// Rational-surface location and the Chebyshev radial domain layout.
//
// The ideal-MHD eigenfunction is singular where q = m/n. A single global
// Chebyshev expansion across that singularity converges algebraically rather
// than spectrally, and the loss is worst exactly where the unstable modes
// localize -- so the radial domain is broken at each resonance and a separate
// expansion is built on each smooth subinterval.
//
// -- Both kernels are single-threaded, and that is not laziness ---------------
// Each produces an ORDERED list whose entries are found by a sequential scan
// with a running index. Parallelizing either would require a prefix sum over a
// few dozen entries to reproduce the same ordering, which costs more complexity
// than the scan costs time. The scan is over n_psi surfaces -- tens -- and runs
// once per toroidal mode number.
//
// -- Resolution limits what "at the resonance" can mean -----------------------
// q is known only at the traced surfaces, so a resonance is located by
// interpolating within the bracketing pair. The breakpoint is therefore
// accurate to the surface spacing, not to machine precision. That is adequate
// -- the subinterval boundary only has to separate the singularity from the
// smooth region, not sit on it exactly -- but it does mean the radial surface
// count, not the Chebyshev order, sets how well a resonant layer can be
// isolated.

#include "quasar/physics/stability/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace quasar::stability {
namespace {

__global__ void locate_rational_kernel(const Real* __restrict__ psi_n,
                                       const Real* __restrict__ q,
                                       const int* __restrict__ valid,
                                       int n_psi, int n_toroidal,
                                       RationalSurfaces* __restrict__ out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  RationalSurfaces res{};
  if (n_toroidal == 0) {
    // n = 0 has no resonance: q = m/0 is not a condition. The axisymmetric
    // mode is a legitimate thing to analyse, so this is an empty result rather
    // than an error.
    *out = res;
    return;
  }

  const Real n_real = static_cast<Real>(n_toroidal);

  for (int i = 0; i + 1 < n_psi; ++i) {
    if (valid[i] == 0 || valid[i + 1] == 0) continue;
    const Real q0 = q[i];
    const Real q1 = q[i + 1];
    if (q0 == q1) continue;

    // Poloidal numbers whose resonance could lie in this interval. Scanning the
    // bracketed range rather than a fixed window keeps this correct for a
    // reversed-shear profile, where q is not monotone and the same m can
    // resonate more than once.
    // Multiplication by a negative n reverses the interval.  Form the bounds
    // after multiplication so n and -n locate the same physical surfaces
    // with opposite poloidal-mode labels.
    const Real lo = fmin(q0 * n_real, q1 * n_real);
    const Real hi = fmax(q0 * n_real, q1 * n_real);
    const int m_lo = static_cast<int>(ceil(lo));
    const int m_hi = static_cast<int>(floor(hi));

    for (int m = m_lo; m <= m_hi; ++m) {
      const Real target = static_cast<Real>(m) / n_real;
      // Half-open in q so a resonance sitting exactly on a surface is claimed
      // by one interval only, never both.
      const bool inside = (q0 <= q1) ? (target >= q0 && target < q1)
                                     : (target > q1 && target <= q0);
      if (!inside) continue;

      if (res.count >= RationalSurfaces::kMaxRational) {
        res.overflow = true;
        break;
      }
      const Real frac = (target - q0) / (q1 - q0);
      res.psi_n[res.count] = psi_n[i] + frac * (psi_n[i + 1] - psi_n[i]);
      res.m[res.count] = m;
      ++res.count;
    }
  }

  // The scan visits surfaces in ascending psi_n, but within one interval the
  // m loop runs in ascending m, which is descending psi_n when q decreases.
  // Sort so the list is ascending in psi_n regardless of the shear sign.
  for (int a = 1; a < res.count; ++a) {
    const Real key_psi = res.psi_n[a];
    const int key_m = res.m[a];
    int b = a - 1;
    while (b >= 0 && res.psi_n[b] > key_psi) {
      res.psi_n[b + 1] = res.psi_n[b];
      res.m[b + 1] = res.m[b];
      --b;
    }
    res.psi_n[b + 1] = key_psi;
    res.m[b + 1] = key_m;
  }

  *out = res;
}

__global__ void build_domains_kernel(const RationalSurfaces* __restrict__ rational,
                                     Real psi_lo, Real psi_hi, int min_domains,
                                     Real min_width,
                                     RadialDomains* __restrict__ out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  RadialDomains d{};
  if (!(psi_hi > psi_lo)) {
    *out = d;
    return;
  }

  // Start from a uniform subdivision, so a mode number with no resonance in
  // range still gets a usable layout instead of one interval spanning the whole
  // plasma.
  int uniform = min_domains > 0 ? min_domains : 1;
  if (uniform > RadialDomains::kMaxDomains) uniform = RadialDomains::kMaxDomains;

  Real pts[RadialDomains::kMaxDomains + 1];
  int n = 0;
  for (int k = 0; k <= uniform; ++k) {
    pts[n++] = psi_lo + (psi_hi - psi_lo) * static_cast<Real>(k)
                            / static_cast<Real>(uniform);
  }

  // Insert each resonance in order.  A resonance closer than min_width to an
  // existing point is snapped onto that point rather than inserted: the thin
  // interval would be ill-conditioned, but the surviving point is still
  // tagged as rational below.  This distinction matters because continuity is
  // harmonic-specific at a rational cut.
  const Real width = min_width > Real{0} ? min_width : Real{0};
  const int rational_count =
      rational->count < RationalSurfaces::kMaxRational
          ? rational->count
          : RationalSurfaces::kMaxRational;
  if (rational->count < 0
      || rational->count > RationalSurfaces::kMaxRational) {
    d.overflow = true;
  }
  Real assigned_breakpoint[RationalSurfaces::kMaxRational]{};
  bool represented[RationalSurfaces::kMaxRational]{};
  for (int r = 0; r < rational_count; ++r) {
    const Real p = rational->psi_n[r];
    if (p <= psi_lo || p >= psi_hi) continue;

    int nearest = -1;
    Real nearest_distance = Real{0};
    // Never snap an interior rational surface onto either physical endpoint:
    // endpoint tags are not internal interfaces and therefore cannot carry
    // the required one-sided spectral topology.  A narrow end interval is
    // preferable to silently erasing the resonance.
    for (int k = 1; k + 1 < n; ++k) {
      const Real distance = fabs(pts[k] - p);
      if ((distance == Real{0} || distance < width)
          && (nearest < 0 || distance < nearest_distance)) {
        nearest = k;
        nearest_distance = distance;
      }
    }
    if (nearest >= 0) {
      assigned_breakpoint[r] = pts[nearest];
      represented[r] = true;
      continue;
    }

    if (n >= RadialDomains::kMaxDomains + 1) {
      d.overflow = true;
      break;
    }
    // Insertion sort into the ascending list.
    int k = n;
    while (k > 0 && pts[k - 1] > p) {
      pts[k] = pts[k - 1];
      --k;
    }
    pts[k] = p;
    ++n;
    assigned_breakpoint[r] = p;
    represented[r] = true;
  }

  for (int k = 0; k < n; ++k) d.breakpoints[k] = pts[k];
  d.n_domains = n - 1;

  // Re-associate every represented resonance with its final breakpoint.
  // Doing this after all insertions avoids having to shift variable-length
  // metadata during insertion.  Tags are grouped by breakpoint so callers can
  // query one interface without scanning unrelated resonances.
  int tag = 0;
  for (int breakpoint = 0; breakpoint < n; ++breakpoint) {
    d.resonance_offsets[breakpoint] = tag;
    for (int r = 0; r < rational_count; ++r) {
      const Real p = rational->psi_n[r];
      // assigned_breakpoint records the exact cut selected at insertion time.
      // A later insertion may be geometrically closer to this resonance, but
      // must not steal provenance from a regular cut already chosen for it.
      if (represented[r] && assigned_breakpoint[r] == pts[breakpoint]) {
        if (tag >= RadialDomains::kMaxResonanceTags) {
          d.overflow = true;
          continue;
        }
        d.resonant_m[tag] = rational->m[r];
        d.resonant_psi_n[tag] = p;
        ++tag;
      }
    }
  }
  d.resonance_offsets[n] = tag;
  d.resonance_count = tag;
  d.overflow = d.overflow || rational->overflow;
  *out = d;
}

}  // namespace

void launch_locate_rational_surfaces(const FluxCoordinateGrid& coords,
                                     int n_toroidal, RationalSurfaces* d_out,
                                     stream_t stream) {
  hipLaunchKernelGGL(locate_rational_kernel, dim3{1}, dim3{1}, 0,
                     static_cast<hipStream_t>(stream),
                     coords.psi_n.device_ptr(), coords.q.device_ptr(),
                     coords.valid.device_ptr(), coords.n_psi, n_toroidal,
                     d_out);
  QUASAR_HIP_CHECK(hipGetLastError());
}

void launch_build_radial_domains(const RationalSurfaces* d_rational,
                                 Real psi_lo, Real psi_hi, int min_domains,
                                 Real min_width, RadialDomains* d_out,
                                 stream_t stream) {
  hipLaunchKernelGGL(build_domains_kernel, dim3{1}, dim3{1}, 0,
                     static_cast<hipStream_t>(stream), d_rational, psi_lo,
                     psi_hi, min_domains, min_width, d_out);
  QUASAR_HIP_CHECK(hipGetLastError());
}

}  // namespace quasar::stability
