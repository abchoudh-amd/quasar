// PEST straight-field-line coordinates from traced flux surfaces.
//
// See physics/stability/kernels.hpp for why theta* is the antiderivative of the
// q integrand, and why a straight-field-line angle is what makes the
// discretized operator banded.
//
// -- Legacy three-stage path, parallelized where it admits it ------------------
//   1. theta*(l) along each contour: a cumulative integral, so sequential
//      within a surface and parallel across surfaces. One thread per surface.
//   2. Resampling (r, z) onto uniform theta*: also one thread per surface,
//      because it walks the monotone theta* list once. Doing it per output
//      point would mean a search per point.
//   3. Metric: one thread per (surface, theta*) node, since every entry is a
//      local finite difference of the resampled mapping.
//
// -- The angular difference is periodic ---------------------------------------
// d/dtheta* uses a centred difference that wraps, because theta* is an angle:
// node 0's left neighbour is node n_theta-1. Using a one-sided difference at
// the ends instead would put an O(h) error exactly at the outboard midplane on
// an up-down symmetric equilibrium, which is where the ballooning drive peaks.
//
// -- The radial difference is not -----------------------------------------------
// d/dpsi is one-sided at the innermost and outermost surfaces. Those are the
// axis side, where the coordinate is singular anyway, and the plasma edge. The
// energy functional does not evaluate there; the surfaces exist to give the
// interior a centred stencil.

#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/stability/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace quasar::stability {
namespace {

inline constexpr unsigned kBlock1D = 64;
inline constexpr unsigned kBlock2D = 16;
inline constexpr Real kPi = Real{3.14159265358979323846};

struct GridView {
  int nr;
  int nz;
  Real r_min;
  Real z_min;
  Real dr;
  Real dz;

  __device__ std::size_t index(int i, int j) const {
    return static_cast<std::size_t>(i) + static_cast<std::size_t>(nr) * j;
  }
};

GridView view_of(const EllipticGrid& g) {
  return GridView{g.nr, g.nz, g.r_min, g.z_min, g.dr(), g.dz()};
}

// Bilinear sampling of an equilibrium-grid field, matching the sampler the
// equilibrium module uses for the same quantities.  This remains the legacy
// finite-difference coordinate path's sampler; the spectral path below has its
// own cubic sampler.
//
// A fourth-order cubic-Lagrange sampler was tried here and REJECTED on
// measurement, which is worth recording so it is not tried again on the same
// reasoning. The hypothesis was that bilinear's second-order error in the grid
// spacing put a floor under the straight-field-line construction. It does not:
//
//   * At a 129x129 equilibrium the two samplers agree to within a few percent
//     of each other on every surface, so the sampler is not what limits
//     accuracy there.
//   * At 33x33 cubic was modestly better through the mid-radius (2.2e-3 versus
//     3.3e-3 around psi_N = 0.6) and modestly WORSE on the innermost surface
//     (3.3e-2 versus 2.7e-2), where B_poloidal approaches zero and the higher
//     order stencil overshoots.
//
// What actually sets the accuracy of that legacy diagnostic is the equilibrium
// grid: refining it
// 33 -> 65 -> 129 moves the innermost surface 2.7e-2 -> 8.9e-3 -> 2.1e-3, and
// the bulk to a ~1.1e-3 floor. Bilinear is kept for consistency with the
// equilibrium module and because the alternative bought nothing there.  It is
// not reused by the exact-Chebyshev path: a second-order sampler would cap the
// convergence of that spectral representation.
__device__ inline Real sample_field(const GridView& g, const Real* f, Real r,
                                    Real z) {
  if (!isfinite(r) || !isfinite(z)) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  Real fi = (r - g.r_min) / g.dr;
  Real fj = (z - g.z_min) / g.dz;
  if (!isfinite(fi) || !isfinite(fj)) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  fi = fmin(fmax(fi, Real{0}), static_cast<Real>(g.nr - 1));
  fj = fmin(fmax(fj, Real{0}), static_cast<Real>(g.nz - 1));
  int i = min(static_cast<int>(fi), g.nr - 2);
  int j = min(static_cast<int>(fj), g.nz - 2);
  const Real tr = fi - static_cast<Real>(i);
  const Real tz = fj - static_cast<Real>(j);
  return (Real{1} - tr) * (Real{1} - tz) * f[g.index(i, j)]
       + tr * (Real{1} - tz) * f[g.index(i + 1, j)]
       + (Real{1} - tr) * tz * f[g.index(i, j + 1)]
       + tr * tz * f[g.index(i + 1, j + 1)];
}

// Tensor-product cubic Lagrange interpolation for the spectral geometry path.
//
// The four-point stencil is shifted wholly inside the field grid at an edge,
// so no out-of-range access or repeated/clamped sample enters the polynomial.
// The evaluation coordinate is allowed to span [0, 3] within that shifted
// stencil; the same cubic formula is therefore used in the interior and at the
// boundary.  Fixed loop ordering is intentional: field sampling participates
// in the ordered contour integral that defines theta*.
__device__ inline void cubic_lagrange_weights(Real x, Real* w) {
  w[0] = -(x - Real{1}) * (x - Real{2}) * (x - Real{3}) / Real{6};
  w[1] = x * (x - Real{2}) * (x - Real{3}) / Real{2};
  w[2] = -x * (x - Real{1}) * (x - Real{3}) / Real{2};
  w[3] = x * (x - Real{1}) * (x - Real{2}) / Real{6};
}

__device__ inline Real sample_field_cubic(const GridView& g, const Real* f,
                                          Real r, Real z) {
  if (!isfinite(r) || !isfinite(z)) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  Real fi = (r - g.r_min) / g.dr;
  Real fj = (z - g.z_min) / g.dz;
  if (!isfinite(fi) || !isfinite(fj)) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  fi = fmin(fmax(fi, Real{0}), static_cast<Real>(g.nr - 1));
  fj = fmin(fmax(fj, Real{0}), static_cast<Real>(g.nz - 1));

  int i0 = static_cast<int>(floor(fi)) - 1;
  int j0 = static_cast<int>(floor(fj)) - 1;
  i0 = max(0, min(i0, g.nr - 4));
  j0 = max(0, min(j0, g.nz - 4));

  Real wr[4];
  Real wz[4];
  cubic_lagrange_weights(fi - static_cast<Real>(i0), wr);
  cubic_lagrange_weights(fj - static_cast<Real>(j0), wz);

  Real value = Real{0};
  for (int jj = 0; jj < 4; ++jj) {
    Real row = Real{0};
    for (int ii = 0; ii < 4; ++ii) {
      row += wr[ii] * f[g.index(i0 + ii, j0 + jj)];
    }
    value += wz[jj] * row;
  }
  return value;
}

__device__ inline void invalidate_surface(std::size_t output_base,
                                          int n_theta, int surface,
                                          Real* out_r, Real* out_z,
                                          Real* out_q, int* out_valid) {
  out_valid[surface] = 0;
  out_q[surface] = Real{0};
  for (int j = 0; j < n_theta; ++j) {
    out_r[output_base + j] = Real{0};
    out_z[output_base + j] = Real{0};
  }
}

// Stage 1 and 2, fused: walk the contour accumulating theta*, then resample.
//
// Fused because both are sequential walks over the same data and splitting them
// would mean writing and re-reading a per-contour-point theta* array for no
// gain -- the surface count, not the point count, bounds the parallelism either
// way.
__global__ void build_surface_kernel(
    GridView g, const Real* __restrict__ surf_r,
    const Real* __restrict__ surf_z, const int* __restrict__ surf_count,
    const int* __restrict__ surf_closed, const Real* __restrict__ surf_psi_n,
    const Real* __restrict__ b_pol, const Real* __restrict__ b_phi,
    int n_surfaces, int n_contour, int n_theta, Real* __restrict__ out_r,
    Real* __restrict__ out_z, Real* __restrict__ out_q,
    Real* __restrict__ out_psi_n, int* __restrict__ out_valid,
    Real* __restrict__ scratch_theta) {
  const int s = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (s >= n_surfaces) return;

  out_psi_n[s] = surf_psi_n[s];
  const std::size_t cbase = static_cast<std::size_t>(s) * n_contour;
  const std::size_t obase = static_cast<std::size_t>(s) * n_theta;
  const int n = surf_count[s];

  if (surf_closed[s] != 1 || n < 4 || n > n_contour
      || !isfinite(surf_psi_n[s])) {
    invalidate_surface(obase, n_theta, s, out_r, out_z, out_q, out_valid);
    return;
  }

  // -- Stage 1: cumulative toroidal advance around the contour ---------------
  // scratch_theta[k] holds the running integral of B_phi/(r B_pol) dl up to
  // contour point k, with entry 0 equal to zero and entry n the full circuit.
  Real acc = Real{0};
  bool malformed = false;
  scratch_theta[cbase] = Real{0};
  for (int k = 0; k < n; ++k) {
    const int m = (k + 1) % n;
    const Real r0 = surf_r[cbase + k];
    const Real z0 = surf_z[cbase + k];
    const Real r1 = surf_r[cbase + m];
    const Real z1 = surf_z[cbase + m];
    if (!isfinite(r0) || !isfinite(z0) || !isfinite(r1)
        || !isfinite(z1)) {
      malformed = true;
      break;
    }
    const Real dr = r1 - r0;
    const Real dz = z1 - z0;
    const Real dl = sqrt(dr * dr + dz * dz);
    const Real rm = Real{0.5} * (r0 + r1);
    const Real zm = Real{0.5} * (z0 + z1);
    if (!isfinite(dl) || !isfinite(rm) || !isfinite(zm)
        || !(rm > Real{0})) {
      malformed = true;
      break;
    }
    const Real bp = sample_field(g, b_pol, rm, zm);
    const Real bt = sample_field(g, b_phi, rm, zm);
    if (!isfinite(bp) || !isfinite(bt) || !(bp > Real{0})) {
      malformed = true;
      break;
    }
    const Real denominator = rm * bp;
    if (!isfinite(denominator) || !(denominator > Real{0})) {
      malformed = true;
      break;
    }
    const Real increment = bt / denominator * dl;
    acc += increment;
    if (!isfinite(increment) || !isfinite(acc)) {
      malformed = true;
      break;
    }
    if (k + 1 < n) scratch_theta[cbase + k + 1] = acc;
  }
  const Real total = acc;

  // q is the circuit integral over 2*pi -- the same quantity compute_q returns,
  // recomputed here so theta* and q come from one accumulation rather than two
  // that could disagree at the last bits and leave theta* not quite reaching
  // 2*pi.
  if (malformed || !isfinite(total) || !(fabs(total) > Real{0})) {
    invalidate_surface(obase, n_theta, s, out_r, out_z, out_q, out_valid);
    return;
  }

  // Normalize to [0, 2*pi). Dividing by the measured total rather than by
  // 2*pi*q guarantees theta* closes exactly on the circuit even where the
  // contour is coarsely resolved.
  const Real q = total / (Real{2} * kPi);
  const Real norm = Real{2} * kPi / total;
  if (!isfinite(q) || q == Real{0} || !isfinite(norm)) {
    invalidate_surface(obase, n_theta, s, out_r, out_z, out_q, out_valid);
    return;
  }
  for (int k = 0; k < n; ++k) {
    const Real normalized = scratch_theta[cbase + k] * norm;
    if (!isfinite(normalized)) {
      malformed = true;
      break;
    }
    scratch_theta[cbase + k] = normalized;
  }
  if (malformed) {
    invalidate_surface(obase, n_theta, s, out_r, out_z, out_q, out_valid);
    return;
  }

  // -- Stage 2: resample (r, z) at uniform theta* ----------------------------
  // theta* is monotone in k (the integrand is single-signed on a well-formed
  // surface), so one forward walk suffices.
  int k = 0;
  for (int j = 0; j < n_theta; ++j) {
    const Real target = Real{2} * kPi * static_cast<Real>(j)
                      / static_cast<Real>(n_theta);
    while (k + 1 < n && scratch_theta[cbase + k + 1] <= target) ++k;

    const Real t0 = scratch_theta[cbase + k];
    const Real t1 = (k + 1 < n) ? scratch_theta[cbase + k + 1]
                                : Real{2} * kPi;
    const int k1 = (k + 1) % n;
    const Real span = t1 - t0;
    const Real frac = span > Real{0} ? (target - t0) / span : Real{0};

    const Real resampled_r = surf_r[cbase + k]
                           + frac * (surf_r[cbase + k1] - surf_r[cbase + k]);
    const Real resampled_z = surf_z[cbase + k]
                           + frac * (surf_z[cbase + k1] - surf_z[cbase + k]);
    if (!isfinite(frac) || !isfinite(resampled_r) || !isfinite(resampled_z)) {
      malformed = true;
      break;
    }
    out_r[obase + j] = resampled_r;
    out_z[obase + j] = resampled_z;
  }
  if (malformed) {
    invalidate_surface(obase, n_theta, s, out_r, out_z, out_q, out_valid);
    return;
  }
  out_q[s] = q;
  out_valid[s] = 1;
}

// Spectral counterpart of build_surface_kernel.  The contour integration and
// resampling order deliberately match the legacy path; only the field sampler
// changes, because bilinear interpolation would cap the otherwise spectral
// geometry at second order.
__global__ void build_spectral_surface_kernel(
    GridView g, const Real* __restrict__ surf_r,
    const Real* __restrict__ surf_z, const int* __restrict__ surf_count,
    const int* __restrict__ surf_closed, const Real* __restrict__ surf_psi_n,
    const Real* __restrict__ b_pol, const Real* __restrict__ b_phi,
    int n_surfaces, int n_contour, int n_theta, Real* __restrict__ out_r,
    Real* __restrict__ out_z, Real* __restrict__ out_q,
    Real* __restrict__ out_psi_n, int* __restrict__ out_valid,
    Real* __restrict__ scratch_theta) {
  const int s = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (s >= n_surfaces) return;

  out_psi_n[s] = surf_psi_n[s];
  const std::size_t cbase = static_cast<std::size_t>(s) * n_contour;
  const std::size_t obase = static_cast<std::size_t>(s) * n_theta;
  const int n = surf_count[s];

  if (surf_closed[s] != 1 || n < 4 || n > n_contour
      || !isfinite(surf_psi_n[s])) {
    invalidate_surface(obase, n_theta, s, out_r, out_z, out_q, out_valid);
    return;
  }

  Real acc = Real{0};
  bool malformed = false;
  scratch_theta[cbase] = Real{0};
  for (int k = 0; k < n; ++k) {
    const int m = (k + 1) % n;
    const Real r0 = surf_r[cbase + k];
    const Real z0 = surf_z[cbase + k];
    const Real r1 = surf_r[cbase + m];
    const Real z1 = surf_z[cbase + m];
    if (!isfinite(r0) || !isfinite(z0) || !isfinite(r1)
        || !isfinite(z1)) {
      malformed = true;
      break;
    }
    const Real dr = r1 - r0;
    const Real dz = z1 - z0;
    const Real dl = sqrt(dr * dr + dz * dz);
    const Real rm = Real{0.5} * (r0 + r1);
    const Real zm = Real{0.5} * (z0 + z1);
    if (!isfinite(dl) || !isfinite(rm) || !isfinite(zm)
        || !(rm > Real{0})) {
      malformed = true;
      break;
    }
    const Real bp = sample_field_cubic(g, b_pol, rm, zm);
    const Real bt = sample_field_cubic(g, b_phi, rm, zm);
    if (!isfinite(bp) || !isfinite(bt) || !(bp > Real{0})) {
      malformed = true;
      break;
    }
    const Real denominator = rm * bp;
    if (!isfinite(denominator) || !(denominator > Real{0})) {
      malformed = true;
      break;
    }
    const Real increment = bt / denominator * dl;
    acc += increment;
    if (!isfinite(increment) || !isfinite(acc)) {
      malformed = true;
      break;
    }
    if (k + 1 < n) scratch_theta[cbase + k + 1] = acc;
  }
  const Real total = acc;

  if (malformed || !isfinite(total) || !(fabs(total) > Real{0})) {
    invalidate_surface(obase, n_theta, s, out_r, out_z, out_q, out_valid);
    return;
  }

  const Real q = total / (Real{2} * kPi);
  const Real norm = Real{2} * kPi / total;
  if (!isfinite(q) || q == Real{0} || !isfinite(norm)) {
    invalidate_surface(obase, n_theta, s, out_r, out_z, out_q, out_valid);
    return;
  }
  for (int k = 0; k < n; ++k) {
    const Real normalized = scratch_theta[cbase + k] * norm;
    if (!isfinite(normalized)) {
      malformed = true;
      break;
    }
    scratch_theta[cbase + k] = normalized;
  }
  if (malformed) {
    invalidate_surface(obase, n_theta, s, out_r, out_z, out_q, out_valid);
    return;
  }

  int k = 0;
  for (int j = 0; j < n_theta; ++j) {
    const Real target = Real{2} * kPi * static_cast<Real>(j)
                      / static_cast<Real>(n_theta);
    while (k + 1 < n && scratch_theta[cbase + k + 1] <= target) ++k;

    const Real t0 = scratch_theta[cbase + k];
    const Real t1 = (k + 1 < n) ? scratch_theta[cbase + k + 1]
                                : Real{2} * kPi;
    const int k1 = (k + 1) % n;
    const Real span = t1 - t0;
    const Real frac = span > Real{0} ? (target - t0) / span : Real{0};

    const Real resampled_r = surf_r[cbase + k]
                           + frac * (surf_r[cbase + k1] - surf_r[cbase + k]);
    const Real resampled_z = surf_z[cbase + k]
                           + frac * (surf_z[cbase + k1] - surf_z[cbase + k]);
    if (!isfinite(frac) || !isfinite(resampled_r) || !isfinite(resampled_z)) {
      malformed = true;
      break;
    }
    out_r[obase + j] = resampled_r;
    out_z[obase + j] = resampled_z;
  }
  if (malformed) {
    invalidate_surface(obase, n_theta, s, out_r, out_z, out_q, out_valid);
    return;
  }
  out_q[s] = q;
  out_valid[s] = 1;
}

// Stage 3: metric from local differences of the resampled mapping.
__global__ void metric_kernel(const Real* __restrict__ r,
                              const Real* __restrict__ z,
                              const Real* __restrict__ psi_n,
                              const int* __restrict__ valid, int n_psi,
                              int n_theta, Real* __restrict__ dr_dpsi,
                              Real* __restrict__ dz_dpsi,
                              Real* __restrict__ dr_dtheta,
                              Real* __restrict__ dz_dtheta,
                              Real* __restrict__ jac,
                              Real* __restrict__ g_pp,
                              Real* __restrict__ g_pt,
                              Real* __restrict__ g_tt) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int i = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (j >= n_theta || i >= n_psi) return;

  const std::size_t k = static_cast<std::size_t>(i) * n_theta + j;
  if (valid[i] == 0) {
    dr_dpsi[k] = dz_dpsi[k] = dr_dtheta[k] = dz_dtheta[k] = Real{0};
    jac[k] = g_pp[k] = g_pt[k] = g_tt[k] = Real{0};
    return;
  }

  // Poloidal derivative: centred and WRAPPING, because theta* is an angle.
  const int jm = (j - 1 + n_theta) % n_theta;
  const int jp = (j + 1) % n_theta;
  const Real dtheta = Real{2} * kPi / static_cast<Real>(n_theta);
  const std::size_t km = static_cast<std::size_t>(i) * n_theta + jm;
  const std::size_t kp = static_cast<std::size_t>(i) * n_theta + jp;
  const Real r_t = (r[kp] - r[km]) / (Real{2} * dtheta);
  const Real z_t = (z[kp] - z[km]) / (Real{2} * dtheta);

  // Radial derivative: centred where possible, one-sided at the ends. Both ends
  // are excluded from the energy functional -- the inner one is the coordinate
  // singularity at the axis, the outer one the plasma edge.
  int i_lo = i - 1;
  int i_hi = i + 1;
  if (i_lo < 0) i_lo = i;
  if (i_hi >= n_psi) i_hi = i;
  if (i_lo != i && valid[i_lo] == 0) i_lo = i;
  if (i_hi != i && valid[i_hi] == 0) i_hi = i;

  Real r_p = Real{0};
  Real z_p = Real{0};
  const Real dpsi = psi_n[i_hi] - psi_n[i_lo];
  if (i_hi != i_lo && dpsi != Real{0}) {
    const std::size_t k_lo = static_cast<std::size_t>(i_lo) * n_theta + j;
    const std::size_t k_hi = static_cast<std::size_t>(i_hi) * n_theta + j;
    r_p = (r[k_hi] - r[k_lo]) / dpsi;
    z_p = (z[k_hi] - z[k_lo]) / dpsi;
  }

  dr_dpsi[k] = r_p;
  dz_dpsi[k] = z_p;
  dr_dtheta[k] = r_t;
  dz_dtheta[k] = z_t;

  // Jacobian of (psi_n, theta*, phi) -> physical space. The r factor is the
  // toroidal arc length per radian.
  const Real det = r_p * z_t - r_t * z_p;
  jac[k] = r[k] * det;

  // Covariant components of the poloidal block, then invert the 2x2 to get the
  // contravariant ones the energy functional actually contracts with.
  const Real cov_pp = r_p * r_p + z_p * z_p;
  const Real cov_pt = r_p * r_t + z_p * z_t;
  const Real cov_tt = r_t * r_t + z_t * z_t;
  const Real det2 = cov_pp * cov_tt - cov_pt * cov_pt;

  if (fabs(det2) > Real{0}) {
    g_pp[k] = cov_tt / det2;
    g_pt[k] = -cov_pt / det2;
    g_tt[k] = cov_pp / det2;
  } else {
    // Degenerate mapping: report zeros rather than infinities so a caller sees
    // a dead surface instead of NaNs propagating into the assembly.
    g_pp[k] = g_pt[k] = g_tt[k] = Real{0};
  }
}

// A Chebyshev derivative couples every local node in one radial domain.  If a
// contour is unusable, the whole local polynomial is unusable; mark it so
// downstream assembly cannot accidentally combine zeros from one node with
// valid geometry from its neighbours.
__global__ void invalidate_open_spectral_domains_kernel(
    int* __restrict__ valid, int n_domains, int n_nodes) {
  const int d = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (d >= n_domains) return;

  const int base = d * n_nodes;
  int domain_valid = 1;
  for (int i = 0; i < n_nodes; ++i) domain_valid &= valid[base + i];
  if (domain_valid != 0) return;
  for (int i = 0; i < n_nodes; ++i) valid[base + i] = 0;
}

__device__ inline Real fourier_collocation_derivative(
    const Real* __restrict__ values, std::size_t base, int j, int n_theta) {
  Real derivative = Real{0};
  for (int l = 0; l < n_theta; ++l) {
    if (l == j) continue;
    const int delta = j - l;
    const int magnitude = delta < 0 ? -delta : delta;
    const Real sign = (magnitude & 1) != 0 ? Real{-1} : Real{1};
    const Real angle = kPi * static_cast<Real>(delta)
                     / static_cast<Real>(n_theta);
    const Real weight = (n_theta & 1) == 0
                            ? Real{0.5} * sign / tan(angle)
                            : Real{0.5} * sign / sin(angle);
    derivative += weight * values[base + l];
  }
  return derivative;
}

// Spectral mapping derivatives and the metric on the unchanged local-node
// layout.  Each output entry has one owning thread and every reduction follows
// ascending local/Fourier index order, so no atomics or scheduler-dependent
// accumulation enter the result.
__global__ void spectral_metric_kernel(
    const Real* __restrict__ r, const Real* __restrict__ z,
    const int* __restrict__ valid, const Real* __restrict__ diff,
    int n_domains, int n_nodes, int n_theta, Real* __restrict__ dr_dpsi,
    Real* __restrict__ dz_dpsi, Real* __restrict__ dr_dtheta,
    Real* __restrict__ dz_dtheta, Real* __restrict__ jac,
    Real* __restrict__ g_pp, Real* __restrict__ g_pt,
    Real* __restrict__ g_tt) {
  const int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int s = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int n_surfaces = n_domains * n_nodes;
  if (j >= n_theta || s >= n_surfaces) return;

  const std::size_t k = static_cast<std::size_t>(s) * n_theta + j;
  if (valid[s] == 0) {
    dr_dpsi[k] = dz_dpsi[k] = dr_dtheta[k] = dz_dtheta[k] = Real{0};
    jac[k] = g_pp[k] = g_pt[k] = g_tt[k] = Real{0};
    return;
  }

  const int d = s / n_nodes;
  const int i = s - d * n_nodes;
  const std::size_t node_base = static_cast<std::size_t>(d) * n_nodes;
  const std::size_t diff_row = node_base * n_nodes
                             + static_cast<std::size_t>(i) * n_nodes;

  Real r_p = Real{0};
  Real z_p = Real{0};
  for (int l = 0; l < n_nodes; ++l) {
    const std::size_t source = (node_base + static_cast<std::size_t>(l))
                             * n_theta + j;
    const Real coefficient = diff[diff_row + l];
    r_p += coefficient * r[source];
    z_p += coefficient * z[source];
  }

  const std::size_t theta_base = static_cast<std::size_t>(s) * n_theta;
  const Real r_t = fourier_collocation_derivative(r, theta_base, j, n_theta);
  const Real z_t = fourier_collocation_derivative(z, theta_base, j, n_theta);

  dr_dpsi[k] = r_p;
  dz_dpsi[k] = z_p;
  dr_dtheta[k] = r_t;
  dz_dtheta[k] = z_t;

  const Real det = r_p * z_t - r_t * z_p;
  jac[k] = r[k] * det;

  const Real cov_pp = r_p * r_p + z_p * z_p;
  const Real cov_pt = r_p * r_t + z_p * z_t;
  const Real cov_tt = r_t * r_t + z_t * z_t;
  // For a two-dimensional mapping det(g_cov) is exactly det(mapping)^2.
  // Forming that square directly avoids cancellation in
  // cov_pp*cov_tt-cov_pt^2 when coordinate directions are nearly aligned.
  const Real det2 = det * det;
  if (det2 > Real{0}) {
    g_pp[k] = cov_tt / det2;
    g_pt[k] = -cov_pt / det2;
    g_tt[k] = cov_pp / det2;
  } else {
    g_pp[k] = g_pt[k] = g_tt[k] = Real{0};
  }
}

// One deterministic thread verifies both the basis' local ordering contract
// and the exact surface-to-node correspondence before geometry is built.
// Status values are intentionally coarse: the host exception names the broken
// contract, while avoiding a transfer of either full coordinate array.
__global__ void validate_spectral_order_kernel(
    const Real* __restrict__ nodes, const Real* __restrict__ surface_psi_n,
    const int* __restrict__ surface_count,
    const int* __restrict__ surface_closed, int n_domains, int n_nodes,
    int n_contour, int* __restrict__ status) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  *status = 0;

  const int n_local = n_domains * n_nodes;
  for (int s = 0; s < n_local; ++s) {
    if (!isfinite(nodes[s]) || surface_psi_n[s] != nodes[s]) {
      *status = 1;
      return;
    }
    if ((surface_closed[s] != 0 && surface_closed[s] != 1)
        || surface_count[s] < 0 || surface_count[s] > n_contour) {
      *status = 3;
      return;
    }
  }

  for (int d = 0; d < n_domains; ++d) {
    const int base = d * n_nodes;
    for (int i = 1; i < n_nodes; ++i) {
      if (!(nodes[base + i - 1] > nodes[base + i])) {
        *status = 2;
        return;
      }
    }
    if (d + 1 < n_domains) {
      const Real upper = nodes[base];
      const Real next_lower = nodes[(d + 1) * n_nodes + n_nodes - 1];
      if (upper != next_lower) {
        *status = 2;
        return;
      }
    }
  }
}

// dphi/dtheta* must equal q everywhere on a surface. One thread per surface,
// writing that surface's worst relative deviation.
__global__ void straightness_kernel(GridView g, const Real* __restrict__ r,
                                    const Real* __restrict__ z,
                                    const Real* __restrict__ dr_dtheta,
                                    const Real* __restrict__ dz_dtheta,
                                    const Real* __restrict__ q,
                                    const int* __restrict__ valid, int n_psi,
                                    int n_theta,
                                    const Real* __restrict__ b_pol,
                                    const Real* __restrict__ b_phi,
                                    Real* __restrict__ out) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n_psi) return;

  out[i] = Real{0};
  if (valid[i] == 0) return;
  constexpr Real maximum = std::numeric_limits<Real>::max();
  if (valid[i] != 1 || !isfinite(q[i]) || q[i] == Real{0}) {
    out[i] = maximum;
    return;
  }

  Real worst = Real{0};
  for (int j = 0; j < n_theta; ++j) {
    const std::size_t k = static_cast<std::size_t>(i) * n_theta + j;
    const Real rr = r[k];
    if (!isfinite(rr) || !isfinite(z[k]) || !isfinite(dr_dtheta[k])
        || !isfinite(dz_dtheta[k]) || !(rr > Real{0})) {
      out[i] = maximum;
      return;
    }
    const Real bp = sample_field(g, b_pol, rr, z[k]);
    const Real bt = sample_field(g, b_phi, rr, z[k]);
    if (!isfinite(bp) || !isfinite(bt) || !(bp > Real{0})) {
      out[i] = maximum;
      return;
    }

    // Poloidal arc length per unit theta*, then the toroidal advance per unit
    // theta* the local field produces.
    const Real dl_dtheta = sqrt(dr_dtheta[k] * dr_dtheta[k]
                              + dz_dtheta[k] * dz_dtheta[k]);
    const Real denominator = rr * bp;
    if (!isfinite(dl_dtheta) || !isfinite(denominator)
        || !(denominator > Real{0})) {
      out[i] = maximum;
      return;
    }
    const Real dphi_dtheta = bt / denominator * dl_dtheta;
    const Real relative_deviation =
        fabs(dphi_dtheta - q[i]) / fabs(q[i]);
    if (!isfinite(dphi_dtheta) || !isfinite(relative_deviation)) {
      out[i] = maximum;
      return;
    }
    worst = fmax(worst, relative_deviation);
  }
  out[i] = worst;
}

dim3 grid_2d(int nx, int ny, dim3 block) {
  return dim3((static_cast<unsigned>(nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(ny) + block.y - 1) / block.y);
}

std::size_t checked_product(std::size_t a, std::size_t b,
                            const char* message) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::length_error{message};
  }
  return a * b;
}

template <class T>
void require_size(const backend::DeviceBuffer<T>& buffer,
                  std::size_t expected, const char* name) {
  if (buffer.size() != expected) {
    throw std::invalid_argument{std::string{name} + " has size "
                                + std::to_string(buffer.size())
                                + "; expected " + std::to_string(expected)};
  }
}

template <class T>
void require_owner(const backend::DeviceBuffer<T>& buffer, int expected,
                   const char* name) {
  if (buffer.owner_device() != expected) {
    throw std::invalid_argument{std::string{name}
                                + " is allocated on a different device"};
  }
}

struct LegacyGeometryShape {
  int n_surfaces{0};
  int n_contour{0};
  int n_theta{0};
  int owner{-1};
  std::size_t contour_points{0};
  std::size_t geometry_points{0};
};

LegacyGeometryShape validate_legacy_geometry(
    const EllipticGrid& g, const equilibrium::GsFluxSurfaces& surfaces,
    const equilibrium::GsMagneticField& field,
    const FluxCoordinateGrid& out) {
  g.validate();
  if (surfaces.n_surfaces < 1 || surfaces.n_theta < 4) {
    throw std::invalid_argument{
        "GsFluxSurfaces must contain at least one contour with at least four "
        "points"};
  }
  if (out.n_psi != surfaces.n_surfaces || out.n_theta < 3) {
    throw std::invalid_argument{
        "FluxCoordinateGrid must contain every flux surface and at least "
        "three angular points"};
  }

  const std::size_t n_surfaces =
      static_cast<std::size_t>(surfaces.n_surfaces);
  const std::size_t contour_points = checked_product(
      n_surfaces, static_cast<std::size_t>(surfaces.n_theta),
      "legacy flux-surface storage overflows size_t");
  const std::size_t geometry_points = checked_product(
      n_surfaces, static_cast<std::size_t>(out.n_theta),
      "legacy flux-coordinate storage overflows size_t");

  require_size(surfaces.r, contour_points, "GsFluxSurfaces::r");
  require_size(surfaces.z, contour_points, "GsFluxSurfaces::z");
  require_size(surfaces.count, n_surfaces, "GsFluxSurfaces::count");
  require_size(surfaces.closed, n_surfaces, "GsFluxSurfaces::closed");
  require_size(surfaces.psi_n, n_surfaces, "GsFluxSurfaces::psi_n");

  require_size(field.b_poloidal, g.size(), "GsMagneticField::b_poloidal");
  require_size(field.b_phi, g.size(), "GsMagneticField::b_phi");

  require_size(out.psi_n, n_surfaces, "FluxCoordinateGrid::psi_n");
  require_size(out.q, n_surfaces, "FluxCoordinateGrid::q");
  require_size(out.valid, n_surfaces, "FluxCoordinateGrid::valid");
  require_size(out.r, geometry_points, "FluxCoordinateGrid::r");
  require_size(out.z, geometry_points, "FluxCoordinateGrid::z");
  require_size(out.dr_dpsi, geometry_points,
               "FluxCoordinateGrid::dr_dpsi");
  require_size(out.dz_dpsi, geometry_points,
               "FluxCoordinateGrid::dz_dpsi");
  require_size(out.dr_dtheta, geometry_points,
               "FluxCoordinateGrid::dr_dtheta");
  require_size(out.dz_dtheta, geometry_points,
               "FluxCoordinateGrid::dz_dtheta");
  require_size(out.jacobian, geometry_points,
               "FluxCoordinateGrid::jacobian");
  require_size(out.g_psipsi, geometry_points,
               "FluxCoordinateGrid::g_psipsi");
  require_size(out.g_psitheta, geometry_points,
               "FluxCoordinateGrid::g_psitheta");
  require_size(out.g_thetatheta, geometry_points,
               "FluxCoordinateGrid::g_thetatheta");

  const int owner = surfaces.r.owner_device();
  if (owner < 0) {
    throw std::invalid_argument{
        "legacy flux-coordinate inputs are not device-resident"};
  }
  require_owner(surfaces.z, owner, "GsFluxSurfaces::z");
  require_owner(surfaces.count, owner, "GsFluxSurfaces::count");
  require_owner(surfaces.closed, owner, "GsFluxSurfaces::closed");
  require_owner(surfaces.psi_n, owner, "GsFluxSurfaces::psi_n");
  require_owner(field.b_poloidal, owner, "GsMagneticField::b_poloidal");
  require_owner(field.b_phi, owner, "GsMagneticField::b_phi");
  require_owner(out.psi_n, owner, "FluxCoordinateGrid::psi_n");
  require_owner(out.q, owner, "FluxCoordinateGrid::q");
  require_owner(out.valid, owner, "FluxCoordinateGrid::valid");
  require_owner(out.r, owner, "FluxCoordinateGrid::r");
  require_owner(out.z, owner, "FluxCoordinateGrid::z");
  require_owner(out.dr_dpsi, owner, "FluxCoordinateGrid::dr_dpsi");
  require_owner(out.dz_dpsi, owner, "FluxCoordinateGrid::dz_dpsi");
  require_owner(out.dr_dtheta, owner, "FluxCoordinateGrid::dr_dtheta");
  require_owner(out.dz_dtheta, owner, "FluxCoordinateGrid::dz_dtheta");
  require_owner(out.jacobian, owner, "FluxCoordinateGrid::jacobian");
  require_owner(out.g_psipsi, owner, "FluxCoordinateGrid::g_psipsi");
  require_owner(out.g_psitheta, owner, "FluxCoordinateGrid::g_psitheta");
  require_owner(out.g_thetatheta, owner,
                "FluxCoordinateGrid::g_thetatheta");
  if (backend::current_device() != owner) {
    throw std::invalid_argument{
        "legacy flux-coordinate storage is not on the current device"};
  }

  return LegacyGeometryShape{surfaces.n_surfaces, surfaces.n_theta,
                             out.n_theta, owner, contour_points,
                             geometry_points};
}

struct StraightnessShape {
  int n_psi{0};
  int n_theta{0};
  int owner{-1};
};

StraightnessShape validate_straightness(
    const FluxCoordinateGrid& coords, const EllipticGrid& g,
    const equilibrium::GsMagneticField& field, const Real* d_deviation) {
  g.validate();
  if (d_deviation == nullptr) {
    throw std::invalid_argument{
        "straightness output must be a non-null device pointer"};
  }
  if (coords.n_psi < 1 || coords.n_theta < 1) {
    throw std::invalid_argument{
        "straightness check requires non-empty flux coordinates"};
  }

  const std::size_t radial_points =
      static_cast<std::size_t>(coords.n_psi);
  const std::size_t geometry_points = checked_product(
      radial_points, static_cast<std::size_t>(coords.n_theta),
      "straightness coordinate storage overflows size_t");

  require_size(coords.psi_n, radial_points, "FluxCoordinateGrid::psi_n");
  require_size(coords.q, radial_points, "FluxCoordinateGrid::q");
  require_size(coords.valid, radial_points, "FluxCoordinateGrid::valid");
  require_size(coords.r, geometry_points, "FluxCoordinateGrid::r");
  require_size(coords.z, geometry_points, "FluxCoordinateGrid::z");
  require_size(coords.dr_dpsi, geometry_points,
               "FluxCoordinateGrid::dr_dpsi");
  require_size(coords.dz_dpsi, geometry_points,
               "FluxCoordinateGrid::dz_dpsi");
  require_size(coords.dr_dtheta, geometry_points,
               "FluxCoordinateGrid::dr_dtheta");
  require_size(coords.dz_dtheta, geometry_points,
               "FluxCoordinateGrid::dz_dtheta");
  require_size(coords.jacobian, geometry_points,
               "FluxCoordinateGrid::jacobian");
  require_size(coords.g_psipsi, geometry_points,
               "FluxCoordinateGrid::g_psipsi");
  require_size(coords.g_psitheta, geometry_points,
               "FluxCoordinateGrid::g_psitheta");
  require_size(coords.g_thetatheta, geometry_points,
               "FluxCoordinateGrid::g_thetatheta");
  require_size(field.b_poloidal, g.size(), "GsMagneticField::b_poloidal");
  require_size(field.b_phi, g.size(), "GsMagneticField::b_phi");

  const int owner = coords.psi_n.owner_device();
  if (owner < 0) {
    throw std::invalid_argument{
        "straightness inputs are not device-resident"};
  }
  require_owner(coords.q, owner, "FluxCoordinateGrid::q");
  require_owner(coords.valid, owner, "FluxCoordinateGrid::valid");
  require_owner(coords.r, owner, "FluxCoordinateGrid::r");
  require_owner(coords.z, owner, "FluxCoordinateGrid::z");
  require_owner(coords.dr_dpsi, owner, "FluxCoordinateGrid::dr_dpsi");
  require_owner(coords.dz_dpsi, owner, "FluxCoordinateGrid::dz_dpsi");
  require_owner(coords.dr_dtheta, owner, "FluxCoordinateGrid::dr_dtheta");
  require_owner(coords.dz_dtheta, owner, "FluxCoordinateGrid::dz_dtheta");
  require_owner(coords.jacobian, owner, "FluxCoordinateGrid::jacobian");
  require_owner(coords.g_psipsi, owner, "FluxCoordinateGrid::g_psipsi");
  require_owner(coords.g_psitheta, owner, "FluxCoordinateGrid::g_psitheta");
  require_owner(coords.g_thetatheta, owner,
                "FluxCoordinateGrid::g_thetatheta");
  require_owner(field.b_poloidal, owner, "GsMagneticField::b_poloidal");
  require_owner(field.b_phi, owner, "GsMagneticField::b_phi");
  if (backend::current_device() != owner) {
    throw std::invalid_argument{
        "straightness inputs are not on the current device"};
  }

  return StraightnessShape{coords.n_psi, coords.n_theta, owner};
}

struct SpectralGeometryShape {
  int n_local{0};
  int n_contour{0};
  int n_theta{0};
  int owner{-1};
  std::size_t contour_points{0};
  std::size_t geometry_points{0};
};

SpectralGeometryShape validate_spectral_geometry(
    const EllipticGrid& g, const equilibrium::GsFluxSurfaces& surfaces,
    const equilibrium::GsMagneticField& field, const ChebyshevBasis& basis,
    const FluxCoordinateGrid& out) {
  g.validate();
  if (g.nr < 4 || g.nz < 4) {
    throw std::invalid_argument{
        "spectral flux coordinates require at least four field-grid nodes "
        "per axis for cubic interpolation"};
  }
  if (basis.order < 1
      || static_cast<long long>(basis.n_nodes)
             != static_cast<long long>(basis.order) + 1LL
      || basis.n_domains < 1) {
    throw std::invalid_argument{
        "spectral flux coordinates require a non-empty Chebyshev basis with "
        "n_nodes == order + 1 and order >= 1"};
  }

  const std::size_t n_local_size = checked_product(
      static_cast<std::size_t>(basis.n_domains),
      static_cast<std::size_t>(basis.n_nodes),
      "spectral flux-coordinate local-node count overflows size_t");
  if (n_local_size
      > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error{
        "spectral flux-coordinate local-node count does not fit in int"};
  }
  const int n_local = static_cast<int>(n_local_size);
  if (surfaces.n_surfaces != n_local || surfaces.n_theta < 4) {
    throw std::invalid_argument{
        "GsFluxSurfaces must contain one contour with at least four points "
        "for every local Chebyshev node"};
  }
  if (out.n_psi != n_local || out.n_theta < 3) {
    throw std::invalid_argument{
        "FluxCoordinateGrid must contain every local Chebyshev node and at "
        "least three Fourier collocation points"};
  }

  const std::size_t n_diff = checked_product(
      n_local_size, static_cast<std::size_t>(basis.n_nodes),
      "spectral flux-coordinate differentiation storage overflows size_t");
  const std::size_t contour_points = checked_product(
      n_local_size, static_cast<std::size_t>(surfaces.n_theta),
      "spectral flux-surface storage overflows size_t");
  const std::size_t geometry_points = checked_product(
      n_local_size, static_cast<std::size_t>(out.n_theta),
      "spectral flux-coordinate storage overflows size_t");

  require_size(basis.nodes, n_local_size, "ChebyshevBasis::nodes");
  require_size(basis.weights, n_local_size, "ChebyshevBasis::weights");
  require_size(basis.diff, n_diff, "ChebyshevBasis::diff");

  require_size(surfaces.r, contour_points, "GsFluxSurfaces::r");
  require_size(surfaces.z, contour_points, "GsFluxSurfaces::z");
  require_size(surfaces.count, n_local_size, "GsFluxSurfaces::count");
  require_size(surfaces.closed, n_local_size, "GsFluxSurfaces::closed");
  require_size(surfaces.q, n_local_size, "GsFluxSurfaces::q");
  require_size(surfaces.area, n_local_size, "GsFluxSurfaces::area");
  require_size(surfaces.volume, n_local_size, "GsFluxSurfaces::volume");
  require_size(surfaces.psi_n, n_local_size, "GsFluxSurfaces::psi_n");

  require_size(field.b_poloidal, g.size(), "GsMagneticField::b_poloidal");
  require_size(field.b_phi, g.size(), "GsMagneticField::b_phi");

  require_size(out.psi_n, n_local_size, "FluxCoordinateGrid::psi_n");
  require_size(out.q, n_local_size, "FluxCoordinateGrid::q");
  require_size(out.valid, n_local_size, "FluxCoordinateGrid::valid");
  require_size(out.r, geometry_points, "FluxCoordinateGrid::r");
  require_size(out.z, geometry_points, "FluxCoordinateGrid::z");
  require_size(out.dr_dpsi, geometry_points,
               "FluxCoordinateGrid::dr_dpsi");
  require_size(out.dz_dpsi, geometry_points,
               "FluxCoordinateGrid::dz_dpsi");
  require_size(out.dr_dtheta, geometry_points,
               "FluxCoordinateGrid::dr_dtheta");
  require_size(out.dz_dtheta, geometry_points,
               "FluxCoordinateGrid::dz_dtheta");
  require_size(out.jacobian, geometry_points,
               "FluxCoordinateGrid::jacobian");
  require_size(out.g_psipsi, geometry_points,
               "FluxCoordinateGrid::g_psipsi");
  require_size(out.g_psitheta, geometry_points,
               "FluxCoordinateGrid::g_psitheta");
  require_size(out.g_thetatheta, geometry_points,
               "FluxCoordinateGrid::g_thetatheta");

  const int owner = basis.nodes.owner_device();
  if (owner < 0) {
    throw std::invalid_argument{
        "spectral flux-coordinate inputs are not device-resident"};
  }
  require_owner(basis.weights, owner, "ChebyshevBasis::weights");
  require_owner(basis.diff, owner, "ChebyshevBasis::diff");
  require_owner(surfaces.r, owner, "GsFluxSurfaces::r");
  require_owner(surfaces.z, owner, "GsFluxSurfaces::z");
  require_owner(surfaces.count, owner, "GsFluxSurfaces::count");
  require_owner(surfaces.closed, owner, "GsFluxSurfaces::closed");
  require_owner(surfaces.q, owner, "GsFluxSurfaces::q");
  require_owner(surfaces.area, owner, "GsFluxSurfaces::area");
  require_owner(surfaces.volume, owner, "GsFluxSurfaces::volume");
  require_owner(surfaces.psi_n, owner, "GsFluxSurfaces::psi_n");
  require_owner(field.b_poloidal, owner, "GsMagneticField::b_poloidal");
  require_owner(field.b_phi, owner, "GsMagneticField::b_phi");
  require_owner(out.psi_n, owner, "FluxCoordinateGrid::psi_n");
  require_owner(out.q, owner, "FluxCoordinateGrid::q");
  require_owner(out.valid, owner, "FluxCoordinateGrid::valid");
  require_owner(out.r, owner, "FluxCoordinateGrid::r");
  require_owner(out.z, owner, "FluxCoordinateGrid::z");
  require_owner(out.dr_dpsi, owner, "FluxCoordinateGrid::dr_dpsi");
  require_owner(out.dz_dpsi, owner, "FluxCoordinateGrid::dz_dpsi");
  require_owner(out.dr_dtheta, owner, "FluxCoordinateGrid::dr_dtheta");
  require_owner(out.dz_dtheta, owner, "FluxCoordinateGrid::dz_dtheta");
  require_owner(out.jacobian, owner, "FluxCoordinateGrid::jacobian");
  require_owner(out.g_psipsi, owner, "FluxCoordinateGrid::g_psipsi");
  require_owner(out.g_psitheta, owner, "FluxCoordinateGrid::g_psitheta");
  require_owner(out.g_thetatheta, owner,
                "FluxCoordinateGrid::g_thetatheta");
  if (backend::current_device() != owner) {
    throw std::invalid_argument{
        "spectral flux-coordinate storage is not on the current device"};
  }

  return SpectralGeometryShape{n_local, surfaces.n_theta, out.n_theta, owner,
                               contour_points, geometry_points};
}

}  // namespace

void FluxCoordinateGrid::resize(int n_psi_in, int n_theta_in) {
  if (n_psi_in < 0 || n_theta_in < 0) {
    throw std::invalid_argument{
        "FluxCoordinateGrid dimensions must be non-negative"};
  }
  const std::size_t s = static_cast<std::size_t>(n_psi_in);
  const std::size_t n = checked_product(
      s, static_cast<std::size_t>(n_theta_in),
      "FluxCoordinateGrid storage overflows size_t");

  // Preflight byte counts before mutating either buffers or shape metadata.
  // This keeps extreme dimensions deterministic and host-safe.
  (void)backend::detail::checked_buffer_bytes<Real>(s);
  (void)backend::detail::checked_buffer_bytes<int>(s);
  (void)backend::detail::checked_buffer_bytes<Real>(n);

  psi_n = backend::DeviceBuffer<Real>{s};
  q = backend::DeviceBuffer<Real>{s};
  valid = backend::DeviceBuffer<int>{s};

  r = backend::DeviceBuffer<Real>{n};
  z = backend::DeviceBuffer<Real>{n};
  dr_dpsi = backend::DeviceBuffer<Real>{n};
  dz_dpsi = backend::DeviceBuffer<Real>{n};
  dr_dtheta = backend::DeviceBuffer<Real>{n};
  dz_dtheta = backend::DeviceBuffer<Real>{n};
  jacobian = backend::DeviceBuffer<Real>{n};
  g_psipsi = backend::DeviceBuffer<Real>{n};
  g_psitheta = backend::DeviceBuffer<Real>{n};
  g_thetatheta = backend::DeviceBuffer<Real>{n};

  n_psi = n_psi_in;
  n_theta = n_theta_in;
}

void launch_build_flux_coordinates(const EllipticGrid& g,
                                   const equilibrium::GsFluxSurfaces& surfaces,
                                   const equilibrium::GsMagneticField& field,
                                   Real /*axis_r*/, Real /*axis_z*/,
                                   FluxCoordinateGrid& out, stream_t stream) {
  const LegacyGeometryShape shape =
      validate_legacy_geometry(g, surfaces, field, out);
  auto* s = static_cast<hipStream_t>(stream);

  backend::DeviceBuffer<Real> scratch{
      shape.contour_points, backend::uninitialized,
      backend::on_device(shape.owner)};

  const dim3 sblock{kBlock1D};
  const dim3 sgrid{
      (static_cast<unsigned>(shape.n_surfaces) + kBlock1D - 1) / kBlock1D};
  hipLaunchKernelGGL(build_surface_kernel, sgrid, sblock, 0, s, view_of(g),
                     surfaces.r.device_ptr(), surfaces.z.device_ptr(),
                     surfaces.count.device_ptr(), surfaces.closed.device_ptr(),
                     surfaces.psi_n.device_ptr(),
                     field.b_poloidal.device_ptr(), field.b_phi.device_ptr(),
                     shape.n_surfaces, shape.n_contour, shape.n_theta,
                     out.r.device_ptr(),
                     out.z.device_ptr(), out.q.device_ptr(),
                     out.psi_n.device_ptr(), out.valid.device_ptr(),
                     scratch.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  const dim3 mblock{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(metric_kernel,
                     grid_2d(shape.n_theta, shape.n_surfaces, mblock),
                     mblock, 0, s, out.r.device_ptr(), out.z.device_ptr(),
                     out.psi_n.device_ptr(), out.valid.device_ptr(),
                     shape.n_surfaces, shape.n_theta, out.dr_dpsi.device_ptr(),
                     out.dz_dpsi.device_ptr(), out.dr_dtheta.device_ptr(),
                     out.dz_dtheta.device_ptr(), out.jacobian.device_ptr(),
                     out.g_psipsi.device_ptr(), out.g_psitheta.device_ptr(),
                     out.g_thetatheta.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  // The scratch buffer must outlive the kernels that read it.
  backend::device_synchronize(stream);
}

void launch_build_spectral_flux_coordinates(
    const EllipticGrid& g, const equilibrium::GsFluxSurfaces& surfaces,
    const equilibrium::GsMagneticField& field, const ChebyshevBasis& basis,
    FluxCoordinateGrid& out, stream_t stream) {
  const SpectralGeometryShape shape =
      validate_spectral_geometry(g, surfaces, field, basis, out);
  auto* s = static_cast<hipStream_t>(stream);

  // Only this one-word validation result crosses to the host.  The surfaces,
  // basis, fields, and constructed geometry remain device-resident.
  backend::DeviceBuffer<int> order_status{
      1, backend::on_device(shape.owner)};
  hipLaunchKernelGGL(validate_spectral_order_kernel, dim3{1}, dim3{1}, 0, s,
                     basis.nodes.device_ptr(), surfaces.psi_n.device_ptr(),
                     surfaces.count.device_ptr(), surfaces.closed.device_ptr(),
                     basis.n_domains, basis.n_nodes, shape.n_contour,
                     order_status.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());
  backend::device_synchronize(stream);
  int host_order_status = 0;
  order_status.copy_to_host(&host_order_status, 1);
  if (host_order_status == 1) {
    throw std::invalid_argument{
        "GsFluxSurfaces::psi_n must match ChebyshevBasis::nodes exactly in "
        "domain-major/node-major order"};
  }
  if (host_order_status == 2) {
    throw std::invalid_argument{
        "ChebyshevBasis nodes must descend within each domain and duplicate "
        "adjacent domain interfaces exactly"};
  }
  if (host_order_status == 3) {
    throw std::invalid_argument{
        "GsFluxSurfaces count/closed metadata is outside its allocated "
        "contour shape"};
  }

  backend::DeviceBuffer<Real> scratch{
      shape.contour_points, backend::uninitialized,
      backend::on_device(shape.owner)};

  const dim3 surface_block{kBlock1D};
  const dim3 surface_grid{
      (static_cast<unsigned>(shape.n_local) + kBlock1D - 1) / kBlock1D};
  hipLaunchKernelGGL(
      build_spectral_surface_kernel, surface_grid, surface_block, 0, s,
      view_of(g), surfaces.r.device_ptr(), surfaces.z.device_ptr(),
      surfaces.count.device_ptr(), surfaces.closed.device_ptr(),
      surfaces.psi_n.device_ptr(), field.b_poloidal.device_ptr(),
      field.b_phi.device_ptr(), shape.n_local, shape.n_contour, shape.n_theta,
      out.r.device_ptr(), out.z.device_ptr(), out.q.device_ptr(),
      out.psi_n.device_ptr(), out.valid.device_ptr(), scratch.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  const dim3 domain_grid{
      (static_cast<unsigned>(basis.n_domains) + kBlock1D - 1) / kBlock1D};
  hipLaunchKernelGGL(invalidate_open_spectral_domains_kernel, domain_grid,
                     surface_block, 0, s, out.valid.device_ptr(),
                     basis.n_domains, basis.n_nodes);
  QUASAR_HIP_CHECK(hipGetLastError());

  const dim3 metric_block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(
      spectral_metric_kernel,
      grid_2d(shape.n_theta, shape.n_local, metric_block), metric_block, 0, s,
      out.r.device_ptr(), out.z.device_ptr(), out.valid.device_ptr(),
      basis.diff.device_ptr(), basis.n_domains, basis.n_nodes, shape.n_theta,
      out.dr_dpsi.device_ptr(), out.dz_dpsi.device_ptr(),
      out.dr_dtheta.device_ptr(), out.dz_dtheta.device_ptr(),
      out.jacobian.device_ptr(), out.g_psipsi.device_ptr(),
      out.g_psitheta.device_ptr(), out.g_thetatheta.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  // The per-contour cumulative-angle scratch must outlive both dependent
  // kernels.  This mirrors the synchronization contract of the legacy launch.
  backend::device_synchronize(stream);
}

void launch_check_straightness(const FluxCoordinateGrid& coords,
                               const EllipticGrid& g,
                               const equilibrium::GsMagneticField& field,
                               Real* d_deviation, stream_t stream) {
  const StraightnessShape shape =
      validate_straightness(coords, g, field, d_deviation);
  const dim3 block{kBlock1D};
  const dim3 grid{
      (static_cast<unsigned>(shape.n_psi) + kBlock1D - 1) / kBlock1D};
  hipLaunchKernelGGL(straightness_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream), view_of(g),
                     coords.r.device_ptr(), coords.z.device_ptr(),
                     coords.dr_dtheta.device_ptr(),
                     coords.dz_dtheta.device_ptr(), coords.q.device_ptr(),
                     coords.valid.device_ptr(), shape.n_psi, shape.n_theta,
                     field.b_poloidal.device_ptr(), field.b_phi.device_ptr(),
                     d_deviation);
  QUASAR_HIP_CHECK(hipGetLastError());
}

}  // namespace quasar::stability
