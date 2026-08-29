// Device derived diagnostics: magnetic field, traced flux surfaces, safety
// factor, and the aggregate shape bundle.
//
// -- Parallelism follows the data dependence, not the loop nesting -------------
// Three different shapes here, and each is parallelized where it actually
// admits it:
//
//   * The field is one thread per node. Trivially parallel.
//   * Ray marching is one thread per (surface, ray). Rays are independent --
//     each marches outward from the axis until it crosses the target flux --
//     so 32 surfaces x 128 rays is 4096 independent threads doing up to 4000
//     serial samples each. This is the expensive part and the one worth
//     spreading.
//   * Compaction and the per-surface integrals are one thread per SURFACE.
//     They cannot be spread further without changing the answer: the host
//     appends hit points in ray order, so the polygon's vertex ordering is the
//     ray ordering, and both the shoelace area and the q contour integral
//     accumulate sequentially around that polygon. Reducing those in tree order
//     would give a different sum.
//
// -- The F profile is serial by construction -----------------------------------
// F(psi_N) is recovered by integrating FF' inward from the boundary, so each
// sample depends on the one outside it. It runs single-threaded over 257
// points, which costs nothing and preserves the host's accumulation exactly.

#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace quasar::equilibrium {
namespace {

inline constexpr unsigned kBlock2D = 16;
inline constexpr unsigned kBlock1D = 128;
inline constexpr Real kPi = Real{3.14159265358979323846};

// Ray-marching resolution, matching the host's trace_surface.
inline constexpr int kNTheta = 128;
inline constexpr int kNSteps = 4000;

struct GridView {
  int nr;
  int nz;
  Real r_min;
  Real z_min;
  Real r_max;
  Real z_max;
  Real dr;
  Real dz;

  __device__ Real r(int i) const { return r_min + static_cast<Real>(i) * dr; }
  __device__ std::size_t index(int i, int j) const {
    return static_cast<std::size_t>(i) + static_cast<std::size_t>(nr) * j;
  }
};

GridView view_of(const EllipticGrid& g) {
  return GridView{g.nr,    g.nz,    g.r_min, g.z_min,
                  g.r_max, g.z_max, g.dr(),  g.dz()};
}

__device__ inline Real sample_bilinear_dev(const GridView& g, const Real* f,
                                           Real r, Real z) {
  Real fi = (r - g.r_min) / g.dr;
  Real fj = (z - g.z_min) / g.dz;
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

__device__ inline Real normalized_flux_dev(Real psi, Real psi_axis,
                                           Real psi_bdry) {
  const Real denom = psi_bdry - psi_axis;
  if (denom == Real{0}) return Real{1};
  const Real n = (psi - psi_axis) / denom;
  return fmin(fmax(n, Real{0}), Real{1});
}

// Horner, descending -- same order as the host profile evaluation.
__device__ inline Real horner(const Real* c, int n, Real x) {
  Real acc = Real{0};
  for (int k = n; k-- > 0;) acc = acc * x + c[k];
  return acc;
}

// F^2(psi_N) = F_vac^2 + 2 (psi_b - psi_a) * profile_scale * integral of FF',
// integrated inward from psi_N = 1 by the trapezoid rule.
__global__ void integrate_f_kernel(ProfileCoefficients profile, Real f_vacuum,
                                   Real psi_axis, Real psi_bdry,
                                   Real profile_scale, int samples,
                                   Real* __restrict__ f_table) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  const Real dpn = Real{1} / static_cast<Real>(samples - 1);
  const Real scale = Real{2} * (psi_bdry - psi_axis) * profile_scale;

  f_table[samples - 1] = f_vacuum * f_vacuum;
  for (int k = samples - 2; k >= 0; --k) {
    const Real pn_hi = static_cast<Real>(k + 1) * dpn;
    const Real pn_lo = static_cast<Real>(k) * dpn;
    const Real avg = Real{0.5} * (horner(profile.f_coeffs, profile.n_f, pn_hi)
                                + horner(profile.f_coeffs, profile.n_f, pn_lo));
    f_table[k] = f_table[k + 1] - scale * avg * dpn;
  }

  for (int k = 0; k < samples; ++k) {
    const Real v = f_table[k];
    // F^2 < 0 means the requested profile is not realizable with this vacuum
    // field. Clamp rather than emit NaN, and let the caller notice through q.
    f_table[k] = v > Real{0} ? sqrt(v) : Real{0};
  }
}

// Nearest-neighbour lookup, matching the host driver's f_of lambda.
__device__ inline Real f_lookup(const Real* f_table, int samples, Real pn) {
  const Real clamped = fmin(fmax(pn, Real{0}), Real{1});
  const int k = static_cast<int>(clamped * static_cast<Real>(samples - 1)
                                 + Real{0.5});
  return f_table[k];
}

__global__ void compute_field_kernel(GridView g, const Real* __restrict__ psi,
                                     const Real* __restrict__ d_r_field,
                                     const Real* __restrict__ d_z_field,
                                     const Real* __restrict__ f_table,
                                     int f_samples, Real psi_axis,
                                     Real psi_boundary, Real* __restrict__ b_r,
                                     Real* __restrict__ b_z,
                                     Real* __restrict__ b_phi,
                                     Real* __restrict__ b_pol) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= g.nr || j >= g.nz) return;

  const std::size_t k = g.index(i, j);
  const Real r = g.r(i);
  b_r[k] = -d_z_field[k] / r;
  b_z[k] = d_r_field[k] / r;
  const Real pn = normalized_flux_dev(psi[k], psi_axis, psi_boundary);
  b_phi[k] = f_lookup(f_table, f_samples, pn) / r;
  b_pol[k] = sqrt(b_r[k] * b_r[k] + b_z[k] * b_z[k]);
}

// One thread per (surface, ray). Writes the crossing point, or marks a miss.
__global__ void trace_rays_kernel(GridView g, const Real* __restrict__ psi,
                                  Real axis_r, Real axis_z, Real psi_axis,
                                  Real psi_boundary, int n_surfaces,
                                  int n_theta, Real* __restrict__ hit_r,
                                  Real* __restrict__ hit_z,
                                  int* __restrict__ hit_ok) {
  const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int s = static_cast<int>(blockIdx.y);
  if (t >= n_theta || s >= n_surfaces) return;

  const std::size_t slot = static_cast<std::size_t>(s) * n_theta + t;
  hit_ok[slot] = 0;

  const Real pn = static_cast<Real>(s + 1) / static_cast<Real>(n_surfaces + 1);
  const Real target = psi_axis + pn * (psi_boundary - psi_axis);

  const Real max_reach = fmax(g.r_max - g.r_min, g.z_max - g.z_min);
  const Real ds = max_reach / static_cast<Real>(kNSteps);
  const Real th = Real{2} * kPi * static_cast<Real>(t) / static_cast<Real>(n_theta);
  const Real ur = cos(th);
  const Real uz = sin(th);

  Real prev_val = psi_axis;
  Real prev_len = Real{0};

  for (int k = 1; k <= kNSteps; ++k) {
    const Real len = static_cast<Real>(k) * ds;
    const Real rr = axis_r + ur * len;
    const Real zz = axis_z + uz * len;
    if (rr <= g.r_min || rr >= g.r_max || zz <= g.z_min || zz >= g.z_max) break;

    const Real val = sample_bilinear_dev(g, psi, rr, zz);
    if ((prev_val - target) * (val - target) <= Real{0} && k > 1) {
      const Real denom = val - prev_val;
      const Real frac = denom != Real{0} ? (target - prev_val) / denom : Real{0};
      const Real hit_len = prev_len + frac * ds;
      hit_r[slot] = axis_r + ur * hit_len;
      hit_z[slot] = axis_z + uz * hit_len;
      hit_ok[slot] = 1;
      return;
    }
    prev_val = val;
    prev_len = len;
  }
}

// One thread per surface: compact the hits in ray order, then integrate around
// the resulting polygon. Both integrals are sequential by construction.
__global__ void surface_reduce_kernel(GridView g, const Real* __restrict__ hit_r,
                                      const Real* __restrict__ hit_z,
                                      const int* __restrict__ hit_ok,
                                      const Real* __restrict__ b_pol,
                                      const Real* __restrict__ b_phi,
                                      int n_surfaces, int n_theta,
                                      Real* __restrict__ out_r,
                                      Real* __restrict__ out_z,
                                      int* __restrict__ out_count,
                                      int* __restrict__ out_closed,
                                      Real* __restrict__ out_q,
                                      Real* __restrict__ out_area,
                                      Real* __restrict__ out_volume,
                                      Real* __restrict__ out_psi_n) {
  const int s = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (s >= n_surfaces) return;

  const std::size_t base = static_cast<std::size_t>(s) * n_theta;
  out_psi_n[s] = static_cast<Real>(s + 1) / static_cast<Real>(n_surfaces + 1);

  int n = 0;
  bool closed = true;
  for (int t = 0; t < n_theta; ++t) {
    if (hit_ok[base + t] == 0) {
      closed = false;
      continue;
    }
    out_r[base + n] = hit_r[base + t];
    out_z[base + n] = hit_z[base + t];
    ++n;
  }
  out_count[s] = n;
  out_closed[s] = closed ? 1 : 0;

  if (!closed || n < 3) {
    out_area[s] = Real{0};
    out_volume[s] = Real{0};
    out_q[s] = Real{0};
    return;
  }

  // Shoelace area, and the centroid radius Pappus needs for the volume.
  Real area2 = Real{0};
  Real cx = Real{0};
  for (int k = 0; k < n; ++k) {
    const int m = (k + 1) % n;
    const Real cross = out_r[base + k] * out_z[base + m]
                     - out_r[base + m] * out_z[base + k];
    area2 += cross;
    cx += (out_r[base + k] + out_r[base + m]) * cross;
  }
  const Real area = fabs(area2) * Real{0.5};
  const Real centroid_r = area2 != Real{0} ? cx / (Real{3} * area2) : Real{0};
  out_area[s] = area;
  out_volume[s] = Real{2} * kPi * fabs(centroid_r) * area;

  // q = (1/2pi) * contour integral of B_phi / (r B_pol) dl.
  Real acc = Real{0};
  for (int k = 0; k < n; ++k) {
    const int m = (k + 1) % n;
    const Real dr = out_r[base + m] - out_r[base + k];
    const Real dz = out_z[base + m] - out_z[base + k];
    const Real dl = sqrt(dr * dr + dz * dz);
    const Real rm = Real{0.5} * (out_r[base + k] + out_r[base + m]);
    const Real zm = Real{0.5} * (out_z[base + k] + out_z[base + m]);
    const Real bp = sample_bilinear_dev(g, b_pol, rm, zm);
    const Real bt = sample_bilinear_dev(g, b_phi, rm, zm);
    if (bp <= Real{0}) continue;
    acc += bt / (rm * bp) * dl;
  }
  out_q[s] = acc / (Real{2} * kPi);
}

// Single-threaded aggregate fold. The selections are first-wins and
// order-dependent, exactly like the critical-point merge.
__global__ void diagnostics_kernel(const Real* __restrict__ r,
                                   const Real* __restrict__ z,
                                   const int* __restrict__ count,
                                   const int* __restrict__ closed,
                                   const Real* __restrict__ q,
                                   const Real* __restrict__ volume,
                                   const Real* __restrict__ psi_n,
                                   int n_surfaces, int n_theta,
                                   GsDiagnostics* __restrict__ out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  GsDiagnostics d{};
  int last_closed = -1;
  int first_closed = -1;

  for (int s = 0; s < n_surfaces; ++s) {
    if (closed[s] == 0) {
      ++d.n_open_surfaces;
      continue;
    }
    if (first_closed < 0) first_closed = s;
    last_closed = s;
  }

  if (first_closed < 0) {
    *out = d;
    return;
  }
  d.have_closed = true;
  d.q_axis = q[first_closed];

  // q_95: the closed surface nearest psi_N = 0.95, strict `<` so ties go to the
  // innermost.
  int best = first_closed;
  Real best_gap = Real{1e30};
  for (int s = 0; s < n_surfaces; ++s) {
    if (closed[s] == 0) continue;
    const Real gap = fabs(psi_n[s] - Real{0.95});
    if (gap < best_gap) {
      best_gap = gap;
      best = s;
    }
  }
  d.q_95 = q[best];
  d.total_volume = volume[last_closed];
  d.psi_n_boundary = psi_n[last_closed];

  // Shape of the outermost closed surface.
  const std::size_t base = static_cast<std::size_t>(last_closed) * n_theta;
  const int n = count[last_closed];
  if (n >= 3) {
    Real r_lo = r[base], r_hi = r[base], z_lo = z[base], z_hi = z[base];
    Real r_at_zmax = r[base], r_at_zmin = r[base];
    for (int k = 0; k < n; ++k) {
      r_lo = fmin(r_lo, r[base + k]);
      r_hi = fmax(r_hi, r[base + k]);
      if (z[base + k] > z_hi) { z_hi = z[base + k]; r_at_zmax = r[base + k]; }
      if (z[base + k] < z_lo) { z_lo = z[base + k]; r_at_zmin = r[base + k]; }
    }
    d.r_major = Real{0.5} * (r_hi + r_lo);
    d.r_minor = Real{0.5} * (r_hi - r_lo);
    if (d.r_minor > Real{0}) {
      d.elongation = Real{0.5} * (z_hi - z_lo) / d.r_minor;
      const Real upper = (d.r_major - r_at_zmax) / d.r_minor;
      const Real lower = (d.r_major - r_at_zmin) / d.r_minor;
      d.triangularity = Real{0.5} * (upper + lower);
    }
  }

  *out = d;
}

dim3 grid_2d(int nx, int ny, dim3 block) {
  return dim3((static_cast<unsigned>(nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(ny) + block.y - 1) / block.y);
}

}  // namespace

void GsMagneticField::resize(const EllipticGrid& g) {
  const std::size_t n = g.size();
  b_r = backend::DeviceBuffer<Real>{n};
  b_z = backend::DeviceBuffer<Real>{n};
  b_phi = backend::DeviceBuffer<Real>{n};
  b_poloidal = backend::DeviceBuffer<Real>{n};
}

void GsFluxSurfaces::resize(int n_surfaces_in, int n_theta_in) {
  n_surfaces = n_surfaces_in;
  n_theta = n_theta_in;
  const std::size_t n = static_cast<std::size_t>(n_surfaces) * n_theta;
  r = backend::DeviceBuffer<Real>{n};
  z = backend::DeviceBuffer<Real>{n};
  count = backend::DeviceBuffer<int>{static_cast<std::size_t>(n_surfaces)};
  closed = backend::DeviceBuffer<int>{static_cast<std::size_t>(n_surfaces)};
  q = backend::DeviceBuffer<Real>{static_cast<std::size_t>(n_surfaces)};
  area = backend::DeviceBuffer<Real>{static_cast<std::size_t>(n_surfaces)};
  volume = backend::DeviceBuffer<Real>{static_cast<std::size_t>(n_surfaces)};
  psi_n = backend::DeviceBuffer<Real>{static_cast<std::size_t>(n_surfaces)};
}

void launch_gs_integrate_f_profile(const ProfileCoefficients& profile,
                                   Real f_vacuum, Real psi_axis, Real psi_bdry,
                                   Real profile_scale, int samples,
                                   Real* d_f_table, stream_t stream) {
  if (samples < 2) samples = 2;
  hipLaunchKernelGGL(integrate_f_kernel, dim3{1}, dim3{1}, 0,
                     static_cast<hipStream_t>(stream), profile, f_vacuum,
                     psi_axis, psi_bdry, profile_scale, samples, d_f_table);
  QUASAR_HIP_CHECK(hipGetLastError());
}

void launch_gs_compute_field(const EllipticGrid& g, const Real* d_psi,
                             const GsDerivativeFields& derivatives,
                             const Real* d_f_table, int f_samples,
                             Real psi_axis, Real psi_boundary,
                             GsMagneticField& out, stream_t stream) {
  const dim3 block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(compute_field_kernel, grid_2d(g.nr, g.nz, block), block, 0,
                     static_cast<hipStream_t>(stream), view_of(g), d_psi,
                     derivatives.d_r.device_ptr(), derivatives.d_z.device_ptr(),
                     d_f_table, f_samples, psi_axis, psi_boundary,
                     out.b_r.device_ptr(), out.b_z.device_ptr(),
                     out.b_phi.device_ptr(), out.b_poloidal.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());
}

void launch_gs_trace_surfaces(const EllipticGrid& g, const Real* d_psi,
                              const GsMagneticField& field, Real axis_r,
                              Real axis_z, Real psi_axis, Real psi_boundary,
                              GsFluxSurfaces& out, stream_t stream) {
  auto* s = static_cast<hipStream_t>(stream);
  const std::size_t n = static_cast<std::size_t>(out.n_surfaces) * out.n_theta;

  // Ray hits land in scratch first; compaction then rewrites them in place
  // into the dense output arrays.
  backend::DeviceBuffer<Real> hit_r{n};
  backend::DeviceBuffer<Real> hit_z{n};
  backend::DeviceBuffer<int> hit_ok{n};

  const dim3 ray_block{kBlock1D};
  const dim3 ray_grid{(static_cast<unsigned>(out.n_theta) + kBlock1D - 1) / kBlock1D,
                      static_cast<unsigned>(out.n_surfaces)};
  hipLaunchKernelGGL(trace_rays_kernel, ray_grid, ray_block, 0, s, view_of(g),
                     d_psi, axis_r, axis_z, psi_axis, psi_boundary,
                     out.n_surfaces, out.n_theta, hit_r.device_ptr(),
                     hit_z.device_ptr(), hit_ok.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  const unsigned sblock = 64;
  const dim3 sgrid{(static_cast<unsigned>(out.n_surfaces) + sblock - 1) / sblock};
  hipLaunchKernelGGL(surface_reduce_kernel, sgrid, dim3{sblock}, 0, s,
                     view_of(g), hit_r.device_ptr(), hit_z.device_ptr(),
                     hit_ok.device_ptr(), field.b_poloidal.device_ptr(),
                     field.b_phi.device_ptr(), out.n_surfaces, out.n_theta,
                     out.r.device_ptr(), out.z.device_ptr(),
                     out.count.device_ptr(), out.closed.device_ptr(),
                     out.q.device_ptr(), out.area.device_ptr(),
                     out.volume.device_ptr(), out.psi_n.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());
}

void launch_gs_reduce_diagnostics(GsFluxSurfaces& surfaces, GsDiagnostics* d_out,
                                  stream_t stream) {
  hipLaunchKernelGGL(diagnostics_kernel, dim3{1}, dim3{1}, 0,
                     static_cast<hipStream_t>(stream),
                     surfaces.r.device_ptr(), surfaces.z.device_ptr(),
                     surfaces.count.device_ptr(), surfaces.closed.device_ptr(),
                     surfaces.q.device_ptr(), surfaces.volume.device_ptr(),
                     surfaces.psi_n.device_ptr(), surfaces.n_surfaces,
                     surfaces.n_theta, d_out);
  QUASAR_HIP_CHECK(hipGetLastError());
}

GsDiagnostics copy_diagnostics_to_host(
    const backend::DeviceBuffer<GsDiagnostics>& buffer, stream_t stream) {
  backend::device_synchronize(stream);
  GsDiagnostics out{};
  buffer.copy_to_host(&out, 1);
  return out;
}

}  // namespace quasar::equilibrium
