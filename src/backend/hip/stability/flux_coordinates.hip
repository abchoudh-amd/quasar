// PEST straight-field-line coordinates from traced flux surfaces.
//
// See physics/stability/kernels.hpp for why theta* is the antiderivative of the
// q integrand, and why a straight-field-line angle is what makes the
// discretized operator banded.
//
// -- Three stages, each parallelized where it admits it ------------------------
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

#include <cstddef>

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

// Cubic Lagrange interpolation on four equally spaced samples, with the
// evaluation point at offset t in [0, 1] between p1 and p2.
// Bilinear sampling of an equilibrium-grid field, matching the sampler the
// equilibrium module uses for the same quantities.
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
// What actually sets the accuracy is the equilibrium grid: refining it
// 33 -> 65 -> 129 moves the innermost surface 2.7e-2 -> 8.9e-3 -> 2.1e-3, and
// the bulk to a ~1.1e-3 floor. Bilinear is kept for consistency with the
// equilibrium module and because the alternative bought nothing.
__device__ inline Real sample_field(const GridView& g, const Real* f, Real r,
                                    Real z) {
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

  if (surf_closed[s] == 0 || n < 4) {
    out_valid[s] = 0;
    out_q[s] = Real{0};
    for (int j = 0; j < n_theta; ++j) {
      out_r[obase + j] = Real{0};
      out_z[obase + j] = Real{0};
    }
    return;
  }

  // -- Stage 1: cumulative toroidal advance around the contour ---------------
  // scratch_theta[k] holds the running integral of B_phi/(r B_pol) dl up to
  // contour point k, with entry 0 equal to zero and entry n the full circuit.
  Real acc = Real{0};
  scratch_theta[cbase] = Real{0};
  for (int k = 0; k < n; ++k) {
    const int m = (k + 1) % n;
    const Real dr = surf_r[cbase + m] - surf_r[cbase + k];
    const Real dz = surf_z[cbase + m] - surf_z[cbase + k];
    const Real dl = sqrt(dr * dr + dz * dz);
    const Real rm = Real{0.5} * (surf_r[cbase + k] + surf_r[cbase + m]);
    const Real zm = Real{0.5} * (surf_z[cbase + k] + surf_z[cbase + m]);
    const Real bp = sample_field(g, b_pol, rm, zm);
    const Real bt = sample_field(g, b_phi, rm, zm);
    if (bp > Real{0}) acc += bt / (rm * bp) * dl;
    if (k + 1 < n) scratch_theta[cbase + k + 1] = acc;
  }
  const Real total = acc;

  // q is the circuit integral over 2*pi -- the same quantity compute_q returns,
  // recomputed here so theta* and q come from one accumulation rather than two
  // that could disagree at the last bits and leave theta* not quite reaching
  // 2*pi.
  const Real q = total / (Real{2} * kPi);
  out_q[s] = q;

  if (!(fabs(total) > Real{0})) {
    out_valid[s] = 0;
    for (int j = 0; j < n_theta; ++j) {
      out_r[obase + j] = Real{0};
      out_z[obase + j] = Real{0};
    }
    return;
  }

  // Normalize to [0, 2*pi). Dividing by the measured total rather than by
  // 2*pi*q guarantees theta* closes exactly on the circuit even where the
  // contour is coarsely resolved.
  const Real norm = Real{2} * kPi / total;
  for (int k = 0; k < n; ++k) scratch_theta[cbase + k] *= norm;

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

    out_r[obase + j] = surf_r[cbase + k]
                     + frac * (surf_r[cbase + k1] - surf_r[cbase + k]);
    out_z[obase + j] = surf_z[cbase + k]
                     + frac * (surf_z[cbase + k1] - surf_z[cbase + k]);
  }
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
  if (valid[i] == 0 || q[i] == Real{0}) return;

  Real worst = Real{0};
  for (int j = 0; j < n_theta; ++j) {
    const std::size_t k = static_cast<std::size_t>(i) * n_theta + j;
    const Real rr = r[k];
    const Real bp = sample_field(g, b_pol, rr, z[k]);
    const Real bt = sample_field(g, b_phi, rr, z[k]);
    if (!(bp > Real{0})) continue;

    // Poloidal arc length per unit theta*, then the toroidal advance per unit
    // theta* the local field produces.
    const Real dl_dtheta = sqrt(dr_dtheta[k] * dr_dtheta[k]
                              + dz_dtheta[k] * dz_dtheta[k]);
    const Real dphi_dtheta = bt / (rr * bp) * dl_dtheta;
    worst = fmax(worst, fabs(dphi_dtheta - q[i]) / fabs(q[i]));
  }
  out[i] = worst;
}

dim3 grid_2d(int nx, int ny, dim3 block) {
  return dim3((static_cast<unsigned>(nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(ny) + block.y - 1) / block.y);
}

}  // namespace

void FluxCoordinateGrid::resize(int n_psi_in, int n_theta_in) {
  n_psi = n_psi_in;
  n_theta = n_theta_in;
  const std::size_t n = static_cast<std::size_t>(n_psi) * n_theta;
  const std::size_t s = static_cast<std::size_t>(n_psi);

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
}

void launch_build_flux_coordinates(const EllipticGrid& g,
                                   const equilibrium::GsFluxSurfaces& surfaces,
                                   const equilibrium::GsMagneticField& field,
                                   Real /*axis_r*/, Real /*axis_z*/,
                                   FluxCoordinateGrid& out, stream_t stream) {
  auto* s = static_cast<hipStream_t>(stream);
  const int n_surfaces = surfaces.n_surfaces;
  const int n_contour = surfaces.n_theta;

  backend::DeviceBuffer<Real> scratch{
      static_cast<std::size_t>(n_surfaces) * n_contour};

  const dim3 sblock{kBlock1D};
  const dim3 sgrid{(static_cast<unsigned>(n_surfaces) + kBlock1D - 1) / kBlock1D};
  hipLaunchKernelGGL(build_surface_kernel, sgrid, sblock, 0, s, view_of(g),
                     surfaces.r.device_ptr(), surfaces.z.device_ptr(),
                     surfaces.count.device_ptr(), surfaces.closed.device_ptr(),
                     surfaces.psi_n.device_ptr(),
                     field.b_poloidal.device_ptr(), field.b_phi.device_ptr(),
                     n_surfaces, n_contour, out.n_theta, out.r.device_ptr(),
                     out.z.device_ptr(), out.q.device_ptr(),
                     out.psi_n.device_ptr(), out.valid.device_ptr(),
                     scratch.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  const dim3 mblock{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(metric_kernel, grid_2d(out.n_theta, out.n_psi, mblock),
                     mblock, 0, s, out.r.device_ptr(), out.z.device_ptr(),
                     out.psi_n.device_ptr(), out.valid.device_ptr(), out.n_psi,
                     out.n_theta, out.dr_dpsi.device_ptr(),
                     out.dz_dpsi.device_ptr(), out.dr_dtheta.device_ptr(),
                     out.dz_dtheta.device_ptr(), out.jacobian.device_ptr(),
                     out.g_psipsi.device_ptr(), out.g_psitheta.device_ptr(),
                     out.g_thetatheta.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  // The scratch buffer must outlive the kernels that read it.
  backend::device_synchronize(stream);
}

void launch_check_straightness(const FluxCoordinateGrid& coords,
                               const EllipticGrid& g,
                               const equilibrium::GsMagneticField& field,
                               Real* d_deviation, stream_t stream) {
  const dim3 block{kBlock1D};
  const dim3 grid{(static_cast<unsigned>(coords.n_psi) + kBlock1D - 1) / kBlock1D};
  hipLaunchKernelGGL(straightness_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream), view_of(g),
                     coords.r.device_ptr(), coords.z.device_ptr(),
                     coords.dr_dtheta.device_ptr(),
                     coords.dz_dtheta.device_ptr(), coords.q.device_ptr(),
                     coords.valid.device_ptr(), coords.n_psi, coords.n_theta,
                     field.b_poloidal.device_ptr(), field.b_phi.device_ptr(),
                     d_deviation);
  QUASAR_HIP_CHECK(hipGetLastError());
}

}  // namespace quasar::stability
