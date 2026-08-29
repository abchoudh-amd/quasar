// Device source terms for the Grad-Shafranov Picard/Newton iteration: the
// toroidal current density, its I_p normalization, the elliptic right-hand
// side, the Newton Jacobian diagonal, and the two field-update helpers.
//
// -- Profiles cross as data, not as objects -----------------------------------
// IEquilibriumProfile is virtual, and a vtable pointer is meaningless on the
// device: the object would have to be constructed there to have a valid one.
// The registry and the polymorphism stay on the host, and the selected profile
// is lowered to a flat ProfileCoefficients POD at configuration time (see
// equilibrium_profile.hpp). The kernels below evaluate a plain polynomial with
// no indirection, and the POD is small enough to pass by value.
//
// -- Bit-exactness ------------------------------------------------------------
// Every kernel here is elementwise -- one thread per node, no reduction, no
// cross-thread communication -- so each output is produced by the same sequence
// of operations the host performs on that node. All the equivalence tests are
// equalities.
//
// The Horner loops in particular run in the host's descending-index order. That
// is not incidental: ascending Horner is a different (and slightly less
// accurate) evaluation, and reversing it would silently break the equality.

#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace quasar::equilibrium {
namespace {

inline constexpr unsigned kBlock2D = 16;

struct GridView {
  int nr;
  int nz;
  Real r_min;
  Real dr;

  __device__ Real r(int i) const { return r_min + static_cast<Real>(i) * dr; }
  __device__ bool on_boundary(int i, int j) const {
    return i == 0 || j == 0 || i == nr - 1 || j == nz - 1;
  }
  __device__ std::size_t index(int i, int j) const {
    return static_cast<std::size_t>(i) + static_cast<std::size_t>(nr) * j;
  }
};

GridView view_of(const EllipticGrid& g) {
  return GridView{g.nr, g.nz, g.r_min, g.dr()};
}

// Horner, descending index -- the order PolynomialProfile::horner uses.
__device__ inline Real horner(const Real* c, int n, Real x) {
  Real acc = Real{0};
  for (int k = n; k-- > 0;) acc = acc * x + c[k];
  return acc;
}

// Derivative of the same polynomial, also descending, stopping at k == 1.
__device__ inline Real horner_derivative(const Real* c, int n, Real x) {
  Real acc = Real{0};
  for (int k = n; k-- > 1;) acc = acc * x + c[k] * static_cast<Real>(k);
  return acc;
}

// psi_N = (psi - psi_axis) / (psi_bdry - psi_axis), clamped to [0, 1].
// Outside the plasma psi_N clamps to 1, where the default profiles vanish, so
// no current is driven there.
__device__ inline Real normalized_flux_dev(Real psi, Real psi_axis,
                                           Real psi_bdry) {
  const Real denom = psi_bdry - psi_axis;
  if (denom == Real{0}) return Real{1};
  const Real n = (psi - psi_axis) / denom;
  return fmin(fmax(n, Real{0}), Real{1});
}

__global__ void build_current_kernel(GridView g, const Real* __restrict__ psi,
                                     ProfileCoefficients profile, Real psi_axis,
                                     Real psi_boundary,
                                     Real* __restrict__ j_phi) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= g.nr || j >= g.nz) return;
  const std::size_t k = g.index(i, j);

  // The host assigns the whole field to zero first, then fills the interior.
  if (g.on_boundary(i, j)) {
    j_phi[k] = Real{0};
    return;
  }

  const Real pn = normalized_flux_dev(psi[k], psi_axis, psi_boundary);
  if (pn >= Real{1}) {  // outside the last closed surface: no current
    j_phi[k] = Real{0};
    return;
  }
  const Real r = g.r(i);
  j_phi[k] = r * horner(profile.p_coeffs, profile.n_p, pn)
           + horner(profile.f_coeffs, profile.n_f, pn) / (kMu0 * r);
}

__global__ void scale_field_kernel(Real scale, Real* field, std::size_t n) {
  const std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  field[k] *= scale;
}

__global__ void build_rhs_kernel(GridView g, const Real* __restrict__ j_phi,
                                 Real* __restrict__ rhs) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= g.nr || j >= g.nz) return;
  const std::size_t k = g.index(i, j);

  // Boundary entries stay zero: the host allocates rhs zero-filled and writes
  // only the interior, so writing an explicit zero here matches while keeping
  // the kernel self-contained rather than relying on the caller's allocation.
  rhs[k] = g.on_boundary(i, j) ? Real{0}
                               : -kMu0 * g.r(i) * j_phi[k];
}

__global__ void jacobian_diagonal_kernel(GridView g,
                                         const Real* __restrict__ psi,
                                         ProfileCoefficients profile,
                                         Real psi_axis, Real psi_boundary,
                                         Real profile_scale,
                                         Real* __restrict__ jac) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= g.nr || j >= g.nz) return;
  const std::size_t k = g.index(i, j);

  const Real denom = psi_boundary - psi_axis;
  if (g.on_boundary(i, j) || denom == Real{0}) {
    jac[k] = Real{0};
    return;
  }

  const Real pn = normalized_flux_dev(psi[k], psi_axis, psi_boundary);
  // Both ends are excluded: pn >= 1 is outside the plasma, and pn <= 0 is the
  // clamped axis side where the chain rule through psi_N does not apply.
  if (pn >= Real{1} || pn <= Real{0}) {
    jac[k] = Real{0};
    return;
  }
  const Real r = g.r(i);
  jac[k] = profile_scale
         * (-kMu0 * r * r * horner_derivative(profile.p_coeffs, profile.n_p, pn)
            - horner_derivative(profile.f_coeffs, profile.n_f, pn))
         / denom;
}

__global__ void blend_kernel(Real* target, const Real* __restrict__ candidate,
                             Real weight, std::size_t n) {
  const std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  target[k] += weight * (candidate[k] - target[k]);
}

__global__ void restore_boundary_kernel(GridView g,
                                        const Real* __restrict__ source,
                                        Real* __restrict__ target) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= g.nr || j >= g.nz) return;
  if (!g.on_boundary(i, j)) return;
  const std::size_t k = g.index(i, j);
  target[k] = source[k];
}

// Seeded plasma column: a quadratic-with-compact-support well.
//
// The (1-s^2)^2 form is C^1 at the seed edge, so it does not inject a kink the
// sixth-order residual would then have to smooth away over many iterations.
// Everything else -- the centre, the radius, the depth scaling by mu0*|I_p|,
// and the sign -- is resolved by the caller; the kernel carries no policy.
__global__ void plasma_seed_kernel(GridView g, Real rc, Real zc, Real a,
                                   Real depth, Real sign, Real dz, Real z_min,
                                   Real* __restrict__ psi) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= g.nr || j >= g.nz) return;
  if (g.on_boundary(i, j)) return;  // boundary is Green's-function data

  const Real dr = (g.r(i) - rc) / a;
  const Real dzn = ((z_min + static_cast<Real>(j) * dz) - zc) / a;
  const Real s2 = dr * dr + dzn * dzn;
  if (s2 >= Real{1}) return;

  const Real w = Real{1} - s2;
  psi[g.index(i, j)] += depth * w * w * sign;
}

__global__ void axpy_interior_kernel(GridView g, Real* __restrict__ x,
                                     const Real* __restrict__ delta,
                                     Real alpha) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + 1;
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) + 1;
  if (i >= g.nr - 1 || j >= g.nz - 1) return;
  const std::size_t k = g.index(i, j);
  x[k] += alpha * delta[k];
}

dim3 grid_2d(int nx, int ny, dim3 block) {
  return dim3((static_cast<unsigned>(nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(ny) + block.y - 1) / block.y);
}

dim3 grid_1d(std::size_t n, unsigned block) {
  return dim3(static_cast<unsigned>((n + block - 1) / block));
}

}  // namespace

void launch_gs_add_plasma_seed(const EllipticGrid& g, Real r_center,
                               Real z_center, Real minor_radius, Real depth,
                               Real sign, Real* d_psi, stream_t stream) {
  const dim3 block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(plasma_seed_kernel, grid_2d(g.nr, g.nz, block), block, 0,
                     static_cast<hipStream_t>(stream), view_of(g), r_center,
                     z_center, minor_radius, depth, sign, g.dz(), g.z_min,
                     d_psi);
  QUASAR_HIP_CHECK(hipGetLastError());
}

void launch_gs_axpy_interior(const EllipticGrid& g, Real* d_x,
                             const Real* d_delta, Real alpha,
                             stream_t stream) {
  const dim3 block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(axpy_interior_kernel, grid_2d(g.nr - 2, g.nz - 2, block),
                     block, 0, static_cast<hipStream_t>(stream), view_of(g),
                     d_x, d_delta, alpha);
  QUASAR_HIP_CHECK(hipGetLastError());
}

void launch_gs_build_current(const EllipticGrid& g, const Real* d_psi,
                             const ProfileCoefficients& profile, Real psi_axis,
                             Real psi_boundary, Real* d_j_phi,
                             stream_t stream) {
  const dim3 block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(build_current_kernel, grid_2d(g.nr, g.nz, block), block, 0,
                     static_cast<hipStream_t>(stream), view_of(g), d_psi,
                     profile, psi_axis, psi_boundary, d_j_phi);
  QUASAR_HIP_CHECK(hipGetLastError());
}

void launch_gs_scale_field(const EllipticGrid& g, Real scale, Real* d_field,
                           stream_t stream) {
  constexpr unsigned kBlock = 256;
  hipLaunchKernelGGL(scale_field_kernel, grid_1d(g.size(), kBlock),
                     dim3{kBlock}, 0, static_cast<hipStream_t>(stream), scale,
                     d_field, g.size());
  QUASAR_HIP_CHECK(hipGetLastError());
}

void launch_gs_build_rhs(const EllipticGrid& g, const Real* d_j_phi,
                         Real* d_rhs, stream_t stream) {
  const dim3 block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(build_rhs_kernel, grid_2d(g.nr, g.nz, block), block, 0,
                     static_cast<hipStream_t>(stream), view_of(g), d_j_phi,
                     d_rhs);
  QUASAR_HIP_CHECK(hipGetLastError());
}

void launch_gs_build_jacobian_diagonal(const EllipticGrid& g,
                                       const Real* d_psi,
                                       const ProfileCoefficients& profile,
                                       Real psi_axis, Real psi_boundary,
                                       Real profile_scale, Real* d_jac,
                                       stream_t stream) {
  const dim3 block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(jacobian_diagonal_kernel, grid_2d(g.nr, g.nz, block),
                     block, 0, static_cast<hipStream_t>(stream), view_of(g),
                     d_psi, profile, psi_axis, psi_boundary, profile_scale,
                     d_jac);
  QUASAR_HIP_CHECK(hipGetLastError());
}

void launch_gs_blend(const EllipticGrid& g, Real* d_target,
                     const Real* d_candidate, Real weight, stream_t stream) {
  constexpr unsigned kBlock = 256;
  hipLaunchKernelGGL(blend_kernel, grid_1d(g.size(), kBlock), dim3{kBlock}, 0,
                     static_cast<hipStream_t>(stream), d_target, d_candidate,
                     weight, g.size());
  QUASAR_HIP_CHECK(hipGetLastError());
}

void launch_gs_restore_boundary(const EllipticGrid& g, const Real* d_source,
                                Real* d_target, stream_t stream) {
  const dim3 block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(restore_boundary_kernel, grid_2d(g.nr, g.nz, block), block,
                     0, static_cast<hipStream_t>(stream), view_of(g), d_source,
                     d_target);
  QUASAR_HIP_CHECK(hipGetLastError());
}

}  // namespace quasar::equilibrium
