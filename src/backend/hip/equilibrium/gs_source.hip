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
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace quasar::equilibrium {
namespace {

inline constexpr unsigned kBlock2D = 16;

struct GridView {
  int nr;
  int nz;
  Real r_min;
  Real z_min;
  Real dr;
  Real dz;

  __device__ Real r(int i) const { return r_min + static_cast<Real>(i) * dr; }
  __device__ Real z(int j) const { return z_min + static_cast<Real>(j) * dz; }
  __device__ bool on_boundary(int i, int j) const {
    return i == 0 || j == 0 || i == nr - 1 || j == nz - 1;
  }
  __device__ std::size_t index(int i, int j) const {
    return static_cast<std::size_t>(i) + static_cast<std::size_t>(nr) * j;
  }
};

GridView view_of(const EllipticGrid& g) {
  return GridView{g.nr, g.nz, g.r_min, g.z_min, g.dr(), g.dz()};
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
  if (!isfinite(psi) || !isfinite(psi_axis) || !isfinite(psi_bdry)) {
    return Real{1};
  }
  const Real denom = psi_bdry - psi_axis;
  if (denom == Real{0} || !isfinite(denom)) return Real{1};
  const Real numerator = psi - psi_axis;
  if (!isfinite(numerator)) return Real{1};
  const Real n = numerator / denom;
  if (!isfinite(n)) return Real{1};
  return fmin(fmax(n, Real{0}), Real{1});
}

__device__ inline bool plasma_eligible(GridView g,
                                       const Real* __restrict__ psi, int i,
                                       int j, Real psi_axis,
                                       Real psi_boundary) {
  if (g.on_boundary(i, j)) return false;
  const Real value = psi[g.index(i, j)];
  const Real denominator = psi_boundary - psi_axis;
  if (!isfinite(value) || !isfinite(denominator)
      || denominator == Real{0}) {
    return false;
  }
  return (value - psi_axis) / denominator < Real{1};
}

struct EdgeDerivativeView {
  const Real* d_r;
  const Real* d_z;
  const Real* d_rr;
  const Real* d_zz;
};

__device__ inline bool bezier_edge_strictly_inside(const Real controls[6]) {
  constexpr int kMaxDepth = 8;
  constexpr int kStackCapacity = kMaxDepth + 2;
  constexpr Real kTolerance =
      Real{128} * std::numeric_limits<Real>::epsilon();
  struct Segment {
    Real controls[6];
    int depth;
  };

  Segment stack[kStackCapacity];
  int top = 1;
  stack[0].depth = 0;
  for (int k = 0; k < 6; ++k) stack[0].controls[k] = controls[k];

  while (top > 0) {
    const Segment segment = stack[--top];
    bool all_inside = true;
    for (int k = 0; k < 6; ++k) {
      const Real value = segment.controls[k];
      if (!isfinite(value)) return false;
      if (!(value < -kTolerance)) all_inside = false;
    }
    if (all_inside) continue;
    if (segment.depth == kMaxDepth) return false;

    Real work[6];
    Segment left;
    Segment right;
    for (int k = 0; k < 6; ++k) work[k] = segment.controls[k];
    left.depth = segment.depth + 1;
    right.depth = segment.depth + 1;
    left.controls[0] = work[0];
    right.controls[5] = work[5];
    for (int level = 1; level < 6; ++level) {
      for (int k = 0; k < 6 - level; ++k) {
        work[k] = Real{0.5} * (work[k] + work[k + 1]);
      }
      left.controls[level] = work[0];
      right.controls[5 - level] = work[5 - level];
    }
    stack[top++] = right;
    stack[top++] = left;
  }
  return true;
}

__device__ inline bool flux_edge_strictly_inside(
    GridView g, const Real* __restrict__ psi, EdgeDerivativeView derivatives,
    int i0, int j0, int i1, int j1, Real psi_axis, Real psi_boundary) {
  const Real denom = psi_boundary - psi_axis;
  const bool radial = j0 == j1;
  const Real h = radial ? g.r(i1) - g.r(i0) : g.z(j1) - g.z(j0);
  const std::size_t k0 = g.index(i0, j0);
  const std::size_t k1 = g.index(i1, j1);
  const Real* first = radial ? derivatives.d_r : derivatives.d_z;
  const Real* second = radial ? derivatives.d_rr : derivatives.d_zz;

  const Real q0 = (psi[k0] - psi_boundary) / denom;
  const Real q1 = (psi[k1] - psi_boundary) / denom;
  const Real v0 = h * first[k0] / denom;
  const Real v1 = h * first[k1] / denom;
  const Real a0 = h * h * second[k0] / denom;
  const Real a1 = h * h * second[k1] / denom;
  const Real controls[6] = {
      q0,
      q0 + v0 / Real{5},
      q0 + Real{2} * v0 / Real{5} + a0 / Real{20},
      q1 - Real{2} * v1 / Real{5} + a1 / Real{20},
      q1 - v1 / Real{5},
      q1};
  return bezier_edge_strictly_inside(controls);
}

// The grid sizes used by the free-boundary solve contain only a few thousand
// nodes. A single device thread can therefore perform the complete ordered BFS
// more cheaply than a host-synchronized sequence of wavefront launches, while
// keeping psi and the component label resident on the owning device.
__global__ void build_plasma_mask_kernel(
    GridView g, const Real* __restrict__ psi, EdgeDerivativeView derivatives,
    Real axis_r, Real axis_z, Real psi_axis, Real psi_boundary,
    int* __restrict__ mask, std::size_t* __restrict__ queue) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  const std::size_t size = static_cast<std::size_t>(g.nr) * g.nz;
  for (std::size_t k = 0; k < size; ++k) mask[k] = 0;
  if (!isfinite(axis_r) || !isfinite(axis_z)) return;

  std::size_t seed = size;
  Real nearest = std::numeric_limits<Real>::infinity();
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      if (!plasma_eligible(g, psi, i, j, psi_axis, psi_boundary)) continue;
      const Real di = (g.r(i) - axis_r) / g.dr;
      const Real dj = (g.z(j) - axis_z) / g.dz;
      const Real distance = di * di + dj * dj;
      if (distance < nearest) {
        nearest = distance;
        seed = g.index(i, j);
      }
    }
  }
  if (seed == size) return;

  std::size_t head = 0;
  std::size_t tail = 0;
  queue[tail++] = seed;
  mask[seed] = 1;
  while (head < tail) {
    const std::size_t current = queue[head++];
    const int i = static_cast<int>(current % static_cast<std::size_t>(g.nr));
    const int j = static_cast<int>(current / static_cast<std::size_t>(g.nr));
    const int neighbour_i[4] = {i - 1, i + 1, i, i};
    const int neighbour_j[4] = {j, j, j - 1, j + 1};
    for (int direction = 0; direction < 4; ++direction) {
      const int ni = neighbour_i[direction];
      const int nj = neighbour_j[direction];
      if (!plasma_eligible(g, psi, ni, nj, psi_axis, psi_boundary)) continue;
      if (!flux_edge_strictly_inside(g, psi, derivatives, i, j, ni, nj,
                                     psi_axis, psi_boundary)) {
        continue;
      }
      const std::size_t next = g.index(ni, nj);
      if (mask[next] != 0) continue;
      mask[next] = 1;
      queue[tail++] = next;
    }
  }
}

__global__ void build_current_kernel(GridView g, const Real* __restrict__ psi,
                                     ProfileCoefficients profile, Real psi_axis,
                                     Real psi_boundary,
                                     const int* __restrict__ plasma_mask,
                                     Real* __restrict__ j_phi) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= g.nr || j >= g.nz) return;
  const std::size_t k = g.index(i, j);

  // The host assigns the whole field to zero first, then fills the interior.
  if (g.on_boundary(i, j) || plasma_mask[k] == 0) {
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
                                         const int* __restrict__ plasma_mask,
                                         Real* __restrict__ jac) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= g.nr || j >= g.nz) return;
  const std::size_t k = g.index(i, j);

  const Real denom = psi_boundary - psi_axis;
  if (g.on_boundary(i, j) || plasma_mask[k] == 0 || denom == Real{0}) {
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

}  // namespace

void GsPlasmaMaskScratch::resize(const EllipticGrid& g) {
  g.validate();
  const std::size_t n = g.size();
  (void)backend::detail::checked_buffer_bytes<int>(n);
  (void)backend::detail::checked_buffer_bytes<std::size_t>(n);

  backend::DeviceBuffer<int> next_mask{n};
  backend::DeviceBuffer<std::size_t> next_queue{
      n, backend::uninitialized};
  mask = std::move(next_mask);
  queue = std::move(next_queue);
}

void launch_gs_build_plasma_mask(const EllipticGrid& g, const Real* d_psi,
                                 const GsDerivativeFields& derivatives,
                                 Real axis_r, Real axis_z, Real psi_axis,
                                 Real psi_boundary,
                                 GsPlasmaMaskScratch& scratch,
                                 stream_t stream) {
  g.validate();
  if (d_psi == nullptr) {
    throw std::invalid_argument{
        "launch_gs_build_plasma_mask: psi buffer must not be null"};
  }

  const std::size_t n = g.size();
  require_size(scratch.mask, n, "GsPlasmaMaskScratch::mask");
  require_size(scratch.queue, n, "GsPlasmaMaskScratch::queue");
  require_size(derivatives.d_r, n, "GsDerivativeFields::d_r");
  require_size(derivatives.d_z, n, "GsDerivativeFields::d_z");
  require_size(derivatives.d_rr, n, "GsDerivativeFields::d_rr");
  require_size(derivatives.d_zz, n, "GsDerivativeFields::d_zz");

  const int owner = scratch.mask.owner_device();
  if (owner < 0) {
    throw std::invalid_argument{
        "launch_gs_build_plasma_mask: storage is not device-resident"};
  }
  require_owner(scratch.queue, owner, "GsPlasmaMaskScratch::queue");
  require_owner(derivatives.d_r, owner, "GsDerivativeFields::d_r");
  require_owner(derivatives.d_z, owner, "GsDerivativeFields::d_z");
  require_owner(derivatives.d_rr, owner, "GsDerivativeFields::d_rr");
  require_owner(derivatives.d_zz, owner, "GsDerivativeFields::d_zz");
  if (backend::current_device() != owner) {
    throw std::invalid_argument{
        "launch_gs_build_plasma_mask: storage is not on the current device"};
  }

  const EdgeDerivativeView derivative_view{
      derivatives.d_r.device_ptr(), derivatives.d_z.device_ptr(),
      derivatives.d_rr.device_ptr(), derivatives.d_zz.device_ptr()};
  hipLaunchKernelGGL(build_plasma_mask_kernel, dim3{1}, dim3{1}, 0,
                     static_cast<hipStream_t>(stream), view_of(g), d_psi,
                     derivative_view, axis_r, axis_z, psi_axis, psi_boundary,
                     scratch.mask.device_ptr(), scratch.queue.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());
}

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
                             Real psi_boundary, const int* d_plasma_mask,
                             Real* d_j_phi,
                             stream_t stream) {
  const dim3 block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(build_current_kernel, grid_2d(g.nr, g.nz, block), block, 0,
                     static_cast<hipStream_t>(stream), view_of(g), d_psi,
                     profile, psi_axis, psi_boundary, d_plasma_mask, d_j_phi);
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
                                       Real profile_scale,
                                       const int* d_plasma_mask, Real* d_jac,
                                       stream_t stream) {
  const dim3 block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(jacobian_diagonal_kernel, grid_2d(g.nr, g.nz, block),
                     block, 0, static_cast<hipStream_t>(stream), view_of(g),
                     d_psi, profile, psi_axis, psi_boundary, profile_scale,
                     d_plasma_mask, d_jac);
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
