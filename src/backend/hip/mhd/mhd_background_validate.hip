// Device sweeps that validate a prescribed MHD background field.
//
// These replace three host loops that downloaded b0x/b0y/b0z in full and walked
// the padded grid on the CPU. The arithmetic is unchanged: every metric comes
// from quasar/numerics/mhd_background_metrics.hpp, which is the same code the
// host used, promoted to QUASAR_HOST_DEVICE and moved onto the numerics axis so
// both sides can share one definition rather than drifting.
//
// Reduction choices, and why they do not violate the no-float-atomics rule that
// the rest of the tree follows:
//
//   * The two ScaledValue maxima and the curl-free Real maximum use the same
//     two-pass block tree as every other reduction here. Maximum is associative
//     and rounds nothing, so the tree and a sequential loop agree bit for bit --
//     this is an equality port, not an accuracy-ordering one.
//   * The boolean outcomes (a non-finite sample, an exactly-nonzero stored
//     difference, a violated closure rule) use INTEGER atomics. Bitwise or/and
//     are exact, commutative and associative, so an integer atomic is
//     order-independent and costs nothing in reproducibility. The prohibition
//     is on floating-point atomics, which are neither.
//
// The maxima deliberately reproduce std::max's NaN behaviour rather than
// fmax's. std::max(a, NaN) returns a, so a NaN defect is dropped while an
// infinite one propagates; the host check that follows tests isfinite anyway.
// Silently "improving" this would change which inputs are rejected.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/numerics/mhd_background_metrics.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/reduction_detail.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace quasar::mhd {
namespace {

namespace hd = ::quasar::backend::hip_detail;
namespace nm = ::quasar::numerics;

inline constexpr unsigned kThreads = hd::kReduceThreads;

// Boundary mode codes shared with the launcher. Kept as plain ints across the
// ABI so the kernel header stays free of the boundary axis's string vocabulary.
inline constexpr int kModeIgnored = 0;
inline constexpr int kModePeriodic = 1;
inline constexpr int kModeWall = 2;
inline constexpr int kModeAxis = 3;

// ScaledValue carries default member initializers, so it cannot be declared as
// a __shared__ array. This is the same aggregate without them.
struct SharedScaled {
  Real mantissa;
  int exponent;
};

__device__ inline nm::ScaledValue widen(const SharedScaled& v) {
  return nm::ScaledValue{v.mantissa, v.exponent};
}

__device__ inline SharedScaled narrow(const nm::ScaledValue& v) {
  return SharedScaled{v.mantissa, v.exponent};
}

// std::max semantics exactly: returns the first argument unless it is strictly
// less than the second. NaN compares false, so NaN is dropped and infinity is
// kept -- see the file header.
__device__ inline Real host_like_max(Real a, Real b) { return (a < b) ? b : a; }

__device__ inline void merge_scaled_max(SharedScaled& into,
                                        const SharedScaled& candidate) {
  nm::ScaledValue maximum = widen(into);
  nm::retain_scaled_abs_max(widen(candidate), maximum);
  into = narrow(maximum);
}

template <unsigned Threads>
__device__ inline void block_scaled_max(SharedScaled* sdata, unsigned tid) {
  for (unsigned s = Threads / 2; s > 0; s >>= 1) {
    if (tid < s) merge_scaled_max(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
}

template <unsigned Threads>
__device__ inline void block_real_max(Real* sdata, unsigned tid) {
  for (unsigned s = Threads / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = host_like_max(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
}

// -- Solenoidality -----------------------------------------------------------

__global__ QUASAR_LAUNCH_BOUNDS_256 void solenoidal_pass1(
    Grid2D grid, int cylindrical, const Real* __restrict__ b0x,
    const Real* __restrict__ b0y, const Real* __restrict__ b0z,
    std::size_t storage, SharedScaled* __restrict__ partial_residual,
    SharedScaled* __restrict__ partial_scale, int* __restrict__ nonfinite) {
  __shared__ SharedScaled s_residual[kThreads];
  __shared__ SharedScaled s_scale[kThreads];
  const unsigned tid = threadIdx.x;

  // Finiteness covers all of padded storage, not just the interior: a
  // non-finite ghost would poison a stencil later even though it never enters
  // the divergence sum.
  for (std::size_t k = blockIdx.x * blockDim.x + tid; k < storage;
       k += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (!(isfinite(b0x[k]) && isfinite(b0y[k]) && isfinite(b0z[k]))) {
      atomicOr(nonfinite, 1);
    }
  }

  const std::size_t n_cells = static_cast<std::size_t>(grid.nx)
                            * static_cast<std::size_t>(grid.ny);
  SharedScaled residual_linf{Real{0}, 0};
  SharedScaled scale_linf{Real{0}, 0};
  for (std::size_t t = blockIdx.x * blockDim.x + tid; t < n_cells;
       t += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const int i = static_cast<int>(t % static_cast<std::size_t>(grid.nx));
    const int j = static_cast<int>(t / static_cast<std::size_t>(grid.nx));

    const Real bx_lo = b0x[grid.index(i, j)];
    const Real bx_hi = b0x[grid.index(i + 1, j)];
    const Real by_lo = b0y[grid.index(i, j)];
    const Real by_hi = b0y[grid.index(i, j + 1)];
    const nm::ScaledValue radial = cylindrical
        ? nm::scaled_annular_radial_divergence(
              bx_hi, bx_lo, grid.dx(), grid.r_at_cell_center(i))
        : nm::scaled_directional_derivative(bx_hi, bx_lo, grid.dx());
    const nm::ScaledValue axial =
        nm::scaled_directional_derivative(by_hi, by_lo, grid.dy());
    const nm::ScaledValue radial_roundoff = cylindrical
        ? nm::scaled_annular_radial_roundoff(
              bx_hi, bx_lo, grid.dx(), grid.r_at_cell_center(i))
        : nm::scaled_directional_roundoff(bx_hi, bx_lo, grid.dx());
    const nm::ScaledValue axial_roundoff =
        nm::scaled_directional_roundoff(by_hi, by_lo, grid.dy());
    nm::ScaledValue residual =
        nm::scaled_directional_sum(radial, axial, Real{1});
    const nm::ScaledValue roundoff =
        nm::scaled_directional_sum(radial_roundoff, axial_roundoff, Real{1});
    // A residual entirely explained by the representation roundoff of its own
    // operands is not a resolved divergence; zeroing it here is what keeps a
    // legitimately-seeded field from failing on its own last bits.
    if (nm::residual_is_roundoff_explained(radial, axial, residual, roundoff)) {
      residual = nm::ScaledValue{};
    }
    merge_scaled_max(residual_linf, narrow(residual));
    merge_scaled_max(scale_linf,
                     narrow(nm::scaled_directional_magnitude_sum(radial, axial)));
  }
  s_residual[tid] = residual_linf;
  s_scale[tid] = scale_linf;
  __syncthreads();

  block_scaled_max<kThreads>(s_residual, tid);
  block_scaled_max<kThreads>(s_scale, tid);
  if (tid == 0) {
    partial_residual[blockIdx.x] = s_residual[0];
    partial_scale[blockIdx.x] = s_scale[0];
  }
}

__global__ QUASAR_LAUNCH_BOUNDS_256 void scaled_max_pass2(
    const SharedScaled* __restrict__ partial, SharedScaled* __restrict__ out,
    unsigned n_partial) {
  __shared__ SharedScaled sdata[kThreads];
  const unsigned tid = threadIdx.x;

  SharedScaled acc{Real{0}, 0};
  for (unsigned k = tid; k < n_partial; k += kThreads) {
    merge_scaled_max(acc, partial[k]);
  }
  sdata[tid] = acc;
  __syncthreads();

  block_scaled_max<kThreads>(sdata, tid);
  if (tid == 0) out[0] = sdata[0];
}

__global__ QUASAR_LAUNCH_BOUNDS_256 void real_max_pass2(
    const Real* __restrict__ partial, Real* __restrict__ out,
    unsigned n_partial) {
  __shared__ Real sdata[kThreads];
  const unsigned tid = threadIdx.x;

  Real acc = Real{0};
  for (unsigned k = tid; k < n_partial; k += kThreads) {
    acc = host_like_max(acc, partial[k]);
  }
  sdata[tid] = acc;
  __syncthreads();

  block_real_max<kThreads>(sdata, tid);
  if (tid == 0) out[0] = sdata[0];
}

// -- Boundary closure --------------------------------------------------------

__device__ inline void require_pair(Real actual, Real source, Real source_sign,
                                    unsigned rule, unsigned* violated) {
  const Real defect = nm::normalized_pair_defect(actual, source, source_sign);
  if (!(isfinite(defect) && defect <= nm::kDiscreteSolenoidalTolerance)) {
    atomicOr(violated, rule);
  }
}

__device__ inline void require_zero(Real value, unsigned rule,
                                    unsigned* violated) {
  if (value != Real{0}) atomicOr(violated, rule);
}

// B0 is never ghost-filled, so every prescribed padded sample must already be
// the fixed point of the configured homogeneous closure. The index maps below
// mirror the device ghost kernels exactly.
__global__ QUASAR_LAUNCH_BOUNDS_256 void boundary_x_kernel(
    Grid2D grid, int mode, int low, const Real* __restrict__ b0x,
    const Real* __restrict__ b0y, const Real* __restrict__ b0z,
    unsigned* __restrict__ violated) {
  const std::size_t rows = static_cast<std::size_t>(grid.ny) + 1;
  const std::size_t layers = static_cast<std::size_t>(grid.nghost);
  const std::size_t total = rows * layers;
  for (std::size_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
       t += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const int j = static_cast<int>(t / layers);
    const int layer = static_cast<int>(t % layers) + 1;

    if (mode == kModePeriodic) {
      const int target = low ? -layer : grid.nx - 1 + layer;
      const int source = low ? grid.nx - layer : layer - 1;
      const unsigned rule = mhd_background_rule_periodic_x;
      require_pair(b0x[grid.index(target, j)], b0x[grid.index(source, j)],
                   Real{1}, rule, violated);
      require_pair(b0y[grid.index(target, j)], b0y[grid.index(source, j)],
                   Real{1}, rule, violated);
      require_pair(b0z[grid.index(target, j)], b0z[grid.index(source, j)],
                   Real{1}, rule, violated);
      continue;
    }

    const bool axis = mode == kModeAxis;
    const unsigned parity_rule = axis ? mhd_background_rule_axis_parity
                                      : mhd_background_rule_x_wall_parity;
    const int target = low ? -layer : grid.nx - 1 + layer;
    const int source = low ? layer - 1 : grid.nx - layer;
    require_pair(b0y[grid.index(target, j)], b0y[grid.index(source, j)],
                 Real{1}, parity_rule, violated);
    require_pair(b0z[grid.index(target, j)], b0z[grid.index(source, j)],
                 axis ? Real{-1} : Real{1}, parity_rule, violated);
    if (low) {
      require_zero(b0x[grid.index(0, j)],
                   axis ? mhd_background_rule_axis_constraint
                        : mhd_background_rule_x_wall_normal,
                   violated);
      require_pair(b0x[grid.index(-layer, j)], b0x[grid.index(layer, j)],
                   Real{-1}, parity_rule, violated);
    } else {
      require_zero(b0x[grid.index(grid.nx, j)],
                   mhd_background_rule_x_wall_normal, violated);
      if (layer > 1) {
        const int offset = layer - 1;
        require_pair(b0x[grid.index(grid.nx + offset, j)],
                     b0x[grid.index(grid.nx - offset, j)], Real{-1},
                     mhd_background_rule_x_wall_parity, violated);
      }
    }
  }
}

__global__ QUASAR_LAUNCH_BOUNDS_256 void boundary_y_kernel(
    Grid2D grid, int mode, int low, const Real* __restrict__ b0x,
    const Real* __restrict__ b0y, const Real* __restrict__ b0z,
    unsigned* __restrict__ violated) {
  const std::size_t columns =
      static_cast<std::size_t>(grid.nx + 2 * grid.nghost);
  const std::size_t layers = static_cast<std::size_t>(grid.nghost);
  const std::size_t total = columns * layers;
  for (std::size_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total;
       t += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const int i = static_cast<int>(t / layers) - grid.nghost;
    const int layer = static_cast<int>(t % layers) + 1;

    if (mode == kModePeriodic) {
      const int target = low ? -layer : grid.ny - 1 + layer;
      const int source = low ? grid.ny - layer : layer - 1;
      const unsigned rule = mhd_background_rule_periodic_y;
      require_pair(b0x[grid.index(i, target)], b0x[grid.index(i, source)],
                   Real{1}, rule, violated);
      require_pair(b0y[grid.index(i, target)], b0y[grid.index(i, source)],
                   Real{1}, rule, violated);
      require_pair(b0z[grid.index(i, target)], b0z[grid.index(i, source)],
                   Real{1}, rule, violated);
      continue;
    }

    const int target = low ? -layer : grid.ny - 1 + layer;
    const int source = low ? layer - 1 : grid.ny - layer;
    require_pair(b0x[grid.index(i, target)], b0x[grid.index(i, source)],
                 Real{1}, mhd_background_rule_y_wall_parity, violated);
    require_pair(b0z[grid.index(i, target)], b0z[grid.index(i, source)],
                 Real{1}, mhd_background_rule_y_wall_parity, violated);
    if (low) {
      require_zero(b0y[grid.index(i, 0)], mhd_background_rule_y_wall_normal,
                   violated);
      require_pair(b0y[grid.index(i, -layer)], b0y[grid.index(i, layer)],
                   Real{-1}, mhd_background_rule_y_wall_parity, violated);
    } else {
      require_zero(b0y[grid.index(i, grid.ny)],
                   mhd_background_rule_y_wall_normal, violated);
      if (layer > 1) {
        const int offset = layer - 1;
        require_pair(b0y[grid.index(i, grid.ny + offset)],
                     b0y[grid.index(i, grid.ny - offset)], Real{-1},
                     mhd_background_rule_y_wall_parity, violated);
      }
    }
  }
}

// -- Curl-free ---------------------------------------------------------------

// Four concatenated sweeps, decoded from one flat index so the whole check is
// a single reduction:
//   A: interior cells, cylindrical axis-touching only -- Bphi must be zero
//   B: interior corners -- poloidal curl defect (tolerance)
//   C: (ny+1) x nx     -- axial difference of Bphi must be exactly zero
//   D: ny x (nx+1)     -- radial derivative of Bphi
struct CurlPlan {
  std::size_t end_a;
  std::size_t end_b;
  std::size_t end_c;
  std::size_t end_d;
  int width_a;
  int width_b;
  int width_c;
  int width_d;
  int cylindrical;
  int axis_touching;
};

__global__ QUASAR_LAUNCH_BOUNDS_256 void curl_free_pass1(
    Grid2D grid, CurlPlan plan, const Real* __restrict__ b0x,
    const Real* __restrict__ b0y, const Real* __restrict__ b0z,
    Real* __restrict__ partial, int* __restrict__ not_exactly_zero,
    int* __restrict__ negative_radius) {
  __shared__ Real sdata[kThreads];
  const unsigned tid = threadIdx.x;

  Real linf = Real{0};
  for (std::size_t t = blockIdx.x * blockDim.x + tid; t < plan.end_d;
       t += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    int i = 0;
    int j = 0;
    int region = 0;
    if (t < plan.end_a) {
      region = 0;
      i = static_cast<int>(t % static_cast<std::size_t>(plan.width_a));
      j = static_cast<int>(t / static_cast<std::size_t>(plan.width_a));
    } else if (t < plan.end_b) {
      const std::size_t u = t - plan.end_a;
      region = 1;
      i = static_cast<int>(u % static_cast<std::size_t>(plan.width_b));
      j = static_cast<int>(u / static_cast<std::size_t>(plan.width_b));
    } else if (t < plan.end_c) {
      const std::size_t u = t - plan.end_b;
      region = 2;
      i = static_cast<int>(u % static_cast<std::size_t>(plan.width_c));
      j = static_cast<int>(u / static_cast<std::size_t>(plan.width_c));
    } else {
      const std::size_t u = t - plan.end_c;
      region = 3;
      i = static_cast<int>(u % static_cast<std::size_t>(plan.width_d));
      j = static_cast<int>(u / static_cast<std::size_t>(plan.width_d));
    }

    switch (region) {
      case 0: {
        // A regular axisymmetric curl-free field on a simply connected domain
        // containing r=0 has no toroidal component at all. The formal annular
        // solution Bphi=C/r is singular at the axis and represents a
        // distributional axial current, so skipping only the r=0 difference is
        // not a valid proof -- the component itself must vanish.
        if (b0z[grid.index(i, j)] != Real{0}) atomicAnd(not_exactly_zero, 0);
        break;
      }
      case 1: {
        // curl(B0)_out = d_y B0x - d_x B0y at every interior corner, including
        // corners on the physical boundary: their one-sided stencil samples are
        // already present in the fixed padded background.
        const nm::ScaledValue dy_b0x = nm::scaled_directional_derivative(
            b0x[grid.index(i, j)], b0x[grid.index(i, j - 1)], grid.dy());
        const nm::ScaledValue dx_b0y = nm::scaled_directional_derivative(
            b0y[grid.index(i, j)], b0y[grid.index(i - 1, j)], grid.dx());
        linf = host_like_max(
            linf, nm::normalized_directional_sum_defect(dy_b0x, dx_b0y,
                                                        Real{-1}));
        break;
      }
      case 2: {
        const nm::ScaledValue difference = nm::scaled_difference_to_value(
            b0z[grid.index(i, j)], b0z[grid.index(i, j - 1)]);
        if (!isfinite(difference.mantissa) ||
            difference.mantissa != Real{0}) {
          atomicAnd(not_exactly_zero, 0);
        }
        break;
      }
      default: {
        if (plan.cylindrical) {
          const Real r_face = grid.r_at_edge(i);
          if (r_face == Real{0}) break;
          if (!(r_face > Real{0})) {
            atomicOr(negative_radius, 1);
            break;
          }
          // Compare the C invariant inferred from each uniform-dr cell average
          // rather than differencing stored values: inversion of the
          // logarithmic moment is admitted under the same relative tolerance
          // as the poloidal curl check.
          const nm::ScaledValue upper =
              nm::scaled_toroidal_flux_from_uniform_average(
                  b0z[grid.index(i, j)], grid.r_at_edge(i), grid.dx());
          const nm::ScaledValue lower =
              nm::scaled_toroidal_flux_from_uniform_average(
                  b0z[grid.index(i - 1, j)], grid.r_at_edge(i - 1), grid.dx());
          linf = host_like_max(
              linf,
              nm::normalized_directional_sum_defect(upper, lower, Real{-1}));
        } else {
          const nm::ScaledValue difference = nm::scaled_difference_to_value(
              b0z[grid.index(i, j)], b0z[grid.index(i - 1, j)]);
          if (!isfinite(difference.mantissa) ||
              difference.mantissa != Real{0}) {
            atomicAnd(not_exactly_zero, 0);
          }
        }
        break;
      }
    }
  }
  sdata[tid] = linf;
  __syncthreads();

  block_real_max<kThreads>(sdata, tid);
  if (tid == 0) partial[blockIdx.x] = sdata[0];
}

std::size_t extent(int w, int h) {
  return static_cast<std::size_t>(w > 0 ? w : 0)
       * static_cast<std::size_t>(h > 0 ? h : 0);
}

}  // namespace

MhdBackgroundSolenoidalResult launch_mhd_validate_background_solenoidal(
    const MhdBackgroundField<Real>& b0, Grid2D grid, int cylindrical,
    stream_t stream_in) {
  const auto stream = static_cast<hipStream_t>(stream_in);
  const std::size_t storage = grid.storage_size();
  const std::size_t n_cells = static_cast<std::size_t>(grid.nx)
                            * static_cast<std::size_t>(grid.ny);
  const std::size_t span = storage > n_cells ? storage : n_cells;
  MhdBackgroundSolenoidalResult out;
  if (span == 0) return out;

  const unsigned blocks = hd::reduce_block_count(span);
  backend::DeviceBuffer<SharedScaled> partial_residual{blocks};
  backend::DeviceBuffer<SharedScaled> partial_scale{blocks};
  backend::DeviceBuffer<SharedScaled> result{1};
  backend::DeviceBuffer<int> nonfinite{1};

  hipLaunchKernelGGL(solenoidal_pass1, dim3{blocks}, dim3{kThreads}, 0, stream,
                     grid, cylindrical, b0.b0x_face.device_ptr(),
                     b0.b0y_face.device_ptr(), b0.b0z_cell.device_ptr(),
                     storage, partial_residual.device_ptr(),
                     partial_scale.device_ptr(), nonfinite.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  SharedScaled host{};
  hipLaunchKernelGGL(scaled_max_pass2, dim3{1}, dim3{kThreads}, 0, stream,
                     partial_residual.device_ptr(), result.device_ptr(),
                     blocks);
  QUASAR_HIP_CHECK(hipGetLastError());
  QUASAR_HIP_CHECK(hipStreamSynchronize(stream));
  result.copy_to_host(&host, 1);
  out.residual_linf = nm::ScaledValue{host.mantissa, host.exponent};

  hipLaunchKernelGGL(scaled_max_pass2, dim3{1}, dim3{kThreads}, 0, stream,
                     partial_scale.device_ptr(), result.device_ptr(), blocks);
  QUASAR_HIP_CHECK(hipGetLastError());
  QUASAR_HIP_CHECK(hipStreamSynchronize(stream));
  result.copy_to_host(&host, 1);
  out.directional_scale_linf = nm::ScaledValue{host.mantissa, host.exponent};

  nonfinite.copy_to_host(&out.nonfinite_sample, 1);
  return out;
}

MhdBackgroundBoundaryResult launch_mhd_validate_background_boundaries(
    const MhdBackgroundField<Real>& b0, Grid2D grid, const int field_modes[4],
    stream_t stream_in) {
  const auto stream = static_cast<hipStream_t>(stream_in);
  MhdBackgroundBoundaryResult out;
  if (grid.nghost <= 0) return out;

  backend::DeviceBuffer<unsigned> violated{1};
  for (int side = 0; side < 4; ++side) {
    const int mode = field_modes[side];
    if (mode != kModePeriodic && mode != kModeWall && mode != kModeAxis) {
      continue;
    }
    const bool x_side = side < 2;
    const int low = (side == 0 || side == 2) ? 1 : 0;
    const std::size_t rows = x_side
        ? static_cast<std::size_t>(grid.ny) + 1
        : static_cast<std::size_t>(grid.nx + 2 * grid.nghost);
    const std::size_t total = rows * static_cast<std::size_t>(grid.nghost);
    if (total == 0) continue;
    const unsigned blocks = hd::reduce_block_count(total);
    if (x_side) {
      hipLaunchKernelGGL(boundary_x_kernel, dim3{blocks}, dim3{kThreads}, 0,
                         stream, grid, mode, low, b0.b0x_face.device_ptr(),
                         b0.b0y_face.device_ptr(), b0.b0z_cell.device_ptr(),
                         violated.device_ptr());
    } else {
      hipLaunchKernelGGL(boundary_y_kernel, dim3{blocks}, dim3{kThreads}, 0,
                         stream, grid, mode, low, b0.b0x_face.device_ptr(),
                         b0.b0y_face.device_ptr(), b0.b0z_cell.device_ptr(),
                         violated.device_ptr());
    }
    QUASAR_HIP_CHECK(hipGetLastError());
  }
  QUASAR_HIP_CHECK(hipStreamSynchronize(stream));
  violated.copy_to_host(&out.violated_rules, 1);
  return out;
}

MhdBackgroundCurlFreeResult launch_mhd_validate_background_curl_free(
    const MhdBackgroundField<Real>& b0, Grid2D grid, int cylindrical,
    stream_t stream_in) {
  const auto stream = static_cast<hipStream_t>(stream_in);
  MhdBackgroundCurlFreeResult out;

  CurlPlan plan{};
  plan.cylindrical = cylindrical;
  plan.axis_touching = (cylindrical && grid.origin_x == Real{0}) ? 1 : 0;
  plan.width_a = plan.axis_touching ? grid.nx : 0;
  plan.width_b = grid.nx + 1;
  plan.width_c = grid.nx;
  plan.width_d = grid.nx + 1;
  const int height_a = plan.axis_touching ? grid.ny : 0;
  const int height_b = grid.ny + 1;
  const int height_c = grid.ny + 1;
  const int height_d = grid.ny;
  plan.end_a = extent(plan.width_a, height_a);
  plan.end_b = plan.end_a + extent(plan.width_b, height_b);
  plan.end_c = plan.end_b + extent(plan.width_c, height_c);
  plan.end_d = plan.end_c + extent(plan.width_d, height_d);
  if (plan.end_d == 0) return out;

  const unsigned blocks = hd::reduce_block_count(plan.end_d);
  backend::DeviceBuffer<Real> partial{blocks};
  backend::DeviceBuffer<Real> result{1};
  // Seeded to 1 = "so far exactly zero"; the kernel only ever clears it.
  backend::DeviceBuffer<int> exactly_zero{1};
  const int one = 1;
  exactly_zero.copy_from_host(&one, 1);
  backend::DeviceBuffer<int> negative_radius{1};

  hipLaunchKernelGGL(curl_free_pass1, dim3{blocks}, dim3{kThreads}, 0, stream,
                     grid, plan, b0.b0x_face.device_ptr(),
                     b0.b0y_face.device_ptr(), b0.b0z_cell.device_ptr(),
                     partial.device_ptr(), exactly_zero.device_ptr(),
                     negative_radius.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  hipLaunchKernelGGL(real_max_pass2, dim3{1}, dim3{kThreads}, 0, stream,
                     partial.device_ptr(), result.device_ptr(), blocks);
  QUASAR_HIP_CHECK(hipGetLastError());
  QUASAR_HIP_CHECK(hipStreamSynchronize(stream));

  result.copy_to_host(&out.relative_linf, 1);
  exactly_zero.copy_to_host(&out.single_derivative_is_zero, 1);
  negative_radius.copy_to_host(&out.negative_interior_radius, 1);
  return out;
}

}  // namespace quasar::mhd
