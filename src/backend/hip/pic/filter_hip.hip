#include "quasar/core/yee_field.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

namespace {

__device__ inline int clamp_axis(int i, int lo, int hi) {
  return i < lo ? lo : (i > hi ? hi : i);
}

__device__ inline std::size_t filter_index(const quasar::Grid2D& g, int i, int j,
                                           bool periodic_x, bool periodic_y) {
  const int ii = periodic_x ? g.wrap_i(i) : clamp_axis(i, 0, g.nx - 1);
  const int jj = periodic_y ? g.wrap_j(j) : clamp_axis(j, 0, g.ny - 1);
  return g.index(ii, jj);
}

// Separable digital filter pass: out[k] = a*left + b*center + a*right applied
// first along x (Axis == 0) then along y (Axis == 1). (a, b) = (0.25, 0.5) is the
// binomial (1-2-1)/4 smoother; the compensator pass uses (a, b) = (-0.5, 2.0) =
// (1-2-1) with a sharpening sign so that smoother∘compensator has flat response at
// low k. Periodic axes wrap; wall axes clamp at the edge, avoiding far-edge
// current coupling on mixed/non-periodic domains.
//
// All three current components (jx, jy, jz) are filtered in one launch: they
// share the same stencil geometry and index arithmetic, so fusing them cuts the
// launch count and the per-cell index/wrap recomputation 3x versus one kernel
// per component. The three stencil neighbours per axis are L2-cache friendly, so
// fusion captures the bulk of the win without the occupancy cost of a large LDS
// tile.
template <int Axis>
__global__ void stencil_kernel(quasar::Grid2D g,
                               const double* __restrict__ in_x,
                               const double* __restrict__ in_y,
                               const double* __restrict__ in_z,
                               double* __restrict__ out_x,
                               double* __restrict__ out_y,
                               double* __restrict__ out_z,
                               double a, double b, bool periodic_x, bool periodic_y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const int di = (Axis == 0) ? 1 : 0;
  const int dj = (Axis == 0) ? 0 : 1;
  // The three neighbour indices are shared by all components, so compute the
  // wrap/clamp once and reuse for jx/jy/jz.
  const std::size_t kl = filter_index(g, i - di, j - dj, periodic_x, periodic_y);
  const std::size_t kc = filter_index(g, i, j, periodic_x, periodic_y);
  const std::size_t kr = filter_index(g, i + di, j + dj, periodic_x, periodic_y);
  const std::size_t ko = g.index(i, j);
  out_x[ko] = a * in_x[kl] + b * in_x[kc] + a * in_x[kr];
  out_y[ko] = a * in_y[kl] + b * in_y[kc] + a * in_y[kr];
  out_z[ko] = a * in_z[kl] + b * in_z[kc] + a * in_z[kr];
}

// One separable (x then y) pass over all three current components, using the
// three per-component `scratch` strips as ping-pong storage. Leaves the result
// back in the JField2D buffers.
void filter_pass(const quasar::Grid2D& g, quasar::JField2D<double>& j,
                 double* sx, double* sy, double* sz,
                 double a, double b, bool periodic_x, bool periodic_y,
                 hipStream_t stream) {
  const dim3 block(quasar::backend::pic::kLaunchBlock2D,
                   quasar::backend::pic::kLaunchBlock2D);
  const dim3 grid = quasar::backend::pic::grid_2d(g, block);
  hipLaunchKernelGGL(stencil_kernel<0>, grid, block, 0, stream,
                     g, j.jx.device_ptr(), j.jy.device_ptr(), j.jz.device_ptr(),
                     sx, sy, sz, a, b, periodic_x, periodic_y);
  hipLaunchKernelGGL(stencil_kernel<1>, grid, block, 0, stream,
                     g, sx, sy, sz,
                     j.jx.device_ptr(), j.jy.device_ptr(), j.jz.device_ptr(),
                     a, b, periodic_x, periodic_y);
}

void run_passes(const quasar::Grid2D& g, quasar::JField2D<double>& j, double* scratch,
                int passes, double a, double b, bool periodic_x, bool periodic_y,
                hipStream_t stream) {
  if (passes <= 0) return;
  // `scratch` holds three contiguous storage_size() strips, one per component.
  const std::size_t n = g.storage_size();
  double* sx = scratch;
  double* sy = scratch + n;
  double* sz = scratch + 2 * n;
  for (int p = 0; p < passes; ++p) {
    filter_pass(g, j, sx, sy, sz, a, b, periodic_x, periodic_y, stream);
  }
}

}  // namespace

// `scratch` is caller-owned ping-pong storage of at least 3*g.storage_size()
// Real values (one storage_size() strip per current component, since the three
// components are filtered in one fused launch), hoisted out of the per-step hot
// path (see numerics/filter.cpp).
extern "C" void launch_pic_filter_binomial(const quasar::Grid2D& g,
                                           quasar::JField2D<quasar::Real>& j,
                                           quasar::Real* scratch, int passes, int periodic_x,
                                           int periodic_y, quasar_stream_t stream) {
  run_passes(g, j, scratch, passes, 0.25, 0.5,
             periodic_x != 0, periodic_y != 0, static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_filter_compensated(const quasar::Grid2D& g,
                                              quasar::JField2D<quasar::Real>& j,
                                              quasar::Real* scratch, int passes, int periodic_x,
                                              int periodic_y, quasar_stream_t stream) {
  // Standard compensated binomial: `passes` smoothing passes followed by one
  // sharpening pass that restores the low-wavenumber amplitude the smoother
  // attenuated.
  const auto hip_stream = static_cast<hipStream_t>(stream);
  const bool px = periodic_x != 0;
  const bool py = periodic_y != 0;
  run_passes(g, j, scratch, passes, 0.25, 0.5, px, py, hip_stream);
  run_passes(g, j, scratch, 1, -0.5, 2.0, px, py, hip_stream);
  QUASAR_HIP_CHECK(::hipGetLastError());
}
