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
template <int Axis>
__global__ void stencil_kernel(quasar::Grid2D g, const double* in, double* out,
                               double a, double b, bool periodic_x, bool periodic_y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const int di = (Axis == 0) ? 1 : 0;
  const int dj = (Axis == 0) ? 0 : 1;
  const double l = in[filter_index(g, i - di, j - dj, periodic_x, periodic_y)];
  const double c = in[filter_index(g, i, j, periodic_x, periodic_y)];
  const double r = in[filter_index(g, i + di, j + dj, periodic_x, periodic_y)];
  out[g.index(i, j)] = a * l + b * c + a * r;
}

// One separable (x then y) pass over a single current component, using `scratch`
// as ping-pong storage. Leaves the result back in `data`.
void filter_component(const quasar::Grid2D& g, double* data, double* scratch,
                      double a, double b, bool periodic_x, bool periodic_y,
                      hipStream_t stream) {
  const dim3 block(quasar::backend::pic::kLaunchBlock2D,
                   quasar::backend::pic::kLaunchBlock2D);
  const dim3 grid = quasar::backend::pic::grid_2d(g, block);
  hipLaunchKernelGGL(stencil_kernel<0>, grid, block, 0, stream,
                     g, data, scratch, a, b, periodic_x, periodic_y);
  hipLaunchKernelGGL(stencil_kernel<1>, grid, block, 0, stream,
                     g, scratch, data, a, b, periodic_x, periodic_y);
}

void run_passes(const quasar::Grid2D& g, quasar::JField2D<double>& j, double* scratch,
                int passes, double a, double b, bool periodic_x, bool periodic_y,
                hipStream_t stream) {
  if (passes <= 0) return;
  for (int p = 0; p < passes; ++p) {
    filter_component(g, j.jx.device_ptr(), scratch, a, b, periodic_x, periodic_y, stream);
    filter_component(g, j.jy.device_ptr(), scratch, a, b, periodic_x, periodic_y, stream);
    filter_component(g, j.jz.device_ptr(), scratch, a, b, periodic_x, periodic_y, stream);
  }
}

}  // namespace

// `scratch` is caller-owned ping-pong storage of at least g.storage_size()
// doubles, hoisted out of the per-step hot path (see numerics/filter.cpp).
extern "C" void launch_pic_filter_binomial(const quasar::Grid2D& g, quasar::JField2D<double>& j,
                                           double* scratch, int passes, int periodic_x,
                                           int periodic_y, quasar_stream_t stream) {
  run_passes(g, j, scratch, passes, 0.25, 0.5,
             periodic_x != 0, periodic_y != 0, static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_filter_compensated(const quasar::Grid2D& g, quasar::JField2D<double>& j,
                                              double* scratch, int passes, int periodic_x,
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
