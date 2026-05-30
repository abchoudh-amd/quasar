#include "quasar/core/yee_field.hpp"

#include "backend/hip/pic/launch.hpp"

#include <hip/hip_runtime.h>

namespace {

// Separable digital filter pass: out[k] = a*left + b*center + a*right applied
// first along x then along y. (a, b) = (0.25, 0.5) is the binomial (1-2-1)/4
// smoother; the compensator pass uses (a, b) = (-0.5, 2.0) = (1-2-1) with a
// sharpening sign so that smoother∘compensator has flat response at low k.
// Neighbor lookup uses periodic_index so the stencil wraps on the torus.
__global__ void stencil_x_kernel(quasar::Grid2D g, const double* in, double* out,
                                 double a, double b) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const double l = in[g.periodic_index(i - 1, j)];
  const double c = in[g.periodic_index(i, j)];
  const double r = in[g.periodic_index(i + 1, j)];
  out[g.index(i, j)] = a * l + b * c + a * r;
}

__global__ void stencil_y_kernel(quasar::Grid2D g, const double* in, double* out,
                                 double a, double b) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const double l = in[g.periodic_index(i, j - 1)];
  const double c = in[g.periodic_index(i, j)];
  const double r = in[g.periodic_index(i, j + 1)];
  out[g.index(i, j)] = a * l + b * c + a * r;
}

// One separable (x then y) pass over a single current component, using `scratch`
// as ping-pong storage. Leaves the result back in `data`.
void filter_component(const quasar::Grid2D& g, double* data, double* scratch,
                      double a, double b, hipStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((g.nx + block.x - 1) / block.x, (g.ny + block.y - 1) / block.y);
  hipLaunchKernelGGL(stencil_x_kernel, grid, block, 0, stream, g, data, scratch, a, b);
  hipLaunchKernelGGL(stencil_y_kernel, grid, block, 0, stream, g, scratch, data, a, b);
}

void run_passes(const quasar::Grid2D& g, quasar::JField2D<double>& j, double* scratch,
                int passes, double a, double b, hipStream_t stream) {
  if (passes <= 0) return;
  for (int p = 0; p < passes; ++p) {
    filter_component(g, j.jx.device_ptr(), scratch, a, b, stream);
    filter_component(g, j.jy.device_ptr(), scratch, a, b, stream);
    filter_component(g, j.jz.device_ptr(), scratch, a, b, stream);
  }
}

}  // namespace

// `scratch` is caller-owned ping-pong storage of at least g.storage_size()
// doubles, hoisted out of the per-step hot path (see numerics/filter.cpp).
extern "C" void launch_pic_filter_binomial(const quasar::Grid2D& g, quasar::JField2D<double>& j,
                                           double* scratch, int passes, hipStream_t stream) {
  run_passes(g, j, scratch, passes, 0.25, 0.5, stream);
}

extern "C" void launch_pic_filter_compensated(const quasar::Grid2D& g, quasar::JField2D<double>& j,
                                              double* scratch, int passes, hipStream_t stream) {
  // Standard compensated binomial: `passes` smoothing passes followed by one
  // sharpening pass that restores the low-wavenumber amplitude the smoother
  // attenuated.
  run_passes(g, j, scratch, passes, 0.25, 0.5, stream);
  run_passes(g, j, scratch, 1, -0.5, 2.0, stream);
}
