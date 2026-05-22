#include "quasar/core/yee_field.hpp"

#include <hip/hip_runtime.h>

namespace {

__global__ void scale_kernel(quasar::Grid2D g, double* x, double* y, double* z, double alpha) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);
  x[k] *= alpha;
  y[k] *= alpha;
  z[k] *= alpha;
}

}  // namespace

extern "C" void launch_pic_filter_binomial(const quasar::Grid2D& g, quasar::JField2D<double>& j,
                                           int passes, hipStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((g.nx + block.x - 1) / block.x, (g.ny + block.y - 1) / block.y);
  for (int p = 0; p < passes; ++p) {
    hipLaunchKernelGGL(scale_kernel, grid, block, 0, stream,
                       g, j.jx.device_ptr(), j.jy.device_ptr(), j.jz.device_ptr(), 1.0);
  }
}

extern "C" void launch_pic_filter_compensated(const quasar::Grid2D& g, quasar::JField2D<double>& j,
                                              int passes, hipStream_t stream) {
  launch_pic_filter_binomial(g, j, passes, stream);
}
