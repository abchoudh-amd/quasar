#include "quasar/core/yee_field.hpp"
#include "quasar/numerics/shape.hpp"
#include "quasar/physics/pic/species.hpp"

#include <hip/hip_runtime.h>

namespace {

template <int ShapeOrder>
__global__ void deposit_kernel(quasar::Grid2D g, std::size_t n, double q,
                               const double* x, const double* y,
                               const double* vx, const double* vy, const double* vz,
                               const double* weight, const unsigned char* alive,
                               double* jx, double* jy, double* jz) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;
  const auto w = quasar::numerics::shape_weights_2d<ShapeOrder>(x[p], y[p], g);
  for (int jj = 0; jj < w.ny; ++jj) {
    for (int ii = 0; ii < w.nx; ++ii) {
      const double ww = q * weight[p] * w.wx[ii] * w.wy[jj];
      const std::size_t k = g.periodic_index(w.ix[ii], w.iy[jj]);
      atomicAdd(jx + k, ww * vx[p]);
      atomicAdd(jy + k, ww * vy[p]);
      atomicAdd(jz + k, ww * vz[p]);
    }
  }
}

template <int ShapeOrder>
void launch(const quasar::Grid2D& g, const quasar::pic::ParticleSpecies& s,
            quasar::JField2D<double>& j, hipStream_t stream) {
  const dim3 block(256);
  const dim3 grid((s.size() + block.x - 1) / block.x);
  hipLaunchKernelGGL((deposit_kernel<ShapeOrder>), grid, block, 0, stream,
                     g, s.size(), s.charge(), s.x(), s.y(), s.vx(), s.vy(), s.vz(),
                     s.weight(), s.alive(), j.jx.device_ptr(), j.jy.device_ptr(),
                     j.jz.device_ptr());
}

}  // namespace

extern "C" void launch_pic_deposit_shape1(const quasar::Grid2D& g,
                                          const quasar::pic::ParticleSpecies& s,
                                          quasar::JField2D<double>& j,
                                          double, hipStream_t stream) {
  launch<1>(g, s, j, stream);
}

extern "C" void launch_pic_deposit_shape2(const quasar::Grid2D& g,
                                          const quasar::pic::ParticleSpecies& s,
                                          quasar::JField2D<double>& j,
                                          double, hipStream_t stream) {
  launch<2>(g, s, j, stream);
}
