#include "quasar/core/yee_field.hpp"
#include "quasar/numerics/shape.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/gather_common.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

namespace {

// gather_index, gather_pair, and the Boris half-rotation are shared with the
// cylindrical gather+push and live in gather_common.hpp.
using namespace quasar::backend::pic;

template <int ShapeOrder>
__global__ void gather_push_kernel(quasar::Grid2D g, std::size_t n, double qm,
                                   double* __restrict__ x, double* __restrict__ y,
                                   double* __restrict__ x_prev, double* __restrict__ y_prev,
                                   double* __restrict__ vx, double* __restrict__ vy,
                                   double* __restrict__ vz,
                                   const unsigned char* __restrict__ alive,
                                   const double* __restrict__ ex, const double* __restrict__ ey,
                                   const double* __restrict__ ez,
                                   const double* __restrict__ bx, const double* __restrict__ by,
                                   const double* __restrict__ bz,
                                   const double* __restrict__ ext_ex, const double* __restrict__ ext_ey,
                                   const double* __restrict__ ext_ez,
                                   const double* __restrict__ ext_bx, const double* __restrict__ ext_by,
                                   const double* __restrict__ ext_bz,
                                   bool periodic_x, bool periodic_y,
                                   double dt) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;

  // The shape weights depend only on the particle position, so compute them once
  // and reuse them for the self+external E and self+external B gathers.
  const auto w = quasar::numerics::shape_weights_2d<ShapeOrder>(x[p], y[p], g);
  quasar::Vec3 e{0, 0, 0}, ee{0, 0, 0}, b{0, 0, 0}, be{0, 0, 0};
  gather_pair(g, w, ex, ey, ez, ext_ex, ext_ey, ext_ez, periodic_x, periodic_y, e, ee);
  gather_pair(g, w, bx, by, bz, ext_bx, ext_by, ext_bz, periodic_x, periodic_y, b, be);
  e = e + ee;
  b = b + be;
  quasar::Vec3 v = boris(qm, quasar::Vec3{vx[p], vy[p], vz[p]}, e, b, dt);
  vx[p] = v.x;
  vy[p] = v.y;
  vz[p] = v.z;
  // Record the pre-move position so the charge-conserving deposit can form the
  // current from the (x_prev -> x) displacement.
  x_prev[p] = x[p];
  y_prev[p] = y[p];
  x[p] = x[p] + dt * v.x;
  y[p] = y[p] + dt * v.y;
}

template <int ShapeOrder>
void launch(const quasar::Grid2D& g, quasar::pic::ParticleSpecies& s,
            const quasar::YeeField2D<double>& self,
            const quasar::YeeField2D<double>& ext,
            bool periodic_x, bool periodic_y, double dt, hipStream_t stream) {
  if (s.size() == 0) return;
  const dim3 block(256);
  const dim3 grid((s.size() + block.x - 1) / block.x);
  hipLaunchKernelGGL((gather_push_kernel<ShapeOrder>), grid, block, 0, stream,
                     g, s.size(), s.charge_to_mass(), s.x(), s.y(), s.x_prev(), s.y_prev(),
                     s.vx(), s.vy(), s.vz(),
                     s.alive(), self.ex.device_ptr(), self.ey.device_ptr(), self.ez.device_ptr(),
                     self.bx.device_ptr(), self.by.device_ptr(), self.bz.device_ptr(),
                     ext.ex.device_ptr(), ext.ey.device_ptr(), ext.ez.device_ptr(),
                     ext.bx.device_ptr(), ext.by.device_ptr(), ext.bz.device_ptr(),
                     periodic_x, periodic_y, dt);
}

}  // namespace

QUASAR_PIC_GATHER_PUSH_ABI(launch_pic_gather_push_shape1, 1)
QUASAR_PIC_GATHER_PUSH_ABI(launch_pic_gather_push_shape2, 2)
