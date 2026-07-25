#include "quasar/core/yee_field.hpp"
#include "quasar/numerics/shape.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/gather_common.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

#include <stdexcept>

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
                                   unsigned int* __restrict__ particle_error,
                                   const double* __restrict__ ex, const double* __restrict__ ey,
                                   const double* __restrict__ ez,
                                   const double* __restrict__ bx, const double* __restrict__ by,
                                   const double* __restrict__ bz,
                                   const double* __restrict__ ext_ex, const double* __restrict__ ext_ey,
                                   const double* __restrict__ ext_ez,
                                   const double* __restrict__ ext_bx, const double* __restrict__ ext_by,
                                   const double* __restrict__ ext_bz,
                                   const double* __restrict__ old_bx,
                                   const double* __restrict__ old_by,
                                   const double* __restrict__ old_bz,
                                   bool periodic_x, bool periodic_y,
                                   double force_dt, double position_dt,
                                   double previous_b_weight,
                                   double current_b_weight) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;

  const quasar::Vec3 initial_v{vx[p], vy[p], vz[p]};
  if (!stencil_coordinate_representable(x[p], g.origin_x, g.dx())
      || !stencil_coordinate_representable(y[p], g.origin_y, g.dy())
      || !valid_nonrelativistic_velocity(initial_v)) {
    mark_particle_error(particle_error);
    return;
  }

  // Cartesian Yee locations in cell units:
  //   Ex/By=(0,1/2), Ey/Bx=(1/2,0), Ez=(1/2,1/2), Bz=(0,0).
  // A single cell-centred stencil for all six components introduces a systematic
  // half-cell force error and breaks the gather/deposit adjoint pairing.
  const auto wx_face = quasar::numerics::shape_weights_2d_at_offset<ShapeOrder>(
      x[p], y[p], g, 0.0, 0.5);
  const auto wy_face = quasar::numerics::shape_weights_2d_at_offset<ShapeOrder>(
      x[p], y[p], g, 0.5, 0.0);
  const auto w_center = quasar::numerics::shape_weights_2d_at_offset<ShapeOrder>(
      x[p], y[p], g, 0.5, 0.5);
  const auto w_corner = quasar::numerics::shape_weights_2d_at_offset<ShapeOrder>(
      x[p], y[p], g, 0.0, 0.0);
  quasar::Vec3 e{0, 0, 0}, ee{0, 0, 0}, b{0, 0, 0}, be{0, 0, 0};
  gather_component_pair(g, wx_face, ex, ext_ex, periodic_x, periodic_y,
                        g.nx, g.ny - 1, e.x, ee.x);
  gather_component_pair(g, wy_face, ey, ext_ey, periodic_x, periodic_y,
                        g.nx - 1, g.ny, e.y, ee.y);
  gather_component_pair(g, w_center, ez, ext_ez, periodic_x, periodic_y,
                        g.nx - 1, g.ny - 1, e.z, ee.z);
  gather_component_pair(g, wy_face, bx, ext_bx, periodic_x, periodic_y,
                        g.nx - 1, g.ny, b.x, be.x);
  gather_component_pair(g, wx_face, by, ext_by, periodic_x, periodic_y,
                        g.nx, g.ny - 1, b.y, be.y);
  gather_component_pair(g, w_corner, bz, ext_bz, periodic_x, periodic_y,
                        g.nx, g.ny, b.z, be.z);
  // Self B is leapfrogged at half steps. The Lorentz force at t^n uses the
  // arithmetic midpoint of the adjacent half-step states; external B is static
  // and is added once.
  b.x = weighted_sum2(
      current_b_weight, b.x, previous_b_weight,
      gather_component(g, wy_face, old_bx, periodic_x, periodic_y,
                       g.nx - 1, g.ny));
  b.y = weighted_sum2(
      current_b_weight, b.y, previous_b_weight,
      gather_component(g, wx_face, old_by, periodic_x, periodic_y,
                       g.nx, g.ny - 1));
  b.z = weighted_sum2(
      current_b_weight, b.z, previous_b_weight,
      gather_component(g, w_corner, old_bz, periodic_x, periodic_y,
                       g.nx, g.ny));
  e = e + ee;
  b = b + be;
  if (!finite_vec(e) || !finite_vec(b)) {
    mark_particle_error(particle_error);
    return;
  }
  quasar::Vec3 v = boris(qm, quasar::Vec3{vx[p], vy[p], vz[p]}, e, b,
                         force_dt);
  const double x_new = fma(position_dt, v.x, x[p]);
  const double y_new = fma(position_dt, v.y, y[p]);
  if (!valid_nonrelativistic_velocity(v)
      || !stencil_coordinate_representable(x_new, g.origin_x, g.dx())
      || !stencil_coordinate_representable(y_new, g.origin_y, g.dy())) {
    mark_particle_error(particle_error);
    return;
  }
  vx[p] = v.x;
  vy[p] = v.y;
  vz[p] = v.z;
  // Record the pre-move position so the charge-conserving deposit can form the
  // current from the (x_prev -> x) displacement.
  x_prev[p] = x[p];
  y_prev[p] = y[p];
  x[p] = x_new;
  y[p] = y_new;
}

template <int ShapeOrder>
void launch(const quasar::Grid2D& g, quasar::pic::ParticleSpecies& s,
            const quasar::YeeField2D<double>& self,
            const quasar::YeeField2D<double>& ext,
            const quasar::BField2D<double>& previous_b,
            bool periodic_x, bool periodic_y, double force_dt,
            double position_dt, double previous_b_weight,
            double current_b_weight, hipStream_t stream) {
  if (s.size() == 0) return;
  const dim3 block(256);
  const dim3 grid((s.size() + block.x - 1) / block.x);
  hipLaunchKernelGGL((gather_push_kernel<ShapeOrder>), grid, block, 0, stream,
                     g, s.size(), s.charge_to_mass(), s.x(), s.y(), s.x_prev(), s.y_prev(),
                     s.vx(), s.vy(), s.vz(),
                     s.alive(), s.particle_error(), self.ex.device_ptr(),
                     self.ey.device_ptr(), self.ez.device_ptr(),
                     self.bx.device_ptr(), self.by.device_ptr(), self.bz.device_ptr(),
                     ext.ex.device_ptr(), ext.ey.device_ptr(), ext.ez.device_ptr(),
                     ext.bx.device_ptr(), ext.by.device_ptr(), ext.bz.device_ptr(),
                     previous_b.bx.device_ptr(), previous_b.by.device_ptr(),
                     previous_b.bz.device_ptr(),
                     periodic_x, periodic_y, force_dt, position_dt,
                     previous_b_weight, current_b_weight);
}

}  // namespace

QUASAR_PIC_GATHER_PUSH_ABI(launch_pic_gather_push_shape1, 1)
QUASAR_PIC_GATHER_PUSH_ABI(launch_pic_gather_push_shape2, 2)

extern "C" void launch_pic_particle_error_check(
    const quasar::pic::ParticleSpecies& s, quasar_stream_t stream) {
  auto hip_stream = static_cast<hipStream_t>(stream);
  unsigned int error = 0;
  QUASAR_HIP_CHECK(::hipMemcpyAsync(&error, s.particle_error(),
                                    sizeof(unsigned int),
                                    ::hipMemcpyDeviceToHost, hip_stream));
  QUASAR_HIP_CHECK(::hipStreamSynchronize(hip_stream));
  if (error == 0) return;
  QUASAR_HIP_CHECK(::hipMemsetAsync(s.particle_error(), 0,
                                    sizeof(unsigned int), hip_stream));
  QUASAR_HIP_CHECK(::hipStreamSynchronize(hip_stream));
  throw std::runtime_error{
      "PIC particle push produced or received a non-finite/nonrepresentable "
      "state, or |v| >= c=1 violated the nonrelativistic model"};
}
