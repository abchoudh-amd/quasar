#include "quasar/core/yee_field.hpp"
#include "quasar/numerics/shape.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/gather_common.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

// Axisymmetric (m=0) cylindrical gather + Boris push in the meridional (r,z)
// plane. The grid axis i indexes the radius r (x-axis, r=0 at i=0) and j the
// axial coordinate z (y-axis). Particle position slots map x->r, y->z; velocity
// slots map vx->vr, vy->vz, vz->vphi. Stored field components map (x,y,z)->
// (r,z,phi) exactly as the cylindrical FDTD kernels.
//
// Each component is interpolated from its cylindrical Yee sub-lattice. The
// physical right-handed basis is (e_r,e_phi,e_z); the storage order (r,z,phi) is
// left-handed and must be permuted before a Cartesian cross product is used.
//
// The cylindrical correction enters in the POSITION advance: a particle moving on
// a straight line in real space sweeps an azimuth even with vphi != 0, so after
// advancing the in-plane Cartesian coordinates we must rotate the (vr, vphi)
// velocity pair into the particle's new radial direction. Concretely, treating
// the meridional plane as a Cartesian (X = r, Y = phi-arc) plane for one step:
//   X_new = r + vr*dt,   Y_new = vphi*dt,
//   r_new = sqrt(X_new^2 + Y_new^2),
//   dphi  = atan2(Y_new, X_new),
// and the velocity (vr, vphi) is rotated by +dphi so it stays expressed in the
// new radial/azimuthal basis. z advances linearly. This is the standard
// "Boris push in cylindrical coordinates" coordinate rotation; it conserves
// kinetic energy exactly (a pure rotation of the (vr,vphi) pair) so a particle
// gains no spurious energy as it approaches or crosses the axis. Reflection
// across r=0 (r -> -r, vr -> -vr) is handled separately by the axis particle BC,
// so this kernel only pushes.

namespace {

// gather_index, gather_pair, and the Boris half-rotation are shared with the
// Cartesian gather+push and live in gather_common.hpp. Only the cylindrical
// position-advance coordinate rotation below is cylindrical-specific.
using namespace quasar::backend::pic;

template <int ShapeOrder>
__global__ void gather_push_cyl_kernel(quasar::Grid2D g, std::size_t n, double qm,
                                       double* __restrict__ x, double* __restrict__ y,
                                       double* __restrict__ x_prev, double* __restrict__ y_prev,
                                       double* __restrict__ vx, double* __restrict__ vy,
                                       double* __restrict__ vz,
                                       double* __restrict__ vphi_deposit,
                                       const unsigned char* __restrict__ alive,
                                       unsigned int* __restrict__ particle_error,
                                       const double* __restrict__ ex, const double* __restrict__ ey,
                                       const double* __restrict__ ez,
                                       const double* __restrict__ bx, const double* __restrict__ by,
                                       const double* __restrict__ bz,
                                       const double* __restrict__ ext_ex,
                                       const double* __restrict__ ext_ey,
                                       const double* __restrict__ ext_ez,
                                       const double* __restrict__ ext_bx,
                                       const double* __restrict__ ext_by,
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

  // Cylindrical Yee locations in (r,z) cell units:
  //   Er/Ephi=(0,1/2), Ez=(1/2,0), Br/Bphi=(0,0), Bz=(1/2,1/2).
  const auto wr_face = quasar::numerics::shape_weights_2d_at_offset<ShapeOrder>(
      x[p], y[p], g, 0.0, 0.5);
  const auto wz_face = quasar::numerics::shape_weights_2d_at_offset<ShapeOrder>(
      x[p], y[p], g, 0.5, 0.0);
  const auto w_center = quasar::numerics::shape_weights_2d_at_offset<ShapeOrder>(
      x[p], y[p], g, 0.5, 0.5);
  const auto w_corner = quasar::numerics::shape_weights_2d_at_offset<ShapeOrder>(
      x[p], y[p], g, 0.0, 0.0);
  quasar::Vec3 e{0, 0, 0}, ee{0, 0, 0}, b{0, 0, 0}, be{0, 0, 0};
  gather_component_pair(g, wr_face, ex, ext_ex, periodic_x, periodic_y,
                        g.nx, g.ny - 1, e.x, ee.x);
  gather_component_pair(g, wz_face, ey, ext_ey, periodic_x, periodic_y,
                        g.nx - 1, g.ny, e.y, ee.y);
  gather_component_pair(g, wr_face, ez, ext_ez, periodic_x, periodic_y,
                        g.nx, g.ny - 1, e.z, ee.z);
  gather_component_pair(g, w_corner, bx, ext_bx, periodic_x, periodic_y,
                        g.nx, g.ny, b.x, be.x);
  gather_component_pair(g, w_center, by, ext_by, periodic_x, periodic_y,
                        g.nx - 1, g.ny - 1, b.y, be.y);
  gather_component_pair(g, w_corner, bz, ext_bz, periodic_x, periodic_y,
                        g.nx, g.ny, b.z, be.z);
  b.x = weighted_sum2(
      current_b_weight, b.x, previous_b_weight,
      gather_component(g, w_corner, old_bx, periodic_x, periodic_y,
                       g.nx, g.ny));
  b.y = weighted_sum2(
      current_b_weight, b.y, previous_b_weight,
      gather_component(g, w_center, old_by, periodic_x, periodic_y,
                       g.nx - 1, g.ny - 1));
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

  // Permute storage (r,z,phi) to the right-handed physical basis (r,phi,z), do
  // the Boris rotation there, then map back. Applying the ordinary cross product
  // directly to storage order reverses the magnetic force.
  const quasar::Vec3 v_rpz = boris(
      qm, quasar::Vec3{vx[p], vz[p], vy[p]},
      quasar::Vec3{e.x, e.z, e.y}, quasar::Vec3{b.x, b.z, b.y}, force_dt);
  const quasar::Vec3 v{v_rpz.x, v_rpz.z, v_rpz.y};
  if (!valid_nonrelativistic_velocity(v)) {
    mark_particle_error(particle_error);
    return;
  }

  // Axial advance is linear.
  const double z_new = fma(position_dt, v.y, y[p]);

  // Meridional-plane advance with cylindrical coordinate rotation. Treat
  // (X = r, Y = phi-arc) locally; advance, then rotate (vr, vphi) by the swept
  // azimuth so the velocity stays in the new radial basis. This is a pure
  // rotation of the (vr, vphi) pair: |(vr,vphi)| is preserved, so no energy is
  // gained even as r -> 0.
  const double r_old = x[p];
  const double xx = fma(position_dt, v.x, r_old);  // new radial Cartesian coord
  const double yy = position_dt * v.z;          // azimuthal arc swept this step
  const double r_new = hypot(xx, yy);

  if (!stencil_coordinate_representable(r_new, g.origin_x, g.dx())
      || !stencil_coordinate_representable(z_new, g.origin_y, g.dy())) {
    mark_particle_error(particle_error);
    return;
  }

  // Rotation angle swept; atan2 handles the r -> 0 region robustly.
  const double dphi = (xx != 0.0 || yy != 0.0) ? atan2(yy, xx) : 0.0;
  const double cphi = cos(dphi);
  const double sphi = sin(dphi);
  const double vr = v.x;
  const double vphi = v.z;
  // Rotate (vr, vphi) by +dphi into the new radial/azimuthal basis. Form and
  // validate the complete stored state before committing any component. The
  // rotation is norm preserving analytically, but its two rounded sums can
  // still turn a velocity immediately below c into a non-finite or |v|>=c
  // state; that must fail at this push rather than reaching the next kernel.
  const quasar::Vec3 rotated_v{
      fma(vr, cphi, vphi * sphi),
      v.y,
      fma(-vr, sphi, vphi * cphi)};
  if (!valid_nonrelativistic_velocity(rotated_v)) {
    mark_particle_error(particle_error);
    return;
  }
  // Record the pre-move position for the charge-conserving deposit only after
  // the complete candidate state has passed every fatal-state check.
  x_prev[p] = x[p];
  y_prev[p] = y[p];
  vphi_deposit[p] = v.z;
  vx[p] = rotated_v.x;
  vy[p] = rotated_v.y;
  vz[p] = rotated_v.z;

  x[p] = r_new;
  y[p] = z_new;
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
  hipLaunchKernelGGL((gather_push_cyl_kernel<ShapeOrder>), grid, block, 0, stream,
                     g, s.size(), s.charge_to_mass(), s.x(), s.y(), s.x_prev(), s.y_prev(),
                     s.vx(), s.vy(), s.vz(), s.vphi_deposit(),
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

QUASAR_PIC_GATHER_PUSH_ABI(launch_pic_gather_push_cyl_shape1, 1)
QUASAR_PIC_GATHER_PUSH_ABI(launch_pic_gather_push_cyl_shape2, 2)
