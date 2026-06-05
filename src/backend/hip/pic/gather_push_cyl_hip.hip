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
// Field interpolation to the particle is identical to the Cartesian gather (the
// same shape weights over the (r,z) grid). The Boris velocity rotation is also
// the standard one -- the local Cartesian-like frame (e_r, e_z, e_phi) is
// orthonormal, so the Lorentz force / Boris half-rotation is unchanged.
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
                                       const unsigned char* __restrict__ alive,
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
                                       bool periodic_x, bool periodic_y,
                                       double dt) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;

  const auto w = quasar::numerics::shape_weights_2d<ShapeOrder>(x[p], y[p], g);
  quasar::Vec3 e{0, 0, 0}, ee{0, 0, 0}, b{0, 0, 0}, be{0, 0, 0};
  gather_pair(g, w, ex, ey, ez, ext_ex, ext_ey, ext_ez, periodic_x, periodic_y, e, ee);
  gather_pair(g, w, bx, by, bz, ext_bx, ext_by, ext_bz, periodic_x, periodic_y, b, be);
  e = e + ee;
  b = b + be;

  // Boris half-rotation in the local (e_r, e_z, e_phi) frame. v.x=vr, v.y=vz,
  // v.z=vphi.
  quasar::Vec3 v = boris(qm, quasar::Vec3{vx[p], vy[p], vz[p]}, e, b, dt);

  // Record the pre-move position for the charge-conserving deposit.
  x_prev[p] = x[p];
  y_prev[p] = y[p];

  // Axial advance is linear.
  y[p] = y[p] + dt * v.y;

  // Meridional-plane advance with cylindrical coordinate rotation. Treat
  // (X = r, Y = phi-arc) locally; advance, then rotate (vr, vphi) by the swept
  // azimuth so the velocity stays in the new radial basis. This is a pure
  // rotation of the (vr, vphi) pair: |(vr,vphi)| is preserved, so no energy is
  // gained even as r -> 0.
  const double r_old = x[p];
  const double xx = r_old + dt * v.x;  // new radial Cartesian coord
  const double yy = dt * v.z;          // azimuthal arc swept this step
  const double r_new = sqrt(xx * xx + yy * yy);

  // Rotation angle swept; atan2 handles the r -> 0 region robustly.
  const double dphi = (xx != 0.0 || yy != 0.0) ? atan2(yy, xx) : 0.0;
  const double cphi = cos(dphi);
  const double sphi = sin(dphi);
  const double vr = v.x;
  const double vphi = v.z;
  // Rotate (vr, vphi) by +dphi into the new radial/azimuthal basis.
  vx[p] = vr * cphi + vphi * sphi;
  vz[p] = -vr * sphi + vphi * cphi;
  vy[p] = v.y;

  x[p] = r_new;
}

template <int ShapeOrder>
void launch(const quasar::Grid2D& g, quasar::pic::ParticleSpecies& s,
            const quasar::YeeField2D<double>& self,
            const quasar::YeeField2D<double>& ext,
            bool periodic_x, bool periodic_y, double dt, hipStream_t stream) {
  if (s.size() == 0) return;
  const dim3 block(256);
  const dim3 grid((s.size() + block.x - 1) / block.x);
  hipLaunchKernelGGL((gather_push_cyl_kernel<ShapeOrder>), grid, block, 0, stream,
                     g, s.size(), s.charge_to_mass(), s.x(), s.y(), s.x_prev(), s.y_prev(),
                     s.vx(), s.vy(), s.vz(),
                     s.alive(), self.ex.device_ptr(), self.ey.device_ptr(), self.ez.device_ptr(),
                     self.bx.device_ptr(), self.by.device_ptr(), self.bz.device_ptr(),
                     ext.ex.device_ptr(), ext.ey.device_ptr(), ext.ez.device_ptr(),
                     ext.bx.device_ptr(), ext.by.device_ptr(), ext.bz.device_ptr(),
                     periodic_x, periodic_y, dt);
}

}  // namespace

QUASAR_PIC_GATHER_PUSH_ABI(launch_pic_gather_push_cyl_shape1, 1)
QUASAR_PIC_GATHER_PUSH_ABI(launch_pic_gather_push_cyl_shape2, 2)
