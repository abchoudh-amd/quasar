#include "quasar/core/yee_field.hpp"
#include "quasar/numerics/shape.hpp"
#include "quasar/physics/pic/species.hpp"

#include <hip/hip_runtime.h>

namespace {

template <int ShapeOrder>
__device__ quasar::Vec3 gather_field(const quasar::Grid2D& g,
                                     const double* x_comp, const double* y_comp,
                                     const double* z_comp,
                                     double x, double y) {
  const auto w = quasar::numerics::shape_weights_2d<ShapeOrder>(x, y, g);
  quasar::Vec3 out{0, 0, 0};
  for (int jj = 0; jj < w.ny; ++jj) {
    for (int ii = 0; ii < w.nx; ++ii) {
      const double ww = w.wx[ii] * w.wy[jj];
      const std::size_t k = g.periodic_index(w.ix[ii], w.iy[jj]);
      out.x += ww * x_comp[k];
      out.y += ww * y_comp[k];
      out.z += ww * z_comp[k];
    }
  }
  return out;
}

__device__ quasar::Vec3 boris(double qm, quasar::Vec3 v, quasar::Vec3 e,
                              quasar::Vec3 b, double dt) {
  const quasar::Vec3 v_minus = v + (qm * dt * 0.5) * e;
  const quasar::Vec3 t = (qm * dt * 0.5) * b;
  const double t2 = quasar::dot(t, t);
  const quasar::Vec3 s = (2.0 / (1.0 + t2)) * t;
  const quasar::Vec3 v_prime = v_minus + quasar::cross(v_minus, t);
  const quasar::Vec3 v_plus = v_minus + quasar::cross(v_prime, s);
  return v_plus + (qm * dt * 0.5) * e;
}

template <int ShapeOrder>
__global__ void gather_push_kernel(quasar::Grid2D g, std::size_t n, double qm,
                                   double* x, double* y,
                                   double* vx, double* vy, double* vz,
                                   const unsigned char* alive,
                                   const double* ex, const double* ey, const double* ez,
                                   const double* bx, const double* by, const double* bz,
                                   const double* ext_ex, const double* ext_ey, const double* ext_ez,
                                   const double* ext_bx, const double* ext_by, const double* ext_bz,
                                   double dt) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;

  quasar::Vec3 e = gather_field<ShapeOrder>(g, ex, ey, ez, x[p], y[p]);
  const quasar::Vec3 ee = gather_field<ShapeOrder>(g, ext_ex, ext_ey, ext_ez, x[p], y[p]);
  e = e + ee;
  quasar::Vec3 b = gather_field<ShapeOrder>(g, bx, by, bz, x[p], y[p]);
  const quasar::Vec3 be = gather_field<ShapeOrder>(g, ext_bx, ext_by, ext_bz, x[p], y[p]);
  b = b + be;
  quasar::Vec3 v = boris(qm, quasar::Vec3{vx[p], vy[p], vz[p]}, e, b, dt);
  vx[p] = v.x;
  vy[p] = v.y;
  vz[p] = v.z;
  x[p] = x[p] + dt * v.x;
  y[p] = y[p] + dt * v.y;
}

template <int ShapeOrder>
void launch(const quasar::Grid2D& g, quasar::pic::ParticleSpecies& s,
            const quasar::YeeField2D<double>& self,
            const quasar::YeeField2D<double>& ext,
            double dt, hipStream_t stream) {
  const dim3 block(256);
  const dim3 grid((s.size() + block.x - 1) / block.x);
  hipLaunchKernelGGL((gather_push_kernel<ShapeOrder>), grid, block, 0, stream,
                     g, s.size(), s.charge_to_mass(), s.x(), s.y(), s.vx(), s.vy(), s.vz(),
                     s.alive(), self.ex.device_ptr(), self.ey.device_ptr(), self.ez.device_ptr(),
                     self.bx.device_ptr(), self.by.device_ptr(), self.bz.device_ptr(),
                     ext.ex.device_ptr(), ext.ey.device_ptr(), ext.ez.device_ptr(),
                     ext.bx.device_ptr(), ext.by.device_ptr(), ext.bz.device_ptr(), dt);
}

}  // namespace

extern "C" void launch_pic_gather_push_shape1(const quasar::Grid2D& g,
                                              quasar::pic::ParticleSpecies& s,
                                              const quasar::YeeField2D<double>& self,
                                              const quasar::YeeField2D<double>& ext,
                                              double dt, hipStream_t stream) {
  launch<1>(g, s, self, ext, dt, stream);
}

extern "C" void launch_pic_gather_push_shape2(const quasar::Grid2D& g,
                                              quasar::pic::ParticleSpecies& s,
                                              const quasar::YeeField2D<double>& self,
                                              const quasar::YeeField2D<double>& ext,
                                              double dt, hipStream_t stream) {
  launch<2>(g, s, self, ext, dt, stream);
}
