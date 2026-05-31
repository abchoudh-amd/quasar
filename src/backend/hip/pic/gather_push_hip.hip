#include "quasar/core/yee_field.hpp"
#include "quasar/numerics/shape.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"

#include <hip/hip_runtime.h>

namespace {

// Per-axis gather indexing. A periodic axis wraps (historical behaviour); a
// non-periodic (wall) axis clamps the shape node into the ghost layer via
// Grid2D::index, reading the field values the field-boundary layer maintains
// there (mirror/one-sided closure for the self field, edge-replicated external
// field) instead of wrapping to the far edge. Mirrors deposit_index().
__device__ inline int clamp_axis(int i, int lo, int hi) {
  return i < lo ? lo : (i > hi ? hi : i);
}

__device__ inline std::size_t gather_index(const quasar::Grid2D& g, int i, int j,
                                           bool periodic_x, bool periodic_y) {
  const int ii = periodic_x ? g.wrap_i(i) : clamp_axis(i, -g.nghost, g.nx - 1 + g.nghost);
  const int jj = periodic_y ? g.wrap_j(j) : clamp_axis(j, -g.nghost, g.ny - 1 + g.nghost);
  return g.index(ii, jj);
}

template <int ShapeOrder>
__device__ quasar::Vec3 gather_field(const quasar::Grid2D& g,
                                     const double* __restrict__ x_comp,
                                     const double* __restrict__ y_comp,
                                     const double* __restrict__ z_comp,
                                     double x, double y,
                                     bool periodic_x, bool periodic_y) {
  const auto w = quasar::numerics::shape_weights_2d<ShapeOrder>(x, y, g);
  quasar::Vec3 out{0, 0, 0};
  for (int jj = 0; jj < w.ny; ++jj) {
    for (int ii = 0; ii < w.nx; ++ii) {
      const double ww = w.wx[ii] * w.wy[jj];
      const std::size_t k = gather_index(g, w.ix[ii], w.iy[jj], periodic_x, periodic_y);
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

  quasar::Vec3 e = gather_field<ShapeOrder>(g, ex, ey, ez, x[p], y[p], periodic_x, periodic_y);
  const quasar::Vec3 ee =
      gather_field<ShapeOrder>(g, ext_ex, ext_ey, ext_ez, x[p], y[p], periodic_x, periodic_y);
  e = e + ee;
  quasar::Vec3 b = gather_field<ShapeOrder>(g, bx, by, bz, x[p], y[p], periodic_x, periodic_y);
  const quasar::Vec3 be =
      gather_field<ShapeOrder>(g, ext_bx, ext_by, ext_bz, x[p], y[p], periodic_x, periodic_y);
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

extern "C" void launch_pic_gather_push_shape1(const quasar::Grid2D& g,
                                              quasar::pic::ParticleSpecies& s,
                                              const quasar::YeeField2D<double>& self,
                                              const quasar::YeeField2D<double>& ext,
                                              int periodic_x, int periodic_y,
                                              double dt, quasar_stream_t stream) {
  launch<1>(g, s, self, ext, periodic_x != 0, periodic_y != 0, dt,
            static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_gather_push_shape2(const quasar::Grid2D& g,
                                              quasar::pic::ParticleSpecies& s,
                                              const quasar::YeeField2D<double>& self,
                                              const quasar::YeeField2D<double>& ext,
                                              int periodic_x, int periodic_y,
                                              double dt, quasar_stream_t stream) {
  launch<2>(g, s, self, ext, periodic_x != 0, periodic_y != 0, dt,
            static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}
