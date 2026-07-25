// Range-safety regressions for complete Maxwell component updates.  Every
// stencil value below is finite.  A naive implementation nevertheless creates
// an infinite curl/product before cancellation with another signed term; the
// exponent-scaled update must retain the finite mathematical result.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/yee_field.hpp"
#include "quasar/physics/pic/kernels.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace {

using quasar::Real;

Real read_at(quasar::backend::DeviceBuffer<Real>& values,
             const quasar::Grid2D& grid, int i, int j) {
  std::vector<Real> host(values.size());
  values.copy_to_host(host.data(), host.size());
  return host[grid.index(i, j)];
}

}  // namespace

TEST(PicFdtdRange, CartesianFaradayCombinesBothCurlTermsWithOldState) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const quasar::Grid2D g{3, 3, 3.0, 3.0, 0.0, 0.0, 1};
  quasar::YeeField2D<Real> f{g};
  const Real a = Real{0.75} * std::numeric_limits<Real>::max();
  const int i = 1, j = 1;
  std::vector<Real> ex(g.storage_size(), Real{0});
  std::vector<Real> ey(g.storage_size(), Real{0});
  std::vector<Real> bz(g.storage_size(), Real{0});
  ey[g.index(i - 1, j)] = -Real{0.5} * a;
  ey[g.index(i, j)] = Real{0.5} * a;       // dEy/dx = +a
  ex[g.index(i, j - 1)] = Real{0.5} * a;
  ex[g.index(i, j)] = -Real{0.5} * a;      // dEx/dy = -a
  bz[g.index(i, j)] = a;
  f.ex.copy_from_host(ex.data(), ex.size());
  f.ey.copy_from_host(ey.data(), ey.size());
  f.bz.copy_from_host(bz.data(), bz.size());

  launch_pic_fdtd_b_order2(g, f.bx.device_ptr(), f.by.device_ptr(),
                           f.bz.device_ptr(), f.ex.device_ptr(),
                           f.ey.device_ptr(), f.ez.device_ptr(), Real{1}, nullptr);
  quasar::backend::device_synchronize(nullptr);

  // a - (+a) + (-a) = -a, although (+a)-(-a) overflows as a standalone curl.
  EXPECT_DOUBLE_EQ(read_at(f.bz, g, i, j), -a);
}

TEST(PicFdtdRange, CartesianAmpereCancelsUnrepresentableCurlAndCurrent) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const quasar::Grid2D g{3, 3, 3.0, 3.0, 0.0, 0.0, 1};
  quasar::YeeField2D<Real> f{g};
  quasar::JField2D<Real> current{g};
  const Real a = Real{0.75} * std::numeric_limits<Real>::max();
  const int i = 1, j = 1;
  std::vector<Real> by(g.storage_size(), Real{0});
  std::vector<Real> jz(g.storage_size(), Real{0});
  by[g.index(i, j)] = -Real{0.5} * a;
  by[g.index(i + 1, j)] = Real{0.5} * a;   // dBy/dx = a
  jz[g.index(i, j)] = a;
  f.by.copy_from_host(by.data(), by.size());
  current.jz.copy_from_host(jz.data(), jz.size());

  launch_pic_fdtd_e_order2(
      g, f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(),
      f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
      current.jx.device_ptr(), current.jy.device_ptr(), current.jz.device_ptr(),
      Real{2}, nullptr);
  quasar::backend::device_synchronize(nullptr);

  // dt*dBy/dx and dt*Jz are both 1.5*DBL_MAX, but cancel exactly.
  EXPECT_DOUBLE_EQ(read_at(f.ez, g, i, j), Real{0});
}

TEST(PicFdtdRange, CylindricalFaradayCombinesBothCurlTermsWithOldState) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const quasar::Grid2D g{3, 3, 3.0, 3.0, 0.0, 0.0, 1};
  quasar::YeeField2D<Real> f{g};
  const Real a = Real{0.75} * std::numeric_limits<Real>::max();
  const int i = 1, j = 1;
  std::vector<Real> er(g.storage_size(), Real{0});
  std::vector<Real> ez(g.storage_size(), Real{0});
  std::vector<Real> bphi(g.storage_size(), Real{0});
  er[g.index(i, j - 1)] = -Real{0.5} * a;
  er[g.index(i, j)] = Real{0.5} * a;       // dEr/dz = +a
  ez[g.index(i - 1, j)] = Real{0.5} * a;
  ez[g.index(i, j)] = -Real{0.5} * a;      // dEz/dr = -a
  bphi[g.index(i, j)] = a;
  f.ex.copy_from_host(er.data(), er.size());
  f.ey.copy_from_host(ez.data(), ez.size());
  f.bz.copy_from_host(bphi.data(), bphi.size());

  launch_pic_fdtd_b_cyl_order2(
      g, f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
      f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(), Real{1}, nullptr);
  quasar::backend::device_synchronize(nullptr);

  EXPECT_DOUBLE_EQ(read_at(f.bz, g, i, j), -a);
}

TEST(PicFdtdRange, CylindricalAxisAmpereCancelsRadialCurlAndCurrent) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const quasar::Grid2D g{3, 3, 3.0, 3.0, 0.0, 0.0, 1};
  quasar::YeeField2D<Real> f{g};
  quasar::JField2D<Real> current{g};
  const Real a = Real{0.75} * std::numeric_limits<Real>::max();
  const int i = 0, j = 1;
  std::vector<Real> bphi(g.storage_size(), Real{0});
  std::vector<Real> jz(g.storage_size(), Real{0});
  bphi[g.index(0, j)] = Real{0};
  bphi[g.index(1, j)] = Real{0.5} * a;
  jz[g.index(i, j)] = a;
  f.bz.copy_from_host(bphi.data(), bphi.size());
  current.jy.copy_from_host(jz.data(), jz.size());

  launch_pic_fdtd_e_cyl_order2(
      g, f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(),
      f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
      current.jx.device_ptr(), current.jy.device_ptr(), current.jz.device_ptr(),
      Real{2}, nullptr);
  quasar::backend::device_synchronize(nullptr);

  // At r=dr/2 the radial flux is 2*Bphi(dr)/dr = a.  Its dt-scaled
  // contribution and dt*Jz are both out of range but cancel before conversion.
  EXPECT_DOUBLE_EQ(read_at(f.ey, g, i, j), Real{0});
}
