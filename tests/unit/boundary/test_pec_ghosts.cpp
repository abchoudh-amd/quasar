#include "quasar/backend/device.hpp"
#include "quasar/boundary/axis.hpp"
#include "quasar/boundary/wall.hpp"

#include <gtest/gtest.h>

#include <vector>

TEST(PecGhosts, LauncherIsCallable) {
  if (!quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  quasar::YeeField2D<double> field{quasar::Grid2D{4, 4, 1.0, 1.0, 0.0, 0.0, 2}};
  quasar::boundary::PecFieldBC bc;
  EXPECT_NO_THROW(bc.fill_ghosts(field, quasar::Side::y_lo));
}

TEST(PecGhosts, CartesianWallUsesParityAtPhysicalWall) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const quasar::Grid2D g{4, 3, 1.0, 1.0, 0.0, 0.0, 2};
  quasar::YeeField2D<double> f{g};
  std::vector<double> ex(g.storage_size(), 0.0), ey(g.storage_size(), 0.0),
      ez(g.storage_size(), 0.0), bx(g.storage_size(), 0.0),
      by(g.storage_size(), 0.0), bz(g.storage_size(), 0.0);
  const int j = 1;
  ex[g.index(0, j)] = 10.0;
  ex[g.index(1, j)] = 11.0;
  ey[g.index(0, j)] = 20.0;
  ey[g.index(1, j)] = 21.0;
  ez[g.index(0, j)] = 30.0;
  ez[g.index(1, j)] = 31.0;
  bx[g.index(0, j)] = 40.0;
  bx[g.index(1, j)] = 41.0;
  by[g.index(0, j)] = 50.0;
  by[g.index(1, j)] = 51.0;
  bz[g.index(0, j)] = 60.0;
  bz[g.index(1, j)] = 61.0;
  f.ex.copy_from_host(ex.data(), ex.size());
  f.ey.copy_from_host(ey.data(), ey.size());
  f.ez.copy_from_host(ez.data(), ez.size());
  f.bx.copy_from_host(bx.data(), bx.size());
  f.by.copy_from_host(by.data(), by.size());
  f.bz.copy_from_host(bz.data(), bz.size());

  quasar::boundary::PecFieldBC bc;
  bc.fill_ghosts(f, quasar::Side::x_lo);
  f.ex.copy_to_host(ex.data(), ex.size());
  f.ey.copy_to_host(ey.data(), ey.size());
  f.ez.copy_to_host(ez.data(), ez.size());
  f.bx.copy_to_host(bx.data(), bx.size());
  f.by.copy_to_host(by.data(), by.size());
  f.bz.copy_to_host(bz.data(), bz.size());

  // Normal E and tangential B are face-located/even.
  EXPECT_DOUBLE_EQ(ex[g.index(-1, j)], 11.0);
  EXPECT_DOUBLE_EQ(by[g.index(-1, j)], 51.0);
  EXPECT_DOUBLE_EQ(bz[g.index(-1, j)], 61.0);
  // Tangential E and normal B are cell-centred/odd. Their first physical sample
  // remains live; only its mirror is negated, so interpolation at x=0 is zero.
  EXPECT_DOUBLE_EQ(ey[g.index(0, j)], 20.0);
  EXPECT_DOUBLE_EQ(ez[g.index(0, j)], 30.0);
  EXPECT_DOUBLE_EQ(bx[g.index(0, j)], 40.0);
  EXPECT_DOUBLE_EQ(ey[g.index(-1, j)], -20.0);
  EXPECT_DOUBLE_EQ(ez[g.index(-1, j)], -30.0);
  EXPECT_DOUBLE_EQ(bx[g.index(-1, j)], -40.0);
  EXPECT_DOUBLE_EQ(ey[g.index(-2, j)], -21.0);
}

TEST(PecGhosts, HighFaceIsPhysicalAndHaloStartsBeyondIt) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const quasar::Grid2D g{4, 3, 1.0, 1.0, 0.0, 0.0, 2};
  quasar::YeeField2D<double> f{g};
  std::vector<double> ex(g.storage_size(), 0.0), ey(g.storage_size(), 0.0),
      by(g.storage_size(), 0.0);
  const int j = 1;
  ex[g.index(g.nx, j)] = 7.0;       // physical normal-E high face
  ex[g.index(g.nx - 1, j)] = 6.0;
  by[g.index(g.nx, j)] = 9.0;       // physical tangential-B high face
  by[g.index(g.nx - 1, j)] = 8.0;
  ey[g.index(g.nx - 1, j)] = 5.0;   // cell-centred tangential E
  ey[g.index(g.nx - 2, j)] = 4.0;
  f.ex.copy_from_host(ex.data(), ex.size());
  f.ey.copy_from_host(ey.data(), ey.size());
  f.by.copy_from_host(by.data(), by.size());

  quasar::boundary::PecFieldBC bc;
  bc.fill_ghosts(f, quasar::Side::x_hi);
  f.ex.copy_to_host(ex.data(), ex.size());
  f.ey.copy_to_host(ey.data(), ey.size());
  f.by.copy_to_host(by.data(), by.size());

  EXPECT_DOUBLE_EQ(ex[g.index(g.nx, j)], 7.0);
  EXPECT_DOUBLE_EQ(by[g.index(g.nx, j)], 9.0);
  EXPECT_DOUBLE_EQ(ex[g.index(g.nx + 1, j)], 6.0);
  EXPECT_DOUBLE_EQ(by[g.index(g.nx + 1, j)], 8.0);
  EXPECT_DOUBLE_EQ(ey[g.index(g.nx, j)], -5.0);
  EXPECT_DOUBLE_EQ(ey[g.index(g.nx + 1, j)], -4.0);
}

TEST(PecGhosts, CylindricalAxisClosesPhysicalHighZRowAndCorner) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const quasar::Grid2D g{4, 3, 1.0, 1.0, 0.0, 0.0, 2};
  quasar::YeeField2D<double> f{g};
  std::vector<double> ey(g.storage_size(), 0.0), bx(g.storage_size(), 0.0),
      bz(g.storage_size(), 0.0);
  // Ez is radial-centred/even; Br and Bphi are radial-face/odd. All three have
  // a meaningful j=ny path in the cylindrical layout (Ez/Br/Bphi high row).
  ey[g.index(0, g.ny)] = 3.0;
  bx[g.index(0, g.ny)] = 8.0;
  bx[g.index(1, g.ny)] = 4.0;
  bz[g.index(0, g.ny)] = 9.0;
  bz[g.index(1, g.ny)] = 5.0;
  f.ey.copy_from_host(ey.data(), ey.size());
  f.bx.copy_from_host(bx.data(), bx.size());
  f.bz.copy_from_host(bz.data(), bz.size());

  quasar::boundary::AxisFieldBC axis;
  axis.fill_ghosts(f, quasar::Side::x_lo);
  f.ey.copy_to_host(ey.data(), ey.size());
  f.bx.copy_to_host(bx.data(), bx.size());
  f.bz.copy_to_host(bz.data(), bz.size());

  EXPECT_DOUBLE_EQ(ey[g.index(-1, g.ny)], 3.0);
  EXPECT_DOUBLE_EQ(bx[g.index(0, g.ny)], 0.0);
  EXPECT_DOUBLE_EQ(bz[g.index(0, g.ny)], 0.0);
  EXPECT_DOUBLE_EQ(bx[g.index(-1, g.ny)], -4.0);
  EXPECT_DOUBLE_EQ(bz[g.index(-1, g.ny)], -5.0);
}
