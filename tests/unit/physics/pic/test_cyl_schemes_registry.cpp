// Guards registration + constructibility of the axisymmetric (cylindrical, r-z,
// m=0) scheme family alongside the on-axis boundary condition. The cylindrical
// field-solver / pusher / deposit schemes are registered via namespace-scope
// static initializers in pic_solver.cpp; that TU is pulled into the link by
// referencing an EmPic2D3V symbol below (the same force-link hack the Cartesian
// linkage test uses). Without it, identical-code elimination of the unreferenced
// TU would drop the QUASAR_REGISTER_* initializers and these contains()/create()
// checks would fail spuriously.
//
// The "axis" boundary lives in src/boundary/axis.cpp, which is whole-archived by
// quasar_add_module(... REGISTERS ...), so its registration does NOT need the
// force-link hack — but the cylindrical numerics schemes in pic_solver.cpp do.

#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/numerics/field_solver.hpp"
#include "quasar/numerics/particle_pusher.hpp"
#include "quasar/numerics/deposit.hpp"
#include "quasar/boundary/boundary_condition.hpp"
#include "quasar/boundary/axis.hpp"
#include "quasar/backend/device.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <utility>
#include <vector>

// Reference an EmPic2D3V symbol so the linker keeps pic_solver.cpp (which owns
// the QUASAR_REGISTER_* static initializers, including the cylindrical schemes)
// in this executable. A pointer to the out-of-line step() member is defined in
// that TU, so naming it forces the link.
namespace {
auto kForceLink = &quasar::pic::EmPic2D3V::step;
}

TEST(CylSchemeRegistry, ForceLinkSymbolResolved) {
  ASSERT_NE(kForceLink, nullptr);
}

TEST(CylSchemeRegistry, CylindricalFieldSolverRegisteredAndConstructible) {
  const auto& reg = ::quasar::Registry<::quasar::numerics::IFieldSolver>::instance();
  ASSERT_TRUE(reg.contains("yee_cyl_o2"));
  ASSERT_TRUE(reg.contains("yee_cyl_o4"));
  EXPECT_NE(reg.create("yee_cyl_o2"), nullptr);
  EXPECT_NE(reg.create("yee_cyl_o4"), nullptr);
}

TEST(CylAxisBoundary, FaceAndCellCentreGhostsUseTheirOwnMirrors) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  quasar::Grid2D g{8, 4, 1.0, 1.0, 0.0, 0.0, 2};
  quasar::YeeField2D<quasar::Real> f{g};
  std::vector<double> ex(g.storage_size(), 0.0);
  std::vector<double> ey(g.storage_size(), 0.0);
  for (int j = 0; j < g.ny; ++j) {
    ex[g.index(0, j)] = 99.0;
    ex[g.index(1, j)] = 3.0;
    ex[g.index(2, j)] = 5.0;
    ey[g.index(0, j)] = 7.0;
    ey[g.index(1, j)] = 11.0;
  }
  f.ex.copy_from_host(ex.data(), ex.size());
  f.ey.copy_from_host(ey.data(), ey.size());
  quasar::boundary::AxisFieldBC bc;
  bc.fill_ghosts(f, quasar::Side::x_lo);
  f.ex.copy_to_host(ex.data(), ex.size());
  f.ey.copy_to_host(ey.data(), ey.size());
  for (int j = 0; j < g.ny; ++j) {
    EXPECT_DOUBLE_EQ(ex[g.index(0, j)], 0.0);
    EXPECT_DOUBLE_EQ(ex[g.index(-1, j)], -3.0);
    EXPECT_DOUBLE_EQ(ex[g.index(-2, j)], -5.0);
    EXPECT_DOUBLE_EQ(ey[g.index(-1, j)], 7.0);
    EXPECT_DOUBLE_EQ(ey[g.index(-2, j)], 11.0);
  }
}

TEST(CylAxisBoundary, ParticleCrossingFlipsRadialAndAzimuthalVelocity) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  quasar::Grid2D g{8, 4, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"test", 1.0, 1.0, 1}};
  sp.set_grid(g);
  sp.set_host_particles({-0.02}, {0.5}, {-0.3}, {0.1}, {0.4}, {1.0});
  quasar::boundary::AxisParticleBC bc;
  bc.apply(sp, quasar::Side::x_lo);
  const auto snap = sp.to_host();
  EXPECT_NEAR(snap.x[0], 0.02, 1.0e-15);
  EXPECT_NEAR(snap.vx[0], 0.3, 1.0e-15);
  EXPECT_NEAR(snap.vy[0], 0.1, 1.0e-15);
  EXPECT_NEAR(snap.vz[0], -0.4, 1.0e-15);
}

TEST(CylAnnulus, KeepsConfiguredInnerBoundaryAndStepsAtFourthOrder) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  quasar::Grid2D g{16, 16, 1.0, 1.0, 0.5, 0.0, 2};
  quasar::pic::EmPicConfig cfg{g, 4, "cic"};
  cfg.geometry = "cylindrical";
  cfg.boundary.field[0] = "pec";
  cfg.boundary.field[1] = "pec";
  cfg.boundary.particle[0] = "specular";
  cfg.boundary.particle[1] = "specular";
  quasar::pic::EmPic2D3V solver{cfg};
  EXPECT_EQ(solver.config().boundary.field[0], "pec");
  EXPECT_EQ(solver.config().boundary.particle[0], "specular");

  quasar::pic::ParticleSpecies sp{
      quasar::pic::SpeciesConfig{"neutral", 0.0, 1.0, 1}};
  sp.set_host_particles({0.75}, {0.5}, {0.1}, {0.0}, {0.0}, {0.0});
  solver.add_species(std::move(sp));
  EXPECT_NO_THROW(solver.step(0.02));
}

TEST(CylAnnulus, FirstColumnIsNotTreatedAsAxisAndUsesCylindricalWalls) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  quasar::Grid2D g{8, 4, 1.0, 1.0, 0.5, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  cfg.geometry = "cylindrical";
  cfg.boundary.field[0] = "pec";
  cfg.boundary.field[1] = "pec";
  cfg.boundary.particle[0] = "specular";
  cfg.boundary.particle[1] = "specular";
  quasar::pic::EmPic2D3V solver{cfg};

  std::vector<double> ex(g.storage_size(), 1.0);
  // Cylindrical PEC treats high radial Ex as a physical normal face. PEC pins
  // tangential E but leaves this normal degree of freedom intact (surface charge
  // may support it); only its ghosts receive an even continuation.
  for (int j = 0; j < g.ny; ++j) ex[g.index(g.nx, j)] = 7.0;
  solver.fields().ex.copy_from_host(ex.data(), ex.size());
  solver.step(0.01);
  solver.fields().ex.copy_to_host(ex.data(), ex.size());
  for (int j = 0; j < g.ny; ++j) {
    EXPECT_NEAR(ex[g.index(0, j)], 1.0, 1.0e-14);
    EXPECT_NEAR(ex[g.index(g.nx, j)], 7.0, 1.0e-14);
  }
}

TEST(CylAnnulus, RejectsNegativeRadiusOrigin) {
  quasar::Grid2D g{16, 16, 1.0, 1.0, -0.1, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  cfg.geometry = "cylindrical";
  cfg.boundary.field[1] = "pec";
  cfg.boundary.particle[1] = "specular";
  EXPECT_THROW(quasar::pic::EmPic2D3V solver{cfg}, std::invalid_argument);
}

TEST(CylSchemeRegistry, RejectsFakeZeroGradientOutflow) {
  quasar::Grid2D g{8, 8, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPicConfig cfg{g, 2, "cic"};
  cfg.geometry = "cylindrical";
  cfg.boundary.field[1] = "pec";
  cfg.boundary.particle[1] = "specular";
  cfg.boundary.field[3] = "outflow";
  EXPECT_THROW(quasar::pic::EmPic2D3V solver{cfg}, std::invalid_argument);
}

TEST(CylSchemeRegistry, CylindricalPushersRegisteredAndConstructible) {
  const auto& reg = ::quasar::Registry<::quasar::numerics::IParticlePusher>::instance();
  ASSERT_TRUE(reg.contains("boris_cyl_cic"));
  ASSERT_TRUE(reg.contains("boris_cyl_tsc"));
  EXPECT_NE(reg.create("boris_cyl_cic"), nullptr);
  EXPECT_NE(reg.create("boris_cyl_tsc"), nullptr);
}

TEST(CylSchemeRegistry, CylindricalDepositSchemesRegisteredAndConstructible) {
  const auto& reg = ::quasar::Registry<::quasar::numerics::IDepositScheme>::instance();
  ASSERT_TRUE(reg.contains("esirkepov_cyl_cic"));
  ASSERT_TRUE(reg.contains("esirkepov_cyl_tsc"));
  EXPECT_NE(reg.create("esirkepov_cyl_cic"), nullptr);
  EXPECT_NE(reg.create("esirkepov_cyl_tsc"), nullptr);
}

TEST(CylSchemeRegistry, AxisFieldBoundaryRegisteredAndConstructible) {
  const auto& reg = ::quasar::Registry<::quasar::boundary::IFieldBoundary>::instance();
  ASSERT_TRUE(reg.contains("axis"));
  EXPECT_NE(reg.create("axis"), nullptr);
}

TEST(CylSchemeRegistry, AxisParticleBoundaryRegisteredAndConstructible) {
  const auto& reg = ::quasar::Registry<::quasar::boundary::IParticleBoundary>::instance();
  ASSERT_TRUE(reg.contains("axis"));
  EXPECT_NE(reg.create("axis"), nullptr);
}

// Sanity: the additive cylindrical change must not have dropped the Cartesian
// schemes that the rest of the suite relies on.
TEST(CylSchemeRegistry, CartesianSchemesStillPresent) {
  EXPECT_TRUE(::quasar::Registry<::quasar::numerics::IFieldSolver>::instance().contains("yee_o2"));
  EXPECT_TRUE(::quasar::Registry<::quasar::numerics::IParticlePusher>::instance().contains("boris_cic"));
  EXPECT_TRUE(::quasar::Registry<::quasar::numerics::IDepositScheme>::instance().contains("esirkepov_cic"));
}
