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
#include "quasar/core/registry.hpp"

#include <gtest/gtest.h>

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
  EXPECT_NE(reg.create("yee_cyl_o2"), nullptr);
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
