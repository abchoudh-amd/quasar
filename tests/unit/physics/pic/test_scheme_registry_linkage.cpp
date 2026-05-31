// Guards the registry-driven field-solver / pusher / deposit construction the
// PIC solver relies on. These schemes are registered via namespace-scope static
// initializers in pic_solver.cpp; that TU is pulled into the link by referencing
// an EmPic2D3V symbol below (the same TU that owns the registrations). If the
// registrations ever regress, EmPic2D3V's Registry::create(...) would throw at
// runtime; this CPU-only test fails loudly instead of needing a device run.

#include "quasar/physics/pic/pic_solver.hpp"
#include "quasar/numerics/field_solver.hpp"
#include "quasar/numerics/particle_pusher.hpp"
#include "quasar/numerics/deposit.hpp"
#include "quasar/core/registry.hpp"

#include <gtest/gtest.h>

// Reference an EmPic2D3V symbol so the linker keeps pic_solver.cpp (which owns
// the QUASAR_REGISTER_* static initializers) in this executable. A pointer to the
// out-of-line step() member is defined in that TU, so naming it forces the link.
namespace {
auto kForceLink = &quasar::pic::EmPic2D3V::step;
}

TEST(SchemeRegistryLinkage, FieldSolversAreRegistered) {
  ASSERT_NE(kForceLink, nullptr);
  const auto& reg = ::quasar::Registry<::quasar::numerics::IFieldSolver>::instance();
  EXPECT_TRUE(reg.contains("yee_o2"));
  EXPECT_TRUE(reg.contains("yee_o4"));
}

TEST(SchemeRegistryLinkage, PushersAreRegistered) {
  const auto& reg = ::quasar::Registry<::quasar::numerics::IParticlePusher>::instance();
  EXPECT_TRUE(reg.contains("boris_cic"));
  EXPECT_TRUE(reg.contains("boris_tsc"));
}

TEST(SchemeRegistryLinkage, DepositSchemesAreRegistered) {
  const auto& reg = ::quasar::Registry<::quasar::numerics::IDepositScheme>::instance();
  EXPECT_TRUE(reg.contains("esirkepov_cic"));
  EXPECT_TRUE(reg.contains("esirkepov_tsc"));
}
