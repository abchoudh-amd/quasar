// Guards the registry-driven field-solver / pusher / deposit construction the
// PIC solver relies on. These schemes are registered via namespace-scope static
// initializers in pic_solver.cpp. The pic module is linked WHOLE_ARCHIVE through
// quasar::core, so this registry-only executable must see them without naming an
// EmPic2D3V symbol. If that linkage regresses, Registry::create(...) would throw
// at runtime; this CPU-only test fails loudly instead of needing a device run.

#include "quasar/numerics/field_solver.hpp"
#include "quasar/numerics/particle_pusher.hpp"
#include "quasar/numerics/deposit.hpp"
#include "quasar/core/registry.hpp"

#include <gtest/gtest.h>

TEST(SchemeRegistryLinkage, FieldSolversAreRegistered) {
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
