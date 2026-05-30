// Guards the registry-driven boundary construction the PIC solver relies on:
// the particle/field BC factories live in their own static archive and are
// registered via namespace-scope static initializers. If the linker drops those
// objects (nothing else references them), the registrations vanish and the
// solver's Registry::create(...) would throw at runtime. This test fails loudly
// if that regression ever happens.

#include "quasar/boundary/boundary_condition.hpp"
#include "quasar/core/registry.hpp"

#include <gtest/gtest.h>

TEST(BoundaryRegistryLinkage, ParticleBoundariesAreRegistered) {
  const auto& reg =
      ::quasar::Registry<::quasar::boundary::IParticleBoundary>::instance();
  EXPECT_TRUE(reg.contains("periodic"));
  EXPECT_TRUE(reg.contains("specular"));
  EXPECT_TRUE(reg.contains("absorbing"));
}

TEST(BoundaryRegistryLinkage, FieldBoundariesAreRegistered) {
  const auto& reg =
      ::quasar::Registry<::quasar::boundary::IFieldBoundary>::instance();
  EXPECT_TRUE(reg.contains("periodic"));
  EXPECT_TRUE(reg.contains("pec"));
}
