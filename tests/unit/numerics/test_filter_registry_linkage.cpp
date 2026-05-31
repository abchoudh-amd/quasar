// Guards the registry-driven current-filter construction the PIC solver relies
// on. The filter factories are registered via namespace-scope static
// initializers in the numerics module (linked WHOLE_ARCHIVE into quasar::core).
// If that linkage ever regresses, the solver's Registry::create(...) for a deck
// current_filter would throw at runtime; this test fails loudly instead.

#include "quasar/numerics/filter.hpp"
#include "quasar/core/registry.hpp"

#include <gtest/gtest.h>

TEST(FilterRegistryLinkage, CurrentFiltersAreRegistered) {
  const auto& reg =
      ::quasar::Registry<::quasar::numerics::ICurrentFilter>::instance();
  EXPECT_TRUE(reg.contains("binomial"));
  EXPECT_TRUE(reg.contains("compensated_binomial"));
}
