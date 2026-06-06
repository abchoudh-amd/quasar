// RED-phase tests for the IMhdBackgroundProfile pluggable kind: a registry-
// selected analytic background-magnetic-field profile. The interface and its
// "uniform" registration do not exist yet, so this binary fails to compile/link
// until the implementation lands. Pure host: only registry create()/contains()/
// names() and the analytic sample() are exercised, touching no device memory.

#include "quasar/numerics/mhd_background_profile.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/core/types.hpp"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

namespace {

using quasar::Real;
using quasar::numerics::IMhdBackgroundProfile;

// 1. The default "uniform" profile is registered and constructs to a live object.
TEST(MhdBackgroundProfile, UniformIsRegistered) {
  const auto& reg = ::quasar::Registry<IMhdBackgroundProfile>::instance();
  ASSERT_TRUE(reg.contains("uniform"));
  std::unique_ptr<IMhdBackgroundProfile> profile = reg.create("uniform");
  EXPECT_NE(profile, nullptr);
}

// 2. Creating an unknown name throws std::out_of_range (registry contract).
TEST(MhdBackgroundProfile, UnknownNameThrows) {
  const auto& reg = ::quasar::Registry<IMhdBackgroundProfile>::instance();
  EXPECT_THROW(reg.create("does_not_exist"), std::out_of_range);
}

// 3. The uniform profile is spatially constant per component: sampling each of
//    comp {0=b0x, 1=b0y, 2=b0z} at two distinct (x,y) points yields the same
//    value (to round-off).
TEST(MhdBackgroundProfile, UniformIsSpatiallyConstant) {
  const auto& reg = ::quasar::Registry<IMhdBackgroundProfile>::instance();
  std::unique_ptr<IMhdBackgroundProfile> profile = reg.create("uniform");
  ASSERT_NE(profile, nullptr);

  const Real x1 = 0.0;
  const Real y1 = 0.0;
  const Real x2 = 1.25;
  const Real y2 = -0.75;

  for (int comp = 0; comp < 3; ++comp) {
    const Real v1 = profile->sample(comp, x1, y1);
    const Real v2 = profile->sample(comp, x2, y2);
    EXPECT_DOUBLE_EQ(v1, v2) << "uniform profile not constant for comp " << comp;
  }
}

// 4. Registry introspection lists "uniform" among the registered profiles.
TEST(MhdBackgroundProfile, NamesContainUniform) {
  const auto& reg = ::quasar::Registry<IMhdBackgroundProfile>::instance();
  const std::vector<std::string> names = reg.names();
  EXPECT_NE(std::find(names.begin(), names.end(), "uniform"), names.end());
}

}  // namespace
