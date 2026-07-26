// Registry-selected analytic background-magnetic-field profiles. Pure host:
// only registry creation/configuration and analytic sampling are exercised.

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
  EXPECT_TRUE(profile->globally_curl_free());

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

TEST(MhdBackgroundProfile, UniformParametersSetComponents) {
  auto profile = ::quasar::Registry<IMhdBackgroundProfile>::instance().create("uniform");
  ASSERT_TRUE(profile->set_parameter("bx0", Real{1.25}));
  ASSERT_TRUE(profile->set_parameter("by0", Real{-0.5}));
  ASSERT_TRUE(profile->set_parameter("bz0", Real{3}));
  EXPECT_FALSE(profile->set_parameter("unknown", Real{1}));
  EXPECT_DOUBLE_EQ(profile->sample(0, Real{4}, Real{-2}), Real{1.25});
  EXPECT_DOUBLE_EQ(profile->sample(1, Real{4}, Real{-2}), Real{-0.5});
  EXPECT_DOUBLE_EQ(profile->sample(2, Real{4}, Real{-2}), Real{3});
}

TEST(MhdBackgroundProfile, LinearVacuumIsConfiguredAndSolenoidal) {
  auto profile =
      ::quasar::Registry<IMhdBackgroundProfile>::instance().create("linear_vacuum");
  EXPECT_TRUE(profile->globally_curl_free());
  ASSERT_TRUE(profile->set_parameter("gradient", Real{2}));
  ASSERT_TRUE(profile->set_parameter("shear", Real{-0.25}));
  EXPECT_FALSE(profile->set_parameter("unknown", Real{1}));
  EXPECT_DOUBLE_EQ(profile->sample(0, Real{3}, Real{4}), Real{5});
  EXPECT_DOUBLE_EQ(profile->sample(1, Real{3}, Real{4}), Real{-8.75});
  EXPECT_DOUBLE_EQ(profile->sample(2, Real{3}, Real{4}), Real{0});
}

}  // namespace
