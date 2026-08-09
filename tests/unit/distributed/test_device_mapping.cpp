#include "quasar/distributed/device_mapping.hpp"

#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

using quasar::distributed::DeviceIdentity;
using quasar::distributed::RankDeviceAssignment;
using quasar::distributed::RankDeviceVisibility;
using quasar::distributed::auto_assign_devices;
using quasar::distributed::make_auto_endpoint_mapping;
using quasar::distributed::make_endpoint_mapping;
using quasar::distributed::select_device_ordinals;

namespace {

DeviceIdentity gpu(int ordinal, std::string uuid) {
  return DeviceIdentity{ordinal, std::move(uuid), {}};
}

}  // namespace

TEST(DeviceMapping, PartitionsIdenticalNodeVisiblePool) {
  const std::vector<DeviceIdentity> pool{
      gpu(0, "gpu-a"), gpu(1, "gpu-b"),
      gpu(2, "gpu-c"), gpu(3, "gpu-d")};
  const std::array<RankDeviceVisibility, 2> reports{{
      {0, 0, "node-a", pool},
      {1, 1, "node-a", pool},
  }};

  const auto assignments = auto_assign_devices(reports);
  ASSERT_EQ(assignments.size(), 2u);
  ASSERT_EQ(assignments[0].owned_devices.size(), 2u);
  ASSERT_EQ(assignments[1].owned_devices.size(), 2u);
  EXPECT_EQ(assignments[0].owned_devices[0].uuid, "gpu-a");
  EXPECT_EQ(assignments[0].owned_devices[1].uuid, "gpu-b");
  EXPECT_EQ(assignments[1].owned_devices[0].uuid, "gpu-c");
  EXPECT_EQ(assignments[1].owned_devices[1].uuid, "gpu-d");
}

TEST(DeviceMapping, SharedPoolUsesEachRanksLocalOrdinalForPhysicalGpu) {
  const std::array<RankDeviceVisibility, 2> reports{{
      {0, 0, "node-a", {gpu(0, "gpu-a"), gpu(1, "gpu-b"),
                         gpu(2, "gpu-c"), gpu(3, "gpu-d")}},
      {1, 1, "node-a", {gpu(9, "gpu-d"), gpu(8, "gpu-c"),
                         gpu(7, "gpu-b"), gpu(6, "gpu-a")}},
  }};

  const auto assignments = auto_assign_devices(reports);
  ASSERT_EQ(assignments[1].owned_devices.size(), 2u);
  EXPECT_EQ(assignments[1].owned_devices[0].uuid, "gpu-c");
  EXPECT_EQ(assignments[1].owned_devices[0].ordinal, 8);
  EXPECT_EQ(assignments[1].owned_devices[1].uuid, "gpu-d");
  EXPECT_EQ(assignments[1].owned_devices[1].ordinal, 9);
}

TEST(DeviceMapping, AcceptsAlreadyDisjointLauncherMasks) {
  const std::array<RankDeviceVisibility, 2> reports{{
      {0, 0, "node-a", {gpu(0, "gpu-a"), gpu(1, "gpu-b")}},
      {1, 1, "node-a", {gpu(0, "gpu-c"), gpu(1, "gpu-d")}},
  }};

  const auto mapping = make_auto_endpoint_mapping(reports);
  EXPECT_EQ(mapping.rank_count(), 2u);
  EXPECT_EQ(mapping.devices_per_rank(), 2u);
  ASSERT_EQ(mapping.size(), 4u);
  EXPECT_EQ(mapping.endpoint(0).world_rank, 0);
  EXPECT_EQ(mapping.endpoint(0).rank_local_index, 0u);
  EXPECT_EQ(mapping.endpoint(0).device.uuid, "gpu-a");
  EXPECT_EQ(mapping.endpoint(1).device.uuid, "gpu-b");
  EXPECT_EQ(mapping.endpoint(2).world_rank, 1);
  EXPECT_EQ(mapping.endpoint(2).rank_local_index, 0u);
  EXPECT_EQ(mapping.endpoint(2).device.uuid, "gpu-c");
  EXPECT_EQ(mapping.endpoint_for(1, 1).index, 3u);
  EXPECT_EQ(mapping.endpoints_for_rank(1).size(), 2u);
}

TEST(DeviceMapping, OrdersEndpointsByWorldRankAcrossNodes) {
  const std::array<RankDeviceAssignment, 4> reports{{
      {2, 0, "node-b", {gpu(7, "gpu-c")}},
      {0, 0, "node-a", {gpu(3, "gpu-a")}},
      {3, 1, "node-b", {gpu(8, "gpu-d")}},
      {1, 1, "node-a", {gpu(4, "gpu-b")}},
  }};

  const auto mapping = make_endpoint_mapping(reports);
  ASSERT_EQ(mapping.size(), 4u);
  for (std::size_t i = 0; i < mapping.size(); ++i) {
    EXPECT_EQ(mapping.endpoint(i).index, i);
    EXPECT_EQ(mapping.endpoint(i).world_rank, static_cast<int>(i));
  }
}

TEST(DeviceMapping, ExplicitOrdinalsAreAnEligiblePoolNotPerRankOwnership) {
  const std::vector<DeviceIdentity> pool{
      gpu(0, "gpu-a"), gpu(1, "gpu-b"), gpu(2, "gpu-c")};
  const std::array<RankDeviceVisibility, 2> reports{{
      {0, 0, "node-a", pool},
      {1, 1, "node-a", pool},
  }};
  const std::array<int, 2> ordinals{2, 0};

  const auto selected = select_device_ordinals(reports, ordinals);
  const auto assignments = auto_assign_devices(selected);
  ASSERT_EQ(assignments[0].owned_devices.size(), 1u);
  ASSERT_EQ(assignments[1].owned_devices.size(), 1u);
  EXPECT_EQ(assignments[0].owned_devices[0].uuid, "gpu-c");
  EXPECT_EQ(assignments[1].owned_devices[0].uuid, "gpu-a");
}

TEST(DeviceMapping, RejectsPartiallyOverlappingMasks) {
  const std::array<RankDeviceVisibility, 2> reports{{
      {0, 0, "node-a", {gpu(0, "gpu-a"), gpu(1, "gpu-b")}},
      {1, 1, "node-a", {gpu(0, "gpu-b"), gpu(1, "gpu-c")}},
  }};
  EXPECT_THROW((void)auto_assign_devices(reports), std::invalid_argument);
}

TEST(DeviceMapping, RejectsDuplicatePhysicalOwnership) {
  const std::array<RankDeviceAssignment, 2> reports{{
      {0, 0, "node-a", {gpu(0, " GPU-A ")}},
      {1, 1, "node-a", {gpu(7, "gpu-a")}},
  }};
  EXPECT_THROW((void)make_endpoint_mapping(reports), std::invalid_argument);
}

TEST(DeviceMapping, RejectsNonUniformOrZeroGpuRanks) {
  const std::array<RankDeviceAssignment, 2> unequal{{
      {0, 0, "node-a", {gpu(0, "gpu-a"), gpu(1, "gpu-b")}},
      {1, 1, "node-a", {gpu(2, "gpu-c")}},
  }};
  EXPECT_THROW((void)make_endpoint_mapping(unequal), std::invalid_argument);

  const std::array<RankDeviceVisibility, 2> indivisible{{
      {0, 0, "node-a", {gpu(0, "gpu-a")}},
      {1, 1, "node-a", {gpu(0, "gpu-a")}},
  }};
  EXPECT_THROW((void)auto_assign_devices(indivisible), std::invalid_argument);
}

TEST(DeviceMapping, RequiresStablePhysicalIdentityAndValidRanks) {
  const std::array<RankDeviceVisibility, 1> no_identity{{
      {0, 0, "node-a", {{0, {}, {}}}},
  }};
  EXPECT_THROW((void)auto_assign_devices(no_identity), std::invalid_argument);

  const std::array<RankDeviceVisibility, 1> missing_rank_zero{{
      {1, 0, "node-a", {gpu(0, "gpu-a")}},
  }};
  EXPECT_THROW((void)auto_assign_devices(missing_rank_zero),
               std::invalid_argument);
}

TEST(DeviceMapping, RejectsUnavailableOrRepeatedExplicitOrdinal) {
  const std::array<RankDeviceVisibility, 1> reports{{
      {0, 0, "node-a", {gpu(0, "gpu-a")}},
  }};
  const std::array<int, 1> unavailable{1};
  const std::array<int, 2> repeated{0, 0};
  EXPECT_THROW((void)select_device_ordinals(reports, unavailable),
               std::invalid_argument);
  EXPECT_THROW((void)select_device_ordinals(reports, repeated),
               std::invalid_argument);
  EXPECT_THROW((void)select_device_ordinals(reports, std::span<const int>{}),
               std::invalid_argument);
}

TEST(DeviceMapping, EndpointQueriesRejectOutOfRangeIndices) {
  const std::array<RankDeviceAssignment, 1> reports{{
      {0, 0, "node-a", {gpu(0, "gpu-a")}},
  }};
  const auto mapping = make_endpoint_mapping(reports);
  EXPECT_THROW((void)mapping.endpoint(1), std::out_of_range);
  EXPECT_THROW((void)mapping.endpoint_for(1, 0), std::out_of_range);
  EXPECT_THROW((void)mapping.endpoint_for(0, 1), std::out_of_range);
  EXPECT_THROW((void)mapping.endpoints_for_rank(-1), std::out_of_range);
}
