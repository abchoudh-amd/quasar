#include "quasar/distributed/transport.hpp"

#include <gtest/gtest.h>

TEST(DistributedTransportPolicy, AutoFallsBackWhenDirectProbeFails) {
  const auto resolution = quasar::distributed::resolve_transport_policy(
      quasar::distributed::TransportPolicy::automatic,
      quasar::distributed::DirectCapability{true, false});
  EXPECT_EQ(resolution.interprocess,
            quasar::distributed::TransportPolicy::staged);
  EXPECT_FALSE(resolution.uses_direct_mpi());
}

TEST(DistributedTransportPolicy, AutoUsesDirectOnlyAfterQueryAndProbe) {
  const auto resolution = quasar::distributed::resolve_transport_policy(
      quasar::distributed::TransportPolicy::automatic,
      quasar::distributed::DirectCapability{true, true});
  EXPECT_EQ(resolution.interprocess,
            quasar::distributed::TransportPolicy::direct);
  EXPECT_TRUE(resolution.uses_direct_mpi());
}

TEST(DistributedTransportPolicy, DirectFailsInsteadOfFallingBack) {
  EXPECT_THROW(
      (void)quasar::distributed::resolve_transport_policy(
          quasar::distributed::TransportPolicy::direct,
          quasar::distributed::DirectCapability{false, false}),
      std::runtime_error);
}

TEST(DistributedTransportPolicy, ParserAcceptsOnlyPublicVocabulary) {
  EXPECT_EQ(quasar::distributed::parse_transport_policy("auto"),
            quasar::distributed::TransportPolicy::automatic);
  EXPECT_EQ(quasar::distributed::parse_transport_policy("staged"),
            quasar::distributed::TransportPolicy::staged);
  EXPECT_EQ(quasar::distributed::parse_transport_policy("direct"),
            quasar::distributed::TransportPolicy::direct);
  EXPECT_THROW(
      (void)quasar::distributed::parse_transport_policy("fallback"),
      std::invalid_argument);
}
