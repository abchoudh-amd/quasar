#include "quasar/distributed/collective_error.hpp"
#include "quasar/distributed/mpi_runtime.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <type_traits>

using quasar::distributed::CollectiveDecision;
using quasar::distributed::CollectiveErrorRecord;
using quasar::distributed::ErrorDisposition;
using quasar::distributed::collective_message_capacity;
using quasar::distributed::collective_protocol_error;
using quasar::distributed::resolve_collective_errors;

TEST(CollectiveError, AcceptsWhenEveryRankSucceeds) {
  const std::array records{
      CollectiveErrorRecord::success(7, 2, "halo-x"),
      CollectiveErrorRecord::success(7, 0, "halo-x"),
      CollectiveErrorRecord::success(7, 1, "halo-x"),
  };
  const auto result = resolve_collective_errors(records);
  EXPECT_EQ(result.decision, CollectiveDecision::accept);
  EXPECT_TRUE(result.accepted());
  EXPECT_EQ(result.participant_count, 3u);
  EXPECT_EQ(result.retry_count, 0u);
  EXPECT_EQ(result.failure_count, 0u);
  EXPECT_EQ(result.representative.rank, 0);
}

TEST(CollectiveError, RetryIsCollectiveAndDeterministic) {
  const std::array records{
      CollectiveErrorRecord::retry(9, 2, 5, 42, "positivity", "tile 5"),
      CollectiveErrorRecord::success(9, 0, "positivity"),
      CollectiveErrorRecord::retry(9, 1, 3, 41, "positivity", "tile 3"),
  };
  const auto result = resolve_collective_errors(records);
  EXPECT_EQ(result.decision, CollectiveDecision::retry);
  EXPECT_EQ(result.retry_count, 2u);
  EXPECT_EQ(result.failure_count, 0u);
  EXPECT_EQ(result.representative.rank, 1);
  EXPECT_EQ(result.representative.endpoint, 3);
  EXPECT_EQ(result.representative.message_text(), "tile 3");
}

TEST(CollectiveError, FatalOverridesRetry) {
  const std::array records{
      CollectiveErrorRecord::retry(11, 0, 0, 1, "deposit", "retry"),
      CollectiveErrorRecord::failure(11, 1, 2, 8, "deposit", "fatal 1"),
      CollectiveErrorRecord::failure(11, 2, 4, 7, "deposit", "fatal 2"),
  };
  const auto result = resolve_collective_errors(records);
  EXPECT_EQ(result.decision, CollectiveDecision::fail);
  EXPECT_EQ(result.retry_count, 1u);
  EXPECT_EQ(result.failure_count, 2u);
  EXPECT_EQ(result.representative.rank, 1);
  EXPECT_EQ(result.representative.message_text(), "fatal 1");
}

TEST(CollectiveError, EpochDisagreementBecomesProtocolFailure) {
  const std::array records{
      CollectiveErrorRecord::success(3, 0, "migration"),
      CollectiveErrorRecord::success(4, 1, "migration"),
  };
  const auto result = resolve_collective_errors(records);
  EXPECT_EQ(result.decision, CollectiveDecision::fail);
  EXPECT_EQ(result.representative.code, collective_protocol_error);
  EXPECT_EQ(result.representative.rank, -1);
  EXPECT_NE(result.representative.message_text().find("epoch"),
            std::string_view::npos);
}

TEST(CollectiveError, PhaseDisagreementBecomesProtocolFailure) {
  const std::array records{
      CollectiveErrorRecord::success(3, 0, "migration"),
      CollectiveErrorRecord::success(3, 1, "source-exchange"),
  };
  const auto result = resolve_collective_errors(records);
  EXPECT_EQ(result.decision, CollectiveDecision::fail);
  EXPECT_EQ(result.representative.code, collective_protocol_error);
  EXPECT_NE(result.representative.message_text().find("phase"),
            std::string_view::npos);
}

TEST(CollectiveError, MissingOrDuplicateRankBecomesProtocolFailure) {
  const std::array records{
      CollectiveErrorRecord::success(3, 0, "halo"),
      CollectiveErrorRecord::success(3, 0, "halo"),
  };
  const auto result = resolve_collective_errors(records);
  EXPECT_EQ(result.decision, CollectiveDecision::fail);
  EXPECT_EQ(result.representative.code, collective_protocol_error);

  const auto empty = resolve_collective_errors({});
  EXPECT_EQ(empty.decision, CollectiveDecision::fail);
  EXPECT_EQ(empty.participant_count, 0u);
}

TEST(CollectiveError, OutOfRangeRankBecomesSpecificProtocolFailure) {
  const std::array records{
      CollectiveErrorRecord::success(3, 0, "halo"),
      CollectiveErrorRecord::success(3, 2, "halo"),
  };
  const auto result = resolve_collective_errors(records);
  EXPECT_EQ(result.decision, CollectiveDecision::fail);
  EXPECT_EQ(result.representative.code, collective_protocol_error);
  EXPECT_NE(result.representative.message_text().find("out-of-range"),
            std::string_view::npos);
}

TEST(CollectiveError, InvalidWireDispositionBecomesProtocolFailure) {
  std::array records{
      CollectiveErrorRecord::success(3, 0, "halo"),
  };
  records[0].disposition = static_cast<ErrorDisposition>(255);
  const auto result = resolve_collective_errors(records);
  EXPECT_EQ(result.decision, CollectiveDecision::fail);
  EXPECT_EQ(result.representative.code, collective_protocol_error);
}

TEST(CollectiveError, FixedTextIsTerminatedAndTruncated) {
  const std::string long_message(collective_message_capacity + 50, 'x');
  const auto record = CollectiveErrorRecord::failure(
      1, 0, -1, 5, "checkpoint", long_message);
  EXPECT_EQ(record.message_text().size(), collective_message_capacity - 1);
  EXPECT_EQ(record.message.back(), '\0');
}

TEST(CollectiveError, DistributedExceptionConstructionIsNoexceptAndStable) {
  static_assert(std::is_nothrow_constructible_v<
                quasar::distributed::DistributedCollectiveError,
                quasar::distributed::CollectiveResolution>);
  const std::array records{
      CollectiveErrorRecord::failure(
          17, 0, 3, 29, "transport-startup", "injected startup failure"),
      CollectiveErrorRecord::success(17, 1, "transport-startup"),
  };
  const auto resolution = resolve_collective_errors(records);
  const quasar::distributed::DistributedCollectiveError error{resolution};
  const std::string_view message{error.what()};
  EXPECT_NE(message.find("transport-startup"), std::string_view::npos);
  EXPECT_NE(message.find("injected startup failure"), std::string_view::npos);
  EXPECT_EQ(error.resolution().representative.rank, 0);
  EXPECT_EQ(error.resolution().representative.endpoint, 3);
}
