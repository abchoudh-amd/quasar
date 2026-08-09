#include "quasar/distributed/diagnostics.hpp"

#include <gtest/gtest.h>

#include <string>

namespace {

quasar::distributed::ShardedDiagnosticsManifest manifest() {
  using quasar::distributed::DiagnosticShard;
  return {
      .schema = "quasar-diagnostics-shards/v1",
      .physics = "mhd",
      .geometry = "cartesian",
      .global_nx = 9,
      .global_ny = 7,
      .step = 12,
      .time = 0.125,
      .px = 2,
      .py = 2,
      .shards = {
          DiagnosticShard{0, 0, 0, 0, "uuid:a", 0, 0, 0, 0, 5, 4,
                          "out.rank0.gpu0.npz"},
          DiagnosticShard{0, 0, 1, 1, "uuid:b", 1, 0, 5, 0, 4, 4,
                          "out.rank0.gpu1.npz"},
          DiagnosticShard{1, 0, 0, 2, "uuid:c", 0, 1, 0, 4, 5, 3,
                          "out.rank1.gpu0.npz"},
          DiagnosticShard{1, 0, 1, 3, "uuid:d", 1, 1, 5, 4, 4, 3,
                          "out.rank1.gpu1.npz"},
      },
  };
}

}  // namespace

TEST(DistributedDiagnosticsManifest, EmitsVersionedOwnedExtentMapping) {
  const std::string document =
      quasar::distributed::diagnostics_manifest_json(manifest());
  EXPECT_NE(document.find("\"schema\": \"quasar-diagnostics-shards/v1\""),
            std::string::npos);
  EXPECT_NE(document.find("\"global_shape\": [7, 9]"), std::string::npos);
  EXPECT_NE(document.find("\"endpoint\": 3"), std::string::npos);
  EXPECT_NE(document.find("\"offset\": [4, 5]"), std::string::npos);
  EXPECT_NE(document.find("\"owned_shape\": [3, 4]"),
            std::string::npos);
}

TEST(DistributedDiagnosticsManifest, AcceptsThirdPhysicsName) {
  auto extensible = manifest();
  extensible.physics = "radiation";

  EXPECT_NO_THROW(
      quasar::distributed::validate_diagnostics_manifest(extensible));
  const std::string document =
      quasar::distributed::diagnostics_manifest_json(extensible);
  EXPECT_NE(document.find("\"physics\": \"radiation\""),
            std::string::npos);
}

TEST(DistributedDiagnosticsManifest, RejectsOverlapAndIncompleteCoverage) {
  auto overlapping = manifest();
  overlapping.shards[3].offset_x = 4;
  EXPECT_THROW(
      quasar::distributed::validate_diagnostics_manifest(overlapping),
      std::invalid_argument);

  auto incomplete = manifest();
  incomplete.shards[3].owned_nx = 3;
  EXPECT_THROW(
      quasar::distributed::validate_diagnostics_manifest(incomplete),
      std::invalid_argument);
}

TEST(DistributedDiagnosticsManifest, RejectsCheckpointSchema) {
  auto invalid = manifest();
  invalid.schema = "quasar-checkpoint/v1";
  EXPECT_THROW(quasar::distributed::validate_diagnostics_manifest(invalid),
               std::invalid_argument);
}

TEST(DistributedDiagnosticsManifest, RejectsEmptyPhysicsName) {
  auto invalid = manifest();
  invalid.physics.clear();
  EXPECT_THROW(quasar::distributed::validate_diagnostics_manifest(invalid),
               std::invalid_argument);
}

TEST(DistributedDiagnosticsManifest, RejectsNormalizedShardPathAliases) {
  auto invalid = manifest();
  invalid.shards[1].path = "nested/../out.rank0.gpu0.npz";
  EXPECT_THROW(quasar::distributed::validate_diagnostics_manifest(invalid),
               std::invalid_argument);
}

TEST(DistributedDiagnosticsManifest, RejectsNonUtf8JsonFields) {
  auto invalid = manifest();
  invalid.shards[0].path = std::string{"invalid-"} + static_cast<char>(0xff)
      + ".npz";
  EXPECT_THROW(quasar::distributed::validate_diagnostics_manifest(invalid),
               std::invalid_argument);
}
