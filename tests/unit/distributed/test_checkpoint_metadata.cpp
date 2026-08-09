#include "quasar/distributed/checkpoint.hpp"

#include <gtest/gtest.h>

namespace {

quasar::distributed::CheckpointMetadata metadata() {
  return {
      .schema = "quasar-checkpoint/v1",
      .physics = "pic",
      .precision = "float64",
      .geometry = "cartesian",
      .unit_system = "normalized",
      .global_nx = 64,
      .global_ny = 48,
      .boundary_signature = "periodic,periodic,pec,pec",
      .species_signature = "electron:-1:1;ion:1:1836",
      .background_signature = "neutralizing=true",
      .numerics_signature = "yee=4;shape=tsc;filter=binomial:2",
      .step = 120,
      .time = 0.75,
  };
}

}  // namespace

TEST(DistributedCheckpointMetadata,
     RestartMayChangeStepAndTerminationWithoutChangingPhysics) {
  auto stored = metadata();
  auto requested = metadata();
  requested.step = 0;
  requested.time = 9.0;
  EXPECT_NO_THROW(
      quasar::distributed::validate_restart_compatibility(stored, requested));
}

TEST(DistributedCheckpointMetadata, RejectsEveryPhysicsCompatibilityAxis) {
  const auto stored = metadata();
  const auto expect_rejected = [&](auto mutation) {
    auto requested = metadata();
    mutation(requested);
    EXPECT_THROW(
        quasar::distributed::validate_restart_compatibility(stored, requested),
        std::invalid_argument);
  };

  expect_rejected([](auto& value) { value.physics = "mhd"; });
  expect_rejected([](auto& value) { value.precision = "float32"; });
  expect_rejected([](auto& value) { value.geometry = "cylindrical"; });
  expect_rejected([](auto& value) { value.unit_system = "SI"; });
  expect_rejected([](auto& value) { value.global_nx = 65; });
  expect_rejected([](auto& value) { value.global_ny = 49; });
  expect_rejected([](auto& value) { value.boundary_signature = "outflow"; });
  expect_rejected([](auto& value) { value.species_signature = "electron"; });
  expect_rejected([](auto& value) { value.background_signature = "none"; });
  expect_rejected([](auto& value) { value.numerics_signature = "yee=2"; });
}

TEST(DistributedCheckpointMetadata, RejectsDiagnosticsNpzSchema) {
  auto value = metadata();
  value.schema = "quasar-diagnostics-shards/v1";
  EXPECT_THROW(quasar::distributed::validate_checkpoint_metadata(value),
               std::invalid_argument);
}

TEST(DistributedCheckpointMetadata, RejectsEmbeddedNullText) {
  auto value = metadata();
  value.boundary_signature = std::string{"periodic\0outflow", 16};
  EXPECT_THROW(quasar::distributed::validate_checkpoint_metadata(value),
               std::invalid_argument);
}
