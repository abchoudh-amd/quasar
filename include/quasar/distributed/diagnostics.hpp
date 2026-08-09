#pragma once

#include "quasar/distributed/mpi_runtime.hpp"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace quasar::distributed {

inline constexpr const char* diagnostics_shard_schema =
    "quasar-diagnostics-shards/v1";

struct DiagnosticShard {
  int rank{-1};
  int node_rank{-1};
  std::uint64_t local_device{0};
  std::uint64_t endpoint{0};
  std::string device_identity{};
  std::uint64_t tile_x{0};
  std::uint64_t tile_y{0};
  std::uint64_t offset_x{0};
  std::uint64_t offset_y{0};
  std::uint64_t owned_nx{0};
  std::uint64_t owned_ny{0};
  std::filesystem::path path{};
};

struct ShardedDiagnosticsManifest {
  std::string schema{diagnostics_shard_schema};
  std::string physics{};
  std::string geometry{};
  std::uint64_t global_nx{0};
  std::uint64_t global_ny{0};
  std::uint64_t step{0};
  double time{0.0};
  std::uint64_t px{0};
  std::uint64_t py{0};
  std::vector<DiagnosticShard> shards{};
};

void validate_diagnostics_manifest(const ShardedDiagnosticsManifest& manifest);
[[nodiscard]] std::string diagnostics_manifest_json(
    const ShardedDiagnosticsManifest& manifest);

// Gathers each rank's local shard descriptions, validates global coverage on
// rank zero, and atomically publishes the JSON manifest only after every rank
// confirms its shard files are complete.  The manifest is therefore the run's
// completion marker.
void publish_diagnostics_manifest(
    MpiRuntime& runtime,
    std::filesystem::path manifest_path,
    ShardedDiagnosticsManifest common,
    std::span<const DiagnosticShard> local_shards);

}  // namespace quasar::distributed
