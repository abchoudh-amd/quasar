#include "quasar/distributed/checkpoint.hpp"
#include "quasar/distributed/diagnostics.hpp"

#include <hdf5.h>
#include <mpi.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::string broadcast_string(quasar::distributed::MpiRuntime& runtime,
                             std::string value) {
  std::uint64_t size = value.size();
  quasar::distributed::check_mpi(
      MPI_Bcast(&size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD),
      "MPI_Bcast(test path size)");
  if (size > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::length_error{"test path exceeds MPI count range"};
  }
  if (runtime.rank() != 0) value.resize(static_cast<std::size_t>(size));
  quasar::distributed::check_mpi(
      MPI_Bcast(value.data(), static_cast<int>(size), MPI_CHAR, 0,
                MPI_COMM_WORLD),
      "MPI_Bcast(test path)");
  return value;
}

std::filesystem::path shared_test_stem(
    quasar::distributed::MpiRuntime& runtime) {
  std::string path;
  if (runtime.rank() == 0) {
    const auto nonce = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    path = (std::filesystem::temp_directory_path()
            / ("quasar-distributed-io-" + std::to_string(nonce)))
               .string();
  }
  return broadcast_string(runtime, std::move(path));
}

quasar::distributed::CheckpointMetadata metadata() {
  return {
      .schema = "quasar-checkpoint/v1",
      .physics = "mhd",
      .precision = "float64",
      .geometry = "cartesian",
      .unit_system = "normalized",
      .global_nx = 8,
      .global_ny = 6,
      .boundary_signature = "periodic,periodic",
      .species_signature = "",
      .background_signature = "none",
      .numerics_signature = "muscl,ct",
      .step = 7,
      .time = 0.25,
  };
}

bool corrupt_schema_to_variable_length(const std::filesystem::path& path) {
  hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  if (file < 0) return false;
  bool success = H5Adelete(file, "schema") >= 0;
  hid_t type = success ? H5Tcopy(H5T_C_S1) : -1;
  if (type >= 0) success = H5Tset_size(type, H5T_VARIABLE) >= 0;
  hid_t space = success ? H5Screate(H5S_SCALAR) : -1;
  hid_t attribute = success
      ? H5Acreate2(file, "schema", type, space, H5P_DEFAULT, H5P_DEFAULT)
      : -1;
  success = success && attribute >= 0;
  const char* value = "quasar-checkpoint/v1";
  if (success) success = H5Awrite(attribute, type, &value) >= 0;
  if (attribute >= 0) success = H5Aclose(attribute) >= 0 && success;
  if (space >= 0) success = H5Sclose(space) >= 0 && success;
  if (type >= 0) success = H5Tclose(type) >= 0 && success;
  success = H5Fclose(file) >= 0 && success;
  return success;
}

bool corrupt_schema_to_integer(const std::filesystem::path& path) {
  hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  if (file < 0) return false;
  bool success = H5Adelete(file, "schema") >= 0;
  hid_t space = success ? H5Screate(H5S_SCALAR) : -1;
  hid_t attribute = success
      ? H5Acreate2(file, "schema", H5T_STD_U64LE, space, H5P_DEFAULT,
                   H5P_DEFAULT)
      : -1;
  success = success && attribute >= 0;
  const std::uint64_t value = 1;
  if (success) {
    success = H5Awrite(attribute, H5T_NATIVE_UINT64, &value) >= 0;
  }
  if (attribute >= 0) success = H5Aclose(attribute) >= 0 && success;
  if (space >= 0) success = H5Sclose(space) >= 0 && success;
  success = H5Fclose(file) >= 0 && success;
  return success;
}

template <class Function>
bool collectively_rejected(Function&& function) {
  try {
    function();
  } catch (const quasar::distributed::DistributedCollectiveError&) {
    return true;
  }
  return false;
}

void overwrite_u64_le(std::vector<std::uint8_t>& bytes, std::size_t offset,
                      std::uint64_t value) {
  if (offset + sizeof(value) > bytes.size()) {
    throw std::out_of_range{"test envelope field lies outside the buffer"};
  }
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    bytes[offset++] = static_cast<std::uint8_t>((value >> shift) & 0xffU);
  }
}

std::uint64_t read_u64_le(const std::vector<std::uint8_t>& bytes,
                          std::size_t offset) {
  if (offset + sizeof(std::uint64_t) > bytes.size()) {
    throw std::out_of_range{"test envelope field lies outside the buffer"};
  }
  std::uint64_t value = 0;
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    value |= static_cast<std::uint64_t>(bytes[offset++]) << shift;
  }
  return value;
}

bool diagnostic_envelope_rejected(const std::vector<std::uint8_t>& bytes) {
  try {
    (void)quasar::distributed::decode_checkpoint_diagnostic_state(bytes);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
bool configure_test_failure(quasar::distributed::MpiRuntime& runtime,
                            const char* name, bool enabled) {
  const bool local_success = enabled
      ? ::setenv(name, "1", 1) == 0
      : ::unsetenv(name) == 0;
  return runtime.allreduce_all(local_success);
}
#endif

}  // namespace

int main(int argc, char** argv) {
  try {
    quasar::distributed::MpiRuntime runtime{&argc, &argv};
    if (runtime.size() < 2) {
      if (runtime.rank() == 0) {
        std::cerr << "test_distributed_io requires at least two MPI ranks\n";
      }
      runtime.close();
      return 77;
    }

    const std::filesystem::path stem = shared_test_stem(runtime);
    const std::filesystem::path checkpoint = stem.string() + ".h5";
    const std::array<std::uint64_t, 2> shape{
        static_cast<std::uint64_t>(runtime.size() * 2), 3};
    const quasar::distributed::DatasetHyperslab owned{
        .offset = {static_cast<std::uint64_t>(runtime.rank() * 2), 0},
        .count = {2, 3},
    };
    std::vector<std::uint64_t> values(6);
    for (std::size_t index = 0; index < values.size(); ++index) {
      values[index] = static_cast<std::uint64_t>(runtime.rank() * 100)
          + static_cast<std::uint64_t>(index);
    }
    const std::string diagnostic_text =
        "rank-" + std::to_string(runtime.rank()) + "-history";
    const std::vector<std::uint8_t> diagnostic_fragment(
        diagnostic_text.begin(), diagnostic_text.end());
    const std::vector<std::uint8_t> encoded_diagnostics =
        quasar::distributed::collect_checkpoint_diagnostic_state(
            runtime, diagnostic_fragment);

    const auto decoded_diagnostics =
        quasar::distributed::decode_checkpoint_diagnostic_state(
            encoded_diagnostics);
    if (decoded_diagnostics.size()
        != static_cast<std::size_t>(runtime.size())) return 27;
    for (int rank = 0; rank < runtime.size(); ++rank) {
      const std::string expected =
          "rank-" + std::to_string(rank) + "-history";
      if (decoded_diagnostics[static_cast<std::size_t>(rank)]
          != std::vector<std::uint8_t>{expected.begin(), expected.end()}) {
        return 28;
      }
    }
    if (!diagnostic_envelope_rejected({})) return 29;
    {
      auto malformed_envelope = encoded_diagnostics;
      malformed_envelope.front() ^= 0xffU;
      if (!diagnostic_envelope_rejected(malformed_envelope)) return 30;
    }
    {
      auto malformed_envelope = encoded_diagnostics;
      overwrite_u64_le(malformed_envelope, 16, 2);  // version
      if (!diagnostic_envelope_rejected(malformed_envelope)) return 31;
    }
    {
      auto malformed_envelope = encoded_diagnostics;
      overwrite_u64_le(malformed_envelope, 24, 0);  // part count
      if (!diagnostic_envelope_rejected(malformed_envelope)) return 32;
    }
    {
      auto malformed_envelope = encoded_diagnostics;
      overwrite_u64_le(malformed_envelope, 24, 1U << 20);  // truncated table
      if (!diagnostic_envelope_rejected(malformed_envelope)) return 33;
    }
    {
      auto malformed_envelope = encoded_diagnostics;
      overwrite_u64_le(malformed_envelope, 48,
                       std::numeric_limits<std::uint64_t>::max());
      if (!diagnostic_envelope_rejected(malformed_envelope)) return 34;
    }
    {
      auto malformed_envelope = encoded_diagnostics;
      overwrite_u64_le(
          malformed_envelope, 32,
          read_u64_le(malformed_envelope, 32) + 1);  // payload length
      if (!diagnostic_envelope_rejected(malformed_envelope)) return 35;
    }
    {
      auto malformed_envelope = encoded_diagnostics;
      malformed_envelope.back() ^= 0xffU;  // checksum
      if (!diagnostic_envelope_rejected(malformed_envelope)) return 36;
    }

    {
      auto checkpoint_metadata = metadata();
      checkpoint_metadata.diagnostic_state_bytes =
          encoded_diagnostics.size();
      quasar::distributed::ParallelCheckpointWriter writer{
          runtime, checkpoint, checkpoint_metadata};
      writer.write_dataset("mhd/state", quasar::distributed::CheckpointValueType::uint64,
                           shape, std::span{&owned, 1}, values.data(),
                           values.size());
      writer.write_diagnostic_state(encoded_diagnostics);
      writer.commit();
    }
    {
      std::vector<std::uint64_t> restored(values.size());
      quasar::distributed::ParallelCheckpointReader reader{runtime, checkpoint};
      reader.read_dataset("mhd/state", quasar::distributed::CheckpointValueType::uint64,
                          shape, std::span{&owned, 1}, restored.data(),
                          restored.size());
      if (restored != values || reader.metadata().step != 7) return 1;
      const auto diagnostic_parts = reader.read_diagnostic_state();
      if (diagnostic_parts.size()
          != static_cast<std::size_t>(runtime.size())) return 10;
      for (int rank = 0; rank < runtime.size(); ++rank) {
        const std::string expected =
            "rank-" + std::to_string(rank) + "-history";
        if (diagnostic_parts[static_cast<std::size_t>(rank)]
            != std::vector<std::uint8_t>{expected.begin(), expected.end()}) {
          return 11;
        }
      }
      reader.close();
    }

#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
    const std::filesystem::path path_failure_checkpoint =
        stem.string() + ".path-failure.h5";
    if (!configure_test_failure(
            runtime, "QUASAR_TEST_CHECKPOINT_PATH_PREPARATION_FAILURE", true)) {
      return 13;
    }
    const bool path_preparation_rejected = collectively_rejected([&] {
      quasar::distributed::ParallelCheckpointWriter writer{
          runtime, path_failure_checkpoint, metadata()};
    });
    if (!configure_test_failure(
            runtime, "QUASAR_TEST_CHECKPOINT_PATH_PREPARATION_FAILURE", false)) {
      return 14;
    }
    if (!runtime.allreduce_all(path_preparation_rejected)) return 15;

    {
      auto replacement_metadata = metadata();
      replacement_metadata.step = 8;
      quasar::distributed::ParallelCheckpointWriter writer{
          runtime, checkpoint, replacement_metadata};
      std::vector<std::uint64_t> replacement(values.size());
      for (std::size_t index = 0; index < replacement.size(); ++index) {
        replacement[index] = values[index] + 1000;
      }
      writer.write_dataset(
          "mhd/state", quasar::distributed::CheckpointValueType::uint64,
          shape, std::span{&owned, 1}, replacement.data(), replacement.size());
      if (!configure_test_failure(
              runtime, "QUASAR_TEST_CHECKPOINT_RENAME_FAILURE", true)) {
        return 16;
      }
      const bool rename_rejected = collectively_rejected([&] {
        writer.commit();
      });
      if (!configure_test_failure(
              runtime, "QUASAR_TEST_CHECKPOINT_RENAME_FAILURE", false)) {
        return 17;
      }
      const bool replacement_state_valid = rename_rejected
          && !writer.committed();
      if (!runtime.allreduce_all(replacement_state_valid)) return 18;
      writer.close();
    }
    {
      std::vector<std::uint64_t> restored(values.size());
      quasar::distributed::ParallelCheckpointReader reader{runtime, checkpoint};
      reader.read_dataset(
          "mhd/state", quasar::distributed::CheckpointValueType::uint64,
          shape, std::span{&owned, 1}, restored.data(), restored.size());
      const bool original_preserved = restored == values
          && reader.metadata().step == 7
          && reader.metadata().diagnostic_state_bytes
                 == encoded_diagnostics.size();
      if (!runtime.allreduce_all(original_preserved)) return 19;
      reader.close();
    }
#endif

    const std::filesystem::path gap_path = stem.string() + ".gap.h5";
    {
      quasar::distributed::ParallelCheckpointWriter writer{
          runtime, gap_path, metadata()};
      auto gap = owned;
      if (runtime.rank() == 0) gap.count[0] = 1;
      const std::size_t elements =
          static_cast<std::size_t>(gap.count[0] * gap.count[1]);
      const bool rejected = collectively_rejected([&] {
        writer.write_dataset("gap", quasar::distributed::CheckpointValueType::uint64,
                             shape, std::span{&gap, 1}, values.data(), elements);
      });
      if (!runtime.allreduce_all(rejected)) return 2;
      writer.close();
    }

    const std::filesystem::path overlap_path = stem.string() + ".overlap.h5";
    {
      quasar::distributed::ParallelCheckpointWriter writer{
          runtime, overlap_path, metadata()};
      auto overlap = owned;
      if (runtime.rank() == 1) --overlap.offset[0];
      const bool rejected = collectively_rejected([&] {
        writer.write_dataset("overlap", quasar::distributed::CheckpointValueType::uint64,
                             shape, std::span{&overlap, 1}, values.data(),
                             values.size());
      });
      if (!runtime.allreduce_all(rejected)) return 3;
      writer.close();
    }

    const std::filesystem::path malformed = stem.string() + ".vlen.h5";
    {
      quasar::distributed::ParallelCheckpointWriter writer{
          runtime, malformed, metadata()};
      writer.commit();
    }
    bool corrupted = true;
    if (runtime.rank() == 0) {
      corrupted = corrupt_schema_to_variable_length(malformed);
    }
    if (!runtime.allreduce_all(corrupted)) return 4;
    const bool malformed_rejected = collectively_rejected([&] {
      quasar::distributed::ParallelCheckpointReader reader{runtime, malformed};
    });
    if (!runtime.allreduce_all(malformed_rejected)) return 5;
    if (runtime.rank() == 0) {
      corrupted = corrupt_schema_to_integer(malformed);
    }
    if (!runtime.allreduce_all(corrupted)) return 6;
    const bool wrong_type_rejected = collectively_rejected([&] {
      quasar::distributed::ParallelCheckpointReader reader{runtime, malformed};
    });
    if (!runtime.allreduce_all(wrong_type_rejected)) return 7;

    const std::filesystem::path manifest_path = stem.string() + ".manifest.json";
    const std::filesystem::path shard_path = runtime.rank() == 0
        ? manifest_path
        : std::filesystem::path{stem.string() + ".rank"
                                + std::to_string(runtime.rank()) + ".npz"};
    {
      std::ofstream shard{shard_path, std::ios::binary | std::ios::trunc};
      shard << "complete";
      shard.close();
      if (!shard) return 8;
    }
    quasar::distributed::ShardedDiagnosticsManifest manifest{
        .schema = "quasar-diagnostics-shards/v1",
        .physics = "mhd",
        .geometry = "cartesian",
        .global_nx = static_cast<std::uint64_t>(runtime.size()),
        .global_ny = 1,
        .step = 7,
        .time = 0.25,
        .px = static_cast<std::uint64_t>(runtime.size()),
        .py = 1,
    };
    const quasar::distributed::DiagnosticShard shard{
        .rank = runtime.rank(),
        .node_rank = runtime.node_rank(),
        .local_device = 0,
        .endpoint = static_cast<std::uint64_t>(runtime.rank()),
        .device_identity = "test-device-" + std::to_string(runtime.rank()),
        .tile_x = static_cast<std::uint64_t>(runtime.rank()),
        .tile_y = 0,
        .offset_x = static_cast<std::uint64_t>(runtime.rank()),
        .offset_y = 0,
        .owned_nx = 1,
        .owned_ny = 1,
        .path = shard_path,
    };
    const bool collision_rejected = collectively_rejected([&] {
      quasar::distributed::publish_diagnostics_manifest(
          runtime, manifest_path, manifest, std::span{&shard, 1});
    });
    if (!runtime.allreduce_all(collision_rejected)) return 9;

    std::error_code ignored;
    std::filesystem::remove(shard_path, ignored);
    const std::filesystem::path good_manifest =
        stem.string() + ".good.manifest.json";
    const std::filesystem::path good_shard_path =
        stem.string() + ".good.rank" + std::to_string(runtime.rank()) + ".npz";
    {
      std::ofstream good_shard{good_shard_path,
                               std::ios::binary | std::ios::trunc};
      good_shard << "complete";
      good_shard.close();
      if (!good_shard) return 10;
    }
    auto good_shard = shard;
    good_shard.path = good_shard_path;

#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
    const std::filesystem::path preparation_failure_manifest =
        stem.string() + ".preparation-failure.manifest.json";
    if (!configure_test_failure(
            runtime,
            "QUASAR_TEST_DIAGNOSTICS_MANIFEST_PREPARATION_FAILURE", true)) {
      return 20;
    }
    const bool preparation_rejected = collectively_rejected([&] {
      quasar::distributed::publish_diagnostics_manifest(
          runtime, preparation_failure_manifest, manifest,
          std::span{&good_shard, 1});
    });
    if (!configure_test_failure(
            runtime,
            "QUASAR_TEST_DIAGNOSTICS_MANIFEST_PREPARATION_FAILURE", false)) {
      return 21;
    }
    bool preparation_manifest_absent = preparation_rejected;
    if (runtime.rank() == 0) {
      std::error_code state_error;
      preparation_manifest_absent = preparation_manifest_absent
          && !std::filesystem::exists(preparation_failure_manifest,
                                      state_error)
          && !state_error;
    }
    if (!runtime.allreduce_all(preparation_manifest_absent)) return 22;

    const std::filesystem::path publication_failure_manifest =
        stem.string() + ".publication-failure.manifest.json";
    if (!configure_test_failure(
            runtime, "QUASAR_TEST_DIAGNOSTICS_PUBLICATION_FAILURE", true)) {
      return 23;
    }
    const bool publication_rejected = collectively_rejected([&] {
      quasar::distributed::publish_diagnostics_manifest(
          runtime, publication_failure_manifest, manifest,
          std::span{&good_shard, 1});
    });
    if (!configure_test_failure(
            runtime, "QUASAR_TEST_DIAGNOSTICS_PUBLICATION_FAILURE", false)) {
      return 24;
    }
    bool publication_manifest_absent = publication_rejected;
    if (runtime.rank() == 0) {
      std::error_code state_error;
      publication_manifest_absent = publication_manifest_absent
          && !std::filesystem::exists(publication_failure_manifest,
                                      state_error)
          && !state_error;
    }
    if (!runtime.allreduce_all(publication_manifest_absent)) return 25;

    quasar::distributed::publish_diagnostics_manifest(
        runtime, publication_failure_manifest, manifest,
        std::span{&good_shard, 1});
    bool publication_retry_complete = true;
    if (runtime.rank() == 0) {
      std::error_code state_error;
      publication_retry_complete = std::filesystem::is_regular_file(
          publication_failure_manifest, state_error) && !state_error;
    }
    if (!runtime.allreduce_all(publication_retry_complete)) return 26;
#endif

    quasar::distributed::publish_diagnostics_manifest(
        runtime, good_manifest, manifest, std::span{&good_shard, 1});
    bool manifest_complete = true;
    if (runtime.rank() == 0) {
      std::ifstream published{good_manifest, std::ios::binary};
      const std::string document{
          std::istreambuf_iterator<char>{published},
          std::istreambuf_iterator<char>{}};
      manifest_complete = published.is_open()
          && document.find("quasar-diagnostics-shards/v1")
              != std::string::npos;
    }
    if (!runtime.allreduce_all(manifest_complete)) return 11;
    std::filesystem::remove(good_shard_path, ignored);
    runtime.barrier();
    if (runtime.rank() == 0) {
      std::filesystem::remove(checkpoint, ignored);
      std::filesystem::remove(malformed, ignored);
      std::filesystem::remove(manifest_path, ignored);
      std::filesystem::remove(good_manifest, ignored);
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
      std::filesystem::remove(preparation_failure_manifest, ignored);
      std::filesystem::remove(publication_failure_manifest, ignored);
#endif
    }
    runtime.close();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 12;
  }
}
