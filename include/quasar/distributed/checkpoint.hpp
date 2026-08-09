#pragma once

#include "quasar/distributed/mpi_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace quasar::distributed {

inline constexpr std::string_view checkpoint_schema = "quasar-checkpoint/v1";
inline constexpr std::string_view checkpoint_diagnostic_state_schema =
    "quasar-checkpoint-diagnostics/v1";
inline constexpr std::uint64_t max_checkpoint_diagnostic_state_bytes =
    512ULL * 1024ULL * 1024ULL;

enum class CheckpointValueType {
  float32,
  float64,
  uint8,
  uint64,
  int32,
};

struct CheckpointMetadata {
  std::string schema{checkpoint_schema};
  std::string physics{};
  std::string precision{};
  std::string geometry{};
  std::string unit_system{};
  std::uint64_t global_nx{0};
  std::uint64_t global_ny{0};
  std::string boundary_signature{};
  std::string species_signature{};
  std::string background_signature{};
  std::string numerics_signature{};
  std::uint64_t step{0};
  double time{0.0};
  // Size of the checksummed, versioned rank-fragment envelope stored in
  // diagnostics/state.  This is restart state, not a physics compatibility
  // axis: diagnostics placement and policy may change across restart.
  std::uint64_t diagnostic_state_bytes{0};
};

// One half-open file-space rectangle.  Dataset calls concatenate local slabs
// in lexicographic file-offset order into one packed memory buffer.
struct DatasetHyperslab {
  std::vector<std::uint64_t> offset{};
  std::vector<std::uint64_t> count{};
};

void validate_checkpoint_metadata(const CheckpointMetadata& metadata);
void validate_restart_compatibility(const CheckpointMetadata& stored,
                                    const CheckpointMetadata& requested);

// Build one topology-independent envelope from rank-local opaque fragments.
// Empty fragments are allowed.  The result is identical on every rank and is
// bounded by max_checkpoint_diagnostic_state_bytes.  The envelope itself is
// versioned and checksummed; callers remain responsible for validating the
// safe, non-pickle application format carried by each fragment.
[[nodiscard]] std::vector<std::uint8_t>
collect_checkpoint_diagnostic_state(
    MpiRuntime& runtime, std::span<const std::uint8_t> local_fragment);

// Validate and split an envelope without invoking MPI or HDF5.  Primarily
// useful to test corruption handling independently of a checkpoint file.
[[nodiscard]] std::vector<std::vector<std::uint8_t>>
decode_checkpoint_diagnostic_state(
    std::span<const std::uint8_t> encoded_state);

class ParallelCheckpointWriter {
 public:
  ParallelCheckpointWriter(MpiRuntime& runtime,
                           std::filesystem::path committed_path,
                           CheckpointMetadata metadata);
  ~ParallelCheckpointWriter() noexcept;
  ParallelCheckpointWriter(const ParallelCheckpointWriter&) = delete;
  ParallelCheckpointWriter& operator=(const ParallelCheckpointWriter&) = delete;
  ParallelCheckpointWriter(ParallelCheckpointWriter&&) = delete;
  ParallelCheckpointWriter& operator=(ParallelCheckpointWriter&&) = delete;

  [[nodiscard]] const std::filesystem::path& committed_path() const noexcept;
  [[nodiscard]] const std::filesystem::path& temporary_path() const noexcept;
  [[nodiscard]] bool committed() const noexcept;

  // Every rank calls this once per dataset and in the same order.  Ranks with
  // no owned elements pass empty slabs, a null buffer, and zero elements.
  void write_dataset(std::string_view name,
                     CheckpointValueType value_type,
                     std::span<const std::uint64_t> global_shape,
                     std::span<const DatasetHyperslab> local_slabs,
                     const void* packed_values,
                     std::size_t packed_elements);

  // Store an envelope returned by collect_checkpoint_diagnostic_state().
  // Rank zero owns the sole file hyperslab; every rank still participates in
  // the collective write and must supply the identical validated envelope.
  void write_diagnostic_state(
      std::span<const std::uint8_t> encoded_state);

  // Flushes and closes collectively, then rank zero atomically replaces the
  // committed path and broadcasts the result.  A failed commit never removes
  // or truncates the preceding committed file.
  void commit();

  // Collectively closes an uncommitted temporary file and removes it on rank
  // zero.  Destruction itself performs no HDF5, MPI, rename, or removal calls.
  void close();

 private:
  struct Impl;
  Impl* implementation_{nullptr};
};

class ParallelCheckpointReader {
 public:
  ParallelCheckpointReader(MpiRuntime& runtime,
                           std::filesystem::path path);
  ~ParallelCheckpointReader() noexcept;
  ParallelCheckpointReader(const ParallelCheckpointReader&) = delete;
  ParallelCheckpointReader& operator=(const ParallelCheckpointReader&) = delete;
  ParallelCheckpointReader(ParallelCheckpointReader&&) = delete;
  ParallelCheckpointReader& operator=(ParallelCheckpointReader&&) = delete;

  [[nodiscard]] const CheckpointMetadata& metadata() const noexcept;

  void read_dataset(std::string_view name,
                    CheckpointValueType value_type,
                    std::span<const std::uint64_t> expected_global_shape,
                    std::span<const DatasetHyperslab> local_slabs,
                    void* packed_values,
                    std::size_t packed_elements);

  // Read, broadcast, validate, and split the topology-independent diagnostic
  // envelope.  A metadata size of zero represents a checkpoint written by a
  // lower-level caller that supplied no diagnostic continuation state.
  [[nodiscard]] std::vector<std::vector<std::uint8_t>>
  read_diagnostic_state();

  void close();

 private:
  struct Impl;
  Impl* implementation_{nullptr};
};

}  // namespace quasar::distributed
