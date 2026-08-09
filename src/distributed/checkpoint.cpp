#include "quasar/distributed/checkpoint.hpp"

#include "collective_helpers.hpp"
#include "test_hooks.hpp"

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#  include <unistd.h>
#endif

namespace quasar::distributed {
namespace {

constexpr CollectiveStringBroadcastMessages checkpoint_string_broadcast{
    .preparation_failure =
        "collective checkpoint string preparation failed",
    .range_failure = "collective checkpoint string exceeds MPI count range",
    .allocation_failure =
        "collective checkpoint string allocation failed",
    .length_operation = "MPI_Bcast(string length)",
    .bytes_operation = "MPI_Bcast(string bytes)",
    .preparation_code = -46,
    .range_code = -47,
    .allocation_code = -39,
};

constexpr std::array<std::uint8_t, 16> diagnostic_state_magic{
    'Q', 'U', 'A', 'S', 'A', 'R', '-', 'D',
    'I', 'A', 'G', '-', 'V', '1', '\r', '\n'};
constexpr std::size_t diagnostic_state_header_bytes =
    diagnostic_state_magic.size() + 4 * sizeof(std::uint64_t);

void append_u64_le(std::vector<std::uint8_t>& destination,
                   std::uint64_t value) {
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    destination.push_back(
        static_cast<std::uint8_t>((value >> shift) & 0xffU));
  }
}

std::uint64_t read_u64_le(std::span<const std::uint8_t> source,
                          std::size_t& cursor) {
  if (source.size() - std::min(source.size(), cursor) < sizeof(std::uint64_t)) {
    throw std::invalid_argument{
        "checkpoint diagnostic-state envelope is truncated"};
  }
  std::uint64_t value = 0;
  for (unsigned int shift = 0; shift < 64; shift += 8) {
    value |= static_cast<std::uint64_t>(source[cursor++]) << shift;
  }
  return value;
}

std::uint64_t diagnostic_state_hash(
    std::span<const std::uint8_t> bytes) noexcept {
  // FNV-1a is not an authenticity primitive; it is a deterministic corruption
  // check layered over HDF5 and the application-level NPZ/JSON validation.
  std::uint64_t hash = 14695981039346656037ULL;
  for (const std::uint8_t byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::vector<std::uint8_t> encode_diagnostic_state(
    std::span<const std::uint64_t> sizes,
    std::span<const std::uint8_t> payload) {
  if (sizes.empty()) {
    throw std::invalid_argument{
        "checkpoint diagnostic state requires at least one rank fragment"};
  }
  const std::uint64_t table_bytes =
      static_cast<std::uint64_t>(sizes.size()) * sizeof(std::uint64_t);
  const std::uint64_t total = diagnostic_state_header_bytes + table_bytes
                            + static_cast<std::uint64_t>(payload.size());
  if (total > max_checkpoint_diagnostic_state_bytes
      || total > std::numeric_limits<std::size_t>::max()) {
    throw std::length_error{
        "checkpoint diagnostic state exceeds the bounded envelope size"};
  }
  std::vector<std::uint8_t> body;
  body.reserve(static_cast<std::size_t>(table_bytes) + payload.size());
  for (const auto size : sizes) append_u64_le(body, size);
  body.insert(body.end(), payload.begin(), payload.end());

  std::vector<std::uint8_t> result;
  result.reserve(static_cast<std::size_t>(total));
  result.insert(result.end(), diagnostic_state_magic.begin(),
                diagnostic_state_magic.end());
  append_u64_le(result, 1);  // binary envelope version
  append_u64_le(result, static_cast<std::uint64_t>(sizes.size()));
  append_u64_le(result, static_cast<std::uint64_t>(payload.size()));
  append_u64_le(result, diagnostic_state_hash(body));
  result.insert(result.end(), body.begin(), body.end());
  return result;
}

struct Hdf5Types {
  hid_t file{-1};
  hid_t memory{-1};
};

Hdf5Types hdf5_types(CheckpointValueType type) {
  switch (type) {
    case CheckpointValueType::float32:
      return {H5T_IEEE_F32LE, H5T_NATIVE_FLOAT};
    case CheckpointValueType::float64:
      return {H5T_IEEE_F64LE, H5T_NATIVE_DOUBLE};
    case CheckpointValueType::uint8:
      return {H5T_STD_U8LE, H5T_NATIVE_UINT8};
    case CheckpointValueType::uint64:
      return {H5T_STD_U64LE, H5T_NATIVE_UINT64};
    case CheckpointValueType::int32:
      return {H5T_STD_I32LE, H5T_NATIVE_INT32};
  }
  throw std::invalid_argument{"unknown checkpoint value type"};
}

std::string serialize_metadata(const CheckpointMetadata& metadata) {
  std::ostringstream stream;
  const auto append = [&stream](std::string_view value) {
    stream << value.size() << ':' << value;
  };
  append(metadata.schema);
  append(metadata.physics);
  append(metadata.precision);
  append(metadata.geometry);
  append(metadata.unit_system);
  stream << metadata.global_nx << ',' << metadata.global_ny << ';';
  append(metadata.boundary_signature);
  append(metadata.species_signature);
  append(metadata.background_signature);
  append(metadata.numerics_signature);
  stream << metadata.step << ';'
         << std::hexfloat << metadata.time << ';'
         << metadata.diagnostic_state_bytes;
  return stream.str();
}

std::string unique_temporary_path(const std::filesystem::path& committed) {
  const auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
#if defined(__unix__) || defined(__APPLE__)
  const auto process = static_cast<unsigned long long>(::getpid());
#else
  const auto process = 0ULL;
#endif
  std::ostringstream suffix;
  suffix << ".tmp." << process << '.'
         << static_cast<unsigned long long>(ticks);
  return committed.string() + suffix.str();
}

CollectiveFixedMessage local_error_text(std::string_view operation) noexcept {
  CollectiveFixedMessage result{operation};
  result.append(" failed in the parallel HDF5 checkpoint");
  return result;
}

class Hdf5Handle {
 public:
  using Closer = herr_t (*)(hid_t);

  Hdf5Handle() = default;
  Hdf5Handle(hid_t value, Closer closer) : value_{value}, closer_{closer} {}
  ~Hdf5Handle() noexcept { (void)close(); }

  Hdf5Handle(const Hdf5Handle&) = delete;
  Hdf5Handle& operator=(const Hdf5Handle&) = delete;
  Hdf5Handle(Hdf5Handle&& other) noexcept
    : value_{std::exchange(other.value_, -1)}, closer_{other.closer_} {}
  Hdf5Handle& operator=(Hdf5Handle&& other) noexcept {
    if (this != &other) {
      (void)close();
      value_ = std::exchange(other.value_, -1);
      closer_ = other.closer_;
    }
    return *this;
  }

  [[nodiscard]] hid_t get() const noexcept { return value_; }
  [[nodiscard]] bool valid() const noexcept { return value_ >= 0; }
  [[nodiscard]] hid_t release() noexcept { return std::exchange(value_, -1); }

  herr_t close() noexcept {
    if (!valid()) return 0;
    const hid_t value = std::exchange(value_, -1);
    return closer_ == nullptr ? 0 : closer_(value);
  }

 private:
  hid_t value_{-1};
  Closer closer_{nullptr};
};

void collective_require_collective_handle(
    MpiRuntime& runtime, std::uint64_t& epoch, bool local_success,
    std::string_view phase, std::string_view message, Hdf5Handle& handle,
    int code = -1) {
  const CollectiveErrorRecord record = local_success
      ? CollectiveErrorRecord::success(epoch, runtime.rank(), phase)
      : CollectiveErrorRecord::failure(
            epoch, runtime.rank(), -1, code, phase, message);
  ++epoch;
  const CollectiveResolution resolution = runtime.consensus(record);
  if (!resolution.accepted()) {
    // A handle returned by a collective HDF5 operation cannot safely be
    // closed by only the ranks on which creation succeeded.  Abandon a
    // partial handle instead; the run is already entering its fatal path.
    (void)handle.release();
    throw DistributedCollectiveError{resolution};
  }
}

Hdf5Handle create_parallel_access(MpiRuntime& runtime,
                                  std::uint64_t& epoch,
                                  std::string_view phase) {
  Hdf5Handle property{H5Pcreate(H5P_FILE_ACCESS), H5Pclose};
  collective_require(runtime, epoch, property.valid(), phase,
                     local_error_text("H5Pcreate(H5P_FILE_ACCESS)"));

  // H5Pset_fapl_mpio may duplicate the communicator internally.  Do not let
  // one rank skip it because an earlier rank-local property-list setup failed.
  bool success =
      H5Pset_fapl_mpio(property.get(),
                      detail::MpiRuntimeNativeAccess::world(runtime),
                      MPI_INFO_NULL) >= 0;
  collective_require_collective_handle(
      runtime, epoch, success, phase, local_error_text("H5Pset_fapl_mpio"),
      property);
#if H5_VERSION_GE(1, 10, 0)
  success = H5Pset_all_coll_metadata_ops(property.get(), 1) >= 0;
  if (success) {
    success = H5Pset_coll_metadata_write(property.get(), 1) >= 0;
  }
  collective_require(runtime, epoch, success, phase,
                     local_error_text("parallel metadata property setup"));
#endif
  return property;
}

void write_string_attribute(MpiRuntime& runtime, hid_t object,
                            std::uint64_t& epoch, const char* name,
                            const std::string& value) {
  Hdf5Handle type{H5Tcopy(H5T_C_S1), H5Tclose};
  Hdf5Handle space{H5Screate(H5S_SCALAR), H5Sclose};
  bool success = type.valid() && space.valid();
  if (success) success = H5Tset_size(type.get(), value.size() + 1) >= 0;
  if (success) success = H5Tset_strpad(type.get(), H5T_STR_NULLTERM) >= 0;
  collective_require(runtime, epoch, success, "checkpoint-metadata-setup",
                     local_error_text(name));

  Hdf5Handle attribute{
      H5Acreate2(object, name, type.get(), space.get(), H5P_DEFAULT,
                 H5P_DEFAULT),
      H5Aclose};
  success = attribute.valid();
  collective_require_collective_handle(
      runtime, epoch, success, "checkpoint-metadata", local_error_text(name),
      attribute);
  success = H5Awrite(attribute.get(), type.get(), value.c_str()) >= 0;
  collective_require(runtime, epoch, success, "checkpoint-metadata",
                     local_error_text(name));
  success = attribute.close() >= 0;
  success = space.close() >= 0 && success;
  success = type.close() >= 0 && success;
  collective_require(runtime, epoch, success, "checkpoint-metadata-close",
                     local_error_text(name));
}

template <class T>
void write_scalar_attribute(MpiRuntime& runtime, hid_t object,
                            std::uint64_t& epoch, const char* name,
                            hid_t file_type, hid_t memory_type, T value) {
  Hdf5Handle space{H5Screate(H5S_SCALAR), H5Sclose};
  collective_require(runtime, epoch, space.valid(),
                     "checkpoint-metadata-setup", local_error_text(name));
  Hdf5Handle attribute{
      H5Acreate2(object, name, file_type, space.get(), H5P_DEFAULT,
                 H5P_DEFAULT),
      H5Aclose};
  collective_require_collective_handle(
      runtime, epoch, attribute.valid(), "checkpoint-metadata",
      local_error_text(name), attribute);
  bool success = H5Awrite(attribute.get(), memory_type, &value) >= 0;
  collective_require(runtime, epoch, success, "checkpoint-metadata",
                     local_error_text(name));
  success = attribute.close() >= 0;
  success = space.close() >= 0 && success;
  collective_require(runtime, epoch, success, "checkpoint-metadata-close",
                     local_error_text(name));
}

void write_metadata(MpiRuntime& runtime, hid_t file, std::uint64_t& epoch,
                    const CheckpointMetadata& metadata) {
  write_string_attribute(runtime, file, epoch, "schema", metadata.schema);
  write_string_attribute(runtime, file, epoch, "physics", metadata.physics);
  write_string_attribute(runtime, file, epoch, "precision", metadata.precision);
  write_string_attribute(runtime, file, epoch, "geometry", metadata.geometry);
  write_string_attribute(runtime, file, epoch, "unit_system",
                         metadata.unit_system);
  write_scalar_attribute(runtime, file, epoch, "global_nx", H5T_STD_U64LE,
                         H5T_NATIVE_UINT64, metadata.global_nx);
  write_scalar_attribute(runtime, file, epoch, "global_ny", H5T_STD_U64LE,
                         H5T_NATIVE_UINT64, metadata.global_ny);
  write_string_attribute(runtime, file, epoch, "boundary_signature",
                         metadata.boundary_signature);
  write_string_attribute(runtime, file, epoch, "species_signature",
                         metadata.species_signature);
  write_string_attribute(runtime, file, epoch, "background_signature",
                         metadata.background_signature);
  write_string_attribute(runtime, file, epoch, "numerics_signature",
                         metadata.numerics_signature);
  write_scalar_attribute(runtime, file, epoch, "step", H5T_STD_U64LE,
                         H5T_NATIVE_UINT64, metadata.step);
  write_scalar_attribute(runtime, file, epoch, "time", H5T_IEEE_F64LE,
                         H5T_NATIVE_DOUBLE, metadata.time);
  write_scalar_attribute(runtime, file, epoch, "diagnostic_state_bytes",
                         H5T_STD_U64LE, H5T_NATIVE_UINT64,
                         metadata.diagnostic_state_bytes);
}

std::string read_string_attribute(MpiRuntime& runtime, hid_t object,
                                  std::uint64_t& epoch, const char* name) {
  Hdf5Handle attribute{H5Aopen(object, name, H5P_DEFAULT), H5Aclose};
  collective_require_collective_handle(
      runtime, epoch, attribute.valid(), "restart-metadata-open",
      local_error_text(name), attribute);

  Hdf5Handle type{H5Aget_type(attribute.get()), H5Tclose};
  Hdf5Handle space{H5Aget_space(attribute.get()), H5Sclose};
  bool valid = type.valid() && space.valid();
  std::size_t size = 0;
  if (valid) {
    valid = H5Tget_class(type.get()) == H5T_STRING;
  }
  if (valid) {
    const htri_t variable = H5Tis_variable_str(type.get());
    size = H5Tget_size(type.get());
    valid = variable == 0
        && H5Sget_simple_extent_type(space.get()) == H5S_SCALAR
        && size != 0 && size <= (1U << 20);
  }
  collective_require(
      runtime, epoch, valid, "restart-metadata-validate",
      "checkpoint string metadata must be a bounded fixed-length scalar", -14);

  std::vector<char> storage;
  collective_try_with_fallback(
      runtime, epoch, "restart-metadata-setup",
      "checkpoint string metadata allocation failed", -15,
      [&] { storage.assign(size + 1, '\0'); });

  bool success = H5Aread(attribute.get(), type.get(), storage.data()) >= 0;
  collective_require(runtime, epoch, success, "restart-metadata-read",
                     local_error_text(name));

  const auto terminator = std::find(storage.begin(), storage.begin() + size,
                                    '\0');
  const bool terminated = terminator != storage.begin() + size;
  collective_require(runtime, epoch, terminated, "restart-metadata-validate",
                     "checkpoint string metadata is not null terminated", -16);
  std::string result;
  collective_try_with_fallback(
      runtime, epoch, "restart-metadata-setup",
      "checkpoint string metadata allocation failed", -23, [&] {
        result.assign(storage.data(), static_cast<std::size_t>(
                                          terminator - storage.begin()));
      });

  success = space.close() >= 0;
  success = type.close() >= 0 && success;
  success = attribute.close() >= 0 && success;
  collective_require(runtime, epoch, success, "restart-metadata-close",
                     local_error_text(name));
  return result;
}

template <class T>
T read_scalar_attribute(MpiRuntime& runtime, hid_t object,
                        std::uint64_t& epoch, const char* name,
                        hid_t memory_type) {
  Hdf5Handle attribute{H5Aopen(object, name, H5P_DEFAULT), H5Aclose};
  collective_require_collective_handle(
      runtime, epoch, attribute.valid(), "restart-metadata-open",
      local_error_text(name), attribute);
  Hdf5Handle space{H5Aget_space(attribute.get()), H5Sclose};
  const bool valid = space.valid()
      && H5Sget_simple_extent_type(space.get()) == H5S_SCALAR;
  collective_require(runtime, epoch, valid, "restart-metadata-validate",
                     "checkpoint scalar metadata has a non-scalar dataspace",
                     -17);
  T value{};
  bool success = H5Aread(attribute.get(), memory_type, &value) >= 0;
  collective_require(runtime, epoch, success, "restart-metadata-read",
                     local_error_text(name));
  success = space.close() >= 0;
  success = attribute.close() >= 0 && success;
  collective_require(runtime, epoch, success, "restart-metadata-close",
                     local_error_text(name));
  return value;
}

CheckpointMetadata read_metadata(MpiRuntime& runtime, hid_t file,
                                 std::uint64_t& epoch) {
  CheckpointMetadata metadata;
  metadata.schema = read_string_attribute(runtime, file, epoch, "schema");
  metadata.physics = read_string_attribute(runtime, file, epoch, "physics");
  metadata.precision = read_string_attribute(runtime, file, epoch, "precision");
  metadata.geometry = read_string_attribute(runtime, file, epoch, "geometry");
  metadata.unit_system =
      read_string_attribute(runtime, file, epoch, "unit_system");
  metadata.global_nx = read_scalar_attribute<std::uint64_t>(
      runtime, file, epoch, "global_nx", H5T_NATIVE_UINT64);
  metadata.global_ny = read_scalar_attribute<std::uint64_t>(
      runtime, file, epoch, "global_ny", H5T_NATIVE_UINT64);
  metadata.boundary_signature =
      read_string_attribute(runtime, file, epoch, "boundary_signature");
  metadata.species_signature =
      read_string_attribute(runtime, file, epoch, "species_signature");
  metadata.background_signature =
      read_string_attribute(runtime, file, epoch, "background_signature");
  metadata.numerics_signature =
      read_string_attribute(runtime, file, epoch, "numerics_signature");
  metadata.step = read_scalar_attribute<std::uint64_t>(
      runtime, file, epoch, "step", H5T_NATIVE_UINT64);
  metadata.time = read_scalar_attribute<double>(
      runtime, file, epoch, "time", H5T_NATIVE_DOUBLE);
  metadata.diagnostic_state_bytes = read_scalar_attribute<std::uint64_t>(
      runtime, file, epoch, "diagnostic_state_bytes", H5T_NATIVE_UINT64);
  return metadata;
}

std::size_t checked_element_count(std::span<const std::uint64_t> count) {
  std::size_t result = 1;
  for (const std::uint64_t dimension : count) {
    if (dimension == 0) {
      throw std::invalid_argument{"checkpoint hyperslabs must be non-empty"};
    }
    if (dimension > std::numeric_limits<std::size_t>::max() / result) {
      throw std::overflow_error{"checkpoint hyperslab element count overflows"};
    }
    result *= static_cast<std::size_t>(dimension);
  }
  return result;
}

bool slabs_overlap(const DatasetHyperslab& left,
                   const DatasetHyperslab& right) {
  for (std::size_t axis = 0; axis < left.offset.size(); ++axis) {
    const std::uint64_t left_end = left.offset[axis] + left.count[axis];
    const std::uint64_t right_end = right.offset[axis] + right.count[axis];
    if (left_end <= right.offset[axis] || right_end <= left.offset[axis]) {
      return false;
    }
  }
  return true;
}

std::size_t validate_dataset_selection(
    std::span<const std::uint64_t> global_shape,
    std::span<const DatasetHyperslab> slabs,
    const void* packed_values,
    std::size_t packed_elements) {
  if (global_shape.empty()) {
    throw std::invalid_argument{"checkpoint datasets must have positive rank"};
  }
  if (global_shape.size()
      > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument{
        "checkpoint dataset rank exceeds the HDF5 rank range"};
  }
  for (const auto dimension : global_shape) {
    if (dimension == 0
        || dimension > static_cast<std::uint64_t>(
                           std::numeric_limits<hsize_t>::max())) {
      throw std::invalid_argument{
          "checkpoint dataset dimensions must be representable and positive"};
    }
  }
  std::size_t selected = 0;
  for (std::size_t index = 0; index < slabs.size(); ++index) {
    const auto& slab = slabs[index];
    if (slab.offset.size() != global_shape.size()
        || slab.count.size() != global_shape.size()) {
      throw std::invalid_argument{
          "checkpoint hyperslab rank differs from its dataset rank"};
    }
    for (std::size_t axis = 0; axis < global_shape.size(); ++axis) {
      if (slab.offset[axis] > global_shape[axis]
          || slab.count[axis] > global_shape[axis] - slab.offset[axis]) {
        throw std::out_of_range{
            "checkpoint hyperslab lies outside the global dataset"};
      }
    }
    const std::size_t count = checked_element_count(slab.count);
    if (count > std::numeric_limits<std::size_t>::max() - selected) {
      throw std::overflow_error{
          "checkpoint local selection element count overflows"};
    }
    selected += count;
    if (index != 0 && !(slabs[index - 1].offset < slab.offset)) {
      throw std::invalid_argument{
          "checkpoint hyperslabs must be in strict lexicographic offset order"};
    }
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (slabs_overlap(slabs[prior], slab)) {
        throw std::invalid_argument{
            "checkpoint hyperslabs on one rank must not overlap"};
      }
    }
  }
  if (selected != packed_elements) {
    throw std::invalid_argument{
        "packed checkpoint element count does not match its hyperslabs"};
  }
  if (packed_elements != 0 && packed_values == nullptr) {
    throw std::invalid_argument{
        "packed checkpoint buffer is null for a non-empty selection"};
  }
  return selected;
}

std::uint64_t checked_volume(std::span<const std::uint64_t> dimensions) {
  std::uint64_t result = 1;
  for (const std::uint64_t dimension : dimensions) {
    if (dimension == 0
        || dimension > std::numeric_limits<std::uint64_t>::max() / result) {
      throw std::overflow_error{
          "checkpoint dataset volume overflows global coverage validation"};
    }
    result *= dimension;
  }
  return result;
}

std::vector<std::uint64_t> encode_selection(
    std::span<const DatasetHyperslab> slabs, std::size_t rank) {
  if (rank > static_cast<std::size_t>(std::numeric_limits<int>::max())
      || (rank != 0
          && slabs.size()
              > (static_cast<std::size_t>(std::numeric_limits<int>::max()) - 1)
                    / (2 * rank))) {
    throw std::length_error{
        "checkpoint hyperslab metadata exceeds MPI count range"};
  }
  std::vector<std::uint64_t> encoded;
  encoded.reserve(1 + slabs.size() * 2 * rank);
  encoded.push_back(static_cast<std::uint64_t>(slabs.size()));
  for (const auto& slab : slabs) {
    encoded.insert(encoded.end(), slab.offset.begin(), slab.offset.end());
    encoded.insert(encoded.end(), slab.count.begin(), slab.count.end());
  }
  return encoded;
}

struct GatheredDatasetSelections {
  std::vector<int> counts;
  std::vector<int> offsets;
  std::vector<std::uint64_t> words;
};

bool prepare_selection_layout(std::span<const int> counts,
                              std::span<int> offsets,
                              std::int64_t& total) noexcept {
  total = 0;
  for (std::size_t index = 0; index < counts.size(); ++index) {
    if (counts[index] <= 0
        || total > std::numeric_limits<int>::max() - counts[index]) {
      return false;
    }
    offsets[index] = static_cast<int>(total);
    total += counts[index];
  }
  return true;
}

GatheredDatasetSelections gather_dataset_selections(
    MpiRuntime& runtime, std::uint64_t& epoch,
    std::span<const std::uint64_t> global_shape,
    std::span<const DatasetHyperslab> local_slabs,
    std::string_view phase) {
  std::vector<std::uint64_t> local;
  collective_try(runtime, epoch, phase,
                 "checkpoint selection encoding failed", -18, [&] {
    local = encode_selection(local_slabs, global_shape.size());
  });

  GatheredDatasetSelections gathered;
  collective_try_with_fallback(
      runtime, epoch, phase,
      "checkpoint partition metadata allocation failed", -19, [&] {
        gathered.counts.resize(static_cast<std::size_t>(runtime.size()));
        gathered.offsets.resize(static_cast<std::size_t>(runtime.size()));
      });
  const int local_count = static_cast<int>(local.size());
  check_mpi(MPI_Allgather(
                &local_count, 1, MPI_INT, gathered.counts.data(), 1, MPI_INT,
                detail::MpiRuntimeNativeAccess::world(runtime)),
            "MPI_Allgather(checkpoint hyperslab sizes)");

  std::int64_t total = 0;
  const bool counts_valid = prepare_selection_layout(
      gathered.counts, gathered.offsets, total);
  collective_require(
      runtime, epoch, counts_valid, phase,
      "global checkpoint partition metadata exceeds MPI count range", -20);
  collective_try_with_fallback(
      runtime, epoch, phase,
      "global checkpoint partition metadata allocation failed", -21,
      [&] { gathered.words.resize(static_cast<std::size_t>(total)); });
  check_mpi(MPI_Allgatherv(
                local.data(), local_count, MPI_UINT64_T,
                gathered.words.data(), gathered.counts.data(),
                gathered.offsets.data(), MPI_UINT64_T,
                detail::MpiRuntimeNativeAccess::world(runtime)),
            "MPI_Allgatherv(checkpoint hyperslabs)");
  return gathered;
}

std::vector<DatasetHyperslab> decode_dataset_selections(
    std::span<const std::uint64_t> global_shape,
    const GatheredDatasetSelections& gathered) {
  std::vector<DatasetHyperslab> slabs;
  for (std::size_t process = 0; process < gathered.counts.size(); ++process) {
    const std::size_t begin =
        static_cast<std::size_t>(gathered.offsets[process]);
    const std::size_t end =
        begin + static_cast<std::size_t>(gathered.counts[process]);
    std::size_t cursor = begin;
    if (cursor == end) {
      throw std::invalid_argument{
          "checkpoint rank omitted its partition metadata"};
    }
    const std::uint64_t slab_count = gathered.words[cursor++];
    const std::uint64_t words_per_slab = 2 * global_shape.size();
    if (words_per_slab != 0
        && slab_count > (end - cursor) / words_per_slab) {
      throw std::invalid_argument{
          "checkpoint partition metadata is malformed"};
    }
    if (cursor + slab_count * words_per_slab != end) {
      throw std::invalid_argument{
          "checkpoint partition metadata has trailing data"};
    }
    for (std::uint64_t index = 0; index < slab_count; ++index) {
      DatasetHyperslab slab;
      slab.offset.assign(
          gathered.words.begin() + static_cast<std::ptrdiff_t>(cursor),
          gathered.words.begin() + static_cast<std::ptrdiff_t>(
                                       cursor + global_shape.size()));
      cursor += global_shape.size();
      slab.count.assign(
          gathered.words.begin() + static_cast<std::ptrdiff_t>(cursor),
          gathered.words.begin() + static_cast<std::ptrdiff_t>(
                                       cursor + global_shape.size()));
      cursor += global_shape.size();
      slabs.push_back(std::move(slab));
    }
  }
  return slabs;
}

void validate_exact_dataset_coverage(
    std::span<const std::uint64_t> global_shape,
    std::span<const DatasetHyperslab> slabs) {
  std::uint64_t selected = 0;
  for (std::size_t index = 0; index < slabs.size(); ++index) {
    const std::uint64_t volume = checked_volume(slabs[index].count);
    if (volume > std::numeric_limits<std::uint64_t>::max() - selected) {
      throw std::overflow_error{
          "checkpoint selected volume overflows global coverage validation"};
    }
    selected += volume;
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (slabs_overlap(slabs[prior], slabs[index])) {
        throw std::invalid_argument{
            "checkpoint hyperslabs overlap across MPI ranks"};
      }
    }
  }
  if (selected != checked_volume(global_shape)) {
    throw std::invalid_argument{
        "checkpoint hyperslabs do not exactly cover the global dataset"};
  }
}

void validate_global_dataset_selection(
    MpiRuntime& runtime, std::uint64_t& epoch,
    std::span<const std::uint64_t> global_shape,
    std::span<const DatasetHyperslab> local_slabs,
    std::string_view phase) {
  const GatheredDatasetSelections gathered = gather_dataset_selections(
      runtime, epoch, global_shape, local_slabs, phase);
  collective_try(runtime, epoch, phase,
                 "checkpoint partition validation failed", -22, [&] {
    const auto slabs = decode_dataset_selections(global_shape, gathered);
    validate_exact_dataset_coverage(global_shape, slabs);
  });
}

std::string dataset_descriptor(std::string_view name,
                               CheckpointValueType type,
                               std::span<const std::uint64_t> shape) {
  std::ostringstream stream;
  stream << name << '#' << static_cast<int>(type);
  for (const auto dimension : shape) stream << ':' << dimension;
  return stream.str();
}

void require_common_descriptor(MpiRuntime& runtime, std::uint64_t& epoch,
                               const std::string& local,
                               std::string_view phase) {
  collective_require_common_string(
      runtime, epoch, local, phase, checkpoint_string_broadcast,
      "ranks disagree on checkpoint operation ordering or shape", -2);
}

std::vector<hsize_t> hdf5_shape(std::span<const std::uint64_t> shape) {
  std::vector<hsize_t> result;
  result.reserve(shape.size());
  for (const auto dimension : shape) result.push_back(dimension);
  return result;
}

void apply_file_selection(hid_t space,
                          std::span<const DatasetHyperslab> slabs) {
  if (slabs.empty()) {
    if (H5Sselect_none(space) < 0) {
      throw std::runtime_error{
          "H5Sselect_none failed in the parallel HDF5 checkpoint"};
    }
    return;
  }
  bool first = true;
  for (const auto& slab : slabs) {
    const auto offset = hdf5_shape(slab.offset);
    const auto count = hdf5_shape(slab.count);
    const H5S_seloper_t operation = first ? H5S_SELECT_SET : H5S_SELECT_OR;
    if (H5Sselect_hyperslab(space, operation, offset.data(), nullptr,
                            count.data(), nullptr) < 0) {
      throw std::runtime_error{
          "H5Sselect_hyperslab failed in the parallel HDF5 checkpoint"};
    }
    first = false;
  }
}

hid_t packed_memory_space(std::size_t elements) {
  const hsize_t size = elements == 0 ? 1 : static_cast<hsize_t>(elements);
  const hid_t space = H5Screate_simple(1, &size, nullptr);
  if (space < 0) return space;
  if (elements == 0 && H5Sselect_none(space) < 0) {
    (void)H5Sclose(space);
    return -1;
  }
  return space;
}

void validate_dataset_name(std::string_view name) {
  if (name.empty() || name.find('\0') != std::string_view::npos
      || name == "/" || name.find("..") != std::string_view::npos) {
    throw std::invalid_argument{"invalid checkpoint dataset path"};
  }
}

struct PreparedDatasetOperation {
  std::string name;
  std::string descriptor;
  Hdf5Types types;
};

PreparedDatasetOperation prepare_dataset_operation(
    MpiRuntime& runtime, std::uint64_t& epoch, std::string_view name,
    CheckpointValueType value_type,
    std::span<const std::uint64_t> global_shape,
    std::span<const DatasetHyperslab> local_slabs, const void* packed_values,
    std::size_t packed_elements, std::string_view validation_phase,
    std::string_view validation_fallback, int validation_code,
    std::string_view order_phase, std::string_view partition_phase) {
  PreparedDatasetOperation operation;
  collective_try(runtime, epoch, validation_phase, validation_fallback,
                 validation_code, [&] {
    validate_dataset_name(name);
    operation.name.assign(name);
    operation.types = hdf5_types(value_type);
    operation.descriptor = dataset_descriptor(name, value_type, global_shape);
    (void)validate_dataset_selection(global_shape, local_slabs, packed_values,
                                     packed_elements);
  });
  require_common_descriptor(runtime, epoch, operation.descriptor, order_phase);
  validate_global_dataset_selection(runtime, epoch, global_shape, local_slabs,
                                    partition_phase);
  return operation;
}

void select_dataset_hyperslabs(
    MpiRuntime& runtime, std::uint64_t& epoch, Hdf5Handle& file_space,
    std::span<const DatasetHyperslab> local_slabs,
    std::string_view phase, std::string_view fallback, int code) {
  collective_try(runtime, epoch, phase, fallback, code, [&] {
    apply_file_selection(file_space.get(), local_slabs);
  });
}

struct DatasetTransferHandles {
  Hdf5Handle memory_space;
  Hdf5Handle transfer;
};

enum class DatasetCloseOrder {
  dataset_before_space,
  space_before_dataset,
};

DatasetTransferHandles prepare_dataset_transfer(
    MpiRuntime& runtime, std::uint64_t& epoch, std::size_t packed_elements,
    std::string_view phase) {
  DatasetTransferHandles handles{
      .memory_space = Hdf5Handle{packed_memory_space(packed_elements),
                                H5Sclose},
      .transfer = Hdf5Handle{H5Pcreate(H5P_DATASET_XFER), H5Pclose},
  };
  bool success = handles.memory_space.valid() && handles.transfer.valid();
  collective_require(runtime, epoch, success, phase,
                     local_error_text("H5Screate_simple/H5Pcreate"));
  success = H5Pset_dxpl_mpio(handles.transfer.get(),
                            H5FD_MPIO_COLLECTIVE) >= 0;
  collective_require(runtime, epoch, success, phase,
                     local_error_text("H5Pset_dxpl_mpio"));
  return handles;
}

void close_dataset_handles(MpiRuntime& runtime, std::uint64_t& epoch,
                           DatasetTransferHandles& transfer,
                           Hdf5Handle& file_space, Hdf5Handle& dataset,
                           std::string_view phase,
                           DatasetCloseOrder close_order) {
  bool success = transfer.transfer.close() >= 0;
  success = transfer.memory_space.close() >= 0 && success;
  if (close_order == DatasetCloseOrder::dataset_before_space) {
    success = dataset.close() >= 0 && success;
    success = file_space.close() >= 0 && success;
  } else {
    success = file_space.close() >= 0 && success;
    success = dataset.close() >= 0 && success;
  }
  collective_require(runtime, epoch, success, phase,
                     local_error_text("checkpoint dataset handle close"));
}

bool dataset_is_compatible(
    hid_t file_space, hid_t stored_type, const Hdf5Types& requested_types,
    std::span<const std::uint64_t> expected_global_shape) {
  if (H5Tequal(stored_type, requested_types.file) <= 0) return false;
  const int rank = H5Sget_simple_extent_ndims(file_space);
  if (rank != static_cast<int>(expected_global_shape.size())) return false;
  std::vector<hsize_t> dimensions(expected_global_shape.size());
  if (H5Sget_simple_extent_dims(file_space, dimensions.data(), nullptr)
      != rank) {
    return false;
  }
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
    if (dimensions[axis] != expected_global_shape[axis]) return false;
  }
  return true;
}

struct CheckpointDiagnosticLayout {
  std::vector<std::uint64_t> sizes;
  std::vector<int> counts;
  std::vector<int> offsets;
  std::uint64_t payload_size{0};
};

bool prepare_diagnostic_layout(CheckpointDiagnosticLayout& layout) noexcept {
  layout.payload_size = 0;
  for (std::size_t rank = 0; rank < layout.sizes.size(); ++rank) {
    if (layout.sizes[rank]
            > static_cast<std::uint64_t>(std::numeric_limits<int>::max())
        || layout.payload_size
            > static_cast<std::uint64_t>(std::numeric_limits<int>::max())
        || layout.sizes[rank]
            > static_cast<std::uint64_t>(std::numeric_limits<int>::max())
                  - layout.payload_size) {
      return false;
    }
    layout.counts[rank] = static_cast<int>(layout.sizes[rank]);
    layout.offsets[rank] = static_cast<int>(layout.payload_size);
    layout.payload_size += layout.sizes[rank];
  }
  const std::uint64_t table_bytes =
      static_cast<std::uint64_t>(layout.sizes.size())
      * sizeof(std::uint64_t);
  return layout.payload_size <= max_checkpoint_diagnostic_state_bytes
      && table_bytes <= max_checkpoint_diagnostic_state_bytes
      && diagnostic_state_header_bytes + table_bytes
             <= max_checkpoint_diagnostic_state_bytes - layout.payload_size;
}

CheckpointDiagnosticLayout gather_diagnostic_layout(
    MpiRuntime& runtime, std::uint64_t& epoch, std::uint64_t local_size) {
  CheckpointDiagnosticLayout layout;
  collective_try_with_fallback(
      runtime, epoch, "checkpoint-diagnostics-size",
      "checkpoint diagnostic size-table allocation failed", -31, [&] {
        layout.sizes.resize(static_cast<std::size_t>(runtime.size()));
      });
  check_mpi(MPI_Allgather(
                &local_size, 1, MPI_UINT64_T, layout.sizes.data(), 1,
                MPI_UINT64_T,
                detail::MpiRuntimeNativeAccess::world(runtime)),
            "MPI_Allgather(checkpoint diagnostic sizes)");
  collective_try_with_fallback(
      runtime, epoch, "checkpoint-diagnostics-size",
      "checkpoint diagnostic offset-table allocation failed", -40, [&] {
        layout.counts.resize(layout.sizes.size());
        layout.offsets.resize(layout.sizes.size());
      });
  collective_require(
      runtime, epoch, prepare_diagnostic_layout(layout),
      "checkpoint-diagnostics-size",
      "global checkpoint diagnostic state exceeds the bounded size", -32);
  return layout;
}

std::vector<std::uint8_t> gather_diagnostic_payload(
    MpiRuntime& runtime, std::uint64_t& epoch,
    std::span<const std::uint8_t> local_fragment,
    const CheckpointDiagnosticLayout& layout) {
  std::vector<std::uint8_t> payload;
  collective_try_with_fallback(
      runtime, epoch, "checkpoint-diagnostics-gather",
      "checkpoint diagnostic payload allocation failed", -33,
      [&] { payload.resize(static_cast<std::size_t>(layout.payload_size)); });
  std::uint8_t dummy{};
  check_mpi(MPI_Allgatherv(
                local_fragment.empty() ? &dummy : local_fragment.data(),
                static_cast<int>(local_fragment.size()), MPI_UINT8_T,
                payload.empty() ? &dummy : payload.data(),
                layout.counts.data(), layout.offsets.data(), MPI_UINT8_T,
                detail::MpiRuntimeNativeAccess::world(runtime)),
            "MPI_Allgatherv(checkpoint diagnostic payload)");
  return payload;
}

void require_common_serialized_metadata(
    MpiRuntime& runtime, std::uint64_t& epoch,
    const CheckpointMetadata& metadata, std::string_view phase,
    std::string_view serialization_failure, int serialization_code,
    std::string_view mismatch, int mismatch_code) {
  std::string serialized;
  collective_try(runtime, epoch, phase, serialization_failure,
                 serialization_code,
                 [&] { serialized = serialize_metadata(metadata); });
  collective_require_common_string(
      runtime, epoch, serialized, phase, checkpoint_string_broadcast,
      mismatch, mismatch_code);
}

void prepare_checkpoint_temporary_path(
    MpiRuntime& runtime, std::uint64_t& epoch,
    const std::filesystem::path& committed_path,
    std::filesystem::path& temporary_path) {
  std::string local_temporary;
  collective_try_on_root(
      runtime, epoch, "checkpoint-open",
      "checkpoint temporary-path preparation failed", -49,
      [&] { local_temporary = unique_temporary_path(committed_path); });
  const std::string temporary = collective_broadcast_string(
      runtime, epoch, local_temporary, "checkpoint-open",
      checkpoint_string_broadcast);
  collective_try(runtime, epoch, "checkpoint-open",
                 "checkpoint temporary path allocation failed", -50,
                 [&] { temporary_path = temporary; });
}

std::vector<DatasetHyperslab> prepare_root_diagnostic_selection(
    MpiRuntime& runtime, std::uint64_t& epoch, std::uint64_t byte_count,
    std::string_view phase, std::string_view allocation_failure, int code) {
  std::vector<DatasetHyperslab> slabs;
  collective_try_with_fallback(
      runtime, epoch, phase, allocation_failure, code, [&] {
        if (runtime.rank() == 0) {
          slabs.push_back(DatasetHyperslab{
              .offset = {0}, .count = {byte_count}});
        }
      });
  return slabs;
}

}  // namespace

struct ParallelCheckpointWriter::Impl {
  MpiRuntime* runtime{nullptr};
  std::filesystem::path committed_path{};
  std::filesystem::path temporary_path{};
  CheckpointMetadata metadata{};
  hid_t file{-1};
  bool file_collectively_open{false};
  std::uint64_t epoch{0};
  bool committed{false};
  bool closed{false};
};

struct ParallelCheckpointReader::Impl {
  MpiRuntime* runtime{nullptr};
  std::filesystem::path path{};
  CheckpointMetadata metadata{};
  hid_t file{-1};
  bool file_collectively_open{false};
  std::uint64_t epoch{0};
  bool closed{false};
};

std::vector<std::uint8_t> collect_checkpoint_diagnostic_state(
    MpiRuntime& runtime, std::span<const std::uint8_t> local_fragment) {
  runtime.require_orchestration_thread();
  std::uint64_t epoch = 0;
  const std::uint64_t local_size =
      static_cast<std::uint64_t>(local_fragment.size());
  collective_require(
      runtime, epoch,
      local_size <= max_checkpoint_diagnostic_state_bytes
          && local_size <= static_cast<std::uint64_t>(
                               std::numeric_limits<int>::max()),
      "checkpoint-diagnostics-size",
      "rank-local checkpoint diagnostic state exceeds the bounded size",
      -30);
  const CheckpointDiagnosticLayout layout =
      gather_diagnostic_layout(runtime, epoch, local_size);
  const std::vector<std::uint8_t> payload =
      gather_diagnostic_payload(runtime, epoch, local_fragment, layout);
  std::vector<std::uint8_t> encoded;
  collective_try(
      runtime, epoch, "checkpoint-diagnostics-encode",
      "checkpoint diagnostic-state encoding failed", -34, [&] {
        encoded = encode_diagnostic_state(layout.sizes, payload);
  });
  return encoded;
}

std::vector<std::vector<std::uint8_t>>
decode_checkpoint_diagnostic_state(
    std::span<const std::uint8_t> encoded_state) {
  if (encoded_state.size() > max_checkpoint_diagnostic_state_bytes
      || encoded_state.size() < diagnostic_state_header_bytes
      || !std::equal(diagnostic_state_magic.begin(),
                     diagnostic_state_magic.end(), encoded_state.begin())) {
    throw std::invalid_argument{
        "checkpoint diagnostic-state envelope has an invalid header"};
  }
  std::size_t cursor = diagnostic_state_magic.size();
  const std::uint64_t version = read_u64_le(encoded_state, cursor);
  const std::uint64_t part_count = read_u64_le(encoded_state, cursor);
  const std::uint64_t payload_size = read_u64_le(encoded_state, cursor);
  const std::uint64_t expected_hash = read_u64_le(encoded_state, cursor);
  if (version != 1 || part_count == 0
      || part_count > static_cast<std::uint64_t>(1U << 20)) {
    throw std::invalid_argument{
        "checkpoint diagnostic-state envelope version or part count is invalid"};
  }
  if (part_count > (encoded_state.size() - cursor) / sizeof(std::uint64_t)) {
    throw std::invalid_argument{
        "checkpoint diagnostic-state size table is truncated"};
  }
  const std::size_t body_begin = cursor;
  std::vector<std::uint64_t> sizes(static_cast<std::size_t>(part_count));
  std::uint64_t summed_size = 0;
  for (auto& size : sizes) {
    size = read_u64_le(encoded_state, cursor);
    if (size > payload_size - std::min(payload_size, summed_size)) {
      throw std::invalid_argument{
          "checkpoint diagnostic-state part sizes overflow the payload"};
    }
    summed_size += size;
  }
  if (summed_size != payload_size
      || payload_size != encoded_state.size() - cursor
      || diagnostic_state_hash(encoded_state.subspan(body_begin))
             != expected_hash) {
    throw std::invalid_argument{
        "checkpoint diagnostic-state envelope length or checksum is invalid"};
  }
  std::vector<std::vector<std::uint8_t>> parts;
  parts.reserve(sizes.size());
  for (const auto size : sizes) {
    const std::size_t count = static_cast<std::size_t>(size);
    parts.emplace_back(encoded_state.begin()
                           + static_cast<std::ptrdiff_t>(cursor),
                       encoded_state.begin()
                           + static_cast<std::ptrdiff_t>(cursor + count));
    cursor += count;
  }
  return parts;
}

void validate_checkpoint_metadata(const CheckpointMetadata& metadata) {
  const auto valid_text = [](const std::string& value) {
    return value.size() < (1U << 20)
        && value.find('\0') == std::string::npos;
  };
  if (metadata.schema != checkpoint_schema) {
    throw std::invalid_argument{
        "checkpoint schema must be 'quasar-checkpoint/v1'"};
  }
  if (metadata.physics != "pic" && metadata.physics != "mhd") {
    throw std::invalid_argument{"checkpoint physics must be 'pic' or 'mhd'"};
  }
  if (metadata.precision.empty() || metadata.geometry.empty()
      || metadata.unit_system.empty() || metadata.global_nx == 0
      || metadata.global_ny == 0 || metadata.boundary_signature.empty()
      || metadata.numerics_signature.empty()) {
    throw std::invalid_argument{
        "checkpoint compatibility metadata is incomplete"};
  }
  if (!valid_text(metadata.schema) || !valid_text(metadata.physics)
      || !valid_text(metadata.precision) || !valid_text(metadata.geometry)
      || !valid_text(metadata.unit_system)
      || !valid_text(metadata.boundary_signature)
      || !valid_text(metadata.species_signature)
      || !valid_text(metadata.background_signature)
      || !valid_text(metadata.numerics_signature)) {
    throw std::invalid_argument{
        "checkpoint metadata strings must be bounded and contain no null bytes"};
  }
  if (metadata.physics == "pic" && metadata.species_signature.empty()) {
    throw std::invalid_argument{
        "PIC checkpoint metadata requires a species signature"};
  }
  if (!(std::isfinite(metadata.time) && metadata.time >= 0.0)) {
    throw std::invalid_argument{
        "checkpoint time must be finite and non-negative"};
  }
  if (metadata.diagnostic_state_bytes
      > max_checkpoint_diagnostic_state_bytes) {
    throw std::invalid_argument{
        "checkpoint diagnostic state exceeds the bounded envelope size"};
  }
}

void validate_restart_compatibility(const CheckpointMetadata& stored,
                                    const CheckpointMetadata& requested) {
  validate_checkpoint_metadata(stored);
  validate_checkpoint_metadata(requested);
  std::vector<std::string> mismatches;
  const auto compare = [&mismatches](std::string_view label,
                                     const auto& left, const auto& right) {
    if (left != right) mismatches.emplace_back(label);
  };
  compare("schema", stored.schema, requested.schema);
  compare("physics", stored.physics, requested.physics);
  compare("precision", stored.precision, requested.precision);
  compare("geometry", stored.geometry, requested.geometry);
  compare("unit system", stored.unit_system, requested.unit_system);
  compare("global mesh x", stored.global_nx, requested.global_nx);
  compare("global mesh y", stored.global_ny, requested.global_ny);
  compare("boundaries", stored.boundary_signature,
          requested.boundary_signature);
  compare("species", stored.species_signature,
          requested.species_signature);
  compare("backgrounds", stored.background_signature,
          requested.background_signature);
  compare("numerical schemes", stored.numerics_signature,
          requested.numerics_signature);
  if (!mismatches.empty()) {
    std::ostringstream message;
    message << "checkpoint is incompatible with the requested run: ";
    for (std::size_t index = 0; index < mismatches.size(); ++index) {
      if (index != 0) message << ", ";
      message << mismatches[index];
    }
    throw std::invalid_argument{message.str()};
  }
}

ParallelCheckpointWriter::ParallelCheckpointWriter(
    MpiRuntime& runtime, std::filesystem::path committed_path,
    CheckpointMetadata metadata)
  : implementation_{new (std::nothrow) Impl{}} {
  std::uint64_t allocation_epoch = 0;
  try {
    collective_require(
        runtime, allocation_epoch, implementation_ != nullptr,
        "checkpoint-object-allocation",
        "checkpoint writer allocation failed", -44);
  } catch (...) {
    delete implementation_;
    implementation_ = nullptr;
    throw;
  }
  Impl& impl = *implementation_;
  impl.epoch = allocation_epoch;
  impl.runtime = &runtime;
  impl.committed_path = std::move(committed_path);
  impl.metadata = std::move(metadata);
  try {
    std::string local_path;
    collective_try(runtime, impl.epoch, "checkpoint-open",
                   "checkpoint path preparation failed", -3, [&] {
      validate_checkpoint_metadata(impl.metadata);
      if (impl.committed_path.empty()) {
        throw std::invalid_argument{"checkpoint path must not be empty"};
      }
      if (runtime.rank() == 0
          && distributed_test_failure_enabled(
              "QUASAR_TEST_CHECKPOINT_PATH_PREPARATION_FAILURE")) {
        throw std::bad_alloc{};
      }
      local_path = impl.committed_path.string();
    });
    collective_require_common_string(
        runtime, impl.epoch, local_path, "checkpoint-open",
        checkpoint_string_broadcast,
        "ranks supplied different checkpoint paths", -4);
    require_common_serialized_metadata(
        runtime, impl.epoch, impl.metadata, "checkpoint-open",
        "checkpoint metadata serialization failed", -48,
        "ranks supplied different checkpoint metadata", -5);
    prepare_checkpoint_temporary_path(
        runtime, impl.epoch, impl.committed_path, impl.temporary_path);

    Hdf5Handle access =
        create_parallel_access(runtime, impl.epoch, "checkpoint-open");
    impl.file = H5Fcreate(impl.temporary_path.c_str(), H5F_ACC_EXCL,
                         H5P_DEFAULT, access.get());
    collective_require(runtime, impl.epoch, impl.file >= 0,
                       "checkpoint-open", local_error_text("H5Fcreate"));
    impl.file_collectively_open = true;
    const bool access_closed = access.close() >= 0;
    collective_require(runtime, impl.epoch, access_closed,
                       "checkpoint-open-close", local_error_text("H5Pclose"));
    write_metadata(runtime, impl.file, impl.epoch, impl.metadata);
  } catch (...) {
    // H5Fclose on a parallel file is collective.  Only enter it after every
    // rank confirmed that H5Fcreate returned a valid handle.
    if (impl.file_collectively_open) (void)H5Fclose(impl.file);
    delete implementation_;
    implementation_ = nullptr;
    throw;
  }
}

ParallelCheckpointWriter::~ParallelCheckpointWriter() noexcept {
  delete implementation_;
}

const std::filesystem::path& ParallelCheckpointWriter::committed_path() const noexcept {
  return implementation_->committed_path;
}

const std::filesystem::path& ParallelCheckpointWriter::temporary_path() const noexcept {
  return implementation_->temporary_path;
}

bool ParallelCheckpointWriter::committed() const noexcept {
  return implementation_->committed;
}

void ParallelCheckpointWriter::write_dataset(
    std::string_view name, CheckpointValueType value_type,
    std::span<const std::uint64_t> global_shape,
    std::span<const DatasetHyperslab> local_slabs,
    const void* packed_values, std::size_t packed_elements) {
  Impl& impl = *implementation_;
  if (impl.closed || impl.file < 0) {
    throw std::logic_error{"checkpoint writer is closed"};
  }
  const PreparedDatasetOperation operation = prepare_dataset_operation(
      *impl.runtime, impl.epoch, name, value_type, global_shape, local_slabs,
      packed_values, packed_elements, "checkpoint-dataset-validate",
      "checkpoint dataset validation failed", -6,
      "checkpoint-dataset-order", "checkpoint-dataset-partition");
  std::vector<hsize_t> shape;
  collective_try_with_fallback(
      *impl.runtime, impl.epoch, "checkpoint-dataset-setup",
      "checkpoint dataset shape allocation failed", -24,
      [&] { shape = hdf5_shape(global_shape); });
  Hdf5Handle file_space{
      H5Screate_simple(static_cast<int>(shape.size()), shape.data(), nullptr),
      H5Sclose};
  Hdf5Handle link_properties{H5Pcreate(H5P_LINK_CREATE), H5Pclose};
  bool success = file_space.valid() && link_properties.valid();
  collective_require(*impl.runtime, impl.epoch, success,
                     "checkpoint-dataset-setup",
                     local_error_text("H5Screate_simple/H5Pcreate"));
  if (success) {
    success = H5Pset_create_intermediate_group(link_properties.get(), 1) >= 0;
  }
  collective_require(*impl.runtime, impl.epoch, success,
                     "checkpoint-dataset-setup",
                     local_error_text("H5Pset_create_intermediate_group"));
  Hdf5Handle dataset{
      H5Dcreate2(impl.file, operation.name.c_str(), operation.types.file,
                 file_space.get(), link_properties.get(), H5P_DEFAULT,
                 H5P_DEFAULT),
      H5Dclose};
  success = dataset.valid();
  collective_require_collective_handle(
      *impl.runtime, impl.epoch, success, "checkpoint-dataset-create",
      local_error_text("H5Dcreate2"), dataset);
  success = link_properties.close() >= 0;
  collective_require(*impl.runtime, impl.epoch, success,
                     "checkpoint-dataset-setup-close",
                     local_error_text("H5Pclose"));

  select_dataset_hyperslabs(
      *impl.runtime, impl.epoch, file_space, local_slabs,
      "checkpoint-dataset-select", "checkpoint dataset selection failed",
      -7);
  DatasetTransferHandles transfer = prepare_dataset_transfer(
      *impl.runtime, impl.epoch, packed_elements,
      "checkpoint-dataset-select");

  std::byte dummy{};
  const void* values = packed_elements == 0 ? &dummy : packed_values;
  success = H5Dwrite(dataset.get(), operation.types.memory,
                     transfer.memory_space.get(), file_space.get(),
                     transfer.transfer.get(), values) >= 0;
  collective_require(*impl.runtime, impl.epoch, success,
                     "checkpoint-dataset-write",
                     local_error_text("H5Dwrite"));
  close_dataset_handles(
      *impl.runtime, impl.epoch, transfer, file_space, dataset,
      "checkpoint-dataset-close", DatasetCloseOrder::dataset_before_space);
}

void ParallelCheckpointWriter::write_diagnostic_state(
    std::span<const std::uint8_t> encoded_state) {
  Impl& impl = *implementation_;
  collective_try_if(
      *impl.runtime, impl.epoch,
      encoded_state.size() == impl.metadata.diagnostic_state_bytes,
      "checkpoint diagnostic-state size differs from its metadata",
      "checkpoint-diagnostics-validate",
      "checkpoint diagnostic-state validation failed", -35, [&] {
        (void)decode_checkpoint_diagnostic_state(encoded_state);
      });
  std::string descriptor;
  collective_try_with_fallback(
      *impl.runtime, impl.epoch, "checkpoint-diagnostics-agreement",
      "checkpoint diagnostic descriptor allocation failed", -41, [&] {
        std::ostringstream stream;
        stream << encoded_state.size() << ':'
               << diagnostic_state_hash(encoded_state);
        descriptor = stream.str();
      });
  require_common_descriptor(*impl.runtime, impl.epoch, descriptor,
                            "checkpoint-diagnostics-agreement");

  const std::array<std::uint64_t, 1> shape{
      static_cast<std::uint64_t>(encoded_state.size())};
  const std::vector<DatasetHyperslab> slabs =
      prepare_root_diagnostic_selection(
          *impl.runtime, impl.epoch, shape.front(),
          "checkpoint-diagnostics-selection",
          "checkpoint diagnostic selection allocation failed", -42);
  const void* values =
      impl.runtime->rank() == 0 ? encoded_state.data() : nullptr;
  const std::size_t elements =
      impl.runtime->rank() == 0 ? encoded_state.size() : 0;
  write_dataset("diagnostics/state", CheckpointValueType::uint8, shape,
                slabs, values, elements);
}

void ParallelCheckpointWriter::commit() {
  Impl& impl = *implementation_;
  if (impl.closed) {
    if (impl.committed) return;
    throw std::logic_error{"checkpoint writer is already closed"};
  }
  bool success = H5Fflush(impl.file, H5F_SCOPE_GLOBAL) >= 0;
  collective_require(*impl.runtime, impl.epoch, success,
                     "checkpoint-flush", local_error_text("H5Fflush"));
  success = H5Fclose(impl.file) >= 0;
  impl.file = -1;
  impl.file_collectively_open = false;
  collective_require(*impl.runtime, impl.epoch, success,
                     "checkpoint-close", local_error_text("H5Fclose"));

  // This is the final throwing collective before the irreversible rename.
  // Once rank zero replaces the committed path, later notification failures
  // cannot truthfully turn the completed filesystem commit into a failed
  // checkpoint without violating the previous-file guarantee.
  collective_require(*impl.runtime, impl.epoch, true,
                     "checkpoint-commit-ready", {});
  const RootPublicationNotification notification = publish_from_root(
      *impl.runtime, [&impl]() noexcept {
        if (distributed_test_failure_enabled(
                "QUASAR_TEST_CHECKPOINT_RENAME_FAILURE")) {
          return EIO;
        }
        errno = 0;
        if (std::rename(impl.temporary_path.c_str(),
                        impl.committed_path.c_str()) != 0) {
          return errno == 0 ? EIO : errno;
        }
        return 0;
      });
  if (notification.mpi_status != MPI_SUCCESS) {
    // Process-failure recovery is outside the runtime contract.  Keep this
    // necessary post-rename notification nonthrowing: a successful rename is
    // already committed, while a failed notification cannot restore the old
    // file or support another safe collective.
    impl.committed = notification.publication_error == 0;
    impl.closed = true;
    return;
  }
  if (notification.publication_error != 0) {
    collective_require(*impl.runtime, impl.epoch, false,
                       "checkpoint-commit",
                       "failed to atomically replace the committed checkpoint",
                       notification.publication_error);
  }
  impl.committed = true;
  impl.closed = true;
}

void ParallelCheckpointWriter::close() {
  Impl& impl = *implementation_;
  if (impl.closed) return;
  bool success = true;
  if (impl.file >= 0) {
    success = H5Fclose(impl.file) >= 0;
    impl.file = -1;
    impl.file_collectively_open = false;
  }
  collective_require(*impl.runtime, impl.epoch, success,
                     "checkpoint-close", local_error_text("H5Fclose"));
  impl.runtime->barrier();
  int remove_error = 0;
  if (impl.runtime->rank() == 0
      && std::remove(impl.temporary_path.c_str()) != 0 && errno != ENOENT) {
    remove_error = errno == 0 ? EIO : errno;
  }
  check_mpi(MPI_Bcast(
                &remove_error, 1, MPI_INT, 0,
                detail::MpiRuntimeNativeAccess::world(*impl.runtime)),
            "MPI_Bcast(checkpoint cleanup status)");
  collective_require(*impl.runtime, impl.epoch, remove_error == 0,
                     "checkpoint-cleanup",
                     remove_error == 0
                         ? std::string_view{}
                         : std::string_view{"failed to remove temporary checkpoint"},
                     remove_error);
  impl.closed = true;
}

ParallelCheckpointReader::ParallelCheckpointReader(
    MpiRuntime& runtime, std::filesystem::path path)
  : implementation_{new (std::nothrow) Impl{}} {
  std::uint64_t allocation_epoch = 0;
  try {
    collective_require(
        runtime, allocation_epoch, implementation_ != nullptr,
        "restart-object-allocation",
        "checkpoint reader allocation failed", -45);
  } catch (...) {
    delete implementation_;
    implementation_ = nullptr;
    throw;
  }
  Impl& impl = *implementation_;
  impl.epoch = allocation_epoch;
  impl.runtime = &runtime;
  impl.path = std::move(path);
  try {
    std::string local_path;
    collective_try(runtime, impl.epoch, "restart-open",
                   "restart path preparation failed", -8, [&] {
      if (impl.path.empty()) {
        throw std::invalid_argument{"restart path must not be empty"};
      }
      local_path = impl.path.string();
    });
    collective_require_common_string(
        runtime, impl.epoch, local_path, "restart-open",
        checkpoint_string_broadcast,
        "ranks supplied different or empty restart paths", -8);
    Hdf5Handle access =
        create_parallel_access(runtime, impl.epoch, "restart-open");
    impl.file = H5Fopen(impl.path.c_str(), H5F_ACC_RDONLY, access.get());
    collective_require(runtime, impl.epoch, impl.file >= 0, "restart-open",
                       local_error_text("H5Fopen"));
    impl.file_collectively_open = true;
    const bool access_closed = access.close() >= 0;
    collective_require(runtime, impl.epoch, access_closed,
                       "restart-open-close", local_error_text("H5Pclose"));

    collective_try(runtime, impl.epoch, "restart-metadata",
                   "restart metadata validation failed", -9, [&] {
      impl.metadata = read_metadata(runtime, impl.file, impl.epoch);
      validate_checkpoint_metadata(impl.metadata);
    });
    require_common_serialized_metadata(
        runtime, impl.epoch, impl.metadata, "restart-metadata",
        "restart metadata serialization failed", -51,
        "ranks read different checkpoint metadata", -10);
  } catch (...) {
    // Do not let error-path cleanup make a subset of ranks enter collective
    // parallel-file close after an asymmetric H5Fopen failure.
    if (impl.file_collectively_open) (void)H5Fclose(impl.file);
    delete implementation_;
    implementation_ = nullptr;
    throw;
  }
}

ParallelCheckpointReader::~ParallelCheckpointReader() noexcept {
  delete implementation_;
}

const CheckpointMetadata& ParallelCheckpointReader::metadata() const noexcept {
  return implementation_->metadata;
}

void ParallelCheckpointReader::read_dataset(
    std::string_view name, CheckpointValueType value_type,
    std::span<const std::uint64_t> expected_global_shape,
    std::span<const DatasetHyperslab> local_slabs,
    void* packed_values, std::size_t packed_elements) {
  Impl& impl = *implementation_;
  if (impl.closed || impl.file < 0) {
    throw std::logic_error{"checkpoint reader is closed"};
  }
  const PreparedDatasetOperation operation = prepare_dataset_operation(
      *impl.runtime, impl.epoch, name, value_type, expected_global_shape,
      local_slabs, packed_values, packed_elements,
      "restart-dataset-validate", "restart dataset validation failed", -11,
      "restart-dataset-order", "restart-dataset-partition");
  Hdf5Handle dataset{
      H5Dopen2(impl.file, operation.name.c_str(), H5P_DEFAULT), H5Dclose};
  collective_require_collective_handle(
      *impl.runtime, impl.epoch, dataset.valid(), "restart-dataset-open",
      local_error_text("H5Dopen2"), dataset);
  Hdf5Handle file_space{H5Dget_space(dataset.get()), H5Sclose};
  Hdf5Handle stored_type{H5Dget_type(dataset.get()), H5Tclose};
  bool success = file_space.valid() && stored_type.valid();
  collective_require(*impl.runtime, impl.epoch, success,
                     "restart-dataset-setup",
                     local_error_text("H5Dget_space/H5Dget_type"));
  collective_require_predicate(
      *impl.runtime, impl.epoch, "restart-dataset-open",
      "checkpoint dataset type or global shape is incompatible",
      "checkpoint dataset shape allocation failed", -12, [&] {
        return dataset_is_compatible(file_space.get(), stored_type.get(),
                                     operation.types, expected_global_shape);
      });
  success = stored_type.close() >= 0;
  collective_require(*impl.runtime, impl.epoch, success,
                     "restart-dataset-setup-close",
                     local_error_text("H5Tclose"));

  select_dataset_hyperslabs(
      *impl.runtime, impl.epoch, file_space, local_slabs,
      "restart-dataset-select", "restart dataset selection failed", -13);
  DatasetTransferHandles transfer = prepare_dataset_transfer(
      *impl.runtime, impl.epoch, packed_elements, "restart-dataset-select");
  std::byte dummy{};
  void* values = packed_elements == 0 ? &dummy : packed_values;
  success = H5Dread(dataset.get(), operation.types.memory,
                    transfer.memory_space.get(), file_space.get(),
                    transfer.transfer.get(), values) >= 0;
  collective_require(*impl.runtime, impl.epoch, success,
                     "restart-dataset-read", local_error_text("H5Dread"));
  close_dataset_handles(
      *impl.runtime, impl.epoch, transfer, file_space, dataset,
      "restart-dataset-close", DatasetCloseOrder::space_before_dataset);
}

std::vector<std::vector<std::uint8_t>>
ParallelCheckpointReader::read_diagnostic_state() {
  Impl& impl = *implementation_;
  if (impl.metadata.diagnostic_state_bytes == 0) return {};
  const std::uint64_t byte_count = impl.metadata.diagnostic_state_bytes;
  std::vector<std::uint8_t> encoded;
  collective_try_with_fallback(
      *impl.runtime, impl.epoch, "restart-diagnostics-allocation",
      "checkpoint diagnostic-state allocation failed", -36, [&] {
        if (impl.runtime->rank() == 0) {
          encoded.resize(static_cast<std::size_t>(byte_count));
        }
      });

  const std::array<std::uint64_t, 1> shape{byte_count};
  const std::vector<DatasetHyperslab> slabs =
      prepare_root_diagnostic_selection(
          *impl.runtime, impl.epoch, byte_count,
          "restart-diagnostics-selection",
          "checkpoint diagnostic selection allocation failed", -43);
  void* values = impl.runtime->rank() == 0 ? encoded.data() : nullptr;
  const std::size_t elements =
      impl.runtime->rank() == 0 ? encoded.size() : 0;
  read_dataset("diagnostics/state", CheckpointValueType::uint8, shape,
               slabs, values, elements);

  collective_try_with_fallback(
      *impl.runtime, impl.epoch, "restart-diagnostics-allocation",
      "checkpoint diagnostic-state allocation failed", -37, [&] {
        if (impl.runtime->rank() != 0) {
          encoded.resize(static_cast<std::size_t>(byte_count));
        }
      });
  check_mpi(MPI_Bcast(encoded.data(), static_cast<int>(encoded.size()),
                      MPI_UINT8_T, 0,
                      detail::MpiRuntimeNativeAccess::world(*impl.runtime)),
            "MPI_Bcast(checkpoint diagnostic state)");

  std::vector<std::vector<std::uint8_t>> parts;
  collective_try(
      *impl.runtime, impl.epoch, "restart-diagnostics-validate",
      "checkpoint diagnostic-state decoding failed", -38, [&] {
        parts = decode_checkpoint_diagnostic_state(encoded);
  });
  return parts;
}

void ParallelCheckpointReader::close() {
  Impl& impl = *implementation_;
  if (impl.closed) return;
  const bool success = impl.file < 0 || H5Fclose(impl.file) >= 0;
  impl.file = -1;
  impl.file_collectively_open = false;
  collective_require(*impl.runtime, impl.epoch, success,
                     "restart-close", local_error_text("H5Fclose"));
  impl.closed = true;
}

}  // namespace quasar::distributed
