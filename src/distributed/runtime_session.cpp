#include "quasar/distributed/runtime_session.hpp"

#include "quasar/distributed/collective_error.hpp"
#include "quasar/distributed/mpi_device_mapping.hpp"
#include "quasar/distributed/mpi_runtime.hpp"

#include "fixed_message.hpp"
#include "mpi_runtime_native.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace quasar::distributed {
namespace {

enum class RuntimeSessionState : std::uint8_t {
  inactive,
  active,
  abandoned,
};

std::atomic<RuntimeSessionState>& runtime_session_state() {
  static std::atomic<RuntimeSessionState> state{RuntimeSessionState::inactive};
  return state;
}

template <class T>
void append_wire(std::vector<std::byte>& bytes, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (bytes.size() > std::numeric_limits<std::size_t>::max() - sizeof(T)) {
    throw std::length_error{"endpoint mapping record is too large"};
  }
  const std::size_t offset = bytes.size();
  bytes.resize(offset + sizeof(T));
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

void append_wire_string(std::vector<std::byte>& bytes,
                        const std::string& value) {
  append_wire(bytes, static_cast<std::uint64_t>(value.size()));
  if (value.size() > std::numeric_limits<std::size_t>::max() - bytes.size()) {
    throw std::length_error{"endpoint mapping record is too large"};
  }
  const std::size_t offset = bytes.size();
  bytes.resize(offset + value.size());
  if (!value.empty()) {
    std::memcpy(bytes.data() + offset, value.data(), value.size());
  }
}

template <class T>
T read_wire(std::span<const std::byte> bytes, std::size_t& cursor) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (cursor > bytes.size() || sizeof(T) > bytes.size() - cursor) {
    throw std::runtime_error{"truncated endpoint mapping record"};
  }
  T value{};
  std::memcpy(&value, bytes.data() + cursor, sizeof(T));
  cursor += sizeof(T);
  return value;
}

std::string read_wire_string(std::span<const std::byte> bytes,
                             std::size_t& cursor) {
  const std::uint64_t length = read_wire<std::uint64_t>(bytes, cursor);
  if (length > bytes.size() - cursor) {
    throw std::runtime_error{"truncated endpoint mapping string"};
  }
  std::string result{
      reinterpret_cast<const char*>(bytes.data() + cursor),
      static_cast<std::size_t>(length)};
  cursor += static_cast<std::size_t>(length);
  return result;
}

std::vector<std::byte> serialize_assignment(
    const RankDeviceAssignment& assignment, int node_leader) {
  std::vector<std::byte> bytes;
  append_wire(bytes, static_cast<std::int32_t>(assignment.world_rank));
  append_wire(bytes, static_cast<std::int32_t>(assignment.node_local_rank));
  append_wire(bytes, static_cast<std::int32_t>(node_leader));
  append_wire(bytes,
              static_cast<std::uint64_t>(assignment.owned_devices.size()));
  for (const auto& device : assignment.owned_devices) {
    append_wire(bytes, static_cast<std::int32_t>(device.ordinal));
    append_wire_string(bytes, device.uuid);
    append_wire_string(bytes, device.pci_bus_id);
  }
  return bytes;
}

RankDeviceAssignment deserialize_assignment(std::span<const std::byte> bytes) {
  std::size_t cursor = 0;
  const auto world_rank = read_wire<std::int32_t>(bytes, cursor);
  const auto node_rank = read_wire<std::int32_t>(bytes, cursor);
  const auto node_leader = read_wire<std::int32_t>(bytes, cursor);
  const std::uint64_t device_count = read_wire<std::uint64_t>(bytes, cursor);
  if (device_count > bytes.size() - cursor) {
    throw std::runtime_error{"invalid endpoint mapping device count"};
  }
  std::vector<DeviceIdentity> devices;
  devices.reserve(static_cast<std::size_t>(device_count));
  for (std::uint64_t index = 0; index < device_count; ++index) {
    const auto ordinal = read_wire<std::int32_t>(bytes, cursor);
    devices.push_back(DeviceIdentity{
        ordinal, read_wire_string(bytes, cursor),
        read_wire_string(bytes, cursor)});
  }
  if (cursor != bytes.size()) {
    throw std::runtime_error{"endpoint mapping record has trailing bytes"};
  }
  return {world_rank, node_rank,
          "mpi-shared-" + std::to_string(node_leader), std::move(devices)};
}

}  // namespace

class RuntimeSession::Impl {
 public:
  Impl() {
    RuntimeSessionState expected = RuntimeSessionState::inactive;
    if (!runtime_session_state().compare_exchange_strong(
            expected, RuntimeSessionState::active)) {
      if (expected == RuntimeSessionState::abandoned) {
        throw std::logic_error{
            "a prior RuntimeSession was destroyed without collective close; "
            "the process MPI state cannot be safely reused"};
      }
      throw std::logic_error{
          "only one live RuntimeSession is permitted in a process"};
    }
    try {
      runtime_ = std::make_unique<MpiRuntime>();
      registered_ = true;
    } catch (...) {
      runtime_session_state().store(RuntimeSessionState::inactive);
      throw;
    }
  }

  ~Impl() noexcept {
    // Destruction must not initiate MPI work.  Keep an abandoned registration
    // sticky so a later object cannot reuse communicators or MPI state that the
    // caller failed to tear down collectively.
    if (registered_) {
      runtime_session_state().store(RuntimeSessionState::abandoned);
    }
  }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;
  Impl(Impl&&) = delete;
  Impl& operator=(Impl&&) = delete;

  [[nodiscard]] int rank() const noexcept { return runtime_->rank(); }
  [[nodiscard]] int size() const noexcept { return runtime_->size(); }
  [[nodiscard]] int node_rank() const noexcept { return runtime_->node_rank(); }
  [[nodiscard]] int node_size() const noexcept { return runtime_->node_size(); }
  [[nodiscard]] int thread_level() const noexcept {
    return runtime_->thread_level();
  }
  [[nodiscard]] bool owns_mpi() const {
    const std::scoped_lock lock{state_mutex_};
    return runtime_->owns_mpi();
  }
  [[nodiscard]] bool closed() const {
    const std::scoped_lock lock{state_mutex_};
    return runtime_->closed();
  }

  [[nodiscard]] EndpointMapping endpoint_mapping() const {
    const std::scoped_lock lock{state_mutex_};
    return endpoint_mapping_;
  }

  [[nodiscard]] std::optional<VirtualTopology> topology() const {
    const std::scoped_lock lock{state_mutex_};
    return topology_;
  }

  [[nodiscard]] SessionTelemetrySnapshot telemetry() const {
    const std::scoped_lock lock{state_mutex_};
    return {telemetry_,
            mhd_runtime_ ? std::optional{mhd_runtime_->telemetry()}
                         : std::nullopt,
            mhd_runtime_
                ? std::optional{mhd_runtime_->transport_resolution()}
                : std::nullopt,
            pic_runtime_ ? std::optional{pic_runtime_->telemetry()}
                         : std::nullopt,
            pic_runtime_
                ? std::optional{pic_runtime_->transport_resolution()}
                : std::nullopt,
            endpoint_mapping_.size(), endpoint_mapping_.devices_per_rank()};
  }

  void barrier() {
    const std::scoped_lock lock{state_mutex_};
    runtime_->barrier();
    ++telemetry_.barriers;
  }

  void inject_candidate_cleanup_failure_for_testing(bool enabled) {
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
    const std::scoped_lock lock{state_mutex_};
    inject_candidate_cleanup_failure_ = enabled;
#else
    (void)enabled;
#endif
  }

  void inject_mhd_close_failure_for_testing(bool enabled) {
    const std::scoped_lock lock{state_mutex_};
    if (!mhd_runtime_) {
      throw std::logic_error{
          "cannot inject an MHD close failure without an active MHD runtime"};
    }
    mhd_runtime_->inject_next_worker_task_allocation_failure_for_testing(
        enabled);
  }

  void consensus(bool success, std::string_view phase,
                 std::string_view message) {
    const std::scoped_lock lock{state_mutex_};
    collective_require(success, phase, message);
  }

  void require_same_string(std::string_view value, std::string_view phase,
                           std::string_view message) {
    const std::scoped_lock lock{state_mutex_};
    const bool local_valid =
        value.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max());
    collective_require(local_valid, phase,
                       "collective argument exceeds the MPI count range");
    const int local_size = static_cast<int>(value.size());
    std::vector<int> sizes;
    bool storage_valid = true;
    try {
      sizes.resize(static_cast<std::size_t>(size()));
    } catch (...) {
      storage_valid = false;
    }
    collective_require(storage_valid, phase,
                       "collective argument size storage allocation failed");
    int status = MPI_Allgather(&local_size, 1, MPI_INT, sizes.data(), 1,
                               MPI_INT, detail::MpiRuntimeNativeAccess::world(*runtime_));
    collective_require(
        status == MPI_SUCCESS, phase,
        "MPI_Allgather of collective argument sizes failed");
    std::vector<int> offsets;
    std::int64_t total = 0;
    bool sizes_valid = true;
    try {
      offsets.resize(sizes.size());
      for (std::size_t index = 0; index < sizes.size(); ++index) {
        if (sizes[index] < 0 ||
            total > std::numeric_limits<int>::max() - sizes[index]) {
          sizes_valid = false;
          break;
        }
        offsets[index] = static_cast<int>(total);
        total += sizes[index];
      }
    } catch (...) {
      sizes_valid = false;
    }
    collective_require(sizes_valid, phase,
                       "collective arguments are too large or offset storage allocation failed");
    std::vector<char> gathered;
    storage_valid = true;
    try {
      gathered.resize(static_cast<std::size_t>(total));
    } catch (...) {
      storage_valid = false;
    }
    collective_require(storage_valid, phase,
                       "collective argument payload allocation failed");
    char dummy{};
    status = MPI_Allgatherv(
        value.data(), local_size, MPI_CHAR,
        gathered.empty() ? &dummy : gathered.data(), sizes.data(),
        offsets.data(), MPI_CHAR, detail::MpiRuntimeNativeAccess::world(*runtime_));
    collective_require(
        status == MPI_SUCCESS, phase,
        "MPI_Allgatherv of collective arguments failed");
    bool identical = true;
    for (std::size_t index = 0; index < sizes.size(); ++index) {
      identical = identical && sizes[index] == local_size &&
          std::equal(value.begin(), value.end(),
                     gathered.begin() + offsets[index]);
    }
    collective_require(identical, phase, message);
  }

  void configure_owned_devices(std::vector<DeviceIdentity> local_devices,
                               std::string_view parse_error) {
    const std::scoped_lock lock{state_mutex_};
    collective_require(parse_error.empty(), "python-endpoint-input",
                       parse_error);
    collective_require(
        !mhd_runtime_ && !pic_runtime_, "python-endpoint-runtime",
        "owned devices cannot be reconfigured while a physics runtime is active");

    int node_leader = rank();
    const int leader_status = MPI_Allreduce(
        MPI_IN_PLACE, &node_leader, 1, MPI_INT, MPI_MIN,
        detail::MpiRuntimeNativeAccess::shared_memory(*runtime_));
    collective_require(
        leader_status == MPI_SUCCESS, "python-endpoint-node",
        "MPI_Allreduce of the node leader failed");

    std::vector<std::byte> local_record;
    const CollectiveLocalError serialization_error =
        capture_collective_local_error([&] {
          local_record = serialize_assignment(
              RankDeviceAssignment{
                  rank(), node_rank(),
                  "mpi-shared-" + std::to_string(node_leader),
                  std::move(local_devices)},
              node_leader);
          if (local_record.size()
              > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::length_error{
                "one rank's endpoint mapping exceeds the MPI count range"};
          }
        });
    collective_require(serialization_error.ok(), "python-endpoint-encode",
                       serialization_error.message());

    const int local_bytes = static_cast<int>(local_record.size());
    std::vector<int> counts;
    bool gather_storage_valid = true;
    try {
      counts.resize(static_cast<std::size_t>(size()));
    } catch (...) {
      gather_storage_valid = false;
    }
    collective_require(gather_storage_valid, "python-endpoint-size-storage",
                       "endpoint size storage allocation failed");
    int status = MPI_Allgather(
        &local_bytes, 1, MPI_INT, counts.data(), 1, MPI_INT,
        detail::MpiRuntimeNativeAccess::world(*runtime_));
    collective_require(
        status == MPI_SUCCESS, "python-endpoint-size-gather",
        "MPI_Allgather of endpoint sizes failed");

    std::vector<int> offsets;
    std::int64_t total_bytes = 0;
    bool counts_valid = true;
    try {
      offsets.resize(counts.size());
      for (std::size_t index = 0; index < counts.size(); ++index) {
        if (counts[index] < 0
            || total_bytes > std::numeric_limits<int>::max() - counts[index]) {
          counts_valid = false;
          break;
        }
        offsets[index] = static_cast<int>(total_bytes);
        total_bytes += counts[index];
      }
    } catch (...) {
      counts_valid = false;
    }
    collective_require(counts_valid, "python-endpoint-size-validate",
                       "global endpoint mapping is too large or offset storage allocation failed");

    std::vector<std::byte> gathered;
    gather_storage_valid = true;
    try {
      gathered.resize(static_cast<std::size_t>(total_bytes));
    } catch (...) {
      gather_storage_valid = false;
    }
    collective_require(gather_storage_valid,
                       "python-endpoint-payload-storage",
                       "endpoint payload storage allocation failed");
    std::byte dummy{};
    status = MPI_Allgatherv(
        local_record.data(), local_bytes, MPI_BYTE,
        gathered.empty() ? &dummy : gathered.data(), counts.data(),
        offsets.data(), MPI_BYTE,
        detail::MpiRuntimeNativeAccess::world(*runtime_));
    collective_require(
        status == MPI_SUCCESS, "python-endpoint-gather",
        "MPI_Allgatherv of endpoint records failed");

    std::vector<RankDeviceAssignment> assignments;
    EndpointMapping candidate;
    const CollectiveLocalError mapping_error =
        capture_collective_local_error([&] {
          assignments.reserve(counts.size());
          for (std::size_t index = 0; index < counts.size(); ++index) {
            assignments.push_back(deserialize_assignment(
                std::span<const std::byte>{gathered}.subspan(
                    static_cast<std::size_t>(offsets[index]),
                    static_cast<std::size_t>(counts[index]))));
          }
          candidate = make_endpoint_mapping(assignments);
        });
    collective_require(mapping_error.ok(), "python-endpoint-validate",
                       mapping_error.message());
    endpoint_mapping_ = std::move(candidate);
    topology_.reset();
    ++telemetry_.endpoint_configurations;
  }

  void configure_devices(std::vector<int> eligible_ordinals,
                         std::string_view parse_error) {
    const std::scoped_lock lock{state_mutex_};
    collective_require(parse_error.empty(), "python-device-input",
                       parse_error);
    collective_require(!mhd_runtime_ && !pic_runtime_, "python-device-runtime",
                       "devices cannot be reconfigured while a physics runtime is active");
    endpoint_mapping_ = discover_endpoint_mapping(*runtime_, eligible_ordinals);
    topology_.reset();
    ++telemetry_.endpoint_configurations;
  }

  void select_topology(std::size_t global_nx, std::size_t global_ny,
                       std::optional<DecompositionShape> shape,
                       std::size_t minimum_tile_width,
                       std::string_view parse_error) {
    const std::scoped_lock lock{state_mutex_};
    collective_require(parse_error.empty(), "python-topology-input",
                       parse_error);
    collective_require(
        !mhd_runtime_ && !pic_runtime_, "python-topology-runtime",
        "topology cannot be reconfigured while a physics runtime is active");
    collective_require(!endpoint_mapping_.empty(), "python-topology-mapping",
                       "configure_owned_devices must be called first");

    const std::array<std::uint64_t, 6> local_descriptor{
        static_cast<std::uint64_t>(global_nx),
        static_cast<std::uint64_t>(global_ny),
        static_cast<std::uint64_t>(minimum_tile_width),
        shape ? 1ULL : 0ULL,
        shape ? static_cast<std::uint64_t>(shape->px) : 0ULL,
        shape ? static_cast<std::uint64_t>(shape->py) : 0ULL};
    std::vector<std::uint64_t> descriptors;
    bool descriptor_storage_valid = true;
    try {
      descriptors.resize(
          static_cast<std::size_t>(size()) * local_descriptor.size());
    } catch (...) {
      descriptor_storage_valid = false;
    }
    collective_require(descriptor_storage_valid,
                       "python-topology-agreement-storage",
                       "topology agreement storage allocation failed");
    const int status = MPI_Allgather(
        local_descriptor.data(), static_cast<int>(local_descriptor.size()),
        MPI_UINT64_T, descriptors.data(),
        static_cast<int>(local_descriptor.size()), MPI_UINT64_T,
        detail::MpiRuntimeNativeAccess::world(*runtime_));
    collective_require(
        status == MPI_SUCCESS, "python-topology-agreement",
        "MPI_Allgather of topology descriptors failed");
    bool arguments_match = true;
    for (int participant = 0; participant < size(); ++participant) {
      const auto begin = descriptors.begin()
          + static_cast<std::ptrdiff_t>(participant)
                * static_cast<std::ptrdiff_t>(local_descriptor.size());
      arguments_match = arguments_match
          && std::equal(local_descriptor.begin(), local_descriptor.end(), begin);
    }
    collective_require(
        arguments_match, "python-topology-agreement",
        "ranks supplied different topology arguments");

    std::optional<VirtualTopology> candidate;
    const CollectiveLocalError topology_error =
        capture_collective_local_error([&] {
          candidate = shape
              ? VirtualTopology::create(
                    global_nx, global_ny,
                    endpoint_mapping_.size(), *shape,
                    minimum_tile_width)
              : VirtualTopology::create_auto(
                    global_nx, global_ny,
                    endpoint_mapping_.size(), minimum_tile_width);
        });
    collective_require(topology_error.ok(), "python-topology-create",
                       topology_error.message());
    topology_ = std::move(candidate);
    ++telemetry_.topology_selections;
  }

  void start_mhd(mhd::MhdConfig config, MhdGlobalState state,
                 std::optional<MhdGlobalBackground> background,
                 TransportPolicy transport_policy,
                 std::string_view parse_error) {
    const std::scoped_lock lock{state_mutex_};
    collective_require(parse_error.empty(), "python-mhd-input", parse_error);
    collective_require_transport_policy(
        transport_policy, "python-mhd-transport");
    collective_require(!mhd_runtime_ && !pic_runtime_, "python-mhd-start",
                       "a physics runtime is already active");
    collective_require(!endpoint_mapping_.empty() && topology_.has_value(),
                       "python-mhd-start",
                       "device mapping and topology must be selected first");
    mhd_runtime_ = construct_candidate<MhdTileRuntime>(
        "python-mhd-runtime-storage",
        *runtime_, endpoint_mapping_, *topology_, std::move(config),
        transport_policy);
    try {
      mhd_runtime_->seed(state, background ? &*background : nullptr);
    } catch (...) {
      close_failed_candidate(mhd_runtime_, std::current_exception());
    }
  }

  [[nodiscard]] CheckpointMetadata restart_mhd(
      mhd::MhdConfig config, const std::string& path,
      const std::string& unit_system,
      const std::optional<MhdGlobalBackground>& expected_background,
      TransportPolicy transport_policy,
      std::string_view parse_error,
      std::vector<std::vector<std::uint8_t>>& diagnostic_state) {
    const std::scoped_lock lock{state_mutex_};
    collective_require(parse_error.empty(), "python-mhd-restart-input",
                       parse_error);
    collective_require_transport_policy(
        transport_policy, "python-mhd-restart-transport");
    collective_require(!mhd_runtime_ && !pic_runtime_, "python-mhd-restart",
                       "a physics runtime is already active");
    collective_require(!endpoint_mapping_.empty() && topology_.has_value(),
                       "python-mhd-restart",
                       "device mapping and topology must be selected first");
    mhd_runtime_ = construct_candidate<MhdTileRuntime>(
        "python-mhd-restart-runtime-storage",
        *runtime_, endpoint_mapping_, *topology_, std::move(config),
        transport_policy);
    CheckpointMetadata metadata;
    try {
      metadata = mhd_runtime_->restart_from_checkpoint(
          path, unit_system,
          expected_background ? &*expected_background : nullptr,
          &diagnostic_state);
    } catch (...) {
      close_failed_candidate(mhd_runtime_, std::current_exception());
    }
    return metadata;
  }

  [[nodiscard]] Real mhd_cfl_limit() {
    const std::scoped_lock lock{state_mutex_};
    if (!mhd_runtime_) throw std::logic_error{"no active distributed MHD runtime"};
    return mhd_runtime_->cfl_limit();
  }

  void mhd_step(Real dt, bool check_cfl) {
    const std::scoped_lock lock{state_mutex_};
    if (!mhd_runtime_) throw std::logic_error{"no active distributed MHD runtime"};
    mhd_runtime_->step(dt, check_cfl);
  }

  [[nodiscard]] Real mhd_divergence_b_max() {
    const std::scoped_lock lock{state_mutex_};
    if (!mhd_runtime_) throw std::logic_error{"no active distributed MHD runtime"};
    return mhd_runtime_->divergence_b_max();
  }

  [[nodiscard]] MhdGlobalState mhd_gather_state() {
    const std::scoped_lock lock{state_mutex_};
    if (!mhd_runtime_) throw std::logic_error{"no active distributed MHD runtime"};
    return mhd_runtime_->gather_state();
  }

  [[nodiscard]] std::vector<Real> mhd_gather_cell_component(
      std::string_view component) {
    const std::scoped_lock lock{state_mutex_};
    if (!mhd_runtime_) throw std::logic_error{"no active distributed MHD runtime"};
    return mhd_runtime_->gather_cell_component(component);
  }

  [[nodiscard]] std::vector<MhdOwnedShard> mhd_local_owned_shards() {
    const std::scoped_lock lock{state_mutex_};
    if (!mhd_runtime_) throw std::logic_error{"no active distributed MHD runtime"};
    return mhd_runtime_->local_owned_shards();
  }

  [[nodiscard]] MhdGlobalCellSums mhd_global_cell_sums() {
    const std::scoped_lock lock{state_mutex_};
    if (!mhd_runtime_) throw std::logic_error{"no active distributed MHD runtime"};
    return mhd_runtime_->global_cell_sums();
  }

  void mhd_write_checkpoint(const std::string& path, std::uint64_t step,
                            double time, const std::string& unit_system,
                            std::span<const std::uint8_t> diagnostic_state) {
    const std::scoped_lock lock{state_mutex_};
    if (!mhd_runtime_) throw std::logic_error{"no active distributed MHD runtime"};
    mhd_runtime_->write_checkpoint(
        path, step, time, unit_system, diagnostic_state);
  }

  void close_mhd() {
    const std::scoped_lock lock{state_mutex_};
    if (!mhd_runtime_) return;
    mhd_runtime_->close();
    mhd_runtime_.reset();
  }

  void start_pic(pic::EmPicConfig config, PicGlobalFields fields,
                 std::optional<PicGlobalFields> external_fields,
                 std::vector<PicSpeciesState> species,
                 TransportPolicy transport_policy,
                 std::string_view parse_error) {
    const std::scoped_lock lock{state_mutex_};
    collective_require(parse_error.empty(), "python-pic-input", parse_error);
    collective_require_transport_policy(
        transport_policy, "python-pic-transport");
    collective_require(!mhd_runtime_ && !pic_runtime_, "python-pic-start",
                       "a physics runtime is already active");
    collective_require(!endpoint_mapping_.empty() && topology_.has_value(),
                       "python-pic-start",
                       "device mapping and topology must be selected first");
    pic_runtime_ = construct_candidate<PicTileRuntime>(
        "python-pic-runtime-storage",
        *runtime_, endpoint_mapping_, *topology_, std::move(config),
        transport_policy);
    try {
      pic_runtime_->seed(fields, external_fields ? &*external_fields : nullptr,
                         species);
    } catch (...) {
      close_failed_candidate(pic_runtime_, std::current_exception());
    }
  }

  [[nodiscard]] CheckpointMetadata restart_pic(
      pic::EmPicConfig config, const std::string& path,
      const std::string& unit_system,
      const std::vector<pic::SpeciesConfig>& expected_species,
      TransportPolicy transport_policy,
      std::string_view parse_error,
      std::vector<std::vector<std::uint8_t>>& diagnostic_state) {
    const std::scoped_lock lock{state_mutex_};
    collective_require(parse_error.empty(), "python-pic-restart-input",
                       parse_error);
    collective_require_transport_policy(
        transport_policy, "python-pic-restart-transport");
    collective_require(!mhd_runtime_ && !pic_runtime_, "python-pic-restart",
                       "a physics runtime is already active");
    collective_require(!endpoint_mapping_.empty() && topology_.has_value(),
                       "python-pic-restart",
                       "device mapping and topology must be selected first");
    pic_runtime_ = construct_candidate<PicTileRuntime>(
        "python-pic-restart-runtime-storage",
        *runtime_, endpoint_mapping_, *topology_, std::move(config),
        transport_policy);
    CheckpointMetadata metadata;
    try {
      metadata = pic_runtime_->restart_from_checkpoint(
          path, unit_system, expected_species, &diagnostic_state);
    } catch (...) {
      close_failed_candidate(pic_runtime_, std::current_exception());
    }
    return metadata;
  }

  [[nodiscard]] Real pic_cfl_limit() {
    const std::scoped_lock lock{state_mutex_};
    if (!pic_runtime_) throw std::logic_error{"no active distributed PIC runtime"};
    return pic_runtime_->cfl_limit();
  }

  void pic_sample_external_fields(
      numerics::IFieldEvaluator& evaluator,
      const core::IFieldSource& source, Real length_scale,
      Real e_field_scale, Real b_field_scale) {
    const std::scoped_lock lock{state_mutex_};
    if (!pic_runtime_) throw std::logic_error{"no active distributed PIC runtime"};
    pic_runtime_->sample_external_fields(
        evaluator, source, length_scale, e_field_scale, b_field_scale);
  }

  void pic_step(Real dt) {
    const std::scoped_lock lock{state_mutex_};
    if (!pic_runtime_) throw std::logic_error{"no active distributed PIC runtime"};
    pic_runtime_->step(dt);
  }

  [[nodiscard]] PicGlobalState pic_gather_state() {
    const std::scoped_lock lock{state_mutex_};
    if (!pic_runtime_) throw std::logic_error{"no active distributed PIC runtime"};
    return pic_runtime_->gather_state();
  }

  [[nodiscard]] std::vector<PicOwnedShard> pic_local_owned_shards(
      bool include_particles) {
    const std::scoped_lock lock{state_mutex_};
    if (!pic_runtime_) throw std::logic_error{"no active distributed PIC runtime"};
    return pic_runtime_->local_owned_shards(include_particles);
  }

  [[nodiscard]] std::vector<std::uint64_t> pic_alive_counts() {
    const std::scoped_lock lock{state_mutex_};
    if (!pic_runtime_) throw std::logic_error{"no active distributed PIC runtime"};
    return pic_runtime_->alive_counts();
  }

  [[nodiscard]] std::vector<Real> pic_kinetic_energies() {
    const std::scoped_lock lock{state_mutex_};
    if (!pic_runtime_) throw std::logic_error{"no active distributed PIC runtime"};
    return pic_runtime_->kinetic_energies();
  }

  [[nodiscard]] Real pic_total_em_energy() {
    const std::scoped_lock lock{state_mutex_};
    if (!pic_runtime_) throw std::logic_error{"no active distributed PIC runtime"};
    return pic_runtime_->total_em_energy();
  }

  [[nodiscard]] Real pic_gauss_residual() {
    const std::scoped_lock lock{state_mutex_};
    if (!pic_runtime_) throw std::logic_error{"no active distributed PIC runtime"};
    return pic_runtime_->gauss_residual();
  }

  void pic_write_checkpoint(const std::string& path, std::uint64_t step,
                            double time, const std::string& unit_system,
                            std::span<const std::uint8_t> diagnostic_state) {
    const std::scoped_lock lock{state_mutex_};
    if (!pic_runtime_) throw std::logic_error{"no active distributed PIC runtime"};
    pic_runtime_->write_checkpoint(
        path, step, time, unit_system, diagnostic_state);
  }

  void close_pic() {
    const std::scoped_lock lock{state_mutex_};
    if (!pic_runtime_) return;
    pic_runtime_->close();
    pic_runtime_.reset();
  }

  void close() {
    const std::scoped_lock lock{state_mutex_};
    std::exception_ptr failure;
    const auto remember_failure = [&failure] {
      if (!failure) failure = std::current_exception();
    };

    // Physics close and MPI close are separate teardown obligations. Preserve
    // the first exception for the caller, but do not let it skip MPI_Finalize
    // when this session owns MPI.
    try {
      if (mhd_runtime_) {
        mhd_runtime_->close();
        mhd_runtime_.reset();
      }
    } catch (...) {
      remember_failure();
    }
    try {
      if (pic_runtime_) {
        pic_runtime_->close();
        pic_runtime_.reset();
      }
    } catch (...) {
      remember_failure();
    }
    try {
      runtime_->close();
    } catch (...) {
      remember_failure();
    }
    if (runtime_->closed()) release_registration();
    if (failure) std::rethrow_exception(failure);
  }

 private:
  template <class Runtime>
  [[noreturn]] void close_failed_candidate(
      std::unique_ptr<Runtime>& candidate,
      std::exception_ptr original_failure) {
    // The candidate is published in the session slot before initialization.
    // If collective close itself fails, leave that ownership intact so the
    // caller can retry explicit collective close; unwinding a local unique_ptr
    // here would silently discard the only closeable handle.
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
    if (std::exchange(inject_candidate_cleanup_failure_, false)) {
      std::rethrow_exception(original_failure);
    }
#endif
    try {
      candidate->close();
    } catch (...) {
      std::rethrow_exception(original_failure);
    }
    candidate.reset();
    std::rethrow_exception(original_failure);
  }

  template <class Runtime, class... Arguments>
  std::unique_ptr<Runtime> construct_candidate(
      std::string_view phase, Arguments&&... arguments) {
    void* storage = ::operator new(sizeof(Runtime), std::nothrow);
    try {
      collective_require(
          storage != nullptr, phase,
          "physics runtime object allocation failed");
    } catch (...) {
      ::operator delete(storage);
      throw;
    }
    try {
      Runtime* runtime = ::new (storage) Runtime(
          std::forward<Arguments>(arguments)...);
      return std::unique_ptr<Runtime>{runtime};
    } catch (...) {
      ::operator delete(storage);
      throw;
    }
  }

  void collective_require_transport_policy(TransportPolicy policy,
                                           std::string_view phase) {
    const int local = static_cast<int>(policy);
    int minimum = 0;
    int maximum = 0;
    int status = MPI_Allreduce(&local, &minimum, 1, MPI_INT, MPI_MIN,
                               detail::MpiRuntimeNativeAccess::world(*runtime_));
    collective_require(
        status == MPI_SUCCESS, phase,
        "MPI_Allreduce of minimum transport policy failed");
    status = MPI_Allreduce(&local, &maximum, 1, MPI_INT, MPI_MAX,
                           detail::MpiRuntimeNativeAccess::world(*runtime_));
    collective_require(
        status == MPI_SUCCESS, phase,
        "MPI_Allreduce of maximum transport policy failed");
    collective_require(
        minimum == maximum, phase,
        "ranks supplied different transport policies");
  }

  void release_registration() noexcept {
    if (!registered_) return;
    registered_ = false;
    runtime_session_state().store(RuntimeSessionState::inactive);
  }

  void collective_require(bool success, std::string_view phase,
                          std::string_view message) {
    const auto record = success
        ? CollectiveErrorRecord::success(consensus_epoch_, rank(), phase)
        : CollectiveErrorRecord::failure(
              consensus_epoch_, rank(), -1, -1, phase, message);
    ++consensus_epoch_;
    runtime_->require_collective_success(record);
  }

  std::unique_ptr<MpiRuntime> runtime_{};
  std::unique_ptr<MhdTileRuntime> mhd_runtime_{};
  std::unique_ptr<PicTileRuntime> pic_runtime_{};
  EndpointMapping endpoint_mapping_{};
  std::optional<VirtualTopology> topology_{};
  SessionTelemetry telemetry_{};
  std::uint64_t consensus_epoch_{0};
  mutable std::mutex state_mutex_{};
  bool registered_{false};
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
  bool inject_candidate_cleanup_failure_{false};
#endif
};

RuntimeSession::RuntimeSession()
    : impl_{std::make_unique<Impl>()} {}

RuntimeSession::~RuntimeSession() noexcept = default;

int RuntimeSession::rank() const noexcept { return impl_->rank(); }
int RuntimeSession::size() const noexcept { return impl_->size(); }
int RuntimeSession::node_rank() const noexcept { return impl_->node_rank(); }
int RuntimeSession::node_size() const noexcept { return impl_->node_size(); }
int RuntimeSession::thread_level() const noexcept {
  return impl_->thread_level();
}
bool RuntimeSession::owns_mpi() const { return impl_->owns_mpi(); }
bool RuntimeSession::closed() const { return impl_->closed(); }

EndpointMapping RuntimeSession::endpoint_mapping() const {
  return impl_->endpoint_mapping();
}

std::optional<VirtualTopology> RuntimeSession::topology() const {
  return impl_->topology();
}

SessionTelemetrySnapshot RuntimeSession::telemetry() const {
  return impl_->telemetry();
}

void RuntimeSession::barrier() { impl_->barrier(); }

void RuntimeSession::inject_candidate_cleanup_failure_for_testing(
    bool enabled) {
  impl_->inject_candidate_cleanup_failure_for_testing(enabled);
}

void RuntimeSession::inject_mhd_close_failure_for_testing(bool enabled) {
  impl_->inject_mhd_close_failure_for_testing(enabled);
}

void RuntimeSession::consensus(bool success, std::string_view phase,
                               std::string_view message) {
  impl_->consensus(success, phase, message);
}

void RuntimeSession::require_same_string(std::string_view value,
                                         std::string_view phase,
                                         std::string_view message) {
  impl_->require_same_string(value, phase, message);
}

void RuntimeSession::configure_owned_devices(
    std::vector<DeviceIdentity> local_devices,
    std::string_view parse_error) {
  impl_->configure_owned_devices(std::move(local_devices), parse_error);
}

void RuntimeSession::configure_devices(
    std::vector<int> eligible_ordinals, std::string_view parse_error) {
  impl_->configure_devices(std::move(eligible_ordinals), parse_error);
}

void RuntimeSession::select_topology(
    std::size_t global_nx, std::size_t global_ny,
    std::optional<DecompositionShape> shape,
    std::size_t minimum_tile_width, std::string_view parse_error) {
  impl_->select_topology(global_nx, global_ny, std::move(shape),
                         minimum_tile_width, parse_error);
}

void RuntimeSession::start_mhd(
    mhd::MhdConfig config, MhdGlobalState state,
    std::optional<MhdGlobalBackground> background,
    TransportPolicy transport_policy, std::string_view parse_error) {
  impl_->start_mhd(std::move(config), std::move(state),
                   std::move(background), transport_policy, parse_error);
}

CheckpointMetadata RuntimeSession::restart_mhd(
    mhd::MhdConfig config, const std::string& path,
    const std::string& unit_system,
    const std::optional<MhdGlobalBackground>& expected_background,
    TransportPolicy transport_policy, std::string_view parse_error,
    std::vector<std::vector<std::uint8_t>>& diagnostic_state) {
  return impl_->restart_mhd(
      std::move(config), path, unit_system, expected_background,
      transport_policy, parse_error, diagnostic_state);
}

Real RuntimeSession::mhd_cfl_limit() { return impl_->mhd_cfl_limit(); }

void RuntimeSession::mhd_step(Real dt, bool check_cfl) {
  impl_->mhd_step(dt, check_cfl);
}

Real RuntimeSession::mhd_divergence_b_max() {
  return impl_->mhd_divergence_b_max();
}

MhdGlobalState RuntimeSession::mhd_gather_state() {
  return impl_->mhd_gather_state();
}

std::vector<Real> RuntimeSession::mhd_gather_cell_component(
    std::string_view component) {
  return impl_->mhd_gather_cell_component(component);
}

std::vector<MhdOwnedShard> RuntimeSession::mhd_local_owned_shards() {
  return impl_->mhd_local_owned_shards();
}

MhdGlobalCellSums RuntimeSession::mhd_global_cell_sums() {
  return impl_->mhd_global_cell_sums();
}

void RuntimeSession::mhd_write_checkpoint(
    const std::string& path, std::uint64_t step, double time,
    const std::string& unit_system,
    std::span<const std::uint8_t> diagnostic_state) {
  impl_->mhd_write_checkpoint(
      path, step, time, unit_system, diagnostic_state);
}

void RuntimeSession::close_mhd() { impl_->close_mhd(); }

void RuntimeSession::start_pic(
    pic::EmPicConfig config, PicGlobalFields fields,
    std::optional<PicGlobalFields> external_fields,
    std::vector<PicSpeciesState> species,
    TransportPolicy transport_policy, std::string_view parse_error) {
  impl_->start_pic(std::move(config), std::move(fields),
                   std::move(external_fields), std::move(species),
                   transport_policy, parse_error);
}

CheckpointMetadata RuntimeSession::restart_pic(
    pic::EmPicConfig config, const std::string& path,
    const std::string& unit_system,
    const std::vector<pic::SpeciesConfig>& expected_species,
    TransportPolicy transport_policy, std::string_view parse_error,
    std::vector<std::vector<std::uint8_t>>& diagnostic_state) {
  return impl_->restart_pic(
      std::move(config), path, unit_system, expected_species,
      transport_policy, parse_error, diagnostic_state);
}

Real RuntimeSession::pic_cfl_limit() { return impl_->pic_cfl_limit(); }

void RuntimeSession::pic_sample_external_fields(
    numerics::IFieldEvaluator& evaluator,
    const core::IFieldSource& source, Real length_scale,
    Real e_field_scale, Real b_field_scale) {
  impl_->pic_sample_external_fields(
      evaluator, source, length_scale, e_field_scale, b_field_scale);
}

void RuntimeSession::pic_step(Real dt) { impl_->pic_step(dt); }

PicGlobalState RuntimeSession::pic_gather_state() {
  return impl_->pic_gather_state();
}

std::vector<PicOwnedShard> RuntimeSession::pic_local_owned_shards(
    bool include_particles) {
  return impl_->pic_local_owned_shards(include_particles);
}

std::vector<std::uint64_t> RuntimeSession::pic_alive_counts() {
  return impl_->pic_alive_counts();
}

std::vector<Real> RuntimeSession::pic_kinetic_energies() {
  return impl_->pic_kinetic_energies();
}

Real RuntimeSession::pic_total_em_energy() {
  return impl_->pic_total_em_energy();
}

Real RuntimeSession::pic_gauss_residual() {
  return impl_->pic_gauss_residual();
}

void RuntimeSession::pic_write_checkpoint(
    const std::string& path, std::uint64_t step, double time,
    const std::string& unit_system,
    std::span<const std::uint8_t> diagnostic_state) {
  impl_->pic_write_checkpoint(
      path, step, time, unit_system, diagnostic_state);
}

void RuntimeSession::close_pic() { impl_->close_pic(); }

void RuntimeSession::close() { impl_->close(); }

}  // namespace quasar::distributed
