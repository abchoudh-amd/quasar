#include "quasar/distributed/transport.hpp"

#include "collective_helpers.hpp"
#include "mpi_runtime_native.hpp"

#include "quasar/backend/memory.hpp"

#if __has_include(<mpi-ext.h>)
#  include <mpi-ext.h>
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace quasar::distributed {

TransportPolicy parse_transport_policy(std::string_view value) {
  if (value == "auto") return TransportPolicy::automatic;
  if (value == "staged") return TransportPolicy::staged;
  if (value == "direct") return TransportPolicy::direct;
  throw std::invalid_argument{
      "transport must be 'auto', 'staged', or 'direct'"};
}

TransportResolution resolve_transport_policy(
    TransportPolicy requested, DirectCapability capability) {
  TransportResolution result{requested, TransportPolicy::staged, capability};
  switch (requested) {
    case TransportPolicy::staged:
      return result;
    case TransportPolicy::automatic:
      if (capability.available()) {
        result.interprocess = TransportPolicy::direct;
      }
      return result;
    case TransportPolicy::direct:
      if (!capability.available()) {
        std::ostringstream message;
        message << "direct device-buffer MPI was requested but ";
        if (!capability.recognized_query) {
          message << "the MPI implementation did not report ROCm-aware support";
        } else {
          message << "the collective device-buffer startup probe failed";
        }
        throw std::runtime_error{message.str()};
      }
      result.interprocess = TransportPolicy::direct;
      return result;
  }
  throw std::invalid_argument{"unknown transport policy"};
}

DirectCapability probe_direct_mpi(MpiRuntime& runtime) {
  runtime.require_orchestration_thread();
  DirectCapability result;

  bool local_query = false;
#if defined(OMPI_HAVE_MPI_EXT_ROCM)
  local_query = MPIX_Query_rocm_support() != 0;
#elif defined(MPIX_ROCM_AWARE_SUPPORT)
  local_query = MPIX_ROCM_AWARE_SUPPORT != 0;
#endif
  result.recognized_query = runtime.allreduce_all(local_query);
  if (!result.recognized_query) return result;

  backend::DeviceBuffer<unsigned char> send;
  backend::DeviceBuffer<unsigned char> receive;
  const bool local_preflight = local_operation_succeeds([&] {
    send = backend::DeviceBuffer<unsigned char>{1};
    receive = backend::DeviceBuffer<unsigned char>{1};
    const unsigned char value = static_cast<unsigned char>(runtime.rank() & 0xff);
    backend::device_memcpy_h2d(send.device_ptr(), &value, 1);
  });
  if (!runtime.allreduce_all(local_preflight)) {
    return result;
  }

  const int source = (runtime.rank() + runtime.size() - 1) % runtime.size();
  const int destination = (runtime.rank() + 1) % runtime.size();
  const int mpi_status = MPI_Sendrecv(
      send.device_ptr(), 1, MPI_BYTE, destination, 0,
      receive.device_ptr(), 1, MPI_BYTE, source, 0,
      detail::MpiRuntimeNativeAccess::world(runtime), MPI_STATUS_IGNORE);
  bool local_probe = mpi_status == MPI_SUCCESS;
  if (local_probe) {
    bool received_expected_value = false;
    const bool copy_succeeded = local_operation_succeeds([&] {
      unsigned char received = 0;
      backend::device_memcpy_d2h(&received, receive.device_ptr(), 1);
      received_expected_value =
          received == static_cast<unsigned char>(source & 0xff);
    });
    local_probe = copy_succeeded && received_expected_value;
  }
  result.startup_probe = runtime.allreduce_all(local_probe);
  return result;
}

namespace {

struct StagedPayload {
  ByteTransfer transfer{};
  std::byte* send{nullptr};
  std::byte* receive{nullptr};
};

struct PreparedRemoteTransfer {
  ByteTransfer transfer{};
  const void* send{nullptr};
  void* receive{nullptr};
  bool direct_device{false};
};

struct RequestStorage {
  std::vector<MPI_Request> requests{};
  backend::PinnedHostBuffer<std::byte> staged_send_storage{};
  backend::PinnedHostBuffer<std::byte> staged_receive_storage{};
  std::vector<StagedPayload> staged{};
  backend::PinnedHostBuffer<std::byte> local_staged_storage{};
  std::vector<StagedPayload> local_staged{};
  std::vector<ByteTransfer> direct{};
};

// RequestStorage owns every staging allocation touched by an MPI request.  If
// MPI cannot prove that all requests have completed, deliberately retain that
// storage until process teardown instead of freeing a buffer that MPI may
// still access.  A poisoned transport forbids subsequent epochs/tag reuse.
struct BatchResources {
  BatchResources() : storage{std::make_unique<RequestStorage>()} {}

  ~BatchResources() noexcept {
    if (!drain_proven) {
      (void)storage.release();
    }
  }

  std::unique_ptr<RequestStorage> storage{};
  bool drain_proven{true};
  bool terminal{false};
  bool completed{false};
};

struct PairingPlanKey {
  std::array<std::uint64_t, 3> words{};

  [[nodiscard]] bool operator==(const PairingPlanKey& other) const noexcept {
    return words == other.words;
  }
};

inline constexpr std::size_t pairing_plan_cache_capacity = 64;

struct TransportState {
  MpiRuntime* runtime{nullptr};
  MPI_Comm communicator{MPI_COMM_NULL};
  TransportResolution resolution{};
  TransportTelemetry telemetry{};
  backend::PinnedHostBuffer<std::byte> pooled_staged_send{};
  backend::PinnedHostBuffer<std::byte> pooled_staged_receive{};
  backend::PinnedHostBuffer<std::byte> pooled_local_staged{};
  std::shared_ptr<BatchResources> active_resources{};
  std::uint64_t next_epoch{0};
  std::uint64_t tag_generations{0};
  std::array<PairingPlanKey, pairing_plan_cache_capacity>
      pairing_plan_cache{};
  std::size_t pairing_plan_cache_size{0};
  std::size_t pairing_plan_cache_next{0};
  std::string poison_reason{};
  bool active{false};
  bool poisoned{false};
  bool closed{false};
};

struct MessageWireRecord {
  int source{-1};
  int peer{-1};
  std::uint32_t tag_slot{0};
  std::uint64_t bytes{0};
  std::uint64_t chunks{0};
};

static_assert(std::is_trivially_copyable_v<MessageWireRecord>);
static_assert(sizeof(MessageWireRecord)
              <= static_cast<std::size_t>(std::numeric_limits<int>::max()));
static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t));
static_assert(std::is_trivially_copyable_v<PairingPlanKey>);

std::size_t mpi_chunk_count(std::size_t bytes) {
  constexpr std::size_t maximum =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  return bytes == 0 ? 0 : 1 + (bytes - 1) / maximum;
}

PairingPlanKey pairing_plan_signature(
    int local_rank, std::span<const ByteTransfer> transfers) noexcept {
  std::uint64_t primary = UINT64_C(1469598103934665603);
  std::uint64_t secondary = UINT64_C(0x9e3779b97f4a7c15);
  std::uint64_t count = 0;
  const auto mix = [&primary, &secondary](std::uint64_t value) {
    for (unsigned int byte = 0; byte < 8; ++byte) {
      primary ^= (value >> (byte * 8)) & UINT64_C(0xff);
      primary *= UINT64_C(1099511628211);
    }
    secondary ^= value + UINT64_C(0x9e3779b97f4a7c15)
        + (secondary << 6) + (secondary >> 2);
    secondary *= UINT64_C(0xbf58476d1ce4e5b9);
  };
  // Rank-mixing prevents identical local plans on an even number of ranks
  // from cancelling when the global key is reduced with bitwise XOR.
  mix(static_cast<std::uint64_t>(local_rank));
  for (const auto& transfer : transfers) {
    if (transfer.peer_rank == local_rank || transfer.bytes == 0) continue;
    mix(static_cast<std::uint64_t>(transfer.peer_rank));
    mix(transfer.tag_slot);
    mix(static_cast<std::uint64_t>(transfer.bytes));
    mix(static_cast<std::uint64_t>(mpi_chunk_count(transfer.bytes)));
    ++count;
  }
  mix(count);
  return PairingPlanKey{{
      primary,
      secondary,
      count ^ (UINT64_C(0xd6e8feb86659fd93)
               * (static_cast<std::uint64_t>(local_rank) + 1U))}};
}

bool pairing_plan_cached(const TransportState& state,
                         const PairingPlanKey& key) noexcept {
  return std::find(state.pairing_plan_cache.begin(),
                   state.pairing_plan_cache.begin()
                       + static_cast<std::ptrdiff_t>(
                           state.pairing_plan_cache_size),
                   key)
      != state.pairing_plan_cache.begin()
             + static_cast<std::ptrdiff_t>(state.pairing_plan_cache_size);
}

void remember_pairing_plan(TransportState& state,
                           const PairingPlanKey& key) noexcept {
  state.pairing_plan_cache[state.pairing_plan_cache_next] = key;
  if (state.pairing_plan_cache_size < pairing_plan_cache_capacity) {
    ++state.pairing_plan_cache_size;
  }
  state.pairing_plan_cache_next =
      (state.pairing_plan_cache_next + 1) % pairing_plan_cache_capacity;
}

template <class Callback>
void for_each_mpi_chunk(std::size_t bytes, Callback&& callback) {
  constexpr std::size_t maximum =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  std::size_t offset = 0;
  while (offset < bytes) {
    const std::size_t count = std::min(maximum, bytes - offset);
    callback(offset, static_cast<int>(count));
    offset += count;
  }
}

struct RequestCompletion {
  bool operations_succeeded{true};
  bool drained{true};
  int first_error{MPI_SUCCESS};
  const char* failed_operation{nullptr};
};

void remember_request_error(RequestCompletion& result, int error,
                            const char* operation) noexcept {
  result.operations_succeeded = false;
  if (result.first_error == MPI_SUCCESS) {
    result.first_error = error == MPI_SUCCESS ? MPI_ERR_OTHER : error;
    result.failed_operation = operation;
  }
}

bool all_requests_null(const std::vector<MPI_Request>& requests) noexcept {
  return std::all_of(requests.begin(), requests.end(), [](MPI_Request request) {
    return request == MPI_REQUEST_NULL;
  });
}

void cancel_and_drain(std::vector<MPI_Request>& requests,
                      RequestCompletion& result) noexcept {
  for (auto& request : requests) {
    if (request == MPI_REQUEST_NULL) continue;

    const int cancel_status = MPI_Cancel(&request);
    if (cancel_status != MPI_SUCCESS) {
      remember_request_error(result, cancel_status, "MPI_Cancel");

      // A failed cancellation does not justify a blocking wait: the matching
      // operation may never have been posted.  A successful test still proves
      // that this particular request no longer references its buffer.
      int complete = 0;
      MPI_Status status{};
      const int test_status = MPI_Test(&request, &complete, &status);
      if (test_status != MPI_SUCCESS) {
        remember_request_error(result, test_status, "MPI_Test(after cancel)");
      }
      if (complete == 0 || request != MPI_REQUEST_NULL) result.drained = false;
      continue;
    }

    MPI_Status status{};
    const int wait_status = MPI_Wait(&request, &status);
    if (wait_status != MPI_SUCCESS) {
      remember_request_error(result, wait_status, "MPI_Wait(after cancel)");
    }
    if (request != MPI_REQUEST_NULL) {
      int complete = 0;
      const int test_status = MPI_Test(&request, &complete, &status);
      if (test_status != MPI_SUCCESS) {
        remember_request_error(result, test_status,
                               "MPI_Test(after failed wait)");
      }
      if (complete == 0 || request != MPI_REQUEST_NULL) result.drained = false;
    }
  }
  result.drained = result.drained && all_requests_null(requests);
}

RequestCompletion complete_requests(std::vector<MPI_Request>& requests,
                                    bool cancel_first = false) noexcept {
  RequestCompletion result;
  if (cancel_first) {
    cancel_and_drain(requests, result);
    return result;
  }

  // Keep status storage bounded and allocation-free.  MPI_ERR_IN_STATUS can
  // leave requests later in a Waitall array pending, so statuses may not be
  // ignored and all still-live handles must be cancelled and drained below.
  constexpr std::size_t wait_chunk = 256;
  std::array<MPI_Status, wait_chunk> statuses{};
  std::size_t offset = 0;
  while (offset < requests.size()) {
    const int count = static_cast<int>(
        std::min(wait_chunk, requests.size() - offset));
    const int wait_status = MPI_Waitall(
        count, requests.data() + offset, statuses.data());
    if (wait_status != MPI_SUCCESS) {
      int request_error = wait_status;
      if (wait_status == MPI_ERR_IN_STATUS) {
        for (int index = 0; index < count; ++index) {
          const int status_error = statuses[static_cast<std::size_t>(index)].MPI_ERROR;
          if (status_error != MPI_SUCCESS && status_error != MPI_ERR_PENDING) {
            request_error = status_error;
            break;
          }
        }
      }
      remember_request_error(result, request_error, "MPI_Waitall");
      cancel_and_drain(requests, result);
      return result;
    }
    offset += static_cast<std::size_t>(count);
  }
  result.drained = all_requests_null(requests);
  if (!result.drained) {
    remember_request_error(result, MPI_ERR_REQUEST,
                           "MPI_Waitall(request remained active)");
    cancel_and_drain(requests, result);
  }
  return result;
}

std::string request_completion_error(const RequestCompletion& completion) {
  if (completion.first_error == MPI_SUCCESS) {
    return completion.drained
        ? std::string{}
        : std::string{"MPI request drain could not be proven"};
  }
  std::ostringstream message;
  message << (completion.failed_operation == nullptr
                  ? "MPI request completion"
                  : completion.failed_operation)
          << " failed: " << mpi_error_string(completion.first_error);
  if (!completion.drained) {
    message << "; MPI request drain could not be proven";
  }
  return message.str();
}

void poison_transport(TransportState& state, std::string_view reason) noexcept {
  state.poisoned = true;
  if (state.poison_reason.empty()) {
    try {
      state.poison_reason.assign(reason);
    } catch (...) {
      // `poisoned` is the authoritative state.  Retaining detailed text is
      // best-effort and must not divert one rank from the next collective.
    }
  }
}

[[noreturn]] void fail_transport_startup(
    MpiRuntime& runtime, MPI_Comm& communicator,
    const CollectiveResolution& startup_resolution) {
  // Every caller reaches this helper only after agreeing that the duplicated
  // communicator exists on every rank.  Its release is therefore collective;
  // report cleanup failure through the still-live runtime communicator before
  // leaving construction.
  const int free_status = MPI_Comm_free(&communicator);
  const std::uint64_t cleanup_epoch =
      startup_resolution.representative.epoch + 1;
  const CollectiveResolution cleanup_resolution = runtime.consensus(
      free_status == MPI_SUCCESS
          ? CollectiveErrorRecord::success(
                cleanup_epoch, runtime.rank(), "transport-startup-cleanup")
          : CollectiveErrorRecord::failure(
                cleanup_epoch, runtime.rank(), -1, free_status,
                "transport-startup-cleanup",
                "MPI_Comm_free failed during transport startup cleanup"));
  if (!cleanup_resolution.accepted()) {
    throw DistributedCollectiveError{cleanup_resolution};
  }
  throw DistributedCollectiveError{startup_resolution};
}

void require_transport_startup_phase(
    MpiRuntime& runtime, MPI_Comm& communicator, std::uint64_t epoch,
    bool local_success, std::string_view phase, int code,
    std::string_view message) {
  const CollectiveResolution resolution = runtime.consensus(
      local_success
          ? CollectiveErrorRecord::success(epoch, runtime.rank(), phase)
          : CollectiveErrorRecord::failure(
                epoch, runtime.rank(), -1, code, phase, message));
  if (!resolution.accepted()) {
    fail_transport_startup(runtime, communicator, resolution);
  }
}

bool is_valid_device_transfer(const ByteTransfer& transfer) {
  return transfer.send_device >= 0 && transfer.receive_device >= 0;
}

std::pair<bool, std::string> validate_message_pairs(
    MpiRuntime& runtime, MPI_Comm communicator,
    std::span<const ByteTransfer> transfers) {
  std::vector<MessageWireRecord> local;
  collective_try_with_fallback_at_epoch(
      runtime, 0, "transport-pairing-local-storage",
      "message descriptor allocation failed", -1, [&] {
        local.reserve(transfers.size());
        for (const auto& transfer : transfers) {
          if (transfer.peer_rank == runtime.rank() || transfer.bytes == 0) {
            continue;
          }
          local.push_back({runtime.rank(), transfer.peer_rank,
                           transfer.tag_slot, transfer.bytes,
                           mpi_chunk_count(transfer.bytes)});
        }
      });
  const bool local_size_valid =
      local.size() <= static_cast<std::size_t>(
                          std::numeric_limits<int>::max())
                          / sizeof(MessageWireRecord);
  if (!runtime.allreduce_all(local_size_valid)) {
    return {false, "message descriptor set exceeds MPI count range"};
  }
  const int local_bytes = static_cast<int>(
      local.size() * sizeof(MessageWireRecord));
  std::vector<int> sizes;
  collective_try_with_fallback_at_epoch(
      runtime, 0, "transport-pairing-size-storage",
      "message descriptor size storage allocation failed", -1, [&] {
        sizes.resize(static_cast<std::size_t>(runtime.size()));
      });
  check_mpi(MPI_Allgather(&local_bytes, 1, MPI_INT,
                          sizes.data(), 1, MPI_INT, communicator),
            "MPI_Allgather(message descriptor sizes)");
  std::vector<int> offsets;
  std::vector<std::byte> bytes;
  std::int64_t total_bytes = 0;
  constexpr int wire_record_bytes = static_cast<int>(sizeof(MessageWireRecord));
  bool payload_layout_valid = true;
  const bool payload_storage_valid = local_operation_succeeds([&] {
    offsets.resize(sizes.size());
    for (std::size_t rank = 0; rank < sizes.size(); ++rank) {
      if (sizes[rank] < 0 || sizes[rank] % wire_record_bytes != 0
          || total_bytes > std::numeric_limits<int>::max() - sizes[rank]) {
        payload_layout_valid = false;
        break;
      }
      offsets[rank] = static_cast<int>(total_bytes);
      total_bytes += sizes[rank];
    }
    if (payload_layout_valid) {
      bytes.resize(static_cast<std::size_t>(total_bytes));
    }
  });
  collective_require_at_epoch(
      runtime, 0, payload_storage_valid && payload_layout_valid,
      "transport-pairing-payload-storage",
      "global message descriptor storage is invalid or unavailable");
  MessageWireRecord local_dummy{};
  std::byte receive_dummy{};
  check_mpi(MPI_Allgatherv(local.empty() ? &local_dummy : local.data(),
                           local_bytes, MPI_BYTE,
                           bytes.empty() ? &receive_dummy : bytes.data(),
                           sizes.data(),
                           offsets.data(), MPI_BYTE,
                           communicator),
            "MPI_Allgatherv(message descriptors)");
  const std::size_t record_count = bytes.size() / sizeof(MessageWireRecord);
  std::vector<MessageWireRecord> records;
  collective_try_with_fallback_at_epoch(
      runtime, 0, "transport-pairing-record-storage",
      "decoded message descriptor allocation failed", -1, [&] {
        records.resize(record_count);
      });
  if (!bytes.empty()) {
    std::memcpy(records.data(), bytes.data(), bytes.size());
  }
  const auto key = [](const MessageWireRecord& record) {
    return std::tuple{record.source, record.peer, record.tag_slot,
                      record.bytes, record.chunks};
  };
  std::vector<MessageWireRecord> reverse = records;
  for (auto& record : reverse) std::swap(record.source, record.peer);
  std::sort(records.begin(), records.end(),
            [&](const auto& left, const auto& right) {
              return key(left) < key(right);
            });
  std::sort(reverse.begin(), reverse.end(),
            [&](const auto& left, const auto& right) {
              return key(left) < key(right);
            });
  for (std::size_t index = 0; index < records.size(); ++index) {
    if (key(records[index]) != key(reverse[index])) {
      const auto& record = records[index];
      std::ostringstream message;
      message << "rank " << record.source << " and rank " << record.peer
              << " disagree on tag slot " << record.tag_slot
              << ", payload size, or MPI chunk layout";
      return {false, message.str()};
    }
  }
  return {true, {}};
}

void stage_device_send(StagedPayload& payload) {
  const auto& transfer = payload.transfer;
  backend::DeviceGuard guard{transfer.send_device};
  backend::device_memcpy_d2h_async(
      payload.send, transfer.send_buffer, transfer.bytes,
      transfer.send_stream);
}

template <class Payload, class Device, class Stream>
void synchronize_unique_streams(std::span<const Payload> payloads,
                                Device&& device, Stream&& stream) {
  std::vector<std::pair<int, backend::stream_t>> unique;
  unique.reserve(payloads.size());
  for (const auto& payload : payloads) {
    unique.emplace_back(device(payload), stream(payload));
  }
  const auto less = [](const auto& left, const auto& right) {
    if (left.first != right.first) return left.first < right.first;
    return std::less<backend::stream_t>{}(left.second, right.second);
  };
  std::sort(unique.begin(), unique.end(), less);
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
  for (const auto& [ordinal, selected_stream] : unique) {
    backend::DeviceGuard guard{ordinal};
    backend::device_synchronize(selected_stream);
  }
}

void finish_staged_sends(std::span<const StagedPayload> payloads) {
  synchronize_unique_streams(
      payloads,
      [](const StagedPayload& payload) {
        return payload.transfer.send_device;
      },
      [](const StagedPayload& payload) {
        return payload.transfer.send_stream;
      });
}

void finish_staged_receives(std::span<StagedPayload> payloads) {
  try {
    for (auto& payload : payloads) {
      const auto& transfer = payload.transfer;
      backend::DeviceGuard guard{transfer.receive_device};
      backend::device_memcpy_h2d_async(
          transfer.receive_buffer, payload.receive, transfer.bytes,
          transfer.receive_stream);
    }
    synchronize_unique_streams(
        std::span<const StagedPayload>{payloads.data(), payloads.size()},
        [](const StagedPayload& payload) {
          return payload.transfer.receive_device;
        },
        [](const StagedPayload& payload) {
          return payload.transfer.receive_stream;
        });
  } catch (...) {
    try {
      synchronize_unique_streams(
          std::span<const StagedPayload>{payloads.data(), payloads.size()},
          [](const StagedPayload& payload) {
            return payload.transfer.receive_device;
          },
          [](const StagedPayload& payload) {
            return payload.transfer.receive_stream;
          });
    } catch (...) {
    }
    throw;
  }
}

void recycle_staging_buffers(TransportState& state,
                             RequestStorage& storage) noexcept {
  if (storage.staged_send_storage.size()
      > state.pooled_staged_send.size()) {
    state.pooled_staged_send = std::move(storage.staged_send_storage);
  }
  if (storage.staged_receive_storage.size()
      > state.pooled_staged_receive.size()) {
    state.pooled_staged_receive = std::move(storage.staged_receive_storage);
  }
  if (storage.local_staged_storage.size()
      > state.pooled_local_staged.size()) {
    state.pooled_local_staged = std::move(storage.local_staged_storage);
  }
}

bool can_copy_device_to_device(const ByteTransfer& transfer) {
  return transfer.send_device == transfer.receive_device
      || backend::device_can_access_peer(transfer.receive_device,
                                         transfer.send_device);
}

void copy_local_device_transfers(
    std::span<const ByteTransfer> transfers) {
  synchronize_unique_streams(
      transfers,
      [](const ByteTransfer& transfer) { return transfer.send_device; },
      [](const ByteTransfer& transfer) { return transfer.send_stream; });
  try {
    for (const auto& transfer : transfers) {
      backend::DeviceGuard guard{transfer.receive_device};
      if (transfer.send_device != transfer.receive_device) {
        backend::device_enable_peer_access(transfer.send_device);
      }
      backend::device_memcpy_peer_async(
          transfer.receive_buffer, transfer.receive_device,
          transfer.send_buffer, transfer.send_device,
          transfer.bytes, transfer.receive_stream);
    }
    synchronize_unique_streams(
        transfers,
        [](const ByteTransfer& transfer) { return transfer.receive_device; },
        [](const ByteTransfer& transfer) { return transfer.receive_stream; });
  } catch (...) {
    // Copies issued before a later launch failure still reference caller-owned
    // buffers.  Drain every receive stream once before unwinding.
    try {
      synchronize_unique_streams(
          transfers,
          [](const ByteTransfer& transfer) {
            return transfer.receive_device;
          },
          [](const ByteTransfer& transfer) {
            return transfer.receive_stream;
          });
    } catch (...) {
    }
    throw;
  }
}

void prepare_direct_transfers(std::span<const ByteTransfer> transfers) {
  synchronize_unique_streams(
      transfers,
      [](const ByteTransfer& transfer) { return transfer.send_device; },
      [](const ByteTransfer& transfer) { return transfer.send_stream; });
  synchronize_unique_streams(
      transfers,
      [](const ByteTransfer& transfer) { return transfer.receive_device; },
      [](const ByteTransfer& transfer) { return transfer.receive_stream; });
}

void finish_direct_receives(std::span<const ByteTransfer> transfers) {
  // ROCm-aware MPI is not assumed to integrate with arbitrary HIP streams.
  // MPI_Wait proves the external write is host-complete; synchronize each
  // receiving stream once before workers enqueue consumers.
  synchronize_unique_streams(
      transfers,
      [](const ByteTransfer& transfer) { return transfer.receive_device; },
      [](const ByteTransfer& transfer) { return transfer.receive_stream; });
}

std::size_t validate_transfer_descriptors_at_epoch(
    TransportState& state, std::uint64_t epoch,
    std::span<const ByteTransfer> transfers) {
  bool descriptors_valid = true;
  CollectiveFixedMessage validation_error;
  std::size_t request_count = 0;
  std::set<std::pair<int, std::uint32_t>> remote_tags;
  const bool validation_succeeded = capture_local_failure_with_fallback(
      validation_error, "descriptor allocation failed", [&] {
        for (const auto& transfer : transfers) {
          if (transfer.peer_rank < 0
              || transfer.peer_rank >= state.runtime->size()
              || transfer.tag_slot >= Transport::tag_slots_per_epoch
              || (transfer.bytes != 0
                  && (transfer.send_buffer == nullptr
                      || transfer.receive_buffer == nullptr))
              || (transfer.residence == BufferResidence::device
                  && !is_valid_device_transfer(transfer))) {
            descriptors_valid = false;
            validation_error.assign("invalid byte-transfer descriptor");
            break;
          }
          if (transfer.peer_rank != state.runtime->rank()) {
            if (transfer.bytes != 0
                && !remote_tags.emplace(transfer.peer_rank,
                                        transfer.tag_slot).second) {
              descriptors_valid = false;
              validation_error.assign(
                  "duplicate non-empty (peer rank, tag slot) in one communication epoch");
              break;
            }
            const std::size_t chunks = mpi_chunk_count(transfer.bytes);
            if (chunks
                > (std::numeric_limits<std::size_t>::max() - request_count)
                    / 2) {
              descriptors_valid = false;
              validation_error.assign("MPI request count overflows host size");
              break;
            }
            request_count += 2 * chunks;
          }
        }
      });
  collective_require_at_epoch(
      *state.runtime, epoch, validation_succeeded && descriptors_valid,
      "transport-begin", validation_error);
  return request_count;
}

void validate_pairing_plan_at_epoch(
    TransportState& state, std::uint64_t epoch,
    std::span<const ByteTransfer> transfers) {
  const PairingPlanKey local_pairing_key =
      pairing_plan_signature(state.runtime->rank(), transfers);
  PairingPlanKey global_pairing_key{};
  check_mpi(MPI_Allreduce(
                local_pairing_key.words.data(),
                global_pairing_key.words.data(),
                static_cast<int>(global_pairing_key.words.size()),
                MPI_UINT64_T, MPI_BXOR, state.communicator),
            "MPI_Allreduce(transport pairing plan key)");
  if (pairing_plan_cached(state, global_pairing_key)) return;

  bool pairs_valid = true;
  CollectiveFixedMessage pairing_error;
  const bool pairing_check_succeeded = capture_local_failure(
      pairing_error, "transport pairing validation failed", [&] {
        auto [valid, error] = validate_message_pairs(
            *state.runtime, state.communicator, transfers);
        pairs_valid = valid;
        if (!valid) pairing_error.assign(error);
      });
  collective_require_at_epoch(
      *state.runtime, epoch, pairing_check_succeeded && pairs_valid,
      "transport-pairing",
      pairing_error.empty()
          ? std::string_view{"transport pairing validation failed"}
          : pairing_error.view());
  remember_pairing_plan(state, global_pairing_key);
}

struct StagingStorageRequirements {
  std::size_t staged_bytes{0};
  std::size_t local_staged_bytes{0};
};

StagingStorageRequirements staging_storage_requirements(
    const TransportState& state,
    std::span<const ByteTransfer> transfers) {
  StagingStorageRequirements requirements;
  for (const auto& transfer : transfers) {
    const bool needs_staging = transfer.bytes != 0
        && transfer.peer_rank != state.runtime->rank()
        && transfer.residence == BufferResidence::device
        && !state.resolution.uses_direct_mpi();
    if (needs_staging) {
      if (transfer.bytes > std::numeric_limits<std::size_t>::max()
                               - requirements.staged_bytes) {
        throw std::length_error{"staged transport storage size overflows"};
      }
      requirements.staged_bytes += transfer.bytes;
    }
    const bool local_device_transfer = transfer.bytes != 0
        && transfer.peer_rank == state.runtime->rank()
        && transfer.residence == BufferResidence::device;
    const bool needs_local_staging = local_device_transfer
        && !can_copy_device_to_device(transfer);
    if (needs_local_staging) {
      if (transfer.bytes > std::numeric_limits<std::size_t>::max()
                               - requirements.local_staged_bytes) {
        throw std::length_error{
            "local staged transport storage size overflows"};
      }
      requirements.local_staged_bytes += transfer.bytes;
    }
  }
  return requirements;
}

void acquire_staging_storage(
    TransportState& state, RequestStorage& storage,
    const StagingStorageRequirements& requirements) {
  if (requirements.staged_bytes != 0) {
    if (state.pooled_staged_send.size() >= requirements.staged_bytes) {
      storage.staged_send_storage = std::move(state.pooled_staged_send);
    } else {
      storage.staged_send_storage =
          backend::PinnedHostBuffer<std::byte>{requirements.staged_bytes};
    }
    if (state.pooled_staged_receive.size() >= requirements.staged_bytes) {
      storage.staged_receive_storage =
          std::move(state.pooled_staged_receive);
    } else {
      storage.staged_receive_storage =
          backend::PinnedHostBuffer<std::byte>{requirements.staged_bytes};
    }
  }
  if (requirements.local_staged_bytes != 0) {
    if (state.pooled_local_staged.size()
        >= requirements.local_staged_bytes) {
      storage.local_staged_storage = std::move(state.pooled_local_staged);
    } else {
      storage.local_staged_storage =
          backend::PinnedHostBuffer<std::byte>{
              requirements.local_staged_bytes};
    }
  }
}

struct PreparedTransfers {
  std::vector<PreparedRemoteTransfer> remote{};
  std::vector<ByteTransfer> local_device_copies{};
};

void reserve_transfer_storage(
    RequestStorage& storage, PreparedTransfers& prepared,
    std::size_t request_count, std::size_t transfer_count) {
  storage.requests.reserve(request_count);
  storage.staged.reserve(transfer_count);
  storage.local_staged.reserve(transfer_count);
  storage.direct.reserve(transfer_count);
  prepared.remote.reserve(transfer_count);
  prepared.local_device_copies.reserve(transfer_count);
}

void classify_transfers(
    TransportState& state, RequestStorage& storage,
    std::span<const ByteTransfer> transfers, PreparedTransfers& prepared) {
  std::size_t staged_offset = 0;
  std::size_t local_staged_offset = 0;
  for (const auto& transfer : transfers) {
    ++state.telemetry.messages;
    state.telemetry.bytes += transfer.bytes;
    if (transfer.bytes == 0) continue;

    if (transfer.peer_rank == state.runtime->rank()) {
      if (transfer.residence == BufferResidence::host) {
        std::memmove(transfer.receive_buffer, transfer.send_buffer,
                     transfer.bytes);
        state.telemetry.local_staged_bytes += transfer.bytes;
      } else if (can_copy_device_to_device(transfer)) {
        prepared.local_device_copies.push_back(transfer);
        state.telemetry.peer_bytes += transfer.bytes;
      } else {
        std::byte* const staging =
            storage.local_staged_storage.data() + local_staged_offset;
        storage.local_staged.push_back(
            StagedPayload{transfer, staging, staging});
        local_staged_offset += transfer.bytes;
        state.telemetry.local_staged_bytes += transfer.bytes;
      }
      continue;
    }

    const bool direct_device =
        transfer.residence == BufferResidence::device
        && state.resolution.uses_direct_mpi();
    const void* send_buffer = transfer.send_buffer;
    void* receive_buffer = transfer.receive_buffer;
    if (transfer.residence == BufferResidence::device && !direct_device) {
      storage.staged.push_back(StagedPayload{
          transfer,
          storage.staged_send_storage.data() + staged_offset,
          storage.staged_receive_storage.data() + staged_offset});
      StagedPayload& payload = storage.staged.back();
      send_buffer = payload.send;
      receive_buffer = payload.receive;
      staged_offset += transfer.bytes;
    } else if (direct_device) {
      storage.direct.push_back(transfer);
    }

    prepared.remote.push_back(
        {transfer, send_buffer, receive_buffer, direct_device});
  }
}

void perform_local_transfers(
    RequestStorage& storage,
    std::span<const ByteTransfer> local_device_copies) {
  copy_local_device_transfers(local_device_copies);

  try {
    for (auto& payload : storage.local_staged) {
      stage_device_send(payload);
    }
    finish_staged_sends(storage.local_staged);
    finish_staged_receives(storage.local_staged);
  } catch (...) {
    // A launch failure may leave earlier D2H operations referring to the
    // batch-owned pinned buffer. Drain every participating send stream
    // before unwinding; finish_staged_receives drains receive streams itself.
    try {
      finish_staged_sends(storage.local_staged);
    } catch (...) {
    }
    throw;
  }
}

void prepare_remote_transfers(RequestStorage& storage) {
  prepare_direct_transfers(storage.direct);

  try {
    for (auto& payload : storage.staged) {
      stage_device_send(payload);
    }
    finish_staged_sends(storage.staged);
  } catch (...) {
    // Any copies issued before a later launch failed still reference the
    // batch-owned pinned storage. Drain their streams before unwinding.
    try {
      finish_staged_sends(storage.staged);
    } catch (...) {
    }
    throw;
  }
}

void post_remote_requests(
    TransportState& state, BatchResources& resources,
    std::span<const PreparedRemoteTransfer> prepared, int tag_base) {
  for (const auto& item : prepared) {
    const auto& transfer = item.transfer;
    const int tag = tag_base + static_cast<int>(transfer.tag_slot);
    auto* receive_bytes = static_cast<std::byte*>(item.receive);
    const auto* send_bytes = static_cast<const std::byte*>(item.send);
    for_each_mpi_chunk(transfer.bytes, [&](std::size_t offset, int count) {
      resources.storage->requests.push_back(MPI_REQUEST_NULL);
      resources.drain_proven = false;
      check_mpi(MPI_Irecv(receive_bytes + offset, count, MPI_BYTE,
                          transfer.peer_rank, tag, state.communicator,
                          &resources.storage->requests.back()),
                "MPI_Irecv(distributed transfer)");
    });
    for_each_mpi_chunk(transfer.bytes, [&](std::size_t offset, int count) {
      resources.storage->requests.push_back(MPI_REQUEST_NULL);
      resources.drain_proven = false;
      check_mpi(MPI_Isend(send_bytes + offset, count, MPI_BYTE,
                          transfer.peer_rank, tag, state.communicator,
                          &resources.storage->requests.back()),
                "MPI_Isend(distributed transfer)");
    });
    if (item.direct_device) {
      state.telemetry.direct_mpi_bytes += transfer.bytes;
    } else {
      state.telemetry.staged_mpi_bytes += transfer.bytes;
    }
  }
}

void start_transfer_requests(
    TransportState& state, BatchResources& resources,
    std::span<const ByteTransfer> transfers, std::size_t request_count,
    int tag_base) {
  RequestStorage& storage = *resources.storage;
  PreparedTransfers prepared;
  reserve_transfer_storage(
      storage, prepared, request_count, transfers.size());
  const StagingStorageRequirements requirements =
      staging_storage_requirements(state, transfers);
  acquire_staging_storage(state, storage, requirements);
  classify_transfers(state, storage, transfers, prepared);
  perform_local_transfers(storage, prepared.local_device_copies);
  prepare_remote_transfers(storage);
  post_remote_requests(state, resources, prepared.remote, tag_base);
}

void require_startup_resolution(
    TransportState& state, BatchResources& resources,
    const CollectiveErrorRecord& started) {
  CollectiveResolution start_resolution;
  try {
    start_resolution = state.runtime->consensus(started);
  } catch (...) {
    const RequestCompletion cleanup =
        complete_requests(resources.storage->requests, true);
    resources.drain_proven = cleanup.drained;
    resources.terminal = true;
    if (cleanup.drained) {
      resources.completed = true;
      state.active = false;
      state.active_resources.reset();
    }
    poison_transport(
        state,
        "transport startup consensus failed before coordinated cleanup");
    ++state.telemetry.epochs;
    throw;
  }
  if (start_resolution.accepted()) return;

  const RequestCompletion cleanup =
      complete_requests(resources.storage->requests, true);
  resources.drain_proven = cleanup.drained;
  resources.terminal = true;
  bool globally_drained = false;
  try {
    globally_drained = state.runtime->allreduce_all(cleanup.drained);
  } catch (...) {
    poison_transport(state, "startup-drain consensus failed");
    ++state.telemetry.epochs;
    throw;
  }
  if (globally_drained) {
    resources.completed = true;
    state.active = false;
    state.active_resources.reset();
  } else {
    poison_transport(
        state, "at least one rank could not prove startup request drainage");
  }
  ++state.telemetry.epochs;
  throw DistributedCollectiveError{start_resolution};
}

void finish_empty_epoch_if_collective(
    TransportState& state, BatchResources& resources) {
  bool no_rank_has_requests = false;
  try {
    no_rank_has_requests =
        state.runtime->allreduce_all(resources.storage->requests.empty());
  } catch (const std::exception&) {
    const RequestCompletion cleanup =
        complete_requests(resources.storage->requests, true);
    resources.drain_proven = cleanup.drained;
    resources.terminal = true;
    if (cleanup.drained) {
      resources.completed = true;
      state.active = false;
      state.active_resources.reset();
    }
    poison_transport(state, "active-epoch consensus failed");
    ++state.telemetry.epochs;
    throw;
  }
  if (!no_rank_has_requests) return;

  resources.drain_proven = true;
  resources.terminal = true;
  resources.completed = true;
  recycle_staging_buffers(state, *resources.storage);
  state.active = false;
  state.active_resources.reset();
  ++state.telemetry.epochs;
}

}  // namespace

struct Transport::Impl {
  std::shared_ptr<TransportState> state{};
};

struct TransferBatch::Impl {
  std::shared_ptr<TransportState> state{};
  std::shared_ptr<BatchResources> resources{};
  std::uint64_t epoch{0};
};

TransferBatch::TransferBatch(std::unique_ptr<Impl> implementation) noexcept
  : implementation_{std::move(implementation)} {}

TransferBatch::~TransferBatch() noexcept = default;
TransferBatch::TransferBatch(TransferBatch&&) noexcept = default;
TransferBatch& TransferBatch::operator=(TransferBatch&&) noexcept = default;

bool TransferBatch::valid() const noexcept {
  return implementation_ != nullptr;
}

bool TransferBatch::complete() const noexcept {
  return implementation_ == nullptr || implementation_->resources->completed;
}

std::uint64_t TransferBatch::epoch() const noexcept {
  return implementation_ == nullptr ? 0 : implementation_->epoch;
}

void TransferBatch::wait() {
  if (implementation_ == nullptr || implementation_->resources->completed) return;
  Impl& batch = *implementation_;
  auto state = batch.state;
  state->runtime->require_orchestration_thread();
  auto resources = batch.resources;
  if (resources->terminal) {
    throw std::logic_error{
        "distributed transfer batch is terminal but its MPI drain was not proven"};
  }
  if (state->poisoned) {
    throw std::logic_error{"distributed transport is poisoned: "
                           + state->poison_reason};
  }
  if (!state->active || state->active_resources != resources) {
    throw std::logic_error{"distributed transfer batch is no longer active"};
  }

  RequestCompletion completion =
      complete_requests(resources->storage->requests);
  resources->drain_proven = completion.drained;
  bool globally_drained = false;
  try {
    globally_drained = state->runtime->allreduce_all(completion.drained);
  } catch (const std::exception&) {
    resources->terminal = true;
    poison_transport(*state, "request-drain consensus failed");
    ++state->telemetry.epochs;
    throw;
  }

  bool local_success = completion.operations_succeeded;
  CollectiveFixedMessage local_error;
  if (!local_success) {
    (void)capture_local_failure_with_fallback(
        local_error, "transport completion failed", [&] {
          local_error.assign(request_completion_error(completion));
        });
  }
  if (!globally_drained) {
    resources->terminal = true;
    poison_transport(*state,
                     "at least one rank could not prove MPI request drainage");
    local_success = false;
    if (!state->poison_reason.empty()) {
      local_error.assign(state->poison_reason);
    }
  }

  if (completion.drained && completion.operations_succeeded) {
    CollectiveFixedMessage receive_error;
    const LocalFailureKind receive_failure = capture_local_exception(
        receive_error, "unspecified distributed failure", [&] {
          finish_staged_receives(resources->storage->staged);
        });
    if (receive_failure != LocalFailureKind::none) {
      local_success = false;
      if (receive_failure == LocalFailureKind::standard_exception
          && local_error.empty()) {
        local_error.assign(receive_error);
      }
    }
  }

  if (completion.drained) {
    CollectiveFixedMessage receive_error;
    const LocalFailureKind receive_failure = capture_local_exception(
        receive_error, "unspecified distributed failure", [&] {
          finish_direct_receives(resources->storage->direct);
        });
    if (receive_failure != LocalFailureKind::none) {
      local_success = false;
      if (receive_failure == LocalFailureKind::standard_exception
          && local_error.empty()) {
        local_error.assign(receive_error);
      }
    }
    recycle_staging_buffers(*state, *resources->storage);
  }

  if (globally_drained) {
    resources->terminal = true;
    resources->completed = true;
    state->active = false;
    state->active_resources.reset();
  }
  ++state->telemetry.epochs;

  const auto record = make_collective_record(
      *state->runtime, batch.epoch, local_success, "transport-wait",
      local_error.empty() ? std::string_view{"transport completion failed"}
                          : local_error.view());
  try {
    state->runtime->require_collective_success(record);
  } catch (const DistributedCollectiveError&) {
    throw;
  } catch (const std::exception&) {
    poison_transport(*state, "transport-wait consensus failed");
    throw;
  }
}

Transport::Transport(MpiRuntime& runtime, TransportPolicy policy) {
  runtime.require_orchestration_thread();
  std::unique_ptr<Impl> implementation;
  std::shared_ptr<TransportState> state;
  collective_try_with_fallback_at_epoch(
      runtime, 0, "transport-object-storage",
      "transport object allocation failed", -1, [&] {
        implementation = std::make_unique<Impl>();
        state = std::make_shared<TransportState>();
        state->runtime = &runtime;
      });
  implementation_ = std::move(implementation);

  const DirectCapability capability = policy == TransportPolicy::staged
      ? DirectCapability{}
      : probe_direct_mpi(runtime);
  collective_try_at_epoch(
      runtime, 0, "transport-startup",
      "transport policy resolution failed", -1, [&] {
        state->resolution = resolve_transport_policy(policy, capability);
      });

  MPI_Comm communicator = MPI_COMM_NULL;
  const int duplication_status = MPI_Comm_dup(
      detail::MpiRuntimeNativeAccess::world(runtime), &communicator);
  const CollectiveResolution duplication_resolution = runtime.consensus(
      duplication_status == MPI_SUCCESS && communicator != MPI_COMM_NULL
          ? CollectiveErrorRecord::success(
                1, runtime.rank(), "transport-communicator-duplicate")
          : CollectiveErrorRecord::failure(
                1, runtime.rank(), -1, duplication_status,
                "transport-communicator-duplicate",
                "MPI_Comm_dup failed during transport startup"));
  if (!duplication_resolution.accepted()) {
    // A failed collective duplication can leave a communicator on only a
    // subset of ranks.  Such a partial handle cannot be freed collectively.
    communicator = MPI_COMM_NULL;
    throw DistributedCollectiveError{duplication_resolution};
  }

  const int handler_status =
      MPI_Comm_set_errhandler(communicator, MPI_ERRORS_RETURN);
  require_transport_startup_phase(
      runtime, communicator, 2, handler_status == MPI_SUCCESS,
      "transport-communicator-error-handler", handler_status,
      "MPI_Comm_set_errhandler failed during transport startup");

  int* tag_upper_bound = nullptr;
  int attribute_present = 0;
  const int attribute_status = MPI_Comm_get_attr(
      communicator, MPI_TAG_UB, &tag_upper_bound, &attribute_present);
  require_transport_startup_phase(
      runtime, communicator, 3, attribute_status == MPI_SUCCESS,
      "transport-tag-query", attribute_status,
      "MPI_Comm_get_attr(MPI_TAG_UB) failed during transport startup");
  const bool tags_valid = attribute_present != 0 && tag_upper_bound != nullptr
      && *tag_upper_bound >= static_cast<int>(tag_slots_per_epoch - 1);
  require_transport_startup_phase(
      runtime, communicator, 4, tags_valid, "transport-tag-space", -1,
      "MPI_TAG_UB is too small for a communication epoch");
  state->communicator = communicator;
  state->tag_generations =
      (static_cast<std::uint64_t>(*tag_upper_bound) + 1)
      / tag_slots_per_epoch;
  implementation_->state = std::move(state);
}

Transport::~Transport() noexcept = default;

const TransportResolution& Transport::resolution() const noexcept {
  return implementation_->state->resolution;
}

const TransportTelemetry& Transport::telemetry() const noexcept {
  return implementation_->state->telemetry;
}

bool Transport::has_active_epoch() const noexcept {
  return implementation_->state->active;
}

bool Transport::poisoned() const noexcept {
  return implementation_->state->poisoned;
}

TransferBatch Transport::begin(std::span<const ByteTransfer> transfers) {
  auto state = implementation_->state;
  state->runtime->require_orchestration_thread();
  if (state->closed) throw std::logic_error{"distributed transport is closed"};
  if (state->poisoned) {
    throw std::logic_error{"distributed transport is poisoned: "
                           + state->poison_reason};
  }
  if (state->active) {
    throw std::logic_error{
        "the preceding communication epoch has not completed"};
  }

  const std::uint64_t epoch = state->next_epoch++;
  const std::size_t request_count =
      validate_transfer_descriptors_at_epoch(*state, epoch, transfers);
  validate_pairing_plan_at_epoch(*state, epoch, transfers);

  std::unique_ptr<TransferBatch::Impl> batch;
  std::shared_ptr<BatchResources> resources;
  collective_try_with_fallback_at_epoch(
      *state->runtime, epoch, "transport-batch-storage",
      "transport batch storage allocation failed", -1, [&] {
        batch = std::make_unique<TransferBatch::Impl>();
        batch->state = state;
        batch->resources = std::make_shared<BatchResources>();
        batch->epoch = epoch;
        resources = batch->resources;
      });
  state->active_resources = resources;
  state->active = true;
  const int tag_base = static_cast<int>(
      (epoch % state->tag_generations) * tag_slots_per_epoch);

  const auto started = collective_try_record(
      *state->runtime, epoch, "transport-start",
      "transport request startup failed", -1, [&] {
        start_transfer_requests(
            *state, *resources, transfers, request_count, tag_base);
      });
  require_startup_resolution(*state, *resources, started);
  finish_empty_epoch_if_collective(*state, *resources);
  return TransferBatch{std::move(batch)};
}

void Transport::close() {
  auto state = implementation_->state;
  state->runtime->require_orchestration_thread();
  if (state->closed) return;
  bool local_ready = true;
  CollectiveFixedMessage local_error;
  if (state->active) {
    if (state->active_resources == nullptr
        || state->active_resources->storage == nullptr) {
      local_ready = false;
      local_error.assign(
          "active communication epoch has no retained request storage");
    } else {
      auto resources = state->active_resources;
      const RequestCompletion cleanup =
          complete_requests(resources->storage->requests, true);
      resources->drain_proven = cleanup.drained;
      bool globally_drained = false;
      try {
        globally_drained = state->runtime->allreduce_all(cleanup.drained);
      } catch (const std::exception&) {
        poison_transport(*state, "close-drain consensus failed");
        throw;
      }
      if (!globally_drained) {
        resources->terminal = true;
        poison_transport(*state,
                         "at least one rank could not prove close-time request drainage");
        local_ready = false;
        if (!state->poison_reason.empty()) {
          local_error.assign(state->poison_reason);
        }
      } else {
        CollectiveFixedMessage receive_error;
        const LocalFailureKind receive_failure = capture_local_exception(
            receive_error, "unspecified distributed failure", [&] {
              finish_direct_receives(resources->storage->direct);
            });
        if (receive_failure != LocalFailureKind::none) {
          local_ready = false;
          if (receive_failure == LocalFailureKind::standard_exception) {
            local_error.assign(receive_error);
          }
        }
        if (!cleanup.operations_succeeded) {
          local_ready = false;
          if (local_error.empty()) {
            (void)capture_local_failure_with_fallback(
                local_error, "transport close preparation failed", [&] {
                  local_error.assign(request_completion_error(cleanup));
                });
          }
        }
        resources->terminal = true;
        resources->completed = true;
        state->active = false;
        state->active_resources.reset();
        ++state->telemetry.epochs;
      }
    }
  }
  const CollectiveResolution close_resolution = state->runtime->consensus(
      make_collective_record(
          *state->runtime, state->next_epoch, local_ready, "transport-close",
          local_error.empty()
              ? std::string_view{"transport close preparation failed"}
              : local_error.view()));

  // Never free a communicator while any rank still has an unproven active
  // request.  Every participant takes this agreement point even when its local
  // close preparation failed, keeping close retryable in the unsafe case.
  const bool safe_to_free = state->runtime->allreduce_all(!state->active);
  if (!safe_to_free) {
    poison_transport(*state,
                     "transport communicator retained after incomplete drain");
    if (!close_resolution.accepted()) {
      throw DistributedCollectiveError{close_resolution};
    }
    throw std::runtime_error{
        "transport communicator retained after incomplete drain"};
  }

  const int free_status = state->communicator == MPI_COMM_NULL
      ? MPI_SUCCESS
      : MPI_Comm_free(&state->communicator);
  if (free_status != MPI_SUCCESS) {
    poison_transport(*state, "transport communicator close failed");
  }
  CollectiveResolution communicator_resolution;
  try {
    communicator_resolution = state->runtime->consensus(
        free_status == MPI_SUCCESS
            ? CollectiveErrorRecord::success(
                  state->next_epoch, state->runtime->rank(),
                  "transport-communicator-close")
            : CollectiveErrorRecord::failure(
                  state->next_epoch, state->runtime->rank(), -1, free_status,
                  "transport-communicator-close",
                  "MPI_Comm_free failed"));
  } catch (...) {
    // MPI_Comm_free may already have invalidated the handle on some ranks.
    // Retrying could therefore call a collective on only a subset.
    state->closed = true;
    poison_transport(*state, "transport communicator close consensus failed");
    throw;
  }
  if (!communicator_resolution.accepted()) {
    state->closed = true;
    poison_transport(*state, "transport communicator close failed");
    throw DistributedCollectiveError{communicator_resolution};
  }
  state->closed = true;
  if (!close_resolution.accepted()) {
    throw DistributedCollectiveError{close_resolution};
  }
}

}  // namespace quasar::distributed
