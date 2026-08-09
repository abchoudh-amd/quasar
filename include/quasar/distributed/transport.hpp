#pragma once

#include "quasar/backend/device.hpp"
#include "quasar/distributed/mpi_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

namespace quasar::distributed {

enum class TransportPolicy {
  automatic,
  staged,
  direct,
};

enum class TransportPath {
  peer_copy,
  staged_mpi,
  direct_mpi,
};

enum class BufferResidence {
  host,
  device,
};

struct DirectCapability {
  bool recognized_query{false};
  bool startup_probe{false};

  [[nodiscard]] constexpr bool available() const noexcept {
    return recognized_query && startup_probe;
  }
};

struct TransportResolution {
  TransportPolicy requested{TransportPolicy::automatic};
  TransportPolicy interprocess{TransportPolicy::staged};
  DirectCapability direct{};

  [[nodiscard]] constexpr bool uses_direct_mpi() const noexcept {
    return interprocess == TransportPolicy::direct;
  }
};

[[nodiscard]] TransportPolicy parse_transport_policy(std::string_view value);
[[nodiscard]] TransportResolution resolve_transport_policy(
    TransportPolicy requested, DirectCapability capability);

// One already-packed byte payload.  Pack/unpack and additive reduction remain
// worker operations; this layer moves the packed representation only.  For a
// node-local transfer, send_device and receive_device identify the two endpoint
// owners.  For an inter-process transfer they identify the local source and
// destination allocation owners used by staged copies.  A non-empty
// inter-process (peer_rank, tag_slot) pair must be unique within one begin()
// call.  All caller-owned buffers and streams must remain alive until the
// returned batch reports complete or Transport::close() completes.
struct ByteTransfer {
  int peer_rank{-1};
  std::uint32_t tag_slot{0};
  const void* send_buffer{nullptr};
  void* receive_buffer{nullptr};
  std::size_t bytes{0};
  BufferResidence residence{BufferResidence::device};
  int send_device{-1};
  int receive_device{-1};
  backend::stream_t send_stream{nullptr};
  backend::stream_t receive_stream{nullptr};
};

struct TransportTelemetry {
  std::uint64_t epochs{0};
  std::uint64_t messages{0};
  std::uint64_t bytes{0};
  std::uint64_t peer_bytes{0};
  std::uint64_t local_staged_bytes{0};
  std::uint64_t staged_mpi_bytes{0};
  std::uint64_t direct_mpi_bytes{0};
};

class Transport;

// A batch owns all MPI requests and staging buffers for one communication
// epoch.  wait() is explicit and completes every request before the transport
// permits another epoch/tag reuse.  Its destructor deliberately performs no
// MPI calls.
class TransferBatch {
 public:
  TransferBatch() noexcept = default;
  ~TransferBatch() noexcept;
  TransferBatch(TransferBatch&&) noexcept;
  TransferBatch& operator=(TransferBatch&&) noexcept;
  TransferBatch(const TransferBatch&) = delete;
  TransferBatch& operator=(const TransferBatch&) = delete;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] bool complete() const noexcept;
  [[nodiscard]] std::uint64_t epoch() const noexcept;
  void wait();

 private:
  struct Impl;
  explicit TransferBatch(std::unique_ptr<Impl> implementation) noexcept;
  std::unique_ptr<Impl> implementation_{};
  friend class Transport;
};

class Transport {
 public:
  static constexpr std::uint32_t tag_slots_per_epoch = 64;

  explicit Transport(MpiRuntime& runtime,
                     TransportPolicy policy = TransportPolicy::automatic);
  ~Transport() noexcept;
  Transport(const Transport&) = delete;
  Transport& operator=(const Transport&) = delete;
  Transport(Transport&&) = delete;
  Transport& operator=(Transport&&) = delete;

  [[nodiscard]] const TransportResolution& resolution() const noexcept;
  [[nodiscard]] const TransportTelemetry& telemetry() const noexcept;
  [[nodiscard]] bool has_active_epoch() const noexcept;
  [[nodiscard]] bool poisoned() const noexcept;

  [[nodiscard]] TransferBatch begin(std::span<const ByteTransfer> transfers);
  void close();

 private:
  struct Impl;
  std::unique_ptr<Impl> implementation_;
  friend class TransferBatch;
};

[[nodiscard]] DirectCapability probe_direct_mpi(MpiRuntime& runtime);

}  // namespace quasar::distributed
