#pragma once

#include "quasar/backend/memory.hpp"
#include "quasar/distributed/mpi_runtime.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace quasar::distributed {

struct WorkerContext {
  std::size_t endpoint{0};
  std::size_t rank_local_index{0};
  int device_ordinal{-1};
  backend::DeviceStream compute_stream;
  backend::DeviceStream communication_stream;
  backend::DeviceEvent compute_ready;
  backend::DeviceEvent communication_ready;

  WorkerContext(std::size_t endpoint_index,
                std::size_t local_index,
                int device);
};

class RetryableWorkerError : public std::runtime_error {
 public:
  RetryableWorkerError(int code, std::string message)
    : std::runtime_error{std::move(message)}, code_{code} {}
  [[nodiscard]] int code() const noexcept { return code_; }

 private:
  int code_{0};
};

using WorkerTask = std::function<void(WorkerContext&)>;

// One persistent thread per local GPU endpoint.  Worker threads own device
// selection, streams, events, and all task-side HIP calls; the orchestration
// thread alone submits phases and performs MPI/HDF5 operations.
class EndpointWorkerPool {
 public:
  EndpointWorkerPool(std::span<const int> device_ordinals,
                     std::size_t first_global_endpoint = 0);
  ~EndpointWorkerPool() noexcept;
  EndpointWorkerPool(const EndpointWorkerPool&) = delete;
  EndpointWorkerPool& operator=(const EndpointWorkerPool&) = delete;
  EndpointWorkerPool(EndpointWorkerPool&&) = delete;
  EndpointWorkerPool& operator=(EndpointWorkerPool&&) = delete;

  [[nodiscard]] std::size_t size() const noexcept;
  [[nodiscard]] bool closed() const noexcept;

  // Execute exactly one task on every worker.  All tasks are released before
  // the orchestration thread waits, so independent endpoint kernels overlap.
  // A local exception is converted to one deterministic rank record after all
  // workers finish; no worker exception escapes while peers are still running.
  [[nodiscard]] CollectiveErrorRecord execute(
      std::uint64_t epoch, int rank, std::string_view phase,
      std::span<const WorkerTask> tasks);
  [[nodiscard]] CollectiveErrorRecord execute(
      std::uint64_t epoch, int rank, std::string_view phase,
      const WorkerTask& task_for_every_endpoint);

  [[nodiscard]] CollectiveResolution execute_collective(
      MpiRuntime& runtime, std::uint64_t epoch, std::string_view phase,
      std::span<const WorkerTask> tasks);
  [[nodiscard]] CollectiveResolution execute_collective(
      MpiRuntime& runtime, std::uint64_t epoch, std::string_view phase,
      const WorkerTask& task_for_every_endpoint);

  // Local and non-collective. Distributed owners call this only as part of
  // their explicit collective close protocol.
  void close();

 private:
  struct Impl;
  std::unique_ptr<Impl> implementation_;
};

}  // namespace quasar::distributed
