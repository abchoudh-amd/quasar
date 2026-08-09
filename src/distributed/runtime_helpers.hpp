#pragma once

#include "mpi_runtime_native.hpp"

#include "quasar/distributed/worker_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace quasar::distributed {

inline std::size_t checked_product(std::size_t left, std::size_t right,
                                   const char* label) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::length_error{label};
  }
  return left * right;
}

inline void capture_error_text(
    std::string& destination, const char* message,
    std::string_view fallback = "unspecified distributed failure") noexcept {
  try {
    destination.assign(message == nullptr ? fallback : std::string_view{message});
  } catch (...) {
    // Callers supply a fixed literal to their collective record when this
    // best-effort diagnostic allocation is unavailable.
  }
}

inline std::string_view error_text_or(
    const std::string& value, std::string_view fallback) noexcept {
  return value.empty() ? fallback : std::string_view{value};
}

inline std::size_t row_major(std::size_t x, std::size_t y,
                             std::size_t width) noexcept {
  return y * width + x;
}

inline void hash_bytes(std::uint64_t& hash, const void* data,
                       std::size_t size) noexcept {
  constexpr std::uint64_t prime = 1099511628211ULL;
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t index = 0; index < size; ++index) {
    hash ^= bytes[index];
    hash *= prime;
  }
}

template <class T>
inline void hash_span(std::uint64_t& hash, std::span<const T> values) noexcept {
  hash_bytes(hash, values.data(), values.size_bytes());
}

template <class T>
inline void hash_scalar(std::uint64_t& hash, const T& value) noexcept {
  hash_bytes(hash, &value, sizeof(value));
}

inline void hash_string(std::uint64_t& hash, std::string_view value) noexcept {
  hash_bytes(hash, value.data(), value.size());
  constexpr unsigned char terminator = 0xff;
  hash_scalar(hash, terminator);
}

template <class T>
inline void require_finite(std::span<const T> values, const char* label) {
  if (!std::all_of(values.begin(), values.end(),
                   [](T value) { return std::isfinite(value); })) {
    throw std::invalid_argument{std::string{label}
                                + " must contain only finite values"};
  }
}

template <class T>
inline void allreduce_sum_in_place(MpiRuntime& runtime,
                                   std::vector<T>& values,
                                   MPI_Datatype datatype,
                                   const char* operation) {
  runtime.require_orchestration_thread();
  constexpr std::size_t maximum =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  std::size_t offset = 0;
  while (offset < values.size()) {
    const int count = static_cast<int>(
        std::min(maximum, values.size() - offset));
    check_mpi(MPI_Allreduce(
                  MPI_IN_PLACE, values.data() + offset, count, datatype,
                  MPI_SUM, detail::MpiRuntimeNativeAccess::world(runtime)),
              operation);
    offset += static_cast<std::size_t>(count);
  }
}

inline std::size_t allreduce_max_size(MpiRuntime& runtime,
                                      std::size_t local,
                                      const char* operation) {
  static_assert(std::numeric_limits<std::size_t>::digits <=
                std::numeric_limits<std::uint64_t>::digits);
  runtime.require_orchestration_thread();
  const std::uint64_t local_value = static_cast<std::uint64_t>(local);
  std::uint64_t global_value = 0;
  check_mpi(MPI_Allreduce(
                &local_value, &global_value, 1, MPI_UINT64_T, MPI_MAX,
                detail::MpiRuntimeNativeAccess::world(runtime)),
            operation);
  return static_cast<std::size_t>(global_value);
}

inline void require_exact_coverage(std::span<const int> counts,
                                   std::string_view component) {
  if (std::find_if(counts.begin(), counts.end(),
                   [](int count) { return count != 1; }) != counts.end()) {
    throw std::runtime_error{"distributed ownership does not cover "
                             + std::string{component} + " exactly once"};
  }
}

inline void require_worker_success(EndpointWorkerPool& workers,
                                   MpiRuntime& runtime,
                                   std::uint64_t& epoch,
                                   std::string_view phase,
                                   std::span<const WorkerTask> tasks) {
  const auto resolution =
      workers.execute_collective(runtime, epoch++, phase, tasks);
  if (!resolution.accepted()) throw DistributedCollectiveError{resolution};
}

template <class Function>
inline std::vector<WorkerTask> local_indexed_tasks(std::size_t count,
                                                   Function function) {
  std::vector<WorkerTask> tasks;
  tasks.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    tasks.emplace_back([index, function](WorkerContext& context) mutable {
      function(index, context);
    });
  }
  return tasks;
}

[[noreturn]] inline void poison_runtime_collectively(
    MpiRuntime& runtime, std::uint64_t& epoch, bool& poisoned,
    std::string_view phase, std::string_view local_message) {
  poisoned = true;
  const auto resolution = runtime.consensus(CollectiveErrorRecord::failure(
      epoch++, runtime.rank(), -1, -1, phase, local_message));
  throw DistributedCollectiveError{resolution};
}

}  // namespace quasar::distributed
