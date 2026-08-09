#pragma once

#include "fixed_message.hpp"
#include "runtime_helpers.hpp"

#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace quasar::distributed {

using CollectiveFixedMessage = FixedMessage<collective_message_capacity>;

inline CollectiveErrorRecord make_collective_record(
    MpiRuntime& runtime, std::uint64_t epoch, bool local_success,
    std::string_view phase, std::string_view message, int code = -1) {
  return local_success
      ? CollectiveErrorRecord::success(epoch, runtime.rank(), phase)
      : CollectiveErrorRecord::failure(
            epoch, runtime.rank(), -1, code, phase, message);
}

inline void collective_require_at_epoch(
    MpiRuntime& runtime, std::uint64_t epoch, bool local_success,
    std::string_view phase, std::string_view message, int code = -1) {
  runtime.require_collective_success(make_collective_record(
      runtime, epoch, local_success, phase, message, code));
}

inline void collective_require(MpiRuntime& runtime, std::uint64_t& epoch,
                               bool local_success, std::string_view phase,
                               std::string_view message, int code = -1) {
  const CollectiveErrorRecord record = make_collective_record(
      runtime, epoch, local_success, phase, message, code);
  ++epoch;
  runtime.require_collective_success(record);
}

template <class Operation>
CollectiveErrorRecord collective_try_record(
    MpiRuntime& runtime, std::uint64_t epoch, std::string_view phase,
    std::string_view fallback, int code, Operation&& operation) {
  CollectiveFixedMessage error;
  const bool success = capture_local_failure(
      error, fallback, std::forward<Operation>(operation));
  return make_collective_record(runtime, epoch, success, phase, error, code);
}

template <class Operation>
CollectiveErrorRecord collective_try_with_fallback_record(
    MpiRuntime& runtime, std::uint64_t epoch, std::string_view phase,
    std::string_view failure, int code, Operation&& operation) {
  CollectiveFixedMessage error;
  const bool success = capture_local_failure_with_fallback(
      error, failure, std::forward<Operation>(operation));
  return make_collective_record(runtime, epoch, success, phase, error, code);
}

template <class Operation>
void collective_try_at_epoch(
    MpiRuntime& runtime, std::uint64_t epoch, std::string_view phase,
    std::string_view fallback, int code, Operation&& operation) {
  runtime.require_collective_success(collective_try_record(
      runtime, epoch, phase, fallback, code,
      std::forward<Operation>(operation)));
}

template <class Operation>
void collective_try_with_fallback_at_epoch(
    MpiRuntime& runtime, std::uint64_t epoch, std::string_view phase,
    std::string_view failure, int code, Operation&& operation) {
  runtime.require_collective_success(collective_try_with_fallback_record(
      runtime, epoch, phase, failure, code,
      std::forward<Operation>(operation)));
}

// Execute a rank-local preparation step and converge on its outcome before
// any rank can enter the next collective.  The failure record stays in fixed
// storage so even std::bad_alloc follows the same consensus path.
template <class Operation>
void collective_try(MpiRuntime& runtime, std::uint64_t& epoch,
                    std::string_view phase, std::string_view fallback,
                    int code, Operation&& operation) {
  const CollectiveErrorRecord record = collective_try_record(
      runtime, epoch, phase, fallback, code,
      std::forward<Operation>(operation));
  ++epoch;
  runtime.require_collective_success(record);
}

// Some existing collective contracts intentionally expose one stable message
// for every exception (most often allocation failure), rather than forwarding
// implementation-specific exception text.
template <class Operation>
void collective_try_with_fallback(MpiRuntime& runtime, std::uint64_t& epoch,
                                  std::string_view phase,
                                  std::string_view failure, int code,
                                  Operation&& operation) {
  const CollectiveErrorRecord record = collective_try_with_fallback_record(
      runtime, epoch, phase, failure, code,
      std::forward<Operation>(operation));
  ++epoch;
  runtime.require_collective_success(record);
}

// Build one task per local endpoint under the same allocation-consensus
// contract used by both physics runtimes. `prepare` runs inside the guarded
// local phase so test-only one-shot fault injection can remain physics-owned
// without duplicating the try/catch/record machinery.
template <class Prepare, class Function>
std::vector<WorkerTask> collectively_indexed_tasks(
    MpiRuntime& runtime, std::uint64_t& epoch, std::size_t count,
    std::string_view storage_phase, std::string_view storage_failure,
    Prepare&& prepare, Function&& function) {
  std::vector<WorkerTask> tasks;
  collective_try_with_fallback(
      runtime, epoch, storage_phase, storage_failure, -1, [&] {
        std::forward<Prepare>(prepare)();
        tasks = local_indexed_tasks(
            count, std::forward<Function>(function));
      });
  return tasks;
}

template <class Function>
std::vector<WorkerTask> collectively_indexed_tasks(
    MpiRuntime& runtime, std::uint64_t& epoch, std::size_t count,
    std::string_view storage_phase, std::string_view storage_failure,
    Function&& function) {
  return collectively_indexed_tasks(
      runtime, epoch, count, storage_phase, storage_failure, [] {},
      std::forward<Function>(function));
}

template <class Operation>
void collective_try_if(MpiRuntime& runtime, std::uint64_t& epoch,
                       bool precondition,
                       std::string_view precondition_failure,
                       std::string_view phase, std::string_view fallback,
                       int code, Operation&& operation) {
  CollectiveFixedMessage error;
  bool success = precondition;
  if (success) {
    success = capture_local_failure(
        error, fallback, std::forward<Operation>(operation));
  } else {
    error.assign(precondition_failure);
  }
  collective_require(runtime, epoch, success, phase, error, code);
}

template <class Predicate>
void collective_require_predicate(
    MpiRuntime& runtime, std::uint64_t& epoch, std::string_view phase,
    std::string_view rejection, std::string_view exception_failure, int code,
    Predicate&& predicate) {
  bool accepted = false;
  std::string_view message = rejection;
  try {
    accepted = static_cast<bool>(std::forward<Predicate>(predicate)());
  } catch (const std::exception&) {
    message = exception_failure;
  } catch (...) {
    message = exception_failure;
  }
  collective_require(runtime, epoch, accepted, phase, message, code);
}

template <class Operation>
void collective_try_on_root(MpiRuntime& runtime, std::uint64_t& epoch,
                            std::string_view phase,
                            std::string_view fallback, int code,
                            Operation&& operation, int root = 0) {
  CollectiveFixedMessage error;
  bool success = true;
  if (runtime.rank() == root) {
    success = capture_local_failure(
        error, fallback, std::forward<Operation>(operation));
  }
  collective_require(runtime, epoch, success, phase, error, code);
}

template <class Operation, class FailureOperation>
void collective_try_on_root_with_failure(
    MpiRuntime& runtime, std::uint64_t& epoch, std::string_view phase,
    std::string_view fallback, int code, Operation&& operation,
    FailureOperation&& failure_operation, int root = 0) {
  static_assert(std::is_nothrow_invocable_v<FailureOperation>);
  CollectiveFixedMessage error;
  bool success = true;
  if (runtime.rank() == root) {
    success = capture_local_failure(
        error, fallback, std::forward<Operation>(operation));
    if (!success) {
      std::invoke(std::forward<FailureOperation>(failure_operation));
    }
  }
  collective_require(runtime, epoch, success, phase, error, code);
}

struct CollectiveStringBroadcastMessages {
  std::string_view preparation_failure;
  std::string_view range_failure;
  std::string_view allocation_failure;
  const char* length_operation;
  const char* bytes_operation;
  int preparation_code{-1};
  int range_code{-1};
  int allocation_code{-1};
};

inline std::string collective_broadcast_string(
    MpiRuntime& runtime, std::uint64_t& epoch,
    const std::string& root_value, std::string_view phase,
    const CollectiveStringBroadcastMessages& messages, int root = 0) {
  runtime.require_orchestration_thread();
  std::string value;
  collective_try(runtime, epoch, phase, messages.preparation_failure,
                 messages.preparation_code, [&] {
                   if (runtime.rank() == root) value = root_value;
                 });

  std::uint64_t length = runtime.rank() == root ? value.size() : 0;
  check_mpi(MPI_Bcast(&length, 1, MPI_UINT64_T, root,
                      detail::MpiRuntimeNativeAccess::world(runtime)),
            messages.length_operation);
  collective_require(
      runtime, epoch,
      length <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
      phase, messages.range_failure, messages.range_code);
  collective_try_with_fallback(
      runtime, epoch, phase, messages.allocation_failure,
      messages.allocation_code, [&] {
        if (runtime.rank() != root) {
          value.resize(static_cast<std::size_t>(length));
        }
      });
  if (length != 0) {
    check_mpi(MPI_Bcast(value.data(), static_cast<int>(length), MPI_CHAR,
                        root, detail::MpiRuntimeNativeAccess::world(runtime)),
              messages.bytes_operation);
  }
  return value;
}

inline void collective_require_common_string(
    MpiRuntime& runtime, std::uint64_t& epoch, const std::string& local,
    std::string_view phase,
    const CollectiveStringBroadcastMessages& broadcast_messages,
    std::string_view mismatch_message, int mismatch_code = -1,
    int root = 0) {
  const std::string root_value = collective_broadcast_string(
      runtime, epoch, local, phase, broadcast_messages, root);
  collective_require(runtime, epoch, local == root_value, phase,
                     mismatch_message, mismatch_code);
}

struct RootPublicationNotification {
  int publication_error{0};
  int mpi_status{MPI_SUCCESS};
};

// The root operation must report errno-style status and must not throw: this
// helper is used only after every rank has agreed that an irreversible atomic
// rename may proceed.
template <class Operation>
RootPublicationNotification publish_from_root(
    MpiRuntime& runtime, Operation&& operation, int root = 0) {
  static_assert(std::is_nothrow_invocable_r_v<int, Operation>);
  RootPublicationNotification result;
  if (runtime.rank() == root) {
    result.publication_error = std::invoke(std::forward<Operation>(operation));
  }
  result.mpi_status = MPI_Bcast(
      &result.publication_error, 1, MPI_INT, root,
      detail::MpiRuntimeNativeAccess::world(runtime));
  return result;
}

}  // namespace quasar::distributed
