#include "quasar/distributed/worker_pool.hpp"

#include "quasar/backend/device.hpp"

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

namespace quasar::distributed {

WorkerContext::WorkerContext(std::size_t endpoint_index,
                             std::size_t local_index,
                             int device)
  : endpoint{endpoint_index},
    rank_local_index{local_index},
    device_ordinal{device},
    compute_stream{backend::on_device(device)},
    communication_stream{backend::on_device(device)},
    compute_ready{backend::on_device(device)},
    communication_ready{backend::on_device(device)} {}

struct EndpointWorkerPool::Impl {
  struct Slot {
    std::size_t endpoint{0};
    std::size_t local_index{0};
    int device{-1};
    std::thread thread{};
    std::unique_ptr<WorkerContext> context{};
    WorkerTask task{};
    std::exception_ptr error{};
    std::uint64_t requested_generation{0};
    std::uint64_t completed_generation{0};
    bool ready{false};
    bool stop{false};
  };

  std::thread::id orchestration_thread{std::this_thread::get_id()};
  std::mutex mutex{};
  std::condition_variable command_ready{};
  std::condition_variable phase_complete{};
  std::vector<std::unique_ptr<Slot>> slots{};
  std::uint64_t generation{0};
  bool closed{false};

  void require_orchestration_thread() const {
    if (std::this_thread::get_id() != orchestration_thread) {
      throw std::logic_error{
          "GPU worker phases must be submitted by the orchestration thread"};
    }
  }

  void worker_main(Slot& slot) noexcept {
    try {
      backend::set_device(slot.device);
      slot.context = std::make_unique<WorkerContext>(
          slot.endpoint, slot.local_index, slot.device);
    } catch (...) {
      slot.error = std::current_exception();
    }
    {
      std::lock_guard lock{mutex};
      slot.ready = true;
    }
    phase_complete.notify_one();

    std::unique_lock lock{mutex};
    while (true) {
      command_ready.wait(lock, [&] {
        return slot.stop
            || slot.requested_generation > slot.completed_generation;
      });
      if (slot.stop) break;
      const std::uint64_t active_generation = slot.requested_generation;
      WorkerTask task = std::move(slot.task);
      slot.error = nullptr;
      lock.unlock();
      try {
        if (!slot.context) {
          throw std::runtime_error{"GPU worker initialization failed"};
        }
        task(*slot.context);
      } catch (...) {
        slot.error = std::current_exception();
      }
      lock.lock();
      slot.completed_generation = active_generation;
      phase_complete.notify_one();
    }
    // Context teardown happens on its owning worker thread.
    slot.context.reset();
  }

  void close_noexcept() noexcept {
    {
      std::lock_guard lock{mutex};
      if (closed) return;
      for (auto& slot : slots) slot->stop = true;
      closed = true;
    }
    command_ready.notify_all();
    for (auto& slot : slots) {
      if (slot->thread.joinable()) slot->thread.join();
    }
  }
};

namespace {

std::string exception_message(const std::exception_ptr& error) {
  try {
    std::rethrow_exception(error);
  } catch (const std::exception& exception) {
    return exception.what();
  } catch (...) {
    return "non-standard worker exception";
  }
}

}  // namespace

EndpointWorkerPool::EndpointWorkerPool(
    std::span<const int> device_ordinals,
    std::size_t first_global_endpoint)
  : implementation_{std::make_unique<Impl>()} {
  if (device_ordinals.empty()) {
    throw std::invalid_argument{
        "a distributed rank must own at least one GPU worker"};
  }
  if (first_global_endpoint
          > std::numeric_limits<std::size_t>::max() - device_ordinals.size()
      || first_global_endpoint + device_ordinals.size() - 1
          > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    throw std::overflow_error{"global GPU endpoint indices overflow"};
  }
  std::vector<int> seen;
  seen.reserve(device_ordinals.size());
  for (const int device : device_ordinals) {
    if (device < 0
        || std::find(seen.begin(), seen.end(), device) != seen.end()) {
      throw std::invalid_argument{
          "rank-local GPU worker ordinals must be unique and non-negative"};
    }
    seen.push_back(device);
  }

  Impl& impl = *implementation_;
  impl.slots.reserve(device_ordinals.size());
  try {
    for (std::size_t local = 0; local < device_ordinals.size(); ++local) {
      auto slot = std::make_unique<Impl::Slot>();
      slot->endpoint = first_global_endpoint + local;
      slot->local_index = local;
      slot->device = device_ordinals[local];
      Impl::Slot* slot_pointer = slot.get();
      impl.slots.push_back(std::move(slot));
      slot_pointer->thread = std::thread{[&impl, slot_pointer] {
        impl.worker_main(*slot_pointer);
      }};
    }
  } catch (...) {
    impl.close_noexcept();
    throw;
  }

  std::unique_lock lock{impl.mutex};
  impl.phase_complete.wait(lock, [&] {
    return std::all_of(impl.slots.begin(), impl.slots.end(),
                       [](const auto& slot) { return slot->ready; });
  });
  std::exception_ptr initialization_error;
  for (const auto& slot : impl.slots) {
    if (slot->error) {
      initialization_error = slot->error;
      break;
    }
  }
  lock.unlock();
  if (initialization_error) {
    impl.close_noexcept();
    std::rethrow_exception(initialization_error);
  }
}

EndpointWorkerPool::~EndpointWorkerPool() noexcept {
  if (implementation_) implementation_->close_noexcept();
}

std::size_t EndpointWorkerPool::size() const noexcept {
  return implementation_->slots.size();
}

bool EndpointWorkerPool::closed() const noexcept {
  return implementation_->closed;
}

CollectiveErrorRecord EndpointWorkerPool::execute(
    std::uint64_t epoch, int rank, std::string_view phase,
    std::span<const WorkerTask> tasks) {
  Impl& impl = *implementation_;
  impl.require_orchestration_thread();
  if (impl.closed) throw std::logic_error{"GPU worker pool is closed"};
  if (tasks.size() != impl.slots.size()) {
    throw std::invalid_argument{
        "GPU worker phase requires exactly one task per endpoint"};
  }

  std::uint64_t active_generation = 0;
  {
    std::lock_guard lock{impl.mutex};
    active_generation = ++impl.generation;
    for (std::size_t index = 0; index < impl.slots.size(); ++index) {
      auto& slot = *impl.slots[index];
      slot.task = tasks[index];
      slot.requested_generation = active_generation;
    }
  }
  impl.command_ready.notify_all();
  std::unique_lock lock{impl.mutex};
  impl.phase_complete.wait(lock, [&] {
    return std::all_of(
        impl.slots.begin(), impl.slots.end(),
        [active_generation](const auto& slot) {
          return slot->completed_generation >= active_generation;
        });
  });

  std::optional<CollectiveErrorRecord> first_retry;
  std::optional<CollectiveErrorRecord> first_failure;
  for (const auto& slot : impl.slots) {
    if (!slot->error) continue;
    try {
      std::rethrow_exception(slot->error);
    } catch (const RetryableWorkerError& error) {
      if (!first_retry) {
        first_retry = CollectiveErrorRecord::retry(
            epoch, rank, static_cast<std::int32_t>(slot->endpoint),
            error.code(), phase, error.what());
      }
    } catch (...) {
      if (!first_failure) {
        first_failure = CollectiveErrorRecord::failure(
            epoch, rank, static_cast<std::int32_t>(slot->endpoint), -1,
            phase, exception_message(slot->error));
      }
    }
  }
  if (first_failure) return *first_failure;
  if (first_retry) return *first_retry;
  return CollectiveErrorRecord::success(epoch, rank, phase);
}

CollectiveErrorRecord EndpointWorkerPool::execute(
    std::uint64_t epoch, int rank, std::string_view phase,
    const WorkerTask& task_for_every_endpoint) {
  std::vector<WorkerTask> tasks(size(), task_for_every_endpoint);
  return execute(epoch, rank, phase, tasks);
}

CollectiveResolution EndpointWorkerPool::execute_collective(
    MpiRuntime& runtime, std::uint64_t epoch, std::string_view phase,
    std::span<const WorkerTask> tasks) {
  CollectiveErrorRecord record;
  try {
    record = execute(epoch, runtime.rank(), phase, tasks);
  } catch (const std::exception& error) {
    record = CollectiveErrorRecord::failure(
        epoch, runtime.rank(), -1, -1, phase, error.what());
  } catch (...) {
    record = CollectiveErrorRecord::failure(
        epoch, runtime.rank(), -1, -1, phase,
        "non-standard worker-pool orchestration exception");
  }
  return runtime.consensus(record);
}

CollectiveResolution EndpointWorkerPool::execute_collective(
    MpiRuntime& runtime, std::uint64_t epoch, std::string_view phase,
    const WorkerTask& task_for_every_endpoint) {
  CollectiveErrorRecord record;
  try {
    record = execute(epoch, runtime.rank(), phase, task_for_every_endpoint);
  } catch (const std::exception& error) {
    record = CollectiveErrorRecord::failure(
        epoch, runtime.rank(), -1, -1, phase, error.what());
  } catch (...) {
    record = CollectiveErrorRecord::failure(
        epoch, runtime.rank(), -1, -1, phase,
        "non-standard worker-pool orchestration exception");
  }
  return runtime.consensus(record);
}

void EndpointWorkerPool::close() {
  implementation_->require_orchestration_thread();
  implementation_->close_noexcept();
}

}  // namespace quasar::distributed
