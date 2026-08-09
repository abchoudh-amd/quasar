#include "quasar/distributed/mpi_runtime.hpp"
#include "quasar/distributed/mpi_device_mapping.hpp"

#include <atomic>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <thread>

int main(int argc, char** argv) {
  try {
    quasar::distributed::MpiRuntime runtime{&argc, &argv};
    if (runtime.size() < 2) {
      if (runtime.rank() == 0) {
        std::cerr << "test_mpi_runtime requires at least two MPI ranks\n";
      }
      runtime.close();
      return 77;
    }

    const std::uint64_t local = static_cast<std::uint64_t>(runtime.rank() + 1);
    const std::uint64_t expected =
        static_cast<std::uint64_t>(runtime.size())
        * static_cast<std::uint64_t>(runtime.size() + 1) / 2;
    if (runtime.allreduce_sum(local) != expected) return 1;

    const auto status = runtime.rank() == 0
        ? quasar::distributed::CollectiveErrorRecord::retry(
              4, runtime.rank(), 0, 19, "positivity", "retry requested")
        : quasar::distributed::CollectiveErrorRecord::success(
              4, runtime.rank(), "positivity");
    const auto resolution = runtime.consensus(status);
    if (resolution.decision
            != quasar::distributed::CollectiveDecision::retry
        || resolution.representative.rank != 0
        || resolution.retry_count != 1) {
      return 2;
    }

    const auto mismatched_epoch =
        quasar::distributed::CollectiveErrorRecord::success(
            runtime.rank() == 0 ? 7 : 8, runtime.rank(), "epoch-check");
    const auto epoch_resolution = runtime.consensus(mismatched_epoch);
    if (epoch_resolution.decision
            != quasar::distributed::CollectiveDecision::fail
        || epoch_resolution.representative.code
            != quasar::distributed::collective_protocol_error
        || epoch_resolution.representative.message_text()
            != "collective records disagree on communication epoch") {
      return 7;
    }

    const auto mismatched_phase =
        quasar::distributed::CollectiveErrorRecord::success(
            9, runtime.rank(), runtime.rank() == 0 ? "phase-a" : "phase-b");
    const auto phase_resolution = runtime.consensus(mismatched_phase);
    if (phase_resolution.decision
            != quasar::distributed::CollectiveDecision::fail
        || phase_resolution.representative.code
            != quasar::distributed::collective_protocol_error
        || phase_resolution.representative.message_text()
            != "collective records disagree on execution phase") {
      return 8;
    }

    std::atomic<bool> rejected{false};
    std::thread non_orchestration_thread{[&] {
      try {
        runtime.barrier();
      } catch (const std::logic_error&) {
        rejected.store(true);
      }
    }};
    non_orchestration_thread.join();
    if (!rejected.load()) return 3;

    const int selected_ordinal = runtime.rank() == 0 ? 0 : 1;
    bool mismatched_selection_rejected = false;
    try {
      (void)quasar::distributed::discover_endpoint_mapping(
          runtime, std::span<const int>{&selected_ordinal, 1});
    } catch (const quasar::distributed::DistributedCollectiveError& error) {
      mismatched_selection_rejected =
          error.resolution().representative.phase_text()
          == "device-selection-agreement";
    }
    if (!mismatched_selection_rejected) return 5;

    runtime.close();
    if (!runtime.closed()) return 6;
    runtime.close();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 4;
  }
}
