#include "quasar/distributed/transport.hpp"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

namespace {

std::vector<quasar::distributed::ByteTransfer> host_exchange(
    int rank, std::uint32_t tag, const void* send, void* receive,
    std::size_t bytes) {
  if (rank > 1) return {};
  quasar::distributed::ByteTransfer transfer;
  transfer.peer_rank = 1 - rank;
  transfer.tag_slot = tag;
  transfer.send_buffer = send;
  transfer.receive_buffer = receive;
  transfer.bytes = bytes;
  transfer.residence = quasar::distributed::BufferResidence::host;
  return {transfer};
}

bool expect_collective_rejection(
    quasar::distributed::Transport& transport,
    std::span<const quasar::distributed::ByteTransfer> transfers) {
  try {
    (void)transport.begin(transfers);
  } catch (const quasar::distributed::DistributedCollectiveError&) {
    return !transport.has_active_epoch() && !transport.poisoned();
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    quasar::distributed::MpiRuntime runtime{&argc, &argv};
    if (runtime.size() < 2) {
      if (runtime.rank() == 0) {
        std::cerr << "test_transport_mpi requires at least two MPI ranks\n";
      }
      runtime.close();
      return 77;
    }

    quasar::distributed::Transport transport{
        runtime, quasar::distributed::TransportPolicy::staged};
    int local_failures = 0;

    // Explicit direct policy is a collective contract: when the recognized
    // ROCm-aware query/startup probe is unavailable every rank rejects it,
    // rather than silently selecting staged MPI or throwing on only one rank.
    const auto direct_capability =
        quasar::distributed::probe_direct_mpi(runtime);
    bool direct_rejected = false;
    try {
      quasar::distributed::Transport direct{
          runtime, quasar::distributed::TransportPolicy::direct};
      direct.close();
    } catch (const quasar::distributed::DistributedCollectiveError&) {
      direct_rejected = true;
    }
    const bool every_rank_rejected = runtime.allreduce_all(direct_rejected);
    const bool every_rank_accepted = runtime.allreduce_all(!direct_rejected);
    if (!every_rank_rejected && !every_rank_accepted) ++local_failures;
    if (direct_capability.available() != every_rank_accepted) {
      ++local_failures;
    }

    // A valid host exchange exercises ranks with and without local requests:
    // ranks above one still participate in wait() and epoch completion.
    std::uint64_t send = static_cast<std::uint64_t>(runtime.rank() + 100);
    std::uint64_t receive = 0;
    auto transfers = host_exchange(
        runtime.rank(), 7, &send, &receive, sizeof(send));
    auto batch = transport.begin(transfers);
    batch.wait();
    if (!batch.complete() || transport.has_active_epoch()) ++local_failures;
    if (runtime.rank() < 2
        && receive != static_cast<std::uint64_t>((1 - runtime.rank()) + 100)) {
      ++local_failures;
    }
    if (transport.telemetry().staged_mpi_bytes != sizeof(send)
        || transport.telemetry().direct_mpi_bytes != 0) {
      ++local_failures;
    }

    // Distinct transports own distinct MPI communication contexts, so equal
    // epoch/tag values may be in flight concurrently without cross-matching.
    quasar::distributed::Transport isolated_transport{
        runtime, quasar::distributed::TransportPolicy::staged};
    std::uint64_t first_send =
        static_cast<std::uint64_t>(runtime.rank() + 1000);
    std::uint64_t first_receive = 0;
    std::uint64_t second_send =
        static_cast<std::uint64_t>(runtime.rank() + 2000);
    std::uint64_t second_receive = 0;
    auto first_transfers = host_exchange(
        runtime.rank(), 0, &first_send, &first_receive, sizeof(first_send));
    auto second_transfers = host_exchange(
        runtime.rank(), 0, &second_send, &second_receive, sizeof(second_send));
    auto first_batch = transport.begin(first_transfers);
    auto second_batch = isolated_transport.begin(second_transfers);
    second_batch.wait();
    first_batch.wait();
    if (runtime.rank() < 2) {
      if (first_receive
          != static_cast<std::uint64_t>((1 - runtime.rank()) + 1000)) {
        ++local_failures;
      }
      if (second_receive
          != static_cast<std::uint64_t>((1 - runtime.rank()) + 2000)) {
        ++local_failures;
      }
    }
    isolated_transport.close();

    // Two messages with one peer/tag are ambiguous under MPI's ordered
    // matching rules (especially if peers list different sizes/order).  They
    // must be rejected collectively before anything is posted.
    std::array<std::uint64_t, 2> duplicate_send{
        static_cast<std::uint64_t>(runtime.rank()),
        static_cast<std::uint64_t>(runtime.rank() + 1)};
    std::array<std::uint64_t, 2> duplicate_receive{};
    std::vector<quasar::distributed::ByteTransfer> duplicates;
    if (runtime.rank() < 2) {
      duplicates = host_exchange(
          runtime.rank(), 11, &duplicate_send[0], &duplicate_receive[0],
          sizeof(std::uint64_t));
      auto second = host_exchange(
          runtime.rank(), 11, &duplicate_send[1], &duplicate_receive[1],
          sizeof(std::uint64_t));
      duplicates.push_back(second.front());
    }
    if (!expect_collective_rejection(transport, duplicates)) ++local_failures;

    // Payload-size disagreement also implies a different chunk sequence and
    // is rejected before posting, leaving the transport reusable.
    std::array<std::uint64_t, 2> mismatch_send{};
    std::array<std::uint64_t, 2> mismatch_receive{};
    auto mismatch = host_exchange(
        runtime.rank(), 12, mismatch_send.data(), mismatch_receive.data(),
        runtime.rank() == 1 ? sizeof(mismatch_send)
                            : sizeof(std::uint64_t));
    if (!expect_collective_rejection(transport, mismatch)) ++local_failures;

    send += 10;
    receive = 0;
    transfers = host_exchange(
        runtime.rank(), 13, &send, &receive, sizeof(send));
    batch = transport.begin(transfers);
    batch.wait();
    if (!batch.complete() || transport.poisoned()) ++local_failures;
    if (runtime.rank() < 2
        && receive != static_cast<std::uint64_t>((1 - runtime.rank()) + 110)) {
      ++local_failures;
    }

    // Cache more than one validated pairing plan, but key the cache by the
    // complete global combination.  A local-only multi-plan cache would
    // incorrectly accept the mixed A/B plan below and post unmatched tags.
    std::uint64_t alternating_send =
        static_cast<std::uint64_t>(runtime.rank() + 5000);
    std::uint64_t alternating_receive = 0;
    auto plan_a = host_exchange(
        runtime.rank(), 20, &alternating_send, &alternating_receive,
        sizeof(alternating_send));
    auto plan_b = host_exchange(
        runtime.rank(), 21, &alternating_send, &alternating_receive,
        sizeof(alternating_send));
    auto plan_batch = transport.begin(plan_a);
    plan_batch.wait();
    alternating_receive = 0;
    plan_batch = transport.begin(plan_b);
    plan_batch.wait();

    auto mixed_plan = runtime.rank() == 0 ? plan_a : plan_b;
    if (!expect_collective_rejection(transport, mixed_plan)) {
      ++local_failures;
    }

    alternating_receive = 0;
    plan_batch = transport.begin(plan_a);
    plan_batch.wait();
    alternating_receive = 0;
    plan_batch = transport.begin(plan_b);
    plan_batch.wait();
    if (transport.poisoned()) ++local_failures;

    // Dropping wait() must not drop staging/request storage.  Collective
    // close cancels and drains the retained active epoch before returning.
    send += 10;
    receive = 0;
    transfers = host_exchange(
        runtime.rank(), 14, &send, &receive, sizeof(send));
    batch = transport.begin(transfers);
    transport.close();
    if (!batch.complete() || transport.has_active_epoch()) ++local_failures;

    // Replacing the handle for an active batch must not destroy its request
    // storage.  Its owning transport retains that storage for close-time drain.
    quasar::distributed::Transport abandoned_transport{
        runtime, quasar::distributed::TransportPolicy::staged};
    quasar::distributed::Transport replacement_transport{
        runtime, quasar::distributed::TransportPolicy::staged};
    std::uint64_t abandoned_send =
        static_cast<std::uint64_t>(runtime.rank() + 3000);
    std::uint64_t abandoned_receive = 0;
    std::uint64_t replacement_send =
        static_cast<std::uint64_t>(runtime.rank() + 4000);
    std::uint64_t replacement_receive = 0;
    auto abandoned_transfers = host_exchange(
        runtime.rank(), 3, &abandoned_send, &abandoned_receive,
        sizeof(abandoned_send));
    auto replacement_transfers = host_exchange(
        runtime.rank(), 3, &replacement_send, &replacement_receive,
        sizeof(replacement_send));
    auto replaced_batch = abandoned_transport.begin(abandoned_transfers);
    auto replacement_batch =
        replacement_transport.begin(replacement_transfers);
    replaced_batch = std::move(replacement_batch);
    replaced_batch.wait();
    abandoned_transport.close();
    replacement_transport.close();
    if (!replaced_batch.complete()
        || abandoned_transport.has_active_epoch()
        || replacement_transport.has_active_epoch()) {
      ++local_failures;
    }

    const auto global_failures = runtime.allreduce_sum(
        static_cast<std::uint64_t>(local_failures));
    runtime.close();
    return global_failures == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
