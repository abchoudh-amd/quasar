#include "quasar/backend/memory.hpp"
#include "quasar/distributed/transport.hpp"

#include <mpi.h>

#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

namespace {

constexpr int skip = 77;
constexpr std::size_t payload_elements = 257;
constexpr int exchange_epochs = 3;

std::uint64_t value_for(int rank, int epoch, std::size_t index) {
  return static_cast<std::uint64_t>(rank + 1) * 1000000ULL
      + static_cast<std::uint64_t>(epoch + 1) * 1000ULL
      + static_cast<std::uint64_t>(index);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || (std::string_view{argv[1]} != "staged"
                    && std::string_view{argv[1]} != "direct")) {
    std::cerr << "usage: test_transport_device_mpi staged|direct\n";
    return 2;
  }

  std::unique_ptr<quasar::distributed::MpiRuntime> runtime;
  MPI_Comm local = MPI_COMM_NULL;
  try {
    runtime =
        std::make_unique<quasar::distributed::MpiRuntime>(&argc, &argv);
    if (runtime->size() != 2) {
      runtime->close();
      return skip;
    }

    quasar::distributed::check_mpi(
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                            runtime->rank(), MPI_INFO_NULL, &local),
        "MPI_Comm_split_type(transport device test)");
    int local_rank = -1;
    int local_size = 0;
    quasar::distributed::check_mpi(
        MPI_Comm_rank(local, &local_rank),
        "MPI_Comm_rank(transport device test)");
    quasar::distributed::check_mpi(
        MPI_Comm_size(local, &local_size),
        "MPI_Comm_size(transport device test)");
    const bool enough_local_devices =
        quasar::backend::device_count() >= local_size;
    if (!runtime->allreduce_all(enough_local_devices)) {
      MPI_Comm_free(&local);
      runtime->close();
      return skip;
    }

    const int device = local_rank;
    const auto requested = std::string_view{argv[1]} == "direct"
        ? quasar::distributed::TransportPolicy::direct
        : quasar::distributed::TransportPolicy::staged;
    std::unique_ptr<quasar::distributed::Transport> transport;
    bool direct_rejected = false;
    try {
      transport = std::make_unique<quasar::distributed::Transport>(
          *runtime, requested);
    } catch (const quasar::distributed::DistributedCollectiveError&) {
      direct_rejected = true;
    }
    if (requested == quasar::distributed::TransportPolicy::direct) {
      const bool all_rejected = runtime->allreduce_all(direct_rejected);
      const bool all_accepted = runtime->allreduce_all(!direct_rejected);
      if (all_rejected) {
        MPI_Comm_free(&local);
        runtime->close();
        return skip;
      }
      if (!all_accepted) {
        throw std::runtime_error{
            "direct transport policy was not resolved collectively"};
      }
    } else if (direct_rejected) {
      throw std::runtime_error{"staged transport was unexpectedly rejected"};
    }

    int local_failures = 0;
    {
      quasar::backend::DeviceBuffer<std::uint64_t> send{
          payload_elements, quasar::backend::on_device(device)};
      quasar::backend::DeviceBuffer<std::uint64_t> receive{
          payload_elements, quasar::backend::on_device(device)};
      quasar::backend::DeviceStream send_stream{
          quasar::backend::on_device(device)};
      quasar::backend::DeviceStream receive_stream{
          quasar::backend::on_device(device)};
      std::vector<std::uint64_t> host_send(payload_elements);
      std::vector<std::uint64_t> host_receive(payload_elements);

      for (int epoch = 0; epoch < exchange_epochs; ++epoch) {
        for (std::size_t index = 0; index < payload_elements; ++index) {
          host_send[index] = value_for(runtime->rank(), epoch, index);
          host_receive[index] = 0;
        }
        send.copy_from_host_async(
            host_send.data(), host_send.size(), send_stream.get());
        receive.copy_from_host_async(
            host_receive.data(), host_receive.size(), receive_stream.get());

        quasar::distributed::ByteTransfer transfer;
        transfer.peer_rank = 1 - runtime->rank();
        transfer.tag_slot = 5;
        transfer.send_buffer = send.device_ptr();
        transfer.receive_buffer = receive.device_ptr();
        transfer.bytes = send.bytes();
        transfer.residence = quasar::distributed::BufferResidence::device;
        transfer.send_device = device;
        transfer.receive_device = device;
        transfer.send_stream = send_stream.get();
        transfer.receive_stream = receive_stream.get();
        auto batch = transport->begin(std::span{&transfer, 1u});
        if (batch.epoch() != static_cast<std::uint64_t>(epoch)) {
          ++local_failures;
        }
        batch.wait();
        if (!batch.complete() || transport->has_active_epoch()) {
          ++local_failures;
        }

        receive.copy_to_host(host_receive.data(), host_receive.size());
        for (std::size_t index = 0; index < payload_elements; ++index) {
          if (host_receive[index]
              != value_for(1 - runtime->rank(), epoch, index)) {
            ++local_failures;
            break;
          }
        }
      }
    }

    const std::uint64_t expected_bytes =
        exchange_epochs * payload_elements * sizeof(std::uint64_t);
    const auto& telemetry = transport->telemetry();
    if (telemetry.epochs != exchange_epochs
        || telemetry.messages != exchange_epochs
        || telemetry.bytes != expected_bytes) {
      ++local_failures;
    }
    if (requested == quasar::distributed::TransportPolicy::staged) {
      if (telemetry.staged_mpi_bytes != expected_bytes
          || telemetry.direct_mpi_bytes != 0
          || transport->resolution().uses_direct_mpi()) {
        ++local_failures;
      }
    } else {
      if (telemetry.direct_mpi_bytes != expected_bytes
          || telemetry.staged_mpi_bytes != 0
          || !transport->resolution().uses_direct_mpi()) {
        ++local_failures;
      }
    }

    const auto failures = runtime->allreduce_sum(
        static_cast<std::uint64_t>(local_failures));
    transport->close();
    MPI_Comm_free(&local);
    runtime->close();
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "test_transport_device_mpi: " << error.what() << '\n';
    if (local != MPI_COMM_NULL) (void)MPI_Comm_free(&local);
    if (runtime && !runtime->closed()) {
      try {
        runtime->close();
      } catch (...) {
      }
    }
    return 2;
  }
}
