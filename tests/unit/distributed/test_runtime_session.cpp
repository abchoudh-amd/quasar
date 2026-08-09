#include "quasar/distributed/runtime_session.hpp"

#include "quasar/backend/device.hpp"

#include <mpi.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using quasar::Real;
using quasar::distributed::DecompositionShape;
using quasar::distributed::DeviceIdentity;
using quasar::distributed::DistributedCollectiveError;
using quasar::distributed::MhdGlobalState;
using quasar::distributed::RuntimeSession;
using quasar::distributed::TransportPolicy;

constexpr int kSkip = 77;

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error{std::string{message}};
}

void require_contains(std::string_view text, std::string_view expected,
                      std::string_view context) {
  if (text.find(expected) == std::string_view::npos) {
    throw std::runtime_error{
        std::string{context} + ": expected '" + std::string{expected}
        + "' in '" + std::string{text} + "'"};
  }
}

quasar::mhd::MhdConfig make_mhd_config(std::size_t nx, std::size_t ny) {
  quasar::mhd::MhdConfig config;
  config.grid = quasar::Grid2D{
      static_cast<int>(nx), static_cast<int>(ny), Real{1}, Real{1},
      Real{0}, Real{0}, /*nghost=*/2};
  config.geometry = "cartesian";
  config.reconstruction = "muscl_minmod";
  config.riemann = "hlld";
  config.integrator = "ssprk3";
  config.ct = "fd_ct_christlieb";
  config.positivity = "troubled_cell";
  for (int side = 0; side < 4; ++side) {
    config.boundary.fluid[side] = "outflow";
    config.boundary.field[side] = "outflow";
  }
  return config;
}

MhdGlobalState make_mhd_state(std::size_t nx, std::size_t ny) {
  MhdGlobalState state;
  state.global_nx = nx;
  state.global_ny = ny;
  const std::size_t cells = nx * ny;
  state.rho.assign(cells, Real{1});
  state.mx.assign(cells, Real{0});
  state.my.assign(cells, Real{0});
  state.mz.assign(cells, Real{0});
  state.energy.assign(cells, Real{3});
  state.bx_face.assign((nx + 1) * ny, Real{0.5});
  state.by_face.assign(nx * (ny + 1), Real{-0.25});
  state.bz_cell.assign(cells, Real{0.125});
  return state;
}

int test_owned_runtime_closes_after_physics_close_failure() {
#if !defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
  return kSkip;
#else
  if (quasar::backend::device_count() < 1) return kSkip;

  constexpr std::size_t nx = 8;
  constexpr std::size_t ny = 8;
  RuntimeSession session;
  require(session.owns_mpi(),
          "standalone RuntimeSession did not own MPI initialization");
  session.configure_devices({0});
  session.select_topology(nx, ny, DecompositionShape{1, 1},
                          /*minimum_tile_width=*/2);
  session.start_mhd(make_mhd_config(nx, ny), make_mhd_state(nx, ny),
                    std::nullopt, TransportPolicy::staged);
  session.inject_mhd_close_failure_for_testing(true);

  bool preserved_first_error = false;
  try {
    session.close();
  } catch (const DistributedCollectiveError& error) {
    preserved_first_error =
        error.resolution().representative.phase_text()
            == "mhd-worker-task-storage";
    require_contains(error.what(),
                     "distributed MHD worker task allocation failed",
                     "RuntimeSession close failure");
  }
  require(preserved_first_error,
          "RuntimeSession did not preserve the physics close failure");
  require(session.closed(),
          "RuntimeSession did not close its MPI runtime after physics close failed");

  int finalized = 0;
  require(MPI_Finalized(&finalized) == MPI_SUCCESS,
          "MPI_Finalized failed after RuntimeSession close");
  require(finalized != 0,
          "RuntimeSession skipped MPI_Finalize after physics close failed");
  return 0;
#endif
}

void test_lifecycle_and_collectives() {
  RuntimeSession session;
  require(!session.owns_mpi(),
          "session unexpectedly owns externally initialized MPI");
  require(!session.closed(), "new session is already closed");

  bool duplicate_rejected = false;
  try {
    RuntimeSession duplicate;
  } catch (const std::logic_error& error) {
    require_contains(error.what(), "one live RuntimeSession",
                     "duplicate session error");
    duplicate_rejected = true;
  }
  require(duplicate_rejected, "a second active session was accepted");

  session.barrier();
  session.require_same_string(
      "shared-value", "runtime-session-test-agreement",
      "ranks supplied different test agreement values");

  if (session.size() > 1) {
    bool disagreement_rejected = false;
    try {
      session.require_same_string(
          "rank-" + std::to_string(session.rank()),
          "runtime-session-test-disagreement",
          "ranks supplied different test agreement values");
    } catch (const DistributedCollectiveError& error) {
      require_contains(error.what(),
                       "ranks supplied different test agreement values",
                       "collective disagreement error");
      disagreement_rejected = true;
    }
    require(disagreement_rejected,
            "different per-rank strings were accepted");
  }

  bool parse_failure_rejected = false;
  try {
    session.configure_owned_devices(
        {}, session.rank() == 0 ? "synthetic bounded parse failure" : "");
  } catch (const DistributedCollectiveError& error) {
    require_contains(error.what(), "synthetic bounded parse failure",
                     "bounded parse failure");
    parse_failure_rejected = true;
  }
  require(parse_failure_rejected,
          "rank-local endpoint parse failure was not collective");

  std::vector<DeviceIdentity> devices;
  devices.push_back(DeviceIdentity{
      0, "runtime-session-rank-" + std::to_string(session.rank()), ""});
  session.configure_owned_devices(std::move(devices));

  const auto mapping = session.endpoint_mapping();
  require(mapping.rank_count() == static_cast<std::size_t>(session.size()),
          "endpoint assignment did not preserve every MPI rank");
  require(mapping.devices_per_rank() == 1,
          "endpoint assignment changed devices per rank");
  require(mapping.size() == static_cast<std::size_t>(session.size()),
          "endpoint assignment produced the wrong endpoint count");

  const std::size_t global_nx =
      std::max<std::size_t>(16, 8 * static_cast<std::size_t>(session.size()));
  session.select_topology(
      global_nx, 16,
      DecompositionShape{static_cast<std::size_t>(session.size()), 1},
      2);
  const auto topology = session.topology();
  require(topology.has_value(), "topology selection produced no topology");
  require(topology->tiles().size() == mapping.size(),
          "topology does not contain one tile per endpoint");

  const auto telemetry = session.telemetry();
  require(telemetry.counters.barriers == 1,
          "session barrier telemetry is incorrect");
  require(telemetry.counters.endpoint_configurations == 1,
          "session endpoint telemetry is incorrect");
  require(telemetry.counters.topology_selections == 1,
          "session topology telemetry is incorrect");
  require(telemetry.endpoint_count == mapping.size(),
          "session endpoint snapshot is incorrect");

  session.close();
  require(session.closed(), "session close did not close MPI runtime");
  session.close();

  // A successful explicit close releases the process registration even while
  // the closed C++ object itself remains alive.
  RuntimeSession replacement;
  replacement.close();
  require(replacement.closed(), "replacement session did not close");
}

void test_abandoned_session_poisoning() {
  {
    RuntimeSession abandoned;
    require(!abandoned.closed(), "new abandonment fixture is closed");
  }

  bool poisoned = false;
  try {
    RuntimeSession replacement;
  } catch (const std::logic_error& error) {
    require_contains(error.what(), "destroyed without collective close",
                     "abandoned session error");
    poisoned = true;
  }
  require(poisoned, "an abandoned process session was reused");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2
      && std::string_view{argv[1]} == "owned-close-failure") {
    try {
      return test_owned_runtime_closes_after_physics_close_failure();
    } catch (const std::exception& error) {
      std::cerr << "owned RuntimeSession close-failure test: "
                << error.what() << '\n';
      return 1;
    }
  }

  int provided = MPI_THREAD_SINGLE;
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided)
      != MPI_SUCCESS) {
    std::cerr << "MPI_Init_thread failed\n";
    return 2;
  }

  int result = 0;
  try {
    require(provided >= MPI_THREAD_FUNNELED,
            "MPI does not provide MPI_THREAD_FUNNELED");
    if (argc != 2) {
      throw std::invalid_argument{
          "usage: test_runtime_session "
          "<lifecycle|abandoned|owned-close-failure>"};
    }
    const std::string_view mode{argv[1]};
    if (mode == "lifecycle") {
      test_lifecycle_and_collectives();
    } else if (mode == "abandoned") {
      test_abandoned_session_poisoning();
    } else {
      throw std::invalid_argument{"unknown runtime-session test mode"};
    }
  } catch (const std::exception& error) {
    int rank = -1;
    (void)MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "runtime-session test rank " << rank << ": "
              << error.what() << '\n';
    result = 1;
  }

  if (MPI_Finalize() != MPI_SUCCESS && result == 0) result = 3;
  return result;
}
