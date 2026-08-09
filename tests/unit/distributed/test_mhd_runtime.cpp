#include "quasar/backend/device.hpp"
#include "quasar/distributed/collective_error.hpp"
#include "quasar/distributed/device_mapping.hpp"
#include "quasar/distributed/mhd_runtime.hpp"
#include "quasar/distributed/mpi_runtime.hpp"
#include "quasar/distributed/topology.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using quasar::Real;
using quasar::distributed::DecompositionShape;
using quasar::distributed::DeviceIdentity;
using quasar::distributed::DistributedCollectiveError;
using quasar::distributed::EndpointMapping;
using quasar::distributed::MhdGlobalState;
using quasar::distributed::MhdTileRuntime;
using quasar::distributed::MpiRuntime;
using quasar::distributed::RankDeviceAssignment;
using quasar::distributed::TransportPolicy;
using quasar::distributed::VirtualTopology;

constexpr int kSkip = 77;

[[noreturn]] void fail(std::string message) {
  throw std::runtime_error{std::move(message)};
}

void require(bool condition, std::string message) {
  if (!condition) fail(std::move(message));
}

void require_same_text(MpiRuntime& mpi, const std::string& text,
                       std::string_view label) {
  constexpr int capacity = 1024;
  const bool local_fits = text.size() < static_cast<std::size_t>(capacity);
  require(mpi.allreduce_all(local_fits),
          std::string{label} + " exceeds the comparison buffer");

  std::array<char, capacity> local{};
  text.copy(local.data(), text.size());
  std::vector<std::array<char, capacity>> gathered(
      static_cast<std::size_t>(mpi.size()));
  quasar::distributed::check_mpi(
      MPI_Allgather(local.data(), capacity, MPI_CHAR, gathered.data(),
                    capacity, MPI_CHAR, MPI_COMM_WORLD),
      "MPI_Allgather(MHD exception text)");
  bool identical = true;
  for (const auto& candidate : gathered) {
    if (candidate != gathered.front()) {
      identical = false;
      break;
    }
  }
  require(identical, std::string{label} + " differs across MPI ranks");
}

void require_equal(std::span<const Real> expected,
                   std::span<const Real> actual,
                   std::string_view component) {
  if (expected.size() != actual.size()) {
    fail(std::string{component} + " has the wrong gathered size");
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (expected[index] != actual[index]) {
      std::ostringstream message;
      message << component << '[' << index << "] differs: expected "
              << expected[index] << ", got " << actual[index];
      fail(message.str());
    }
  }
}

void require_equal(const MhdGlobalState& expected,
                   const MhdGlobalState& actual) {
  require(expected.global_nx == actual.global_nx,
          "gathered global_nx differs");
  require(expected.global_ny == actual.global_ny,
          "gathered global_ny differs");
  require_equal(expected.rho, actual.rho, "rho");
  require_equal(expected.mx, actual.mx, "mx");
  require_equal(expected.my, actual.my, "my");
  require_equal(expected.mz, actual.mz, "mz");
  require_equal(expected.energy, actual.energy, "energy");
  require_equal(expected.bx_face, actual.bx_face, "bx_face");
  require_equal(expected.by_face, actual.by_face, "by_face");
  require_equal(expected.bz_cell, actual.bz_cell, "bz_cell");
}

void require_close(std::span<const Real> expected,
                   std::span<const Real> actual, Real tolerance,
                   std::string_view component) {
  require(expected.size() == actual.size(),
          std::string{component} + " has the wrong size");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const Real error = std::abs(expected[index] - actual[index]);
    if (!(error <= tolerance)) {
      std::ostringstream message;
      message << component << '[' << index << "] differs by " << error
              << " (expected " << expected[index] << ", got "
              << actual[index] << ')';
      fail(message.str());
    }
  }
}

void require_close(const MhdGlobalState& expected,
                   const MhdGlobalState& actual, Real tolerance) {
  require_close(expected.rho, actual.rho, tolerance, "rho");
  require_close(expected.mx, actual.mx, tolerance, "mx");
  require_close(expected.my, actual.my, tolerance, "my");
  require_close(expected.mz, actual.mz, tolerance, "mz");
  require_close(expected.energy, actual.energy, tolerance, "energy");
  require_close(expected.bx_face, actual.bx_face, tolerance, "bx_face");
  require_close(expected.by_face, actual.by_face, tolerance, "by_face");
  require_close(expected.bz_cell, actual.bz_cell, tolerance, "bz_cell");
}

void require_distributed_stage_telemetry(const MhdTileRuntime& runtime) {
  const auto& telemetry = runtime.telemetry();
  require(telemetry.register_halo_epochs > 0,
          "MHD step did not record register-halo transport epochs");
  require(telemetry.canonical_face_record_passes > 0,
          "MHD step did not reconcile canonical HLLD face records");
  require(telemetry.dense_residual_face_reconciliations == 0,
          "MHD residual faces unexpectedly used dense reconciliation");
  require(telemetry.dense_emf_input_reconciliations == 0,
          "MHD EMF inputs unexpectedly used dense reconciliation");
  require(telemetry.dense_ct_collective_bytes == 0,
          "MHD CT unexpectedly recorded dense collective bytes");
  require(telemetry.emf_reconciliations > 0,
          "MHD step did not record logical EMF reconciliation phases");
  require(telemetry.transport.epochs > telemetry.register_halo_epochs,
          "MHD transport did not include CT exchanges beyond register halos");
  require(telemetry.transport.bytes > 0,
          "MHD transport recorded no halo/CT payload bytes");
}

MhdGlobalState make_state(std::size_t nx, std::size_t ny) {
  MhdGlobalState state;
  state.global_nx = nx;
  state.global_ny = ny;
  const std::size_t cells = nx * ny;
  state.rho.resize(cells);
  state.mx.resize(cells);
  state.my.resize(cells);
  state.mz.resize(cells);
  state.energy.resize(cells);
  state.bz_cell.resize(cells);
  for (std::size_t index = 0; index < cells; ++index) {
    // Binary fractions make the host -> device -> host identity check exact.
    const Real offset = static_cast<Real>(index) / Real{1024};
    state.rho[index] = Real{1} + offset;
    state.mx[index] = Real{0.125} + offset;
    state.my[index] = Real{-0.25} + offset;
    state.mz[index] = Real{0.0625} - offset;
    state.energy[index] = Real{4} + offset;
    state.bz_cell[index] = Real{-0.03125} + offset;
  }

  state.bx_face.resize((nx + 1) * ny);
  for (std::size_t y = 0; y < ny; ++y) {
    for (std::size_t x = 0; x <= nx; ++x) {
      state.bx_face[y * (nx + 1) + x] =
          Real{0.5} + static_cast<Real>(17 * y + x) / Real{128};
    }
  }
  state.by_face.resize(nx * (ny + 1));
  for (std::size_t y = 0; y <= ny; ++y) {
    for (std::size_t x = 0; x < nx; ++x) {
      state.by_face[y * nx + x] =
          Real{-0.75} + static_cast<Real>(19 * y + x) / Real{128};
    }
  }
  return state;
}

quasar::mhd::MhdConfig make_config(std::size_t nx, std::size_t ny,
                                   bool periodic = false) {
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
  const std::string boundary = periodic ? "periodic" : "outflow";
  for (int side = 0; side < 4; ++side) {
    config.boundary.fluid[side] = boundary;
    config.boundary.field[side] = boundary;
  }
  return config;
}

quasar::mhd::MhdConfig make_cylindrical_axis_config(
    std::size_t nx = 4, std::size_t ny = 6) {
  auto config = make_config(nx, ny);
  config.geometry = "cylindrical";
  config.grid = quasar::Grid2D{
      static_cast<int>(nx), static_cast<int>(ny), Real{1}, Real{1.5},
      Real{0}, Real{0}, /*nghost=*/2};
  config.boundary.fluid[0] = "axis";
  config.boundary.field[0] = "axis";
  return config;
}

EndpointMapping make_mapping(std::size_t device_count) {
  std::vector<DeviceIdentity> devices;
  devices.reserve(device_count);
  for (std::size_t index = 0; index < device_count; ++index) {
    devices.push_back(DeviceIdentity{
        static_cast<int>(index), "mhd-runtime-test-" + std::to_string(index),
        {}});
  }
  const std::vector<RankDeviceAssignment> assignments{{
      /*world_rank=*/0, /*node_local_rank=*/0, "mhd-runtime-test-node",
      std::move(devices)}};
  return quasar::distributed::make_endpoint_mapping(assignments);
}

EndpointMapping make_multirank_mapping(const MpiRuntime& mpi) {
  std::vector<RankDeviceAssignment> assignments;
  assignments.reserve(static_cast<std::size_t>(mpi.size()));
  for (int rank = 0; rank < mpi.size(); ++rank) {
    assignments.push_back(RankDeviceAssignment{
        rank, rank, "mhd-runtime-multirank-node",
        {DeviceIdentity{rank, "mhd-runtime-rank-device-" +
                               std::to_string(rank),
                        {}}}});
  }
  return quasar::distributed::make_endpoint_mapping(assignments);
}

EndpointMapping make_worker_failure_mapping(const MpiRuntime& mpi) {
  const int invalid_ordinal = quasar::backend::device_count();
  std::vector<RankDeviceAssignment> assignments;
  assignments.reserve(static_cast<std::size_t>(mpi.size()));
  for (int rank = 0; rank < mpi.size(); ++rank) {
    assignments.push_back(RankDeviceAssignment{
        rank, rank, "mhd-worker-failure-node",
        {DeviceIdentity{
            rank == 1 ? invalid_ordinal : 0,
            "mhd-worker-failure-device-" + std::to_string(rank), {}}}});
  }
  return quasar::distributed::make_endpoint_mapping(assignments);
}

MhdGlobalState make_uniform_state(std::size_t nx, std::size_t ny) {
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

MhdGlobalState make_smooth_state(std::size_t nx, std::size_t ny) {
  auto state = make_uniform_state(nx, ny);
  constexpr Real tau = Real{6.283185307179586476925286766559};
  for (std::size_t y = 0; y < ny; ++y) {
    for (std::size_t x = 0; x < nx; ++x) {
      const std::size_t index = y * nx + x;
      const Real phase_x =
          tau * static_cast<Real>(x) / static_cast<Real>(nx);
      const Real phase_y =
          tau * static_cast<Real>(y) / static_cast<Real>(ny);
      state.rho[index] = Real{1} + Real{0.01} * std::sin(phase_x);
      state.mx[index] = Real{0.02} * std::cos(phase_x);
      state.my[index] = Real{0.01} * std::sin(phase_y);
      state.mz[index] = Real{0.005} * std::cos(phase_y);
      state.energy[index] = Real{3} + Real{0.01} * std::cos(phase_x);
      state.bz_cell[index] = Real{0.02} * std::sin(phase_x);
    }
  }
  return state;
}

quasar::distributed::MhdGlobalBackground make_uniform_background(
    std::size_t nx, std::size_t ny) {
  quasar::distributed::MhdGlobalBackground background;
  background.global_nx = nx;
  background.global_ny = ny;
  background.b0x_face.assign((nx + 1) * ny, Real{0.25});
  background.b0y_face.assign(nx * (ny + 1), Real{-0.125});
  background.b0z_cell.assign(nx * ny, Real{0.0625});
  return background;
}

std::vector<Real> padded_component(const quasar::Grid2D& grid,
                                   std::span<const Real> global,
                                   bool face_x, bool face_y) {
  std::vector<Real> host(grid.storage_size(), Real{0});
  const int nx = grid.nx;
  const int ny = grid.ny;
  const int limit_x = nx + (face_x ? 1 : 0);
  const int limit_y = ny + (face_y ? 1 : 0);
  const std::size_t width = static_cast<std::size_t>(limit_x);
  for (int j = 0; j < limit_y; ++j) {
    for (int i = 0; i < limit_x; ++i) {
      host[grid.index(i, j)] =
          global[static_cast<std::size_t>(j) * width +
                 static_cast<std::size_t>(i)];
    }
  }
  return host;
}

void seed_serial(quasar::mhd::MhdSolver2D& solver,
                 const MhdGlobalState& state) {
  const auto grid = solver.grid();
  solver.seed_state("rho", padded_component(grid, state.rho, false, false));
  solver.seed_state("mx", padded_component(grid, state.mx, false, false));
  solver.seed_state("my", padded_component(grid, state.my, false, false));
  solver.seed_state("mz", padded_component(grid, state.mz, false, false));
  solver.seed_state(
      "energy", padded_component(grid, state.energy, false, false));
  solver.seed_state(
      "bx_face", padded_component(grid, state.bx_face, true, false));
  solver.seed_state(
      "by_face", padded_component(grid, state.by_face, false, true));
  solver.seed_state(
      "bz_cell", padded_component(grid, state.bz_cell, false, false));
}

MhdGlobalState gather_serial(const quasar::mhd::MhdSolver2D& solver) {
  const auto grid = solver.grid();
  MhdGlobalState state;
  state.global_nx = static_cast<std::size_t>(grid.nx);
  state.global_ny = static_cast<std::size_t>(grid.ny);
  const auto gather = [&](std::string_view component, bool face_x,
                          bool face_y) {
    const auto host = solver.state_component_to_host(component);
    const int limit_x = grid.nx + (face_x ? 1 : 0);
    const int limit_y = grid.ny + (face_y ? 1 : 0);
    std::vector<Real> result(
        static_cast<std::size_t>(limit_x) *
        static_cast<std::size_t>(limit_y));
    for (int j = 0; j < limit_y; ++j) {
      for (int i = 0; i < limit_x; ++i) {
        result[static_cast<std::size_t>(j) *
                   static_cast<std::size_t>(limit_x) +
               static_cast<std::size_t>(i)] = host[grid.index(i, j)];
      }
    }
    return result;
  };
  state.rho = gather("rho", false, false);
  state.mx = gather("mx", false, false);
  state.my = gather("my", false, false);
  state.mz = gather("mz", false, false);
  state.energy = gather("energy", false, false);
  state.bx_face = gather("bx_face", true, false);
  state.by_face = gather("by_face", false, true);
  state.bz_cell = gather("bz_cell", false, false);
  return state;
}

MhdGlobalState make_brio_wu_state(std::size_t nx, std::size_t ny) {
  MhdGlobalState state;
  state.global_nx = nx;
  state.global_ny = ny;
  const std::size_t cells = nx * ny;
  state.rho.resize(cells);
  state.mx.assign(cells, Real{0});
  state.my.assign(cells, Real{0});
  state.mz.assign(cells, Real{0});
  state.energy.resize(cells);
  state.bz_cell.assign(cells, Real{0});
  for (std::size_t y = 0; y < ny; ++y) {
    for (std::size_t x = 0; x < nx; ++x) {
      const bool left = x < nx / 2;
      const std::size_t index = y * nx + x;
      state.rho[index] = left ? Real{1} : Real{0.125};
      const Real pressure = left ? Real{1} : Real{0.1};
      state.energy[index] = pressure +
          Real{0.5} * (Real{0.75} * Real{0.75} + Real{1});
    }
  }
  state.bx_face.assign((nx + 1) * ny, Real{0.75});
  state.by_face.resize(nx * (ny + 1));
  for (std::size_t y = 0; y <= ny; ++y) {
    for (std::size_t x = 0; x < nx; ++x) {
      state.by_face[y * nx + x] = x < nx / 2 ? Real{1} : Real{-1};
    }
  }
  return state;
}

quasar::mhd::MhdConfig make_brio_wu_config(
    std::size_t nx, std::size_t ny,
    std::string_view reconstruction = "mp7") {
  auto config = make_config(nx, ny);
  const int halo = reconstruction == "mp7" ? 4
      : reconstruction == "mp5" ? 3 : 2;
  config.grid = quasar::Grid2D{
      static_cast<int>(nx), static_cast<int>(ny), Real{1},
      static_cast<Real>(ny) / static_cast<Real>(nx), Real{0}, Real{0},
      halo};
  config.gamma = Real{2};
  config.reconstruction = std::string{reconstruction};
  config.boundary.fluid[2] = "periodic";
  config.boundary.fluid[3] = "periodic";
  config.boundary.field[2] = "periodic";
  config.boundary.field[3] = "periodic";
  return config;
}

MhdGlobalState make_expanding_floor_state(std::size_t nx,
                                          std::size_t ny,
                                          Real gamma) {
  MhdGlobalState state;
  state.global_nx = nx;
  state.global_ny = ny;
  const std::size_t cells = nx * ny;
  state.rho.assign(cells, Real{1});
  state.mx.resize(cells);
  state.my.assign(cells, Real{0});
  state.mz.assign(cells, Real{0});
  state.energy.resize(cells);
  state.bx_face.assign((nx + 1) * ny, Real{0});
  state.by_face.assign(nx * (ny + 1), Real{0});
  state.bz_cell.assign(cells, Real{0});
  constexpr Real pressure = Real{0.1};
  for (std::size_t y = 0; y < ny; ++y) {
    for (std::size_t x = 0; x < nx; ++x) {
      const std::size_t index = y * nx + x;
      const Real coordinate =
          (static_cast<Real>(x) + Real{0.5}) / static_cast<Real>(nx);
      const Real vx = Real{0.8} *
          std::sin(Real{2} * quasar::pi * coordinate);
      state.mx[index] = vx;
      state.energy[index] = pressure / (gamma - Real{1}) +
          Real{0.5} * vx * vx;
    }
  }
  return state;
}

quasar::mhd::MhdConfig make_expanding_floor_config(
    std::size_t nx, std::size_t ny) {
  auto config = make_config(nx, ny, /*periodic=*/true);
  config.grid = quasar::Grid2D{
      static_cast<int>(nx), static_cast<int>(ny), Real{1},
      static_cast<Real>(ny) / static_cast<Real>(nx), Real{0}, Real{0},
      /*nghost=*/4};
  config.reconstruction = "mp7";
  config.p_floor = Real{0.1};
  return config;
}

template <class Function>
void require_collective_rejection(MpiRuntime& mpi, Function&& function,
                                  std::string_view label) {
  (void)mpi;
  bool rejected = false;
  try {
    function();
  } catch (const DistributedCollectiveError&) {
    rejected = true;
  }
  const int local = rejected ? 1 : 0;
  int all = 0;
  quasar::distributed::check_mpi(
      MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD),
      "MPI_Allreduce(MHD rejection test)");
  require(all == 1, std::string{label} +
                        " was not rejected collectively on every rank");
}

template <class Function>
void require_poisoned_rejection(Function&& function,
                                std::string_view label) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception& error) {
    rejected = std::string_view{error.what()}.find("poisoned") !=
        std::string_view::npos;
  }
  require(rejected, std::string{label} +
                        " was not rejected by the poisoned runtime");
}

void run_seed_gather(MpiRuntime& mpi, std::size_t device_count,
                     DecompositionShape shape,
                     quasar::mhd::MhdConfig config,
                     const MhdGlobalState& seed) {
  auto topology = VirtualTopology::create(
      seed.global_nx, seed.global_ny, device_count, shape,
      /*minimum_tile_width=*/2);
  MhdTileRuntime runtime{
      mpi, make_mapping(device_count), std::move(topology), std::move(config)};
  try {
    runtime.seed(seed);
    const auto gathered = runtime.gather_state();
    require_equal(seed, gathered);
    runtime.close();
  } catch (...) {
    try {
      if (!runtime.closed()) runtime.close();
    } catch (...) {
    }
    throw;
  }
}

void run_brio_wu_comparison(
    MpiRuntime& mpi, DecompositionShape shape,
    std::string_view reconstruction = "mp7",
    std::size_t nx = 32, std::size_t ny = 8) {
  const auto seed = make_brio_wu_state(nx, ny);
  const std::size_t endpoint_count = shape.px * shape.py;
  const std::size_t halo = reconstruction == "mp7" ? 4
      : reconstruction == "mp5" ? 3 : 2;
  auto topology = VirtualTopology::create(
      nx, ny, endpoint_count, shape,
      /*minimum_tile_width=*/halo);
  MhdTileRuntime distributed{
      mpi, make_mapping(endpoint_count), std::move(topology),
      make_brio_wu_config(nx, ny, reconstruction)};
  try {
    distributed.seed(seed);
    quasar::mhd::MhdSolver2D serial{
        make_brio_wu_config(nx, ny, reconstruction)};
    seed_serial(serial, seed);
    const Real serial_limit = serial.cfl_limit();
    const Real distributed_limit = distributed.cfl_limit();
    require(std::abs(serial_limit - distributed_limit) <= Real{1e-15},
            "distributed " + std::string{reconstruction} +
                " Brio-Wu CFL differs from serial");
    const Real dt = Real{0.2} * std::min(serial_limit, distributed_limit);
    serial.step_unchecked(dt);
    distributed.step(dt);
    const Real tolerance = reconstruction == "mp7"
        ? Real{5e-13} : Real{5e-12};
    require_close(gather_serial(serial), distributed.gather_state(),
                  tolerance);
    require_distributed_stage_telemetry(distributed);
    if (mpi.size() == 1 && endpoint_count > 1) {
      const auto& transport = distributed.telemetry().transport;
      require(transport.peer_bytes + transport.local_staged_bytes ==
                  transport.bytes,
              "local multi-GPU MHD halo bytes were not fully classified");
      const bool first_edge_has_peer_path =
          quasar::backend::device_can_access_peer(1, 0) ||
          quasar::backend::device_can_access_peer(0, 1);
      if (first_edge_has_peer_path) {
        require(transport.peer_bytes > 0,
                "peer-capable MHD halo edge did not exercise peer copy");
      } else {
        require(transport.local_staged_bytes > 0 &&
                    transport.peer_bytes == 0,
                "non-peer-capable MHD halo edge did not use local staging");
      }
      require(transport.staged_mpi_bytes == 0 &&
                  transport.direct_mpi_bytes == 0,
              "single-rank MHD halos incorrectly recorded MPI bytes");
    }
    distributed.close();
  } catch (...) {
    try {
      if (!distributed.closed()) distributed.close();
    } catch (...) {
    }
    throw;
  }
}

void test_synchronized_positivity_retry(MpiRuntime& mpi) {
  constexpr std::size_t nx = 32;
  constexpr std::size_t ny = 8;
  auto config = make_expanding_floor_config(nx, ny);
  const auto seed = make_expanding_floor_state(nx, ny, config.gamma);
  auto topology = VirtualTopology::create(
      nx, ny, /*endpoint_count=*/2, {2, 1},
      /*minimum_tile_width=*/4);
  MhdTileRuntime distributed{
      mpi, make_mapping(2), std::move(topology), config};
  try {
    distributed.seed(seed);
    quasar::mhd::MhdSolver2D serial{config};
    seed_serial(serial, seed);
    const Real serial_limit = serial.cfl_limit();
    const Real distributed_limit = distributed.cfl_limit();
    require(std::abs(serial_limit - distributed_limit) <= Real{1e-15},
            "distributed positivity-retry CFL differs from serial");
    const Real requested =
        Real{32} * std::min(serial_limit, distributed_limit);
    serial.step_unchecked(requested);
    distributed.step(requested, /*check_cfl=*/false);
    const auto actual = distributed.gather_state();
    require_close(gather_serial(serial), actual, Real{2e-11});
    require(serial.last_positivity_substeps() > 1,
            "serial positivity fixture did not force subcycling");
    require(distributed.telemetry().rejected_attempts > 0,
            "distributed positivity fixture did not reject a stage");
    require(distributed.telemetry().accepted_substeps > 1,
            "distributed positivity fixture did not accept multiple substeps");
    const auto sum = [](std::span<const Real> values) {
      return std::accumulate(values.begin(), values.end(), Real{0});
    };
    require(std::abs(sum(actual.rho) - sum(seed.rho)) <= Real{2e-11},
            "distributed positivity retry did not conserve mass");
    require(std::abs(sum(actual.energy) - sum(seed.energy)) <= Real{2e-11},
            "distributed positivity retry did not conserve energy");
    require(distributed.divergence_b_max() <= Real{1e-12},
            "distributed positivity retry violated the CT constraint");
    require_distributed_stage_telemetry(distributed);
    distributed.close();
  } catch (...) {
    try {
      if (!distributed.closed()) distributed.close();
    } catch (...) {
    }
    throw;
  }
}

void test_cylindrical_axis_step(MpiRuntime& mpi) {
  constexpr std::size_t nx = 9;
  constexpr std::size_t ny = 7;
  auto config = make_cylindrical_axis_config(nx, ny);
  auto seed = make_uniform_state(nx, ny);
  std::fill(seed.bx_face.begin(), seed.bx_face.end(), Real{0});
  std::fill(seed.by_face.begin(), seed.by_face.end(), Real{0});
  std::fill(seed.bz_cell.begin(), seed.bz_cell.end(), Real{0});
  auto topology = VirtualTopology::create(
      nx, ny, /*endpoint_count=*/2, {2, 1},
      /*minimum_tile_width=*/2);
  MhdTileRuntime distributed{
      mpi, make_mapping(2), std::move(topology), config};
  try {
    distributed.seed(seed);
    const auto sums = distributed.global_cell_sums();
    const Real cylindrical_volume =
        quasar::pi * config.grid.lx * config.grid.lx * config.grid.ly;
    require(std::abs(sums.rho - cylindrical_volume) <= Real{2e-14},
            "cylindrical global rho integral omitted the radial measure");
    require(std::abs(sums.energy - Real{3} * cylindrical_volume)
                <= Real{6e-14},
            "cylindrical global energy integral omitted the radial measure");
    quasar::mhd::MhdSolver2D serial{config};
    seed_serial(serial, seed);
    const Real serial_limit = serial.cfl_limit();
    const Real distributed_limit = distributed.cfl_limit();
    require(std::abs(serial_limit - distributed_limit) <= Real{1e-15},
            "distributed cylindrical-axis CFL differs from serial");
    const Real dt =
        Real{0.1} * std::min(serial_limit, distributed_limit);
    serial.step_unchecked(dt);
    distributed.step(dt);
    require_close(gather_serial(serial), distributed.gather_state(),
                  Real{5e-12});
    require(distributed.divergence_b_max() <= Real{1e-12},
            "distributed cylindrical-axis step violated the CT constraint");
    require_distributed_stage_telemetry(distributed);
    distributed.close();
  } catch (...) {
    try {
      if (!distributed.closed()) distributed.close();
    } catch (...) {
    }
    throw;
  }
}

void test_physical_y_high_ct_corner(MpiRuntime& mpi) {
  constexpr std::size_t nx = 13;
  constexpr std::size_t ny = 9;
  const auto config = make_config(nx, ny);
  const auto seed = make_smooth_state(nx, ny);
  require(std::any_of(seed.bx_face.begin(), seed.bx_face.end(),
                      [](Real value) { return value != Real{0}; }) &&
              std::any_of(seed.by_face.begin(), seed.by_face.end(),
                          [](Real value) { return value != Real{0}; }),
          "physical-y-high CT fixture requires nontrivial magnetic fields");
  auto topology = VirtualTopology::create(
      nx, ny, /*endpoint_count=*/2, {2, 1},
      /*minimum_tile_width=*/2);
  MhdTileRuntime distributed{
      mpi, make_mapping(2), std::move(topology), config};
  try {
    distributed.seed(seed);
    quasar::mhd::MhdSolver2D serial{config};
    seed_serial(serial, seed);
    const Real serial_limit = serial.cfl_limit();
    const Real distributed_limit = distributed.cfl_limit();
    const Real dt =
        Real{0.1} * std::min(serial_limit, distributed_limit);
    serial.step_unchecked(dt);
    distributed.step(dt);
    require_close(gather_serial(serial), distributed.gather_state(),
                  Real{5e-12});
    distributed.close();
  } catch (...) {
    try {
      if (!distributed.closed()) distributed.close();
    } catch (...) {
    }
    throw;
  }
}

void test_four_gpu(MpiRuntime& mpi) {
  run_brio_wu_comparison(mpi, {4, 1});
  run_brio_wu_comparison(mpi, {2, 2});
  // Exact MP5 halo width three on an uneven genuine 2-D split. The y-periodic
  // seam and its four-way corners exercise canonical records in both axes.
  run_brio_wu_comparison(mpi, {2, 2}, "mp5", /*nx=*/19, /*ny=*/13);
}

void test_explicit_background(MpiRuntime& mpi) {
  constexpr std::size_t nx = 12;
  constexpr std::size_t ny = 8;
  const auto state = make_uniform_state(nx, ny);
  const auto background = make_uniform_background(nx, ny);
  auto config = make_config(nx, ny);
  config.background.enabled = true;
  config.background.profile = "uniform";
  config.background.bx0 = Real{0};
  config.background.by0 = Real{0};
  config.background.bz0 = Real{0};

  quasar::mhd::MhdSolver2D serial{config};
  seed_serial(serial, state);
  const auto grid = serial.grid();
  serial.seed_background(
      "b0x_face", std::vector<Real>(grid.storage_size(), Real{0.25}));
  serial.seed_background(
      "b0y_face", std::vector<Real>(grid.storage_size(), Real{-0.125}));
  serial.seed_background(
      "b0z_cell", std::vector<Real>(grid.storage_size(), Real{0.0625}));

  auto topology = VirtualTopology::create(
      nx, ny, /*endpoint_count=*/1, {1, 1},
      /*minimum_tile_width=*/2);
  MhdTileRuntime distributed{
      mpi, make_mapping(1), std::move(topology), config};
  try {
    distributed.seed(state, &background);
    const Real serial_limit = serial.cfl_limit();
    const Real distributed_limit = distributed.cfl_limit();
    require(std::abs(serial_limit - distributed_limit) <= Real{1e-15},
            "explicit-background distributed CFL differs from serial");
    const Real dt = Real{0.1} * std::min(serial_limit, distributed_limit);
    serial.step_unchecked(dt);
    distributed.step(dt);
    require_close(gather_serial(serial), distributed.gather_state(),
                  Real{5e-13});
    distributed.close();
  } catch (...) {
    try {
      if (!distributed.closed()) distributed.close();
    } catch (...) {
    }
    throw;
  }
}

void test_explicit_background_multi_gpu(MpiRuntime& mpi) {
  constexpr std::size_t nx = 13;
  constexpr std::size_t ny = 9;
  const auto state = make_smooth_state(nx, ny);
  const auto background = make_uniform_background(nx, ny);
  auto config = make_config(nx, ny, /*periodic=*/true);
  config.background.enabled = true;
  config.background.profile = "uniform";
  config.background.bx0 = Real{0};
  config.background.by0 = Real{0};
  config.background.bz0 = Real{0};

  quasar::mhd::MhdSolver2D serial{config};
  seed_serial(serial, state);
  const auto grid = serial.grid();
  serial.seed_background(
      "b0x_face", std::vector<Real>(grid.storage_size(), Real{0.25}));
  serial.seed_background(
      "b0y_face", std::vector<Real>(grid.storage_size(), Real{-0.125}));
  serial.seed_background(
      "b0z_cell", std::vector<Real>(grid.storage_size(), Real{0.0625}));

  auto topology = VirtualTopology::create(
      nx, ny, /*endpoint_count=*/2, {2, 1},
      /*minimum_tile_width=*/2);
  MhdTileRuntime distributed{
      mpi, make_mapping(2), std::move(topology), config};
  try {
    distributed.seed(state, &background);
    const Real serial_limit = serial.cfl_limit();
    const Real distributed_limit = distributed.cfl_limit();
    require(std::abs(serial_limit - distributed_limit) <= Real{1e-15},
            "multi-GPU explicit-background CFL differs from serial");
    const Real dt =
        Real{0.1} * std::min(serial_limit, distributed_limit);
    serial.step_unchecked(dt);
    distributed.step(dt);
    require_close(gather_serial(serial), distributed.gather_state(),
                  Real{5e-12});
    require(distributed.telemetry().canonical_face_record_passes >= 8,
            "active-background step did not exchange complete auxiliary records");
    require_distributed_stage_telemetry(distributed);
    distributed.close();
  } catch (...) {
    try {
      if (!distributed.closed()) distributed.close();
    } catch (...) {
    }
    throw;
  }
}

void test_single_gpu(MpiRuntime& mpi) {
  {
    constexpr std::size_t nx = 9;
    constexpr std::size_t ny = 7;
    const auto seed = make_state(nx, ny);
    run_seed_gather(mpi, /*device_count=*/1, {1, 1},
                    make_config(nx, ny), seed);
  }

  {
    constexpr std::size_t nx = 8;
    constexpr std::size_t ny = 6;
    auto seed = make_state(nx, ny);
    // Deliberately make periodic high-face interchange duplicates disagree
    // with their canonical low faces. The runtime must ignore these duplicate
    // inputs and rebuild them after collective ownership validation.
    for (std::size_t y = 0; y < ny; ++y) {
      seed.bx_face[y * (nx + 1) + nx] =
          Real{1000} + static_cast<Real>(y);
    }
    for (std::size_t x = 0; x < nx; ++x) {
      seed.by_face[ny * nx + x] =
          Real{-1000} - static_cast<Real>(x);
    }

    auto expected = seed;
    for (std::size_t y = 0; y < ny; ++y) {
      expected.bx_face[y * (nx + 1) + nx] =
          expected.bx_face[y * (nx + 1)];
    }
    for (std::size_t x = 0; x < nx; ++x) {
      expected.by_face[ny * nx + x] = expected.by_face[x];
    }

    auto topology = VirtualTopology::create(
        nx, ny, /*endpoint_count=*/1, {1, 1},
        /*minimum_tile_width=*/2);
    MhdTileRuntime runtime{mpi, make_mapping(1), std::move(topology),
                           make_config(nx, ny, /*periodic=*/true)};
    try {
      runtime.seed(seed);
      require_equal(expected, runtime.gather_state());
      runtime.close();
    } catch (...) {
      try {
        if (!runtime.closed()) runtime.close();
      } catch (...) {
      }
      throw;
    }
  }

  test_explicit_background(mpi);
}

void test_multi_gpu(MpiRuntime& mpi) {
  // Both dimensions are intentionally uneven. Each orientation therefore
  // exercises a distinct half-open ownership seam and remainder placement.
  constexpr std::size_t nx = 9;
  constexpr std::size_t ny = 7;
  const auto seed = make_state(nx, ny);
  run_seed_gather(mpi, /*device_count=*/2, {2, 1}, make_config(nx, ny), seed);
  run_seed_gather(mpi, /*device_count=*/2, {1, 2}, make_config(nx, ny), seed);

  // Explicit decompositions are accepted by the physics-neutral topology
  // layer, so the runtime must reject a tile narrower than the selected
  // reconstruction stencil before constructing any solver workers.
  {
    auto config = make_config(/*nx=*/6, /*ny=*/8);
    config.reconstruction = "mp7";
    bool rejected = false;
    std::string rejection_message;
    try {
      MhdTileRuntime runtime{
          mpi, make_mapping(/*device_count=*/2),
          VirtualTopology::create(
              /*global_nx=*/6, /*global_ny=*/8, /*endpoint_count=*/2,
              {2, 1}, /*minimum_tile_width=*/1),
          std::move(config)};
      runtime.close();
    } catch (const DistributedCollectiveError& error) {
      rejection_message = error.what();
      rejected = error.resolution().representative.phase_text()
              == "mhd-runtime-config" &&
          rejection_message.find("MHD decomposition 2x1") !=
              std::string::npos &&
          rejection_message.find("4-cell stencil reach") !=
              std::string::npos &&
          rejection_message.find("reconstruction=mp7") !=
              std::string::npos;
    }
    require(rejected,
            "thin MHD decomposition lacked a scheme-specific rejection: " +
                rejection_message);
  }

  // The second radial tile begins at r=0.5, has dr=0.25 and a two-cell halo,
  // so its internal low halo reaches r=0 exactly. That is valid because the
  // global axis tile owns the parity closure; an ordinary physical annulus
  // with the same padded extent would remain invalid.
  auto cylindrical = make_cylindrical_axis_config();
  const auto cylindrical_seed = make_state(4, 6);
  run_seed_gather(mpi, /*device_count=*/2, {2, 1},
                  std::move(cylindrical), cylindrical_seed);
  test_cylindrical_axis_step(mpi);

  // The y split is the regression for physical-x/internal-y CT corners; the x
  // split covers canonical face ownership in the orthogonal orientation.
  run_brio_wu_comparison(mpi, {2, 1});
  run_brio_wu_comparison(mpi, {1, 2});
  test_physical_y_high_ct_corner(mpi);
  // MUSCL's exact width-two halo on an uneven y split, including the periodic
  // global seam, must consume the same canonical HLLD records as serial.
  run_brio_wu_comparison(
      mpi, {1, 2}, "muscl_minmod", /*nx=*/19, /*ny=*/9);
  test_explicit_background_multi_gpu(mpi);
  test_synchronized_positivity_retry(mpi);
}

void test_multirank_consensus(MpiRuntime& mpi) {
  constexpr std::size_t nx = 12;
  constexpr std::size_t ny = 8;

  {
    bool rejected = false;
    std::string rejection_message;
    try {
      MhdTileRuntime runtime{
          mpi, make_worker_failure_mapping(mpi),
          VirtualTopology::create(nx, ny, 2, {2, 1}, 2),
          make_config(nx, ny)};
      runtime.close();
    } catch (const DistributedCollectiveError& error) {
      rejected = error.resolution().representative.phase_text()
                     == "mhd-worker-pool-construct"
          && error.resolution().representative.rank == 1;
      rejection_message = error.what();
    }
    require(mpi.allreduce_all(rejected),
            "rank-local MHD worker construction failure was not rejected "
            "collectively");
    require_same_text(mpi, rejection_message,
                      "MHD worker construction collective exception");
  }

  const auto mapping = make_multirank_mapping(mpi);
  const auto make_topology = [&] {
    return VirtualTopology::create(
        nx, ny, static_cast<std::size_t>(mpi.size()), {2, 1},
        /*minimum_tile_width=*/2);
  };

  require_collective_rejection(
      mpi,
      [&] {
        auto config = make_config(nx, ny);
        if (mpi.rank() == 1) config.gamma = Real{1.4};
        MhdTileRuntime runtime{
            mpi, mapping, make_topology(), std::move(config)};
        runtime.close();
      },
      "rank-dependent MHD configuration");

#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
  {
    MhdTileRuntime runtime{
        mpi, mapping, make_topology(), make_config(nx, ny),
        TransportPolicy::staged};
    runtime.inject_seed_post_mutation_failure_for_testing(mpi.rank() == 1);
    bool rejected = false;
    std::string rejection_message;
    try {
      runtime.seed(make_uniform_state(nx, ny));
    } catch (const DistributedCollectiveError& error) {
      rejected = error.resolution().representative.phase_text() ==
          "mhd-seed-failure";
      rejection_message = error.what();
    }
    require(mpi.allreduce_all(rejected),
            "post-mutation MHD seed failure was not rejected collectively");
    require_same_text(mpi, rejection_message,
                      "post-mutation MHD seed collective exception");
    require(mpi.allreduce_all(runtime.poisoned()),
            "post-mutation MHD seed failure did not poison every rank");
    require(!runtime.seeded(),
            "failed MHD seed was marked committed");
    require_poisoned_rejection(
        [&] { runtime.seed(make_uniform_state(nx, ny)); },
        "MHD reseed after post-mutation failure");
    require_poisoned_rejection(
        [&] { runtime.step(Real{1e-6}); },
        "MHD step after post-mutation seed failure");
    require_poisoned_rejection(
        [&] { (void)runtime.gather_state(); },
        "MHD diagnostic after post-mutation seed failure");
    runtime.close();
    require(runtime.closed(),
            "poisoned MHD seed runtime did not close collectively");
  }

  {
    MhdTileRuntime runtime{
        mpi, mapping, make_topology(), make_config(nx, ny),
        TransportPolicy::staged};
    runtime.inject_next_worker_task_allocation_failure_for_testing(
        mpi.rank() == 1);
    bool rejected = false;
    std::string rejection_message;
    try {
      runtime.close();
    } catch (const DistributedCollectiveError& error) {
      rejected = error.resolution().representative.phase_text() ==
                     "mhd-worker-task-storage" &&
          error.resolution().representative.rank == 1;
      rejection_message = error.what();
    }
    require(mpi.allreduce_all(rejected),
            "rank-local MHD close task allocation failure was not rejected "
            "collectively");
    require_same_text(mpi, rejection_message,
                      "MHD close task allocation collective exception");
    require(!runtime.closed(),
            "failed MHD close task allocation marked the runtime closed");
    runtime.close();
    require(runtime.closed(),
            "MHD runtime did not close after one-shot allocation failure");
  }
#endif

  {
    MhdTileRuntime runtime{
        mpi, mapping, make_topology(), make_config(nx, ny),
        TransportPolicy::staged};
    auto inconsistent = make_uniform_state(nx, ny);
    if (mpi.rank() == 1) inconsistent.rho[0] += Real{0.125};
    require_collective_rejection(
        mpi, [&] { runtime.seed(inconsistent); },
        "rank-dependent canonical MHD state seed");
    require(!runtime.seeded(),
            "rejected MHD state seed mutated the runtime");

    const auto seed = make_uniform_state(nx, ny);
    runtime.seed(seed);
    const auto before = runtime.gather_state();
    const Real limit = runtime.cfl_limit();

    const Real invalid_dt = mpi.rank() == 0 ? Real{-1} : Real{0.1} * limit;
    require_collective_rejection(
        mpi, [&] { runtime.step(invalid_dt); },
        "rank-local invalid MHD timestep");
    require_equal(before, runtime.gather_state());

    const Real inconsistent_dt =
        (mpi.rank() == 0 ? Real{0.1} : Real{0.2}) * limit;
    require_collective_rejection(
        mpi, [&] { runtime.step(inconsistent_dt); },
        "rank-dependent MHD timestep");
    require_equal(before, runtime.gather_state());

    // Both validation failures occurred before the request snapshot or any
    // physical update. A subsequent common step must still succeed.
    runtime.step(Real{0.05} * limit);
    require(runtime.telemetry().accepted_steps == 1,
            "MHD runtime was not reusable after timestep rejection");
    require_distributed_stage_telemetry(runtime);
    const auto& transport = runtime.telemetry().transport;
    require(transport.staged_mpi_bytes > 0,
            "explicit staged MHD transport recorded no MPI bytes");
    require(transport.direct_mpi_bytes == 0,
            "explicit staged MHD transport recorded direct MPI bytes");
    runtime.close();
  }

  {
    std::unique_ptr<MhdTileRuntime> runtime;
    bool direct_rejected = false;
    std::string rejection_message;
    try {
      runtime = std::make_unique<MhdTileRuntime>(
          mpi, mapping, make_topology(), make_config(nx, ny),
          TransportPolicy::direct);
    } catch (const DistributedCollectiveError& error) {
      direct_rejected = true;
      rejection_message = error.what();
    }

    const bool all_rejected = mpi.allreduce_all(direct_rejected);
    const bool all_accepted = mpi.allreduce_all(!direct_rejected);
    require(all_rejected || all_accepted,
            "explicit direct MHD transport did not resolve collectively");
    if (all_rejected) {
      require_same_text(
          mpi, rejection_message,
          "explicit direct MHD transport collective exception");
    } else {
      try {
        runtime->seed(make_uniform_state(nx, ny));
        runtime->step(Real{0.05} * runtime->cfl_limit());
        require_distributed_stage_telemetry(*runtime);
        const auto& resolution = runtime->transport_resolution();
        const auto& transport = runtime->telemetry().transport;
        require(resolution.requested == TransportPolicy::direct &&
                    resolution.interprocess == TransportPolicy::direct,
                "explicit direct MHD transport resolved to staging");
        require(transport.direct_mpi_bytes > 0,
                "explicit direct MHD transport recorded no direct MPI bytes");
        require(transport.staged_mpi_bytes == 0,
                "explicit direct MHD transport recorded staged MPI bytes");
        runtime->close();
      } catch (...) {
        try {
          if (runtime && !runtime->closed()) runtime->close();
        } catch (...) {
        }
        throw;
      }
    }
  }

  {
    auto config = make_config(nx, ny);
    config.background.enabled = true;
    config.background.profile = "uniform";
    MhdTileRuntime runtime{
        mpi, mapping, make_topology(), std::move(config)};
    const auto seed = make_uniform_state(nx, ny);
    auto inconsistent = make_uniform_background(nx, ny);
    if (mpi.rank() == 1) inconsistent.b0z_cell[0] += Real{0.125};
    require_collective_rejection(
        mpi, [&] { runtime.seed(seed, &inconsistent); },
        "rank-dependent canonical MHD background seed");
    require(!runtime.seeded(),
            "rejected MHD background seed mutated the runtime");
    const auto background = make_uniform_background(nx, ny);
    runtime.seed(seed, &background);
    (void)runtime.cfl_limit();
    runtime.close();
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || (std::string_view{argv[1]} != "single" &&
                    std::string_view{argv[1]} != "multi" &&
                    std::string_view{argv[1]} != "quad" &&
                    std::string_view{argv[1]} != "multirank")) {
    std::cerr << "usage: test_mhd_runtime single|multi|quad|multirank\n";
    return 2;
  }

  std::unique_ptr<MpiRuntime> mpi;
  try {
    mpi = std::make_unique<MpiRuntime>(&argc, &argv);
    const bool multirank = std::string_view{argv[1]} == "multirank";
    const int expected_ranks = multirank ? 2 : 1;
    if (mpi->size() != expected_ranks) {
      if (mpi->rank() == 0) {
        std::cerr << "test_mhd_runtime " << argv[1]
                  << " requires MPI world size " << expected_ranks << '\n';
      }
      mpi->close();
      return kSkip;
    }

    const bool multi = std::string_view{argv[1]} == "multi";
    const bool quad = std::string_view{argv[1]} == "quad";
    const int required_devices = quad ? 4 : (multi || multirank) ? 2 : 1;
    if (quasar::backend::device_count() < required_devices) {
      std::cerr << "test_mhd_runtime " << argv[1] << " requires "
                << required_devices << " visible GPU(s)\n";
      mpi->close();
      return kSkip;
    }

    if (multirank) {
      test_multirank_consensus(*mpi);
    } else if (quad) {
      test_four_gpu(*mpi);
    } else if (multi) {
      test_multi_gpu(*mpi);
    } else {
      test_single_gpu(*mpi);
    }
    mpi->close();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_mhd_runtime: " << error.what() << '\n';
    if (mpi && !mpi->closed()) {
      try {
        mpi->close();
      } catch (const std::exception& close_error) {
        std::cerr << "test_mhd_runtime close: " << close_error.what() << '\n';
      }
    }
    return 1;
  }
}
