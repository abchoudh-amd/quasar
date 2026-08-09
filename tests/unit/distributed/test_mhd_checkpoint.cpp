#include "quasar/backend/device.hpp"
#include "quasar/distributed/device_mapping.hpp"
#include "quasar/distributed/mhd_runtime.hpp"
#include "quasar/distributed/mpi_runtime.hpp"
#include "quasar/distributed/topology.hpp"

#include <hdf5.h>
#include <mpi.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
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
using quasar::distributed::MhdGlobalBackground;
using quasar::distributed::MhdGlobalState;
using quasar::distributed::MhdTileRuntime;
using quasar::distributed::MpiRuntime;
using quasar::distributed::RankDeviceAssignment;
using quasar::distributed::VirtualTopology;

constexpr int kSkip = 77;

[[noreturn]] void fail(std::string message) {
  throw std::runtime_error{std::move(message)};
}

void require(bool condition, std::string message) {
  if (!condition) fail(std::move(message));
}

void require_equal(std::span<const Real> expected,
                   std::span<const Real> actual,
                   std::string_view component) {
  if (expected.size() != actual.size()) {
    fail(std::string{component} + " has the wrong restored size");
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (expected[index] != actual[index]) {
      std::ostringstream message;
      message << component << '[' << index << "] differs after restart";
      fail(message.str());
    }
  }
}

void require_equal(const MhdGlobalState& expected,
                   const MhdGlobalState& actual) {
  require(expected.global_nx == actual.global_nx,
          "restored global_nx differs");
  require(expected.global_ny == actual.global_ny,
          "restored global_ny differs");
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
                   std::span<const Real> actual,
                   Real tolerance,
                   std::string_view component) {
  if (expected.size() != actual.size()) {
    fail(std::string{component} + " has the wrong restored size");
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const Real error = std::abs(expected[index] - actual[index]);
    if (!(error <= tolerance)) {
      std::ostringstream message;
      message << component << '[' << index << "] differs by " << error
              << " after cross-placement restart (expected "
              << expected[index] << ", got " << actual[index] << ')';
      fail(message.str());
    }
  }
}

void require_close(const MhdGlobalState& expected,
                   const MhdGlobalState& actual,
                   Real tolerance) {
  require(expected.global_nx == actual.global_nx,
          "restored global_nx differs");
  require(expected.global_ny == actual.global_ny,
          "restored global_ny differs");
  require_close(expected.rho, actual.rho, tolerance, "rho");
  require_close(expected.mx, actual.mx, tolerance, "mx");
  require_close(expected.my, actual.my, tolerance, "my");
  require_close(expected.mz, actual.mz, tolerance, "mz");
  require_close(expected.energy, actual.energy, tolerance, "energy");
  require_close(expected.bx_face, actual.bx_face, tolerance, "bx_face");
  require_close(expected.by_face, actual.by_face, tolerance, "by_face");
  require_close(expected.bz_cell, actual.bz_cell, tolerance, "bz_cell");
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
    const Real offset = static_cast<Real>(index) / Real{2048};
    state.rho[index] = Real{1} + offset;
    state.mx[index] = Real{0.125} + offset;
    state.my[index] = Real{-0.25} + offset;
    state.mz[index] = Real{0.0625} - offset;
    state.energy[index] = Real{4} + offset;
    state.bz_cell[index] = Real{-0.03125} + offset;
  }
  state.bx_face.resize((nx + 1) * ny);
  for (std::size_t y = 0; y < ny; ++y) {
    for (std::size_t x = 0; x < nx; ++x) {
      state.bx_face[y * (nx + 1) + x] =
          Real{0.5} + static_cast<Real>(13 * y + x) / Real{256};
    }
    state.bx_face[y * (nx + 1) + nx] =
        state.bx_face[y * (nx + 1)];
  }
  state.by_face.resize(nx * (ny + 1));
  for (std::size_t y = 0; y < ny; ++y) {
    for (std::size_t x = 0; x < nx; ++x) {
      state.by_face[y * nx + x] =
          Real{-0.75} + static_cast<Real>(17 * y + x) / Real{256};
    }
  }
  for (std::size_t x = 0; x < nx; ++x) {
    state.by_face[ny * nx + x] = state.by_face[x];
  }
  return state;
}

MhdGlobalState make_evolvable_state(std::size_t nx, std::size_t ny) {
  MhdGlobalState state;
  state.global_nx = nx;
  state.global_ny = ny;
  const std::size_t cells = nx * ny;
  state.rho.resize(cells);
  state.mx.resize(cells);
  state.my.resize(cells);
  state.mz.resize(cells);
  state.energy.assign(cells, Real{4});
  state.bz_cell.resize(cells);
  constexpr Real tau = Real{6.283185307179586476925286766559};
  for (std::size_t y = 0; y < ny; ++y) {
    for (std::size_t x = 0; x < nx; ++x) {
      const std::size_t index = y * nx + x;
      const Real phase = tau * static_cast<Real>(x) / static_cast<Real>(nx);
      state.rho[index] = Real{1} + Real{0.01} * std::sin(phase);
      state.mx[index] = Real{0.02} * std::cos(phase);
      state.my[index] = Real{0.01} * std::sin(phase);
      state.mz[index] = Real{0.005} * std::cos(phase);
      state.bz_cell[index] = Real{0.02} * std::sin(phase);
    }
  }
  state.bx_face.resize((nx + 1) * ny);
  for (std::size_t y = 0; y < ny; ++y) {
    const Real value = Real{0.5} + Real{0.01} * std::sin(
        tau * static_cast<Real>(y) / static_cast<Real>(ny));
    for (std::size_t x = 0; x <= nx; ++x) {
      state.bx_face[y * (nx + 1) + x] = value;
    }
  }
  state.by_face.resize(nx * (ny + 1));
  for (std::size_t y = 0; y <= ny; ++y) {
    for (std::size_t x = 0; x < nx; ++x) {
      state.by_face[y * nx + x] = Real{-0.25} + Real{0.01} * std::cos(
          tau * static_cast<Real>(x) / static_cast<Real>(nx));
    }
  }
  return state;
}

MhdGlobalBackground make_background(std::size_t nx, std::size_t ny) {
  MhdGlobalBackground background;
  background.global_nx = nx;
  background.global_ny = ny;
  // Deliberately differ from the analytic uniform values in make_config().
  // A restart must load these stored samples over the constructor defaults.
  background.b0x_face.assign((nx + 1) * ny, Real{0.375});
  background.b0y_face.assign(nx * (ny + 1), Real{-0.1875});
  background.b0z_cell.assign(nx * ny, Real{0.09375});
  return background;
}

quasar::mhd::MhdConfig make_config(std::size_t nx, std::size_t ny,
                                   bool background) {
  quasar::mhd::MhdConfig config;
  config.grid = quasar::Grid2D{
      static_cast<int>(nx), static_cast<int>(ny), Real{2}, Real{1.5},
      Real{0.25}, Real{-0.5}, /*nghost=*/2};
  config.geometry = "cartesian";
  config.reconstruction = "muscl_minmod";
  config.riemann = "hlld";
  config.integrator = "ssprk3";
  config.ct = "fd_ct_christlieb";
  config.positivity = "troubled_cell";
  config.timestep_signature = "policy=fixed;dt=0x1p-5";
  for (int side = 0; side < 4; ++side) {
    config.boundary.fluid[side] = "periodic";
    config.boundary.field[side] = "periodic";
  }
  config.background.enabled = background;
  config.background.profile = "uniform";
  config.background.bx0 = Real{0.25};
  config.background.by0 = Real{-0.125};
  config.background.bz0 = Real{0.0625};
  return config;
}

EndpointMapping make_mapping(std::size_t device_count) {
  std::vector<DeviceIdentity> devices;
  devices.reserve(device_count);
  for (std::size_t index = 0; index < device_count; ++index) {
    devices.push_back(DeviceIdentity{
        static_cast<int>(index),
        "mhd-checkpoint-test-" + std::to_string(index), {}});
  }
  const std::vector<RankDeviceAssignment> assignments{{
      /*world_rank=*/0, /*node_local_rank=*/0,
      "mhd-checkpoint-test-node", std::move(devices)}};
  return quasar::distributed::make_endpoint_mapping(assignments);
}

EndpointMapping make_multirank_mapping(int rank_count,
                                       std::size_t devices_per_rank = 1) {
  std::vector<RankDeviceAssignment> assignments;
  assignments.reserve(static_cast<std::size_t>(rank_count));
  for (int rank = 0; rank < rank_count; ++rank) {
    std::vector<DeviceIdentity> devices;
    devices.reserve(devices_per_rank);
    for (std::size_t local = 0; local < devices_per_rank; ++local) {
      const std::size_t ordinal =
          static_cast<std::size_t>(rank) * devices_per_rank + local;
      devices.push_back(DeviceIdentity{
          static_cast<int>(ordinal),
          "mhd-checkpoint-rank-" + std::to_string(rank) + "-device-" +
              std::to_string(local),
          {}});
    }
    assignments.push_back(RankDeviceAssignment{
        rank, rank, "mhd-checkpoint-test-node",
        std::move(devices)});
  }
  return quasar::distributed::make_endpoint_mapping(assignments);
}

std::filesystem::path temporary_checkpoint(std::string_view suffix) {
  const auto nonce = std::chrono::steady_clock::now()
                         .time_since_epoch()
                         .count();
  return std::filesystem::temp_directory_path() /
      ("quasar-mhd-checkpoint-" + std::to_string(nonce) +
       std::string{suffix} + ".h5");
}

std::filesystem::path shared_temporary_checkpoint(
    MpiRuntime& runtime, std::string_view suffix) {
  std::string path = runtime.rank() == 0
      ? temporary_checkpoint(suffix).string()
      : std::string{};
  std::uint64_t size = static_cast<std::uint64_t>(path.size());
  quasar::distributed::check_mpi(
      MPI_Bcast(&size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD),
      "MPI_Bcast(MHD checkpoint test path size)");
  if (runtime.rank() != 0) path.resize(static_cast<std::size_t>(size));
  quasar::distributed::check_mpi(
      MPI_Bcast(path.data(), static_cast<int>(path.size()), MPI_CHAR, 0,
                MPI_COMM_WORLD),
      "MPI_Bcast(MHD checkpoint test path)");
  return path;
}

MhdTileRuntime make_runtime(MpiRuntime& mpi, std::size_t device_count,
                            DecompositionShape shape,
                            quasar::mhd::MhdConfig config) {
  auto topology = VirtualTopology::create(
      static_cast<std::size_t>(config.grid.nx),
      static_cast<std::size_t>(config.grid.ny), device_count, shape,
      /*minimum_tile_width=*/2);
  return MhdTileRuntime{
      mpi, make_mapping(device_count), std::move(topology), std::move(config)};
}

MhdTileRuntime make_multirank_runtime(
    MpiRuntime& mpi, DecompositionShape shape,
    quasar::mhd::MhdConfig config,
    std::size_t devices_per_rank = 1) {
  const std::size_t endpoint_count =
      static_cast<std::size_t>(mpi.size()) * devices_per_rank;
  auto topology = VirtualTopology::create(
      static_cast<std::size_t>(config.grid.nx),
      static_cast<std::size_t>(config.grid.ny), endpoint_count, shape,
      /*minimum_tile_width=*/2);
  return MhdTileRuntime{
      mpi, make_multirank_mapping(mpi.size(), devices_per_rank),
      std::move(topology),
      std::move(config)};
}

std::vector<Real> read_real_dataset(const std::filesystem::path& path,
                                    std::string_view name,
                                    std::size_t elements) {
  hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file < 0) fail("failed to open checkpoint for dataset verification");
  const std::string dataset_name{name};
  hid_t dataset = H5Dopen2(file, dataset_name.c_str(), H5P_DEFAULT);
  std::vector<Real> values(elements);
  bool success = dataset >= 0 &&
      H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              values.data()) >= 0;
  if (dataset >= 0) success = H5Dclose(dataset) >= 0 && success;
  success = H5Fclose(file) >= 0 && success;
  if (!success) fail("failed to read checkpoint dataset for verification");
  return values;
}

bool corrupt_periodic_bx_duplicate(const std::filesystem::path& path,
                                   std::size_t nx) {
  hid_t file = H5Fopen(path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  if (file < 0) return false;
  hid_t dataset = H5Dopen2(file, "mhd/state/bx_face", H5P_DEFAULT);
  hid_t file_space = dataset >= 0 ? H5Dget_space(dataset) : -1;
  const hsize_t offset[2]{0, static_cast<hsize_t>(nx)};
  const hsize_t count[2]{1, 1};
  bool success = file_space >= 0 &&
      H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset, nullptr,
                          count, nullptr) >= 0;
  hid_t memory_space = success ? H5Screate_simple(2, count, nullptr) : -1;
  const Real corrupted = Real{12345.5};
  if (memory_space >= 0) {
    success = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, memory_space, file_space,
                       H5P_DEFAULT, &corrupted) >= 0;
  }
  if (memory_space >= 0) success = H5Sclose(memory_space) >= 0 && success;
  if (file_space >= 0) success = H5Sclose(file_space) >= 0 && success;
  if (dataset >= 0) success = H5Dclose(dataset) >= 0 && success;
  success = H5Fclose(file) >= 0 && success;
  return success;
}

template <class Function>
void require_collective_rejection(Function&& function,
                                  std::string_view description) {
  bool rejected = false;
  try {
    function();
  } catch (const DistributedCollectiveError&) {
    rejected = true;
  }
  require(rejected, std::string{description} + " was not rejected");
}

template <class Function>
void require_poisoned_rejection(Function&& function,
                                std::string_view description) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception& error) {
    rejected = std::string_view{error.what()}.find("poisoned") !=
        std::string_view::npos;
  }
  require(rejected, std::string{description} +
                        " was not rejected as a poisoned-runtime operation");
}

void test_single_gpu(MpiRuntime& mpi) {
  constexpr std::size_t nx = 8;
  constexpr std::size_t ny = 6;
  const MhdGlobalState seed = make_state(nx, ny);
  const MhdGlobalBackground background = make_background(nx, ny);
  const std::filesystem::path path = temporary_checkpoint("-single");
  const std::filesystem::path second = temporary_checkpoint("-roundtrip");
  const std::vector<std::uint8_t> diagnostic_state{
      'm', 'h', 'd', '-', 'h', 'i', 's', 't', 'o', 'r', 'y'};
  std::error_code ignored;
  try {
    {
      auto runtime = make_runtime(
          mpi, /*device_count=*/1, {1, 1},
          make_config(nx, ny, /*background=*/true));
      runtime.seed(seed, &background);
      runtime.write_checkpoint(path, /*step=*/19, /*time=*/0.375,
                               "normalized", diagnostic_state);
      runtime.close();
    }

    {
      auto runtime = make_runtime(
          mpi, /*device_count=*/1, {1, 1},
          make_config(nx, ny, /*background=*/true));
      runtime.seed(seed, &background);
      require_collective_rejection(
          [&] {
            // `path` is an existing regular file, so it cannot be used as the
            // parent directory of another checkpoint. This reaches the real
            // parallel-HDF5 writer construction path and fails collectively
            // without touching the committed file at `path`.
            runtime.write_checkpoint(
                path / "uncreatable-child.h5", /*step=*/20,
                /*time=*/0.5, "normalized");
          },
          "parallel-HDF5 checkpoint write");
      require(runtime.poisoned(),
              "failed MHD checkpoint write did not poison the runtime");
      require_poisoned_rejection(
          [&] { (void)runtime.cfl_limit(); }, "CFL diagnostic");
      require_poisoned_rejection(
          [&] { (void)runtime.gather_state(); }, "state diagnostic");
      require_poisoned_rejection(
          [&] { (void)runtime.global_cell_sums(); }, "sum diagnostic");
      require_poisoned_rejection(
          [&] { runtime.step(Real{1e-6}); }, "step");
      require_poisoned_rejection(
          [&] {
            runtime.write_checkpoint(path, /*step=*/20, /*time=*/0.5,
                                     "normalized");
          },
          "second checkpoint write");
      require_poisoned_rejection(
          [&] {
            (void)runtime.restart_from_checkpoint(
                path, "normalized", &background);
          },
          "restart");
      require_poisoned_rejection(
          [&] {
            (void)runtime.checkpoint_metadata(
                /*step=*/20, /*time=*/0.5, "normalized", &background);
          },
          "checkpoint metadata diagnostic");
      runtime.close();
    }

    {
      auto incompatible_config = make_config(nx, ny, /*background=*/true);
      incompatible_config.grid.lx = Real{3};
      auto runtime = make_runtime(
          mpi, /*device_count=*/1, {1, 1},
          std::move(incompatible_config));
      require_collective_rejection(
          [&] {
            (void)runtime.restart_from_checkpoint(
                path, "normalized", &background);
          },
          "physical-mesh mismatch");
      require(!runtime.seeded(),
              "incompatible restart mutated the runtime");
      runtime.close();
    }

    {
      auto incompatible_config = make_config(nx, ny, /*background=*/true);
      incompatible_config.timestep_signature = "policy=auto";
      auto runtime = make_runtime(
          mpi, /*device_count=*/1, {1, 1},
          std::move(incompatible_config));
      require_collective_rejection(
          [&] {
            (void)runtime.restart_from_checkpoint(
                path, "normalized", &background);
          },
          "timestep-policy mismatch");
      require(!runtime.seeded(),
              "timestep-incompatible restart mutated the runtime");
      runtime.close();
    }

    {
      auto incompatible_config = make_config(nx, ny, /*background=*/true);
      incompatible_config.timestep_signature = "policy=fixed;dt=0x1p-4";
      auto runtime = make_runtime(
          mpi, /*device_count=*/1, {1, 1},
          std::move(incompatible_config));
      require_collective_rejection(
          [&] {
            (void)runtime.restart_from_checkpoint(
                path, "normalized", &background);
          },
          "fixed-timestep value mismatch");
      require(!runtime.seeded(),
              "fixed-timestep-incompatible restart mutated the runtime");
      runtime.close();
    }

    {
      auto compatible_config = make_config(nx, ny, /*background=*/true);
      compatible_config.grid.nghost = 3;
      auto runtime = make_runtime(
          mpi, /*device_count=*/1, {1, 1}, std::move(compatible_config));
      (void)runtime.restart_from_checkpoint(
          path, "normalized", &background);
      require_equal(seed, runtime.gather_state());
      runtime.close();
    }

    {
      auto mismatched_background = background;
      mismatched_background.b0z_cell[0] += Real{0.03125};
      auto runtime = make_runtime(
          mpi, /*device_count=*/1, {1, 1},
          make_config(nx, ny, /*background=*/true));
      require_collective_rejection(
          [&] {
            (void)runtime.restart_from_checkpoint(
                path, "normalized", &mismatched_background);
          },
          "explicit background content mismatch");
      require(!runtime.seeded(),
              "background-incompatible restart mutated the runtime");
      runtime.close();
    }

    {
      auto runtime = make_runtime(
          mpi, /*device_count=*/1, {1, 1},
          make_config(nx, ny, /*background=*/true));
      // This fresh readback is also the atomic-preservation assertion for the
      // failed write above: the pre-existing committed image must retain its
      // original metadata and every state value.
      std::vector<std::vector<std::uint8_t>> restored_diagnostics;
      const auto metadata = runtime.restart_from_checkpoint(
          path, "normalized", &background, &restored_diagnostics);
      require(metadata.step == 19 && metadata.time == 0.375,
              "restart did not return the stored step/time");
      require(restored_diagnostics.size() == 1
                  && restored_diagnostics.front() == diagnostic_state,
              "MHD checkpoint diagnostic state differs");
      require_equal(seed, runtime.gather_state());
      // Writing the restored runtime again exercises background restoration:
      // the second checkpoint stages B0 from the live tile solvers.
      runtime.write_checkpoint(second, metadata.step, metadata.time,
                               "normalized");
      runtime.close();
    }
    require_equal(background.b0x_face,
                  read_real_dataset(second, "mhd/background/b0x_face",
                                    background.b0x_face.size()),
                  "restored b0x_face");
    require_equal(background.b0y_face,
                  read_real_dataset(second, "mhd/background/b0y_face",
                                    background.b0y_face.size()),
                  "restored b0y_face");
    require_equal(background.b0z_cell,
                  read_real_dataset(second, "mhd/background/b0z_cell",
                                    background.b0z_cell.size()),
                  "restored b0z_cell");

    require(corrupt_periodic_bx_duplicate(path, nx),
            "failed to corrupt the periodic duplicate test fixture");
    {
      auto runtime = make_runtime(
          mpi, /*device_count=*/1, {1, 1},
          make_config(nx, ny, /*background=*/true));
      require_collective_rejection(
          [&] {
            (void)runtime.restart_from_checkpoint(
                path, "normalized", &background);
          },
          "periodic high-face corruption");
      require(!runtime.seeded(),
              "corrupt restart mutated the runtime");
      runtime.close();
    }
  } catch (...) {
    std::filesystem::remove(path, ignored);
    std::filesystem::remove(second, ignored);
    throw;
  }
  std::filesystem::remove(path, ignored);
  std::filesystem::remove(second, ignored);
}

void test_repartition(MpiRuntime& mpi) {
  constexpr std::size_t nx = 9;
  constexpr std::size_t ny = 7;
  const MhdGlobalState seed = make_state(nx, ny);
  const std::filesystem::path path = temporary_checkpoint("-repartition");
  std::error_code ignored;
  try {
    {
      auto runtime = make_runtime(
          mpi, /*device_count=*/2, {2, 1},
          make_config(nx, ny, /*background=*/false));
      runtime.seed(seed);
      runtime.write_checkpoint(path, /*step=*/23, /*time=*/0.5,
                               "normalized");
      runtime.close();
    }
    {
      auto runtime = make_runtime(
          mpi, /*device_count=*/2, {1, 2},
          make_config(nx, ny, /*background=*/false));
      const auto metadata =
          runtime.restart_from_checkpoint(path, "normalized");
      require(metadata.step == 23 && metadata.time == 0.5,
              "repartitioned restart lost step/time");
      require_equal(seed, runtime.gather_state());
      runtime.close();
    }
  } catch (...) {
    std::filesystem::remove(path, ignored);
    throw;
  }
  std::filesystem::remove(path, ignored);
}

void test_multirank(MpiRuntime& mpi) {
  constexpr std::size_t nx = 9;
  constexpr std::size_t ny = 7;
  const MhdGlobalState seed = make_state(nx, ny);
  const std::filesystem::path path =
      shared_temporary_checkpoint(mpi, "-multirank");
  std::error_code ignored;
  try {
    {
      auto runtime = make_multirank_runtime(
          mpi, {2, 1}, make_config(nx, ny, /*background=*/false));
      runtime.seed(seed);
      runtime.write_checkpoint(path, /*step=*/31, /*time=*/0.75,
                               "normalized");
      runtime.close();
    }
    {
      auto runtime = make_multirank_runtime(
          mpi, {1, 2}, make_config(nx, ny, /*background=*/false));
      const auto metadata =
          runtime.restart_from_checkpoint(path, "normalized");
      require(metadata.step == 31 && metadata.time == 0.75,
              "multi-rank restart lost step/time");
      require_equal(seed, runtime.gather_state());
      runtime.close();
    }
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
    {
      auto runtime = make_multirank_runtime(
          mpi, {1, 2}, make_config(nx, ny, /*background=*/false));
      runtime.inject_checkpoint_metadata_copy_failure_for_testing(
          mpi.rank() == 1);
      bool rejected = false;
      try {
        (void)runtime.restart_from_checkpoint(path, "normalized");
      } catch (const DistributedCollectiveError& error) {
        rejected = error.resolution().representative.phase_text() ==
                       "mhd-restart-metadata-copy" &&
            error.resolution().representative.rank == 1;
      }
      require(mpi.allreduce_all(rejected),
              "rank-local MHD metadata copy failure was not rejected "
              "collectively");
      require(!runtime.seeded() && !runtime.poisoned(),
              "pre-mutation MHD metadata copy failure changed runtime state");

      // The failed attempt must collectively close its reader. The one-shot
      // injection is consumed, so the same runtime can reopen and restore.
      const auto metadata =
          runtime.restart_from_checkpoint(path, "normalized");
      require(metadata.step == 31 && metadata.time == 0.75,
              "MHD retry after metadata copy failure lost step/time");
      require_equal(seed, runtime.gather_state());
      runtime.close();
    }
    {
      auto runtime = make_multirank_runtime(
          mpi, {1, 2}, make_config(nx, ny, /*background=*/false));
      // Only rank one reports the injected failure. The post-publication
      // consensus must make every rank throw the same collective exception and
      // enter the poisoned state after tile state has already been reconciled.
      runtime.inject_restart_post_reconcile_failure_for_testing(
          mpi.rank() == 1);
      require_collective_rejection(
          [&] {
            (void)runtime.restart_from_checkpoint(path, "normalized");
          },
          "post-reconcile MHD restart failure");
      require(runtime.seeded() &&
                  runtime.telemetry().state_reconciliations > 0,
              "injected MHD restart failure occurred before tile publication "
              "and reconciliation");
      require(mpi.allreduce_all(runtime.poisoned()),
              "post-reconcile MHD restart failure did not poison every rank");
      require_poisoned_rejection(
          [&] { runtime.seed(seed); }, "seed after failed restart");
      require_poisoned_rejection(
          [&] { (void)runtime.cfl_limit(); },
          "CFL diagnostic after failed restart");
      require_poisoned_rejection(
          [&] { (void)runtime.gather_state(); },
          "state diagnostic after failed restart");
      require_poisoned_rejection(
          [&] { (void)runtime.divergence_b_max(); },
          "divergence diagnostic after failed restart");
      require_poisoned_rejection(
          [&] { (void)runtime.gather_cell_component("rho"); },
          "cell diagnostic after failed restart");
      require_poisoned_rejection(
          [&] { (void)runtime.local_owned_shards(); },
          "shard diagnostic after failed restart");
      require_poisoned_rejection(
          [&] { (void)runtime.global_cell_sums(); },
          "sum diagnostic after failed restart");
      require_poisoned_rejection(
          [&] { runtime.step(Real{1e-6}); }, "step after failed restart");
      require_poisoned_rejection(
          [&] {
            runtime.write_checkpoint(path, /*step=*/32, /*time=*/1.0,
                                     "normalized");
          },
          "checkpoint write after failed restart");
      require_poisoned_rejection(
          [&] {
            (void)runtime.restart_from_checkpoint(path, "normalized");
          },
          "second restart after failed restart");
      require_poisoned_rejection(
          [&] {
            (void)runtime.checkpoint_metadata(
                /*step=*/32, /*time=*/1.0, "normalized");
          },
          "checkpoint metadata after failed restart");
      runtime.close();
      require(runtime.closed(),
              "poisoned MHD restart runtime did not close collectively");
    }
#endif
  } catch (...) {
    mpi.barrier();
    if (mpi.rank() == 0) std::filesystem::remove(path, ignored);
    throw;
  }
  mpi.barrier();
  if (mpi.rank() == 0) std::filesystem::remove(path, ignored);
}

void write_cross_process_fixture(MpiRuntime& mpi,
                                 const std::filesystem::path& path) {
  constexpr std::size_t nx = 9;
  constexpr std::size_t ny = 7;
  auto runtime = make_runtime(
      mpi, /*device_count=*/1, {1, 1},
      make_config(nx, ny, /*background=*/false));
  runtime.seed(make_evolvable_state(nx, ny));
  const Real dt = Real{0.1} * runtime.cfl_limit();
  runtime.step(dt);
  runtime.write_checkpoint(path, /*step=*/42, /*time=*/dt, "normalized");
  runtime.close();
}

void read_cross_process_fixture(MpiRuntime& mpi,
                                const std::filesystem::path& path) {
  constexpr std::size_t nx = 9;
  constexpr std::size_t ny = 7;
  auto runtime = make_multirank_runtime(
      mpi, {2, 1}, make_config(nx, ny, /*background=*/false));
  const auto metadata =
      runtime.restart_from_checkpoint(path, "normalized");
  require(metadata.step == 42 && metadata.time > 0.0,
          "cross-process restart lost step/time");
  const MhdGlobalState restored = runtime.gather_state();
  const Real limit = runtime.cfl_limit();
  require(std::isfinite(limit) && limit > Real{0},
          "evolved MHD restart did not restore trusted divergence provenance");
  runtime.step(Real{0.05} * limit);
  runtime.close();

  // Recompute the writer's one-step state through the destination placement.
  // This makes the 1-rank -> 2-rank fixture validate every restored lattice,
  // rather than accepting a finite but scrambled hyperslab repartition.
  auto reference = make_multirank_runtime(
      mpi, {2, 1}, make_config(nx, ny, /*background=*/false));
  reference.seed(make_evolvable_state(nx, ny));
  reference.step(static_cast<Real>(metadata.time));
  require_close(reference.gather_state(), restored, Real{2e-10});
  reference.close();
  mpi.barrier();
  if (mpi.rank() == 0) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
  }
}

std::filesystem::path continued_path(const std::filesystem::path& path) {
  return std::filesystem::path{path.string() + ".continued"};
}

void remove_cross_placement_paths(MpiRuntime& mpi,
                                  const std::filesystem::path& path) {
  if (mpi.rank() == 0) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove(continued_path(path), ignored);
  }
  mpi.barrier();
}

void write_cross_placement_fixture(MpiRuntime& mpi,
                                   const std::filesystem::path& path) {
  constexpr std::size_t nx = 9;
  constexpr std::size_t ny = 7;
  remove_cross_placement_paths(mpi, path);
  auto runtime = make_multirank_runtime(
      mpi, {2, 2}, make_config(nx, ny, /*background=*/false),
      /*devices_per_rank=*/2);
  try {
    runtime.seed(make_evolvable_state(nx, ny));
    const Real dt = Real{0.1} * runtime.cfl_limit();
    runtime.step(dt);
    runtime.write_checkpoint(path, /*step=*/42, /*time=*/dt,
                             "normalized");
    runtime.step(dt);
    runtime.write_checkpoint(continued_path(path), /*step=*/43,
                             /*time=*/Real{2} * dt, "normalized");
    runtime.close();
  } catch (...) {
    try {
      if (!runtime.closed()) runtime.close();
    } catch (...) {
    }
    remove_cross_placement_paths(mpi, path);
    throw;
  }
}

void read_cross_placement_fixture(MpiRuntime& mpi,
                                  const std::filesystem::path& path) {
  constexpr std::size_t nx = 9;
  constexpr std::size_t ny = 7;
  MhdGlobalState continued;
  try {
    {
      auto runtime = make_runtime(
          mpi, /*device_count=*/4, {4, 1},
          make_config(nx, ny, /*background=*/false));
      const auto metadata =
          runtime.restart_from_checkpoint(path, "normalized");
      require(metadata.step == 42 && metadata.time > Real{0},
              "cross-placement MHD base checkpoint metadata differs");
      runtime.step(static_cast<Real>(metadata.time));
      continued = runtime.gather_state();
      runtime.close();
    }
    {
      auto reference = make_runtime(
          mpi, /*device_count=*/4, {4, 1},
          make_config(nx, ny, /*background=*/false));
      const auto metadata = reference.restart_from_checkpoint(
          continued_path(path), "normalized");
      require(metadata.step == 43 &&
                  metadata.time > Real{0},
              "cross-placement MHD continuation checkpoint metadata differs");
      require_close(reference.gather_state(), continued, Real{2e-10});
      reference.close();
    }
  } catch (...) {
    remove_cross_placement_paths(mpi, path);
    throw;
  }
  remove_cross_placement_paths(mpi, path);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string_view mode = argc >= 2 ? argv[1] : "";
  const bool cross_process =
      mode == "write-cross" || mode == "read-cross" ||
      mode == "write-cross-four" || mode == "read-cross-four";
  if ((!cross_process && argc != 2) || (cross_process && argc != 3) ||
      (mode != "single" && mode != "repartition" && mode != "multirank" &&
       mode != "write-cross" && mode != "read-cross" &&
       mode != "write-cross-four" && mode != "read-cross-four")) {
    std::cerr << "usage: test_mhd_checkpoint "
                 "single|repartition|multirank|write-cross PATH|read-cross PATH|"
                 "write-cross-four PATH|read-cross-four PATH\n";
    return 2;
  }

  std::unique_ptr<MpiRuntime> mpi;
  try {
    mpi = std::make_unique<MpiRuntime>(&argc, &argv);
    const bool multirank = mode == "multirank" || mode == "read-cross" ||
        mode == "write-cross-four";
    if ((!multirank && mpi->size() != 1) || (multirank && mpi->size() != 2)) {
      if (mpi->rank() == 0) {
        std::cerr << "test_mhd_checkpoint " << mode
                  << " requires MPI world size " << (multirank ? 2 : 1)
                  << '\n';
      }
      mpi->close();
      return kSkip;
    }
    const bool repartition = mode == "repartition";
    const int required_devices =
        mode == "write-cross-four" || mode == "read-cross-four"
        ? 4
        : repartition || mode == "multirank" || mode == "read-cross" ? 2 : 1;
    if (quasar::backend::device_count() < required_devices) {
      std::cerr << "test_mhd_checkpoint " << mode << " requires "
                << required_devices << " visible GPU(s)\n";
      mpi->close();
      return kSkip;
    }
    if (mode == "write-cross") {
      write_cross_process_fixture(*mpi, argv[2]);
    } else if (mode == "read-cross") {
      read_cross_process_fixture(*mpi, argv[2]);
    } else if (mode == "write-cross-four") {
      write_cross_placement_fixture(*mpi, argv[2]);
    } else if (mode == "read-cross-four") {
      read_cross_placement_fixture(*mpi, argv[2]);
    } else if (mode == "multirank") {
      test_multirank(*mpi);
    } else if (repartition) {
      test_repartition(*mpi);
    } else {
      test_single_gpu(*mpi);
    }
    mpi->close();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_mhd_checkpoint: " << error.what() << '\n';
    if (mpi && !mpi->closed()) {
      try {
        mpi->close();
      } catch (const std::exception& close_error) {
        std::cerr << "test_mhd_checkpoint close: " << close_error.what()
                  << '\n';
      }
    }
    return 1;
  }
}
