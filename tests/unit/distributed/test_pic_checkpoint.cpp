#include "quasar/backend/device.hpp"
#include "quasar/distributed/device_mapping.hpp"
#include "quasar/distributed/mpi_runtime.hpp"
#include "quasar/distributed/pic_runtime.hpp"
#include "quasar/distributed/topology.hpp"

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
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
using quasar::distributed::MpiRuntime;
using quasar::distributed::PicGlobalFields;
using quasar::distributed::PicGlobalState;
using quasar::distributed::PicSpeciesState;
using quasar::distributed::PicTileRuntime;
using quasar::distributed::RankDeviceAssignment;
using quasar::distributed::VirtualTopology;

constexpr int kSkip = 77;

void require(bool condition, std::string message) {
  if (!condition) throw std::runtime_error{std::move(message)};
}

void require_values(std::span<const Real> expected,
                    std::span<const Real> actual,
                    std::string_view name, Real tolerance = Real{0}) {
  require(expected.size() == actual.size(),
          std::string{name} + " size differs");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (std::fabs(expected[index] - actual[index]) > tolerance) {
      throw std::runtime_error{std::string{name} + " differs at " +
                               std::to_string(index)};
    }
  }
}

template <class T>
void require_values(std::span<const T> expected,
                    std::span<const T> actual,
                    std::string_view name) {
  require(std::equal(expected.begin(), expected.end(), actual.begin(),
                     actual.end()),
          std::string{name} + " differs");
}

void require_fields(const PicGlobalFields& expected,
                    const PicGlobalFields& actual,
                    std::string_view prefix,
                    Real tolerance = Real{0}) {
  require(expected.global_nx == actual.global_nx &&
              expected.global_ny == actual.global_ny,
          std::string{prefix} + " mesh differs");
  require_values(expected.ex, actual.ex, std::string{prefix} + " Ex",
                 tolerance);
  require_values(expected.ey, actual.ey, std::string{prefix} + " Ey",
                 tolerance);
  require_values(expected.ez, actual.ez, std::string{prefix} + " Ez",
                 tolerance);
  require_values(expected.bx, actual.bx, std::string{prefix} + " Bx",
                 tolerance);
  require_values(expected.by, actual.by, std::string{prefix} + " By",
                 tolerance);
  require_values(expected.bz, actual.bz, std::string{prefix} + " Bz",
                 tolerance);
}

void require_particles(
    const quasar::pic::ParticleSpecies::HostSnapshot& expected,
    const quasar::pic::ParticleSpecies::HostSnapshot& actual,
    std::string_view prefix, Real tolerance = Real{0}) {
  require_values(expected.x, actual.x, std::string{prefix} + " x",
                 tolerance);
  require_values(expected.y, actual.y, std::string{prefix} + " y",
                 tolerance);
  require_values(expected.x_prev, actual.x_prev,
                 std::string{prefix} + " x_prev", tolerance);
  require_values(expected.y_prev, actual.y_prev,
                 std::string{prefix} + " y_prev", tolerance);
  require_values(expected.vx, actual.vx, std::string{prefix} + " vx",
                 tolerance);
  require_values(expected.vy, actual.vy, std::string{prefix} + " vy",
                 tolerance);
  require_values(expected.vz, actual.vz, std::string{prefix} + " vz",
                 tolerance);
  require_values(expected.vphi_deposit, actual.vphi_deposit,
                 std::string{prefix} + " vphi", tolerance);
  require_values(expected.weight, actual.weight,
                 std::string{prefix} + " weight", tolerance);
  require_values<std::uint8_t>(expected.alive, actual.alive,
                               std::string{prefix} + " alive");
  require_values<std::uint64_t>(expected.id, actual.id,
                                std::string{prefix} + " id");
}

void require_state(const PicGlobalState& expected,
                   const PicGlobalState& actual,
                   Real tolerance = Real{0}) {
  require_fields(expected.fields, actual.fields, "live", tolerance);
  require_fields(expected.external_fields, actual.external_fields,
                 "external", tolerance);
  require_values(expected.previous_bx, actual.previous_bx, "previous Bx",
                 tolerance);
  require_values(expected.previous_by, actual.previous_by, "previous By",
                 tolerance);
  require_values(expected.previous_bz, actual.previous_bz, "previous Bz",
                 tolerance);
  require_values(expected.sources.jx, actual.sources.jx, "Jx", tolerance);
  require_values(expected.sources.jy, actual.sources.jy, "Jy", tolerance);
  require_values(expected.sources.jz, actual.sources.jz, "Jz", tolerance);
  require_values(expected.sources.charge, actual.sources.charge, "charge",
                 tolerance);
  require(expected.step_count == actual.step_count &&
              expected.previous_dt == actual.previous_dt &&
              expected.has_previous_dt == actual.has_previous_dt &&
              expected.background_initialized ==
                  actual.background_initialized &&
              expected.background_charge_density ==
                  actual.background_charge_density,
          "PIC runtime metadata differs");
  require(expected.species.size() == actual.species.size(),
          "PIC species count differs");
  for (std::size_t kind = 0; kind < expected.species.size(); ++kind) {
    require(expected.species[kind].config.name ==
                actual.species[kind].config.name &&
                expected.species[kind].config.charge ==
                    actual.species[kind].config.charge &&
                expected.species[kind].config.mass ==
                    actual.species[kind].config.mass,
            "PIC species configuration differs");
    require_particles(expected.species[kind].particles,
                      actual.species[kind].particles,
                      "species " + std::to_string(kind), tolerance);
  }
  for (int side = 0; side < 4; ++side) {
    require_values(expected.boundary.mur_history[side],
                   actual.boundary.mur_history[side],
                   "Mur side " + std::to_string(side), tolerance);
    require(expected.boundary.mur_primed[side] ==
                actual.boundary.mur_primed[side],
            "Mur priming differs");
  }
  require_values(expected.boundary.outflow_corner_history,
                 actual.boundary.outflow_corner_history,
                 "outflow corners", tolerance);
  require(expected.boundary.outflow_corners_primed ==
              actual.boundary.outflow_corners_primed,
          "outflow corner priming differs");
}

EndpointMapping make_mapping(std::size_t devices) {
  std::vector<DeviceIdentity> identities;
  for (std::size_t index = 0; index < devices; ++index) {
    identities.push_back({static_cast<int>(index),
                          "pic-checkpoint-" + std::to_string(index), {}});
  }
  const std::vector<RankDeviceAssignment> assignments{{
      0, 0, "pic-checkpoint-node", std::move(identities)}};
  return quasar::distributed::make_endpoint_mapping(assignments);
}

EndpointMapping make_multirank_mapping(const MpiRuntime& mpi,
                                       std::size_t devices_per_rank = 1) {
  std::vector<RankDeviceAssignment> assignments;
  for (int rank = 0; rank < mpi.size(); ++rank) {
    std::vector<DeviceIdentity> identities;
    for (std::size_t local = 0; local < devices_per_rank; ++local) {
      const std::size_t ordinal =
          static_cast<std::size_t>(rank) * devices_per_rank + local;
      identities.push_back({
          static_cast<int>(ordinal),
          "pic-checkpoint-rank-" + std::to_string(rank) + "-device-" +
              std::to_string(local), {}});
    }
    assignments.push_back({
        rank, rank, "pic-checkpoint-multirank-node",
        std::move(identities)});
  }
  return quasar::distributed::make_endpoint_mapping(assignments);
}

quasar::pic::EmPicConfig make_config(std::size_t nx, std::size_t ny) {
  quasar::pic::EmPicConfig config;
  config.grid = quasar::Grid2D{static_cast<int>(nx), static_cast<int>(ny),
                               Real{1}, Real{1}, Real{0}, Real{0}, 1};
  config.fdtd_order = 2;
  config.shape = "cic";
  config.geometry = "cartesian";
  config.neutralizing_background = true;
  config.timestep_signature = "policy=auto;dt=0x1p-4";
  for (int side = 0; side < 4; ++side) {
    config.boundary.field[side] = "periodic";
    config.boundary.particle[side] = "periodic";
  }
  return config;
}

quasar::pic::EmPicConfig make_outflow_config(std::size_t nx,
                                              std::size_t ny) {
  auto config = make_config(nx, ny);
  config.neutralizing_background = false;
  for (int side = 0; side < 4; ++side) {
    config.boundary.field[side] = "outflow";
    config.boundary.particle[side] = "absorbing";
  }
  return config;
}

quasar::pic::EmPicConfig make_cylindrical_config(std::size_t nx,
                                                  std::size_t ny) {
  auto config = make_config(nx, ny);
  config.geometry = "cylindrical";
  config.neutralizing_background = false;
  config.grid = quasar::Grid2D{static_cast<int>(nx), static_cast<int>(ny),
                               Real{1}, Real{1.5}, Real{0}, Real{0}, 1};
  config.boundary.field[0] = "axis";
  config.boundary.particle[0] = "axis";
  config.boundary.field[1] = "pec";
  config.boundary.particle[1] = "specular";
  return config;
}

std::vector<Real> component(std::size_t width, std::size_t height,
                            std::size_t nx, std::size_t ny,
                            bool face_x, bool face_y, Real offset) {
  std::vector<Real> values(width * height);
  for (std::size_t y = 0; y < height; ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      const std::size_t cx = face_x && x == nx ? 0 : x;
      const std::size_t cy = face_y && y == ny ? 0 : y;
      values[y * width + x] =
          offset + static_cast<Real>(cx + 11 * cy) / Real{8192};
    }
  }
  return values;
}

PicGlobalFields make_fields(std::size_t nx, std::size_t ny, Real scale) {
  PicGlobalFields fields;
  fields.global_nx = nx;
  fields.global_ny = ny;
  fields.ex = component(nx + 1, ny, nx, ny, true, false,
                        scale * Real{0.01});
  fields.ey = component(nx, ny + 1, nx, ny, false, true,
                        scale * Real{-0.02});
  fields.ez = component(nx, ny, nx, ny, false, false,
                        scale * Real{0.03});
  fields.bx = component(nx, ny + 1, nx, ny, false, true,
                        scale * Real{-0.04});
  fields.by = component(nx + 1, ny, nx, ny, true, false,
                        scale * Real{0.05});
  fields.bz = component(nx + 1, ny + 1, nx, ny, true, true,
                        scale * Real{-0.06});
  return fields;
}

PicGlobalFields make_cylindrical_fields(std::size_t nx, std::size_t ny,
                                        Real scale) {
  PicGlobalFields fields;
  fields.global_nx = nx;
  fields.global_ny = ny;
  fields.ex = component(nx + 1, ny, nx, ny, false, false,
                        scale * Real{0.01});
  fields.ey = component(nx, ny + 1, nx, ny, false, true,
                        scale * Real{-0.02});
  fields.ez = component(nx + 1, ny, nx, ny, false, false,
                        scale * Real{0.03});
  fields.bx = component(nx + 1, ny + 1, nx, ny, false, true,
                        scale * Real{-0.04});
  fields.by = component(nx, ny, nx, ny, false, false,
                        scale * Real{0.05});
  fields.bz = component(nx + 1, ny + 1, nx, ny, false, true,
                        scale * Real{-0.06});
  return fields;
}

PicSpeciesState make_species(std::size_t nx, std::size_t ny) {
  (void)nx;
  (void)ny;
  PicSpeciesState state;
  state.config = {"electrons", Real{-1}, Real{1}, 3};
  auto& p = state.particles;
  p.x = {Real{0.2}, Real{0.52}, Real{0.81}};
  p.y = {Real{0.25}, Real{0.71}, Real{0.43}};
  p.x_prev = p.x;
  p.y_prev = p.y;
  p.vx = {Real{0.01}, Real{-0.015}, Real{0.005}};
  p.vy = {Real{-0.004}, Real{0.006}, Real{-0.003}};
  p.vz = {Real{0.002}, Real{-0.001}, Real{0.004}};
  p.vphi_deposit = p.vz;
  p.weight = {Real{0.2}, Real{0.3}, Real{0.25}};
  p.alive = {1, 1, 1};
  p.id = {9003, 101, 700};
  return state;
}

PicSpeciesState make_uneven_multirank_species() {
  PicSpeciesState state;
  state.config = {"uneven-electrons", Real{-1}, Real{1}, 7};
  auto& p = state.particles;
  // In the 2x1 writer every record belongs to rank 0. After a 1x2 restart the
  // y positions split 4/3 between ranks, so balanced HDF5 readers must route
  // records independently of both their writer and reader rank chunks.
  p.x = {Real{0.11}, Real{0.17}, Real{0.23}, Real{0.29},
         Real{0.34}, Real{0.39}, Real{0.44}};
  p.y = {Real{0.08}, Real{0.21}, Real{0.35}, Real{0.49},
         Real{0.58}, Real{0.73}, Real{0.91}};
  p.x_prev = p.x;
  p.y_prev = p.y;
  p.vx.assign(p.x.size(), Real{0});
  p.vy.assign(p.x.size(), Real{0});
  p.vz.assign(p.x.size(), Real{0});
  p.vphi_deposit.assign(p.x.size(), Real{0});
  p.weight.assign(p.x.size(), Real{0.125});
  p.alive.assign(p.x.size(), std::uint8_t{1});
  p.id = {7007, 11, 505, 42, 9001, 313, 88};
  return state;
}

std::vector<quasar::pic::SpeciesConfig> species_configs(
    std::span<const PicSpeciesState> species) {
  std::vector<quasar::pic::SpeciesConfig> result;
  for (const auto& entry : species) {
    result.push_back({entry.config.name, entry.config.charge,
                      entry.config.mass, entry.config.capacity});
  }
  return result;
}

PicTileRuntime make_runtime(MpiRuntime& mpi, std::size_t devices,
                            DecompositionShape shape,
                            quasar::pic::EmPicConfig config) {
  auto topology = VirtualTopology::create(
      static_cast<std::size_t>(config.grid.nx),
      static_cast<std::size_t>(config.grid.ny), devices, shape, 2);
  return PicTileRuntime{mpi, make_mapping(devices), std::move(topology),
                        std::move(config)};
}

PicTileRuntime make_multirank_runtime(MpiRuntime& mpi,
                                      DecompositionShape shape,
                                      quasar::pic::EmPicConfig config,
                                      std::size_t devices_per_rank = 1) {
  const std::size_t endpoint_count =
      static_cast<std::size_t>(mpi.size()) * devices_per_rank;
  auto topology = VirtualTopology::create(
      static_cast<std::size_t>(config.grid.nx),
      static_cast<std::size_t>(config.grid.ny),
      endpoint_count, shape, 2);
  return PicTileRuntime{mpi,
                        make_multirank_mapping(mpi, devices_per_rank),
                        std::move(topology), std::move(config)};
}

std::filesystem::path temporary_path(std::string_view suffix) {
  const auto nonce = std::chrono::steady_clock::now()
                         .time_since_epoch()
                         .count();
  return std::filesystem::temp_directory_path() /
      ("quasar-pic-checkpoint-" + std::to_string(nonce) +
       std::string{suffix} + ".h5");
}

std::filesystem::path shared_path(MpiRuntime& mpi,
                                  std::string_view suffix) {
  std::string path = mpi.rank() == 0
      ? temporary_path(suffix).string()
      : std::string{};
  std::uint64_t size = path.size();
  quasar::distributed::check_mpi(
      MPI_Bcast(&size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD),
      "MPI_Bcast(PIC checkpoint path size)");
  if (mpi.rank() != 0) path.resize(static_cast<std::size_t>(size));
  quasar::distributed::check_mpi(
      MPI_Bcast(path.data(), static_cast<int>(path.size()), MPI_CHAR, 0,
                MPI_COMM_WORLD),
      "MPI_Bcast(PIC checkpoint path)");
  return path;
}

template <class Function>
void require_rejected(Function&& function, std::string_view name) {
  bool rejected = false;
  try {
    function();
  } catch (const DistributedCollectiveError&) {
    rejected = true;
  }
  require(rejected, std::string{name} + " was not rejected");
}

template <class Function>
void require_poisoned_rejection(Function&& function,
                                std::string_view name) {
  bool rejected = false;
  try {
    function();
  } catch (const std::exception& error) {
    rejected = std::string_view{error.what()}.find("poisoned") !=
        std::string_view::npos;
  }
  require(rejected, std::string{name} +
                        " was not rejected as a poisoned-runtime operation");
}

void test_periodic_roundtrip(MpiRuntime& mpi) {
  constexpr std::size_t nx = 8;
  constexpr std::size_t ny = 6;
  const std::vector<PicSpeciesState> species{make_species(nx, ny)};
  const auto configs = species_configs(species);
  const std::filesystem::path path = temporary_path("-single");
  auto checkpoint_config = make_config(nx, ny);
  checkpoint_config.external_field_signature = "checkpoint-external-v1";
  const std::vector<std::uint8_t> diagnostic_state{
      'p', 'i', 'c', '-', 'h', 'i', 's', 't', 'o', 'r', 'y'};
  std::error_code ignored;
  PicGlobalState expected;
  try {
    {
      auto runtime = make_runtime(mpi, 1, {1, 1}, checkpoint_config);
      const auto fields = make_fields(nx, ny, Real{1});
      const auto external = make_fields(nx, ny, Real{0.25});
      runtime.seed(fields, &external, species);
      runtime.step(Real{0.1} * runtime.cfl_limit());
      expected = runtime.gather_state();
      runtime.write_checkpoint(path, expected.step_count, 0.125,
                               "normalized", diagnostic_state);
      runtime.close();
    }
    {
      auto runtime = make_runtime(mpi, 1, {1, 1}, checkpoint_config);
      std::vector<std::vector<std::uint8_t>> restored_diagnostics;
      const auto metadata = runtime.restart_from_checkpoint(
          path, "normalized", configs, &restored_diagnostics);
      require(metadata.step == expected.step_count && metadata.time == 0.125,
              "PIC checkpoint metadata differs");
      require(restored_diagnostics.size() == 1
                  && restored_diagnostics.front() == diagnostic_state,
              "PIC checkpoint diagnostic state differs");
      require_state(expected, runtime.gather_state());
      runtime.close();
    }
    {
      auto wrong = configs;
      wrong[0].mass = Real{2};
      auto runtime = make_runtime(mpi, 1, {1, 1}, checkpoint_config);
      require_rejected(
          [&] {
            (void)runtime.restart_from_checkpoint(path, "normalized", wrong);
          },
          "PIC species mismatch");
      require(!runtime.seeded(), "rejected PIC restart mutated the runtime");
      runtime.close();
    }
    {
      auto undersized = configs;
      undersized[0].capacity = species[0].particles.x.size() - 1;
      auto runtime = make_runtime(mpi, 1, {1, 1}, checkpoint_config);
      require_rejected(
          [&] {
            (void)runtime.restart_from_checkpoint(
                path, "normalized", undersized);
          },
          "PIC particle count above configured capacity");
      require(!runtime.seeded(),
              "capacity-rejected PIC restart mutated the runtime");
      runtime.close();
    }
    {
      auto wrong_timestep = checkpoint_config;
      wrong_timestep.timestep_signature = "policy=fixed;dt=0x1p-4";
      auto runtime = make_runtime(mpi, 1, {1, 1}, wrong_timestep);
      require_rejected(
          [&] {
            (void)runtime.restart_from_checkpoint(
                path, "normalized", configs);
          },
          "PIC timestep-policy mismatch");
      require(!runtime.seeded(),
              "timestep-incompatible PIC restart mutated the runtime");
      runtime.close();
    }
    {
      auto compatible_halo = checkpoint_config;
      compatible_halo.grid.nghost = 2;
      auto runtime = make_runtime(mpi, 1, {1, 1}, compatible_halo);
      (void)runtime.restart_from_checkpoint(path, "normalized", configs);
      require_state(expected, runtime.gather_state());
      runtime.close();
    }
    {
      auto wrong_background = checkpoint_config;
      wrong_background.external_field_signature = "checkpoint-external-v2";
      auto runtime = make_runtime(mpi, 1, {1, 1}, wrong_background);
      require_rejected(
          [&] {
            (void)runtime.restart_from_checkpoint(
                path, "normalized", configs);
          },
          "PIC external-field mismatch");
      require(!runtime.seeded(),
              "rejected PIC background restart mutated the runtime");
      runtime.close();
    }
  } catch (...) {
    std::filesystem::remove(path, ignored);
    throw;
  }
  std::filesystem::remove(path, ignored);
}

void test_outflow_history(MpiRuntime& mpi) {
  constexpr std::size_t nx = 8;
  constexpr std::size_t ny = 6;
  const std::filesystem::path path = temporary_path("-outflow");
  std::error_code ignored;
  PicGlobalState continued;
  const Real dt_fraction = Real{0.08};
  try {
    {
      auto runtime = make_runtime(
          mpi, 1, {1, 1}, make_outflow_config(nx, ny));
      runtime.seed(make_fields(nx, ny, Real{0.3}), nullptr, {});
      const Real dt = dt_fraction * runtime.cfl_limit();
      runtime.step(dt);
      const auto committed = runtime.gather_state();
      runtime.write_checkpoint(path, committed.step_count, 0.25,
                               "normalized");
      runtime.step(dt);
      continued = runtime.gather_state();
      runtime.close();
    }
    {
      auto runtime = make_runtime(
          mpi, 1, {1, 1}, make_outflow_config(nx, ny));
      (void)runtime.restart_from_checkpoint(path, "normalized", {});
      runtime.step(dt_fraction * runtime.cfl_limit());
      require_state(continued, runtime.gather_state(), Real{2e-13});
      runtime.close();
    }
  } catch (...) {
    std::filesystem::remove(path, ignored);
    throw;
  }
  std::filesystem::remove(path, ignored);
}

void test_failed_write_is_fatal(MpiRuntime& mpi) {
  constexpr std::size_t nx = 8;
  constexpr std::size_t ny = 6;
  const std::filesystem::path path = temporary_path("-fatal-write");
  std::error_code ignored;
  PicGlobalState committed;
  try {
    {
      auto runtime = make_runtime(mpi, 1, {1, 1}, make_config(nx, ny));
      runtime.seed(make_fields(nx, ny, Real{0.45}), nullptr, {});
      runtime.step(Real{0.05} * runtime.cfl_limit());
      committed = runtime.gather_state();
      runtime.write_checkpoint(path, committed.step_count, 0.625,
                               "normalized");

      bool rejected = false;
      try {
        // The mismatched absolute step fails after a temporary parallel-HDF5
        // file has been opened, exercising cleanup without replacing `path`.
        runtime.write_checkpoint(path, committed.step_count + 1, 0.75,
                                 "normalized");
      } catch (const DistributedCollectiveError&) {
        rejected = true;
      }
      require(rejected && runtime.poisoned(),
              "failed PIC checkpoint did not poison the runtime");
      bool only_close_legal = false;
      try {
        (void)runtime.gather_state();
      } catch (const std::logic_error&) {
        only_close_legal = true;
      }
      require(only_close_legal,
              "poisoned PIC checkpoint runtime allowed further work");
      runtime.close();
    }
    {
      auto runtime = make_runtime(mpi, 1, {1, 1}, make_config(nx, ny));
      const auto metadata = runtime.restart_from_checkpoint(
          path, "normalized", {});
      require(metadata.step == committed.step_count &&
                  metadata.time == 0.625,
              "failed PIC checkpoint replaced the preceding committed file");
      require_state(committed, runtime.gather_state());
      runtime.close();
    }
  } catch (...) {
    std::filesystem::remove(path, ignored);
    throw;
  }
  std::filesystem::remove(path, ignored);
}

void test_cylindrical_roundtrip(MpiRuntime& mpi) {
  constexpr std::size_t nx = 8;
  constexpr std::size_t ny = 6;
  const std::filesystem::path path = temporary_path("-cylindrical");
  std::error_code ignored;
  PicGlobalState committed;
  PicGlobalState continued;
  try {
    {
      auto runtime = make_runtime(
          mpi, 1, {1, 1}, make_cylindrical_config(nx, ny));
      runtime.seed(make_cylindrical_fields(nx, ny, Real{0.2}), nullptr, {});
      const Real dt = Real{0.05} * runtime.cfl_limit();
      runtime.step(dt);
      committed = runtime.gather_state();
      runtime.write_checkpoint(path, committed.step_count, 0.375,
                               "normalized");
      runtime.step(dt);
      continued = runtime.gather_state();
      runtime.close();
    }
    {
      auto runtime = make_runtime(
          mpi, 1, {1, 1}, make_cylindrical_config(nx, ny));
      const auto metadata = runtime.restart_from_checkpoint(
          path, "normalized", {});
      require(metadata.geometry == "cylindrical" &&
                  metadata.step == committed.step_count,
              "cylindrical PIC checkpoint metadata differs");
      require_state(committed, runtime.gather_state());
      runtime.step(Real{0.05} * runtime.cfl_limit());
      require_state(continued, runtime.gather_state(), Real{3e-12});
      runtime.close();
    }
  } catch (...) {
    std::filesystem::remove(path, ignored);
    throw;
  }
  std::filesystem::remove(path, ignored);
}

void test_repartition(MpiRuntime& mpi) {
  constexpr std::size_t nx = 9;
  constexpr std::size_t ny = 7;
  const std::vector<PicSpeciesState> species{make_species(nx, ny)};
  const auto configs = species_configs(species);
  const std::filesystem::path path = temporary_path("-repartition");
  std::error_code ignored;
  PicGlobalState expected;
  try {
    {
      auto runtime = make_runtime(mpi, 2, {2, 1}, make_config(nx, ny));
      runtime.seed(make_fields(nx, ny, Real{0.2}), nullptr, species);
      runtime.step(Real{0.05} * runtime.cfl_limit());
      expected = runtime.gather_state();
      runtime.write_checkpoint(path, expected.step_count, 0.5,
                               "normalized");
      require(runtime.telemetry().checkpoint_local_lattice_writes == 19,
              "PIC checkpoint did not write all lattices from local tiles");
      require(runtime.telemetry().checkpoint_global_lattice_materializations ==
                  0,
              "PIC checkpoint write materialized a global lattice");
      runtime.close();
    }
    {
      auto runtime = make_runtime(mpi, 2, {1, 2}, make_config(nx, ny));
      (void)runtime.restart_from_checkpoint(path, "normalized", configs);
      require(runtime.telemetry().checkpoint_local_lattice_reads == 19,
              "PIC restart did not read all lattices into local tiles");
      require(runtime.telemetry().checkpoint_global_lattice_materializations ==
                  0,
              "PIC restart materialized a global lattice");
      require_state(expected, runtime.gather_state());
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
  const std::vector<PicSpeciesState> species{
      make_uneven_multirank_species()};
  const auto configs = species_configs(species);
  const std::filesystem::path path = shared_path(mpi, "-multirank");
  std::error_code ignored;
  PicGlobalState expected;
  try {
    {
      auto runtime = make_multirank_runtime(
          mpi, {2, 1}, make_config(nx, ny));
      runtime.seed(make_fields(nx, ny, Real{0.15}), nullptr, species);
      expected = runtime.gather_state();
      runtime.write_checkpoint(path, expected.step_count, 0.75,
                               "normalized");
      runtime.close();
    }
    {
      auto runtime = make_multirank_runtime(
          mpi, {1, 2}, make_config(nx, ny));
      (void)runtime.restart_from_checkpoint(path, "normalized", configs);
      const auto restored = runtime.gather_state();
      require(restored.species.size() == 1 &&
                  restored.species.front().particles.id.size() == 7,
              "uneven multi-rank PIC restart lost particle records");
      require_state(expected, restored);
      runtime.close();
    }
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
    {
      auto runtime = make_multirank_runtime(
          mpi, {1, 2}, make_config(nx, ny));
      runtime.inject_checkpoint_metadata_copy_failure_for_testing(
          mpi.rank() == 1);
      bool rejected = false;
      try {
        (void)runtime.restart_from_checkpoint(path, "normalized", configs);
      } catch (const DistributedCollectiveError& error) {
        rejected = error.resolution().representative.phase_text() ==
                       "pic-restart-metadata-copy" &&
            error.resolution().representative.rank == 1;
      }
      require(mpi.allreduce_all(rejected),
              "rank-local PIC metadata copy failure was not rejected "
              "collectively");
      require(!runtime.seeded() && !runtime.poisoned(),
              "pre-mutation PIC metadata copy failure changed runtime state");

      // The first attempt must collectively close its reader. Reopening on
      // the same runtime proves that the one-shot failure left no live HDF5
      // collective state behind.
      const auto metadata = runtime.restart_from_checkpoint(
          path, "normalized", configs);
      require(metadata.step == expected.step_count && metadata.time == 0.75,
              "PIC retry after metadata copy failure lost step/time");
      require_state(expected, runtime.gather_state());
      runtime.close();
    }
    {
      auto runtime = make_multirank_runtime(
          mpi, {1, 2}, make_config(nx, ny));
      runtime.inject_restart_post_reconcile_failure_for_testing(
          mpi.rank() == 1);
      bool rejected = false;
      try {
        (void)runtime.restart_from_checkpoint(path, "normalized", configs);
      } catch (const DistributedCollectiveError& error) {
        rejected = error.resolution().representative.phase_text() ==
            "pic-restart-failure";
      }
      require(mpi.allreduce_all(rejected),
              "post-reconcile PIC restart failure was not rejected "
              "collectively");
      require(runtime.seeded() &&
                  runtime.telemetry().checkpoint_local_lattice_reads == 19,
              "injected PIC restart failure occurred before state publication");
      require(mpi.allreduce_all(runtime.poisoned()),
              "post-reconcile PIC restart failure did not poison every rank");
      require_poisoned_rejection(
          [&] { (void)runtime.gather_state(); },
          "PIC diagnostic after failed restart");
      require_poisoned_rejection(
          [&] { runtime.step(Real{1e-6}); },
          "PIC step after failed restart");
      require_poisoned_rejection(
          [&] {
            runtime.write_checkpoint(path, expected.step_count, 1.0,
                                     "normalized");
          },
          "PIC checkpoint write after failed restart");
      require_poisoned_rejection(
          [&] {
            (void)runtime.restart_from_checkpoint(
                path, "normalized", configs);
          },
          "second PIC restart after failed restart");
      runtime.close();
      require(runtime.closed(),
              "poisoned PIC restart runtime did not close collectively");
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

std::filesystem::path continued_path(const std::filesystem::path& path) {
  return std::filesystem::path{path.string() + ".continued"};
}

void remove_cross_rank_paths(MpiRuntime& mpi,
                             const std::filesystem::path& path) {
  if (mpi.rank() == 0) {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove(continued_path(path), ignored);
  }
  mpi.barrier();
}

void test_cross_rank_write(MpiRuntime& mpi,
                           const std::filesystem::path& path) {
  constexpr std::size_t nx = 9;
  constexpr std::size_t ny = 7;
  remove_cross_rank_paths(mpi, path);
  auto runtime = make_multirank_runtime(
      mpi, {2, 2}, make_config(nx, ny), 2);
  try {
    const std::vector<PicSpeciesState> species{make_species(nx, ny)};
    const auto fields = make_fields(nx, ny, Real{0.33});
    const auto external = make_fields(nx, ny, Real{0.17});
    runtime.seed(fields, &external, species);
    const Real dt = Real{0.04} * runtime.cfl_limit();
    runtime.step(dt);
    const auto first = runtime.gather_state();
    runtime.write_checkpoint(path, first.step_count, 0.5, "normalized");
    runtime.step(dt);
    const auto second = runtime.gather_state();
    runtime.write_checkpoint(continued_path(path), second.step_count,
                             0.5 + static_cast<double>(dt), "normalized");
    runtime.close();
  } catch (...) {
    try {
      if (!runtime.closed()) runtime.close();
    } catch (...) {
    }
    remove_cross_rank_paths(mpi, path);
    throw;
  }
}

void test_cross_rank_read(MpiRuntime& mpi,
                          const std::filesystem::path& path) {
  constexpr std::size_t nx = 9;
  constexpr std::size_t ny = 7;
  const std::vector<PicSpeciesState> species{make_species(nx, ny)};
  const auto configs = species_configs(species);
  PicGlobalState continued;
  try {
    {
      auto runtime = make_runtime(mpi, 4, {4, 1}, make_config(nx, ny));
      const auto metadata = runtime.restart_from_checkpoint(
          path, "normalized", configs);
      require(metadata.step == 1,
              "cross-rank PIC base checkpoint step differs");
      runtime.step(Real{0.04} * runtime.cfl_limit());
      continued = runtime.gather_state();
      runtime.close();
    }
    {
      auto reference = make_runtime(
          mpi, 4, {4, 1}, make_config(nx, ny));
      const auto metadata = reference.restart_from_checkpoint(
          continued_path(path), "normalized", configs);
      require(metadata.step == 2,
              "cross-rank PIC continuation checkpoint step differs");
      require_state(reference.gather_state(), continued, Real{4e-12});
      reference.close();
    }
  } catch (...) {
    remove_cross_rank_paths(mpi, path);
    throw;
  }
  remove_cross_rank_paths(mpi, path);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string_view mode = argc >= 2 ? argv[1] : "";
  const bool cross_mode = mode == "cross-write" || mode == "cross-read";
  if ((mode != "single" && mode != "repartition" && mode != "multirank" &&
       !cross_mode) || (cross_mode && argc != 3) ||
      (!cross_mode && argc != 2)) {
    std::cerr << "usage: test_pic_checkpoint "
                 "single|repartition|multirank|cross-write PATH|cross-read PATH\n";
    return 2;
  }
  std::unique_ptr<MpiRuntime> mpi;
  try {
    mpi = std::make_unique<MpiRuntime>(&argc, &argv);
    const bool multirank = mode == "multirank" || mode == "cross-write";
    if ((!multirank && mpi->size() != 1) ||
        (multirank && mpi->size() != 2)) {
      mpi->close();
      return kSkip;
    }
    const int required_devices =
        cross_mode ? 4 : mode == "single" ? 1 : 2;
    if (quasar::backend::device_count() < required_devices) {
      mpi->close();
      return kSkip;
    }
    if (mode == "single") {
      test_periodic_roundtrip(*mpi);
      test_outflow_history(*mpi);
      test_failed_write_is_fatal(*mpi);
      test_cylindrical_roundtrip(*mpi);
    } else if (mode == "repartition") {
      test_repartition(*mpi);
    } else if (mode == "multirank") {
      test_multirank(*mpi);
    } else if (mode == "cross-write") {
      test_cross_rank_write(*mpi, argv[2]);
    } else {
      test_cross_rank_read(*mpi, argv[2]);
    }
    mpi->close();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_pic_checkpoint: " << error.what() << '\n';
    if (mpi && !mpi->closed()) {
      try {
        mpi->close();
      } catch (const std::exception& close_error) {
        std::cerr << "test_pic_checkpoint close: " << close_error.what()
                  << '\n';
      }
    }
    return 1;
  }
}
