#include "quasar/backend/device.hpp"
#include "quasar/distributed/device_mapping.hpp"
#include "quasar/distributed/mpi_runtime.hpp"
#include "quasar/distributed/pic_runtime.hpp"
#include "quasar/distributed/topology.hpp"
#include "quasar/physics/pic/diagnostics.hpp"

#include "host_field_evaluator.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
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
using quasar::distributed::PicGlobalSources;
using quasar::distributed::PicGlobalState;
using quasar::distributed::PicOwnedArray;
using quasar::distributed::PicOwnedFields;
using quasar::distributed::PicOwnedShard;
using quasar::distributed::PicSpeciesState;
using quasar::distributed::PicTileRuntime;
using quasar::distributed::RankDeviceAssignment;
using quasar::distributed::VirtualTopology;

constexpr int kSkip = 77;

class EmptyFieldSource final : public quasar::core::IFieldSource {};

class LinearElectricEvaluator final : public quasar::test::HostFieldEvaluator {
 protected:
  quasar::Field<quasar::Vec3> host_B(
      const quasar::core::IFieldSource&,
      const quasar::core::PointCloud& observations) const override {
    return quasar::Field<quasar::Vec3>(observations.size());
  }

  quasar::Field<quasar::Vec3> host_E(
      const quasar::core::IFieldSource&,
      const quasar::core::PointCloud& observations) const override {
    quasar::Field<quasar::Vec3> values(observations.size());
    for (std::size_t index = 0; index < observations.size(); ++index) {
      const auto point = observations.points()[index];
      values[index] = {Real{0.5} + Real{0.75} * point.x,
                       Real{-0.125} * point.y, Real{0}};
    }
    return values;
  }
};

class RegistryOnlyIdentityFilter final
    : public quasar::numerics::ICurrentFilter {
 public:
  void apply(quasar::JField2D<Real>&,
             const quasar::boundary::BoundarySpec&,
             bool) const override {}

  void set_passes(int passes) override {
    if (passes < 1) throw std::invalid_argument{"filter passes"};
    passes_ = passes;
  }

  std::vector<quasar::numerics::DistributedFilterStencil>
  distributed_stencils() const override {
    return {{passes_, Real{0}, Real{1}}};
  }

 private:
  int passes_{1};
};

QUASAR_REGISTER_CURRENT_FILTER("distributed_registry_identity",
                               RegistryOnlyIdentityFilter)

void require(bool condition, std::string message) {
  if (!condition) throw std::runtime_error{std::move(message)};
}

void require_same_text(MpiRuntime& mpi, const std::string& text,
                       std::string_view label) {
  constexpr int capacity = 1024;
  require(mpi.allreduce_all(
              text.size() < static_cast<std::size_t>(capacity)),
          std::string{label} + " exceeds the comparison buffer");
  std::array<char, capacity> local{};
  text.copy(local.data(), text.size());
  std::vector<std::array<char, capacity>> gathered(
      static_cast<std::size_t>(mpi.size()));
  quasar::distributed::check_mpi(
      MPI_Allgather(local.data(), capacity, MPI_CHAR, gathered.data(),
                    capacity, MPI_CHAR, MPI_COMM_WORLD),
      "MPI_Allgather(PIC exception text)");
  require(std::all_of(gathered.begin(), gathered.end(),
                      [&](const auto& candidate) {
                        return candidate == gathered.front();
                      }),
          std::string{label} + " differs across MPI ranks");
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

EndpointMapping make_mapping(std::size_t device_count) {
  std::vector<DeviceIdentity> devices;
  for (std::size_t index = 0; index < device_count; ++index) {
    devices.push_back({static_cast<int>(index),
                       "pic-runtime-test-" + std::to_string(index), {}});
  }
  const std::vector<RankDeviceAssignment> assignments{{
      0, 0, "pic-runtime-test-node", std::move(devices)}};
  return quasar::distributed::make_endpoint_mapping(assignments);
}

EndpointMapping make_multirank_mapping(
    const MpiRuntime& mpi, std::size_t devices_per_rank = 1) {
  std::vector<RankDeviceAssignment> assignments;
  for (int rank = 0; rank < mpi.size(); ++rank) {
    std::vector<DeviceIdentity> devices;
    for (std::size_t local = 0; local < devices_per_rank; ++local) {
      const std::size_t ordinal =
          static_cast<std::size_t>(rank) * devices_per_rank + local;
      devices.push_back({
          static_cast<int>(ordinal),
          "pic-runtime-multirank-" + std::to_string(rank) + '-' +
              std::to_string(local), {}});
    }
    assignments.push_back({
        rank, rank, "pic-runtime-multirank-node",
        std::move(devices)});
  }
  return quasar::distributed::make_endpoint_mapping(assignments);
}

EndpointMapping make_worker_failure_mapping(const MpiRuntime& mpi) {
  const int invalid_ordinal = quasar::backend::device_count();
  std::vector<RankDeviceAssignment> assignments;
  assignments.reserve(static_cast<std::size_t>(mpi.size()));
  for (int rank = 0; rank < mpi.size(); ++rank) {
    assignments.push_back({
        rank, rank, "pic-worker-failure-node",
        {{rank == 1 ? invalid_ordinal : 0,
          "pic-worker-failure-device-" + std::to_string(rank), {}}}});
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
  for (int side = 0; side < 4; ++side) {
    config.boundary.field[side] = "periodic";
    config.boundary.particle[side] = "periodic";
  }
  return config;
}

quasar::pic::EmPicConfig make_cylindrical_config(std::size_t nx,
                                                  std::size_t ny) {
  auto config = make_config(nx, ny);
  config.geometry = "cylindrical";
  config.grid = quasar::Grid2D{static_cast<int>(nx), static_cast<int>(ny),
                               Real{1}, Real{1.5}, Real{0}, Real{0}, 1};
  config.boundary.field[0] = "axis";
  config.boundary.particle[0] = "axis";
  config.boundary.field[1] = "pec";
  config.boundary.particle[1] = "specular";
  return config;
}

quasar::pic::EmPicConfig make_cylindrical_order_four_config(
    std::size_t nx, std::size_t ny) {
  auto config = make_cylindrical_config(nx, ny);
  config.fdtd_order = 4;
  config.grid = quasar::Grid2D{static_cast<int>(nx), static_cast<int>(ny),
                               Real{1}, Real{1.5}, Real{0}, Real{0}, 2};
  return config;
}

quasar::pic::EmPicConfig make_order_four_config(std::size_t nx,
                                                std::size_t ny) {
  auto config = make_config(nx, ny);
  config.fdtd_order = 4;
  config.grid = quasar::Grid2D{static_cast<int>(nx), static_cast<int>(ny),
                               Real{1}, Real{1}, Real{0}, Real{0}, 2};
  config.filters = {{"binomial", 1}, {"compensated_binomial", 2}};
  return config;
}

quasar::pic::EmPicConfig make_external_config(std::size_t nx,
                                               std::size_t ny) {
  auto config = make_config(nx, ny);
  config.shape = "tsc";
  config.grid = quasar::Grid2D{static_cast<int>(nx), static_cast<int>(ny),
                               Real{1}, Real{1}, Real{0}, Real{0}, 2};
  config.external_field_signature = "linear-electric-v1";
  for (int side = 0; side < 4; ++side) {
    config.boundary.field[side] = "pec";
    config.boundary.particle[side] = "specular";
  }
  return config;
}

quasar::pic::EmPicConfig make_tsc_config(std::size_t nx,
                                         std::size_t ny) {
  auto config = make_config(nx, ny);
  config.shape = "tsc";
  config.grid = quasar::Grid2D{static_cast<int>(nx), static_cast<int>(ny),
                               Real{1}, Real{1}, Real{0}, Real{0}, 2};
  return config;
}

std::vector<Real> make_component(std::size_t width, std::size_t height,
                                 std::size_t cells_x, std::size_t cells_y,
                                 bool face_x, bool face_y, Real offset) {
  std::vector<Real> result(width * height);
  for (std::size_t y = 0; y < height; ++y) {
    for (std::size_t x = 0; x < width; ++x) {
      const std::size_t canonical_x = face_x && x == cells_x ? 0 : x;
      const std::size_t canonical_y = face_y && y == cells_y ? 0 : y;
      result[y * width + x] =
          offset + static_cast<Real>(canonical_x + 17 * canonical_y) /
                       Real{1024};
    }
  }
  return result;
}

PicGlobalFields make_fields(std::size_t nx, std::size_t ny) {
  PicGlobalFields fields;
  fields.global_nx = nx;
  fields.global_ny = ny;
  fields.ex = make_component(nx + 1, ny, nx, ny, true, false, Real{0.1});
  fields.ey = make_component(nx, ny + 1, nx, ny, false, true, Real{-0.2});
  fields.ez = make_component(nx, ny, nx, ny, false, false, Real{0.3});
  fields.bx = make_component(nx, ny + 1, nx, ny, false, true, Real{-0.4});
  fields.by = make_component(nx + 1, ny, nx, ny, true, false, Real{0.5});
  fields.bz = make_component(nx + 1, ny + 1, nx, ny, true, true, Real{-0.6});
  return fields;
}

PicGlobalFields make_cylindrical_fields(std::size_t nx, std::size_t ny) {
  PicGlobalFields fields;
  fields.global_nx = nx;
  fields.global_ny = ny;
  fields.ex = make_component(nx + 1, ny, nx, ny, false, false, Real{0.01});
  fields.ey = make_component(nx, ny + 1, nx, ny, false, true, Real{-0.02});
  fields.ez = make_component(nx + 1, ny, nx, ny, false, false, Real{0.03});
  fields.bx = make_component(nx + 1, ny + 1, nx, ny, false, true, Real{-0.04});
  fields.by = make_component(nx, ny, nx, ny, false, false, Real{0.05});
  fields.bz = make_component(nx + 1, ny + 1, nx, ny, false, true, Real{-0.06});
  return fields;
}

void seed_serial(quasar::pic::EmPic2D3V& solver,
                 const PicGlobalFields& fields) {
  const auto grid = solver.grid();
  const auto upload = [&](quasar::backend::DeviceBuffer<Real>& device,
                          std::span<const Real> source,
                          std::size_t width, std::size_t height) {
    std::vector<Real> padded(grid.storage_size(), Real{0});
    for (std::size_t y = 0; y < height; ++y) {
      for (std::size_t x = 0; x < width; ++x) {
        padded[grid.index(static_cast<int>(x), static_cast<int>(y))] =
            source[y * width + x];
      }
    }
    device.copy_from_host(padded.data(), padded.size());
  };
  auto& live = solver.fields();
  const bool cylindrical = solver.config().geometry == "cylindrical";
  upload(live.ex, fields.ex, fields.global_nx + 1, fields.global_ny);
  upload(live.ey, fields.ey, fields.global_nx, fields.global_ny + 1);
  upload(live.ez, fields.ez,
         fields.global_nx + (cylindrical ? 1 : 0), fields.global_ny);
  upload(live.bx, fields.bx,
         fields.global_nx + (cylindrical ? 1 : 0), fields.global_ny + 1);
  upload(live.by, fields.by,
         fields.global_nx + (cylindrical ? 0 : 1),
         fields.global_ny);
  upload(live.bz, fields.bz, fields.global_nx + 1, fields.global_ny + 1);
}

PicGlobalFields gather_serial(const quasar::pic::EmPic2D3V& solver) {
  const auto grid = solver.grid();
  PicGlobalFields result;
  result.global_nx = static_cast<std::size_t>(grid.nx);
  result.global_ny = static_cast<std::size_t>(grid.ny);
  const auto download = [&](const quasar::backend::DeviceBuffer<Real>& device,
                            std::size_t width, std::size_t height) {
    std::vector<Real> padded(grid.storage_size());
    device.copy_to_host(padded.data(), padded.size());
    std::vector<Real> values(width * height);
    for (std::size_t y = 0; y < height; ++y) {
      for (std::size_t x = 0; x < width; ++x) {
        values[y * width + x] =
            padded[grid.index(static_cast<int>(x), static_cast<int>(y))];
      }
    }
    return values;
  };
  const auto& live = solver.fields();
  const bool cylindrical = solver.config().geometry == "cylindrical";
  result.ex = download(live.ex, result.global_nx + 1, result.global_ny);
  result.ey = download(live.ey, result.global_nx, result.global_ny + 1);
  result.ez = download(live.ez,
                       result.global_nx + (cylindrical ? 1 : 0),
                       result.global_ny);
  result.bx = download(live.bx,
                       result.global_nx + (cylindrical ? 1 : 0),
                       result.global_ny + 1);
  result.by = download(live.by,
                       result.global_nx + (cylindrical ? 0 : 1),
                       result.global_ny);
  result.bz = download(live.bz, result.global_nx + 1,
                       result.global_ny + 1);
  return result;
}

PicGlobalSources gather_serial_sources(quasar::pic::EmPic2D3V& solver) {
  const auto grid = solver.grid();
  PicGlobalSources result;
  result.global_nx = static_cast<std::size_t>(grid.nx);
  result.global_ny = static_cast<std::size_t>(grid.ny);
  const auto download = [&](const quasar::backend::DeviceBuffer<Real>& device,
                            std::size_t width, std::size_t height) {
    std::vector<Real> padded(grid.storage_size());
    device.copy_to_host(padded.data(), padded.size());
    std::vector<Real> values(width * height);
    for (std::size_t y = 0; y < height; ++y) {
      for (std::size_t x = 0; x < width; ++x) {
        values[y * width + x] =
            padded[grid.index(static_cast<int>(x), static_cast<int>(y))];
      }
    }
    return values;
  };
  result.jx = download(solver.current().jx, result.global_nx + 1,
                       result.global_ny);
  result.jy = download(solver.current().jy, result.global_nx,
                       result.global_ny + 1);
  const bool cylindrical = solver.config().geometry == "cylindrical";
  result.jz = download(solver.current().jz,
                       result.global_nx + (cylindrical ? 1 : 0),
                       result.global_ny);
  result.charge = download(solver.charge_density().values, result.global_nx,
                           result.global_ny);
  return result;
}

void require_close(std::span<const Real> expected,
                   std::span<const Real> actual, Real tolerance,
                   std::string_view name) {
  require(expected.size() == actual.size(), std::string{name} + " size differs");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (std::fabs(expected[index] - actual[index]) > tolerance) {
      std::ostringstream message;
      message << std::setprecision(std::numeric_limits<Real>::max_digits10)
              << name << " value differs at " << index << ": "
              << expected[index] << " vs " << actual[index]
              << " (absolute delta "
              << std::fabs(expected[index] - actual[index])
              << ", tolerance " << tolerance << ')';
      throw std::runtime_error{message.str()};
    }
  }
}

void require_close(Real expected, Real actual, Real tolerance,
                   std::string_view name) {
  const Real scale = std::max({Real{1}, std::fabs(expected),
                               std::fabs(actual)});
  if (std::fabs(expected - actual) > tolerance * scale) {
    throw std::runtime_error{std::string{name} + " differs: " +
                             std::to_string(expected) + " vs " +
                             std::to_string(actual)};
  }
}

void require_close(const PicGlobalFields& expected,
                   const PicGlobalFields& actual, Real tolerance) {
  require_close(expected.ex, actual.ex, tolerance, "Ex");
  require_close(expected.ey, actual.ey, tolerance, "Ey");
  require_close(expected.ez, actual.ez, tolerance, "Ez");
  require_close(expected.bx, actual.bx, tolerance, "Bx");
  require_close(expected.by, actual.by, tolerance, "By");
  require_close(expected.bz, actual.bz, tolerance, "Bz");
}

struct OwnedComponentDescriptor {
  PicOwnedArray PicOwnedFields::*owned;
  std::vector<Real> PicGlobalFields::*global;
  std::size_t nx;
  std::size_t ny;
  const char* name;
};

void require_owned_component_reconstructs(
    MpiRuntime& mpi, std::span<const PicOwnedShard> shards,
    const PicGlobalFields& gathered,
    const OwnedComponentDescriptor& component, bool external) {
  const std::size_t size = component.nx * component.ny;
  std::vector<Real> reconstructed(size, Real{0});
  std::vector<int> coverage(size, 0);
  for (const auto& shard : shards) {
    const PicOwnedFields& fields =
        external ? shard.external_fields : shard.fields;
    const PicOwnedArray& owned = fields.*(component.owned);
    require(owned.values.size() == owned.nx * owned.ny,
            std::string{component.name} + " owned payload size differs");
    require(owned.offset_x + owned.nx <= component.nx &&
                owned.offset_y + owned.ny <= component.ny,
            std::string{component.name} + " owned payload is out of bounds");
    for (std::size_t y = 0; y < owned.ny; ++y) {
      for (std::size_t x = 0; x < owned.nx; ++x) {
        const std::size_t destination =
            (owned.offset_y + y) * component.nx + owned.offset_x + x;
        require(coverage[destination] == 0,
                std::string{component.name} +
                    " overlaps another rank-local shard");
        reconstructed[destination] = owned.values[y * owned.nx + x];
        coverage[destination] = 1;
      }
    }
  }
  require(MPI_Allreduce(MPI_IN_PLACE, reconstructed.data(),
                        static_cast<int>(reconstructed.size()), MPI_DOUBLE,
                        MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS,
          std::string{"failed to reconstruct "} + component.name);
  require(MPI_Allreduce(MPI_IN_PLACE, coverage.data(),
                        static_cast<int>(coverage.size()), MPI_INT,
                        MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS,
          std::string{"failed to reduce ownership for "} + component.name);
  require(std::all_of(coverage.begin(), coverage.end(),
                      [](int count) { return count == 1; }),
          std::string{component.name} +
              " shards do not cover the global lattice exactly once");
  require_close(gathered.*(component.global), reconstructed, Real{0},
                component.name);
}

PicGlobalFields run_seed_gather(MpiRuntime& mpi, std::size_t devices,
                                DecompositionShape shape, std::size_t nx,
                                std::size_t ny) {
  auto topology = VirtualTopology::create(nx, ny, devices, shape, 2);
  PicTileRuntime runtime{mpi, make_mapping(devices), std::move(topology),
                         make_config(nx, ny)};
  try {
    const auto fields = make_fields(nx, ny);
    runtime.seed(fields, nullptr, {});
    require_close(fields, runtime.gather_state().fields, Real{0});
    runtime.step(Real{0.25} * runtime.cfl_limit());
    auto evolved = runtime.gather_state().fields;
    runtime.close();
    return evolved;
  } catch (...) {
    try {
      if (!runtime.closed()) runtime.close();
    } catch (...) {
    }
    throw;
  }
}

void test_single(MpiRuntime& mpi) {
  constexpr std::size_t nx = 8;
  constexpr std::size_t ny = 6;
  auto topology = VirtualTopology::create(nx, ny, 1, {1, 1}, 2);
  PicTileRuntime runtime{mpi, make_mapping(1), std::move(topology),
                         make_config(nx, ny)};
  try {
    const auto fields = make_fields(nx, ny);
    runtime.seed(fields, nullptr, {});
    require_close(fields, runtime.gather_state().fields, Real{0});
    const Real dt = Real{0.25} * runtime.cfl_limit();
    runtime.step(dt);
    const auto evolved = runtime.gather_state();
    quasar::pic::EmPic2D3V serial{make_config(nx, ny)};
    seed_serial(serial, fields);
    serial.step(dt);
    require_close(gather_serial(serial), evolved.fields, Real{2e-12});
    require_close(quasar::pic::total_em_energy(serial),
                  runtime.total_em_energy(), Real{5e-12},
                  "Cartesian EM energy");
    require_close(quasar::pic::gauss_residual(serial),
                  runtime.gauss_residual(), Real{5e-12},
                  "Cartesian Gauss residual");
    for (const auto* component : {&evolved.fields.ex, &evolved.fields.ey,
                                  &evolved.fields.ez, &evolved.fields.bx,
                                  &evolved.fields.by, &evolved.fields.bz}) {
      require(std::all_of(component->begin(), component->end(),
                          [](Real value) { return std::isfinite(value); }),
              "evolved PIC field is non-finite");
    }
    require(runtime.telemetry().accepted_steps == 1,
            "PIC runtime did not commit its vacuum step");
    runtime.close();
  } catch (...) {
    try {
      if (!runtime.closed()) runtime.close();
    } catch (...) {
    }
    throw;
  }

  // A filter known only to the registry must work under the distributed
  // runtime; dispatching on the two built-in names would reject this case.
  {
    auto registry_config = make_config(nx, ny);
    registry_config.filters = {{"distributed_registry_identity", 2}};
    auto registry_topology =
        VirtualTopology::create(nx, ny, 1, {1, 1}, 1);
    PicTileRuntime registry_runtime{
        mpi, make_mapping(1), std::move(registry_topology),
        std::move(registry_config)};
    try {
      registry_runtime.seed(make_fields(nx, ny), nullptr, {});
      registry_runtime.step(Real{0.1} * registry_runtime.cfl_limit());
      registry_runtime.close();
    } catch (...) {
      try {
        if (!registry_runtime.closed()) registry_runtime.close();
      } catch (...) {
      }
      throw;
    }
  }

  // Explicit decomposition is checked against the PIC scheme reach before
  // worker/device construction, with enough context to fix the deck.
  {
    bool rejected = false;
    std::string rejection;
    try {
      PicTileRuntime too_thin{
          mpi, make_mapping(3),
          VirtualTopology::create(3, 4, 3, {3, 1}),
          make_order_four_config(3, 4)};
      too_thin.close();
    } catch (const DistributedCollectiveError& error) {
      rejected = error.resolution().representative.phase_text() ==
          "pic-runtime-config";
      rejection = error.what();
    }
    require(rejected && rejection.find("fdtd_order=4") != std::string::npos &&
                rejection.find("3x1") != std::string::npos,
            "thin PIC decomposition lacked a scheme-specific rejection");
  }

  // Stable IDs are global across species, including dead records. Rejection
  // occurs before any solver mutation and leaves the runtime reusable.
  PicGlobalState reusable_state;
  auto validation_topology = VirtualTopology::create(nx, ny, 1, {1, 1}, 2);
  PicTileRuntime validation_runtime{
      mpi, make_mapping(1), std::move(validation_topology),
      make_config(nx, ny)};
  try {
    std::vector<PicSpeciesState> duplicate(2);
    for (std::size_t kind = 0; kind < duplicate.size(); ++kind) {
      duplicate[kind].config = {
          kind == 0 ? "duplicate-live" : "duplicate-dead",
          kind == 0 ? Real{-1} : Real{1}, Real{1}, 1};
      auto& particle = duplicate[kind].particles;
      particle.x = {kind == 0 ? Real{0.25} : Real{0.75}};
      particle.y = {Real{0.5}};
      particle.x_prev = particle.x;
      particle.y_prev = particle.y;
      particle.vx = {Real{0}};
      particle.vy = {Real{0}};
      particle.vz = {Real{0}};
      particle.vphi_deposit = {Real{0}};
      particle.weight = {Real{1}};
      particle.alive = {static_cast<std::uint8_t>(kind == 0 ? 1 : 0)};
      particle.id = {0x5151ULL};
    }
    bool rejected = false;
    try {
      validation_runtime.seed(make_fields(nx, ny), nullptr, duplicate);
    } catch (const quasar::distributed::DistributedCollectiveError&) {
      rejected = true;
    }
    require(rejected && !validation_runtime.seeded() &&
                !validation_runtime.poisoned(),
            "cross-species duplicate ID rejection mutated the PIC runtime");
    validation_runtime.seed(make_fields(nx, ny), nullptr, {});
    reusable_state = validation_runtime.gather_state();
    validation_runtime.close();

    auto restore_topology =
        VirtualTopology::create(nx, ny, 1, {1, 1}, 2);
    PicTileRuntime restore_runtime{
        mpi, make_mapping(1), std::move(restore_topology),
        make_config(nx, ny)};
    try {
      PicGlobalState invalid_restore = reusable_state;
      invalid_restore.species = duplicate;
      bool restore_rejected = false;
      try {
        restore_runtime.restore(invalid_restore);
      } catch (const quasar::distributed::DistributedCollectiveError&) {
        restore_rejected = true;
      }
      require(restore_rejected && !restore_runtime.seeded() &&
                  !restore_runtime.poisoned(),
              "duplicate-ID restore rejection mutated the PIC runtime");
      restore_runtime.restore(reusable_state);
      restore_runtime.close();
    } catch (...) {
      try {
        if (!restore_runtime.closed()) restore_runtime.close();
      } catch (...) {
      }
      throw;
    }
  } catch (...) {
    try {
      if (!validation_runtime.closed()) validation_runtime.close();
    } catch (...) {
    }
    throw;
  }

  // Charged-particle one-tile evolution remains the serial algorithm modulo
  // the canonical gather/reapply round trip.
  auto particle_topology = VirtualTopology::create(nx, ny, 1, {1, 1}, 2);
  PicTileRuntime particles{mpi, make_mapping(1), std::move(particle_topology),
                           make_order_four_config(nx, ny)};
  try {
    std::vector<PicSpeciesState> states(2);
    states[0].config = {"negative", Real{-1}, Real{1}, 1};
    states[1].config = {"positive", Real{1}, Real{2}, 1};
    for (std::size_t kind = 0; kind < states.size(); ++kind) {
      auto& snapshot = states[kind].particles;
      snapshot.x = {kind == 0 ? Real{0.3} : Real{0.7}};
      snapshot.y = {kind == 0 ? Real{0.4} : Real{0.6}};
      snapshot.x_prev = snapshot.x;
      snapshot.y_prev = snapshot.y;
      snapshot.vx = {kind == 0 ? Real{0.02} : Real{-0.01}};
      snapshot.vy = {kind == 0 ? Real{-0.01} : Real{0.015}};
      snapshot.vz = {kind == 0 ? Real{0.01} : Real{-0.005}};
      snapshot.vphi_deposit = snapshot.vz;
      snapshot.weight = {Real{1}};
      snapshot.alive = {1};
      snapshot.id = {100 + kind};
    }
    const auto fields = make_fields(nx, ny);
    particles.seed(fields, nullptr, states);
    const Real dt = Real{0.1} * particles.cfl_limit();
    particles.step(dt);
    const auto distributed = particles.gather_state();

    quasar::pic::EmPic2D3V serial{make_order_four_config(nx, ny)};
    seed_serial(serial, fields);
    for (const auto& state : states) {
      quasar::pic::ParticleSpecies species{state.config};
      species.set_host_particles(
          state.particles.x, state.particles.y, state.particles.vx,
          state.particles.vy, state.particles.vz, state.particles.weight,
          state.particles.id);
      serial.add_species(std::move(species));
    }
    serial.step(dt);
    require_close(gather_serial(serial), distributed.fields, Real{3e-12});
    const auto serial_sources = gather_serial_sources(serial);
    require_close(serial_sources.jx, distributed.sources.jx, Real{3e-12}, "Jx");
    require_close(serial_sources.jy, distributed.sources.jy, Real{3e-12}, "Jy");
    require_close(serial_sources.jz, distributed.sources.jz, Real{3e-12}, "Jz");
    require_close(serial_sources.charge, distributed.sources.charge,
                  Real{3e-12}, "charge");
    for (std::size_t kind = 0; kind < states.size(); ++kind) {
      const auto serial_particles = serial.species()[kind].to_host();
      require_close(serial_particles.x,
                    distributed.species[kind].particles.x, Real{3e-12},
                    "particle x");
      require_close(serial_particles.y,
                    distributed.species[kind].particles.y, Real{3e-12},
                    "particle y");
      require(serial_particles.id == distributed.species[kind].particles.id,
              "stable particle ID differs from serial evolution");
    }
    particles.close();
  } catch (...) {
    try {
      if (!particles.closed()) particles.close();
    } catch (...) {
    }
    throw;
  }

  // Prescribed fields must be sampled on each tile's padded coordinates, not
  // reconstructed from the canonical unpadded lattice. A TSC particle next to
  // a physical wall exercises the otherwise-unrepresentable ghost samples.
  auto external_topology =
      VirtualTopology::create(nx, ny, 1, {1, 1}, 2);
  PicTileRuntime external_runtime{
      mpi, make_mapping(1), std::move(external_topology),
      make_external_config(nx, ny)};
  try {
    auto fields = make_fields(nx, ny);
    for (auto* component : {&fields.ex, &fields.ey, &fields.ez,
                            &fields.bx, &fields.by, &fields.bz}) {
      std::fill(component->begin(), component->end(), Real{0});
    }
    PicSpeciesState state;
    state.config = {"wall-probe", Real{1}, Real{1}, 1};
    state.particles.x = {Real{0.01}};
    state.particles.y = {Real{0.45}};
    state.particles.x_prev = state.particles.x;
    state.particles.y_prev = state.particles.y;
    state.particles.vx = {Real{0}};
    state.particles.vy = {Real{0}};
    state.particles.vz = {Real{0}};
    state.particles.vphi_deposit = {Real{0}};
    state.particles.weight = {Real{1}};
    state.particles.alive = {1};
    state.particles.id = {991};

    LinearElectricEvaluator evaluator;
    EmptyFieldSource source;
    external_runtime.seed(
        fields, nullptr, std::span<const PicSpeciesState>{&state, 1});
    external_runtime.sample_external_fields(evaluator, source);
    const Real dt = Real{0.05} * external_runtime.cfl_limit();
    external_runtime.step(dt);
    const auto distributed = external_runtime.gather_state();

    quasar::pic::EmPic2D3V serial{make_external_config(nx, ny)};
    seed_serial(serial, fields);
    quasar::pic::ParticleSpecies serial_species{state.config};
    serial_species.set_host_particles(
        state.particles.x, state.particles.y, state.particles.vx,
        state.particles.vy, state.particles.vz, state.particles.weight,
        state.particles.id);
    serial.add_species(std::move(serial_species));
    quasar::pic::sample_external_field(
        evaluator, source, serial.external_fields(), Real{1}, Real{1},
        Real{1}, "xy", "cartesian", 2);
    serial.step(dt);
    const auto serial_particles = serial.species().front().to_host();
    require_close(serial_particles.x,
                  distributed.species.front().particles.x, Real{3e-12},
                  "external-field particle x");
    require_close(serial_particles.vx,
                  distributed.species.front().particles.vx, Real{3e-12},
                  "external-field particle vx");
    require_close(gather_serial(serial), distributed.fields, Real{3e-12});
    external_runtime.close();
  } catch (...) {
    try {
      if (!external_runtime.closed()) external_runtime.close();
    } catch (...) {
    }
    throw;
  }
}

void test_multi(MpiRuntime& mpi) {
  const auto baseline = run_seed_gather(mpi, 1, {1, 1}, 9, 7);
  const auto split_x = run_seed_gather(mpi, 2, {2, 1}, 9, 7);
  const auto split_y = run_seed_gather(mpi, 2, {1, 2}, 9, 7);
  require_close(baseline, split_x, Real{2e-12});
  require_close(baseline, split_y, Real{2e-12});

  struct DiagnosticState {
    PicGlobalFields fields;
    Real em_energy{0};
    Real gauss{0};
  };
  const auto run_cylindrical = [&](std::size_t devices,
                                   DecompositionShape shape) {
    auto topology = VirtualTopology::create(6, 8, devices, shape, 1);
    PicTileRuntime runtime{mpi, make_mapping(devices), std::move(topology),
                           make_cylindrical_config(6, 8)};
    try {
      const auto fields = make_cylindrical_fields(6, 8);
      runtime.seed(fields, nullptr, {});
      const auto seeded_fields = runtime.gather_state().fields;
      require(seeded_fields.ex[0] == Real{0},
              "cylindrical axis did not pin Er to zero");
      const Real dt = Real{0.2} * runtime.cfl_limit();
      runtime.step(dt);
      DiagnosticState result{
          runtime.gather_state().fields,
          runtime.total_em_energy(), runtime.gauss_residual()};
      if (devices == 1) {
        quasar::pic::EmPic2D3V serial{make_cylindrical_config(6, 8)};
        seed_serial(serial, fields);
        serial.step(dt);
        require_close(quasar::pic::total_em_energy(serial),
                      result.em_energy, Real{7e-12},
                      "cylindrical EM energy");
        require_close(quasar::pic::gauss_residual(serial),
                      result.gauss, Real{7e-12},
                      "cylindrical Gauss residual");
      }
      runtime.close();
      return result;
    } catch (...) {
      try {
        if (!runtime.closed()) runtime.close();
      } catch (...) {
      }
      throw;
    }
  };
  const auto cylindrical_baseline = run_cylindrical(1, {1, 1});
  const auto cylindrical_split = run_cylindrical(2, {2, 1});
  require_close(cylindrical_baseline.fields, cylindrical_split.fields,
                Real{3e-12});
  require_close(cylindrical_baseline.em_energy,
                cylindrical_split.em_energy, Real{7e-12},
                "split cylindrical EM energy");
  require_close(cylindrical_baseline.gauss, cylindrical_split.gauss,
                Real{7e-12}, "split cylindrical Gauss residual");

  // Order-four cylindrical continuity couples every radial row to corrected
  // face 1. Exercise a non-uniform Jr with both a one-tile runtime (against the
  // device/serial implementation) and a radial split (which must broadcast
  // that same face through the fixed exchange plan).
  constexpr std::size_t cylindrical_o4_nx = 12;
  constexpr std::size_t cylindrical_o4_ny = 8;
  PicSpeciesState cylindrical_particles;
  cylindrical_particles.config = {
      "cylindrical-current-probes", Real{-1}, Real{1}, 4};
  cylindrical_particles.particles.x = {
      Real{0.08}, Real{0.27}, Real{0.63}, Real{0.86}};
  cylindrical_particles.particles.y = {
      Real{0.18}, Real{0.57}, Real{0.91}, Real{1.23}};
  cylindrical_particles.particles.x_prev =
      cylindrical_particles.particles.x;
  cylindrical_particles.particles.y_prev =
      cylindrical_particles.particles.y;
  cylindrical_particles.particles.vx = {
      Real{0.035}, Real{-0.017}, Real{0.011}, Real{-0.026}};
  cylindrical_particles.particles.vy = {
      Real{-0.008}, Real{0.013}, Real{-0.006}, Real{0.009}};
  cylindrical_particles.particles.vz = {
      Real{0.004}, Real{-0.003}, Real{0.002}, Real{-0.001}};
  cylindrical_particles.particles.vphi_deposit =
      cylindrical_particles.particles.vz;
  cylindrical_particles.particles.weight = {
      Real{0.7}, Real{1.1}, Real{0.9}, Real{1.3}};
  cylindrical_particles.particles.alive.assign(4, std::uint8_t{1});
  cylindrical_particles.particles.id = {4101, 4102, 4103, 4104};
  auto cylindrical_o4_fields = make_cylindrical_fields(
      cylindrical_o4_nx, cylindrical_o4_ny);
  for (auto* component : {&cylindrical_o4_fields.ex,
                          &cylindrical_o4_fields.ey,
                          &cylindrical_o4_fields.ez,
                          &cylindrical_o4_fields.bx,
                          &cylindrical_o4_fields.by,
                          &cylindrical_o4_fields.bz}) {
    std::fill(component->begin(), component->end(), Real{0});
  }
  const auto cylindrical_o4_config = make_cylindrical_order_four_config(
      cylindrical_o4_nx, cylindrical_o4_ny);
  quasar::pic::EmPic2D3V cylindrical_serial{cylindrical_o4_config};
  seed_serial(cylindrical_serial, cylindrical_o4_fields);
  quasar::pic::ParticleSpecies cylindrical_serial_particles{
      cylindrical_particles.config};
  cylindrical_serial_particles.set_host_particles(
      cylindrical_particles.particles.x,
      cylindrical_particles.particles.y,
      cylindrical_particles.particles.vx,
      cylindrical_particles.particles.vy,
      cylindrical_particles.particles.vz,
      cylindrical_particles.particles.weight,
      cylindrical_particles.particles.id);
  cylindrical_serial.add_species(std::move(cylindrical_serial_particles));
  const Real cylindrical_o4_dt = Real{0.08} * cylindrical_serial.cfl_limit();
  cylindrical_serial.step(cylindrical_o4_dt);
  const PicGlobalSources cylindrical_serial_sources =
      gather_serial_sources(cylindrical_serial);
  bool nonzero_axis_coupling = false;
  for (std::size_t y = 0; y < cylindrical_o4_ny; ++y) {
    nonzero_axis_coupling = nonzero_axis_coupling ||
        std::fabs(cylindrical_serial_sources.jx[
            y * (cylindrical_o4_nx + 1) + 1]) > Real{1e-12};
  }
  require(nonzero_axis_coupling,
          "cylindrical order-four regression did not excite radial face 1");

  const auto run_cylindrical_o4 = [&](std::size_t devices,
                                      DecompositionShape shape) {
    auto topology = VirtualTopology::create(
        cylindrical_o4_nx, cylindrical_o4_ny, devices, shape, 2);
    PicTileRuntime runtime{
        mpi, make_mapping(devices), std::move(topology),
        make_cylindrical_order_four_config(
            cylindrical_o4_nx, cylindrical_o4_ny)};
    try {
      runtime.seed(cylindrical_o4_fields, nullptr,
                   std::span<const PicSpeciesState>{
                       &cylindrical_particles, 1});
      runtime.step(cylindrical_o4_dt);
      auto state = runtime.gather_state();
      runtime.close();
      return state;
    } catch (...) {
      try {
        if (!runtime.closed()) runtime.close();
      } catch (...) {
      }
      throw;
    }
  };
  const auto cylindrical_o4_one = run_cylindrical_o4(1, {1, 1});
  const auto cylindrical_o4_split = run_cylindrical_o4(2, {2, 1});
  const std::array<std::pair<std::string_view, const PicGlobalState*>, 2>
      cylindrical_o4_distributed{{
          {"one-tile", &cylindrical_o4_one},
          {"radial-split", &cylindrical_o4_split},
      }};
  for (const auto& [layout, distributed] : cylindrical_o4_distributed) {
    const std::string label =
        "cylindrical order-four " + std::string{layout};
    require_close(cylindrical_serial_sources.jx,
                  distributed->sources.jx, Real{3e-10},
                  label + " Jr");
    require_close(cylindrical_serial_sources.jy,
                  distributed->sources.jy, Real{3e-10},
                  label + " Jz");
    require_close(cylindrical_serial_sources.jz,
                  distributed->sources.jz, Real{3e-10},
                  label + " Jphi");
    require_close(cylindrical_serial_sources.charge,
                  distributed->sources.charge, Real{3e-10},
                  label + " charge");
  }

  const auto run_charged = [&](std::size_t devices,
                               DecompositionShape shape) {
    auto topology = VirtualTopology::create(9, 7, devices, shape, 2);
    PicTileRuntime runtime{mpi, make_mapping(devices), std::move(topology),
                           make_order_four_config(9, 7)};
    try {
      std::vector<PicSpeciesState> states(2);
      const Real seam = Real{5} / Real{9};
      for (std::size_t kind = 0; kind < states.size(); ++kind) {
        states[kind].config = {
            kind == 0 ? "electron" : "ion",
            kind == 0 ? Real{-1} : Real{1}, Real{1}, 1};
        auto& snapshot = states[kind].particles;
        snapshot.x = {kind == 0 ? seam - Real{0.01} : seam + Real{0.01}};
        snapshot.y = {kind == 0 ? Real{0.45} : Real{0.55}};
        snapshot.x_prev = snapshot.x;
        snapshot.y_prev = snapshot.y;
        snapshot.vx = {kind == 0 ? Real{0.025} : Real{-0.02}};
        snapshot.vy = {kind == 0 ? Real{-0.01} : Real{0.015}};
        snapshot.vz = {kind == 0 ? Real{0.02} : Real{-0.01}};
        snapshot.vphi_deposit = snapshot.vz;
        snapshot.weight = {Real{1}};
        snapshot.alive = {1};
        snapshot.id = {200 + kind};
      }
      runtime.seed(make_fields(9, 7), nullptr, states);
      runtime.step(Real{0.1} * runtime.cfl_limit());
      auto state = runtime.gather_state();
      runtime.close();
      return state;
    } catch (...) {
      try {
        if (!runtime.closed()) runtime.close();
      } catch (...) {
      }
      throw;
    }
  };
  const auto charged_baseline = run_charged(1, {1, 1});
  const auto charged_split = run_charged(2, {2, 1});
  const auto charged_split_y = run_charged(2, {1, 2});
  const std::array<std::pair<std::string_view, const PicGlobalState*>, 2>
      charged_distributed{{
          {"x-split", &charged_split},
          {"y-split", &charged_split_y},
      }};
  for (const auto& [layout, distributed] : charged_distributed) {
    const std::string label = std::string{layout} + " order-four";
    require_close(charged_baseline.fields, distributed->fields, Real{2e-10});
    require_close(charged_baseline.sources.jx, distributed->sources.jx,
                  Real{2e-10}, label + " Jx");
    require_close(charged_baseline.sources.jy, distributed->sources.jy,
                  Real{2e-10}, label + " Jy");
    require_close(charged_baseline.sources.jz, distributed->sources.jz,
                  Real{2e-10}, label + " Jz");
    require_close(charged_baseline.sources.charge,
                  distributed->sources.charge, Real{2e-10},
                  label + " charge");
  }

  struct TscResult {
    PicGlobalState state;
    std::uint64_t migrated{0};
  };
  const auto run_tsc = [&](std::size_t devices, DecompositionShape shape) {
    auto topology = VirtualTopology::create(9, 7, devices, shape, 2);
    PicTileRuntime runtime{mpi, make_mapping(devices), std::move(topology),
                           make_tsc_config(9, 7)};
    try {
      std::vector<PicSpeciesState> states(2);
      const Real seam = Real{5} / Real{9};
      for (std::size_t kind = 0; kind < states.size(); ++kind) {
        states[kind].config = {
            kind == 0 ? "tsc-electron" : "tsc-ion",
            kind == 0 ? Real{-1} : Real{1}, Real{1}, 1};
        auto& particles = states[kind].particles;
        particles.x = {kind == 0 ? seam - Real{1e-5} : Real{0.2}};
        particles.y = {kind == 0 ? Real{0.51} : Real{0.49}};
        particles.x_prev = particles.x;
        particles.y_prev = particles.y;
        particles.vx = {kind == 0 ? Real{0.2} : Real{0}};
        particles.vy = {Real{0}};
        particles.vz = {kind == 0 ? Real{0.03} : Real{-0.03}};
        particles.vphi_deposit = particles.vz;
        particles.weight = {Real{1}};
        particles.alive = {1};
        particles.id = {900 + kind};
      }
      auto fields = make_fields(9, 7);
      for (auto* component : {&fields.ex, &fields.ey, &fields.ez,
                              &fields.bx, &fields.by, &fields.bz}) {
        std::fill(component->begin(), component->end(), Real{0});
      }
      runtime.seed(fields, nullptr, states);
      runtime.step(Real{0.15} * runtime.cfl_limit());
      TscResult result{runtime.gather_state(),
                       runtime.telemetry().migrated_particles};
      runtime.close();
      return result;
    } catch (...) {
      try {
        if (!runtime.closed()) runtime.close();
      } catch (...) {
      }
      throw;
    }
  };
  const auto tsc_baseline = run_tsc(1, {1, 1});
  const auto tsc_split = run_tsc(2, {2, 1});
  require_close(tsc_baseline.state.fields, tsc_split.state.fields,
                Real{3e-10});
  require_close(tsc_baseline.state.sources.jx, tsc_split.state.sources.jx,
                Real{3e-10}, "TSC split Jx");
  require_close(tsc_baseline.state.sources.jy, tsc_split.state.sources.jy,
                Real{3e-10}, "TSC split Jy");
  require_close(tsc_baseline.state.sources.jz, tsc_split.state.sources.jz,
                Real{3e-10}, "TSC split Jz");
  require_close(tsc_baseline.state.sources.charge,
                tsc_split.state.sources.charge, Real{3e-10},
                "TSC split charge");
  require(tsc_split.migrated > 0,
          "distributed TSC particle did not migrate across the tile seam");
  require(tsc_split.state.species.front().particles.x.front() > Real{5}/Real{9},
          "distributed TSC particle did not finish beyond the tile seam");

  auto restart_topology = VirtualTopology::create(9, 7, 2, {1, 2}, 2);
  PicTileRuntime restarted{mpi, make_mapping(2), std::move(restart_topology),
                           make_order_four_config(9, 7)};
  try {
    restarted.restore(charged_split);
    const auto restored = restarted.gather_state();
    require_close(charged_split.fields, restored.fields, Real{2e-12});
    require_close(charged_split.sources.jx, restored.sources.jx,
                  Real{2e-12}, "restored Jx");
    require_close(charged_split.sources.jy, restored.sources.jy,
                  Real{2e-12}, "restored Jy");
    require(charged_split.species[0].particles.id ==
                restored.species[0].particles.id &&
                charged_split.species[1].particles.id ==
                restored.species[1].particles.id,
            "repartitioned restore changed stable particle IDs");
    restarted.close();
  } catch (...) {
    try {
      if (!restarted.closed()) restarted.close();
    } catch (...) {
    }
    throw;
  }

  // Two neutral particles exercise both an ordinary internal seam and the
  // global periodic seam without field/source forces obscuring ownership.
  auto topology = VirtualTopology::create(9, 7, 2, {2, 1}, 2);
  PicTileRuntime migration{mpi, make_mapping(2), std::move(topology),
                           make_config(9, 7)};
  try {
    PicSpeciesState species;
    species.config = {"markers", Real{0}, Real{1}, 2};
    const Real internal_seam = Real{5} / Real{9};
    species.particles.x = {internal_seam - Real{0.001}, Real{0.999}};
    species.particles.y = {Real{0.4}, Real{0.6}};
    species.particles.x_prev = species.particles.x;
    species.particles.y_prev = species.particles.y;
    species.particles.vx = {Real{0.1}, Real{0.1}};
    species.particles.vy = {Real{0}, Real{0}};
    species.particles.vz = {Real{0}, Real{0}};
    species.particles.vphi_deposit = species.particles.vz;
    species.particles.weight = {Real{1}, Real{1}};
    species.particles.alive = {1, 1};
    species.particles.id = {41, 42};
    const auto fields = make_fields(9, 7);
    migration.seed(fields, nullptr, std::span<const PicSpeciesState>{&species, 1});
    migration.step(Real{0.25} * migration.cfl_limit());
    const auto state = migration.gather_state();
    require(state.species.size() == 1 && state.species[0].particles.id.size() == 2,
            "PIC migration lost a species or particle");
    require(state.species[0].particles.id[0] == 41 &&
                state.species[0].particles.id[1] == 42,
            "PIC migration did not preserve stable-ID ordering");
    require(state.species[0].particles.x[0] > internal_seam,
            "particle did not cross the internal x seam");
    require(state.species[0].particles.x[1] < Real{0.01},
            "particle did not wrap across the global periodic seam");
    require(migration.telemetry().migrated_particles == 2,
            "PIC migration telemetry did not count both transfers");
    require(migration.telemetry().transport_peer_bytes +
                migration.telemetry().transport_local_staged_bytes > 0,
            "same-rank PIC halos did not use a device transport path");
    require(migration.telemetry().transport_staged_mpi_bytes == 0 &&
                migration.telemetry().transport_direct_mpi_bytes == 0,
            "same-rank PIC halos unexpectedly used an MPI transport path");
    migration.close();
  } catch (...) {
    try {
      if (!migration.closed()) migration.close();
    } catch (...) {
    }
    throw;
  }
}

void test_quad(MpiRuntime& mpi) {
  constexpr std::size_t nx = 10;
  constexpr std::size_t ny = 10;
  auto topology = VirtualTopology::create(nx, ny, 4, {2, 2}, 2);
  PicTileRuntime runtime{mpi, make_mapping(4), std::move(topology),
                         make_config(nx, ny)};
  try {
    auto fields = make_fields(nx, ny);
    for (auto* component : {&fields.ex, &fields.ey, &fields.ez,
                            &fields.bx, &fields.by, &fields.bz}) {
      std::fill(component->begin(), component->end(), Real{0});
    }
    PicSpeciesState marker;
    marker.config = {"diagonal-marker", Real{0}, Real{1}, 1};
    marker.particles.x = {Real{0.4995}};
    marker.particles.y = {Real{0.4995}};
    marker.particles.x_prev = marker.particles.x;
    marker.particles.y_prev = marker.particles.y;
    marker.particles.vx = {Real{0.2}};
    marker.particles.vy = {Real{0.2}};
    marker.particles.vz = {Real{0}};
    marker.particles.vphi_deposit = {Real{0}};
    marker.particles.weight = {Real{1}};
    marker.particles.alive = {1};
    marker.particles.id = {0xD1A60AULL};
    runtime.seed(
        fields, nullptr, std::span<const PicSpeciesState>{&marker, 1});
    runtime.step(Real{0.25} * runtime.cfl_limit());
    const auto state = runtime.gather_state();
    require(state.species.size() == 1 &&
                state.species.front().particles.id.size() == 1 &&
                state.species.front().particles.id.front() == 0xD1A60AULL,
            "four-GPU diagonal migration lost the stable particle ID");
    const Real x = state.species.front().particles.x.front();
    const Real y = state.species.front().particles.y.front();
    require(x > Real{0.5} && y > Real{0.5},
            "particle did not cross both 2x2 tile seams");
    const std::size_t ix = static_cast<std::size_t>(
        std::floor(x / (Real{1} / static_cast<Real>(nx))));
    const std::size_t iy = static_cast<std::size_t>(
        std::floor(y / (Real{1} / static_cast<Real>(ny))));
    const auto owner = runtime.topology().owner_of_cell(ix, iy);
    const auto coordinate = runtime.topology().tile(owner).coordinate;
    require(coordinate.x == 1 && coordinate.y == 1,
            "particle did not land on the diagonal GPU endpoint");
    require(runtime.telemetry().migrated_particles == 1,
            "four-GPU diagonal transfer was not counted exactly once");
    runtime.close();
  } catch (...) {
    try {
      if (!runtime.closed()) runtime.close();
    } catch (...) {
    }
    throw;
  }
}

void test_local_shards_multirank(MpiRuntime& mpi) {
  constexpr std::size_t nx = 9;
  constexpr std::size_t ny = 7;
  constexpr std::size_t devices_per_rank = 2;
  constexpr std::size_t endpoint_count = 4;
  auto topology =
      VirtualTopology::create(nx, ny, endpoint_count, {2, 2}, 2);

  PicSpeciesState markers;
  markers.config = {"owned-markers", Real{0}, Real{2}, endpoint_count};
  for (std::size_t endpoint = 0; endpoint < endpoint_count; ++endpoint) {
    const auto& tile = topology.tile(endpoint);
    const Real x = static_cast<Real>(tile.x.begin) / static_cast<Real>(nx) +
                   Real{0.5} * static_cast<Real>(tile.x.size()) /
                       static_cast<Real>(nx);
    const Real y = static_cast<Real>(tile.y.begin) / static_cast<Real>(ny) +
                   Real{0.5} * static_cast<Real>(tile.y.size()) /
                       static_cast<Real>(ny);
    markers.particles.x.push_back(x);
    markers.particles.y.push_back(y);
    markers.particles.x_prev.push_back(x - Real{0.001});
    markers.particles.y_prev.push_back(y + Real{0.001});
    markers.particles.vx.push_back(
        Real{0.01} * static_cast<Real>(endpoint + 1));
    markers.particles.vy.push_back(
        Real{-0.005} * static_cast<Real>(endpoint + 1));
    markers.particles.vz.push_back(
        Real{0.0025} * static_cast<Real>(endpoint + 1));
    markers.particles.vphi_deposit.push_back(
        markers.particles.vz.back());
    markers.particles.weight.push_back(Real{1} + Real{0.1} * endpoint);
    markers.particles.alive.push_back(1);
    markers.particles.id.push_back(1000 + endpoint);
  }

  PicTileRuntime runtime{
      mpi, make_multirank_mapping(mpi, devices_per_rank),
      std::move(topology), make_config(nx, ny)};
  try {
    runtime.seed(make_fields(nx, ny), nullptr,
                 std::span<const PicSpeciesState>{&markers, 1});

    const auto fields_only = runtime.local_owned_shards(false);
    require(fields_only.size() == devices_per_rank,
            "field-only extraction did not return two local shards");
    require(std::all_of(fields_only.begin(), fields_only.end(),
                        [](const PicOwnedShard& shard) {
                          return shard.species.empty();
                        }),
            "field-only extraction unexpectedly materialized particles");

    const auto shards = runtime.local_owned_shards(true);
    require(shards.size() == devices_per_rank,
            "particle extraction did not return two local shards");
    std::size_t local_particles = 0;
    for (const auto& shard : shards) {
      require(runtime.mapping().endpoint(shard.endpoint).world_rank ==
                  mpi.rank(),
              "rank-local extraction returned a remote endpoint");
      require(shard.endpoint >=
                  static_cast<std::size_t>(mpi.rank()) * devices_per_rank &&
                  shard.endpoint <
                      static_cast<std::size_t>(mpi.rank() + 1) *
                          devices_per_rank,
              "rank-local extraction returned the wrong endpoint range");
      require(shard.species.size() == 1,
              "rank-local extraction lost the species record");
      local_particles += shard.species.front().particles.id.size();
    }
    require(local_particles == devices_per_rank,
            "rank-local extraction received remote or lost local particles");
    require(runtime.telemetry().global_state_gathers == 0,
            "local shard extraction performed a global state gather");
    require(runtime.telemetry().local_shard_extractions == 2,
            "local shard extraction telemetry is incorrect");

    const auto counts = runtime.alive_counts();
    const auto energies = runtime.kinetic_energies();
    require(counts.size() == 1 && counts.front() == endpoint_count,
            "scalar alive-count reduction is incorrect");
    require(energies.size() == 1 && energies.front() > Real{0},
            "scalar kinetic-energy reduction is incorrect");
    require(runtime.telemetry().global_state_gathers == 0,
            "scalar diagnostics performed a global state gather");

    const auto gathered = runtime.gather_state();
    require(runtime.telemetry().global_state_gathers == 1,
            "explicit gathered diagnostics were not counted");
    const std::array<OwnedComponentDescriptor, 6> components{{
        {&PicOwnedFields::ex, &PicGlobalFields::ex, nx + 1, ny, "Ex"},
        {&PicOwnedFields::ey, &PicGlobalFields::ey, nx, ny + 1, "Ey"},
        {&PicOwnedFields::ez, &PicGlobalFields::ez, nx, ny, "Ez"},
        {&PicOwnedFields::bx, &PicGlobalFields::bx, nx, ny + 1, "Bx"},
        {&PicOwnedFields::by, &PicGlobalFields::by, nx + 1, ny, "By"},
        {&PicOwnedFields::bz, &PicGlobalFields::bz, nx + 1, ny + 1, "Bz"},
    }};
    for (const auto& component : components) {
      require_owned_component_reconstructs(
          mpi, shards, gathered.fields, component, false);
      require_owned_component_reconstructs(
          mpi, shards, gathered.external_fields, component, true);
    }

    require(gathered.species.size() == 1,
            "gathered state lost the marker species");
    const auto& global_particles = gathered.species.front().particles;
    const std::size_t particle_count = global_particles.id.size();
    require(particle_count == endpoint_count,
            "gathered state lost marker particles");
    constexpr std::size_t real_components = 9;
    std::vector<Real> particle_values(real_components * particle_count,
                                      Real{0});
    std::vector<std::uint64_t> particle_ids(particle_count, 0);
    std::vector<int> particle_alive(particle_count, 0);
    std::vector<int> particle_coverage(particle_count, 0);
    for (const auto& shard : shards) {
      const auto& local = shard.species.front().particles;
      for (std::size_t particle = 0; particle < local.id.size(); ++particle) {
        const auto found = std::lower_bound(
            global_particles.id.begin(), global_particles.id.end(),
            local.id[particle]);
        require(found != global_particles.id.end() &&
                    *found == local.id[particle],
                "local shard contains an unknown stable particle ID");
        const std::size_t index = static_cast<std::size_t>(
            std::distance(global_particles.id.begin(), found));
        require(particle_coverage[index] == 0,
                "particle appears in two rank-local shards");
        const std::array<Real, real_components> values{
            local.x[particle], local.y[particle], local.x_prev[particle],
            local.y_prev[particle], local.vx[particle], local.vy[particle],
            local.vz[particle], local.vphi_deposit[particle],
            local.weight[particle]};
        for (std::size_t component = 0; component < values.size();
             ++component) {
          particle_values[component * particle_count + index] =
              values[component];
        }
        particle_ids[index] = local.id[particle];
        particle_alive[index] = local.alive[particle];
        particle_coverage[index] = 1;
      }
    }
    require(MPI_Allreduce(MPI_IN_PLACE, particle_values.data(),
                          static_cast<int>(particle_values.size()),
                          MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS,
            "failed to reconstruct shard particle values");
    require(MPI_Allreduce(MPI_IN_PLACE, particle_ids.data(),
                          static_cast<int>(particle_ids.size()),
                          MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS,
            "failed to reconstruct shard particle IDs");
    require(MPI_Allreduce(MPI_IN_PLACE, particle_alive.data(),
                          static_cast<int>(particle_alive.size()), MPI_INT,
                          MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS,
            "failed to reconstruct shard particle liveness");
    require(MPI_Allreduce(MPI_IN_PLACE, particle_coverage.data(),
                          static_cast<int>(particle_coverage.size()), MPI_INT,
                          MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS,
            "failed to reconstruct shard particle ownership");
    require(std::all_of(particle_coverage.begin(), particle_coverage.end(),
                        [](int count) { return count == 1; }),
            "particle shards do not cover gathered particles exactly once");
    require(particle_ids == global_particles.id,
            "particle shard IDs differ from gathered diagnostics");
    for (std::size_t particle = 0; particle < particle_count; ++particle) {
      require(particle_alive[particle] == global_particles.alive[particle],
              "particle shard liveness differs from gathered diagnostics");
    }
    const std::array<const std::vector<Real>*, real_components> expected{{
        &global_particles.x, &global_particles.y, &global_particles.x_prev,
        &global_particles.y_prev, &global_particles.vx,
        &global_particles.vy, &global_particles.vz,
        &global_particles.vphi_deposit, &global_particles.weight}};
    for (std::size_t component = 0; component < expected.size(); ++component) {
      require_close(
          *expected[component],
          std::span<const Real>{
              particle_values.data() + component * particle_count,
              particle_count},
          Real{0}, "particle shard component");
    }
    runtime.close();
  } catch (...) {
    try {
      if (!runtime.closed()) runtime.close();
    } catch (...) {
    }
    throw;
  }
}

void test_multirank(MpiRuntime& mpi) {
  constexpr std::size_t nx = 9;
  constexpr std::size_t ny = 7;
  {
    bool rejected = false;
    std::string rejection_message;
    try {
      PicTileRuntime runtime{
          mpi, make_worker_failure_mapping(mpi),
          VirtualTopology::create(nx, ny, 2, {2, 1}, 2),
          make_order_four_config(nx, ny)};
      runtime.close();
    } catch (const DistributedCollectiveError& error) {
      rejected = error.resolution().representative.phase_text()
                     == "pic-worker-pool-construct"
          && error.resolution().representative.rank == 1;
      rejection_message = error.what();
    }
    require(mpi.allreduce_all(rejected),
            "rank-local PIC worker construction failure was not rejected "
            "collectively");
    require_same_text(mpi, rejection_message,
                      "PIC worker construction collective exception");
  }

  bool inconsistent_config_rejected = false;
  try {
    auto inconsistent = make_order_four_config(nx, ny);
    inconsistent.external_field_signature =
        mpi.rank() == 0 ? "shared-background-a" : "shared-background-b";
    auto rejected_topology = VirtualTopology::create(nx, ny, 2, {2, 1}, 2);
    PicTileRuntime rejected{
        mpi, make_multirank_mapping(mpi), std::move(rejected_topology),
        std::move(inconsistent)};
    rejected.close();
  } catch (const quasar::distributed::DistributedCollectiveError&) {
    inconsistent_config_rejected = true;
  }
  require(inconsistent_config_rejected,
          "rank-dependent PIC configuration was not rejected collectively");

#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
  {
    auto topology = VirtualTopology::create(nx, ny, 2, {2, 1}, 2);
    PicTileRuntime poisoned{
        mpi, make_multirank_mapping(mpi), std::move(topology),
        make_order_four_config(nx, ny),
        quasar::distributed::TransportPolicy::staged};
    poisoned.inject_seed_post_mutation_failure_for_testing(mpi.rank() == 1);
    bool rejected = false;
    std::string rejection_message;
    try {
      poisoned.seed(make_fields(nx, ny), nullptr, {});
    } catch (const DistributedCollectiveError& error) {
      rejected = error.resolution().representative.phase_text() ==
          "pic-seed-failure";
      rejection_message = error.what();
    }
    require(mpi.allreduce_all(rejected),
            "post-mutation PIC seed failure was not rejected collectively");
    require_same_text(mpi, rejection_message,
                      "post-mutation PIC seed collective exception");
    require(mpi.allreduce_all(poisoned.poisoned()),
            "post-mutation PIC seed failure did not poison every rank");
    require(!poisoned.seeded(),
            "failed PIC seed was marked committed");
    require_poisoned_rejection(
        [&] { poisoned.seed(make_fields(nx, ny), nullptr, {}); },
        "PIC reseed after post-mutation failure");
    require_poisoned_rejection(
        [&] { poisoned.step(Real{1e-6}); },
        "PIC step after post-mutation seed failure");
    require_poisoned_rejection(
        [&] { (void)poisoned.gather_state(); },
        "PIC diagnostic after post-mutation seed failure");
    poisoned.close();
    require(poisoned.closed(),
            "poisoned PIC seed runtime did not close collectively");
  }

  {
    auto topology = VirtualTopology::create(nx, ny, 2, {2, 1}, 2);
    PicTileRuntime runtime{
        mpi, make_multirank_mapping(mpi), std::move(topology),
        make_order_four_config(nx, ny),
        quasar::distributed::TransportPolicy::staged};
    runtime.inject_next_worker_task_allocation_failure_for_testing(
        mpi.rank() == 1);
    bool rejected = false;
    std::string rejection_message;
    try {
      runtime.close();
    } catch (const DistributedCollectiveError& error) {
      rejected = error.resolution().representative.phase_text() ==
                     "pic-worker-task-storage" &&
          error.resolution().representative.rank == 1;
      rejection_message = error.what();
    }
    require(mpi.allreduce_all(rejected),
            "rank-local PIC close task allocation failure was not rejected "
            "collectively");
    require_same_text(mpi, rejection_message,
                      "PIC close task allocation collective exception");
    require(!runtime.closed(),
            "failed PIC close task allocation marked the runtime closed");
    runtime.close();
    require(runtime.closed(),
            "PIC runtime did not close after one-shot allocation failure");
  }
#endif

  auto topology = VirtualTopology::create(nx, ny, 2, {2, 1}, 2);
  PicTileRuntime runtime{mpi, make_multirank_mapping(mpi), std::move(topology),
                         make_order_four_config(nx, ny),
                         quasar::distributed::TransportPolicy::automatic};
  try {
    const auto fields = make_fields(nx, ny);
    runtime.seed(fields, nullptr, {});
    require_close(fields, runtime.gather_state().fields, Real{0});
    const auto seed_telemetry = runtime.telemetry();
    if (runtime.transport_resolution().uses_direct_mpi()) {
      require(seed_telemetry.transport_direct_mpi_bytes > 0,
              "PIC auto transport resolved direct without device MPI bytes");
    } else {
      require(seed_telemetry.transport_staged_mpi_bytes > 0,
              "PIC staged fixed halos did not record staged MPI bytes");
      require(seed_telemetry.transport_direct_mpi_bytes == 0,
              "PIC staged transport unexpectedly recorded direct MPI bytes");
    }
    const Real limit = runtime.cfl_limit();
    bool inconsistent_rejected = false;
    try {
      runtime.step((mpi.rank() == 0 ? Real{0.1} : Real{0.2}) * limit);
    } catch (const quasar::distributed::DistributedCollectiveError&) {
      inconsistent_rejected = true;
    }
    require(inconsistent_rejected && !runtime.poisoned(),
            "rank-dependent timestep was not rejected before mutation");
    runtime.step(Real{0.1} * limit);
    const auto state = runtime.gather_state();
    require(state.step_count == 1,
            "multi-rank PIC runtime did not commit one shared step");
    runtime.close();
  } catch (...) {
    try {
      if (!runtime.closed()) runtime.close();
    } catch (...) {
    }
    throw;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 || (std::string_view{argv[1]} != "single" &&
                    std::string_view{argv[1]} != "multi" &&
                    std::string_view{argv[1]} != "quad" &&
                    std::string_view{argv[1]} != "multirank" &&
                    std::string_view{argv[1]} != "local-shards")) {
    std::cerr << "usage: test_pic_runtime "
                 "single|multi|quad|multirank|local-shards\n";
    return 2;
  }
  std::unique_ptr<MpiRuntime> mpi;
  try {
    mpi = std::make_unique<MpiRuntime>(&argc, &argv);
    const bool local_shards =
        std::string_view{argv[1]} == "local-shards";
    const bool multirank =
        std::string_view{argv[1]} == "multirank" || local_shards;
    if ((multirank && mpi->size() != 2) || (!multirank && mpi->size() != 1)) {
      mpi->close();
      return kSkip;
    }
    if (multirank) {
      const int required = local_shards ? 4 : 2;
      if (quasar::backend::device_count() < required) {
        mpi->close();
        return kSkip;
      }
      if (local_shards) {
        test_local_shards_multirank(*mpi);
      } else {
        test_multirank(*mpi);
      }
      mpi->close();
      return 0;
    }
    const bool multi = std::string_view{argv[1]} == "multi";
    const bool quad = std::string_view{argv[1]} == "quad";
    const int required = quad ? 4 : multi ? 2 : 1;
    if (quasar::backend::device_count() < required) {
      mpi->close();
      return kSkip;
    }
    if (quad) {
      test_quad(*mpi);
    } else if (multi) {
      test_multi(*mpi);
    } else {
      test_single(*mpi);
    }
    mpi->close();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test_pic_runtime: " << error.what() << '\n';
    if (mpi && !mpi->closed()) {
      try {
        mpi->close();
      } catch (...) {
      }
    }
    return 1;
  }
}
