#include "quasar/distributed/mhd_runtime.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/distributed/mhd_halo.hpp"

#include "collective_helpers.hpp"
#include "mhd_tile_access.hpp"
#include "mpi_runtime_native.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace quasar::distributed {
namespace {

bool periodic_x(const mhd::MhdConfig& config) {
  return config.boundary.field[0] == "periodic" &&
         config.boundary.field[1] == "periodic";
}

bool periodic_y(const mhd::MhdConfig& config) {
  return config.boundary.field[2] == "periodic" &&
         config.boundary.field[3] == "periodic";
}

std::size_t mhd_required_tile_width(std::string_view reconstruction) {
  if (reconstruction == "mp7") return 4;
  if (reconstruction == "mp5") return 3;
  return 2;
}

bool map_cell(long long coordinate, std::size_t extent, bool periodic,
              std::size_t& mapped) {
  if (coordinate >= 0 &&
      static_cast<unsigned long long>(coordinate) < extent) {
    mapped = static_cast<std::size_t>(coordinate);
    return true;
  }
  if (!periodic || extent == 0) return false;
  const long long period = static_cast<long long>(extent);
  long long value = coordinate % period;
  if (value < 0) value += period;
  mapped = static_cast<std::size_t>(value);
  return true;
}

bool map_face(long long coordinate, std::size_t cells, bool periodic,
              std::size_t& mapped) {
  if (coordinate >= 0 &&
      static_cast<unsigned long long>(coordinate) <= cells) {
    mapped = static_cast<std::size_t>(coordinate);
    return true;
  }
  if (!periodic || cells == 0) return false;
  const long long period = static_cast<long long>(cells);
  long long value = coordinate % period;
  if (value < 0) value += period;
  mapped = static_cast<std::size_t>(value);
  return true;
}

int background_boundary_mode(std::string_view name) {
  auto& registry = Registry<boundary::IMhdFieldBoundary>::instance();
  if (!registry.contains(name)) {
    throw std::invalid_argument{
        "unknown MHD background field boundary kind '" +
        std::string{name} + "'"};
  }
  const auto selected = registry.create(name);
  if (selected->is_internal_cut()) return 4;
  const int mode = selected->ghost_continuation_mode();
  if (mode < 0 || mode > 3) {
    throw std::invalid_argument{
        "MHD background field boundary does not provide a supported ghost "
        "continuation mode"};
  }
  return mode;
}

void fill_background_ghosts(
    const Grid2D& grid, const boundary::MhdBoundarySpec& boundaries,
    std::vector<Real>& bx, std::vector<Real>& by, std::vector<Real>& bz) {
  const auto ghost_indices = [](int n, int layer, bool low, int mode,
                                int& ghost, int& source) {
    ghost = low ? -layer : n - 1 + layer;
    if (mode == 0) {
      source = low ? n - layer : layer - 1;
    } else if (mode == 1) {
      source = low ? 0 : n - 1;
    } else {
      source = low ? layer - 1 : n - layer;
    }
  };
  const auto at = [&grid](std::vector<Real>& values, int i, int j) -> Real& {
    return values[grid.index(i, j)];
  };

  // Match MhdSolver2D::fill_ghosts ordering. Both passes span their complete
  // padded transverse range so a physical/internal tile corner is closed from
  // the already-exchanged neighbour halo.
  for (int side = 0; side < 4; ++side) {
    const int mode = background_boundary_mode(boundaries.field[side]);
    if (mode == 4) continue;
    const bool x_side = side < 2;
    const bool low = side == 0 || side == 2;
    const int line_low = -grid.nghost;
    const int line_high =
        (x_side ? grid.ny : grid.nx) + grid.nghost;
    const int extent = x_side ? grid.nx : grid.ny;
    for (int layer = 1; layer <= grid.nghost; ++layer) {
      int ghost = 0;
      int source = 0;
      ghost_indices(extent, layer, low, mode, ghost, source);
      for (int line = line_low; line < line_high; ++line) {
        if (x_side) {
          at(by, ghost, line) = at(by, source, line);
          at(bz, ghost, line) =
              (mode == 3 ? Real{-1} : Real{1}) * at(bz, source, line);
          if (mode == 2 || mode == 3) {
            if (low) {
              at(bx, 0, line) = Real{0};
              at(bx, -layer, line) = -at(bx, layer, line);
            } else if (layer == 1) {
              at(bx, grid.nx, line) = Real{0};
            } else {
              const int offset = layer - 1;
              at(bx, grid.nx + offset, line) =
                  -at(bx, grid.nx - offset, line);
            }
          } else if (mode == 1 && !low) {
            if (layer > 1) {
              at(bx, grid.nx - 1 + layer, line) =
                  at(bx, grid.nx, line);
            }
          } else {
            at(bx, ghost, line) = at(bx, source, line);
          }
        } else {
          at(bx, line, ghost) = at(bx, line, source);
          at(bz, line, ghost) = at(bz, line, source);
          if (mode == 2) {
            if (low) {
              at(by, line, 0) = Real{0};
              at(by, line, -layer) = -at(by, line, layer);
            } else if (layer == 1) {
              at(by, line, grid.ny) = Real{0};
            } else {
              const int offset = layer - 1;
              at(by, line, grid.ny + offset) =
                  -at(by, line, grid.ny - offset);
            }
          } else if (mode == 1 && !low) {
            if (layer > 1) {
              at(by, line, grid.ny - 1 + layer) =
                  at(by, line, grid.ny);
            }
          } else {
            at(by, line, ghost) = at(by, line, source);
          }
        }
      }
    }
  }
}

#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
bool inject_next_mhd_worker_task_allocation_failure = false;
bool inject_mhd_seed_post_mutation_failure = false;
bool inject_mhd_checkpoint_metadata_copy_failure = false;
#endif

void prepare_mhd_worker_task_storage() {
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
  if (std::exchange(inject_next_mhd_worker_task_allocation_failure, false)) {
    throw std::bad_alloc{};
  }
#endif
}

constexpr std::string_view mhd_worker_task_storage_phase =
    "mhd-worker-task-storage";
constexpr std::string_view mhd_worker_task_storage_failure =
    "distributed MHD worker task allocation failed";

template <class Function>
std::vector<WorkerTask> mhd_worker_tasks(
    MpiRuntime& runtime, std::uint64_t& epoch, std::size_t count,
    Function&& function) {
  return collectively_indexed_tasks(
      runtime, epoch, count, mhd_worker_task_storage_phase,
      mhd_worker_task_storage_failure, prepare_mhd_worker_task_storage,
      std::forward<Function>(function));
}

// Every device halo family has the same ordered contract: producers enqueue
// pack kernels, Transport completes the matching sends/receives, then consumers
// enqueue unpack kernels and publish them to the default stream.  Keep that
// sequencing in one place while callers provide only their typed lane views.
// The phases cannot be pipelined with the current single send/receive buffer per
// direction: the transfer consumes the packed bytes, the unpack consumes the
// received bytes, and the next axis may consume guards written by this one.
template <class PackFunction, class TransferFunction, class UnpackFunction>
std::uint64_t run_device_halo_exchange_phase(
    MpiRuntime& runtime, EndpointWorkerPool& workers, std::uint64_t& epoch,
    std::size_t local_count, std::string_view phase,
    PackFunction&& pack, TransferFunction&& transfer,
    UnpackFunction&& unpack) {
  auto pack_tasks = mhd_worker_tasks(
      runtime, epoch, local_count, std::forward<PackFunction>(pack));
  require_worker_success(workers, runtime, epoch, phase, pack_tasks);

  const std::uint64_t batches =
      std::forward<TransferFunction>(transfer)();

  auto unpack_tasks = mhd_worker_tasks(
      runtime, epoch, local_count, std::forward<UnpackFunction>(unpack));
  require_worker_success(workers, runtime, epoch, phase, unpack_tasks);
  return batches;
}

enum class MhdCheckpointLattice {
  cell,
  bx_face,
  by_face,
};

enum class MhdCheckpointField {
  state,
  background,
};

struct MhdCheckpointDataset {
  std::string_view name;
  MhdCheckpointLattice lattice;
  MhdCheckpointField field;
  std::size_t component;
};

constexpr std::array<MhdCheckpointDataset, 11> mhd_checkpoint_datasets{{
    {"mhd/state/rho", MhdCheckpointLattice::cell,
     MhdCheckpointField::state, 0},
    {"mhd/state/mx", MhdCheckpointLattice::cell,
     MhdCheckpointField::state, 1},
    {"mhd/state/my", MhdCheckpointLattice::cell,
     MhdCheckpointField::state, 2},
    {"mhd/state/mz", MhdCheckpointLattice::cell,
     MhdCheckpointField::state, 3},
    {"mhd/state/energy", MhdCheckpointLattice::cell,
     MhdCheckpointField::state, 4},
    {"mhd/state/bx_face", MhdCheckpointLattice::bx_face,
     MhdCheckpointField::state, 5},
    {"mhd/state/by_face", MhdCheckpointLattice::by_face,
     MhdCheckpointField::state, 6},
    {"mhd/state/bz_cell", MhdCheckpointLattice::cell,
     MhdCheckpointField::state, 7},
    {"mhd/background/b0x_face", MhdCheckpointLattice::bx_face,
     MhdCheckpointField::background, 0},
    {"mhd/background/b0y_face", MhdCheckpointLattice::by_face,
     MhdCheckpointField::background, 1},
    {"mhd/background/b0z_cell", MhdCheckpointLattice::cell,
     MhdCheckpointField::background, 2},
}};

static_assert(std::is_same_v<Real, double>,
              "distributed MHD collectives currently require double Real");

CheckpointValueType checkpoint_real_type() noexcept {
  return CheckpointValueType::float64;
}

void append_signature_text(std::ostringstream& output,
                           std::string_view key,
                           std::string_view value) {
  output << key << '=' << value.size() << ':' << value << ';';
}

void append_signature_integer(std::ostringstream& output,
                              std::string_view key,
                              std::uint64_t value) {
  output << key << '=' << value << ';';
}

void append_signature_real(std::ostringstream& output,
                           std::string_view key, Real value) {
  const auto bits = std::bit_cast<std::uint64_t>(value);
  output << key << "=0x" << std::hex << bits << std::dec << ';';
}

std::string mhd_boundary_signature(const mhd::MhdConfig& config) {
  std::ostringstream output;
  for (std::size_t side = 0; side < config.boundary.fluid.size(); ++side) {
    append_signature_text(output, "fluid" + std::to_string(side),
                          config.boundary.fluid[side]);
  }
  for (std::size_t side = 0; side < config.boundary.field.size(); ++side) {
    append_signature_text(output, "field" + std::to_string(side),
                          config.boundary.field[side]);
  }
  return output.str();
}

std::string mhd_background_content_signature(
    const MhdGlobalBackground& background) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_scalar(hash, background.global_nx);
  hash_scalar(hash, background.global_ny);
  hash_span(hash, std::span<const Real>{background.b0x_face});
  hash_span(hash, std::span<const Real>{background.b0y_face});
  hash_span(hash, std::span<const Real>{background.b0z_cell});
  std::ostringstream output;
  output << "fnv1a64:0x" << std::hex << hash;
  return output.str();
}

std::string mhd_background_signature(
    const mhd::MhdConfig& config,
    std::string_view content_signature = {}) {
  std::ostringstream output;
  append_signature_integer(output, "enabled",
                           config.background.enabled ? 1 : 0);
  if (!config.background.enabled) return output.str();

  append_signature_text(output, "profile", config.background.profile);
  append_signature_real(output, "bx0", config.background.bx0);
  append_signature_real(output, "by0", config.background.by0);
  append_signature_real(output, "bz0", config.background.bz0);
  append_signature_real(output, "profile_scale",
                        config.background.profile_scale);
  append_signature_integer(output, "curl_free",
                           config.background.curl_free ? 1 : 0);

  std::vector<std::pair<std::string, Real>> parameters;
  parameters.reserve(config.background.params.size());
  for (const auto& [name, value] : config.background.params) {
    parameters.emplace_back(name, value);
  }
  std::sort(parameters.begin(), parameters.end(),
            [](const auto& left, const auto& right) {
              return left.first < right.first;
            });
  append_signature_integer(output, "parameter_count", parameters.size());
  for (std::size_t index = 0; index < parameters.size(); ++index) {
    append_signature_text(output, "parameter_name" + std::to_string(index),
                          parameters[index].first);
    append_signature_real(output, "parameter_value" + std::to_string(index),
                          parameters[index].second);
  }
  if (!content_signature.empty()) {
    append_signature_text(output, "canonical_content", content_signature);
  }
  return output.str();
}

std::string mhd_numerics_signature(const mhd::MhdConfig& config) {
  std::ostringstream output;
  append_signature_real(output, "lx", config.grid.lx);
  append_signature_real(output, "ly", config.grid.ly);
  append_signature_real(output, "origin_x", config.grid.origin_x);
  append_signature_real(output, "origin_y", config.grid.origin_y);
  append_signature_real(output, "gamma", config.gamma);
  append_signature_text(output, "reconstruction", config.reconstruction);
  append_signature_text(output, "riemann", config.riemann);
  append_signature_text(output, "integrator", config.integrator);
  append_signature_text(output, "ct", config.ct);
  append_signature_text(output, "positivity", config.positivity);
  append_signature_real(output, "rho_floor", config.rho_floor);
  append_signature_real(output, "p_floor", config.p_floor);
  append_signature_real(output, "cfl", config.cfl);
  append_signature_text(output, "timestep", config.timestep_signature);
  return output.str();
}

std::array<std::uint64_t, 2> checkpoint_shape(
    const VirtualTopology& topology, MhdCheckpointLattice lattice) {
  std::uint64_t nx = static_cast<std::uint64_t>(topology.global_nx());
  std::uint64_t ny = static_cast<std::uint64_t>(topology.global_ny());
  if (lattice == MhdCheckpointLattice::bx_face) ++nx;
  if (lattice == MhdCheckpointLattice::by_face) ++ny;
  return {ny, nx};
}

std::vector<DatasetHyperslab> checkpoint_slabs(
    const EndpointMapping& mapping, const VirtualTopology& topology,
    int rank, MhdCheckpointLattice lattice) {
  std::vector<DatasetHyperslab> slabs;
  const auto endpoints = mapping.endpoints_for_rank(rank);
  slabs.reserve(endpoints.size());
  for (const auto& endpoint : endpoints) {
    const TileExtent& tile = topology.tile(endpoint.index);
    std::size_t x_begin = tile.x.begin;
    std::size_t y_begin = tile.y.begin;
    std::size_t x_count = tile.x.size();
    std::size_t y_count = tile.y.size();
    if (lattice == MhdCheckpointLattice::bx_face) {
      const std::size_t skip = tile.coordinate.x == 0 ? 0 : 1;
      x_begin += skip;
      x_count += 1 - skip;
    } else if (lattice == MhdCheckpointLattice::by_face) {
      const std::size_t skip = tile.coordinate.y == 0 ? 0 : 1;
      y_begin += skip;
      y_count += 1 - skip;
    }
    DatasetHyperslab slab{
        .offset = {static_cast<std::uint64_t>(y_begin),
                   static_cast<std::uint64_t>(x_begin)},
        .count = {static_cast<std::uint64_t>(y_count),
                  static_cast<std::uint64_t>(x_count)},
    };
    // HDF5 enumerates the union of selected rectangles in global row-major
    // order.  Coalesce a rank's horizontally adjacent endpoint rectangles so
    // its one-dimensional packed memory buffer has that same order (rather
    // than slab-major rows from alternating rectangles).
    if (!slabs.empty() && slabs.back().offset[0] == slab.offset[0] &&
        slabs.back().count[0] == slab.count[0] &&
        slabs.back().offset[1] + slabs.back().count[1] == slab.offset[1]) {
      slabs.back().count[1] += slab.count[1];
    } else {
      slabs.push_back(std::move(slab));
    }
  }
  return slabs;
}

std::size_t slab_elements(const DatasetHyperslab& slab) {
  if (slab.count.size() != 2) {
    throw std::logic_error{"MHD checkpoint slab must be two-dimensional"};
  }
  const auto rows = static_cast<std::size_t>(slab.count[0]);
  const auto columns = static_cast<std::size_t>(slab.count[1]);
  return checked_product(rows, columns, "MHD checkpoint hyperslab");
}

}  // namespace

struct MhdTileRuntime::DenseField {
  MhdGlobalState state{};
};

struct MhdTileRuntime::RegisterHaloBuffers {
  std::array<backend::DeviceBuffer<Real>, 4> send{};
  std::array<backend::DeviceBuffer<Real>, 4> receive{};
  backend::stream_t communication_stream{nullptr};
};

struct MhdTileRuntime::StepController {
  Real remaining;
  Real trial;
  Real global_anchor;
  bool low_order_interval{false};
  int retries{0};
  int attempts{0};
};

namespace {

constexpr int maximum_mhd_step_retries = 80;
constexpr int maximum_mhd_step_attempts = 100000;

using HostMhdRegister =
    std::array<std::vector<Real>, mhd_register_component_count>;

void communication_waits_for_default_stream(WorkerContext& context) {
  context.compute_ready.record(nullptr);
  backend::stream_wait_event(context.communication_stream.get(),
                             context.compute_ready.get());
}

void default_stream_waits_for_communication(WorkerContext& context) {
  context.communication_ready.record(context.communication_stream);
  backend::stream_wait_event(nullptr, context.communication_ready.get());
}

struct LocalMhdCheckpointTile {
  TileExtent tile{};
  Grid2D grid{};
  HostMhdRegister state{};
  std::array<std::vector<Real>, 3> background{};
};

template <class Field>
HostMhdRegister download_mhd_register(Field& field) {
  HostMhdRegister host;
  const std::size_t size = field.grid.storage_size();
  for (auto& component : host) component.resize(size);
  field.rho.copy_to_host(host[0].data(), size);
  field.mx.copy_to_host(host[1].data(), size);
  field.my.copy_to_host(host[2].data(), size);
  field.mz.copy_to_host(host[3].data(), size);
  field.energy.copy_to_host(host[4].data(), size);
  field.bx_face.copy_to_host(host[5].data(), size);
  field.by_face.copy_to_host(host[6].data(), size);
  field.bz_cell.copy_to_host(host[7].data(), size);
  return host;
}

bool has_neighbor_on_axis(
    const VirtualTopology& topology, std::size_t endpoint,
    const std::array<Direction, 2>& directions,
    bool periodic_x_value, bool periodic_y_value) {
  return std::any_of(
      directions.begin(), directions.end(), [&](Direction direction) {
        const auto neighbor = topology.neighbor(
            endpoint, direction, periodic_x_value, periodic_y_value);
        return neighbor.has_value() && *neighbor != endpoint;
      });
}

using DeviceHaloConstComponents = mhd::MhdDeviceHaloConstComponents;
using DeviceHaloComponents = mhd::MhdDeviceHaloComponents;
using DeviceHaloValueKind = mhd::MhdDeviceHaloValueKind;

mhd::MhdDeviceHaloDirection device_halo_direction(Direction direction) {
  switch (direction) {
    case Direction::x_low: return mhd::MhdDeviceHaloDirection::x_low;
    case Direction::x_high: return mhd::MhdDeviceHaloDirection::x_high;
    case Direction::y_low: return mhd::MhdDeviceHaloDirection::y_low;
    case Direction::y_high: return mhd::MhdDeviceHaloDirection::y_high;
  }
  throw std::invalid_argument{"distributed MHD halo direction is invalid"};
}

mhd::MhdDeviceHaloLayout device_halo_layout(MhdHaloStagger layout) {
  switch (layout) {
    case MhdHaloStagger::cell:
      return mhd::MhdDeviceHaloLayout::cell;
    case MhdHaloStagger::cell_extended_y:
      return mhd::MhdDeviceHaloLayout::cell_extended_y;
    case MhdHaloStagger::x_face:
      return mhd::MhdDeviceHaloLayout::x_face;
    case MhdHaloStagger::x_face_extended_y:
      return mhd::MhdDeviceHaloLayout::x_face_extended_y;
    case MhdHaloStagger::y_face:
      return mhd::MhdDeviceHaloLayout::y_face;
    case MhdHaloStagger::node:
      return mhd::MhdDeviceHaloLayout::node;
  }
  throw std::invalid_argument{"distributed MHD halo layout is invalid"};
}

template <class T>
void set_halo_source(DeviceHaloConstComponents& components,
                     std::size_t component,
                     const backend::DeviceBuffer<T>& values,
                     DeviceHaloValueKind kind) {
  components.component[component] = {values.device_ptr(), kind};
}

template <class T>
void set_halo_destination(DeviceHaloComponents& components,
                          std::size_t component,
                          backend::DeviceBuffer<T>& values,
                          DeviceHaloValueKind kind,
                          MhdHaloStagger layout) {
  components.component[component] = {
      values.device_ptr(), kind, device_halo_layout(layout)};
}

template <class Field>
DeviceHaloConstComponents field_halo_sources(const Field& field) {
  DeviceHaloConstComponents result{};
  set_halo_source(result, 0, field.rho, DeviceHaloValueKind::real);
  set_halo_source(result, 1, field.mx, DeviceHaloValueKind::real);
  set_halo_source(result, 2, field.my, DeviceHaloValueKind::real);
  set_halo_source(result, 3, field.mz, DeviceHaloValueKind::real);
  set_halo_source(result, 4, field.energy, DeviceHaloValueKind::real);
  set_halo_source(result, 5, field.bx_face, DeviceHaloValueKind::real);
  set_halo_source(result, 6, field.by_face, DeviceHaloValueKind::real);
  set_halo_source(result, 7, field.bz_cell, DeviceHaloValueKind::real);
  return result;
}

template <class Field>
DeviceHaloComponents field_halo_destinations(
    Field& field, const MhdHaloLayouts& layouts) {
  DeviceHaloComponents result{};
  set_halo_destination(
      result, 0, field.rho, DeviceHaloValueKind::real, layouts[0]);
  set_halo_destination(
      result, 1, field.mx, DeviceHaloValueKind::real, layouts[1]);
  set_halo_destination(
      result, 2, field.my, DeviceHaloValueKind::real, layouts[2]);
  set_halo_destination(
      result, 3, field.mz, DeviceHaloValueKind::real, layouts[3]);
  set_halo_destination(
      result, 4, field.energy, DeviceHaloValueKind::real, layouts[4]);
  set_halo_destination(
      result, 5, field.bx_face, DeviceHaloValueKind::real, layouts[5]);
  set_halo_destination(
      result, 6, field.by_face, DeviceHaloValueKind::real, layouts[6]);
  set_halo_destination(
      result, 7, field.bz_cell, DeviceHaloValueKind::real, layouts[7]);
  return result;
}

DeviceHaloConstComponents background_halo_sources(
    const mhd::MhdBackgroundField<Real>& background) {
  DeviceHaloConstComponents result{};
  set_halo_source(
      result, 5, background.b0x_face, DeviceHaloValueKind::real);
  set_halo_source(
      result, 6, background.b0y_face, DeviceHaloValueKind::real);
  set_halo_source(
      result, 7, background.b0z_cell, DeviceHaloValueKind::real);
  return result;
}

DeviceHaloComponents background_halo_destinations(
    mhd::MhdBackgroundField<Real>& background) {
  const MhdHaloLayouts layouts = mhd_register_halo_layouts();
  DeviceHaloComponents result{};
  set_halo_destination(result, 5, background.b0x_face,
                       DeviceHaloValueKind::real, layouts[5]);
  set_halo_destination(result, 6, background.b0y_face,
                       DeviceHaloValueKind::real, layouts[6]);
  set_halo_destination(result, 7, background.b0z_cell,
                       DeviceHaloValueKind::real, layouts[7]);
  return result;
}

template <class Buffers>
void pack_neighbor_halos(
    const Grid2D& grid, std::size_t endpoint,
    const std::array<Direction, 2>& directions,
    const VirtualTopology& topology, bool periodic_x_value,
    bool periodic_y_value, const DeviceHaloConstComponents& components,
    Buffers& buffers, backend::stream_t stream) {
  for (const Direction direction : directions) {
    const auto neighbor = topology.neighbor(
        endpoint, direction, periodic_x_value, periodic_y_value);
    if (!neighbor || *neighbor == endpoint) continue;
    const std::size_t index = static_cast<std::size_t>(direction);
    auto& send = buffers.send[index];
    mhd::launch_mhd_device_halo_pack(
        grid, device_halo_direction(direction), components, send, stream);
  }
}

template <class Buffers>
void unpack_neighbor_halos(
    const Grid2D& grid, std::size_t endpoint,
    const std::array<Direction, 2>& directions,
    const VirtualTopology& topology, bool periodic_x_value,
    bool periodic_y_value, const DeviceHaloComponents& components,
    Buffers& buffers, backend::stream_t stream) {
  for (const Direction direction : directions) {
    const auto neighbor = topology.neighbor(
        endpoint, direction, periodic_x_value, periodic_y_value);
    if (!neighbor || *neighbor == endpoint) continue;
    const std::size_t index = static_cast<std::size_t>(direction);
    auto& receive = buffers.receive[index];
    const auto owner = topology.canonical_face_owner(
        endpoint, direction, periodic_x_value, periodic_y_value);
    mhd::launch_mhd_device_halo_unpack(
        grid, device_halo_direction(direction), receive, components,
        owner.has_value() && *owner != endpoint, stream);
  }
}

constexpr std::size_t mhd_face_record_pass_count = 4;

DeviceHaloConstComponents face_record_halo_sources(
    const mhd::MhdField2D<Real>& flux,
    const mhd::MhdMomentumFluxParts2D<Real>& parts, std::size_t pass) {
  if (pass >= mhd_face_record_pass_count) {
    throw std::out_of_range{"distributed MHD face-record pass is invalid"};
  }
  if (pass == 0) return field_halo_sources(flux);

  DeviceHaloConstComponents result{};
  if (pass == 1) {
    set_halo_source(result, 0, parts.wave_x, DeviceHaloValueKind::real);
    set_halo_source(result, 1, parts.wave_y, DeviceHaloValueKind::real);
    set_halo_source(result, 2, parts.wave_z, DeviceHaloValueKind::real);
    set_halo_source(result, 3, parts.cross_b_point[0].x,
                    DeviceHaloValueKind::real);
    set_halo_source(result, 4, parts.cross_b_point[0].y,
                    DeviceHaloValueKind::real);
    set_halo_source(result, 5, parts.cross_b_point[0].z,
                    DeviceHaloValueKind::real);
    set_halo_source(result, 6, parts.cross_b_point[1].x,
                    DeviceHaloValueKind::real);
    set_halo_source(result, 7, parts.cross_b_point[1].y,
                    DeviceHaloValueKind::real);
    return result;
  }
  if (pass == 2) {
    set_halo_source(result, 0, parts.cross_b_point[1].z,
                    DeviceHaloValueKind::real);
    set_halo_source(result, 1, parts.cross_b_point[2].x,
                    DeviceHaloValueKind::real);
    set_halo_source(result, 2, parts.cross_b_point[2].y,
                    DeviceHaloValueKind::real);
    set_halo_source(result, 3, parts.cross_b_point[2].z,
                    DeviceHaloValueKind::real);
    set_halo_source(result, 4, parts.cross_b_point[3].x,
                    DeviceHaloValueKind::real);
    set_halo_source(result, 5, parts.cross_b_point[3].y,
                    DeviceHaloValueKind::real);
    set_halo_source(result, 6, parts.cross_b_point[3].z,
                    DeviceHaloValueKind::real);
    set_halo_source(result, 7, parts.b0_induction_covariance,
                    DeviceHaloValueKind::scaled_mantissa);
    return result;
  }
  set_halo_source(result, 0, parts.b0_induction_covariance,
                  DeviceHaloValueKind::scaled_exponent);
  set_halo_source(result, 1, parts.quadrature_valid,
                  DeviceHaloValueKind::int32);
  return result;
}

DeviceHaloComponents face_record_halo_destinations(
    mhd::MhdField2D<Real>& flux,
    mhd::MhdMomentumFluxParts2D<Real>& parts,
    std::size_t pass, const MhdHaloLayouts& layouts) {
  if (pass >= mhd_face_record_pass_count) {
    throw std::out_of_range{"distributed MHD face-record pass is invalid"};
  }
  if (pass == 0) return field_halo_destinations(flux, layouts);

  DeviceHaloComponents result{};
  if (pass == 1) {
    set_halo_destination(result, 0, parts.wave_x, DeviceHaloValueKind::real,
                         layouts[0]);
    set_halo_destination(result, 1, parts.wave_y, DeviceHaloValueKind::real,
                         layouts[1]);
    set_halo_destination(result, 2, parts.wave_z, DeviceHaloValueKind::real,
                         layouts[2]);
    set_halo_destination(result, 3, parts.cross_b_point[0].x,
                         DeviceHaloValueKind::real, layouts[3]);
    set_halo_destination(result, 4, parts.cross_b_point[0].y,
                         DeviceHaloValueKind::real, layouts[4]);
    set_halo_destination(result, 5, parts.cross_b_point[0].z,
                         DeviceHaloValueKind::real, layouts[5]);
    set_halo_destination(result, 6, parts.cross_b_point[1].x,
                         DeviceHaloValueKind::real, layouts[6]);
    set_halo_destination(result, 7, parts.cross_b_point[1].y,
                         DeviceHaloValueKind::real, layouts[7]);
    return result;
  }
  if (pass == 2) {
    set_halo_destination(result, 0, parts.cross_b_point[1].z,
                         DeviceHaloValueKind::real, layouts[0]);
    set_halo_destination(result, 1, parts.cross_b_point[2].x,
                         DeviceHaloValueKind::real, layouts[1]);
    set_halo_destination(result, 2, parts.cross_b_point[2].y,
                         DeviceHaloValueKind::real, layouts[2]);
    set_halo_destination(result, 3, parts.cross_b_point[2].z,
                         DeviceHaloValueKind::real, layouts[3]);
    set_halo_destination(result, 4, parts.cross_b_point[3].x,
                         DeviceHaloValueKind::real, layouts[4]);
    set_halo_destination(result, 5, parts.cross_b_point[3].y,
                         DeviceHaloValueKind::real, layouts[5]);
    set_halo_destination(result, 6, parts.cross_b_point[3].z,
                         DeviceHaloValueKind::real, layouts[6]);
    set_halo_destination(result, 7, parts.b0_induction_covariance,
                         DeviceHaloValueKind::scaled_mantissa, layouts[7]);
    return result;
  }
  set_halo_destination(result, 0, parts.b0_induction_covariance,
                       DeviceHaloValueKind::scaled_exponent, layouts[0]);
  set_halo_destination(result, 1, parts.quadrature_valid,
                       DeviceHaloValueKind::int32, layouts[1]);
  return result;
}

constexpr std::size_t direction_index(Direction direction) noexcept {
  return static_cast<std::size_t>(direction);
}

template <class Tile>
bool checkpoint_tile_owns(const Tile& local, MhdCheckpointLattice lattice,
                          std::size_t global_x, std::size_t global_y) {
  std::size_t x_begin = local.tile.x.begin;
  std::size_t y_begin = local.tile.y.begin;
  std::size_t x_count = local.tile.x.size();
  std::size_t y_count = local.tile.y.size();
  if (lattice == MhdCheckpointLattice::bx_face) {
    const std::size_t skip = local.tile.coordinate.x == 0 ? 0 : 1;
    x_begin += skip;
    x_count += 1 - skip;
  } else if (lattice == MhdCheckpointLattice::by_face) {
    const std::size_t skip = local.tile.coordinate.y == 0 ? 0 : 1;
    y_begin += skip;
    y_count += 1 - skip;
  }
  return global_x >= x_begin && global_x < x_begin + x_count
      && global_y >= y_begin && global_y < y_begin + y_count;
}

template <class Select>
std::vector<Real> pack_local_checkpoint_values(
    std::span<const LocalMhdCheckpointTile> local_tiles,
    MhdCheckpointLattice lattice,
    std::span<const DatasetHyperslab> slabs,
    Select&& select) {
  std::size_t total = 0;
  for (const auto& slab : slabs) total += slab_elements(slab);
  std::vector<Real> packed;
  packed.reserve(total);
  for (const auto& slab : slabs) {
    const std::size_t row_begin = static_cast<std::size_t>(slab.offset[0]);
    const std::size_t column_begin =
        static_cast<std::size_t>(slab.offset[1]);
    const std::size_t row_count = static_cast<std::size_t>(slab.count[0]);
    const std::size_t column_count =
        static_cast<std::size_t>(slab.count[1]);
    for (std::size_t row = 0; row < row_count; ++row) {
      for (std::size_t column = 0; column < column_count; ++column) {
        const std::size_t global_x = column_begin + column;
        const std::size_t global_y = row_begin + row;
        const auto owner = std::find_if(
            local_tiles.begin(), local_tiles.end(),
            [&](const LocalMhdCheckpointTile& candidate) {
              return checkpoint_tile_owns(
                  candidate, lattice, global_x, global_y);
            });
        if (owner == local_tiles.end()) {
          throw std::logic_error{
              "rank-local MHD checkpoint selection has no tile owner"};
        }
        const int i = static_cast<int>(global_x - owner->tile.x.begin);
        const int j = static_cast<int>(global_y - owner->tile.y.begin);
        packed.push_back(select(*owner)[owner->grid.index(i, j)]);
      }
    }
  }
  if (packed.size() != total) {
    throw std::logic_error{
        "rank-local MHD checkpoint packing produced the wrong size"};
  }
  return packed;
}

template <class Tile, class Select>
void unpack_local_checkpoint_values(
    std::span<Tile> local_tiles, MhdCheckpointLattice lattice,
    std::span<const DatasetHyperslab> slabs,
    std::span<const Real> packed,
    Select&& select) {
  std::size_t cursor = 0;
  for (const auto& slab : slabs) {
    const std::size_t row_begin = static_cast<std::size_t>(slab.offset[0]);
    const std::size_t column_begin =
        static_cast<std::size_t>(slab.offset[1]);
    const std::size_t row_count = static_cast<std::size_t>(slab.count[0]);
    const std::size_t column_count =
        static_cast<std::size_t>(slab.count[1]);
    for (std::size_t row = 0; row < row_count; ++row) {
      for (std::size_t column = 0; column < column_count; ++column) {
        if (cursor >= packed.size()) {
          throw std::logic_error{
              "rank-local MHD checkpoint input is truncated"};
        }
        const std::size_t global_x = column_begin + column;
        const std::size_t global_y = row_begin + row;
        const auto owner = std::find_if(
            local_tiles.begin(), local_tiles.end(),
            [&](const Tile& candidate) {
              return checkpoint_tile_owns(
                  candidate, lattice, global_x, global_y);
            });
        if (owner == local_tiles.end()) {
          throw std::logic_error{
              "rank-local MHD restart selection has no tile owner"};
        }
        const int i = static_cast<int>(global_x - owner->tile.x.begin);
        const int j = static_cast<int>(global_y - owner->tile.y.begin);
        select(*owner)[owner->grid.index(i, j)] = packed[cursor++];
      }
    }
  }
  if (cursor != packed.size()) {
    throw std::logic_error{
        "rank-local MHD checkpoint input has trailing values"};
  }
}

template <class Tile, class Select>
void read_local_checkpoint_lattice(
    MpiRuntime& runtime, std::uint64_t& epoch,
    const EndpointMapping& mapping, const VirtualTopology& topology,
    ParallelCheckpointReader& reader, std::span<Tile> local_tiles,
    std::string_view name, MhdCheckpointLattice lattice, Select&& select) {
  const auto shape = checkpoint_shape(topology, lattice);
  std::vector<DatasetHyperslab> slabs;
  std::vector<Real> packed;
  collective_try_with_fallback(
      runtime, epoch, "mhd-restart-dataset-storage",
      "MHD restart dataset allocation failed", -1, [&] {
    slabs = checkpoint_slabs(mapping, topology, runtime.rank(), lattice);
    std::size_t total = 0;
    for (const auto& slab : slabs) {
      const std::size_t count = slab_elements(slab);
      if (count > std::numeric_limits<std::size_t>::max() - total) {
        throw std::length_error{"MHD restart packed selection overflows"};
      }
      total += count;
    }
    packed.resize(total);
  });
  reader.read_dataset(name, checkpoint_real_type(), shape, slabs,
                      packed.data(), packed.size());

  collective_try_with_fallback(
      runtime, epoch, "mhd-restart-dataset-unpack",
      "MHD restart dataset unpack failed", -1, [&] {
    unpack_local_checkpoint_values(
        local_tiles, lattice, slabs, packed,
        std::forward<Select>(select));
  });
}

std::string checkpoint_background_content_signature(
    std::string_view signature) {
  constexpr std::string_view key = "canonical_content=";
  const std::size_t begin = signature.find(key);
  if (begin == std::string_view::npos) return {};
  const std::size_t length_begin = begin + key.size();
  const std::size_t colon = signature.find(':', length_begin);
  if (colon == std::string_view::npos) {
    throw std::runtime_error{
        "MHD checkpoint background content signature is malformed"};
  }
  std::size_t length = 0;
  const auto digits = signature.substr(length_begin, colon - length_begin);
  if (digits.empty()) {
    throw std::runtime_error{
        "MHD checkpoint background content signature is malformed"};
  }
  for (const char digit : digits) {
    if (digit < '0' || digit > '9'
        || length > (std::numeric_limits<std::size_t>::max()
                     - static_cast<std::size_t>(digit - '0')) / 10) {
      throw std::runtime_error{
          "MHD checkpoint background content signature is malformed"};
    }
    length = 10 * length + static_cast<std::size_t>(digit - '0');
  }
  const std::size_t value_begin = colon + 1;
  if (length > signature.size() - value_begin) {
    throw std::runtime_error{
        "MHD checkpoint background content signature is truncated"};
  }
  return std::string{signature.substr(value_begin, length)};
}

template <class Tile, class BxSelect, class BySelect>
void verify_local_periodic_duplicates(
    MpiRuntime& runtime, const mhd::MhdConfig& config,
    const VirtualTopology& topology,
    std::span<const Tile> local_tiles,
    BxSelect&& bx_select, BySelect&& by_select,
    std::string_view description) {
  if (periodic_x(config)) {
    std::vector<Real> low(topology.global_ny(), Real{0});
    std::vector<Real> high(topology.global_ny(), Real{0});
    std::vector<int> low_counts(topology.global_ny(), 0);
    std::vector<int> high_counts(topology.global_ny(), 0);
    for (const auto& local : local_tiles) {
      for (std::size_t j = 0; j < local.tile.y.size(); ++j) {
        const std::size_t global_y = local.tile.y.begin + j;
        if (local.tile.coordinate.x == 0) {
          low[global_y] = bx_select(local)[
              local.grid.index(0, static_cast<int>(j))];
          low_counts[global_y] = 1;
        }
        if (local.tile.coordinate.x + 1 == topology.shape().px) {
          high[global_y] = bx_select(local)[
              local.grid.index(local.grid.nx, static_cast<int>(j))];
          high_counts[global_y] = 1;
        }
      }
    }
    allreduce_sum_in_place(runtime, low, MPI_DOUBLE,
                           "MPI_Allreduce(MHD periodic Bx low duplicate)");
    allreduce_sum_in_place(runtime, high, MPI_DOUBLE,
                           "MPI_Allreduce(MHD periodic Bx high duplicate)");
    allreduce_sum_in_place(runtime, low_counts, MPI_INT,
                           "MPI_Allreduce(MHD periodic Bx low ownership)");
    allreduce_sum_in_place(runtime, high_counts, MPI_INT,
                           "MPI_Allreduce(MHD periodic Bx high ownership)");
    require_exact_coverage(low_counts, "periodic Bx low duplicate");
    require_exact_coverage(high_counts, "periodic Bx high duplicate");
    if (low != high) {
      throw std::runtime_error{
          std::string{description}
          + " has an inconsistent periodic Bx high-face duplicate"};
    }
  }
  if (periodic_y(config)) {
    std::vector<Real> low(topology.global_nx(), Real{0});
    std::vector<Real> high(topology.global_nx(), Real{0});
    std::vector<int> low_counts(topology.global_nx(), 0);
    std::vector<int> high_counts(topology.global_nx(), 0);
    for (const auto& local : local_tiles) {
      for (std::size_t i = 0; i < local.tile.x.size(); ++i) {
        const std::size_t global_x = local.tile.x.begin + i;
        if (local.tile.coordinate.y == 0) {
          low[global_x] = by_select(local)[
              local.grid.index(static_cast<int>(i), 0)];
          low_counts[global_x] = 1;
        }
        if (local.tile.coordinate.y + 1 == topology.shape().py) {
          high[global_x] = by_select(local)[
              local.grid.index(static_cast<int>(i), local.grid.ny)];
          high_counts[global_x] = 1;
        }
      }
    }
    allreduce_sum_in_place(runtime, low, MPI_DOUBLE,
                           "MPI_Allreduce(MHD periodic By low duplicate)");
    allreduce_sum_in_place(runtime, high, MPI_DOUBLE,
                           "MPI_Allreduce(MHD periodic By high duplicate)");
    allreduce_sum_in_place(runtime, low_counts, MPI_INT,
                           "MPI_Allreduce(MHD periodic By low ownership)");
    allreduce_sum_in_place(runtime, high_counts, MPI_INT,
                           "MPI_Allreduce(MHD periodic By high ownership)");
    require_exact_coverage(low_counts, "periodic By low duplicate");
    require_exact_coverage(high_counts, "periodic By high duplicate");
    if (low != high) {
      throw std::runtime_error{
          std::string{description}
          + " has an inconsistent periodic By high-face duplicate"};
    }
  }
}

}  // namespace

struct MhdTileRuntime::RestartPayload {
  struct Tile {
    TileExtent tile{};
    Grid2D grid{};
    HostMhdRegister state{};
    std::array<std::vector<Real>, 3> background{};
  };

  std::vector<Tile> local_tiles{};
  std::vector<std::vector<std::uint8_t>> diagnostic_state{};
  std::string background_content_signature{};
};

void validate_mhd_global_state(const MhdGlobalState& state) {
  if (state.global_nx == 0 || state.global_ny == 0) {
    throw std::invalid_argument{
        "distributed MHD state requires a non-empty global mesh"};
  }
  const std::size_t cells = checked_product(
      state.global_nx, state.global_ny, "MHD cell lattice");
  const std::size_t bx_size = checked_product(
      state.global_nx + 1, state.global_ny, "MHD Bx lattice");
  const std::size_t by_size = checked_product(
      state.global_nx, state.global_ny + 1, "MHD By lattice");
  const auto require_size = [](std::span<const Real> values,
                               std::size_t expected, const char* label) {
    if (values.size() != expected) {
      throw std::invalid_argument{
          std::string{label} + " has the wrong global lattice size"};
    }
    require_finite(values, label);
  };
  require_size(state.rho, cells, "rho");
  require_size(state.mx, cells, "mx");
  require_size(state.my, cells, "my");
  require_size(state.mz, cells, "mz");
  require_size(state.energy, cells, "energy");
  require_size(state.bx_face, bx_size, "bx_face");
  require_size(state.by_face, by_size, "by_face");
  require_size(state.bz_cell, cells, "bz_cell");
}

void validate_mhd_global_background(const MhdGlobalBackground& background) {
  if (background.global_nx == 0 || background.global_ny == 0) {
    throw std::invalid_argument{
        "distributed MHD background requires a non-empty global mesh"};
  }
  const std::size_t cells = checked_product(
      background.global_nx, background.global_ny, "MHD background cells");
  const std::size_t bx_size = checked_product(
      background.global_nx + 1, background.global_ny,
      "MHD background Bx lattice");
  const std::size_t by_size = checked_product(
      background.global_nx, background.global_ny + 1,
      "MHD background By lattice");
  if (background.b0x_face.size() != bx_size ||
      background.b0y_face.size() != by_size ||
      background.b0z_cell.size() != cells) {
    throw std::invalid_argument{
        "distributed MHD background has an incompatible lattice size"};
  }
  require_finite(std::span<const Real>{background.b0x_face}, "b0x_face");
  require_finite(std::span<const Real>{background.b0y_face}, "b0y_face");
  require_finite(std::span<const Real>{background.b0z_cell}, "b0z_cell");
}

MhdTileRuntime::MhdTileRuntime(MpiRuntime& runtime,
                               EndpointMapping mapping,
                               VirtualTopology topology,
                               mhd::MhdConfig global_config,
                               TransportPolicy transport_policy)
    : runtime_{&runtime},
      mapping_{std::move(mapping)},
      topology_{std::move(topology)},
      global_config_{std::move(global_config)} {
  runtime.require_orchestration_thread();
  collective_try(
      runtime, worker_epoch_, "mhd-runtime-config",
      "distributed MHD runtime configuration is invalid", -1, [&] {
    if (mapping_.empty() || mapping_.rank_count() !=
                                static_cast<std::size_t>(runtime.size())) {
      throw std::invalid_argument{
          "endpoint mapping does not match the MPI world"};
    }
    if (mapping_.size() != topology_.endpoint_count()) {
      throw std::invalid_argument{
          "endpoint mapping and virtual topology sizes differ"};
    }
    if (topology_.global_nx() !=
            static_cast<std::size_t>(global_config_.grid.nx) ||
        topology_.global_ny() !=
            static_cast<std::size_t>(global_config_.grid.ny)) {
      throw std::invalid_argument{
          "virtual topology and MHD configuration mesh differ"};
    }
    const std::size_t required_tile_width =
        mhd_required_tile_width(global_config_.reconstruction);
    for (const TileExtent& tile : topology_.tiles()) {
      if (tile.x.size() < required_tile_width ||
          tile.y.size() < required_tile_width) {
        throw std::invalid_argument{
            "MHD decomposition " +
            std::to_string(topology_.shape().px) + 'x' +
            std::to_string(topology_.shape().py) +
            " creates a tile thinner than the required " +
            std::to_string(required_tile_width) +
            "-cell stencil reach for reconstruction=" +
            global_config_.reconstruction};
      }
    }
    for (const auto& endpoint : mapping_.endpoints_for_rank(runtime.rank())) {
      local_devices_.push_back(endpoint.device.ordinal);
    }
  });

  // Every process constructs the shared runtime independently. Refuse to
  // create workers if ranks supplied different physics, mesh, topology, or
  // numerical configuration; otherwise the first later collective could
  // combine values from incompatible simulations.
  std::uint64_t config_hash = 1469598103934665603ULL;
  hash_scalar(config_hash, global_config_.grid.nx);
  hash_scalar(config_hash, global_config_.grid.ny);
  // Halo width is allowed to change across an unpadded checkpoint restart, but
  // every rank participating in one live run must still construct identical
  // tile storage.
  hash_scalar(config_hash, global_config_.grid.nghost);
  hash_string(config_hash, global_config_.geometry);
  hash_string(config_hash, mhd_boundary_signature(global_config_));
  hash_string(config_hash, mhd_background_signature(global_config_));
  hash_string(config_hash, mhd_numerics_signature(global_config_));
  const auto shape = topology_.shape();
  hash_scalar(config_hash, shape.px);
  hash_scalar(config_hash, shape.py);
  hash_scalar(config_hash, topology_.minimum_tile_width());
  hash_scalar(config_hash, transport_policy);
  std::uint64_t minimum_hash = 0;
  std::uint64_t maximum_hash = 0;
  check_mpi(MPI_Allreduce(&config_hash, &minimum_hash, 1, MPI_UINT64_T,
                          MPI_MIN,
                          detail::MpiRuntimeNativeAccess::world(runtime)),
            "MPI_Allreduce(distributed MHD config minimum hash)");
  check_mpi(MPI_Allreduce(&config_hash, &maximum_hash, 1, MPI_UINT64_T,
                          MPI_MAX,
                          detail::MpiRuntimeNativeAccess::world(runtime)),
            "MPI_Allreduce(distributed MHD config maximum hash)");
  collective_require(
      runtime, worker_epoch_, minimum_hash == maximum_hash,
      "mhd-runtime-agreement",
      "distributed MHD configuration or topology differs between MPI ranks");

  const std::size_t first_endpoint =
      static_cast<std::size_t>(runtime.rank()) * mapping_.devices_per_rank();
  const std::uint64_t worker_construction_epoch = worker_epoch_++;
  const CollectiveErrorRecord worker_construction = collective_try_record(
      runtime, worker_construction_epoch, "mhd-worker-pool-construct",
      "non-standard GPU worker-pool construction failure", -1, [&] {
    workers_ =
        std::make_unique<EndpointWorkerPool>(local_devices_, first_endpoint);
  });
  try {
    runtime.require_collective_success(worker_construction);
  } catch (...) {
    // Worker-pool teardown is rank-local.  Every rank reaches the agreement
    // above before a successful participant releases its workers, so no peer
    // can enter the first worker collective after another rank failed setup.
    if (workers_) {
      try {
        workers_->close();
      } catch (...) {
      }
      workers_.reset();
    }
    throw;
  }
  std::vector<WorkerTask> tasks;
  try {
    collective_try_with_fallback(
        runtime, worker_epoch_, "mhd-tile-storage",
        "distributed MHD tile storage allocation failed", -1, [&] {
    solvers_.resize(local_devices_.size());
    register_halos_.resize(local_devices_.size());
    tasks = local_indexed_tasks(solvers_.size(), [this](std::size_t local,
                                                  WorkerContext& context) {
      const std::size_t endpoint =
          static_cast<std::size_t>(runtime_->rank()) *
              mapping_.devices_per_rank() +
          local;
      solvers_[local] =
          std::make_unique<mhd::MhdSolver2D>(tile_config(endpoint));
      auto buffers = std::make_unique<RegisterHaloBuffers>();
      buffers->communication_stream = context.communication_stream.get();
      const Grid2D grid = solvers_[local]->grid();
      const bool px = periodic_x(global_config_);
      const bool py = periodic_y(global_config_);
      for (int side = 0; side < 4; ++side) {
        const Direction direction = static_cast<Direction>(side);
        const auto neighbor = topology_.neighbor(endpoint, direction, px, py);
        if (!neighbor || *neighbor == endpoint) continue;
        const std::size_t size =
            mhd_register_halo_scalar_count(grid, direction);
        buffers->send[static_cast<std::size_t>(side)] =
            backend::DeviceBuffer<Real>{size};
        buffers->receive[static_cast<std::size_t>(side)] =
            backend::DeviceBuffer<Real>{size};
      }
      register_halos_[local] = std::move(buffers);
    });
    });
  } catch (...) {
    workers_->close();
    workers_.reset();
    solvers_.clear();
    register_halos_.clear();
    throw;
  }
  const auto resolution = workers_->execute_collective(
      runtime, worker_epoch_++, "mhd-tile-construct", tasks);
  if (!resolution.accepted()) {
    auto cleanup = local_indexed_tasks(solvers_.size(), [this](std::size_t local,
                                                         WorkerContext&) {
      register_halos_[local].reset();
      solvers_[local].reset();
    });
    (void)workers_->execute(worker_epoch_++, runtime.rank(),
                            "mhd-tile-cleanup", cleanup);
    workers_->close();
    throw DistributedCollectiveError{resolution};
  }
  try {
    transport_ = std::make_unique<Transport>(runtime, transport_policy);
    telemetry_.transport = transport_->telemetry();
  } catch (...) {
    auto cleanup = local_indexed_tasks(solvers_.size(), [this](std::size_t local,
                                                         WorkerContext&) {
      register_halos_[local].reset();
      solvers_[local].reset();
    });
    (void)workers_->execute(worker_epoch_++, runtime.rank(),
                            "mhd-transport-cleanup", cleanup);
    workers_->close();
    throw;
  }
}

MhdTileRuntime::~MhdTileRuntime() noexcept = default;

const VirtualTopology& MhdTileRuntime::topology() const noexcept {
  return topology_;
}

const EndpointMapping& MhdTileRuntime::mapping() const noexcept {
  return mapping_;
}

bool MhdTileRuntime::closed() const noexcept { return lifecycle_.closed; }
bool MhdTileRuntime::seeded() const noexcept { return lifecycle_.seeded; }
bool MhdTileRuntime::poisoned() const noexcept { return lifecycle_.poisoned; }

void MhdTileRuntime::inject_next_worker_task_allocation_failure_for_testing(
    bool enabled) noexcept {
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
  inject_next_mhd_worker_task_allocation_failure = enabled;
#else
  (void)enabled;
#endif
}

void MhdTileRuntime::inject_seed_post_mutation_failure_for_testing(
    bool enabled) noexcept {
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
  inject_mhd_seed_post_mutation_failure = enabled;
#else
  (void)enabled;
#endif
}

void MhdTileRuntime::inject_checkpoint_metadata_copy_failure_for_testing(
    bool enabled) noexcept {
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
  inject_mhd_checkpoint_metadata_copy_failure = enabled;
#else
  (void)enabled;
#endif
}

void MhdTileRuntime::inject_restart_post_reconcile_failure_for_testing(
    bool enabled) noexcept {
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
  inject_restart_post_reconcile_failure_ = enabled;
#else
  (void)enabled;
#endif
}

const MhdRuntimeTelemetry& MhdTileRuntime::telemetry() const noexcept {
  return telemetry_;
}

const TransportResolution& MhdTileRuntime::transport_resolution() const noexcept {
  return transport_->resolution();
}

mhd::MhdConfig MhdTileRuntime::tile_config(std::size_t endpoint) const {
  mhd::MhdConfig config = global_config_;
  const TileExtent& tile = topology_.tile(endpoint);
  const Real dx = global_config_.grid.dx();
  const Real dy = global_config_.grid.dy();
  const int communication_halo = static_cast<int>(
      mhd_required_tile_width(global_config_.reconstruction));
  config.grid = Grid2D::from_cell_spacing(
      static_cast<int>(tile.x.size()), static_cast<int>(tile.y.size()),
      dx, dy,
      std::fma(static_cast<Real>(tile.x.begin), dx,
               global_config_.grid.origin_x),
      std::fma(static_cast<Real>(tile.y.begin), dy,
               global_config_.grid.origin_y),
      std::max(global_config_.grid.nghost, communication_halo));

  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);
  for (int side = 0; side < 4; ++side) {
    const Direction direction = static_cast<Direction>(side);
    const auto neighbor = topology_.neighbor(endpoint, direction, px, py);
    if (neighbor && *neighbor != endpoint) {
      config.boundary.fluid[side] = "internal";
      config.boundary.field[side] = "internal";
    }
  }
  return config;
}

void MhdTileRuntime::require_open_seeded() const {
  if (lifecycle_.closed) {
    throw std::logic_error{"distributed MHD runtime is closed"};
  }
  if (lifecycle_.poisoned) {
    throw std::logic_error{
        "distributed MHD runtime is poisoned after a failed collective mutation"};
  }
  if (!lifecycle_.seeded) {
    throw std::logic_error{"distributed MHD runtime is not seeded"};
  }
}

[[noreturn]] void MhdTileRuntime::poison_collectively(
    std::string_view phase, std::string_view local_message) {
  poison_runtime_collectively(*runtime_, worker_epoch_, lifecycle_.poisoned,
                               phase, local_message);
}

void MhdTileRuntime::seed(const MhdGlobalState& state,
                          const MhdGlobalBackground* background) {
  if (lifecycle_.closed) {
    throw std::logic_error{"distributed MHD runtime is closed"};
  }
  if (lifecycle_.poisoned) {
    throw std::logic_error{
        "distributed MHD runtime is poisoned after a failed collective mutation"};
  }
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "mhd-seed-validate",
      "distributed MHD seed validation failed", -1, [&] {
    validate_mhd_global_state(state);
    if (state.global_nx != topology_.global_nx() ||
        state.global_ny != topology_.global_ny()) {
      throw std::invalid_argument{
          "distributed MHD seed mesh does not match the topology"};
    }
    if (background != nullptr) {
      validate_mhd_global_background(*background);
      if (!global_config_.background.enabled) {
        throw std::invalid_argument{
            "an explicit MHD background requires background.enabled"};
      }
      if (background->global_nx != state.global_nx ||
          background->global_ny != state.global_ny) {
        throw std::invalid_argument{
            "MHD state and background meshes differ"};
      }
    }
  });

  // The canonical host seed is replicated by contract. Compare its exact
  // topology-independent representation before any worker mutates a tile, so
  // a rank-local deck/file error is rejected collectively and the runtime
  // remains unseeded and reusable.
  std::uint64_t seed_hash = 1469598103934665603ULL;
  hash_scalar(seed_hash, state.global_nx);
  hash_scalar(seed_hash, state.global_ny);
  hash_span(seed_hash, std::span<const Real>{state.rho});
  hash_span(seed_hash, std::span<const Real>{state.mx});
  hash_span(seed_hash, std::span<const Real>{state.my});
  hash_span(seed_hash, std::span<const Real>{state.mz});
  hash_span(seed_hash, std::span<const Real>{state.energy});
  hash_span(seed_hash, std::span<const Real>{state.bx_face});
  hash_span(seed_hash, std::span<const Real>{state.by_face});
  hash_span(seed_hash, std::span<const Real>{state.bz_cell});
  const bool has_background = background != nullptr;
  hash_scalar(seed_hash, has_background);
  if (background != nullptr) {
    hash_scalar(seed_hash, background->global_nx);
    hash_scalar(seed_hash, background->global_ny);
    hash_span(seed_hash, std::span<const Real>{background->b0x_face});
    hash_span(seed_hash, std::span<const Real>{background->b0y_face});
    hash_span(seed_hash, std::span<const Real>{background->b0z_cell});
  }
  std::uint64_t minimum_hash = 0;
  std::uint64_t maximum_hash = 0;
  check_mpi(MPI_Allreduce(&seed_hash, &minimum_hash, 1, MPI_UINT64_T, MPI_MIN,
                          detail::MpiRuntimeNativeAccess::world(*runtime_)),
            "MPI_Allreduce(distributed MHD seed minimum hash)");
  check_mpi(MPI_Allreduce(&seed_hash, &maximum_hash, 1, MPI_UINT64_T, MPI_MAX,
                          detail::MpiRuntimeNativeAccess::world(*runtime_)),
            "MPI_Allreduce(distributed MHD seed maximum hash)");
  collective_require(
      *runtime_, worker_epoch_, minimum_hash == maximum_hash,
      "mhd-seed-consistency",
      "canonical MHD state or background seed differs between MPI ranks");
  std::string prepared_background_signature;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "mhd-seed-signature",
      "distributed MHD seed signature allocation failed", -1, [&] {
    if (background != nullptr) {
      prepared_background_signature =
          mhd_background_content_signature(*background);
    }
  });

  bool mutation_started = false;
  try {
    seed_local_tiles(state, background, mutation_started);
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
    const bool injected =
        std::exchange(inject_mhd_seed_post_mutation_failure, false);
    collective_require(
        *runtime_, worker_epoch_, !injected, "mhd-seed-post-mutation",
        "injected post-mutation MHD seed failure");
#endif
    lifecycle_.seeded = true;
    reconcile_register(0, "mhd-seed-reconcile");
    background_content_signature_ =
        std::move(prepared_background_signature);
  } catch (const std::exception& error) {
    if (mutation_started) {
      poison_collectively("mhd-seed-failure", error.what());
    }
    throw;
  } catch (...) {
    if (mutation_started) {
      poison_collectively("mhd-seed-failure",
                          "non-standard MHD seed failure");
    }
    throw;
  }
}

void MhdTileRuntime::seed_local_tiles(
    const MhdGlobalState& state,
    const MhdGlobalBackground* background,
    bool& mutation_started) {
  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);
  auto tasks = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, &state, background, px, py](std::size_t local, WorkerContext&) {
    auto& solver = *solvers_[local];
    const std::size_t endpoint =
        static_cast<std::size_t>(runtime_->rank()) *
            mapping_.devices_per_rank() +
        local;
    const TileExtent& tile = topology_.tile(endpoint);
    const Grid2D grid = solver.grid();

    auto make_component = [&](std::span<const Real> global,
                              bool face_x, bool face_y) {
      std::vector<Real> result(grid.storage_size(), Real{0});
      for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
        for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
          std::size_t gx = 0;
          std::size_t gy = 0;
          const long long raw_x = static_cast<long long>(tile.x.begin) + i;
          const long long raw_y = static_cast<long long>(tile.y.begin) + j;
          const bool x_ok = face_x
              ? map_face(raw_x, state.global_nx, px, gx)
              : map_cell(raw_x, state.global_nx, px, gx);
          const bool y_ok = face_y
              ? map_face(raw_y, state.global_ny, py, gy)
              : map_cell(raw_y, state.global_ny, py, gy);
          if (!x_ok || !y_ok) continue;
          const std::size_t width = state.global_nx + (face_x ? 1 : 0);
          result[grid.index(i, j)] = global[row_major(gx, gy, width)];
        }
      }
      return result;
    };

    solver.seed_state("rho", make_component(state.rho, false, false));
    solver.seed_state("mx", make_component(state.mx, false, false));
    solver.seed_state("my", make_component(state.my, false, false));
    solver.seed_state("mz", make_component(state.mz, false, false));
    solver.seed_state("energy", make_component(state.energy, false, false));
    solver.seed_state("bx_face", make_component(state.bx_face, true, false));
    solver.seed_state("by_face", make_component(state.by_face, false, true));
    solver.seed_state("bz_cell", make_component(state.bz_cell, false, false));

    if (background != nullptr) {
      auto b0x = make_component(background->b0x_face, true, false);
      auto b0y = make_component(background->b0y_face, false, true);
      auto b0z = make_component(background->b0z_cell, false, false);
      // The canonical interchange omits padding. Internal and periodic guards
      // were populated from the global lattice above; derive true physical
      // exterior samples with the same staggered closure as the serial solver.
      // This is also what completes physical/internal corner blocks.
      fill_background_ghosts(
          grid, solver.config().boundary, b0x, b0y, b0z);
      solver.seed_background("b0x_face", b0x);
      solver.seed_background("b0y_face", b0y);
      solver.seed_background("b0z_cell", b0z);
    }
    backend::device_synchronize(nullptr);
  });
  mutation_started = true;
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-seed-tiles", tasks);
}

MhdTileRuntime::DenseField MhdTileRuntime::collect_register(
    int register_index, std::string_view phase) {
  return collect_field(register_index, false, phase);
}

MhdTileRuntime::DenseField MhdTileRuntime::collect_field(
    int register_index, bool residual, std::string_view phase) {
  if (!residual && (register_index < 0 || register_index >= 3)) {
    throw std::out_of_range{"distributed MHD register is out of range"};
  }
  const std::size_t nx = topology_.global_nx();
  const std::size_t ny = topology_.global_ny();
  const std::size_t cells = checked_product(nx, ny, "MHD cells");
  const std::size_t bx_size = checked_product(nx + 1, ny, "MHD Bx");
  const std::size_t by_size = checked_product(nx, ny + 1, "MHD By");
  DenseField dense;
  dense.state.global_nx = nx;
  dense.state.global_ny = ny;
  dense.state.rho.assign(cells, Real{0});
  dense.state.mx.assign(cells, Real{0});
  dense.state.my.assign(cells, Real{0});
  dense.state.mz.assign(cells, Real{0});
  dense.state.energy.assign(cells, Real{0});
  dense.state.bx_face.assign(bx_size, Real{0});
  dense.state.by_face.assign(by_size, Real{0});
  dense.state.bz_cell.assign(cells, Real{0});
  std::vector<int> cell_counts(cells, 0);
  std::vector<int> bx_counts(bx_size, 0);
  std::vector<int> by_counts(by_size, 0);
  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);

  auto tasks = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, register_index, residual, &dense, &cell_counts, &bx_counts,
       &by_counts, px, py, nx, ny](std::size_t local, WorkerContext&) {
    auto& solver = *solvers_[local];
    const auto& field = residual
        ? MhdTileAccess::residual(solver)
        : MhdTileAccess::state_register(solver, register_index);
    const Grid2D& grid = field.grid;
    const std::size_t endpoint =
        static_cast<std::size_t>(runtime_->rank()) *
            mapping_.devices_per_rank() +
        local;
    const TileExtent& tile = topology_.tile(endpoint);
    std::vector<Real> rho(grid.storage_size()), mx(grid.storage_size());
    std::vector<Real> my(grid.storage_size()), mz(grid.storage_size());
    std::vector<Real> energy(grid.storage_size()), bx(grid.storage_size());
    std::vector<Real> by(grid.storage_size()), bz(grid.storage_size());
    field.rho.copy_to_host(rho.data(), rho.size());
    field.mx.copy_to_host(mx.data(), mx.size());
    field.my.copy_to_host(my.data(), my.size());
    field.mz.copy_to_host(mz.data(), mz.size());
    field.energy.copy_to_host(energy.data(), energy.size());
    field.bx_face.copy_to_host(bx.data(), bx.size());
    field.by_face.copy_to_host(by.data(), by.size());
    field.bz_cell.copy_to_host(bz.data(), bz.size());

    for (int j = 0; j < grid.ny; ++j) {
      const std::size_t gy = tile.y.begin + static_cast<std::size_t>(j);
      for (int i = 0; i < grid.nx; ++i) {
        const std::size_t gx = tile.x.begin + static_cast<std::size_t>(i);
        const std::size_t global = row_major(gx, gy, nx);
        const std::size_t local_index = grid.index(i, j);
        dense.state.rho[global] = rho[local_index];
        dense.state.mx[global] = mx[local_index];
        dense.state.my[global] = my[local_index];
        dense.state.mz[global] = mz[local_index];
        dense.state.energy[global] = energy[local_index];
        dense.state.bz_cell[global] = bz[local_index];
        cell_counts[global] = 1;
      }
    }

    const int bx_begin = tile.coordinate.x == 0 ? 0 : 1;
    for (int j = 0; j < grid.ny; ++j) {
      const std::size_t gy = tile.y.begin + static_cast<std::size_t>(j);
      for (int i = bx_begin; i <= grid.nx; ++i) {
        const std::size_t gx = tile.x.begin + static_cast<std::size_t>(i);
        if (px && gx == nx) continue;
        const std::size_t global = row_major(gx, gy, nx + 1);
        dense.state.bx_face[global] = bx[grid.index(i, j)];
        bx_counts[global] = 1;
      }
    }

    const int by_begin = tile.coordinate.y == 0 ? 0 : 1;
    for (int j = by_begin; j <= grid.ny; ++j) {
      const std::size_t gy = tile.y.begin + static_cast<std::size_t>(j);
      if (py && gy == ny) continue;
      for (int i = 0; i < grid.nx; ++i) {
        const std::size_t gx = tile.x.begin + static_cast<std::size_t>(i);
        const std::size_t global = row_major(gx, gy, nx);
        dense.state.by_face[global] = by[grid.index(i, j)];
        by_counts[global] = 1;
      }
    }
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_, phase, tasks);

  auto reduce = [&](std::vector<Real>& values, std::vector<int>& counts,
                    const char* label) {
    allreduce_sum_in_place(*runtime_, values, MPI_DOUBLE,
                           "MPI_Allreduce(distributed MHD values)");
    allreduce_sum_in_place(*runtime_, counts, MPI_INT,
                           "MPI_Allreduce(distributed MHD ownership)");
    telemetry_.collective_bytes +=
        static_cast<std::uint64_t>(values.size() * sizeof(Real) +
                                   counts.size() * sizeof(int));
    require_exact_coverage(counts, label);
  };
  reduce(dense.state.rho, cell_counts, "cell lattice");
  // The six cell-centred arrays share exactly the same ownership mask.  Their
  // values still require independent reductions, while the proven mask need not
  // be transmitted six times.
  allreduce_sum_in_place(*runtime_, dense.state.mx, MPI_DOUBLE,
                         "MPI_Allreduce(distributed MHD mx)");
  allreduce_sum_in_place(*runtime_, dense.state.my, MPI_DOUBLE,
                         "MPI_Allreduce(distributed MHD my)");
  allreduce_sum_in_place(*runtime_, dense.state.mz, MPI_DOUBLE,
                         "MPI_Allreduce(distributed MHD mz)");
  allreduce_sum_in_place(*runtime_, dense.state.energy, MPI_DOUBLE,
                         "MPI_Allreduce(distributed MHD energy)");
  allreduce_sum_in_place(*runtime_, dense.state.bz_cell, MPI_DOUBLE,
                         "MPI_Allreduce(distributed MHD bz)");
  telemetry_.collective_bytes += static_cast<std::uint64_t>(
      5 * cells * sizeof(Real));

  allreduce_sum_in_place(*runtime_, dense.state.bx_face, MPI_DOUBLE,
                         "MPI_Allreduce(distributed MHD Bx)");
  allreduce_sum_in_place(*runtime_, bx_counts, MPI_INT,
                         "MPI_Allreduce(distributed MHD Bx ownership)");
  allreduce_sum_in_place(*runtime_, dense.state.by_face, MPI_DOUBLE,
                         "MPI_Allreduce(distributed MHD By)");
  allreduce_sum_in_place(*runtime_, by_counts, MPI_INT,
                         "MPI_Allreduce(distributed MHD By ownership)");
  telemetry_.collective_bytes += static_cast<std::uint64_t>(
      (bx_size + by_size) * (sizeof(Real) + sizeof(int)));

  // A periodic high face is a retained interchange duplicate, not a second
  // owned degree of freedom.  Leave it absent from the collective ownership
  // mask, validate every unique face exactly once, then rebuild the duplicate
  // from the canonical low face on every rank.  Synthesizing it before the
  // all-reduce would make every rank claim ownership of the same slot.
  if (px) {
    for (std::size_t j = 0; j < ny; ++j) {
      const std::size_t high = row_major(nx, j, nx + 1);
      if (bx_counts[high] != 0) {
        throw std::runtime_error{
            "periodic Bx high-face duplicate unexpectedly has an owner"};
      }
      bx_counts[high] = 1;
      dense.state.bx_face[high] =
          dense.state.bx_face[row_major(0, j, nx + 1)];
    }
  }
  if (py) {
    for (std::size_t i = 0; i < nx; ++i) {
      const std::size_t high = row_major(i, ny, nx);
      if (by_counts[high] != 0) {
        throw std::runtime_error{
            "periodic By high-face duplicate unexpectedly has an owner"};
      }
      by_counts[high] = 1;
      dense.state.by_face[high] =
          dense.state.by_face[row_major(i, 0, nx)];
    }
  }
  require_exact_coverage(bx_counts, "Bx face lattice");
  require_exact_coverage(by_counts, "By face lattice");
  return dense;
}

MhdGlobalBackground MhdTileRuntime::collect_background(
    std::string_view phase) {
  if (!global_config_.background.enabled) {
    throw std::logic_error{
        "cannot collect a disabled distributed MHD background"};
  }
  const std::size_t nx = topology_.global_nx();
  const std::size_t ny = topology_.global_ny();
  const std::size_t cells = checked_product(nx, ny, "MHD background cells");
  const std::size_t bx_size =
      checked_product(nx + 1, ny, "MHD background Bx");
  const std::size_t by_size =
      checked_product(nx, ny + 1, "MHD background By");
  MhdGlobalBackground background;
  background.global_nx = nx;
  background.global_ny = ny;
  background.b0x_face.assign(bx_size, Real{0});
  background.b0y_face.assign(by_size, Real{0});
  background.b0z_cell.assign(cells, Real{0});
  std::vector<int> cell_counts(cells, 0);
  std::vector<int> bx_counts(bx_size, 0);
  std::vector<int> by_counts(by_size, 0);
  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);

  auto tasks = mhd_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(),
      [this, &background, &cell_counts, &bx_counts, &by_counts, px, py,
       nx, ny](std::size_t local, WorkerContext&) {
        const auto& solver = *solvers_[local];
        const auto& field = MhdTileAccess::background(solver);
        if (!field.active) {
          throw std::logic_error{
              "enabled distributed MHD background is not active"};
        }
        const Grid2D& grid = field.grid;
        const std::size_t endpoint =
            static_cast<std::size_t>(runtime_->rank()) *
                mapping_.devices_per_rank() +
            local;
        const TileExtent& tile = topology_.tile(endpoint);
        std::vector<Real> bx(grid.storage_size());
        std::vector<Real> by(grid.storage_size());
        std::vector<Real> bz(grid.storage_size());
        field.b0x_face.copy_to_host(bx.data(), bx.size());
        field.b0y_face.copy_to_host(by.data(), by.size());
        field.b0z_cell.copy_to_host(bz.data(), bz.size());

        for (int j = 0; j < grid.ny; ++j) {
          const std::size_t gy =
              tile.y.begin + static_cast<std::size_t>(j);
          for (int i = 0; i < grid.nx; ++i) {
            const std::size_t gx =
                tile.x.begin + static_cast<std::size_t>(i);
            const std::size_t global = row_major(gx, gy, nx);
            background.b0z_cell[global] = bz[grid.index(i, j)];
            cell_counts[global] = 1;
          }
        }

        const int bx_begin = tile.coordinate.x == 0 ? 0 : 1;
        for (int j = 0; j < grid.ny; ++j) {
          const std::size_t gy =
              tile.y.begin + static_cast<std::size_t>(j);
          for (int i = bx_begin; i <= grid.nx; ++i) {
            const std::size_t gx =
                tile.x.begin + static_cast<std::size_t>(i);
            if (px && gx == nx) continue;
            const std::size_t global = row_major(gx, gy, nx + 1);
            background.b0x_face[global] = bx[grid.index(i, j)];
            bx_counts[global] = 1;
          }
        }

        const int by_begin = tile.coordinate.y == 0 ? 0 : 1;
        for (int j = by_begin; j <= grid.ny; ++j) {
          const std::size_t gy =
              tile.y.begin + static_cast<std::size_t>(j);
          if (py && gy == ny) continue;
          for (int i = 0; i < grid.nx; ++i) {
            const std::size_t gx =
                tile.x.begin + static_cast<std::size_t>(i);
            const std::size_t global = row_major(gx, gy, nx);
            background.b0y_face[global] = by[grid.index(i, j)];
            by_counts[global] = 1;
          }
        }
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_, phase, tasks);

  allreduce_sum_in_place(*runtime_, background.b0z_cell, MPI_DOUBLE,
                         "MPI_Allreduce(distributed MHD background Bz)");
  allreduce_sum_in_place(*runtime_, cell_counts, MPI_INT,
                         "MPI_Allreduce(distributed MHD background cells)");
  allreduce_sum_in_place(*runtime_, background.b0x_face, MPI_DOUBLE,
                         "MPI_Allreduce(distributed MHD background Bx)");
  allreduce_sum_in_place(*runtime_, bx_counts, MPI_INT,
                         "MPI_Allreduce(distributed MHD background Bx ownership)");
  allreduce_sum_in_place(*runtime_, background.b0y_face, MPI_DOUBLE,
                         "MPI_Allreduce(distributed MHD background By)");
  allreduce_sum_in_place(*runtime_, by_counts, MPI_INT,
                         "MPI_Allreduce(distributed MHD background By ownership)");
  telemetry_.collective_bytes += static_cast<std::uint64_t>(
      cells * (sizeof(Real) + sizeof(int)) +
      (bx_size + by_size) * (sizeof(Real) + sizeof(int)));

  require_exact_coverage(cell_counts, "background cell lattice");
  if (px) {
    for (std::size_t y = 0; y < ny; ++y) {
      const std::size_t high = row_major(nx, y, nx + 1);
      if (bx_counts[high] != 0) {
        throw std::runtime_error{
            "periodic background Bx high-face duplicate unexpectedly has an owner"};
      }
      bx_counts[high] = 1;
      background.b0x_face[high] =
          background.b0x_face[row_major(0, y, nx + 1)];
    }
  }
  if (py) {
    for (std::size_t x = 0; x < nx; ++x) {
      const std::size_t high = row_major(x, ny, nx);
      if (by_counts[high] != 0) {
        throw std::runtime_error{
            "periodic background By high-face duplicate unexpectedly has an owner"};
      }
      by_counts[high] = 1;
      background.b0y_face[high] = background.b0y_face[x];
    }
  }
  require_exact_coverage(bx_counts, "background Bx face lattice");
  require_exact_coverage(by_counts, "background By face lattice");
  validate_mhd_global_background(background);
  return background;
}

std::uint64_t MhdTileRuntime::transfer_halo_axis(
    Direction positive_direction, std::string_view phase) {
  if (positive_direction != Direction::x_high
      && positive_direction != Direction::y_high) {
    throw std::invalid_argument{
        "distributed MHD register exchange requires a positive axis"};
  }
  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);
  constexpr std::size_t slots_per_epoch =
      static_cast<std::size_t>(Transport::tag_slots_per_epoch);
  std::vector<std::vector<ByteTransfer>> epoch_transfers;
  collective_try(
      *runtime_, worker_epoch_, phase,
      "MHD halo descriptor preparation failed", -1, [&] {
    const auto edges = make_mhd_halo_edges(
        topology_, positive_direction, px, py);
    std::vector<std::size_t> remote_message_counts(
        static_cast<std::size_t>(runtime_->size()), 0);
    std::size_t local_message_count = 0;
    const auto append = [&](const Endpoint& source,
                            Direction source_direction,
                            const Endpoint& destination,
                            Direction destination_direction,
                            bool local_copy) {
      const std::size_t source_local = source.rank_local_index;
      const std::size_t destination_local =
          local_copy ? destination.rank_local_index : source_local;
      auto& send = register_halos_.at(source_local)
                       ->send[direction_index(source_direction)];
      auto& receive = register_halos_.at(destination_local)
                          ->receive[direction_index(
                              local_copy ? destination_direction
                                         : source_direction)];
      if (send.empty() || receive.empty() || send.bytes() != receive.bytes()) {
        throw std::logic_error{
            "MHD neighbour halo buffers have incompatible payload sizes"};
      }
      ByteTransfer transfer;
      transfer.peer_rank = local_copy
          ? runtime_->rank() : destination.world_rank;
      std::size_t epoch = 0;
      if (local_copy) {
        transfer.tag_slot = static_cast<std::uint32_t>(
            local_message_count++ % slots_per_epoch);
      } else {
        // Both endpoints filter the same global edge order, so this per-peer
        // ordinal selects the same epoch and tag on both MPI ranks.
        std::size_t& peer_messages = remote_message_counts.at(
            static_cast<std::size_t>(destination.world_rank));
        const std::size_t message = peer_messages++;
        epoch = message / slots_per_epoch;
        transfer.tag_slot = static_cast<std::uint32_t>(
            message % slots_per_epoch);
      }
      transfer.send_buffer = send.device_ptr();
      transfer.receive_buffer = receive.device_ptr();
      transfer.bytes = send.bytes();
      transfer.residence = BufferResidence::device;
      transfer.send_device = send.owner_device();
      transfer.receive_device = receive.owner_device();
      transfer.send_stream = register_halos_.at(source_local)
                                 ->communication_stream;
      transfer.receive_stream = register_halos_.at(destination_local)
                                    ->communication_stream;
      if (epoch_transfers.size() <= epoch) {
        epoch_transfers.resize(epoch + 1);
      }
      epoch_transfers[epoch].push_back(transfer);
    };

    for (const MhdHaloEdge& edge : edges) {
      const Endpoint& first = mapping_.endpoint(edge.first_endpoint);
      const Endpoint& second = mapping_.endpoint(edge.second_endpoint);
      if (first.world_rank == second.world_rank) {
        if (runtime_->rank() == first.world_rank) {
          append(first, edge.first_direction,
                 second, edge.second_direction, true);
          append(second, edge.second_direction,
                 first, edge.first_direction, true);
        }
      } else if (runtime_->rank() == first.world_rank) {
        append(first, edge.first_direction,
               second, edge.second_direction, false);
      } else if (runtime_->rank() == second.world_rank) {
        append(second, edge.second_direction,
               first, edge.first_direction, false);
      }
    }
  });

  // Tags are matched with a peer rank, so unrelated neighbours may reuse all
  // 64 slots in the same epoch.  A collective maximum preserves the required
  // Transport::begin sequence while ranks with less boundary traffic enter
  // the trailing epochs with no descriptors.
  const std::size_t epoch_count = allreduce_max_size(
      *runtime_, epoch_transfers.size(),
      "MPI_Allreduce(distributed MHD transport epoch count)");
  for (std::size_t epoch = 0; epoch < epoch_count; ++epoch) {
    const std::span<const ByteTransfer> transfers =
        epoch < epoch_transfers.size()
            ? std::span<const ByteTransfer>{epoch_transfers[epoch]}
            : std::span<const ByteTransfer>{};
    auto batch = transport_->begin(transfers);
    batch.wait();
    telemetry_.transport = transport_->telemetry();
  }
  return static_cast<std::uint64_t>(epoch_count);
}

void MhdTileRuntime::exchange_register_axis(
    int register_index, Direction positive_direction,
    std::string_view phase) {
  if (register_index < 0 || register_index >= 3) {
    throw std::out_of_range{"distributed MHD register is out of range"};
  }
  if (positive_direction != Direction::x_high
      && positive_direction != Direction::y_high) {
    throw std::invalid_argument{
        "distributed MHD register exchange requires a positive axis"};
  }
  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);
  const bool x_axis = positive_direction == Direction::x_high;
  const std::array<Direction, 2> directions = x_axis
      ? std::array{Direction::x_low, Direction::x_high}
      : std::array{Direction::y_low, Direction::y_high};

  telemetry_.register_halo_epochs += run_device_halo_exchange_phase(
      *runtime_, *workers_, worker_epoch_, solvers_.size(), phase,
      [this, register_index, directions, px, py]
      (std::size_t local, WorkerContext& context) {
        auto& solver = *solvers_[local];
        auto& field = MhdTileAccess::state_register(solver, register_index);
        const std::size_t endpoint =
            static_cast<std::size_t>(runtime_->rank()) *
                mapping_.devices_per_rank() + local;
        if (!has_neighbor_on_axis(topology_, endpoint, directions, px, py)) {
          return;
        }

        auto& buffers = *register_halos_[local];
        const auto components = field_halo_sources(field);
        communication_waits_for_default_stream(context);
        pack_neighbor_halos(field.grid, endpoint, directions, topology_, px,
                            py, components, buffers,
                            context.communication_stream.get());
      },
      [this, positive_direction, phase] {
        return transfer_halo_axis(positive_direction, phase);
      },
      [this, register_index, directions, px, py]
      (std::size_t local, WorkerContext& context) {
        auto& solver = *solvers_[local];
        auto& field = MhdTileAccess::state_register(solver, register_index);
        const std::size_t endpoint =
            static_cast<std::size_t>(runtime_->rank()) *
                mapping_.devices_per_rank() + local;
        if (!has_neighbor_on_axis(topology_, endpoint, directions, px, py)) {
          return;
        }

        auto& buffers = *register_halos_[local];
        const auto components = field_halo_destinations(
            field, mhd_register_halo_layouts());
        unpack_neighbor_halos(field.grid, endpoint, directions, topology_, px,
                              py, components, buffers,
                              context.communication_stream.get());
        default_stream_waits_for_communication(context);
        MhdTileAccess::note_state_reconciled(solver);
      });
}

void MhdTileRuntime::exchange_background_axis(
    Direction positive_direction, std::string_view phase) {
  if (!global_config_.background.enabled) return;
  if (positive_direction != Direction::x_high
      && positive_direction != Direction::y_high) {
    throw std::invalid_argument{
        "distributed MHD background exchange requires a positive axis"};
  }
  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);
  const bool x_axis = positive_direction == Direction::x_high;
  const std::array<Direction, 2> directions = x_axis
      ? std::array{Direction::x_low, Direction::x_high}
      : std::array{Direction::y_low, Direction::y_high};

  (void)run_device_halo_exchange_phase(
      *runtime_, *workers_, worker_epoch_, solvers_.size(), phase,
      [this, directions, px, py](std::size_t local,
                                 WorkerContext& context) {
        auto& solver = *solvers_[local];
        const Grid2D grid = solver.grid();
        const std::size_t endpoint =
            static_cast<std::size_t>(runtime_->rank()) *
                mapping_.devices_per_rank() + local;
        if (!has_neighbor_on_axis(topology_, endpoint, directions, px, py)) {
          return;
        }

        auto& buffers = *register_halos_[local];
        const auto components =
            background_halo_sources(MhdTileAccess::background(solver));
        communication_waits_for_default_stream(context);
        pack_neighbor_halos(grid, endpoint, directions, topology_, px, py,
                            components, buffers,
                            context.communication_stream.get());
      },
      [this, positive_direction, phase] {
        return transfer_halo_axis(positive_direction, phase);
      },
      [this, directions, px, py](std::size_t local,
                                 WorkerContext& context) {
        auto& solver = *solvers_[local];
        const Grid2D grid = solver.grid();
        const std::size_t endpoint =
            static_cast<std::size_t>(runtime_->rank()) *
                mapping_.devices_per_rank() + local;
        if (!has_neighbor_on_axis(topology_, endpoint, directions, px, py)) {
          return;
        }

        auto& buffers = *register_halos_[local];
        const auto components =
            background_halo_destinations(MhdTileAccess::background(solver));
        unpack_neighbor_halos(grid, endpoint, directions, topology_, px, py,
                              components, buffers,
                              context.communication_stream.get());
        default_stream_waits_for_communication(context);
        MhdTileAccess::note_state_reconciled(solver);
      });
}

void MhdTileRuntime::reconcile_background(std::string_view phase) {
  if (!global_config_.background.enabled) return;
  exchange_background_axis(Direction::x_high, phase);
  exchange_background_axis(Direction::y_high, phase);
  auto closures = mhd_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(), [this](std::size_t local, WorkerContext&) {
        auto& solver = *solvers_[local];
        for (int side = 0; side < 4; ++side) {
          const int mode =
              background_boundary_mode(solver.config().boundary.field[side]);
          if (mode == 4) continue;
          mhd::launch_mhd_fill_ghosts_background(
              MhdTileAccess::background(solver), side, mode, nullptr);
        }
        MhdTileAccess::note_background_reconciled(solver);
        backend::device_synchronize(nullptr);
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         phase, closures);
}

void MhdTileRuntime::reconcile_register(int register_index,
                                        std::string_view phase) {
  // X first, then Y over the x-extended pitch.  The second phase therefore
  // propagates diagonal corner guards without a separate diagonal message.
  exchange_register_axis(register_index, Direction::x_high, phase);
  exchange_register_axis(register_index, Direction::y_high, phase);
  ++telemetry_.state_reconciliations;
}

void MhdTileRuntime::exchange_face_records_axis(
    Direction positive_direction, std::string_view phase) {
  if (positive_direction != Direction::x_high
      && positive_direction != Direction::y_high) {
    throw std::invalid_argument{
        "distributed MHD face-record exchange requires a positive axis"};
  }
  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);
  const bool x_axis = positive_direction == Direction::x_high;
  const std::array<Direction, 2> directions = x_axis
      ? std::array{Direction::x_low, Direction::x_high}
      : std::array{Direction::y_low, Direction::y_high};
  MhdHaloLayouts layouts{};
  layouts.fill(x_axis ? MhdHaloStagger::x_face
                      : MhdHaloStagger::y_face);
  const std::size_t passes =
      global_config_.background.enabled ? mhd_face_record_pass_count : 1;

  for (std::size_t pass = 0; pass < passes; ++pass) {
    (void)run_device_halo_exchange_phase(
        *runtime_, *workers_, worker_epoch_, solvers_.size(), phase,
        [this, directions, px, py, x_axis, pass]
          (std::size_t local, WorkerContext& context) {
          auto& solver = *solvers_[local];
          const auto records = MhdTileAccess::face_records(solver, x_axis);
          auto& flux = records.flux;
          auto& parts = records.momentum;
          const std::size_t endpoint =
              static_cast<std::size_t>(runtime_->rank()) *
                  mapping_.devices_per_rank() + local;
          if (!has_neighbor_on_axis(topology_, endpoint, directions, px, py)) {
            return;
          }

          auto& buffers = *register_halos_[local];
          const auto components =
              face_record_halo_sources(flux, parts, pass);
          communication_waits_for_default_stream(context);
          pack_neighbor_halos(flux.grid, endpoint, directions, topology_, px,
                              py, components, buffers,
                              context.communication_stream.get());
        },
        [this, positive_direction, phase] {
          return transfer_halo_axis(positive_direction, phase);
        },
        [this, directions, px, py, x_axis, pass, layouts]
          (std::size_t local, WorkerContext& context) {
          auto& solver = *solvers_[local];
          const auto records = MhdTileAccess::face_records(solver, x_axis);
          auto& flux = records.flux;
          auto& parts = records.momentum;
          const std::size_t endpoint =
              static_cast<std::size_t>(runtime_->rank()) *
                  mapping_.devices_per_rank() + local;
          if (!has_neighbor_on_axis(topology_, endpoint, directions, px, py)) {
            return;
          }

          auto& buffers = *register_halos_[local];
          const auto components =
              face_record_halo_destinations(flux, parts, pass, layouts);
          unpack_neighbor_halos(flux.grid, endpoint, directions, topology_, px,
                                py, components, buffers,
                                context.communication_stream.get());
          default_stream_waits_for_communication(context);
        });
    ++telemetry_.canonical_face_record_passes;
  }
}

void MhdTileRuntime::reconcile_face_records(std::string_view phase) {
  exchange_face_records_axis(Direction::x_high, phase);
  exchange_face_records_axis(Direction::y_high, phase);
}

void MhdTileRuntime::exchange_residual_faces_axis(
    Direction positive_direction, std::string_view phase) {
  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);
  const bool x_axis = positive_direction == Direction::x_high;
  const std::array<Direction, 2> directions = x_axis
      ? std::array{Direction::x_low, Direction::x_high}
      : std::array{Direction::y_low, Direction::y_high};
  (void)run_device_halo_exchange_phase(
      *runtime_, *workers_, worker_epoch_, solvers_.size(), phase,
      [this, directions, px, py]
      (std::size_t local, WorkerContext& context) {
        auto& solver = *solvers_[local];
        const Grid2D grid = solver.grid();
        const std::size_t endpoint =
            static_cast<std::size_t>(runtime_->rank()) *
                mapping_.devices_per_rank() + local;
        auto& buffers = *register_halos_[local];
        DeviceHaloConstComponents components{};
        set_halo_source(components, 5,
                        MhdTileAccess::residual(solver).bx_face,
                        DeviceHaloValueKind::real);
        set_halo_source(components, 6,
                        MhdTileAccess::residual(solver).by_face,
                        DeviceHaloValueKind::real);
        communication_waits_for_default_stream(context);
        pack_neighbor_halos(grid, endpoint, directions, topology_, px, py,
                            components, buffers,
                            context.communication_stream.get());
      },
      [this, positive_direction, phase] {
        return transfer_halo_axis(positive_direction, phase);
      },
      [this, directions, px, py]
      (std::size_t local, WorkerContext& context) {
        auto& solver = *solvers_[local];
        const Grid2D grid = solver.grid();
        const std::size_t endpoint =
            static_cast<std::size_t>(runtime_->rank()) *
                mapping_.devices_per_rank() + local;
        auto& buffers = *register_halos_[local];
        const MhdHaloLayouts layouts = mhd_register_halo_layouts();
        DeviceHaloComponents components{};
        set_halo_destination(components, 5,
                             MhdTileAccess::residual(solver).bx_face,
                             DeviceHaloValueKind::real, layouts[5]);
        set_halo_destination(components, 6,
                             MhdTileAccess::residual(solver).by_face,
                             DeviceHaloValueKind::real, layouts[6]);
        unpack_neighbor_halos(grid, endpoint, directions, topology_, px, py,
                              components, buffers,
                              context.communication_stream.get());
        default_stream_waits_for_communication(context);
      });
}

void MhdTileRuntime::reconcile_residual_faces(std::string_view phase) {
  exchange_residual_faces_axis(Direction::x_high, phase);
  exchange_residual_faces_axis(Direction::y_high, phase);
}

void MhdTileRuntime::exchange_emf_inputs_axis(
    Direction positive_direction, bool masks, std::string_view phase) {
  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);
  const bool x_axis = positive_direction == Direction::x_high;
  const std::array<Direction, 2> directions = x_axis
      ? std::array{Direction::x_low, Direction::x_high}
      : std::array{Direction::y_low, Direction::y_high};
  MhdHaloLayouts layouts{};
  layouts.fill(MhdHaloStagger::cell);
  layouts[0] = MhdHaloStagger::cell_extended_y;
  layouts[5] = MhdHaloStagger::x_face_extended_y;
  layouts[6] = MhdHaloStagger::y_face;

  (void)run_device_halo_exchange_phase(
      *runtime_, *workers_, worker_epoch_, solvers_.size(), phase,
      [this, directions, px, py, masks]
      (std::size_t local, WorkerContext& context) {
        auto& solver = *solvers_[local];
        auto& emf = MhdTileAccess::emf(solver);
        const Grid2D grid = solver.grid();
        const std::size_t endpoint =
            static_cast<std::size_t>(runtime_->rank()) *
                mapping_.devices_per_rank() + local;
        auto& buffers = *register_halos_[local];
        DeviceHaloConstComponents components{};
        if (masks) {
          set_halo_source(components, 5, emf.xface_no_jump,
                          DeviceHaloValueKind::int32);
          set_halo_source(components, 6, emf.yface_no_jump,
                          DeviceHaloValueKind::int32);
        } else {
          set_halo_source(components, 0, emf.cell_ez_average,
                          DeviceHaloValueKind::real);
          set_halo_source(components, 5, emf.xface_ez,
                          DeviceHaloValueKind::real);
          set_halo_source(components, 6, emf.yface_ez,
                          DeviceHaloValueKind::real);
        }
        communication_waits_for_default_stream(context);
        pack_neighbor_halos(grid, endpoint, directions, topology_, px, py,
                            components, buffers,
                            context.communication_stream.get());
      },
      [this, positive_direction, phase] {
        return transfer_halo_axis(positive_direction, phase);
      },
      [this, directions, px, py, masks, layouts]
      (std::size_t local, WorkerContext& context) {
        auto& solver = *solvers_[local];
        auto& emf = MhdTileAccess::emf(solver);
        const Grid2D grid = solver.grid();
        const std::size_t endpoint =
            static_cast<std::size_t>(runtime_->rank()) *
                mapping_.devices_per_rank() + local;
        auto& buffers = *register_halos_[local];
        DeviceHaloComponents components{};
        if (masks) {
          set_halo_destination(components, 5, emf.xface_no_jump,
                               DeviceHaloValueKind::int32, layouts[5]);
          set_halo_destination(components, 6, emf.yface_no_jump,
                               DeviceHaloValueKind::int32, layouts[6]);
        } else {
          set_halo_destination(components, 0, emf.cell_ez_average,
                               DeviceHaloValueKind::real, layouts[0]);
          set_halo_destination(components, 5, emf.xface_ez,
                               DeviceHaloValueKind::real, layouts[5]);
          set_halo_destination(components, 6, emf.yface_ez,
                               DeviceHaloValueKind::real, layouts[6]);
        }
        unpack_neighbor_halos(grid, endpoint, directions, topology_, px, py,
                              components, buffers,
                              context.communication_stream.get());
        default_stream_waits_for_communication(context);
      });
}

void MhdTileRuntime::reconcile_emf_inputs(std::string_view phase) {
  exchange_emf_inputs_axis(Direction::x_high, false, phase);
  exchange_emf_inputs_axis(Direction::y_high, false, phase);
  exchange_emf_inputs_axis(Direction::x_high, true, phase);
  exchange_emf_inputs_axis(Direction::y_high, true, phase);
}

void MhdTileRuntime::exchange_corner_emf_axis(
    Direction positive_direction, std::string_view phase) {
  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);
  const bool x_axis = positive_direction == Direction::x_high;
  const std::array<Direction, 2> directions = x_axis
      ? std::array{Direction::x_low, Direction::x_high}
      : std::array{Direction::y_low, Direction::y_high};
  const std::size_t wire_component = x_axis ? 5u : 6u;
  MhdHaloLayouts layouts{};
  layouts.fill(MhdHaloStagger::cell);
  layouts[wire_component] = MhdHaloStagger::node;
  (void)run_device_halo_exchange_phase(
      *runtime_, *workers_, worker_epoch_, solvers_.size(), phase,
      [this, directions, px, py, wire_component]
      (std::size_t local, WorkerContext& context) {
        auto& solver = *solvers_[local];
        auto& emf = MhdTileAccess::emf(solver);
        const Grid2D grid = solver.grid();
        const std::size_t endpoint =
            static_cast<std::size_t>(runtime_->rank()) *
                mapping_.devices_per_rank() + local;
        auto& buffers = *register_halos_[local];
        DeviceHaloConstComponents components{};
        set_halo_source(components, wire_component, emf.ez_edge,
                        DeviceHaloValueKind::real);
        communication_waits_for_default_stream(context);
        pack_neighbor_halos(grid, endpoint, directions, topology_, px, py,
                            components, buffers,
                            context.communication_stream.get());
      },
      [this, positive_direction, phase] {
        return transfer_halo_axis(positive_direction, phase);
      },
      [this, directions, px, py, wire_component, layouts]
      (std::size_t local, WorkerContext& context) {
        auto& solver = *solvers_[local];
        auto& emf = MhdTileAccess::emf(solver);
        const Grid2D grid = solver.grid();
        const std::size_t endpoint =
            static_cast<std::size_t>(runtime_->rank()) *
                mapping_.devices_per_rank() + local;
        auto& buffers = *register_halos_[local];
        DeviceHaloComponents components{};
        set_halo_destination(components, wire_component, emf.ez_edge,
                             DeviceHaloValueKind::real,
                             layouts[wire_component]);
        unpack_neighbor_halos(grid, endpoint, directions, topology_, px, py,
                              components, buffers,
                              context.communication_stream.get());
        default_stream_waits_for_communication(context);
      });
}

void MhdTileRuntime::reconcile_emf(std::string_view phase) {
  exchange_corner_emf_axis(Direction::x_high, phase);
  exchange_corner_emf_axis(Direction::y_high, phase);
  ++telemetry_.emf_reconciliations;
}

Real MhdTileRuntime::cfl_limit() {
  require_open_seeded();
  reconcile_register(0, "mhd-cfl-state");
  std::vector<Real> limits(solvers_.size(), Real{0});
  auto tasks = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, &limits](std::size_t local, WorkerContext&) {
    limits[local] = MhdTileAccess::cfl_limit(*solvers_[local], 0);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-cfl-local", tasks);
  const Real local = *std::min_element(limits.begin(), limits.end());
  return static_cast<Real>(runtime_->allreduce_min(local));
}

void MhdTileRuntime::restore_request_backups() {
  auto tasks = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this](std::size_t local, WorkerContext&) {
    MhdTileAccess::restore_request(*solvers_[local]);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-request-restore", tasks);
  reconcile_register(0, "mhd-request-restore-reconcile");
}

void MhdTileRuntime::prepare_step_request(Real dt, bool check_cfl) {
  require_open_seeded();
  const bool local_dt_valid = std::isfinite(dt) && dt > Real{0};
  collective_require(
      *runtime_, worker_epoch_, local_dt_valid, "mhd-step-validate-dt",
      "distributed MHD timestep must be finite and positive");
  const Real minimum_dt = static_cast<Real>(runtime_->allreduce_min(dt));
  const Real maximum_dt = static_cast<Real>(runtime_->allreduce_max(dt));
  collective_require(
      *runtime_, worker_epoch_, minimum_dt == maximum_dt,
      "mhd-step-consistent-dt",
      "distributed MHD ranks supplied different timesteps");
  if (check_cfl) {
    const Real limit = cfl_limit();
    collective_require(
        *runtime_, worker_epoch_, dt <= limit, "mhd-step-cfl",
        "distributed MHD timestep exceeds the global CFL limit");
  }

  auto request_snapshot = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this](std::size_t local, WorkerContext&) {
    MhdTileAccess::snapshot_request(*solvers_[local]);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-request-snapshot", request_snapshot);
}

Real MhdTileRuntime::low_order_anchor() {
  std::vector<Real> anchors(solvers_.size(), Real{0});
  auto anchor_tasks = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, &anchors](std::size_t local, WorkerContext&) {
    anchors[local] = MhdTileAccess::low_order_anchor(*solvers_[local]);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-low-order-anchor", anchor_tasks);
  Real global_anchor = *std::min_element(anchors.begin(), anchors.end());
  return static_cast<Real>(runtime_->allreduce_min(global_anchor));
}

void MhdTileRuntime::snapshot_substep(bool low_order_interval,
                                      Real global_anchor) {
  auto tasks = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, low_order_interval, global_anchor]
      (std::size_t local, WorkerContext& context) {
    auto& solver = *solvers_[local];
    context.compute_ready.record(nullptr);
    backend::stream_wait_event(context.compute_stream.get(),
                               context.compute_ready.get());
    MhdTileAccess::snapshot_substep(
        solver, low_order_interval, global_anchor >= Real{1},
        context.compute_stream.get());
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-substep-snapshot", tasks);
}

void MhdTileRuntime::prepare_step_stage(int stage) {
  auto face_records = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, stage](std::size_t local, WorkerContext&) {
    auto& solver = *solvers_[local];
    const std::size_t endpoint =
        static_cast<std::size_t>(runtime_->rank()) *
            mapping_.devices_per_rank() + local;
    const bool px = periodic_x(global_config_);
    const bool py = periodic_y(global_config_);
    mhd::FaceOwnershipFlags4 ownership{};
    for (int side = 0; side < 4; ++side) {
      const Direction direction = static_cast<Direction>(side);
      const auto neighbor = topology_.neighbor(endpoint, direction, px, py);
      if (!neighbor || *neighbor == endpoint) continue;
      const auto owner = topology_.canonical_face_owner(
          endpoint, direction, px, py);
      if (!owner) {
        throw std::logic_error{
            "distributed MHD shared face has no canonical owner"};
      }
      ownership.side[side] = *owner == endpoint ? 1 : 0;
    }
    MhdTileAccess::prepare_face_records(solver, stage, ownership);
    backend::device_synchronize(nullptr);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-stage-face-records", face_records);
  reconcile_face_records("mhd-stage-face-records");

  auto residual_pre = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, stage](std::size_t local, WorkerContext&) {
    auto& solver = *solvers_[local];
    MhdTileAccess::consume_face_records(solver, stage);
    backend::device_synchronize(nullptr);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-stage-flux-emf", residual_pre);
  reconcile_emf_inputs("mhd-stage-emf-inputs");

  auto finish_emf = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this](std::size_t local, WorkerContext&) {
    auto& solver = *solvers_[local];
    MhdTileAccess::finish_emf(solver);
    backend::device_synchronize(nullptr);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-stage-emf-finish", finish_emf);
  reconcile_emf("mhd-stage-emf");

  auto ct_rate = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this](std::size_t local, WorkerContext&) {
    auto& solver = *solvers_[local];
    MhdTileAccess::compute_ct_rate(solver);
    backend::device_synchronize(nullptr);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-stage-ct-rate", ct_rate);
  reconcile_residual_faces("mhd-stage-rate");

  auto finish = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this](std::size_t local, WorkerContext&) {
    MhdTileAccess::finish_split_energy(*solvers_[local]);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-stage-energy", finish);
}

Real MhdTileRuntime::apply_and_assess_step_stage(int stage, Real trial) {
  std::vector<Real> theta(solvers_.size(), Real{1});
  auto combine = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, stage, trial](std::size_t local, WorkerContext&) {
    MhdTileAccess::apply_stage(*solvers_[local], stage, trial);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-stage-combine", combine);

  // A stage combine updates only owned cells/faces. Positivity's MP
  // collocation reads face guards as well, so canonicalize shared faces and
  // refresh true physical closures before asking any tile to accept the stage.
  const int output_register = stage == 2 ? 0 : stage + 1;
  reconcile_register(output_register, "mhd-stage-state");
  auto assess = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, stage, &theta](std::size_t local, WorkerContext&) {
    theta[local] = MhdTileAccess::assess_stage(*solvers_[local], stage);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-stage-positivity", assess);
  Real global_theta = *std::min_element(theta.begin(), theta.end());
  global_theta = static_cast<Real>(runtime_->allreduce_min(global_theta));
  ++telemetry_.stage_evaluations;
  return global_theta;
}

Real MhdTileRuntime::attempt_substep(Real trial) {
  for (int stage = 0; stage < 3; ++stage) {
    prepare_step_stage(stage);
    const Real theta = apply_and_assess_step_stage(stage, trial);
    if (!(theta >= Real{1})) return theta;
  }
  return Real{1};
}

Real MhdTileRuntime::substep_cfl_limit(int order, std::string_view phase) {
  std::vector<Real> limits(solvers_.size(), Real{0});
  auto tasks = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, order, &limits](std::size_t local, WorkerContext&) {
    limits[local] = MhdTileAccess::cfl_limit(*solvers_[local], order);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_, phase, tasks);
  Real global_limit = *std::min_element(limits.begin(), limits.end());
  return static_cast<Real>(runtime_->allreduce_min(global_limit));
}

void MhdTileRuntime::reject_substep(StepController& controller,
                                    Real rejected_theta) {
  ++telemetry_.rejected_attempts;
  const bool high_order_attempt = !controller.low_order_interval;
  auto rollback = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this](std::size_t local, WorkerContext&) {
    MhdTileAccess::rollback_substep(*solvers_[local]);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-substep-rollback", rollback);
  reconcile_register(0, "mhd-substep-rollback-state");
  if (++controller.retries > maximum_mhd_step_retries) {
    throw std::runtime_error{
        "distributed MHD could not find a positive conservative substep"};
  }
  if (high_order_attempt && controller.global_anchor >= Real{1}) {
    controller.low_order_interval = true;
    const Real global_limit = substep_cfl_limit(1, "mhd-low-order-cfl");
    controller.trial = std::min(controller.trial, global_limit);
    return;
  }
  const Real suggested = Real{0.8} * rejected_theta;
  const Real factor =
      std::max(Real{0.05}, std::min(Real{0.8}, suggested));
  controller.trial *= factor;
  if (!(std::isfinite(controller.trial) && controller.trial > Real{0}) ||
      !(controller.remaining - controller.trial < controller.remaining)) {
    throw std::runtime_error{
        "distributed MHD positivity retry cannot advance time"};
  }
}

bool MhdTileRuntime::accept_and_advance_substep(
    StepController& controller) {
  auto accepted = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, &controller](std::size_t local, WorkerContext&) {
    MhdTileAccess::accept_substep(*solvers_[local],
                                  controller.low_order_interval);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-substep-accept", accepted);
  ++telemetry_.accepted_substeps;
  controller.retries = 0;
  if (controller.trial == controller.remaining) {
    controller.remaining = Real{0};
    return false;
  }
  const Real next_remaining = controller.remaining - controller.trial;
  if (!(next_remaining > Real{0}) ||
      !(next_remaining < controller.remaining)) {
    throw std::runtime_error{
        "distributed MHD substep made no time progress"};
  }
  controller.remaining = next_remaining;
  const int order = controller.low_order_interval ? 1 : 0;
  const Real global_limit = substep_cfl_limit(order, "mhd-next-substep-cfl");
  controller.trial = std::min(
      std::min(Real{2} * controller.trial, controller.remaining),
      global_limit);
  return true;
}

void MhdTileRuntime::advance_step(Real dt) {
  const Real global_anchor = low_order_anchor();
  StepController controller{dt, dt, global_anchor};
  while (controller.remaining > Real{0}) {
    if (++controller.attempts > maximum_mhd_step_attempts) {
      throw std::runtime_error{
          "distributed MHD positivity controller exceeded its substep limit"};
    }
    controller.trial = std::min(controller.trial, controller.remaining);
    snapshot_substep(controller.low_order_interval, controller.global_anchor);
    const Real rejected_theta = attempt_substep(controller.trial);
    if (!(rejected_theta >= Real{1})) {
      reject_substep(controller, rejected_theta);
      continue;
    }
    if (!accept_and_advance_substep(controller)) break;
  }
}

void MhdTileRuntime::restore_step_after_failure() noexcept {
  try {
    restore_request_backups();
  } catch (...) {
    // Preserve the original distributed error. A failed restore leaves the
    // runtime unusable, but collective close must remain available to drain
    // Transport and release worker devices.
    lifecycle_.poisoned = true;
  }
}

void MhdTileRuntime::finish_step() {
  auto reset_modes = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this](std::size_t local, WorkerContext&) {
    MhdTileAccess::reset_step_modes(*solvers_[local]);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-step-finish", reset_modes);
  ++telemetry_.accepted_steps;
}

void MhdTileRuntime::step(Real dt, bool check_cfl) {
  prepare_step_request(dt, check_cfl);
  try {
    advance_step(dt);
  } catch (...) {
    restore_step_after_failure();
    throw;
  }
  finish_step();
}

Real MhdTileRuntime::divergence_b_max() {
  require_open_seeded();
  reconcile_register(0, "mhd-divb-state");
  std::vector<Real> values(solvers_.size(), Real{0});
  auto tasks = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, &values](std::size_t local, WorkerContext&) {
    values[local] = solvers_[local]->divergence_b_max();
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-divb-local", tasks);
  const Real local = *std::max_element(values.begin(), values.end());
  return static_cast<Real>(runtime_->allreduce_max(local));
}

MhdGlobalState MhdTileRuntime::gather_state() {
  require_open_seeded();
  reconcile_register(0, "mhd-gather-state");
  return collect_register(0, "mhd-gather-state-final").state;
}

std::vector<Real> MhdTileRuntime::gather_cell_component(
    std::string_view component) {
  require_open_seeded();
  const bool known = component == "rho" || component == "mx" ||
      component == "my" || component == "mz" ||
      component == "energy" || component == "bx" ||
      component == "by" || component == "bz";
  if (!known) {
    throw std::invalid_argument{
        "unknown distributed MHD cell component '" +
        std::string{component} + "'"};
  }
  reconcile_register(0, "mhd-gather-component-state");
  const std::size_t nx = topology_.global_nx();
  const std::size_t ny = topology_.global_ny();
  std::vector<Real> global(checked_product(nx, ny, "MHD cell component"),
                           Real{0});
  std::vector<int> counts(global.size(), 0);
  auto tasks = mhd_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(),
      [this, component, nx, &global, &counts](std::size_t local,
                                              WorkerContext&) {
        const auto& solver = *solvers_[local];
        const std::vector<Real> host =
            solver.state_component_to_host(component);
        const std::size_t endpoint =
            static_cast<std::size_t>(runtime_->rank()) *
                mapping_.devices_per_rank() +
            local;
        const TileExtent& tile = topology_.tile(endpoint);
        const Grid2D grid = solver.grid();
        for (int j = 0; j < grid.ny; ++j) {
          const std::size_t gy = tile.y.begin + static_cast<std::size_t>(j);
          for (int i = 0; i < grid.nx; ++i) {
            const std::size_t gx = tile.x.begin + static_cast<std::size_t>(i);
            const std::size_t index = row_major(gx, gy, nx);
            global[index] = host[grid.index(i, j)];
            counts[index] = 1;
          }
        }
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-gather-component", tasks);
  allreduce_sum_in_place(*runtime_, global, MPI_DOUBLE,
                         "MPI_Allreduce(distributed MHD component)");
  allreduce_sum_in_place(*runtime_, counts, MPI_INT,
                         "MPI_Allreduce(distributed MHD component ownership)");
  telemetry_.collective_bytes += static_cast<std::uint64_t>(
      global.size() * (sizeof(Real) + sizeof(int)));
  require_exact_coverage(counts, "cell diagnostic lattice");
  return global;
}

std::vector<MhdOwnedShard> MhdTileRuntime::local_owned_shards() {
  require_open_seeded();
  reconcile_register(0, "mhd-local-shards-state");
  std::vector<MhdOwnedShard> shards(solvers_.size());
  auto tasks = mhd_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(),
      [this, &shards](std::size_t local, WorkerContext&) {
        auto& solver = *solvers_[local];
        const Grid2D grid = solver.grid();
        const std::size_t endpoint =
            static_cast<std::size_t>(runtime_->rank()) *
                mapping_.devices_per_rank() + local;
        const TileExtent& tile = topology_.tile(endpoint);
        MhdOwnedShard shard;
        shard.endpoint = endpoint;
        shard.tile_x = tile.coordinate.x;
        shard.tile_y = tile.coordinate.y;
        shard.offset_x = tile.x.begin;
        shard.offset_y = tile.y.begin;
        shard.owned_nx = tile.x.size();
        shard.owned_ny = tile.y.size();
        const std::size_t owned = tile.cell_count();
        shard.rho.resize(owned);
        shard.mx.resize(owned);
        shard.my.resize(owned);
        shard.mz.resize(owned);
        shard.energy.resize(owned);
        shard.bx.resize(owned);
        shard.by.resize(owned);
        shard.bz.resize(owned);

        const auto copy_owned = [&](std::string_view component,
                                    std::vector<Real>& destination) {
          const std::vector<Real> padded =
              solver.state_component_to_host(component);
          for (int j = 0; j < grid.ny; ++j) {
            for (int i = 0; i < grid.nx; ++i) {
              destination[static_cast<std::size_t>(j) * tile.x.size()
                          + static_cast<std::size_t>(i)] =
                  padded[grid.index(i, j)];
            }
          }
        };
        copy_owned("rho", shard.rho);
        copy_owned("mx", shard.mx);
        copy_owned("my", shard.my);
        copy_owned("mz", shard.mz);
        copy_owned("energy", shard.energy);
        copy_owned("bx", shard.bx);
        copy_owned("by", shard.by);
        copy_owned("bz", shard.bz);
        shards[local] = std::move(shard);
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-local-shards-extract", tasks);
  ++telemetry_.local_shard_extractions;
  return shards;
}

MhdGlobalCellSums MhdTileRuntime::global_cell_sums() {
  require_open_seeded();
  const bool cylindrical = global_config_.geometry == "cylindrical";
  std::vector<Real> rho(solvers_.size(), Real{0});
  std::vector<Real> energy(solvers_.size(), Real{0});
  auto tasks = mhd_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(), [this, cylindrical, &rho, &energy]
      (std::size_t local, WorkerContext&) {
        const auto& solver = *solvers_[local];
        const Grid2D grid = solver.grid();
        const std::vector<Real> local_rho =
            solver.state_component_to_host("rho");
        const std::vector<Real> local_energy =
            solver.state_component_to_host("energy");
        Real rho_sum = Real{0};
        Real energy_sum = Real{0};
        for (int j = 0; j < grid.ny; ++j) {
          for (int i = 0; i < grid.nx; ++i) {
            const Real weight =
                cylindrical ? grid.cell_volume(i) : Real{1};
            rho_sum += weight * local_rho[grid.index(i, j)];
            energy_sum += weight * local_energy[grid.index(i, j)];
          }
        }
        rho[local] = rho_sum;
        energy[local] = energy_sum;
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-global-cell-sums", tasks);
  const Real local_rho =
      std::accumulate(rho.begin(), rho.end(), Real{0});
  const Real local_energy =
      std::accumulate(energy.begin(), energy.end(), Real{0});
  return MhdGlobalCellSums{
      static_cast<Real>(runtime_->allreduce_sum(local_rho)),
      static_cast<Real>(runtime_->allreduce_sum(local_energy))};
}

CheckpointMetadata MhdTileRuntime::checkpoint_metadata(
    std::uint64_t step, double time,
    std::string_view unit_system,
    const MhdGlobalBackground* expected_background) const {
  if (lifecycle_.poisoned) {
    throw std::logic_error{
        "distributed MHD runtime is poisoned after a failed collective mutation"};
  }
  const std::string expected_content =
      expected_background != nullptr
          ? mhd_background_content_signature(*expected_background)
          : background_content_signature_;
  CheckpointMetadata metadata{
      .schema = std::string{checkpoint_schema},
      .physics = "mhd",
      .precision = "float64",
      .geometry = global_config_.geometry,
      .unit_system = std::string{unit_system},
      .global_nx = static_cast<std::uint64_t>(topology_.global_nx()),
      .global_ny = static_cast<std::uint64_t>(topology_.global_ny()),
      .boundary_signature = mhd_boundary_signature(global_config_),
      .species_signature = {},
      .background_signature =
          mhd_background_signature(global_config_, expected_content),
      .numerics_signature = mhd_numerics_signature(global_config_),
      .step = step,
      .time = time,
  };
  validate_checkpoint_metadata(metadata);
  return metadata;
}

void MhdTileRuntime::write_checkpoint(
    const std::filesystem::path& path, std::uint64_t step, double time,
    std::string_view unit_system,
    std::span<const std::uint8_t> diagnostic_state) {
  runtime_->require_orchestration_thread();
  const bool locally_ready = !lifecycle_.closed && !lifecycle_.poisoned &&
                             lifecycle_.seeded;
  collective_require(
      *runtime_, worker_epoch_, locally_ready, "mhd-checkpoint-ready",
      lifecycle_.closed ? "distributed MHD runtime is closed"
      : lifecycle_.poisoned
          ? "distributed MHD runtime is poisoned after a failed collective mutation"
          : "distributed MHD runtime is not seeded");

  std::optional<ParallelCheckpointWriter> writer;
  try {
    const std::vector<std::uint8_t> encoded_diagnostic_state =
        collect_checkpoint_diagnostic_state(*runtime_, diagnostic_state);
    std::optional<CheckpointMetadata> metadata;
    collective_try_with_fallback(
        *runtime_, worker_epoch_, "mhd-checkpoint-metadata",
        "MHD checkpoint metadata preparation failed", -1, [&] {
      metadata.emplace(checkpoint_metadata(step, time, unit_system));
      metadata->diagnostic_state_bytes = encoded_diagnostic_state.size();
      validate_checkpoint_metadata(*metadata);
    });

    writer.emplace(*runtime_, path, std::move(*metadata));
    reconcile_register(0, "mhd-checkpoint-state");
    reconcile_background("mhd-checkpoint-background");
    std::vector<LocalMhdCheckpointTile> local_tiles;
    std::vector<WorkerTask> stage;
    collective_try_with_fallback(
        *runtime_, worker_epoch_, "mhd-checkpoint-stage-storage",
        "MHD checkpoint tile staging allocation failed", -1, [&] {
      local_tiles.resize(solvers_.size());
      stage = local_indexed_tasks(
          solvers_.size(), [this, &local_tiles]
          (std::size_t local, WorkerContext&) {
            auto& solver = *solvers_[local];
            LocalMhdCheckpointTile staged;
            const std::size_t endpoint =
                static_cast<std::size_t>(runtime_->rank()) *
                    mapping_.devices_per_rank() + local;
            staged.tile = topology_.tile(endpoint);
            staged.grid = solver.grid();
            staged.state = download_mhd_register(
                MhdTileAccess::state_register(solver, 0));
            if (global_config_.background.enabled) {
              for (auto& component : staged.background) {
                component.resize(staged.grid.storage_size());
              }
              const auto& background = MhdTileAccess::background(solver);
              background.b0x_face.copy_to_host(
                  staged.background[0].data(), staged.background[0].size());
              background.b0y_face.copy_to_host(
                  staged.background[1].data(), staged.background[1].size());
              background.b0z_cell.copy_to_host(
                  staged.background[2].data(), staged.background[2].size());
            }
            local_tiles[local] = std::move(staged);
          });
    });
    require_worker_success(*workers_, *runtime_, worker_epoch_,
                           "mhd-checkpoint-stage-local", stage);

    const auto write = [this, &writer, &local_tiles](
                           std::string_view name,
                           MhdCheckpointLattice lattice,
                           auto&& select) {
      const auto shape = checkpoint_shape(topology_, lattice);
      std::vector<DatasetHyperslab> slabs;
      std::vector<Real> packed;
      collective_try_with_fallback(
          *runtime_, worker_epoch_, "mhd-checkpoint-dataset-storage",
          "MHD checkpoint dataset packing failed", -1, [&] {
        slabs = checkpoint_slabs(
            mapping_, topology_, runtime_->rank(), lattice);
        packed = pack_local_checkpoint_values(
            std::span<const LocalMhdCheckpointTile>{local_tiles},
            lattice, slabs, std::forward<decltype(select)>(select));
      });
      writer->write_dataset(name, checkpoint_real_type(), shape, slabs,
                            packed.data(), packed.size());
    };

    for (const auto& dataset : mhd_checkpoint_datasets) {
      if (dataset.field == MhdCheckpointField::background &&
          !global_config_.background.enabled) {
        continue;
      }
      write(dataset.name, dataset.lattice,
            [dataset](const LocalMhdCheckpointTile& tile)
                -> const std::vector<Real>& {
        if (dataset.field == MhdCheckpointField::state) {
          return tile.state[dataset.component];
        }
        return tile.background[dataset.component];
      });
    }
    writer->write_diagnostic_state(encoded_diagnostic_state);
    writer->commit();
  } catch (...) {
    lifecycle_.poisoned = true;
    if (writer) {
      try {
        writer->close();
      } catch (...) {
      }
    }
    throw;
  }
}

CheckpointMetadata MhdTileRuntime::copy_restart_metadata(
    ParallelCheckpointReader& reader) {
  std::optional<CheckpointMetadata> stored;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "mhd-restart-metadata-copy",
      "MHD checkpoint metadata copy failed", -1, [&] {
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
    if (std::exchange(inject_mhd_checkpoint_metadata_copy_failure, false)) {
      throw std::bad_alloc{};
    }
#endif
    stored.emplace(reader.metadata());
  });
  return std::move(*stored);
}

void MhdTileRuntime::require_compatible_restart(
    const CheckpointMetadata& stored, std::string_view unit_system,
    const MhdGlobalBackground* expected_background) {
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "mhd-restart-compatibility",
      "MHD checkpoint is incompatible with this runtime", -1, [&] {
    const CheckpointMetadata requested = checkpoint_metadata(
        stored.step, stored.time, unit_system, expected_background);
    validate_restart_compatibility(stored, requested);
  });
}

MhdTileRuntime::RestartPayload MhdTileRuntime::stage_restart_payload() {
  RestartPayload payload;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "mhd-restart-staging-storage",
      "MHD restart tile staging allocation failed", -1, [&] {
    payload.local_tiles.resize(solvers_.size());
    for (std::size_t local = 0; local < solvers_.size(); ++local) {
      const std::size_t endpoint =
          static_cast<std::size_t>(runtime_->rank()) *
              mapping_.devices_per_rank() + local;
      auto& staged = payload.local_tiles[local];
      staged.tile = topology_.tile(endpoint);
      staged.grid = tile_config(endpoint).grid;
      for (auto& component : staged.state) {
        component.assign(staged.grid.storage_size(), Real{0});
      }
      if (global_config_.background.enabled) {
        for (auto& component : staged.background) {
          component.assign(staged.grid.storage_size(), Real{0});
        }
      }
    }
  });
  return payload;
}

void MhdTileRuntime::read_restart_payload(
    ParallelCheckpointReader& reader, RestartPayload& payload) {
  const auto read = [this, &reader, &payload](
                        std::string_view name,
                        MhdCheckpointLattice lattice, auto&& select) {
    read_local_checkpoint_lattice(
        *runtime_, worker_epoch_, mapping_, topology_, reader,
        std::span<RestartPayload::Tile>{payload.local_tiles}, name, lattice,
        std::forward<decltype(select)>(select));
  };
  for (const auto& dataset : mhd_checkpoint_datasets) {
    if (dataset.field == MhdCheckpointField::background &&
        !global_config_.background.enabled) {
      continue;
    }
    read(dataset.name, dataset.lattice,
         [dataset](RestartPayload::Tile& tile) -> std::vector<Real>& {
      if (dataset.field == MhdCheckpointField::state) {
        return tile.state[dataset.component];
      }
      return tile.background[dataset.component];
    });
  }
  payload.diagnostic_state = reader.read_diagnostic_state();
}

void MhdTileRuntime::validate_restart_payload(
    const CheckpointMetadata& stored, RestartPayload& payload) {
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "mhd-restart-data-validation",
      "MHD checkpoint data validation failed", -1, [&] {
    bool locally_finite = true;
    for (const auto& local : payload.local_tiles) {
      for (const auto& component : local.state) {
        locally_finite = locally_finite && std::all_of(
            component.begin(), component.end(),
            [](Real value) { return std::isfinite(value); });
      }
    }
    if (global_config_.background.enabled) {
      for (const auto& local : payload.local_tiles) {
        for (const auto& component : local.background) {
          locally_finite = locally_finite && std::all_of(
              component.begin(), component.end(),
              [](Real value) { return std::isfinite(value); });
        }
      }
    }
    if (!runtime_->allreduce_all(locally_finite)) {
      throw std::runtime_error{
          "MHD checkpoint contains a non-finite state or background value"};
    }
    verify_local_periodic_duplicates(
        *runtime_, global_config_, topology_,
        std::span<const RestartPayload::Tile>{payload.local_tiles},
        [](const RestartPayload::Tile& tile) -> const std::vector<Real>& {
          return tile.state[5];
        },
        [](const RestartPayload::Tile& tile) -> const std::vector<Real>& {
          return tile.state[6];
        },
        "MHD checkpoint state");
    if (global_config_.background.enabled) {
      payload.background_content_signature =
          checkpoint_background_content_signature(stored.background_signature);
      verify_local_periodic_duplicates(
          *runtime_, global_config_, topology_,
          std::span<const RestartPayload::Tile>{payload.local_tiles},
          [](const RestartPayload::Tile& tile) -> const std::vector<Real>& {
            return tile.background[0];
          },
          [](const RestartPayload::Tile& tile) -> const std::vector<Real>& {
            return tile.background[1];
          },
          "MHD checkpoint background");
    }
  });
}

void MhdTileRuntime::commit_restart_payload(RestartPayload& payload,
                                            bool& mutation_started) {
  std::vector<WorkerTask> restore;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "mhd-restart-restore-storage",
      "MHD restart worker task allocation failed", -1, [&] {
    restore = local_indexed_tasks(
        solvers_.size(), [this, &payload](std::size_t local, WorkerContext&) {
          auto& solver = *solvers_[local];
          const auto& staged = payload.local_tiles[local];
          solver.seed_state("rho", staged.state[0]);
          solver.seed_state("mx", staged.state[1]);
          solver.seed_state("my", staged.state[2]);
          solver.seed_state("mz", staged.state[3]);
          solver.seed_state("energy", staged.state[4]);
          solver.seed_state("bx_face", staged.state[5]);
          solver.seed_state("by_face", staged.state[6]);
          solver.seed_state("bz_cell", staged.state[7]);
          if (global_config_.background.enabled) {
            solver.seed_background("b0x_face", staged.background[0]);
            solver.seed_background("b0y_face", staged.background[1]);
            solver.seed_background("b0z_cell", staged.background[2]);
          }
          backend::device_synchronize(nullptr);
        });
  });
  mutation_started = true;
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-restart-stage-tiles", restore);
  lifecycle_.seeded = true;
  reconcile_register(0, "mhd-restart-state");
  reconcile_background("mhd-restart-background");
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
  collective_require(
      *runtime_, worker_epoch_, !inject_restart_post_reconcile_failure_,
      "mhd-restart-post-reconcile",
      "injected post-reconcile restart failure");
#endif
  background_content_signature_ =
      std::move(payload.background_content_signature);
}

void MhdTileRuntime::publish_restart_payload(
    RestartPayload& payload,
    std::vector<std::vector<std::uint8_t>>* diagnostic_state) {
  // A committed checkpoint was produced only after a solver-owned CT/RK
  // update and halo reconciliation, and its complete canonical field has just
  // passed schema, compatibility, duplicate, and state validation. Restore
  // that trusted provenance after component-wise seeding cleared it. Ordinary
  // external seed() calls deliberately remain on the strict predicate.
  std::vector<WorkerTask> provenance;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "mhd-restart-provenance-storage",
      "MHD restart provenance task allocation failed", -1, [&] {
    provenance = local_indexed_tasks(
        solvers_.size(), [this](std::size_t local, WorkerContext&) {
          MhdTileAccess::mark_restart_state_trusted(*solvers_[local]);
        });
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-restart-provenance", provenance);
  if (diagnostic_state != nullptr) {
    *diagnostic_state = std::move(payload.diagnostic_state);
  }
}

CheckpointMetadata MhdTileRuntime::restart_from_checkpoint(
    const std::filesystem::path& path, std::string_view unit_system,
    const MhdGlobalBackground* expected_background,
    std::vector<std::vector<std::uint8_t>>* diagnostic_state) {
  runtime_->require_orchestration_thread();
  const bool locally_ready = !lifecycle_.closed && !lifecycle_.poisoned &&
                             !lifecycle_.seeded;
  collective_require(
      *runtime_, worker_epoch_, locally_ready, "mhd-restart-ready",
      lifecycle_.closed ? "distributed MHD runtime is closed"
      : lifecycle_.poisoned
          ? "distributed MHD runtime is poisoned after a failed collective mutation"
          : "distributed MHD runtime is already seeded");

  ParallelCheckpointReader reader{*runtime_, path};
  std::optional<CheckpointMetadata> stored;
  RestartPayload payload;
  bool mutation_started = false;
  try {
    stored.emplace(copy_restart_metadata(reader));
    require_compatible_restart(*stored, unit_system, expected_background);
    payload = stage_restart_payload();
    read_restart_payload(reader, payload);
    validate_restart_payload(*stored, payload);
    // No solver state is mutated until the collective reader is closed.
    reader.close();
    commit_restart_payload(payload, mutation_started);
    publish_restart_payload(payload, diagnostic_state);
  } catch (const std::exception& error) {
    try {
      reader.close();
    } catch (...) {
    }
    if (mutation_started) {
      poison_collectively("mhd-restart-failure", error.what());
    }
    throw;
  } catch (...) {
    try {
      reader.close();
    } catch (...) {
    }
    if (mutation_started) {
      poison_collectively("mhd-restart-failure",
                          "non-standard MHD restart failure");
    }
    throw;
  }
  return std::move(*stored);
}

void MhdTileRuntime::close() {
  if (lifecycle_.closed) return;
  runtime_->require_orchestration_thread();
  transport_->close();
  telemetry_.transport = transport_->telemetry();
  auto tasks = mhd_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this](std::size_t local, WorkerContext&) {
    register_halos_[local].reset();
    solvers_[local].reset();
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "mhd-runtime-close", tasks);
  workers_->close();
  lifecycle_.closed = true;
}

}  // namespace quasar::distributed
