#include "quasar/distributed/pic_runtime.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/physics/pic/diagnostics.hpp"
#include "quasar/physics/pic/kernels.hpp"

#include "collective_helpers.hpp"
#include "mpi_runtime_native.hpp"
#include "pic_tile_access.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace quasar::distributed {
namespace {

struct ComponentExtent {
  std::size_t nx{0};
  std::size_t ny{0};
  bool face_x{false};
  bool face_y{false};
};

enum class FieldComponent : std::size_t { ex, ey, ez, bx, by, bz };
enum class SourceComponent : std::size_t { jx, jy, jz, charge };

bool is_cylindrical(std::string_view geometry) {
  return geometry == "cylindrical";
}

bool periodic_x(const pic::EmPicConfig& config) {
  return config.boundary.field[0] == "periodic" &&
         config.boundary.field[1] == "periodic" &&
         config.boundary.particle[0] == "periodic" &&
         config.boundary.particle[1] == "periodic";
}

bool periodic_y(const pic::EmPicConfig& config) {
  return config.boundary.field[2] == "periodic" &&
         config.boundary.field[3] == "periodic" &&
         config.boundary.particle[2] == "periodic" &&
         config.boundary.particle[3] == "periodic";
}

int current_boundary_mode(std::string_view name) {
  auto field_boundary =
      Registry<boundary::IFieldBoundary>::instance().create(name);
  const int mode = field_boundary->ghost_continuation_mode();
  if (mode < 0 || mode > 4 ||
      field_boundary->is_internal_cut() != (mode == 4)) {
    throw std::invalid_argument{
        "registered PIC field boundary has an invalid compact-current "
        "continuation: " + std::string{name}};
  }
  return mode;
}

std::array<ComponentExtent, 6> field_extents(std::size_t nx,
                                              std::size_t ny,
                                              bool cylindrical) {
  if (!cylindrical) {
    return {{{nx + 1, ny, true, false},
             {nx, ny + 1, false, true},
             {nx, ny, false, false},
             {nx, ny + 1, false, true},
             {nx + 1, ny, true, false},
             {nx + 1, ny + 1, true, true}}};
  }
  return {{{nx + 1, ny, true, false},
           {nx, ny + 1, false, true},
           {nx + 1, ny, true, false},
           {nx + 1, ny + 1, true, true},
           {nx, ny, false, false},
           {nx + 1, ny + 1, true, true}}};
}

std::array<ComponentExtent, 3> source_extents(std::size_t nx,
                                               std::size_t ny,
                                               bool cylindrical) {
  const auto fields = field_extents(nx, ny, cylindrical);
  return {fields[static_cast<std::size_t>(FieldComponent::ex)],
          fields[static_cast<std::size_t>(FieldComponent::ey)],
          fields[static_cast<std::size_t>(FieldComponent::ez)]};
}

std::vector<Real>& field_vector(PicGlobalFields& fields,
                                FieldComponent component) {
  switch (component) {
    case FieldComponent::ex: return fields.ex;
    case FieldComponent::ey: return fields.ey;
    case FieldComponent::ez: return fields.ez;
    case FieldComponent::bx: return fields.bx;
    case FieldComponent::by: return fields.by;
    case FieldComponent::bz: return fields.bz;
  }
  throw std::logic_error{"invalid PIC field component"};
}

const std::vector<Real>& field_vector(const PicGlobalFields& fields,
                                      FieldComponent component) {
  return field_vector(const_cast<PicGlobalFields&>(fields), component);
}

PicOwnedArray& owned_field_array(PicOwnedFields& fields,
                                 FieldComponent component) {
  switch (component) {
    case FieldComponent::ex: return fields.ex;
    case FieldComponent::ey: return fields.ey;
    case FieldComponent::ez: return fields.ez;
    case FieldComponent::bx: return fields.bx;
    case FieldComponent::by: return fields.by;
    case FieldComponent::bz: return fields.bz;
  }
  throw std::logic_error{"invalid PIC owned field component"};
}

backend::DeviceBuffer<Real>& field_buffer(YeeField2D<Real>& fields,
                                           FieldComponent component) {
  switch (component) {
    case FieldComponent::ex: return fields.ex;
    case FieldComponent::ey: return fields.ey;
    case FieldComponent::ez: return fields.ez;
    case FieldComponent::bx: return fields.bx;
    case FieldComponent::by: return fields.by;
    case FieldComponent::bz: return fields.bz;
  }
  throw std::logic_error{"invalid PIC field component"};
}

backend::DeviceBuffer<Real>& magnetic_buffer(BField2D<Real>& fields,
                                              FieldComponent component) {
  switch (component) {
    case FieldComponent::bx: return fields.bx;
    case FieldComponent::by: return fields.by;
    case FieldComponent::bz: return fields.bz;
    default: break;
  }
  throw std::logic_error{"requested electric component from magnetic snapshot"};
}

std::vector<Real>& source_vector(PicGlobalSources& sources,
                                 SourceComponent component) {
  switch (component) {
    case SourceComponent::jx: return sources.jx;
    case SourceComponent::jy: return sources.jy;
    case SourceComponent::jz: return sources.jz;
    case SourceComponent::charge: return sources.charge;
  }
  throw std::logic_error{"invalid PIC source component"};
}

const std::vector<Real>& source_vector(const PicGlobalSources& sources,
                                       SourceComponent component) {
  return source_vector(const_cast<PicGlobalSources&>(sources), component);
}

#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
bool inject_next_pic_worker_task_allocation_failure = false;
bool inject_pic_seed_post_mutation_failure = false;
bool inject_pic_checkpoint_metadata_copy_failure = false;
#endif

void prepare_pic_worker_task_storage() {
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
  if (std::exchange(inject_next_pic_worker_task_allocation_failure, false)) {
    throw std::bad_alloc{};
  }
#endif
}

constexpr std::string_view pic_worker_task_storage_phase =
    "pic-worker-task-storage";
constexpr std::string_view pic_worker_task_storage_failure =
    "distributed PIC worker task allocation failed";

template <class Function>
std::vector<WorkerTask> pic_worker_tasks(
    MpiRuntime& runtime, std::uint64_t& epoch, std::size_t count,
    Function&& function) {
  return collectively_indexed_tasks(
      runtime, epoch, count, pic_worker_task_storage_phase,
      pic_worker_task_storage_failure, prepare_pic_worker_task_storage,
      std::forward<Function>(function));
}

bool map_field_coordinate(long long coordinate, std::size_t cells,
                          bool staggered, bool periodic,
                          std::size_t& mapped) {
  const long long high = static_cast<long long>(cells) +
                         (staggered ? 0LL : -1LL);
  if (coordinate >= 0 && coordinate <= high) {
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

// Deposited guards are additive.  On a periodic dimension both the cell guard
// and a normal-current high face represent the unique low coordinate.
bool map_source_coordinate(long long coordinate, std::size_t cells,
                           bool staggered, bool periodic,
                           std::size_t& mapped) {
  if (periodic) {
    if (cells == 0) return false;
    const long long period = static_cast<long long>(cells);
    long long value = coordinate % period;
    if (value < 0) value += period;
    mapped = static_cast<std::size_t>(value);
    return true;
  }
  const long long high = static_cast<long long>(cells) +
                         (staggered ? 0LL : -1LL);
  if (coordinate < 0 || coordinate > high) return false;
  mapped = static_cast<std::size_t>(coordinate);
  return true;
}

struct CanonicalComponentPoint {
  std::size_t x{0};
  std::size_t y{0};
  std::size_t owner{0};
};

CanonicalComponentPoint canonical_component_point(
    const VirtualTopology& topology, std::size_t x, std::size_t y,
    const ComponentExtent& extent, bool periodic_x_axis,
    bool periodic_y_axis) {
  const std::size_t nx = topology.global_nx();
  const std::size_t ny = topology.global_ny();
  if (extent.face_x && periodic_x_axis && x == nx) x = 0;
  if (extent.face_y && periodic_y_axis && y == ny) y = 0;
  const std::size_t cell_x = extent.face_x
      ? (x == 0 ? 0 : std::min(x - 1, nx - 1))
      : x;
  const std::size_t cell_y = extent.face_y
      ? (y == 0 ? 0 : std::min(y - 1, ny - 1))
      : y;
  return {x, y, topology.owner_of_cell(cell_x, cell_y)};
}

Real wrap_periodic(Real value, Real low, Real length) {
  Real reduced = std::fmod(value - low, length);
  if (reduced < Real{0}) reduced += length;
  Real wrapped = low + reduced;
  const Real high = low + length;
  if (!(wrapped < high)) wrapped = low;
  return wrapped;
}

void append_particle(pic::ParticleSpecies::HostSnapshot& destination,
                     const pic::ParticleSpecies::HostSnapshot& source,
                     std::size_t index) {
  destination.x.push_back(source.x[index]);
  destination.y.push_back(source.y[index]);
  destination.x_prev.push_back(source.x_prev[index]);
  destination.y_prev.push_back(source.y_prev[index]);
  destination.vx.push_back(source.vx[index]);
  destination.vy.push_back(source.vy[index]);
  destination.vz.push_back(source.vz[index]);
  destination.vphi_deposit.push_back(source.vphi_deposit[index]);
  destination.weight.push_back(source.weight[index]);
  destination.alive.push_back(source.alive[index]);
  destination.id.push_back(source.id[index]);
}

void sort_particles(pic::ParticleSpecies::HostSnapshot& snapshot) {
  std::vector<std::size_t> order(snapshot.id.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](std::size_t left,
                                                    std::size_t right) {
    return snapshot.id[left] < snapshot.id[right];
  });
  pic::ParticleSpecies::HostSnapshot sorted;
  for (const std::size_t index : order) append_particle(sorted, snapshot, index);
  snapshot = std::move(sorted);
}

struct ParticleRecord {
  Real x{}, y{}, x_prev{}, y_prev{}, vx{}, vy{}, vz{}, vphi_deposit{}, weight{};
  std::uint64_t id{0};
  std::uint64_t source_endpoint{0};
  std::uint8_t alive{0};
};

struct ParticleMigrationRecord {
  ParticleRecord particle{};
  std::uint64_t destination_endpoint{0};
  std::uint64_t species{0};
};

static_assert(std::is_trivially_copyable_v<ParticleRecord>);
static_assert(std::is_trivially_copyable_v<ParticleMigrationRecord>);

template <class Record>
void append_wire_record(std::vector<std::byte>& destination,
                        const Record& record) {
  static_assert(std::is_trivially_copyable_v<Record>);
  const std::size_t offset = destination.size();
  destination.resize(offset + sizeof(Record));
  std::memcpy(destination.data() + offset, &record, sizeof(Record));
}

template <class Record, class Callback>
void for_each_wire_record(std::span<const std::byte> payload,
                          std::string_view description,
                          Callback&& callback) {
  static_assert(std::is_trivially_copyable_v<Record>);
  if (payload.size() % sizeof(Record) != 0) {
    throw std::runtime_error{std::string{description} +
                             " payload has a truncated record"};
  }
  for (std::size_t offset = 0; offset < payload.size();
       offset += sizeof(Record)) {
    Record record;
    std::memcpy(&record, payload.data() + offset, sizeof(Record));
    callback(record);
  }
}

std::vector<ParticleRecord> allgather_records(
    MpiRuntime& runtime, std::span<const ParticleRecord> local,
    std::uint64_t& collective_epoch) {
  runtime.require_orchestration_thread();
  const std::uint64_t local_count = local.size();
  std::vector<std::uint64_t> counts;
  collective_try_with_fallback(
      runtime, collective_epoch, "pic-particle-gather-count-storage",
      "distributed PIC particle count allocation failed", -1, [&] {
    counts.resize(static_cast<std::size_t>(runtime.size()));
  });
  check_mpi(MPI_Allgather(&local_count, 1, MPI_UINT64_T, counts.data(), 1,
                          MPI_UINT64_T,
                          detail::MpiRuntimeNativeAccess::world(runtime)),
            "MPI_Allgather(distributed PIC particle counts)");

  bool shape_valid = true;
  std::uint64_t total_records = 0;
  for (const std::uint64_t count : counts) {
    if (count > std::numeric_limits<std::uint64_t>::max() - total_records) {
      shape_valid = false;
      break;
    }
    total_records += count;
  }
  if (shape_valid) {
    const std::uint64_t host_limit =
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() /
                                   sizeof(ParticleRecord));
    shape_valid = total_records <= host_limit;
  }
  collective_require(
      runtime, collective_epoch, shape_valid, "pic-particle-gather-shape",
      "distributed PIC particle exchange exceeds host size");

  std::vector<std::size_t> offsets;
  std::vector<ParticleRecord> result;
  collective_try_with_fallback(
      runtime, collective_epoch, "pic-particle-gather-allocation",
      "distributed PIC particle gather allocation failed", -1, [&] {
    offsets.resize(counts.size());
    std::size_t offset = 0;
    for (std::size_t rank = 0; rank < counts.size(); ++rank) {
      offsets[rank] = offset;
      offset += static_cast<std::size_t>(counts[rank]);
    }
    result.resize(static_cast<std::size_t>(total_records));
  });

  const std::size_t local_offset =
      offsets[static_cast<std::size_t>(runtime.rank())];
  std::copy(local.begin(), local.end(), result.begin() + local_offset);

  // MPI-3 collectives accept only int element counts.  Broadcasting one
  // rank's byte range at a time, in bounded chunks, removes both the
  // per-rank count and aggregate displacement limits of MPI_Allgatherv while
  // preserving the deterministic rank-major record order.
  constexpr std::uint64_t max_chunk_bytes = 64ULL * 1024ULL * 1024ULL;
  auto* bytes = reinterpret_cast<std::byte*>(result.data());
  for (std::size_t root = 0; root < counts.size(); ++root) {
    const std::uint64_t rank_bytes =
        counts[root] * static_cast<std::uint64_t>(sizeof(ParticleRecord));
    const std::uint64_t byte_offset =
        static_cast<std::uint64_t>(offsets[root]) *
        static_cast<std::uint64_t>(sizeof(ParticleRecord));
    for (std::uint64_t sent = 0; sent < rank_bytes;) {
      const std::uint64_t chunk =
          std::min(max_chunk_bytes, rank_bytes - sent);
      check_mpi(MPI_Bcast(bytes + static_cast<std::size_t>(byte_offset + sent),
                          static_cast<int>(chunk), MPI_BYTE,
                          static_cast<int>(root),
                          detail::MpiRuntimeNativeAccess::world(runtime)),
                "MPI_Bcast(distributed PIC particle chunk)");
      sent += chunk;
    }
  }
  return result;
}

ParticleRecord make_record(const pic::ParticleSpecies::HostSnapshot& snapshot,
                           std::size_t index, std::size_t endpoint) {
  return {snapshot.x[index], snapshot.y[index], snapshot.x_prev[index],
          snapshot.y_prev[index], snapshot.vx[index], snapshot.vy[index],
          snapshot.vz[index], snapshot.vphi_deposit[index],
          snapshot.weight[index], snapshot.id[index], endpoint,
          snapshot.alive[index]};
}

void append_record(pic::ParticleSpecies::HostSnapshot& snapshot,
                   const ParticleRecord& record) {
  snapshot.x.push_back(record.x);
  snapshot.y.push_back(record.y);
  snapshot.x_prev.push_back(record.x_prev);
  snapshot.y_prev.push_back(record.y_prev);
  snapshot.vx.push_back(record.vx);
  snapshot.vy.push_back(record.vy);
  snapshot.vz.push_back(record.vz);
  snapshot.vphi_deposit.push_back(record.vphi_deposit);
  snapshot.weight.push_back(record.weight);
  snapshot.alive.push_back(record.alive);
  snapshot.id.push_back(record.id);
}

static_assert(std::is_same_v<Real, double>,
              "distributed PIC collectives currently require double Real");

CheckpointValueType pic_checkpoint_real_type() noexcept {
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
  output << key << "=0x" << std::hex
         << std::bit_cast<std::uint64_t>(value) << std::dec << ';';
}

std::string pic_boundary_signature(const pic::EmPicConfig& config) {
  std::ostringstream output;
  for (std::size_t side = 0; side < config.boundary.field.size(); ++side) {
    append_signature_text(output, "field" + std::to_string(side),
                          config.boundary.field[side]);
    append_signature_text(output, "particle" + std::to_string(side),
                          config.boundary.particle[side]);
  }
  return output.str();
}

std::string pic_background_signature(const pic::EmPicConfig& config) {
  std::ostringstream output;
  append_signature_integer(output, "neutralizing",
                           config.neutralizing_background ? 1 : 0);
  append_signature_text(output, "external_field",
                        config.external_field_signature);
  return output.str();
}

std::string pic_numerics_signature(const pic::EmPicConfig& config) {
  std::ostringstream output;
  append_signature_real(output, "lx", config.grid.lx);
  append_signature_real(output, "ly", config.grid.ly);
  append_signature_real(output, "origin_x", config.grid.origin_x);
  append_signature_real(output, "origin_y", config.grid.origin_y);
  append_signature_integer(output, "fdtd_order",
                           static_cast<std::uint64_t>(config.fdtd_order));
  append_signature_text(output, "shape", config.shape);
  append_signature_text(output, "plane", config.plane);
  append_signature_real(output, "normalization_n_ref",
                        config.normalization.n_ref);
  append_signature_real(output, "normalization_q_ref",
                        config.normalization.q_ref);
  append_signature_real(output, "normalization_m_ref",
                        config.normalization.m_ref);
  append_signature_real(output, "normalization_omega_p_ref",
                        config.normalization.omega_p_ref);
  append_signature_integer(output, "filter_count", config.filters.size());
  for (std::size_t index = 0; index < config.filters.size(); ++index) {
    append_signature_text(output, "filter_name" + std::to_string(index),
                          config.filters[index].name);
    append_signature_integer(
        output, "filter_passes" + std::to_string(index),
        static_cast<std::uint64_t>(config.filters[index].passes));
  }
  append_signature_text(output, "timestep", config.timestep_signature);
  return output.str();
}

std::string pic_species_signature(
    std::span<const pic::SpeciesConfig> species) {
  std::ostringstream output;
  append_signature_integer(output, "count", species.size());
  for (std::size_t index = 0; index < species.size(); ++index) {
    append_signature_text(output, "name" + std::to_string(index),
                          species[index].name);
    append_signature_real(output, "charge" + std::to_string(index),
                          species[index].charge);
    append_signature_real(output, "mass" + std::to_string(index),
                          species[index].mass);
  }
  return output.str();
}

std::array<std::uint64_t, 2> pic_checkpoint_shape(
    const ComponentExtent& extent) {
  return {static_cast<std::uint64_t>(extent.ny),
          static_cast<std::uint64_t>(extent.nx)};
}

std::vector<DatasetHyperslab> pic_checkpoint_slabs(
    const EndpointMapping& mapping, const VirtualTopology& topology,
    int rank, const ComponentExtent& extent) {
  std::vector<DatasetHyperslab> slabs;
  slabs.reserve(mapping.devices_per_rank());
  for (const auto& endpoint : mapping.endpoints_for_rank(rank)) {
    const TileExtent& tile = topology.tile(endpoint.index);
    const std::size_t skip_x =
        extent.face_x && tile.coordinate.x != 0 ? 1 : 0;
    const std::size_t skip_y =
        extent.face_y && tile.coordinate.y != 0 ? 1 : 0;
    DatasetHyperslab slab{
        .offset = {static_cast<std::uint64_t>(tile.y.begin + skip_y),
                   static_cast<std::uint64_t>(tile.x.begin + skip_x)},
        .count = {static_cast<std::uint64_t>(
                      tile.y.size() + (extent.face_y ? 1 : 0) - skip_y),
                  static_cast<std::uint64_t>(
                      tile.x.size() + (extent.face_x ? 1 : 0) - skip_x)},
    };
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

std::size_t pic_slab_elements(const DatasetHyperslab& slab) {
  if (slab.count.size() != 2) {
    throw std::logic_error{"PIC checkpoint slab must be two-dimensional"};
  }
  return checked_product(static_cast<std::size_t>(slab.count[0]),
                         static_cast<std::size_t>(slab.count[1]),
                         "PIC checkpoint hyperslab");
}

std::vector<DatasetHyperslab> balanced_1d_slabs(
    std::size_t elements, int rank, int size) {
  const std::size_t ranks = static_cast<std::size_t>(size);
  const std::size_t index = static_cast<std::size_t>(rank);
  const std::size_t quotient = elements / ranks;
  const std::size_t remainder = elements % ranks;
  const std::size_t begin =
      index * quotient + std::min(index, remainder);
  const std::size_t end =
      begin + quotient + (index < remainder ? 1u : 0u);
  if (begin == end) return {};
  return {DatasetHyperslab{
      .offset = {static_cast<std::uint64_t>(begin)},
      .count = {static_cast<std::uint64_t>(end - begin)},
  }};
}

std::vector<DatasetHyperslab> rank_zero_1d_slabs(
    std::size_t elements, int rank) {
  if (rank != 0 || elements == 0) return {};
  return {DatasetHyperslab{
      .offset = {0},
      .count = {static_cast<std::uint64_t>(elements)},
  }};
}

template <class T>
void write_pic_checkpoint_1d(
    ParallelCheckpointWriter& writer, MpiRuntime& runtime,
    std::uint64_t& collective_epoch, std::string_view name,
    CheckpointValueType value_type, std::span<const T> values,
    bool balanced) {
  if (values.empty()) return;
  const std::array<std::uint64_t, 1> shape{
      static_cast<std::uint64_t>(values.size())};
  std::vector<DatasetHyperslab> slabs;
  std::size_t begin = 0;
  std::size_t count = 0;
  collective_try_with_fallback(
      runtime, collective_epoch, "pic-checkpoint-vector-slab-storage",
      "PIC checkpoint vector slab allocation failed", -1, [&] {
    slabs = balanced
        ? balanced_1d_slabs(values.size(), runtime.rank(), runtime.size())
        : rank_zero_1d_slabs(values.size(), runtime.rank());
    if (!slabs.empty()) {
      begin = static_cast<std::size_t>(slabs.front().offset.front());
      count = static_cast<std::size_t>(slabs.front().count.front());
    }
  });
  const T* data = count == 0 ? values.data() : values.data() + begin;
  writer.write_dataset(name, value_type, shape, slabs, data, count);
}

template <class T>
void write_pic_checkpoint_local_1d(
    ParallelCheckpointWriter& writer, MpiRuntime& runtime,
    std::uint64_t& collective_epoch, std::string_view name,
    CheckpointValueType value_type, std::uint64_t global_elements,
    std::uint64_t offset, std::span<const T> values) {
  if (global_elements == 0) return;
  std::vector<DatasetHyperslab> slabs;
  collective_try_with_fallback(
      runtime, collective_epoch,
      "pic-checkpoint-local-vector-slab-storage",
      "PIC checkpoint local vector slab allocation failed", -1, [&] {
    if (!values.empty()) {
      slabs.push_back(DatasetHyperslab{
          .offset = {offset},
          .count = {static_cast<std::uint64_t>(values.size())},
      });
    }
  });
  const std::array<std::uint64_t, 1> shape{global_elements};
  writer.write_dataset(name, value_type, shape, slabs, values.data(),
                       values.size());
}

template <class T>
std::vector<T> read_pic_checkpoint_1d(
    ParallelCheckpointReader& reader, MpiRuntime& runtime,
    std::string_view name, CheckpointValueType value_type,
    std::size_t elements, MPI_Datatype mpi_type, bool balanced,
    std::uint64_t& collective_epoch) {
  if (elements == 0) return {};
  const std::array<std::uint64_t, 1> shape{
      static_cast<std::uint64_t>(elements)};
  std::vector<DatasetHyperslab> slabs;
  std::size_t begin = 0;
  std::size_t local_count = 0;
  std::vector<T> local;
  std::vector<T> global;
  collective_try_with_fallback(
      runtime, collective_epoch, "pic-restart-vector-storage",
      "PIC restart vector allocation failed", -1, [&] {
    slabs = balanced
        ? balanced_1d_slabs(elements, runtime.rank(), runtime.size())
        : rank_zero_1d_slabs(elements, runtime.rank());
    if (!slabs.empty()) {
      begin = static_cast<std::size_t>(slabs.front().offset.front());
      local_count = static_cast<std::size_t>(slabs.front().count.front());
    }
    local.resize(local_count);
    global.assign(elements, T{});
  });
  reader.read_dataset(name, value_type, shape, slabs, local.data(),
                      local.size());
  if (!local.empty()) {
    std::copy(local.begin(), local.end(),
              global.begin() + static_cast<std::ptrdiff_t>(begin));
  }
  allreduce_sum_in_place(runtime, global, mpi_type,
                         "MPI_Allreduce(PIC restart vector)");
  return global;
}

template <class T>
std::vector<T> read_pic_checkpoint_chunk(
    ParallelCheckpointReader& reader, MpiRuntime& runtime,
    std::string_view name, CheckpointValueType value_type,
    std::size_t elements, std::uint64_t& collective_epoch) {
  if (elements == 0) return {};
  const std::array<std::uint64_t, 1> shape{
      static_cast<std::uint64_t>(elements)};
  std::vector<DatasetHyperslab> slabs;
  std::size_t local_count = 0;
  std::vector<T> local;
  collective_try_with_fallback(
      runtime, collective_epoch, "pic-restart-chunk-allocation",
      "checkpoint particle read allocation failed", -1, [&] {
    slabs = balanced_1d_slabs(elements, runtime.rank(), runtime.size());
    if (!slabs.empty()) {
      local_count = static_cast<std::size_t>(slabs.front().count.front());
    }
    local.resize(local_count);
  });
  reader.read_dataset(name, value_type, shape, slabs, local.data(),
                      local.size());
  return local;
}

unsigned int global_outflow_corner_mask(const pic::EmPicConfig& config) {
  const auto& boundary = config.boundary.field;
  unsigned int mask = 0;
  if (boundary[0] == "outflow" && boundary[2] == "outflow") mask |= 1u;
  if (boundary[1] == "outflow" && boundary[2] == "outflow") mask |= 2u;
  if (boundary[0] == "outflow" && boundary[3] == "outflow") mask |= 4u;
  if (boundary[1] == "outflow" && boundary[3] == "outflow") mask |= 8u;
  return mask;
}

std::size_t mur_global_stride(const VirtualTopology& topology, int side) {
  return side < 2 ? topology.global_ny() + 1
                  : topology.global_nx() + 1;
}

constexpr std::array<std::string_view, 6> pic_field_dataset_names{
    "pic/fields/ex", "pic/fields/ey", "pic/fields/ez",
    "pic/fields/bx", "pic/fields/by", "pic/fields/bz"};
constexpr std::array<std::string_view, 6> pic_external_field_dataset_names{
    "pic/external_fields/ex", "pic/external_fields/ey",
    "pic/external_fields/ez", "pic/external_fields/bx",
    "pic/external_fields/by", "pic/external_fields/bz"};
constexpr std::array<std::string_view, 3> pic_previous_b_dataset_names{
    "pic/previous_b/bx", "pic/previous_b/by", "pic/previous_b/bz"};
constexpr std::array<std::string_view, 4> pic_source_dataset_names{
    "pic/sources/jx", "pic/sources/jy", "pic/sources/jz",
    "pic/sources/charge"};
constexpr std::array<std::string_view, 4> pic_mur_dataset_names{
    "pic/boundary/mur_side_0", "pic/boundary/mur_side_1",
    "pic/boundary/mur_side_2", "pic/boundary/mur_side_3"};
constexpr std::array<std::string_view, 11> pic_particle_dataset_suffixes{
    "x", "y", "x_prev", "y_prev", "vx", "vy", "vz",
    "vphi_deposit", "weight", "alive", "id"};

}  // namespace

using PicFixedExchangeEntry = pic::PicDeviceHaloEntry;

struct PicDirectedFixedExchange {
  static constexpr std::size_t kind_count = 4;
  std::array<std::vector<PicFixedExchangeEntry>, kind_count> entries{};
};

struct PicFixedExchangePair {
  std::size_t first_endpoint{0};
  std::size_t second_endpoint{0};
  PicDirectedFixedExchange first_to_second{};
  PicDirectedFixedExchange second_to_first{};

  [[nodiscard]] std::size_t capacity(std::size_t kind) const {
    return std::max(first_to_second.entries.at(kind).size(),
                    second_to_first.entries.at(kind).size());
  }

  [[nodiscard]] std::size_t maximum_capacity() const {
    std::size_t result = 0;
    for (std::size_t kind = 0; kind < PicDirectedFixedExchange::kind_count;
         ++kind) {
      result = std::max(result, capacity(kind));
    }
    return result;
  }
};

struct PicFixedExchangePlan {
  std::vector<PicFixedExchangePair> pairs{};
  std::vector<std::vector<PicFixedExchangeEntry>> local_source_entries{};
};

struct PicHaloBuffers {
  std::vector<backend::DeviceBuffer<Real>> send{};
  std::vector<backend::DeviceBuffer<Real>> receive{};
  std::array<std::vector<backend::DeviceBuffer<PicFixedExchangeEntry>>, 4>
      outgoing_entries{};
  std::array<std::vector<backend::DeviceBuffer<PicFixedExchangeEntry>>, 4>
      incoming_entries{};
  backend::DeviceBuffer<PicFixedExchangeEntry> local_source_entries{};
  std::array<backend::DeviceBuffer<Real>, 4> source_work{};
  backend::DeviceBuffer<Real> filter_work{};
  std::array<backend::DeviceBuffer<Real>, 2> order_rhs{};
  std::array<backend::DeviceBuffer<Real>, 2> order_work{};
  backend::DeviceBuffer<Real> axis_values{};
};

namespace {

constexpr std::size_t field_copy_kind = 0;
constexpr std::size_t source_copy_kind = 1;
constexpr std::size_t source_additive_kind = 2;
constexpr std::size_t cylindrical_axis_kind = 3;

std::unique_ptr<PicFixedExchangePlan> make_pic_fixed_exchange_plan(
    const VirtualTopology& topology, const pic::EmPicConfig& config) {
  const std::size_t endpoints = topology.endpoint_count();
  auto plan = std::make_unique<PicFixedExchangePlan>();
  plan->local_source_entries.resize(endpoints);
  std::vector<std::size_t> pair_index(endpoints * endpoints, 0);
  for (std::size_t first = 0; first < endpoints; ++first) {
    for (std::size_t second = first; second < endpoints; ++second) {
      const std::size_t index = plan->pairs.size();
      plan->pairs.push_back(PicFixedExchangePair{
          .first_endpoint = first,
          .second_endpoint = second,
      });
      pair_index[first * endpoints + second] = index;
      pair_index[second * endpoints + first] = index;
    }
  }

  const auto add = [&](std::size_t source, std::size_t destination,
                       std::size_t kind, PicFixedExchangeEntry entry) {
    auto& pair = plan->pairs.at(pair_index.at(source * endpoints + destination));
    auto& direction = source == pair.first_endpoint
        ? pair.first_to_second : pair.second_to_first;
    direction.entries.at(kind).push_back(entry);
  };

  const std::size_t nx = topology.global_nx();
  const std::size_t ny = topology.global_ny();
  const int halo = config.grid.nghost;
  const bool cylindrical = is_cylindrical(config.geometry);
  const bool px = periodic_x(config);
  const bool py = periodic_y(config);
  const auto fields = field_extents(nx, ny, cylindrical);
  const auto currents = source_extents(nx, ny, cylindrical);
  const std::array<ComponentExtent, 4> sources{
      currents[0], currents[1], currents[2],
      ComponentExtent{nx, ny, false, false}};

  const auto append_copy_plan = [&](std::span<const ComponentExtent> extents,
                                    std::size_t kind) {
    for (const TileExtent& destination : topology.tiles()) {
      const int destination_nx = static_cast<int>(destination.x.size());
      const int destination_ny = static_cast<int>(destination.y.size());
      for (std::size_t component = 0; component < extents.size(); ++component) {
        const ComponentExtent extent = extents[component];
        for (int local_y = -halo; local_y < destination_ny + halo; ++local_y) {
          std::size_t mapped_y = 0;
          if (!map_field_coordinate(
                  static_cast<long long>(destination.y.begin) + local_y,
                  ny, extent.face_y, py, mapped_y)) {
            continue;
          }
          for (int local_x = -halo; local_x < destination_nx + halo;
               ++local_x) {
            std::size_t mapped_x = 0;
            if (!map_field_coordinate(
                    static_cast<long long>(destination.x.begin) + local_x,
                    nx, extent.face_x, px, mapped_x)) {
              continue;
            }
            const CanonicalComponentPoint canonical =
                canonical_component_point(topology, mapped_x, mapped_y,
                                          extent, px, py);
            const TileExtent& source = topology.tile(canonical.owner);
            const int source_x = static_cast<int>(canonical.x - source.x.begin);
            const int source_y = static_cast<int>(canonical.y - source.y.begin);
            if (canonical.owner == destination.endpoint &&
                source_x == local_x && source_y == local_y) {
              continue;
            }
            add(canonical.owner, destination.endpoint, kind,
                PicFixedExchangeEntry{
                    .component = static_cast<std::uint32_t>(component),
                    .source_x = source_x,
                    .source_y = source_y,
                    .destination_x = local_x,
                    .destination_y = local_y,
                });
          }
        }
      }
    }
  };
  append_copy_plan(fields, field_copy_kind);
  append_copy_plan(sources, source_copy_kind);

  // Deposited guards are routed to one canonical owner. Contributions whose
  // owner is the source tile are accumulated locally; this plan contains only
  // fixed-size endpoint-to-endpoint payloads.
  for (const TileExtent& source : topology.tiles()) {
    const int source_nx = static_cast<int>(source.x.size());
    const int source_ny = static_cast<int>(source.y.size());
    for (std::size_t component = 0; component < sources.size(); ++component) {
      const ComponentExtent extent = sources[component];
      for (int local_y = -halo; local_y < source_ny + halo; ++local_y) {
        std::size_t mapped_y = 0;
        if (!map_source_coordinate(
                static_cast<long long>(source.y.begin) + local_y,
                ny, extent.face_y, py, mapped_y)) {
          continue;
        }
        for (int local_x = -halo; local_x < source_nx + halo; ++local_x) {
          std::size_t mapped_x = 0;
          if (!map_source_coordinate(
                  static_cast<long long>(source.x.begin) + local_x,
                  nx, extent.face_x, px, mapped_x)) {
            continue;
          }
          const CanonicalComponentPoint canonical =
              canonical_component_point(topology, mapped_x, mapped_y,
                                        extent, px, py);
          if (canonical.owner == source.endpoint) {
            plan->local_source_entries[source.endpoint].push_back(
                PicFixedExchangeEntry{
                    .component = static_cast<std::uint32_t>(component),
                    .source_x = local_x,
                    .source_y = local_y,
                    .destination_x = static_cast<std::int32_t>(
                        canonical.x - source.x.begin),
                    .destination_y = static_cast<std::int32_t>(
                        canonical.y - source.y.begin),
                });
            continue;
          }
          const TileExtent& destination = topology.tile(canonical.owner);
          add(source.endpoint, canonical.owner, source_additive_kind,
              PicFixedExchangeEntry{
                  .component = static_cast<std::uint32_t>(component),
                  .source_x = local_x,
                  .source_y = local_y,
                  .destination_x = static_cast<std::int32_t>(
                      canonical.x - destination.x.begin),
                  .destination_y = static_cast<std::int32_t>(
                      canonical.y - destination.y.begin),
              });
        }
      }
    }
  }

  if (cylindrical && config.grid.origin_x == Real{0}) {
    for (const TileExtent& destination : topology.tiles()) {
      for (std::size_t local_y = 0; local_y < destination.y.size(); ++local_y) {
        const std::size_t global_y = destination.y.begin + local_y;
        const std::size_t source_endpoint = topology.owner_of_cell(0, global_y);
        if (source_endpoint == destination.endpoint) continue;
        const TileExtent& source = topology.tile(source_endpoint);
        add(source_endpoint, destination.endpoint, cylindrical_axis_kind,
            PicFixedExchangeEntry{
                .component = 0,
                .source_x = 1,
                .source_y = static_cast<std::int32_t>(global_y - source.y.begin),
                .destination_x = 0,
                .destination_y = static_cast<std::int32_t>(local_y),
            });
      }
    }
  }

  std::erase_if(plan->pairs, [](const PicFixedExchangePair& pair) {
    return pair.maximum_capacity() == 0;
  });
  return plan;
}

const PicDirectedFixedExchange& fixed_direction(
    const PicFixedExchangePair& pair, std::size_t source) {
  return source == pair.first_endpoint
      ? pair.first_to_second : pair.second_to_first;
}

pic::PicDeviceTileExtent device_tile_extent(
    const VirtualTopology& topology, const TileExtent& tile,
    bool periodic_x_axis, bool periodic_y_axis) {
  const auto shape = topology.shape();
  return pic::PicDeviceTileExtent{
      .global_x_begin = static_cast<std::uint64_t>(tile.x.begin),
      .global_y_begin = static_cast<std::uint64_t>(tile.y.begin),
      .global_nx = static_cast<std::uint64_t>(topology.global_nx()),
      .global_ny = static_cast<std::uint64_t>(topology.global_ny()),
      .tile_x = static_cast<std::int32_t>(tile.coordinate.x),
      .tile_y = static_cast<std::int32_t>(tile.coordinate.y),
      .tiles_x = static_cast<std::int32_t>(shape.px),
      .tiles_y = static_cast<std::int32_t>(shape.py),
      .periodic_x = periodic_x_axis ? 1 : 0,
      .periodic_y = periodic_y_axis ? 1 : 0,
  };
}

}  // namespace

struct PicTileRuntime::DenseFields {
  PicGlobalFields fields{};
};

struct PicTileRuntime::DenseSources {
  PicGlobalSources sources{};
};

struct PicTileRuntime::LocalSources {
  std::vector<std::array<std::vector<Real>, 4>> endpoint{};
};

struct PicTileRuntime::MigrationBatch {
  std::size_t species_count{0};
  std::size_t first_endpoint{0};
  std::vector<std::vector<std::uint64_t>> departure_counts{};
  std::vector<std::vector<pic::ParticleSpecies::HostSnapshot>> departures{};
  std::vector<std::vector<pic::ParticleSpecies::HostSnapshot>> arrivals{};
  std::vector<std::vector<std::byte>> outgoing{};
  std::uint64_t local_migrated{0};
};

struct PicCheckpointTile {
  TileExtent tile{};
  Grid2D grid{};
  std::array<std::vector<Real>, 6> fields{};
  std::array<std::vector<Real>, 6> external_fields{};
  std::array<std::vector<Real>, 3> previous_b{};
  std::array<std::vector<Real>, 4> sources{};
};

struct PicCheckpointSnapshot {
  std::vector<PicCheckpointTile> local_tiles{};
  std::uint64_t step{0};
  Real previous_dt{Real{0}};
  bool has_previous_dt{false};
  bool background_initialized{false};
  Real background_charge_density{Real{0}};
  PicBoundaryState boundary{};
  std::vector<PicSpeciesState> local_species{};
};

struct PicRestartPayload {
  PicGlobalState state{};
  std::vector<PicCheckpointTile> local_tiles{};
  std::array<std::uint8_t, 7> runtime_flags{};
  std::vector<std::vector<std::uint8_t>> diagnostic_state{};
};

namespace {

bool checkpoint_tile_owns(const PicCheckpointTile& local,
                          const ComponentExtent& extent,
                          std::size_t global_x,
                          std::size_t global_y) {
  const std::size_t skip_x =
      extent.face_x && local.tile.coordinate.x != 0 ? 1 : 0;
  const std::size_t skip_y =
      extent.face_y && local.tile.coordinate.y != 0 ? 1 : 0;
  const std::size_t begin_x = local.tile.x.begin + skip_x;
  const std::size_t begin_y = local.tile.y.begin + skip_y;
  const std::size_t count_x =
      local.tile.x.size() + (extent.face_x ? 1 : 0) - skip_x;
  const std::size_t count_y =
      local.tile.y.size() + (extent.face_y ? 1 : 0) - skip_y;
  return global_x >= begin_x && global_x < begin_x + count_x &&
         global_y >= begin_y && global_y < begin_y + count_y;
}

template <class Select>
std::vector<Real> pack_local_pic_checkpoint_values(
    std::span<const PicCheckpointTile> local_tiles,
    const ComponentExtent& extent,
    std::span<const DatasetHyperslab> slabs,
    Select&& select) {
  std::size_t total = 0;
  for (const auto& slab : slabs) total += pic_slab_elements(slab);
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
      const std::size_t global_y = row_begin + row;
      for (std::size_t column = 0; column < column_count; ++column) {
        const std::size_t global_x = column_begin + column;
        const auto found = std::find_if(
            local_tiles.begin(), local_tiles.end(),
            [&](const PicCheckpointTile& local) {
              return checkpoint_tile_owns(local, extent, global_x, global_y);
            });
        if (found == local_tiles.end()) {
          throw std::logic_error{
              "PIC checkpoint hyperslab has no rank-local tile owner"};
        }
        const int local_x = static_cast<int>(global_x - found->tile.x.begin);
        const int local_y = static_cast<int>(global_y - found->tile.y.begin);
        packed.push_back(select(*found)[found->grid.index(local_x, local_y)]);
      }
    }
  }
  if (packed.size() != total) {
    throw std::logic_error{"PIC checkpoint packed hyperslab size mismatch"};
  }
  return packed;
}

template <class Select>
void unpack_local_pic_checkpoint_values(
    std::span<PicCheckpointTile> local_tiles,
    const ComponentExtent& extent,
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
      const std::size_t global_y = row_begin + row;
      for (std::size_t column = 0; column < column_count; ++column) {
        const std::size_t global_x = column_begin + column;
        const auto found = std::find_if(
            local_tiles.begin(), local_tiles.end(),
            [&](const PicCheckpointTile& local) {
              return checkpoint_tile_owns(local, extent, global_x, global_y);
            });
        if (found == local_tiles.end() || cursor >= packed.size()) {
          throw std::logic_error{
              "PIC restart hyperslab has no rank-local tile owner"};
        }
        const int local_x = static_cast<int>(global_x - found->tile.x.begin);
        const int local_y = static_cast<int>(global_y - found->tile.y.begin);
        select(*found)[found->grid.index(local_x, local_y)] = packed[cursor++];
      }
    }
  }
  if (cursor != packed.size()) {
    throw std::logic_error{"PIC restart packed hyperslab size mismatch"};
  }
}

template <class Select>
void verify_and_rebuild_local_periodic_duplicates(
    MpiRuntime& runtime,
    std::span<PicCheckpointTile> local_tiles,
    const ComponentExtent& extent,
    const pic::EmPicConfig& config,
    std::string_view description,
    Select&& select) {
  const auto reconcile_axis = [&](bool x_axis) {
    const bool active = x_axis
        ? extent.face_x && periodic_x(config)
        : extent.face_y && periodic_y(config);
    if (!active) return;
    const std::size_t transverse = x_axis ? extent.ny : extent.nx;
    const std::size_t high_coordinate = x_axis
        ? static_cast<std::size_t>(config.grid.nx)
        : static_cast<std::size_t>(config.grid.ny);
    std::vector<Real> low(transverse, Real{0});
    std::vector<Real> high(transverse, Real{0});
    std::vector<int> low_counts(transverse, 0);
    std::vector<int> high_counts(transverse, 0);
    for (auto& local : local_tiles) {
      for (std::size_t transverse_index = 0;
           transverse_index < transverse; ++transverse_index) {
        const std::size_t low_x = x_axis ? 0 : transverse_index;
        const std::size_t low_y = x_axis ? transverse_index : 0;
        const std::size_t high_x =
            x_axis ? high_coordinate : transverse_index;
        const std::size_t high_y =
            x_axis ? transverse_index : high_coordinate;
        if (checkpoint_tile_owns(local, extent, low_x, low_y)) {
          low[transverse_index] = select(local)[local.grid.index(
              static_cast<int>(low_x - local.tile.x.begin),
              static_cast<int>(low_y - local.tile.y.begin))];
          low_counts[transverse_index] = 1;
        }
        if (checkpoint_tile_owns(local, extent, high_x, high_y)) {
          high[transverse_index] = select(local)[local.grid.index(
              static_cast<int>(high_x - local.tile.x.begin),
              static_cast<int>(high_y - local.tile.y.begin))];
          high_counts[transverse_index] = 1;
        }
      }
    }
    allreduce_sum_in_place(runtime, low, MPI_DOUBLE,
                           "MPI_Allreduce(PIC restart low seam)");
    allreduce_sum_in_place(runtime, high, MPI_DOUBLE,
                           "MPI_Allreduce(PIC restart high seam)");
    allreduce_sum_in_place(runtime, low_counts, MPI_INT,
                           "MPI_Allreduce(PIC restart low seam ownership)");
    allreduce_sum_in_place(runtime, high_counts, MPI_INT,
                           "MPI_Allreduce(PIC restart high seam ownership)");
    for (std::size_t index = 0; index < transverse; ++index) {
      if (low_counts[index] != 1 || high_counts[index] != 1 ||
          low[index] != high[index]) {
        throw std::runtime_error{
            std::string{description} +
            " has an inconsistent periodic high-face duplicate"};
      }
    }
    for (auto& local : local_tiles) {
      for (std::size_t transverse_index = 0;
           transverse_index < transverse; ++transverse_index) {
        const std::size_t high_x =
            x_axis ? high_coordinate : transverse_index;
        const std::size_t high_y =
            x_axis ? transverse_index : high_coordinate;
        if (!checkpoint_tile_owns(local, extent, high_x, high_y)) continue;
        select(local)[local.grid.index(
            static_cast<int>(high_x - local.tile.x.begin),
            static_cast<int>(high_y - local.tile.y.begin))] =
                low[transverse_index];
      }
    }
  };
  reconcile_axis(true);
  reconcile_axis(false);
}

}  // namespace

void validate_pic_global_fields(const PicGlobalFields& fields,
                                std::string_view geometry) {
  if (fields.global_nx == 0 || fields.global_ny == 0) {
    throw std::invalid_argument{
        "distributed PIC fields require a non-empty global mesh"};
  }
  if (geometry != "cartesian" && geometry != "cylindrical") {
    throw std::invalid_argument{"distributed PIC geometry is unsupported"};
  }
  const auto extents = field_extents(fields.global_nx, fields.global_ny,
                                     is_cylindrical(geometry));
  for (std::size_t component = 0; component < extents.size(); ++component) {
    const auto kind = static_cast<FieldComponent>(component);
    const auto& values = field_vector(fields, kind);
    const std::size_t expected = checked_product(
        extents[component].nx, extents[component].ny,
        "PIC field component");
    if (values.size() != expected) {
      throw std::invalid_argument{
          "distributed PIC field component has the wrong Yee lattice size"};
    }
    require_finite(std::span<const Real>{values}, "PIC field component");
  }
}

void validate_pic_species_state(const PicSpeciesState& species) {
  if (species.config.name.empty()) {
    throw std::invalid_argument{"distributed PIC species name is empty"};
  }
  if (!(std::isfinite(species.config.mass) && species.config.mass > Real{0}) ||
      !std::isfinite(species.config.charge)) {
    throw std::invalid_argument{
        "distributed PIC species charge/mass is invalid"};
  }
  const auto& particles = species.particles;
  const std::size_t size = particles.x.size();
  if (particles.y.size() != size || particles.x_prev.size() != size ||
      particles.y_prev.size() != size || particles.vx.size() != size ||
      particles.vy.size() != size || particles.vz.size() != size ||
      particles.vphi_deposit.size() != size ||
      particles.weight.size() != size || particles.alive.size() != size ||
      particles.id.size() != size) {
    throw std::invalid_argument{
        "distributed PIC species has incomplete particle records"};
  }
  std::unordered_set<std::uint64_t> ids;
  ids.reserve(size);
  for (std::size_t index = 0; index < size; ++index) {
    if (!(std::isfinite(particles.x[index]) &&
          std::isfinite(particles.y[index]) &&
          std::isfinite(particles.x_prev[index]) &&
          std::isfinite(particles.y_prev[index]) &&
          std::isfinite(particles.vx[index]) &&
          std::isfinite(particles.vy[index]) &&
          std::isfinite(particles.vz[index]) &&
          std::isfinite(particles.vphi_deposit[index]) &&
          std::isfinite(particles.weight[index]) &&
          particles.weight[index] >= Real{0} &&
          particles.alive[index] <= 1)) {
      throw std::invalid_argument{
          "distributed PIC particle record contains an invalid value"};
    }
    const Real speed = std::hypot(
        std::hypot(particles.vx[index], particles.vy[index]),
        particles.vz[index]);
    if (!(speed < Real{1})) {
      throw std::invalid_argument{
          "distributed PIC particle velocity must satisfy |v| < 1"};
    }
    if (!ids.insert(particles.id[index]).second) {
      throw std::invalid_argument{
          "distributed PIC particle IDs must be unique within a species"};
    }
  }
}

PicTileRuntime::PicTileRuntime(MpiRuntime& runtime, EndpointMapping mapping,
                               VirtualTopology topology,
                               pic::EmPicConfig global_config,
                               TransportPolicy transport_policy)
    : runtime_{&runtime},
      mapping_{std::move(mapping)},
      topology_{std::move(topology)},
      global_config_{std::move(global_config)} {
  runtime.require_orchestration_thread();
  collective_try(
      runtime, worker_epoch_, "pic-runtime-config",
      "distributed PIC runtime configuration is invalid", -1, [&] {
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
          "virtual topology and PIC configuration mesh differ"};
    }
    const std::size_t required_tile_width =
        global_config_.fdtd_order == 4 || global_config_.shape == "tsc"
        ? 2u : 1u;
    for (const TileExtent& tile : topology_.tiles()) {
      if (tile.x.size() < required_tile_width ||
          tile.y.size() < required_tile_width) {
        throw std::invalid_argument{
            "PIC decomposition " +
            std::to_string(topology_.shape().px) + 'x' +
            std::to_string(topology_.shape().py) +
            " creates a tile thinner than the required " +
            std::to_string(required_tile_width) +
            "-cell stencil reach for fdtd_order=" +
            std::to_string(global_config_.fdtd_order) +
            " and shape=" + global_config_.shape};
      }
    }
    for (const auto& spec : global_config_.filters) {
      auto filter = Registry<numerics::ICurrentFilter>::instance().create(
          spec.name);
      filter->set_passes(spec.passes);
      auto stencils = filter->distributed_stencils();
      for (const auto& stencil : stencils) {
        if (stencil.passes < 1 ||
            !std::isfinite(stencil.neighbor_weight) ||
            !std::isfinite(stencil.center_weight)) {
          throw std::invalid_argument{
              "registered PIC current filter returned an invalid "
              "distributed stencil"};
        }
      }
      filter_stencils_.insert(filter_stencils_.end(), stencils.begin(),
                              stencils.end());
    }
    for (const auto& endpoint : mapping_.endpoints_for_rank(runtime.rank())) {
      local_devices_.push_back(endpoint.device.ordinal);
    }
  });

  std::uint64_t config_hash = 1469598103934665603ULL;
  hash_scalar(config_hash, global_config_.grid.nx);
  hash_scalar(config_hash, global_config_.grid.ny);
  // A restart may rebuild omitted halos at a different sufficient width, while
  // ranks in the same live run must continue to agree on allocation geometry.
  hash_scalar(config_hash, global_config_.grid.nghost);
  hash_string(config_hash, global_config_.geometry);
  hash_string(config_hash, pic_boundary_signature(global_config_));
  hash_string(config_hash, pic_background_signature(global_config_));
  hash_string(config_hash, pic_numerics_signature(global_config_));
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
            "MPI_Allreduce(distributed PIC config minimum hash)");
  check_mpi(MPI_Allreduce(&config_hash, &maximum_hash, 1, MPI_UINT64_T,
                          MPI_MAX,
                          detail::MpiRuntimeNativeAccess::world(runtime)),
            "MPI_Allreduce(distributed PIC config maximum hash)");
  collective_require(
      runtime, worker_epoch_, minimum_hash == maximum_hash,
      "pic-runtime-agreement",
      "distributed PIC configuration or topology differs between MPI ranks");

  collective_try_with_fallback(
      runtime, worker_epoch_, "pic-fixed-exchange-plan",
      "distributed PIC fixed exchange plan allocation failed", -1, [&] {
    fixed_exchange_plan_ = make_pic_fixed_exchange_plan(
        topology_, global_config_);
  });

  const std::size_t first_endpoint =
      static_cast<std::size_t>(runtime.rank()) * mapping_.devices_per_rank();
  const std::uint64_t worker_construction_epoch = worker_epoch_++;
  const CollectiveErrorRecord worker_construction = collective_try_record(
      runtime, worker_construction_epoch, "pic-worker-pool-construct",
      "non-standard GPU worker-pool construction failure", -1, [&] {
    workers_ =
        std::make_unique<EndpointWorkerPool>(local_devices_, first_endpoint);
  });
  try {
    runtime.require_collective_success(worker_construction);
  } catch (...) {
    // This cleanup is deliberately local: all participants have already
    // reached the construction consensus, and none may enter a worker phase.
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
        runtime, worker_epoch_, "pic-tile-storage",
        "distributed PIC tile storage allocation failed", -1, [&] {
    solvers_.resize(local_devices_.size());
    halo_buffers_.resize(local_devices_.size());
    tasks = local_indexed_tasks(solvers_.size(),
        [this, first_endpoint](std::size_t local, WorkerContext&) {
      const std::size_t endpoint = first_endpoint + local;
      solvers_[local] =
          std::make_unique<pic::EmPic2D3V>(tile_config(endpoint));
      auto buffers = std::make_unique<PicHaloBuffers>();
      const std::size_t storage_size =
          PicTileAccess::grid(*solvers_[local]).storage_size();
      for (auto& source : buffers->source_work) {
        source = backend::DeviceBuffer<Real>{storage_size};
      }
      buffers->filter_work = backend::DeviceBuffer<Real>{storage_size};
      for (auto& rhs : buffers->order_rhs) {
        rhs = backend::DeviceBuffer<Real>{storage_size};
      }
      for (auto& output : buffers->order_work) {
        output = backend::DeviceBuffer<Real>{storage_size};
      }
      buffers->axis_values = backend::DeviceBuffer<Real>{storage_size};
      const auto& local_entries =
          fixed_exchange_plan_->local_source_entries.at(endpoint);
      if (!local_entries.empty()) {
        buffers->local_source_entries =
            backend::DeviceBuffer<PicFixedExchangeEntry>{
                local_entries.size()};
        buffers->local_source_entries.copy_from_host(
            local_entries.data(), local_entries.size());
      }
      buffers->send.resize(topology_.endpoint_count());
      buffers->receive.resize(topology_.endpoint_count());
      for (std::size_t kind = 0;
           kind < PicDirectedFixedExchange::kind_count; ++kind) {
        buffers->outgoing_entries[kind].resize(topology_.endpoint_count());
        buffers->incoming_entries[kind].resize(topology_.endpoint_count());
      }
      for (const auto& pair : fixed_exchange_plan_->pairs) {
        if (pair.first_endpoint != endpoint &&
            pair.second_endpoint != endpoint) {
          continue;
        }
        const std::size_t peer = pair.first_endpoint == endpoint
            ? pair.second_endpoint : pair.first_endpoint;
        const std::size_t capacity = pair.maximum_capacity();
        buffers->send[peer] = backend::DeviceBuffer<Real>{capacity};
        buffers->receive[peer] = backend::DeviceBuffer<Real>{capacity};
        for (std::size_t kind = 0;
             kind < PicDirectedFixedExchange::kind_count; ++kind) {
          const auto& outgoing =
              fixed_direction(pair, endpoint).entries[kind];
          const auto& incoming = fixed_direction(pair, peer).entries[kind];
          if (!outgoing.empty()) {
            buffers->outgoing_entries[kind][peer] =
                backend::DeviceBuffer<PicFixedExchangeEntry>{outgoing.size()};
            buffers->outgoing_entries[kind][peer].copy_from_host(
                outgoing.data(), outgoing.size());
          }
          if (!incoming.empty()) {
            buffers->incoming_entries[kind][peer] =
                backend::DeviceBuffer<PicFixedExchangeEntry>{incoming.size()};
            buffers->incoming_entries[kind][peer].copy_from_host(
                incoming.data(), incoming.size());
          }
        }
      }
      halo_buffers_[local] = std::move(buffers);
    });
    });
  } catch (...) {
    workers_->close();
    workers_.reset();
    solvers_.clear();
    halo_buffers_.clear();
    throw;
  }
  const auto resolution = workers_->execute_collective(
      runtime, worker_epoch_++, "pic-tile-construct", tasks);
  if (!resolution.accepted()) {
    auto cleanup = local_indexed_tasks(solvers_.size(),
        [this](std::size_t local, WorkerContext&) {
          halo_buffers_[local].reset();
          solvers_[local].reset();
        });
    (void)workers_->execute(worker_epoch_++, runtime.rank(),
                            "pic-tile-cleanup", cleanup);
    workers_->close();
    throw DistributedCollectiveError{resolution};
  }
  try {
    transport_ = std::make_unique<Transport>(runtime, transport_policy);
  } catch (...) {
    auto cleanup = local_indexed_tasks(solvers_.size(),
        [this](std::size_t local, WorkerContext&) {
          halo_buffers_[local].reset();
          solvers_[local].reset();
        });
    (void)workers_->execute(worker_epoch_++, runtime.rank(),
                            "pic-transport-cleanup", cleanup);
    workers_->close();
    throw;
  }
}

PicTileRuntime::~PicTileRuntime() noexcept = default;

const VirtualTopology& PicTileRuntime::topology() const noexcept {
  return topology_;
}

const EndpointMapping& PicTileRuntime::mapping() const noexcept {
  return mapping_;
}

bool PicTileRuntime::closed() const noexcept { return lifecycle_.closed; }
bool PicTileRuntime::seeded() const noexcept { return lifecycle_.seeded; }
bool PicTileRuntime::poisoned() const noexcept { return lifecycle_.poisoned; }

void PicTileRuntime::inject_next_worker_task_allocation_failure_for_testing(
    bool enabled) noexcept {
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
  inject_next_pic_worker_task_allocation_failure = enabled;
#else
  (void)enabled;
#endif
}

void PicTileRuntime::inject_seed_post_mutation_failure_for_testing(
    bool enabled) noexcept {
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
  inject_pic_seed_post_mutation_failure = enabled;
#else
  (void)enabled;
#endif
}

void PicTileRuntime::inject_checkpoint_metadata_copy_failure_for_testing(
    bool enabled) noexcept {
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
  inject_pic_checkpoint_metadata_copy_failure = enabled;
#else
  (void)enabled;
#endif
}

void PicTileRuntime::inject_restart_post_reconcile_failure_for_testing(
    bool enabled) noexcept {
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
  inject_restart_post_reconcile_failure_ = enabled;
#else
  (void)enabled;
#endif
}

const PicRuntimeTelemetry& PicTileRuntime::telemetry() const noexcept {
  return telemetry_;
}

const TransportResolution&
PicTileRuntime::transport_resolution() const noexcept {
  return transport_->resolution();
}

pic::EmPicConfig PicTileRuntime::tile_config(std::size_t endpoint) const {
  pic::EmPicConfig config = global_config_;
  const TileExtent& tile = topology_.tile(endpoint);
  const Real dx = global_config_.grid.dx();
  const Real dy = global_config_.grid.dy();
  config.grid = Grid2D::from_cell_spacing(
      static_cast<int>(tile.x.size()), static_cast<int>(tile.y.size()),
      dx, dy,
      std::fma(static_cast<Real>(tile.x.begin), dx,
               global_config_.grid.origin_x),
      std::fma(static_cast<Real>(tile.y.begin), dy,
               global_config_.grid.origin_y),
      global_config_.grid.nghost);

  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);
  for (int side = 0; side < 4; ++side) {
    const Direction direction = static_cast<Direction>(side);
    const auto neighbor = topology_.neighbor(endpoint, direction, px, py);
    if (neighbor && *neighbor != endpoint) {
      config.boundary.field[side] = "internal";
      config.boundary.particle[side] = "internal";
    }
  }
  return config;
}

void PicTileRuntime::require_usable(bool require_seeded) const {
  if (lifecycle_.closed) {
    throw std::logic_error{"distributed PIC runtime is closed"};
  }
  if (lifecycle_.poisoned) {
    throw std::logic_error{
        "distributed PIC runtime is poisoned; only close() is legal"};
  }
  if (require_seeded && !lifecycle_.seeded) {
    throw std::logic_error{"distributed PIC runtime is not seeded"};
  }
}

void PicTileRuntime::seed(const PicGlobalFields& fields,
                          const PicGlobalFields* external_fields,
                          std::span<const PicSpeciesState> species) {
  require_usable(false);
  if (lifecycle_.seeded) {
    throw std::logic_error{"distributed PIC runtime is already seeded"};
  }
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-seed-validate",
      "distributed PIC seed validation failed", -1, [&] {
    validate_pic_global_fields(fields, global_config_.geometry);
    if (fields.global_nx != topology_.global_nx() ||
        fields.global_ny != topology_.global_ny()) {
      throw std::invalid_argument{
          "distributed PIC seed mesh does not match the topology"};
    }
    if (external_fields != nullptr) {
      validate_pic_global_fields(*external_fields, global_config_.geometry);
      if (external_fields->global_nx != fields.global_nx ||
          external_fields->global_ny != fields.global_ny) {
        throw std::invalid_argument{
            "distributed PIC external-field mesh differs from the live field"};
      }
    }
    std::unordered_set<std::string> names;
    std::unordered_set<std::uint64_t> particle_ids;
    for (const auto& state : species) {
      validate_pic_species_state(state);
      if (!names.insert(state.config.name).second) {
        throw std::invalid_argument{
            "distributed PIC species names must be unique"};
      }
      for (std::size_t particle = 0;
           particle < state.particles.x.size(); ++particle) {
        if (!particle_ids.insert(state.particles.id[particle]).second) {
          throw std::invalid_argument{
              "distributed PIC particle IDs must be globally unique across species"};
        }
        if (state.particles.alive[particle] == 0) continue;
        const Real x = state.particles.x[particle];
        const Real y = state.particles.y[particle];
        const Real x_lo = global_config_.grid.origin_x;
        const Real y_lo = global_config_.grid.origin_y;
        const Real x_hi = x_lo + global_config_.grid.lx;
        const Real y_hi = y_lo + global_config_.grid.ly;
        if (!(x >= x_lo && x <= x_hi && y >= y_lo && y <= y_hi)) {
          throw std::invalid_argument{
              "distributed PIC initial live particle lies outside the global domain"};
        }
      }
    }
  });

  // This API deliberately accepts a replicated canonical seed.  Detect a
  // rank-dependent deck/checkpoint before individual ranks seed their tiles.
  std::uint64_t local_hash = 1469598103934665603ULL;
  hash_scalar(local_hash, fields.global_nx);
  hash_scalar(local_hash, fields.global_ny);
  for (std::size_t component = 0; component < 6; ++component) {
    const auto& values = field_vector(
        fields, static_cast<FieldComponent>(component));
    hash_span(local_hash, std::span<const Real>{values});
  }
  const bool has_external = external_fields != nullptr;
  hash_scalar(local_hash, has_external);
  if (external_fields != nullptr) {
    for (std::size_t component = 0; component < 6; ++component) {
      const auto& values = field_vector(
          *external_fields, static_cast<FieldComponent>(component));
      hash_span(local_hash, std::span<const Real>{values});
    }
  }
  hash_scalar(local_hash, species.size());
  for (const auto& state : species) {
    hash_string(local_hash, state.config.name);
    hash_scalar(local_hash, state.config.charge);
    hash_scalar(local_hash, state.config.mass);
    hash_scalar(local_hash, state.config.capacity);
    const auto& p = state.particles;
    hash_span(local_hash, std::span<const Real>{p.x});
    hash_span(local_hash, std::span<const Real>{p.y});
    hash_span(local_hash, std::span<const Real>{p.x_prev});
    hash_span(local_hash, std::span<const Real>{p.y_prev});
    hash_span(local_hash, std::span<const Real>{p.vx});
    hash_span(local_hash, std::span<const Real>{p.vy});
    hash_span(local_hash, std::span<const Real>{p.vz});
    hash_span(local_hash, std::span<const Real>{p.vphi_deposit});
    hash_span(local_hash, std::span<const Real>{p.weight});
    hash_span(local_hash, std::span<const std::uint8_t>{p.alive});
    hash_span(local_hash, std::span<const std::uint64_t>{p.id});
  }
  std::uint64_t minimum_hash = 0;
  std::uint64_t maximum_hash = 0;
  check_mpi(MPI_Allreduce(&local_hash, &minimum_hash, 1, MPI_UINT64_T, MPI_MIN,
                          detail::MpiRuntimeNativeAccess::world(*runtime_)),
            "MPI_Allreduce(distributed PIC seed minimum hash)");
  check_mpi(MPI_Allreduce(&local_hash, &maximum_hash, 1, MPI_UINT64_T, MPI_MAX,
                          detail::MpiRuntimeNativeAccess::world(*runtime_)),
            "MPI_Allreduce(distributed PIC seed maximum hash)");
  collective_require(
      *runtime_, worker_epoch_, minimum_hash == maximum_hash,
      "pic-seed-consistency",
      "canonical PIC seed differs between MPI ranks");

  bool mutation_started = false;
  try {
    seed_local_fields(fields, external_fields, mutation_started);
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
    const bool injected =
        std::exchange(inject_pic_seed_post_mutation_failure, false);
    collective_require(
        *runtime_, worker_epoch_, !injected, "pic-seed-post-mutation",
        "injected post-mutation PIC seed failure");
#endif
    seed_local_species(species);
    initialize_global_background();
    materialize_charge();
    lifecycle_.seeded = true;
    reconcile_fields("pic-seed-fields");
  } catch (const std::exception& error) {
    if (mutation_started) {
      poison_collectively("pic-seed-failure", error.what());
    }
    throw;
  } catch (...) {
    if (mutation_started) {
      poison_collectively("pic-seed-failure",
                          "non-standard PIC seed failure");
    }
    throw;
  }
}

void PicTileRuntime::seed_local_fields(
    const PicGlobalFields& fields,
    const PicGlobalFields* external_fields,
    bool& mutation_started) {
  DenseFields live;
  DenseFields external;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-seed-field-preparation",
      "distributed PIC seed field allocation failed", -1, [&] {
    live.fields = fields;
    if (external_fields != nullptr) {
      external.fields = *external_fields;
    } else {
      external.fields.global_nx = fields.global_nx;
      external.fields.global_ny = fields.global_ny;
      const auto extents = field_extents(
          fields.global_nx, fields.global_ny,
          is_cylindrical(global_config_.geometry));
      for (std::size_t component = 0; component < extents.size(); ++component) {
        field_vector(external.fields, static_cast<FieldComponent>(component))
            .assign(checked_product(extents[component].nx,
                                    extents[component].ny,
                                    "PIC external field"),
                    Real{0});
      }
    }
  });

  apply_fields(live, false, false, "pic-seed-live-fields",
               &mutation_started);
  apply_fields(external, true, false, "pic-seed-external-fields");

  auto boundary_tasks = pic_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this](std::size_t local, WorkerContext&) {
    PicTileAccess::fill_field_ghosts(*solvers_[local]);
    backend::device_synchronize(nullptr);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-seed-physical-ghosts", boundary_tasks);
}

void PicTileRuntime::sample_external_fields(
    numerics::IFieldEvaluator& evaluator,
    const core::IFieldSource& source, Real length_scale,
    Real e_field_scale, Real b_field_scale) {
  require_usable();
  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(),
      [this, &evaluator, &source, length_scale, e_field_scale,
       b_field_scale](std::size_t local, WorkerContext&) {
        auto& solver = *solvers_[local];
        pic::sample_external_field(
            evaluator, source, PicTileAccess::external_fields(solver), length_scale,
            e_field_scale, b_field_scale, global_config_.plane,
            global_config_.geometry, global_config_.fdtd_order);
        backend::device_synchronize(nullptr);
      });
  const auto resolution = workers_->execute_collective(
      *runtime_, worker_epoch_++, "pic-sample-external-fields", tasks);
  if (!resolution.accepted()) {
    // Sampling commits a complete tile at a time. Other endpoints may already
    // have committed by the time one evaluator rejects its coordinates, so a
    // failed collective cannot safely resume from this in-memory state.
    lifecycle_.poisoned = true;
    throw DistributedCollectiveError{resolution};
  }
}

std::size_t PicTileRuntime::owner_for_position(Real x, Real y) const {
  const Grid2D& grid = global_config_.grid;
  const Real x_hi = grid.origin_x + grid.lx;
  const Real y_hi = grid.origin_y + grid.ly;
  std::size_t ix = 0;
  std::size_t iy = 0;
  if (x == x_hi) {
    ix = topology_.global_nx() - 1;
  } else {
    const Real coordinate = ::quasar::detail::scaled_difference_quotient(
        x, grid.origin_x, grid.dx());
    if (!(std::isfinite(coordinate) && coordinate >= Real{0})) {
      throw std::invalid_argument{
          "distributed PIC particle x coordinate is outside the global mesh"};
    }
    ix = static_cast<std::size_t>(std::floor(coordinate));
  }
  if (y == y_hi) {
    iy = topology_.global_ny() - 1;
  } else {
    const Real coordinate = ::quasar::detail::scaled_difference_quotient(
        y, grid.origin_y, grid.dy());
    if (!(std::isfinite(coordinate) && coordinate >= Real{0})) {
      throw std::invalid_argument{
          "distributed PIC particle y coordinate is outside the global mesh"};
    }
    iy = static_cast<std::size_t>(std::floor(coordinate));
  }
  if (ix >= topology_.global_nx() || iy >= topology_.global_ny()) {
    throw std::invalid_argument{
        "distributed PIC particle position has no half-open tile owner"};
  }
  return topology_.owner_of_cell(ix, iy);
}

void PicTileRuntime::seed_local_species(
    std::span<const PicSpeciesState> species) {
  const std::size_t first_endpoint =
      static_cast<std::size_t>(runtime_->rank()) * mapping_.devices_per_rank();
  std::vector<std::vector<pic::ParticleSpecies::HostSnapshot>> routed;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-seed-species-preparation",
      "distributed PIC seed species allocation failed", -1, [&] {
    routed.resize(solvers_.size());
    for (auto& endpoint : routed) endpoint.resize(species.size());
    for (std::size_t kind = 0; kind < species.size(); ++kind) {
      const auto& input = species[kind].particles;
      for (std::size_t particle = 0; particle < input.x.size(); ++particle) {
        ParticleRecord record = make_record(input, particle, 0);
        std::size_t owner = 0;
        if (input.alive[particle] != 0) {
          if (periodic_x(global_config_)) {
            const Real wrapped = wrap_periodic(
                record.x, global_config_.grid.origin_x,
                global_config_.grid.lx);
            const Real shift = wrapped - record.x;
            record.x = wrapped;
            record.x_prev += shift;
          }
          if (periodic_y(global_config_)) {
            const Real wrapped = wrap_periodic(
                record.y, global_config_.grid.origin_y,
                global_config_.grid.ly);
            const Real shift = wrapped - record.y;
            record.y = wrapped;
            record.y_prev += shift;
          }
          owner = owner_for_position(record.x, record.y);
        } else {
          // Dead restart records have no physical owner. Stable-ID hashing
          // gives them a deterministic home until normal compaction.
          owner = static_cast<std::size_t>(
              input.id[particle] % topology_.endpoint_count());
        }
        if (owner >= first_endpoint &&
            owner < first_endpoint + solvers_.size()) {
          append_record(routed[owner - first_endpoint][kind], record);
        }
      }
    }
    for (auto& by_solver : routed) {
      for (auto& snapshot : by_solver) sort_particles(snapshot);
    }
  });

  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, &species, &routed](std::size_t local, WorkerContext&) {
    auto& solver = *solvers_[local];
    for (std::size_t kind = 0; kind < species.size(); ++kind) {
      pic::SpeciesConfig config = species[kind].config;
      const std::size_t share =
          config.capacity / topology_.endpoint_count() +
          (config.capacity % topology_.endpoint_count() != 0 ? 1u : 0u);
      config.capacity = std::max(share, routed[local][kind].x.size());
      solver.add_species(pic::ParticleSpecies{config});
      PicTileAccess::replace_species_host(
          solver, kind, routed[local][kind]);
    }
    backend::device_synchronize(nullptr);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-seed-species", tasks);
}

void PicTileRuntime::replace_local_species_particles(
    std::span<const PicSpeciesState> species) {
  const std::size_t first_endpoint =
      static_cast<std::size_t>(runtime_->rank()) * mapping_.devices_per_rank();
  std::vector<std::vector<pic::ParticleSpecies::HostSnapshot>> routed;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-restore-local-particles-validate",
      "PIC restore particle routing allocation failed", -1, [&] {
    routed.resize(solvers_.size());
    for (auto& endpoint : routed) endpoint.resize(species.size());
    for (std::size_t kind = 0; kind < species.size(); ++kind) {
      validate_pic_species_state(species[kind]);
      const auto& input = species[kind].particles;
      for (std::size_t particle = 0; particle < input.x.size(); ++particle) {
        ParticleRecord record = make_record(input, particle, 0);
        const std::size_t owner = input.alive[particle] != 0
            ? owner_for_position(record.x, record.y)
            : static_cast<std::size_t>(
                  input.id[particle] % topology_.endpoint_count());
        if (owner < first_endpoint ||
            owner >= first_endpoint + solvers_.size()) {
          throw std::runtime_error{
              "checkpoint particle was routed to the wrong MPI rank"};
        }
        append_record(routed[owner - first_endpoint][kind], record);
      }
    }
    for (auto& by_solver : routed) {
      for (auto& snapshot : by_solver) sort_particles(snapshot);
    }
  });

  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(),
      [this, &species, &routed](std::size_t local, WorkerContext&) {
        auto& solver = *solvers_[local];
        if (PicTileAccess::species(solver).size() != species.size()) {
          throw std::runtime_error{
              "checkpoint species count differs from the seeded solver"};
        }
        for (std::size_t kind = 0; kind < species.size(); ++kind) {
          PicTileAccess::replace_species_host(
              solver, kind, routed[local][kind]);
        }
        backend::device_synchronize(nullptr);
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-restore-local-particles", tasks);
}

std::vector<PicSpeciesState> PicTileRuntime::snapshot_local_particles() {
  const std::size_t species_count =
      PicTileAccess::species(*solvers_.front()).size();
  std::vector<std::vector<pic::ParticleSpecies::HostSnapshot>> snapshots;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-checkpoint-particle-snapshot-storage",
      "PIC checkpoint particle snapshot allocation failed", -1, [&] {
    snapshots.resize(solvers_.size());
    for (auto& endpoint : snapshots) endpoint.resize(species_count);
  });
  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(),
      [this, &snapshots](std::size_t local, WorkerContext&) {
        for (std::size_t kind = 0;
             kind < PicTileAccess::species(*solvers_[local]).size(); ++kind) {
          snapshots[local][kind] =
              PicTileAccess::species(*solvers_[local])[kind].to_host();
        }
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-checkpoint-local-particle-snapshot", tasks);

  std::vector<PicSpeciesState> result;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-checkpoint-particle-result-storage",
      "PIC checkpoint particle result allocation failed", -1, [&] {
    result.resize(species_count);
    for (std::size_t kind = 0; kind < species_count; ++kind) {
      const auto& representative =
          PicTileAccess::species(*solvers_.front())[kind];
      result[kind].config = pic::SpeciesConfig{
          representative.name(), representative.charge(),
          representative.mass(), 0};
      for (const auto& endpoint : snapshots) {
        const auto& source = endpoint[kind];
        for (std::size_t particle = 0; particle < source.x.size();
             ++particle) {
          append_particle(result[kind].particles, source, particle);
        }
      }
      sort_particles(result[kind].particles);
      result[kind].config.capacity = result[kind].particles.x.size();
    }
  });
  return result;
}

PicSpeciesState PicTileRuntime::route_checkpoint_species(
    PicSpeciesState local_chunk, std::uint64_t global_count) {
  std::vector<ParticleRecord> local_records;
  collective_try(
      *runtime_, worker_epoch_, "pic-restart-particle-chunk-validate",
      "PIC checkpoint particle chunk validation failed", -1, [&] {
    validate_pic_species_state(local_chunk);
    if (global_count > static_cast<std::uint64_t>(
                           std::numeric_limits<std::size_t>::max())) {
      throw std::length_error{
          "checkpoint particle count is not representable"};
    }
    const auto slabs = balanced_1d_slabs(
        static_cast<std::size_t>(global_count), runtime_->rank(),
        runtime_->size());
    const std::size_t expected = slabs.empty()
        ? 0
        : static_cast<std::size_t>(slabs.front().count.front());
    if (local_chunk.particles.x.size() != expected) {
      throw std::runtime_error{
          "checkpoint particle chunk has the wrong size"};
    }
    local_records.reserve(expected);
    for (std::size_t particle = 0; particle < expected; ++particle) {
      local_records.push_back(
          make_record(local_chunk.particles, particle, 0));
    }
  });

  constexpr std::size_t max_chunk_bytes = 64u * 1024u * 1024u;
  constexpr std::size_t max_chunk_records =
      max_chunk_bytes / sizeof(ParticleRecord);
  const std::size_t total = static_cast<std::size_t>(global_count);
  const std::size_t ranks = static_cast<std::size_t>(runtime_->size());
  const std::size_t largest_partition =
      total / ranks + (total % ranks != 0 ? 1u : 0u);
  const std::size_t buffer_records =
      std::min(max_chunk_records, largest_partition);
  std::vector<ParticleRecord> incoming;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-restart-particle-route-allocation",
      "checkpoint particle routing allocation failed", -1, [&] {
    incoming.resize(buffer_records);
  });

  PicSpeciesState owned;
  owned.config = local_chunk.config;
  for (int root = 0; root < runtime_->size(); ++root) {
    const auto root_slabs = balanced_1d_slabs(
        total, root, runtime_->size());
    const std::size_t root_count = root_slabs.empty()
        ? 0
        : static_cast<std::size_t>(root_slabs.front().count.front());
    for (std::size_t sent = 0; sent < root_count;) {
      const std::size_t chunk_count =
          std::min(max_chunk_records, root_count - sent);
      ParticleRecord* chunk = runtime_->rank() == root
          ? local_records.data() + sent
          : incoming.data();
      const std::size_t chunk_bytes =
          chunk_count * sizeof(ParticleRecord);
      check_mpi(MPI_Bcast(chunk, static_cast<int>(chunk_bytes), MPI_BYTE,
                          root,
                          detail::MpiRuntimeNativeAccess::world(*runtime_)),
                "MPI_Bcast(PIC restart particle chunk)");

      collective_try(
          *runtime_, worker_epoch_, "pic-restart-particle-route",
          "PIC checkpoint particle routing failed", -1, [&] {
        for (std::size_t index = 0; index < chunk_count; ++index) {
          const ParticleRecord& record = chunk[index];
          const std::size_t owner = record.alive != 0
              ? owner_for_position(record.x, record.y)
              : static_cast<std::size_t>(
                    record.id % topology_.endpoint_count());
          if (mapping_.endpoint(owner).world_rank == runtime_->rank()) {
            append_record(owned.particles, record);
          }
        }
      });
      sent += chunk_count;
    }
  }
  sort_particles(owned.particles);
  owned.config.capacity = owned.particles.x.size();
  const std::uint64_t local_owned = owned.particles.x.size();
  std::uint64_t global_owned = 0;
  check_mpi(MPI_Allreduce(&local_owned, &global_owned, 1, MPI_UINT64_T,
                          MPI_SUM,
                          detail::MpiRuntimeNativeAccess::world(*runtime_)),
            "MPI_Allreduce(PIC restart routed particle count)");
  collective_require(
      *runtime_, worker_epoch_, global_owned == global_count,
      "pic-restart-particle-route-count",
      "checkpoint particle routing lost or duplicated records");
  return owned;
}

PicTileRuntime::DenseFields PicTileRuntime::collect_fields(
    bool external, bool previous_b, std::string_view phase) {
  const std::size_t nx = topology_.global_nx();
  const std::size_t ny = topology_.global_ny();
  const bool cylindrical = is_cylindrical(global_config_.geometry);
  const auto extents = field_extents(nx, ny, cylindrical);
  DenseFields dense;
  dense.fields.global_nx = nx;
  dense.fields.global_ny = ny;
  std::array<std::vector<int>, 6> counts;
  for (std::size_t component = 0; component < extents.size(); ++component) {
    const std::size_t size = checked_product(extents[component].nx,
                                             extents[component].ny,
                                             "PIC dense field");
    field_vector(dense.fields, static_cast<FieldComponent>(component))
        .assign(size, Real{0});
    counts[component].assign(size, 0);
  }
  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);

  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, external, previous_b, &dense, &counts, &extents, nx, ny, px, py]
      (std::size_t local, WorkerContext&) {
    auto& solver = *solvers_[local];
    const std::size_t endpoint =
        static_cast<std::size_t>(runtime_->rank()) *
            mapping_.devices_per_rank() + local;
    const TileExtent& tile = topology_.tile(endpoint);
    const Grid2D& grid = PicTileAccess::grid(solver);
    for (std::size_t component = 0; component < extents.size(); ++component) {
      const auto kind = static_cast<FieldComponent>(component);
      if (previous_b && component < static_cast<std::size_t>(FieldComponent::bx)) {
        continue;
      }
      backend::DeviceBuffer<Real>* device = nullptr;
      if (previous_b) {
        device = &magnetic_buffer(PicTileAccess::previous_b(solver), kind);
      } else {
        device = &field_buffer(
            external ? PicTileAccess::external_fields(solver)
                     : PicTileAccess::fields(solver),
            kind);
      }
      std::vector<Real> host(grid.storage_size());
      device->copy_to_host(host.data(), host.size());
      const bool face_x = extents[component].face_x;
      const bool face_y = extents[component].face_y;
      const int begin_x = face_x && tile.coordinate.x != 0 ? 1 : 0;
      const int begin_y = face_y && tile.coordinate.y != 0 ? 1 : 0;
      const int end_x = grid.nx + (face_x ? 1 : 0);
      const int end_y = grid.ny + (face_y ? 1 : 0);
      auto& global_values = field_vector(dense.fields, kind);
      for (int j = begin_y; j < end_y; ++j) {
        const std::size_t gy = tile.y.begin + static_cast<std::size_t>(j);
        if (face_y && py && gy == ny) continue;
        for (int i = begin_x; i < end_x; ++i) {
          const std::size_t gx = tile.x.begin + static_cast<std::size_t>(i);
          if (face_x && px && gx == nx) continue;
          const std::size_t global = row_major(gx, gy, extents[component].nx);
          global_values[global] = host[grid.index(i, j)];
          counts[component][global] = 1;
        }
      }
    }
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_, phase, tasks);

  for (std::size_t component = 0; component < extents.size(); ++component) {
    if (previous_b && component < static_cast<std::size_t>(FieldComponent::bx)) {
      continue;
    }
    auto& values = field_vector(dense.fields,
                                static_cast<FieldComponent>(component));
    allreduce_sum_in_place(*runtime_, values, MPI_DOUBLE,
                           "MPI_Allreduce(distributed PIC fields)");
    allreduce_sum_in_place(*runtime_, counts[component], MPI_INT,
                           "MPI_Allreduce(distributed PIC field ownership)");
    telemetry_.collective_bytes += static_cast<std::uint64_t>(
        values.size() * sizeof(Real) + counts[component].size() * sizeof(int));

    const auto extent = extents[component];
    if (extent.face_x && px) {
      const std::size_t unique_y = extent.face_y && py ? ny : extent.ny;
      for (std::size_t y = 0; y < unique_y; ++y) {
        const std::size_t high = row_major(nx, y, extent.nx);
        if (counts[component][high] != 0) {
          throw std::runtime_error{
              "periodic PIC high-x field duplicate unexpectedly has an owner"};
        }
        values[high] = values[row_major(0, y, extent.nx)];
        counts[component][high] = 1;
      }
    }
    if (extent.face_y && py) {
      for (std::size_t x = 0; x < extent.nx; ++x) {
        const std::size_t high = row_major(x, ny, extent.nx);
        if (counts[component][high] != 0) {
          throw std::runtime_error{
              "periodic PIC high-y field duplicate unexpectedly has an owner"};
        }
        values[high] = values[row_major(x, 0, extent.nx)];
        counts[component][high] = 1;
      }
    }
    require_exact_coverage(counts[component], "PIC field lattice");
  }
  return dense;
}

void PicTileRuntime::apply_fields(const DenseFields& dense, bool external,
                                  bool previous_b, std::string_view phase,
                                  bool* mutation_started) {
  const auto extents = field_extents(
      dense.fields.global_nx, dense.fields.global_ny,
      is_cylindrical(global_config_.geometry));
  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);
  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, &dense, &extents, external, previous_b, px, py]
      (std::size_t local, WorkerContext&) {
    auto& solver = *solvers_[local];
    const Grid2D& grid = PicTileAccess::grid(solver);
    const std::size_t endpoint =
        static_cast<std::size_t>(runtime_->rank()) *
            mapping_.devices_per_rank() + local;
    const TileExtent& tile = topology_.tile(endpoint);
    for (std::size_t component = 0; component < extents.size(); ++component) {
      const auto kind = static_cast<FieldComponent>(component);
      if (previous_b && component < static_cast<std::size_t>(FieldComponent::bx)) {
        continue;
      }
      backend::DeviceBuffer<Real>* device = nullptr;
      if (previous_b) {
        device = &magnetic_buffer(PicTileAccess::previous_b(solver), kind);
      } else {
        device = &field_buffer(
            external ? PicTileAccess::external_fields(solver)
                     : PicTileAccess::fields(solver),
            kind);
      }
      std::vector<Real> host(grid.storage_size());
      device->copy_to_host(host.data(), host.size());
      const auto extent = extents[component];
      const auto& global = field_vector(dense.fields, kind);
      for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
        const long long raw_y = static_cast<long long>(tile.y.begin) + j;
        std::size_t gy = 0;
        if (!map_field_coordinate(raw_y, dense.fields.global_ny,
                                  extent.face_y, py, gy)) {
          continue;
        }
        for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
          const long long raw_x = static_cast<long long>(tile.x.begin) + i;
          std::size_t gx = 0;
          if (!map_field_coordinate(raw_x, dense.fields.global_nx,
                                    extent.face_x, px, gx)) {
            continue;
          }
          host[grid.index(i, j)] =
              global[row_major(gx, gy, extent.nx)];
        }
      }
      device->copy_from_host(host.data(), host.size());
    }
    backend::device_synchronize(nullptr);
  });
  if (mutation_started != nullptr) *mutation_started = true;
  require_worker_success(*workers_, *runtime_, worker_epoch_, phase, tasks);
}

void PicTileRuntime::transfer_fixed_halos(FixedExchangeKind kind,
                                          std::string_view phase) {
  const std::size_t kind_index = static_cast<std::size_t>(kind);
  constexpr std::size_t slots_per_epoch =
      static_cast<std::size_t>(Transport::tag_slots_per_epoch);
  std::vector<std::vector<ByteTransfer>> epoch_transfers;
  collective_try(
      *runtime_, worker_epoch_, phase,
      "PIC halo descriptor preparation failed", -1, [&] {
    std::vector<std::size_t> remote_message_counts(
        static_cast<std::size_t>(runtime_->size()), 0);
    std::size_t local_message_count = 0;
    const auto append_transfer = [&](ByteTransfer transfer) {
      std::size_t epoch = 0;
      if (transfer.peer_rank == runtime_->rank()) {
        transfer.tag_slot = static_cast<std::uint32_t>(
            local_message_count++ % slots_per_epoch);
      } else {
        // Both endpoints filter the same global pair order, so this per-peer
        // ordinal selects the same epoch and tag on both MPI ranks.
        std::size_t& peer_messages = remote_message_counts.at(
            static_cast<std::size_t>(transfer.peer_rank));
        const std::size_t message = peer_messages++;
        epoch = message / slots_per_epoch;
        transfer.tag_slot = static_cast<std::uint32_t>(
            message % slots_per_epoch);
      }
      if (epoch_transfers.size() <= epoch) {
        epoch_transfers.resize(epoch + 1);
      }
      epoch_transfers[epoch].push_back(transfer);
    };

    for (const auto& pair : fixed_exchange_plan_->pairs) {
      const std::size_t capacity = pair.capacity(kind_index);
      if (capacity == 0) continue;
      const Endpoint& first = mapping_.endpoint(pair.first_endpoint);
      const Endpoint& second = mapping_.endpoint(pair.second_endpoint);
      const auto append_local_copy = [&](const Endpoint& source,
                                         const Endpoint& destination) {
        auto& send = halo_buffers_.at(source.rank_local_index)
                         ->send.at(destination.index);
        auto& receive = halo_buffers_.at(destination.rank_local_index)
                            ->receive.at(source.index);
        if (send.size() < capacity || receive.size() < capacity) {
          throw std::logic_error{
              "PIC fixed halo buffers are smaller than the exchange plan"};
        }
        append_transfer(ByteTransfer{
            .peer_rank = runtime_->rank(),
            .send_buffer = send.device_ptr(),
            .receive_buffer = receive.device_ptr(),
            .bytes = capacity * sizeof(Real),
            .residence = BufferResidence::device,
            .send_device = send.owner_device(),
            .receive_device = receive.owner_device(),
        });
      };
      const auto append_remote = [&](const Endpoint& local,
                                     const Endpoint& remote) {
        auto& send = halo_buffers_.at(local.rank_local_index)
                         ->send.at(remote.index);
        auto& receive = halo_buffers_.at(local.rank_local_index)
                            ->receive.at(remote.index);
        if (send.size() < capacity || receive.size() < capacity) {
          throw std::logic_error{
              "PIC fixed halo buffers are smaller than the exchange plan"};
        }
        append_transfer(ByteTransfer{
            .peer_rank = remote.world_rank,
            .send_buffer = send.device_ptr(),
            .receive_buffer = receive.device_ptr(),
            .bytes = capacity * sizeof(Real),
            .residence = BufferResidence::device,
            .send_device = send.owner_device(),
            .receive_device = receive.owner_device(),
        });
      };

      if (pair.first_endpoint == pair.second_endpoint) {
        if (runtime_->rank() == first.world_rank) {
          append_local_copy(first, first);
        }
      } else if (first.world_rank == second.world_rank) {
        if (runtime_->rank() == first.world_rank) {
          append_local_copy(first, second);
          append_local_copy(second, first);
        }
      } else if (runtime_->rank() == first.world_rank) {
        append_remote(first, second);
      } else if (runtime_->rank() == second.world_rank) {
        append_remote(second, first);
      }
    }
      });

  // A tag is scoped by peer rank.  Each rank can therefore fill the same 64
  // slots independently for every peer instead of consuming slots for global
  // endpoint pairs in which it does not participate.  The maximum keeps the
  // collective Transport::begin sequence identical on all ranks; ranks with
  // fewer local messages enter the remaining epochs with an empty span.
  const std::size_t epoch_count = allreduce_max_size(
      *runtime_, epoch_transfers.size(),
      "MPI_Allreduce(distributed PIC transport epoch count)");
  for (std::size_t epoch = 0; epoch < epoch_count; ++epoch) {
    const std::span<const ByteTransfer> transfers =
        epoch < epoch_transfers.size()
            ? std::span<const ByteTransfer>{epoch_transfers[epoch]}
            : std::span<const ByteTransfer>{};
    auto batch = transport_->begin(transfers);
    batch.wait();
    update_transport_telemetry();
  }
}

void PicTileRuntime::exchange_field_halos(std::string_view phase,
                                          bool external,
                                          bool previous_b) {
  const std::size_t first_endpoint =
      static_cast<std::size_t>(runtime_->rank()) *
      mapping_.devices_per_rank();
  constexpr std::uint32_t all_field_components =
      (std::uint32_t{1} << pic::pic_device_halo_component_count) - 1;
  constexpr std::uint32_t magnetic_components =
      (std::uint32_t{1} << static_cast<std::uint32_t>(FieldComponent::bx)) |
      (std::uint32_t{1} << static_cast<std::uint32_t>(FieldComponent::by)) |
      (std::uint32_t{1} << static_cast<std::uint32_t>(FieldComponent::bz));
  const std::uint32_t component_mask =
      previous_b ? magnetic_components : all_field_components;
  auto pack = pic_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(),
      [this, first_endpoint, external, previous_b, component_mask]
      (std::size_t local, WorkerContext& context) {
        const std::size_t endpoint = first_endpoint + local;
        auto& solver = *solvers_[local];
        const Grid2D& grid = PicTileAccess::grid(solver);
        pic::PicDeviceHaloConstComponents components;
        for (std::size_t component = 0;
             component < pic::pic_device_halo_component_count; ++component) {
          if ((component_mask & (std::uint32_t{1} << component)) == 0) {
            continue;
          }
          components.component[component] = previous_b
              ? magnetic_buffer(
                    PicTileAccess::previous_b(solver),
                    static_cast<FieldComponent>(component)).device_ptr()
              : field_buffer(
                    external ? PicTileAccess::external_fields(solver)
                             : PicTileAccess::fields(solver),
                    static_cast<FieldComponent>(component)).device_ptr();
        }
        context.compute_ready.record(nullptr);
        backend::stream_wait_event(context.communication_stream.get(),
                                   context.compute_ready.get());
        for (const auto& pair : fixed_exchange_plan_->pairs) {
          const std::size_t capacity = pair.capacity(field_copy_kind);
          if (capacity == 0 ||
              (pair.first_endpoint != endpoint &&
               pair.second_endpoint != endpoint)) {
            continue;
          }
          const std::size_t peer = pair.first_endpoint == endpoint
              ? pair.second_endpoint : pair.first_endpoint;
          pic::launch_pic_device_halo_pack(
              grid, components,
              halo_buffers_[local]
                  ->outgoing_entries[field_copy_kind][peer],
              capacity, component_mask,
              halo_buffers_[local]->send[peer],
              context.communication_stream.get());
        }
        context.communication_ready.record(context.communication_stream);
        context.communication_ready.synchronize();
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         phase, pack);

  transfer_fixed_halos(FixedExchangeKind::field_copy, phase);

  auto unpack = pic_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(),
      [this, first_endpoint, external, previous_b, component_mask]
      (std::size_t local, WorkerContext& context) {
        const std::size_t endpoint = first_endpoint + local;
        auto& solver = *solvers_[local];
        const Grid2D& grid = PicTileAccess::grid(solver);
        pic::PicDeviceHaloComponents components;
        for (std::size_t component = 0;
             component < pic::pic_device_halo_component_count; ++component) {
          if ((component_mask & (std::uint32_t{1} << component)) == 0) {
            continue;
          }
          components.component[component] = previous_b
              ? magnetic_buffer(
                    PicTileAccess::previous_b(solver),
                    static_cast<FieldComponent>(component)).device_ptr()
              : field_buffer(
                    external ? PicTileAccess::external_fields(solver)
                             : PicTileAccess::fields(solver),
                    static_cast<FieldComponent>(component)).device_ptr();
        }
        for (const auto& pair : fixed_exchange_plan_->pairs) {
          const std::size_t capacity = pair.capacity(field_copy_kind);
          if (capacity == 0 ||
              (pair.first_endpoint != endpoint &&
               pair.second_endpoint != endpoint)) {
            continue;
          }
          const std::size_t peer = pair.first_endpoint == endpoint
              ? pair.second_endpoint : pair.first_endpoint;
          pic::launch_pic_device_halo_unpack(
              grid, halo_buffers_[local]->receive[peer], capacity,
              halo_buffers_[local]
                  ->incoming_entries[field_copy_kind][peer],
              components, component_mask,
              pic::PicDeviceHaloUpdate::assign,
              context.communication_stream.get());
        }
        context.communication_ready.record(context.communication_stream);
        context.communication_ready.synchronize();
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         phase, unpack);
}

void PicTileRuntime::reconcile_fields(std::string_view phase) {
  exchange_field_halos(phase);
  ++telemetry_.state_reconciliations;
}

void PicTileRuntime::reconcile_magnetic(std::string_view phase) {
  // Collecting all components is intentional: E is unchanged in this phase and
  // the common path keeps every guard coherent before particle gather.
  reconcile_fields(phase);
}

void PicTileRuntime::initialize_global_background() {
  std::vector<long double> local_charge;
  std::vector<long double> local_absolute;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-background-charge-storage",
      "distributed PIC background allocation failed", -1, [&] {
        local_charge.assign(solvers_.size(), 0.0L);
        local_absolute.assign(solvers_.size(), 0.0L);
      });
  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, &local_charge, &local_absolute]
      (std::size_t local, WorkerContext&) {
    for (const auto& species : PicTileAccess::species(*solvers_[local])) {
      const auto snapshot = species.to_host();
      for (std::size_t particle = 0; particle < snapshot.x.size(); ++particle) {
        if (snapshot.alive[particle] == 0) continue;
        const long double contribution =
            static_cast<long double>(species.charge()) *
            static_cast<long double>(snapshot.weight[particle]);
        local_charge[local] += contribution;
        local_absolute[local] += std::fabs(contribution);
      }
    }
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-background-local-charge", tasks);
  long double rank_charge =
      std::accumulate(local_charge.begin(), local_charge.end(), 0.0L);
  long double rank_absolute =
      std::accumulate(local_absolute.begin(), local_absolute.end(), 0.0L);
  long double total_charge = 0.0L;
  long double total_absolute = 0.0L;
  check_mpi(MPI_Allreduce(&rank_charge, &total_charge, 1, MPI_LONG_DOUBLE,
                          MPI_SUM,
                          detail::MpiRuntimeNativeAccess::world(*runtime_)),
            "MPI_Allreduce(distributed PIC total charge)");
  check_mpi(MPI_Allreduce(&rank_absolute, &total_absolute, 1,
                          MPI_LONG_DOUBLE, MPI_SUM,
                          detail::MpiRuntimeNativeAccess::world(*runtime_)),
            "MPI_Allreduce(distributed PIC absolute charge)");
  telemetry_.collective_bytes += 2 * sizeof(long double);

  if (periodic_x(global_config_) && periodic_y(global_config_) &&
      !global_config_.neutralizing_background && total_absolute > 0.0L) {
    constexpr long double tolerance =
        64.0L * static_cast<long double>(std::numeric_limits<Real>::epsilon());
    if (!(std::fabs(total_charge) / total_absolute <= tolerance)) {
      throw std::invalid_argument{
          "distributed PIC doubly periodic domain requires zero net particle "
          "charge or neutralizing_background=true"};
    }
  }

  Real density = Real{0};
  if (global_config_.neutralizing_background && total_charge != 0.0L) {
    long double volume =
        static_cast<long double>(global_config_.grid.lx) *
        static_cast<long double>(global_config_.grid.ly);
    if (is_cylindrical(global_config_.geometry)) {
      const long double radial_mid =
          static_cast<long double>(global_config_.grid.origin_x) +
          0.5L * static_cast<long double>(global_config_.grid.lx);
      volume = 2.0L * static_cast<long double>(pi_v<Real>) *
               static_cast<long double>(global_config_.grid.lx) * radial_mid *
               static_cast<long double>(global_config_.grid.ly);
    }
    const long double computed = -total_charge / volume;
    if (!std::isfinite(computed) ||
        std::fabs(computed) >
            static_cast<long double>(std::numeric_limits<Real>::max())) {
      throw std::overflow_error{
          "distributed PIC neutralizing background is not representable"};
    }
    density = static_cast<Real>(computed);
    if (computed != 0.0L && density == Real{0}) {
      throw std::underflow_error{
          "distributed PIC neutralizing background underflows Real"};
    }
  }

  auto publish = pic_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, density](std::size_t local, WorkerContext&) {
    PicTileAccess::set_background(*solvers_[local], true, density);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-background-publish", publish);
}

void PicTileRuntime::reconcile_sources_device(
    bool use_next_charge, std::string_view phase) {
  constexpr std::uint32_t source_mask = 0x0fu;
  const std::size_t first_endpoint =
      static_cast<std::size_t>(runtime_->rank()) *
      mapping_.devices_per_rank();
  auto pack = pic_worker_tasks(
      *runtime_, worker_epoch_, solvers_.size(),
      [this, first_endpoint, use_next_charge]
      (std::size_t local, WorkerContext& context) {
        const std::size_t endpoint = first_endpoint + local;
        auto& solver = *solvers_[local];
        auto& buffers = *halo_buffers_[local];
        const Grid2D& grid = PicTileAccess::grid(solver);
        pic::PicDeviceHaloConstComponents sources;
        auto& current = PicTileAccess::current(solver);
        sources.component[0] = current.jx.device_ptr();
        sources.component[1] = current.jy.device_ptr();
        sources.component[2] = current.jz.device_ptr();
        sources.component[3] =
            PicTileAccess::charge(solver, use_next_charge).values.device_ptr();
        pic::PicDeviceHaloComponents destinations;
        for (std::size_t component = 0; component < 4; ++component) {
          destinations.component[component] =
              buffers.source_work[component].device_ptr();
          backend::device_memset_async(
              destinations.component[component], 0,
              buffers.source_work[component].bytes(),
              context.communication_stream.get());
        }
        context.compute_ready.record(nullptr);
        backend::stream_wait_event(context.communication_stream.get(),
                                   context.compute_ready.get());
        pic::launch_pic_device_halo_accumulate(
            grid, sources, buffers.local_source_entries, destinations,
            source_mask, context.communication_stream.get());
        for (const auto& pair : fixed_exchange_plan_->pairs) {
          const std::size_t capacity = pair.capacity(source_additive_kind);
          if (capacity == 0 ||
              (pair.first_endpoint != endpoint &&
               pair.second_endpoint != endpoint)) {
            continue;
          }
          const std::size_t peer = pair.first_endpoint == endpoint
              ? pair.second_endpoint : pair.first_endpoint;
          pic::launch_pic_device_halo_pack(
              grid, sources,
              buffers.outgoing_entries[source_additive_kind][peer],
              capacity, source_mask, buffers.send[peer],
              context.communication_stream.get());
        }
        context.communication_ready.record(context.communication_stream);
        context.communication_ready.synchronize();
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_, phase, pack);

  transfer_fixed_halos(FixedExchangeKind::source_additive, phase);

  auto unpack = pic_worker_tasks(
      *runtime_, worker_epoch_, solvers_.size(),
      [this, first_endpoint, use_next_charge]
      (std::size_t local, WorkerContext& context) {
        const std::size_t endpoint = first_endpoint + local;
        auto& solver = *solvers_[local];
        auto& buffers = *halo_buffers_[local];
        const Grid2D& grid = PicTileAccess::grid(solver);
        pic::PicDeviceHaloComponents destinations;
        for (std::size_t component = 0; component < 4; ++component) {
          destinations.component[component] =
              buffers.source_work[component].device_ptr();
        }
        for (const auto& pair : fixed_exchange_plan_->pairs) {
          const std::size_t capacity = pair.capacity(source_additive_kind);
          if (capacity == 0 ||
              (pair.first_endpoint != endpoint &&
               pair.second_endpoint != endpoint)) {
            continue;
          }
          const std::size_t peer = pair.first_endpoint == endpoint
              ? pair.second_endpoint : pair.first_endpoint;
          pic::launch_pic_device_halo_unpack(
              grid, buffers.receive[peer], capacity,
              buffers.incoming_entries[source_additive_kind][peer],
              destinations, source_mask, pic::PicDeviceHaloUpdate::add,
              context.communication_stream.get());
        }
        context.communication_ready.record(context.communication_stream);
        context.communication_ready.synchronize();
        PicTileAccess::swap_sources(
            solver, use_next_charge, buffers.source_work);
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_, phase, unpack);
  ++telemetry_.source_reconciliations;
}

void PicTileRuntime::add_source_background_device(
    bool use_next_charge, Real density, std::string_view phase) {
  auto tasks = pic_worker_tasks(
      *runtime_, worker_epoch_, solvers_.size(),
      [this, use_next_charge, density]
      (std::size_t local, WorkerContext& context) {
        auto& solver = *solvers_[local];
        pic::launch_pic_distributed_source_background(
            PicTileAccess::grid(solver),
            PicTileAccess::charge(solver, use_next_charge).values.device_ptr(),
            density, context.compute_stream.get());
        context.compute_ready.record(context.compute_stream);
        context.compute_ready.synchronize();
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_, phase, tasks);
}

void PicTileRuntime::exchange_source_halos_device(
    bool use_next_charge, std::span<const std::size_t> components,
    std::string_view phase) {
  std::uint32_t component_mask = 0;
  for (const std::size_t component : components) {
    if (component >= 4) {
      throw std::invalid_argument{"PIC source halo component is invalid"};
    }
    component_mask |= std::uint32_t{1} << component;
  }
  const std::size_t first_endpoint =
      static_cast<std::size_t>(runtime_->rank()) *
      mapping_.devices_per_rank();
  auto pack = pic_worker_tasks(
      *runtime_, worker_epoch_, solvers_.size(),
      [this, first_endpoint, use_next_charge, component_mask]
      (std::size_t local, WorkerContext& context) {
        const std::size_t endpoint = first_endpoint + local;
        auto& solver = *solvers_[local];
        auto& buffers = *halo_buffers_[local];
        const Grid2D& grid = PicTileAccess::grid(solver);
        pic::PicDeviceHaloConstComponents sources;
        auto& current = PicTileAccess::current(solver);
        sources.component[0] = current.jx.device_ptr();
        sources.component[1] = current.jy.device_ptr();
        sources.component[2] = current.jz.device_ptr();
        sources.component[3] =
            PicTileAccess::charge(solver, use_next_charge).values.device_ptr();
        context.compute_ready.record(nullptr);
        backend::stream_wait_event(context.communication_stream.get(),
                                   context.compute_ready.get());
        for (const auto& pair : fixed_exchange_plan_->pairs) {
          const std::size_t capacity = pair.capacity(source_copy_kind);
          if (capacity == 0 ||
              (pair.first_endpoint != endpoint &&
               pair.second_endpoint != endpoint)) {
            continue;
          }
          const std::size_t peer = pair.first_endpoint == endpoint
              ? pair.second_endpoint : pair.first_endpoint;
          pic::launch_pic_device_halo_pack(
              grid, sources,
              buffers.outgoing_entries[source_copy_kind][peer], capacity,
              component_mask, buffers.send[peer],
              context.communication_stream.get());
        }
        context.communication_ready.record(context.communication_stream);
        context.communication_ready.synchronize();
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_, phase, pack);

  transfer_fixed_halos(FixedExchangeKind::source_copy, phase);

  auto unpack = pic_worker_tasks(
      *runtime_, worker_epoch_, solvers_.size(),
      [this, first_endpoint, use_next_charge, component_mask]
      (std::size_t local, WorkerContext& context) {
        const std::size_t endpoint = first_endpoint + local;
        auto& solver = *solvers_[local];
        auto& buffers = *halo_buffers_[local];
        const Grid2D& grid = PicTileAccess::grid(solver);
        pic::PicDeviceHaloComponents destinations;
        auto& current = PicTileAccess::current(solver);
        destinations.component[0] = current.jx.device_ptr();
        destinations.component[1] = current.jy.device_ptr();
        destinations.component[2] = current.jz.device_ptr();
        destinations.component[3] =
            PicTileAccess::charge(solver, use_next_charge).values.device_ptr();
        for (const auto& pair : fixed_exchange_plan_->pairs) {
          const std::size_t capacity = pair.capacity(source_copy_kind);
          if (capacity == 0 ||
              (pair.first_endpoint != endpoint &&
               pair.second_endpoint != endpoint)) {
            continue;
          }
          const std::size_t peer = pair.first_endpoint == endpoint
              ? pair.second_endpoint : pair.first_endpoint;
          pic::launch_pic_device_halo_unpack(
              grid, buffers.receive[peer], capacity,
              buffers.incoming_entries[source_copy_kind][peer], destinations,
              component_mask, pic::PicDeviceHaloUpdate::assign,
              context.communication_stream.get());
        }
        context.communication_ready.record(context.communication_stream);
        context.communication_ready.synchronize();
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_, phase, unpack);
}

void PicTileRuntime::exchange_source_halos(
    LocalSources& sources, std::span<const std::size_t> components,
    std::string_view phase) {
  if (std::any_of(components.begin(), components.end(),
                  [](std::size_t component) { return component >= 4; })) {
    throw std::invalid_argument{"PIC source halo component is invalid"};
  }
  const std::size_t first_endpoint =
      static_cast<std::size_t>(runtime_->rank()) *
      mapping_.devices_per_rank();
  const auto selected = [components](std::uint32_t component) {
    return std::find(components.begin(), components.end(),
                     static_cast<std::size_t>(component)) != components.end();
  };
  auto pack = pic_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(), [this, first_endpoint, &sources, selected]
      (std::size_t local, WorkerContext&) {
        const std::size_t endpoint = first_endpoint + local;
        const Grid2D& grid = PicTileAccess::grid(*solvers_[local]);
        for (const auto& pair : fixed_exchange_plan_->pairs) {
          const std::size_t capacity = pair.capacity(source_copy_kind);
          if (capacity == 0 ||
              (pair.first_endpoint != endpoint &&
               pair.second_endpoint != endpoint)) {
            continue;
          }
          const std::size_t peer = pair.first_endpoint == endpoint
              ? pair.second_endpoint : pair.first_endpoint;
          const auto& entries =
              fixed_direction(pair, endpoint).entries[source_copy_kind];
          std::vector<Real> payload(capacity, Real{0});
          for (std::size_t index = 0; index < entries.size(); ++index) {
            const auto& entry = entries[index];
            if (!selected(entry.component)) continue;
            payload[index] = sources.endpoint[local][entry.component]
                [grid.index(entry.source_x, entry.source_y)];
          }
          halo_buffers_[local]->send[peer].copy_from_host(
              payload.data(), payload.size());
        }
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         phase, pack);

  transfer_fixed_halos(FixedExchangeKind::source_copy, phase);

  auto unpack = pic_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(), [this, first_endpoint, &sources, selected]
      (std::size_t local, WorkerContext&) {
        const std::size_t endpoint = first_endpoint + local;
        const Grid2D& grid = PicTileAccess::grid(*solvers_[local]);
        for (const auto& pair : fixed_exchange_plan_->pairs) {
          if (pair.capacity(source_copy_kind) == 0 ||
              (pair.first_endpoint != endpoint &&
               pair.second_endpoint != endpoint)) {
            continue;
          }
          const std::size_t peer = pair.first_endpoint == endpoint
              ? pair.second_endpoint : pair.first_endpoint;
          const auto& entries =
              fixed_direction(pair, peer).entries[source_copy_kind];
          if (entries.empty()) continue;
          std::vector<Real> payload(entries.size());
          halo_buffers_[local]->receive[peer].copy_to_host(
              payload.data(), payload.size());
          for (std::size_t index = 0; index < entries.size(); ++index) {
            const auto& entry = entries[index];
            if (!selected(entry.component)) continue;
            if (entry.component >= sources.endpoint[local].size() ||
                entry.destination_x < -grid.nghost ||
                entry.destination_x >= grid.nx + grid.nghost ||
                entry.destination_y < -grid.nghost ||
                entry.destination_y >= grid.ny + grid.nghost) {
              throw std::logic_error{
                  "PIC fixed source exchange plan is out of range"};
            }
            sources.endpoint[local][entry.component]
                [grid.index(entry.destination_x, entry.destination_y)] =
                    payload[index];
          }
        }
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         phase, unpack);
}

void PicTileRuntime::apply_sources(const LocalSources& sources,
                                   bool use_next_charge,
                                   std::string_view phase) {
  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(),
      [this, &sources, use_next_charge]
      (std::size_t local, WorkerContext& context) {
        auto& solver = *solvers_[local];
        const std::size_t size = PicTileAccess::grid(solver).storage_size();
        const auto stream = context.communication_stream.get();
        PicTileAccess::current(solver).jx.copy_from_host_async(
            sources.endpoint[local][0].data(), size, stream);
        PicTileAccess::current(solver).jy.copy_from_host_async(
            sources.endpoint[local][1].data(), size, stream);
        PicTileAccess::current(solver).jz.copy_from_host_async(
            sources.endpoint[local][2].data(), size, stream);
        auto& charge = PicTileAccess::charge(solver, use_next_charge);
        charge.values.copy_from_host_async(
            sources.endpoint[local][3].data(), size, stream);
        context.communication_ready.record(context.communication_stream);
        backend::stream_wait_event(context.compute_stream.get(),
                                   context.communication_ready.get());
        context.compute_ready.record(context.compute_stream);
        context.compute_ready.synchronize();
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_, phase, tasks);
}

PicTileRuntime::DenseSources PicTileRuntime::collect_sources_owned(
    bool use_next_charge, std::string_view phase) {
  const std::size_t nx = topology_.global_nx();
  const std::size_t ny = topology_.global_ny();
  const auto currents = source_extents(
      nx, ny, is_cylindrical(global_config_.geometry));
  const std::array<ComponentExtent, 4> extents{
      currents[0], currents[1], currents[2],
      ComponentExtent{nx, ny, false, false}};
  DenseSources dense;
  dense.sources.global_nx = nx;
  dense.sources.global_ny = ny;
  std::array<std::vector<int>, 4> counts;
  for (std::size_t component = 0; component < extents.size(); ++component) {
    const std::size_t size = checked_product(extents[component].nx,
                                             extents[component].ny,
                                             "PIC owned source");
    source_vector(dense.sources, static_cast<SourceComponent>(component))
        .assign(size, Real{0});
    counts[component].assign(size, 0);
  }
  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);
  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, &dense, &counts, &extents, use_next_charge, px, py, nx, ny]
      (std::size_t local, WorkerContext&) {
    auto& solver = *solvers_[local];
    const Grid2D& grid = PicTileAccess::grid(solver);
    const std::size_t endpoint =
        static_cast<std::size_t>(runtime_->rank()) *
            mapping_.devices_per_rank() + local;
    const TileExtent& tile = topology_.tile(endpoint);
    std::array<std::vector<Real>, 4> host;
    for (auto& values : host) values.resize(grid.storage_size());
    const auto& current = PicTileAccess::current(solver);
    current.jx.copy_to_host(host[0].data(), host[0].size());
    current.jy.copy_to_host(host[1].data(), host[1].size());
    current.jz.copy_to_host(host[2].data(), host[2].size());
    const auto& charge = PicTileAccess::charge(solver, use_next_charge);
    charge.values.copy_to_host(host[3].data(), host[3].size());
    for (std::size_t component = 0; component < extents.size(); ++component) {
      const auto extent = extents[component];
      const int begin_x = extent.face_x && tile.coordinate.x != 0 ? 1 : 0;
      const int begin_y = extent.face_y && tile.coordinate.y != 0 ? 1 : 0;
      const int end_x = grid.nx + (extent.face_x ? 1 : 0);
      const int end_y = grid.ny + (extent.face_y ? 1 : 0);
      auto& global = source_vector(
          dense.sources, static_cast<SourceComponent>(component));
      for (int j = begin_y; j < end_y; ++j) {
        const std::size_t gy = tile.y.begin + static_cast<std::size_t>(j);
        if (extent.face_y && py && gy == ny) continue;
        for (int i = begin_x; i < end_x; ++i) {
          const std::size_t gx = tile.x.begin + static_cast<std::size_t>(i);
          if (extent.face_x && px && gx == nx) continue;
          const std::size_t index = row_major(gx, gy, extent.nx);
          global[index] = host[component][grid.index(i, j)];
          counts[component][index] = 1;
        }
      }
    }
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_, phase, tasks);
  for (std::size_t component = 0; component < extents.size(); ++component) {
    auto& values = source_vector(
        dense.sources, static_cast<SourceComponent>(component));
    allreduce_sum_in_place(*runtime_, values, MPI_DOUBLE,
                           "MPI_Allreduce(distributed PIC owned source)");
    allreduce_sum_in_place(*runtime_, counts[component], MPI_INT,
                           "MPI_Allreduce(distributed PIC source ownership)");
    telemetry_.collective_bytes += static_cast<std::uint64_t>(
        values.size() * sizeof(Real) + counts[component].size() * sizeof(int));
    const auto extent = extents[component];
    if (extent.face_x && px) {
      const std::size_t unique_y = extent.face_y && py ? ny : extent.ny;
      for (std::size_t y = 0; y < unique_y; ++y) {
        const std::size_t high = row_major(nx, y, extent.nx);
        if (counts[component][high] != 0) {
          throw std::runtime_error{
              "periodic PIC high-x source duplicate unexpectedly has an owner"};
        }
        values[high] = values[row_major(0, y, extent.nx)];
        counts[component][high] = 1;
      }
    }
    if (extent.face_y && py) {
      for (std::size_t x = 0; x < extent.nx; ++x) {
        const std::size_t high = row_major(x, ny, extent.nx);
        if (counts[component][high] != 0) {
          throw std::runtime_error{
              "periodic PIC high-y source duplicate unexpectedly has an owner"};
        }
        values[high] = values[row_major(x, 0, extent.nx)];
        counts[component][high] = 1;
      }
    }
    require_exact_coverage(counts[component], "PIC source lattice");
  }
  return dense;
}

void PicTileRuntime::rebuild_periodic_source_duplicates(
    DenseSources& dense) const {
  const std::size_t nx = dense.sources.global_nx;
  const std::size_t ny = dense.sources.global_ny;
  const auto currents = source_extents(
      nx, ny, is_cylindrical(global_config_.geometry));
  const std::array<ComponentExtent, 3> extents{
      currents[0], currents[1], currents[2]};
  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);
  for (std::size_t component = 0; component < extents.size(); ++component) {
    auto& values = source_vector(
        dense.sources, static_cast<SourceComponent>(component));
    const auto extent = extents[component];
    if (extent.face_x && px) {
      const std::size_t unique_y = extent.face_y && py ? ny : extent.ny;
      for (std::size_t y = 0; y < unique_y; ++y) {
        values[row_major(nx, y, extent.nx)] =
            values[row_major(0, y, extent.nx)];
      }
    }
    if (extent.face_y && py) {
      for (std::size_t x = 0; x < extent.nx; ++x) {
        values[row_major(x, ny, extent.nx)] =
            values[row_major(x, 0, extent.nx)];
      }
    }
  }
}

void PicTileRuntime::apply_sources(const DenseSources& dense,
                                   bool use_next_charge,
                                   std::string_view phase) {
  const std::size_t nx = dense.sources.global_nx;
  const std::size_t ny = dense.sources.global_ny;
  const auto currents = source_extents(
      nx, ny, is_cylindrical(global_config_.geometry));
  const std::array<ComponentExtent, 4> extents{
      currents[0], currents[1], currents[2],
      ComponentExtent{nx, ny, false, false}};
  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);
  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, &dense, &extents, use_next_charge, px, py, nx, ny]
      (std::size_t local, WorkerContext&) {
    auto& solver = *solvers_[local];
    const Grid2D& grid = PicTileAccess::grid(solver);
    const std::size_t endpoint =
        static_cast<std::size_t>(runtime_->rank()) *
            mapping_.devices_per_rank() + local;
    const TileExtent& tile = topology_.tile(endpoint);
    auto& current = PicTileAccess::current(solver);
    std::array<backend::DeviceBuffer<Real>*, 4> devices{
        &current.jx, &current.jy, &current.jz,
        &PicTileAccess::charge(solver, use_next_charge).values};
    for (std::size_t component = 0; component < extents.size(); ++component) {
      std::vector<Real> host(grid.storage_size(), Real{0});
      const auto extent = extents[component];
      const auto& global = source_vector(
          dense.sources, static_cast<SourceComponent>(component));
      for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
        std::size_t gy = 0;
        if (!map_field_coordinate(
                static_cast<long long>(tile.y.begin) + j, ny,
                extent.face_y, py, gy)) {
          continue;
        }
        for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
          std::size_t gx = 0;
          if (!map_field_coordinate(
                  static_cast<long long>(tile.x.begin) + i, nx,
                  extent.face_x, px, gx)) {
            continue;
          }
          host[grid.index(i, j)] = global[row_major(gx, gy, extent.nx)];
        }
      }
      devices[component]->copy_from_host(host.data(), host.size());
    }
    backend::device_synchronize(nullptr);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_, phase, tasks);
}

void PicTileRuntime::materialize_charge() {
  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this](std::size_t local, WorkerContext& context) {
    auto& solver = *solvers_[local];
    PicTileAccess::deposit_initial_charge(
        solver, context.compute_stream.get(), context.compute_ready);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-initial-charge-deposit", tasks);
  reconcile_sources_device(false, "pic-initial-charge-reduce");
  const Real background = solvers_.empty()
      ? Real{0}
      : PicTileAccess::background_density(*solvers_.front());
  add_source_background_device(
      false, background, "pic-initial-charge-background");
  const std::array<std::size_t, 4> all_components{0, 1, 2, 3};
  exchange_source_halos_device(
      false, all_components, "pic-initial-charge-halos");
  auto publish = pic_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this](std::size_t local, WorkerContext& context) {
    auto& solver = *solvers_[local];
    ::launch_pic_validate_finite_sources(
        PicTileAccess::grid(solver), nullptr,
        &PicTileAccess::charge(solver, false),
        PicTileAccess::source_finite_error(solver).device_ptr(),
        context.compute_stream.get());
    PicTileAccess::mark_charge_valid(solver);
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-initial-charge-validate", publish);
}

void PicTileRuntime::filter_sources_device() {
  if (filter_stencils_.empty()) return;
  const bool cylindrical = is_cylindrical(global_config_.geometry);
  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);
  const std::size_t first_endpoint =
      static_cast<std::size_t>(runtime_->rank()) *
      mapping_.devices_per_rank();
  const std::array<std::size_t, 1> jz_component{2};
  const auto one_axis =
      [this, cylindrical, px, py, first_endpoint, &jz_component]
      (int axis, Real neighbor_weight, Real center_weight,
       std::string_view phase) {
        auto tasks = pic_worker_tasks(
            *runtime_, worker_epoch_, solvers_.size(),
            [this, cylindrical, px, py, first_endpoint, axis,
             neighbor_weight, center_weight]
            (std::size_t local, WorkerContext& context) {
              const std::size_t endpoint = first_endpoint + local;
              const TileExtent& tile = topology_.tile(endpoint);
              auto& solver = *solvers_[local];
              auto& buffers = *halo_buffers_[local];
              auto& current = PicTileAccess::current(solver);
              pic::launch_pic_distributed_filter_axis(
                  PicTileAccess::grid(solver),
                  device_tile_extent(topology_, tile, px, py),
                  current.jz.device_ptr(), buffers.filter_work.device_ptr(),
                  axis, neighbor_weight, center_weight,
                  cylindrical ? 1 : 0, context.compute_stream.get());
              context.compute_ready.record(context.compute_stream);
              context.compute_ready.synchronize();
              PicTileAccess::swap_filtered_current(
                  solver, buffers.filter_work);
            });
        require_worker_success(*workers_, *runtime_, worker_epoch_, phase,
                               tasks);
        exchange_source_halos_device(false, jz_component, phase);
      };

  for (const auto& stencil : filter_stencils_) {
    for (int pass = 0; pass < stencil.passes; ++pass) {
      one_axis(0, stencil.neighbor_weight, stencil.center_weight,
               "pic-source-filter-x");
      one_axis(1, stencil.neighbor_weight, stencil.center_weight,
               "pic-source-filter-y");
    }
  }
}

void PicTileRuntime::correct_order_four_sources_device() {
  if (global_config_.fdtd_order != 4) return;
  const bool cylindrical = is_cylindrical(global_config_.geometry);
  const bool on_axis = cylindrical &&
      global_config_.grid.origin_x == Real{0};
  const bool px = periodic_x(global_config_);
  const bool py = periodic_y(global_config_);
  const auto& boundary = global_config_.boundary.field;
  const int x_lo = current_boundary_mode(boundary[0]);
  const int x_hi = current_boundary_mode(boundary[1]);
  const int y_lo = current_boundary_mode(boundary[2]);
  const int y_hi = current_boundary_mode(boundary[3]);
  const std::size_t first_endpoint =
      static_cast<std::size_t>(runtime_->rank()) *
      mapping_.devices_per_rank();
  constexpr std::uint32_t inplane_mask = 0x03u;
  const std::array<std::size_t, 2> inplane_components{0, 1};

  auto setup = pic_worker_tasks(
      *runtime_, worker_epoch_, solvers_.size(),
      [this](std::size_t local, WorkerContext& context) {
        auto& solver = *solvers_[local];
        auto& buffers = *halo_buffers_[local];
        auto& current = PicTileAccess::current(solver);
        pic::PicDeviceHaloConstComponents sources;
        sources.component[0] = current.jx.device_ptr();
        sources.component[1] = current.jy.device_ptr();
        pic::PicDeviceHaloComponents destinations;
        destinations.component[0] = buffers.order_rhs[0].device_ptr();
        destinations.component[1] = buffers.order_rhs[1].device_ptr();
        pic::launch_pic_device_components_copy(
            PicTileAccess::grid(solver), sources, destinations,
            inplane_mask, context.compute_stream.get());
        context.compute_ready.record(context.compute_stream);
        context.compute_ready.synchronize();
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-order-four-setup", setup);

  const int iterations = cylindrical ? 20 : 17;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (on_axis) {
      auto axis_pack = pic_worker_tasks(
          *runtime_, worker_epoch_, solvers_.size(),
          [this, first_endpoint](std::size_t local, WorkerContext& context) {
            const std::size_t endpoint = first_endpoint + local;
            const TileExtent& tile = topology_.tile(endpoint);
            auto& solver = *solvers_[local];
            auto& buffers = *halo_buffers_[local];
            const Grid2D& grid = PicTileAccess::grid(solver);
            auto& radial = PicTileAccess::current(solver).jx;
            backend::device_memset_async(
                buffers.axis_values.device_ptr(), 0,
                buffers.axis_values.bytes(),
                context.communication_stream.get());
            pic::launch_pic_distributed_axis_values(
                grid, radial.device_ptr(), buffers.axis_values.device_ptr(),
                tile.coordinate.x == 0 ? 1 : 0,
                context.communication_stream.get());
            pic::PicDeviceHaloConstComponents sources;
            sources.component[0] = radial.device_ptr();
            for (const auto& pair : fixed_exchange_plan_->pairs) {
              const std::size_t capacity =
                  pair.capacity(cylindrical_axis_kind);
              if (capacity == 0 ||
                  (pair.first_endpoint != endpoint &&
                   pair.second_endpoint != endpoint)) {
                continue;
              }
              const std::size_t peer = pair.first_endpoint == endpoint
                  ? pair.second_endpoint : pair.first_endpoint;
              pic::launch_pic_device_halo_pack(
                  grid, sources,
                  buffers.outgoing_entries[cylindrical_axis_kind][peer],
                  capacity, std::uint32_t{1}, buffers.send[peer],
                  context.communication_stream.get());
            }
            context.communication_ready.record(
                context.communication_stream);
            context.communication_ready.synchronize();
          });
      require_worker_success(*workers_, *runtime_, worker_epoch_,
                             "pic-order-four-axis-pack", axis_pack);
      transfer_fixed_halos(FixedExchangeKind::cylindrical_axis,
                           "pic-order-four-axis-transport");
      auto axis_unpack = pic_worker_tasks(
          *runtime_, worker_epoch_, solvers_.size(),
          [this, first_endpoint](std::size_t local, WorkerContext& context) {
            const std::size_t endpoint = first_endpoint + local;
            auto& solver = *solvers_[local];
            auto& buffers = *halo_buffers_[local];
            pic::PicDeviceHaloComponents destinations;
            destinations.component[0] =
                buffers.axis_values.device_ptr();
            for (const auto& pair : fixed_exchange_plan_->pairs) {
              const std::size_t capacity =
                  pair.capacity(cylindrical_axis_kind);
              if (capacity == 0 ||
                  (pair.first_endpoint != endpoint &&
                   pair.second_endpoint != endpoint)) {
                continue;
              }
              const std::size_t peer = pair.first_endpoint == endpoint
                  ? pair.second_endpoint : pair.first_endpoint;
              pic::launch_pic_device_halo_unpack(
                  PicTileAccess::grid(solver), buffers.receive[peer],
                  capacity,
                  buffers.incoming_entries[cylindrical_axis_kind][peer],
                  destinations, std::uint32_t{1},
                  pic::PicDeviceHaloUpdate::assign,
                  context.communication_stream.get());
            }
            context.communication_ready.record(
                context.communication_stream);
            context.communication_ready.synchronize();
          });
      require_worker_success(*workers_, *runtime_, worker_epoch_,
                             "pic-order-four-axis-unpack", axis_unpack);
    }

    auto correct = pic_worker_tasks(
        *runtime_, worker_epoch_, solvers_.size(),
        [this, first_endpoint, cylindrical, on_axis, px, py, x_lo, x_hi,
         y_lo, y_hi](std::size_t local, WorkerContext& context) {
          const std::size_t endpoint = first_endpoint + local;
          const TileExtent& tile = topology_.tile(endpoint);
          auto& solver = *solvers_[local];
          auto& buffers = *halo_buffers_[local];
          auto& current = PicTileAccess::current(solver);
          pic::launch_pic_distributed_current_correct_order4(
              PicTileAccess::grid(solver),
              device_tile_extent(topology_, tile, px, py),
              current.jx.device_ptr(), current.jy.device_ptr(),
              buffers.order_rhs[0].device_ptr(),
              buffers.order_rhs[1].device_ptr(),
              buffers.axis_values.device_ptr(),
              buffers.order_work[0].device_ptr(),
              buffers.order_work[1].device_ptr(), x_lo, x_hi, y_lo, y_hi,
              cylindrical ? 1 : 0, on_axis ? 1 : 0,
              context.compute_stream.get());
          context.compute_ready.record(context.compute_stream);
          context.compute_ready.synchronize();
          PicTileAccess::swap_inplane_current(
              solver, buffers.order_work);
        });
    const std::string_view phase = cylindrical
        ? "pic-order-four-cylindrical-iteration"
        : "pic-order-four-cartesian-iteration";
    require_worker_success(*workers_, *runtime_, worker_epoch_, phase,
                           correct);
    exchange_source_halos_device(false, inplane_components, phase);
  }
}

void PicTileRuntime::update_transport_telemetry() noexcept {
  if (!transport_) return;
  const TransportTelemetry& source = transport_->telemetry();
  telemetry_.transport_epochs = source.epochs;
  telemetry_.transport_messages = source.messages;
  telemetry_.transport_bytes = source.bytes;
  telemetry_.transport_peer_bytes = source.peer_bytes;
  telemetry_.transport_local_staged_bytes = source.local_staged_bytes;
  telemetry_.transport_staged_mpi_bytes = source.staged_mpi_bytes;
  telemetry_.transport_direct_mpi_bytes = source.direct_mpi_bytes;
}

std::vector<std::vector<std::byte>>
PicTileRuntime::exchange_variable_payloads(
    std::span<const std::vector<std::byte>> outgoing,
    std::string_view phase) {
  const std::size_t rank_count = static_cast<std::size_t>(runtime_->size());
  const bool local_shape_valid = outgoing.size() == rank_count;
  collective_require(
      *runtime_, worker_epoch_, local_shape_valid, phase,
      "variable transport payload rank count is inconsistent", -1);

  std::vector<std::uint64_t> send_counts(rank_count, 0);
  std::vector<std::uint64_t> receive_counts(rank_count, 0);
  for (std::size_t peer = 0; peer < rank_count; ++peer) {
    send_counts[peer] = static_cast<std::uint64_t>(outgoing[peer].size());
  }
  receive_counts[static_cast<std::size_t>(runtime_->rank())] =
      send_counts[static_cast<std::size_t>(runtime_->rank())];

  std::vector<ByteTransfer> count_transfers;
  count_transfers.reserve(rank_count > 0 ? rank_count - 1 : 0);
  for (std::size_t peer = 0; peer < rank_count; ++peer) {
    if (static_cast<int>(peer) == runtime_->rank()) continue;
    count_transfers.push_back(ByteTransfer{
        .peer_rank = static_cast<int>(peer),
        .tag_slot = 0,
        .send_buffer = &send_counts[peer],
        .receive_buffer = &receive_counts[peer],
        .bytes = sizeof(std::uint64_t),
        .residence = BufferResidence::host,
    });
  }
  TransferBatch counts = transport_->begin(count_transfers);
  counts.wait();

  bool local_counts_valid = true;
  for (const std::uint64_t count : receive_counts) {
    if (count > static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max())) {
      local_counts_valid = false;
      break;
    }
  }
  collective_require(
      *runtime_, worker_epoch_, local_counts_valid, phase,
      "incoming variable transport payload exceeds host size", -1);

  std::vector<std::vector<std::byte>> send_buffers;
  std::vector<std::vector<std::byte>> receive_buffers;
  std::vector<ByteTransfer> payload_transfers;
  collective_try(
      *runtime_, worker_epoch_, phase,
      "PIC variable transport payload setup failed", -1, [&] {
        send_buffers.resize(rank_count);
        receive_buffers.resize(rank_count);
        payload_transfers.reserve(rank_count);
        for (std::size_t peer = 0; peer < rank_count; ++peer) {
          const std::size_t incoming =
              static_cast<std::size_t>(receive_counts[peer]);
          const std::size_t transfer_size =
              std::max(outgoing[peer].size(), incoming);
          send_buffers[peer].assign(transfer_size, std::byte{0});
          receive_buffers[peer].assign(transfer_size, std::byte{0});
          std::copy(outgoing[peer].begin(), outgoing[peer].end(),
                    send_buffers[peer].begin());
          payload_transfers.push_back(ByteTransfer{
              .peer_rank = static_cast<int>(peer),
              .tag_slot = 1,
              .send_buffer = send_buffers[peer].data(),
              .receive_buffer = receive_buffers[peer].data(),
              .bytes = transfer_size,
              .residence = BufferResidence::host,
          });
        }
      });
  TransferBatch payloads = transport_->begin(payload_transfers);
  payloads.wait();
  for (std::size_t peer = 0; peer < rank_count; ++peer) {
    receive_buffers[peer].resize(
        static_cast<std::size_t>(receive_counts[peer]));
  }
  update_transport_telemetry();
  return receive_buffers;
}

void PicTileRuntime::validate_distributed_particle_ids(
  std::span<const PicSpeciesState> species,
    std::string_view phase) {
  std::vector<std::vector<std::byte>> outgoing;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-particle-id-validation-pack",
      "PIC particle ID validation allocation failed", -1, [&] {
        outgoing.resize(static_cast<std::size_t>(runtime_->size()));
        for (const auto& state : species) {
          for (const std::uint64_t id : state.particles.id) {
            const std::size_t owner = static_cast<std::size_t>(
                id % static_cast<std::uint64_t>(runtime_->size()));
            append_wire_record(outgoing[owner], id);
          }
        }
      });

  const auto incoming = exchange_variable_payloads(
      outgoing, "pic-particle-id-validation-transport");
  collective_try_with_fallback(
      *runtime_, worker_epoch_, phase,
      "PIC checkpoint particle IDs are not globally unique", -1, [&] {
        std::unordered_set<std::uint64_t> ids;
        for (const auto& payload : incoming) {
          for_each_wire_record<std::uint64_t>(
              payload, "PIC stable particle IDs",
              [&ids](std::uint64_t id) {
                if (!ids.insert(id).second) {
                  throw std::invalid_argument{
                      "distributed PIC checkpoint particle IDs must be globally unique across species"};
                }
              });
        }
      });
}

std::unique_ptr<PicTileRuntime::MigrationBatch>
PicTileRuntime::extract_departing_particles() {
  std::unique_ptr<MigrationBatch> batch;
  collective_try(
      *runtime_, worker_epoch_, "pic-migration-count-storage",
      "PIC migration count allocation failed", -1, [&] {
        batch = std::make_unique<MigrationBatch>();
        batch->species_count =
            PicTileAccess::species(*solvers_.front()).size();
        batch->first_endpoint =
            static_cast<std::size_t>(runtime_->rank()) *
            mapping_.devices_per_rank();
        batch->departure_counts.assign(
            solvers_.size(),
            std::vector<std::uint64_t>(batch->species_count, 0));
      });

  auto departure_tasks = pic_worker_tasks(
      *runtime_, worker_epoch_, solvers_.size(),
      [this, &batch](std::size_t local, WorkerContext& context) {
        const TileExtent& tile =
            topology_.tile(batch->first_endpoint + local);
        const auto shape = topology_.shape();
        const int include_x_high =
            tile.coordinate.x + 1 == shape.px ? 1 : 0;
        const int include_y_high =
            tile.coordinate.y + 1 == shape.py ? 1 : 0;
        auto& species = PicTileAccess::species(*solvers_[local]);
        for (std::size_t kind = 0; kind < species.size(); ++kind) {
          batch->departure_counts[local][kind] =
              ::launch_pic_particle_departure_count(
                  species[kind], include_x_high, include_y_high,
                  context.compute_stream.get());
        }
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-migration-departure-check", departure_tasks);

  std::uint64_t local_departures = 0;
  for (const auto& endpoint : batch->departure_counts) {
    local_departures = std::accumulate(
        endpoint.begin(), endpoint.end(), local_departures);
  }
  std::uint64_t global_departures = 0;
  check_mpi(MPI_Allreduce(&local_departures, &global_departures, 1,
                          MPI_UINT64_T, MPI_SUM,
                          detail::MpiRuntimeNativeAccess::world(*runtime_)),
            "MPI_Allreduce(PIC departing-particle count)");
  telemetry_.collective_bytes += sizeof(std::uint64_t);
  ++telemetry_.particle_migrations;
  if (global_departures == 0) return {};

  collective_try(
      *runtime_, worker_epoch_, "pic-migration-departure-storage",
      "PIC migration departure storage allocation failed", -1, [&] {
        batch->departures.resize(solvers_.size());
        for (auto& endpoint : batch->departures) {
          endpoint.resize(batch->species_count);
        }
      });

  auto extract_tasks = pic_worker_tasks(
      *runtime_, worker_epoch_, solvers_.size(),
      [this, &batch](std::size_t local, WorkerContext& context) {
        const TileExtent& tile =
            topology_.tile(batch->first_endpoint + local);
        const auto shape = topology_.shape();
        const int include_x_high =
            tile.coordinate.x + 1 == shape.px ? 1 : 0;
        const int include_y_high =
            tile.coordinate.y + 1 == shape.py ? 1 : 0;
        auto& species = PicTileAccess::species(*solvers_[local]);
        for (std::size_t kind = 0; kind < species.size(); ++kind) {
          if (batch->departure_counts[local][kind] == 0) continue;
          batch->departures[local][kind] =
              pic::extract_pic_departing_particles(
                  species[kind], include_x_high, include_y_high,
                  context.communication_stream.get());
          if (batch->departures[local][kind].x.size() !=
              batch->departure_counts[local][kind]) {
            throw std::runtime_error{
                "PIC migration departure count changed during extraction"};
          }
        }
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-migration-extract", extract_tasks);
  return batch;
}

void PicTileRuntime::route_departing_particles(MigrationBatch& batch) {
  collective_try(
      *runtime_, worker_epoch_, "pic-migration-pack",
      "PIC migration packing failed", -1, [&] {
    batch.arrivals.resize(solvers_.size());
    for (auto& endpoint : batch.arrivals) {
      endpoint.resize(batch.species_count);
    }
    batch.outgoing.resize(static_cast<std::size_t>(runtime_->size()));
    for (std::size_t kind = 0; kind < batch.species_count; ++kind) {
      for (std::size_t local = 0; local < solvers_.size(); ++local) {
        const auto& snapshot = batch.departures[local][kind];
        const std::size_t endpoint = batch.first_endpoint + local;
        for (std::size_t particle = 0; particle < snapshot.x.size();
             ++particle) {
          ParticleRecord record = make_record(snapshot, particle, endpoint);
          std::size_t owner = endpoint;
          if (record.alive != 0) {
            if (periodic_x(global_config_)) {
              const Real wrapped = wrap_periodic(
                  record.x, global_config_.grid.origin_x,
                  global_config_.grid.lx);
              const Real shift = wrapped - record.x;
              record.x = wrapped;
              record.x_prev += shift;
            }
            if (periodic_y(global_config_)) {
              const Real wrapped = wrap_periodic(
                  record.y, global_config_.grid.origin_y,
                  global_config_.grid.ly);
              const Real shift = wrapped - record.y;
              record.y = wrapped;
              record.y_prev += shift;
            }
            owner = owner_for_position(record.x, record.y);
          }
          if (owner >= topology_.endpoint_count()) {
            throw std::runtime_error{
                "distributed PIC particle retained an invalid endpoint owner"};
          }
          if (owner != endpoint) ++batch.local_migrated;
          const int destination_rank = mapping_.endpoint(owner).world_rank;
          if (destination_rank == runtime_->rank()) {
            if (owner < batch.first_endpoint ||
                owner >= batch.first_endpoint + solvers_.size()) {
              throw std::runtime_error{
                  "PIC migration resolved an invalid local endpoint"};
            }
            append_record(
                batch.arrivals[owner - batch.first_endpoint][kind], record);
          } else {
            append_wire_record(
                batch.outgoing[static_cast<std::size_t>(destination_rank)],
                ParticleMigrationRecord{
                    .particle = record,
                    .destination_endpoint = owner,
                    .species = kind,
                });
          }
        }
      }
    }
      });

  const auto incoming = exchange_variable_payloads(
      batch.outgoing, "pic-migration-variable-transport");
  collective_try(
      *runtime_, worker_epoch_, "pic-migration-unpack",
      "PIC migration unpacking failed", -1, [&] {
        for (const auto& payload : incoming) {
          for_each_wire_record<ParticleMigrationRecord>(
              payload, "PIC particle migration",
              [this, &batch](const ParticleMigrationRecord& migration) {
                const std::size_t endpoint =
                    static_cast<std::size_t>(migration.destination_endpoint);
                const std::size_t kind =
                    static_cast<std::size_t>(migration.species);
                if (endpoint < batch.first_endpoint ||
                    endpoint >= batch.first_endpoint + solvers_.size() ||
                    kind >= batch.species_count ||
                    mapping_.endpoint(endpoint).world_rank !=
                        runtime_->rank()) {
                  throw std::runtime_error{
                      "PIC particle migration received an invalid destination"};
                }
                append_record(
                    batch.arrivals[endpoint - batch.first_endpoint][kind],
                    migration.particle);
              });
        }
      });

  collective_try(
      *runtime_, worker_epoch_, "pic-migration-sort",
      "PIC migration sorting failed", -1, [&] {
        for (auto& endpoint : batch.arrivals) {
          for (auto& species : endpoint) sort_particles(species);
        }
      });
}

void PicTileRuntime::commit_migrated_particles(MigrationBatch& batch) {
  auto append_tasks = pic_worker_tasks(
      *runtime_, worker_epoch_, solvers_.size(),
      [this, &batch](std::size_t local, WorkerContext& context) {
        for (std::size_t kind = 0;
             kind < batch.arrivals[local].size(); ++kind) {
          PicTileAccess::append_migrated_species(
              *solvers_[local], kind, batch.arrivals[local][kind],
              context.communication_stream.get());
        }
        context.communication_ready.record(context.communication_stream);
        backend::stream_wait_event(context.compute_stream.get(),
                                   context.communication_ready.get());
        context.compute_ready.record(context.compute_stream);
        context.compute_ready.synchronize();
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-migration-apply", append_tasks);

  std::uint64_t global_migrated = 0;
  check_mpi(MPI_Allreduce(
                &batch.local_migrated, &global_migrated, 1, MPI_UINT64_T,
                MPI_SUM, detail::MpiRuntimeNativeAccess::world(*runtime_)),
            "MPI_Allreduce(PIC migrated-particle telemetry)");
  telemetry_.collective_bytes += sizeof(std::uint64_t);
  telemetry_.migrated_particles += global_migrated;
}

void PicTileRuntime::migrate_particles() {
  auto batch = extract_departing_particles();
  if (!batch) return;
  route_departing_particles(*batch);
  commit_migrated_particles(*batch);
}

std::vector<PicSpeciesState> PicTileRuntime::gather_particles() {
  const std::size_t species_count =
      PicTileAccess::species(*solvers_.front()).size();
  const std::size_t first_endpoint =
      static_cast<std::size_t>(runtime_->rank()) * mapping_.devices_per_rank();
  std::vector<std::vector<pic::ParticleSpecies::HostSnapshot>> snapshots;
  std::vector<PicSpeciesState> result;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-particle-gather-storage",
      "PIC particle gather allocation failed", -1, [&] {
        snapshots.resize(solvers_.size());
        for (auto& endpoint : snapshots) endpoint.resize(species_count);
        result.resize(species_count);
      });
  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this, &snapshots](std::size_t local, WorkerContext&) {
    const auto& species = PicTileAccess::species(*solvers_[local]);
    for (std::size_t kind = 0; kind < species.size(); ++kind) {
      snapshots[local][kind] = species[kind].to_host();
    }
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-particle-gather-snapshot", tasks);

  for (std::size_t kind = 0; kind < species_count; ++kind) {
    std::vector<ParticleRecord> local_records;
    collective_try_with_fallback(
        *runtime_, worker_epoch_, "pic-particle-gather-local-storage",
        "PIC local particle gather allocation failed", -1, [&] {
          const auto& representative =
              PicTileAccess::species(*solvers_.front())[kind];
          result[kind].config = pic::SpeciesConfig{
              representative.name(), representative.charge(),
              representative.mass(), 0};
          std::size_t local_count = 0;
          for (const auto& endpoint : snapshots) {
            local_count += endpoint[kind].x.size();
          }
          local_records.reserve(local_count);
          for (std::size_t local = 0; local < solvers_.size(); ++local) {
            const auto& snapshot = snapshots[local][kind];
            for (std::size_t particle = 0; particle < snapshot.x.size();
                 ++particle) {
              local_records.push_back(
                  make_record(snapshot, particle, first_endpoint + local));
            }
          }
        });
    const auto records =
        allgather_records(*runtime_, local_records, worker_epoch_);
    collective_try_with_fallback(
        *runtime_, worker_epoch_, "pic-particle-gather-result-storage",
        "PIC global particle gather assembly failed", -1, [&] {
          std::unordered_set<std::uint64_t> ids;
          ids.reserve(records.size());
          for (const auto& record : records) {
            if (!ids.insert(record.id).second) {
              throw std::runtime_error{
                  "distributed PIC gather encountered a duplicate stable ID"};
            }
            append_record(result[kind].particles, record);
          }
          sort_particles(result[kind].particles);
          result[kind].config.capacity = result[kind].particles.x.size();
        });
  }
  return result;
}

PicBoundaryState PicTileRuntime::gather_boundary_state() {
  PicBoundaryState result;
  std::array<std::vector<int>, 4> counts;
  const unsigned int corner_mask = global_outflow_corner_mask(global_config_);
  std::vector<int> corner_counts;
  std::vector<std::array<int, 4>> local_primed;
  std::vector<int> local_corner_primed;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-boundary-snapshot-storage",
      "PIC boundary snapshot allocation failed", -1, [&] {
        for (int side = 0; side < 4; ++side) {
          if (global_config_.boundary.field[side] != "outflow") continue;
          const std::size_t stride = mur_global_stride(topology_, side);
          result.mur_history[side].assign(4 * stride, Real{0});
          counts[side].assign(4 * stride, 0);
        }
        if (corner_mask != 0u) {
          result.outflow_corner_history.assign(8, Real{0});
          corner_counts.assign(8, 0);
        }
        local_primed.resize(solvers_.size());
        local_corner_primed.assign(solvers_.size(), 0);
      });
  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(),
      [this, &result, &counts, &corner_counts, &local_primed,
       &local_corner_primed](std::size_t local, WorkerContext&) {
        auto& solver = *solvers_[local];
        const std::size_t endpoint =
            static_cast<std::size_t>(runtime_->rank()) *
                mapping_.devices_per_rank() +
            local;
        const TileExtent& tile = topology_.tile(endpoint);
        for (int side = 0; side < 4; ++side) {
          if (!PicTileAccess::side_uses_outflow(solver, side)) continue;
          const bool primed = PicTileAccess::side_history_primed(
              solver, side);
          const std::vector<Real> history =
              PicTileAccess::side_history(solver, side);
          const std::size_t local_stride = side < 2
              ? tile.y.size() + 1
              : tile.x.size() + 1;
          if (primed && history.size() != 4 * local_stride) {
            throw std::runtime_error{
                "PIC outflow Mur history has the wrong local size"};
          }
          if (!primed && !history.empty() &&
              history.size() != 4 * local_stride) {
            throw std::runtime_error{
                "PIC unprimed Mur history has the wrong local size"};
          }
          local_primed[local][side] = primed ? 1 : 0;
          const std::size_t coordinate = side < 2
              ? tile.coordinate.y
              : tile.coordinate.x;
          const std::size_t global_begin = side < 2
              ? tile.y.begin
              : tile.x.begin;
          const std::size_t global_stride =
              mur_global_stride(topology_, side);
          const std::size_t skip = coordinate == 0 ? 0 : 1;
          for (std::size_t strip = 0; strip < 4; ++strip) {
            for (std::size_t index = skip; index < local_stride; ++index) {
              const std::size_t global =
                  strip * global_stride + global_begin + index;
              if (primed) {
                result.mur_history[side][global] =
                    history[strip * local_stride + index];
              }
              counts[side][global] = 1;
            }
          }
        }

        const unsigned int corner_mask = PicTileAccess::corner_mask(solver);
        const bool corners_primed = PicTileAccess::corners_primed(solver);
        if (corner_mask != 0u) {
          local_corner_primed[local] =
              corners_primed
                  ? static_cast<int>(std::popcount(corner_mask))
                  : 0;
          std::vector<Real> history(8, Real{0});
          if (corners_primed) {
            const auto& corner_history = PicTileAccess::corner_history(solver);
            if (corner_history.size() != history.size()) {
              throw std::runtime_error{
                  "PIC outflow corner history has the wrong size"};
            }
            corner_history.copy_to_host(history.data(), history.size());
          }
          for (unsigned int corner = 0; corner < 4; ++corner) {
            if ((corner_mask & (1u << corner)) == 0u) {
              continue;
            }
            if (corners_primed) {
              result.outflow_corner_history[corner] = history[corner];
              result.outflow_corner_history[4 + corner] = history[4 + corner];
            }
            corner_counts[corner] = 1;
            corner_counts[4 + corner] = 1;
          }
        }
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-boundary-history-gather", tasks);

  std::array<int, 4> local_primed_counts{};
  for (const auto& solver : local_primed) {
    for (int side = 0; side < 4; ++side) {
      local_primed_counts[side] += solver[side];
    }
  }
  std::array<int, 4> global_primed_counts{};
  check_mpi(MPI_Allreduce(local_primed_counts.data(),
                          global_primed_counts.data(), 4, MPI_INT, MPI_SUM,
                          detail::MpiRuntimeNativeAccess::world(*runtime_)),
            "MPI_Allreduce(PIC Mur history priming)");
  for (int side = 0; side < 4; ++side) {
    if (global_config_.boundary.field[side] != "outflow") continue;
    allreduce_sum_in_place(*runtime_, result.mur_history[side], MPI_DOUBLE,
                           "MPI_Allreduce(PIC Mur history)");
    allreduce_sum_in_place(*runtime_, counts[side], MPI_INT,
                           "MPI_Allreduce(PIC Mur history ownership)");
    require_exact_coverage(counts[side], "Mur boundary history");
    const int segments = side < 2
        ? static_cast<int>(topology_.shape().py)
        : static_cast<int>(topology_.shape().px);
    if (global_primed_counts[side] != 0 &&
        global_primed_counts[side] != segments) {
      throw std::runtime_error{
          "distributed PIC Mur histories disagree on priming state"};
    }
    result.mur_primed[side] =
        global_primed_counts[side] == segments ? 1 : 0;
    telemetry_.collective_bytes += static_cast<std::uint64_t>(
        result.mur_history[side].size() *
        (sizeof(Real) + sizeof(int)));
  }

  if (corner_mask != 0u) {
    allreduce_sum_in_place(*runtime_, result.outflow_corner_history,
                           MPI_DOUBLE,
                           "MPI_Allreduce(PIC outflow corner history)");
    allreduce_sum_in_place(*runtime_, corner_counts, MPI_INT,
                           "MPI_Allreduce(PIC outflow corner ownership)");
    int active_corners = 0;
    for (unsigned int corner = 0; corner < 4; ++corner) {
      if ((corner_mask & (1u << corner)) == 0u) continue;
      ++active_corners;
      if (corner_counts[corner] != 1 ||
          corner_counts[4 + corner] != 1) {
        throw std::runtime_error{
            "distributed PIC outflow corner ownership is incomplete"};
      }
    }
    const int local_corner_count = std::accumulate(
        local_corner_primed.begin(), local_corner_primed.end(), 0);
    int global_corner_count = 0;
    check_mpi(MPI_Allreduce(&local_corner_count, &global_corner_count, 1,
                            MPI_INT, MPI_SUM,
                            detail::MpiRuntimeNativeAccess::world(*runtime_)),
              "MPI_Allreduce(PIC corner history priming)");
    if (global_corner_count != 0 && global_corner_count != active_corners) {
      throw std::runtime_error{
          "distributed PIC corner histories disagree on priming state"};
    }
    result.outflow_corners_primed =
        global_corner_count == active_corners;
    telemetry_.collective_bytes += 8 * (sizeof(Real) + sizeof(int));
  }
  return result;
}

void PicTileRuntime::apply_boundary_state(const PicBoundaryState& state) {
  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(), [this, &state](std::size_t local, WorkerContext&) {
        auto& solver = *solvers_[local];
        const std::size_t endpoint =
            static_cast<std::size_t>(runtime_->rank()) *
                mapping_.devices_per_rank() +
            local;
        const TileExtent& tile = topology_.tile(endpoint);
        for (int side = 0; side < 4; ++side) {
          if (!PicTileAccess::side_uses_outflow(solver, side)) continue;
          const std::size_t local_stride = side < 2
              ? tile.y.size() + 1
              : tile.x.size() + 1;
          const std::size_t global_stride =
              mur_global_stride(topology_, side);
          const std::size_t global_begin = side < 2
              ? tile.y.begin
              : tile.x.begin;
          std::vector<Real> local_history(4 * local_stride);
          for (std::size_t strip = 0; strip < 4; ++strip) {
            for (std::size_t index = 0; index < local_stride; ++index) {
              local_history[strip * local_stride + index] =
                  state.mur_history[side][strip * global_stride +
                                          global_begin + index];
            }
          }
          PicTileAccess::restore_side_history(
              solver, side, std::move(local_history),
              state.mur_primed[side] != 0);
        }
        const unsigned int corner_mask = PicTileAccess::corner_mask(solver);
        if (corner_mask != 0u) {
          std::vector<Real> history(8, Real{0});
          for (unsigned int corner = 0; corner < 4; ++corner) {
            if ((corner_mask & (1u << corner)) == 0u) {
              continue;
            }
            history[corner] = state.outflow_corner_history[corner];
            history[4 + corner] =
                state.outflow_corner_history[4 + corner];
          }
          PicTileAccess::corner_history(solver).copy_from_host(
              history.data(), history.size());
          PicTileAccess::set_corners_primed(
              solver, state.outflow_corners_primed);
        }
        backend::device_synchronize(nullptr);
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-boundary-history-restore", tasks);
}

Real PicTileRuntime::cfl_limit() const {
  require_usable();
  const Real local = is_cylindrical(global_config_.geometry)
      ? cyl_cfl_dt(global_config_.grid, global_config_.fdtd_order, Real{1})
      : cfl_dt(global_config_.grid, global_config_.fdtd_order, Real{1});
  return static_cast<Real>(runtime_->allreduce_min(local));
}

[[noreturn]] void PicTileRuntime::poison_collectively(
    std::string_view phase, std::string_view local_message) {
  poison_runtime_collectively(*runtime_, worker_epoch_,
                               lifecycle_.poisoned, phase, local_message);
}

void PicTileRuntime::step(Real dt) {
  require_usable();
  const bool local_dt_valid = std::isfinite(dt) && dt > Real{0};
  collective_require(
      *runtime_, worker_epoch_, local_dt_valid, "pic-step-validate-dt",
      "distributed PIC timestep must be finite and positive", -1);
  const Real minimum_dt = static_cast<Real>(runtime_->allreduce_min(dt));
  const Real maximum_dt = static_cast<Real>(runtime_->allreduce_max(dt));
  collective_require(
      *runtime_, worker_epoch_, minimum_dt == maximum_dt,
      "pic-step-consistent-dt",
      "distributed PIC ranks supplied different timesteps", -1);
  const Real limit = cfl_limit();
  collective_require(
      *runtime_, worker_epoch_, dt <= limit, "pic-step-cfl",
      "distributed PIC timestep exceeds the global CFL limit", -1);

  Real magnetic_dt = dt;
  Real force_dt = Real{0.5} * dt;
  Real previous_b_weight = Real{0.5};
  Real current_b_weight = Real{0.5};
  if (has_previous_dt_) {
    magnetic_dt = std::midpoint(previous_dt_, dt);
    force_dt = magnetic_dt;
    if (!(std::isfinite(magnetic_dt) && magnetic_dt > Real{0}) ||
        magnetic_dt > limit) {
      throw std::invalid_argument{
          "distributed PIC centered leapfrog timestep is invalid"};
    }
    if (previous_dt_ >= dt) {
      const Real ratio = dt / previous_dt_;
      current_b_weight = Real{1} / (Real{1} + ratio);
      previous_b_weight = ratio * current_b_weight;
    } else {
      const Real ratio = previous_dt_ / dt;
      previous_b_weight = Real{1} / (Real{1} + ratio);
      current_b_weight = ratio * previous_b_weight;
    }
  }

  // Everything above is validation-only.  From the first field mutation onward
  // a failure poisons every replica; recovery is from the last committed file.
  try {
    auto faraday = pic_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
        [this, magnetic_dt](std::size_t local, WorkerContext& context) {
      auto& solver = *solvers_[local];
      PicTileAccess::faraday_phase(
          solver, magnetic_dt, context.compute_stream.get(),
          context.compute_ready, context.communication_ready);
    });
    require_worker_success(*workers_, *runtime_, worker_epoch_,
                           "pic-faraday", faraday);
    reconcile_magnetic("pic-post-faraday-magnetic");

    auto push_deposit = pic_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
        [this, force_dt, dt, previous_b_weight, current_b_weight]
        (std::size_t local, WorkerContext& context) {
      auto& solver = *solvers_[local];
      PicTileAccess::push_deposit_phase(
          solver, force_dt, dt, previous_b_weight, current_b_weight,
          context.compute_stream.get(), context.compute_ready);
    });
    require_worker_success(*workers_, *runtime_, worker_epoch_,
                           "pic-push-deposit", push_deposit);

    reconcile_sources_device(true, "pic-source-additive-reconcile");
    const Real background =
        PicTileAccess::background_density(*solvers_.front());
    add_source_background_device(
        true, background, "pic-source-background");
    const std::array<std::size_t, 4> all_components{0, 1, 2, 3};
    exchange_source_halos_device(
        true, all_components, "pic-source-copy-halos");
    migrate_particles();
    filter_sources_device();
    correct_order_four_sources_device();

    auto ampere = pic_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
        [this, dt](std::size_t local, WorkerContext& context) {
      auto& solver = *solvers_[local];
      PicTileAccess::ampere_phase(
          solver, dt, context.compute_stream.get(), context.compute_ready,
          context.communication_ready);
    });
    require_worker_success(*workers_, *runtime_, worker_epoch_,
                           "pic-ampere", ampere);
    reconcile_fields("pic-step-final-fields");
  } catch (const std::exception& error) {
    poison_collectively("pic-step", error.what());
  } catch (...) {
    poison_collectively("pic-step", "non-standard distributed PIC failure");
  }

  previous_dt_ = dt;
  has_previous_dt_ = true;
  ++step_count_;
  ++telemetry_.accepted_steps;
}

void PicTileRuntime::restore(const PicGlobalState& state) {
  restore_impl(state, false);
}

void PicTileRuntime::restore_impl(const PicGlobalState& state,
                                  bool particles_partitioned) {
  require_usable(false);
  if (lifecycle_.seeded) {
    throw std::logic_error{"distributed PIC runtime is already seeded"};
  }
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-restore-validate",
      "distributed PIC restore validation failed", -1, [&] {
    validate_pic_global_fields(state.fields, global_config_.geometry);
    validate_pic_global_fields(state.external_fields, global_config_.geometry);
    const auto extents = field_extents(
        state.fields.global_nx, state.fields.global_ny,
        is_cylindrical(global_config_.geometry));
    const auto require_size = [](std::span<const Real> values,
                                 std::size_t expected, const char* name) {
      if (values.size() != expected) {
        throw std::invalid_argument{std::string{name} +
                                    " has the wrong checkpoint lattice size"};
      }
      require_finite(values, name);
    };
    require_size(state.previous_bx,
                 checked_product(
                     extents[static_cast<std::size_t>(FieldComponent::bx)].nx,
                     extents[static_cast<std::size_t>(FieldComponent::bx)].ny,
                     "PIC previous Bx"),
                 "previous_bx");
    require_size(state.previous_by,
                 checked_product(
                     extents[static_cast<std::size_t>(FieldComponent::by)].nx,
                     extents[static_cast<std::size_t>(FieldComponent::by)].ny,
                     "PIC previous By"),
                 "previous_by");
    require_size(state.previous_bz,
                 checked_product(
                     extents[static_cast<std::size_t>(FieldComponent::bz)].nx,
                     extents[static_cast<std::size_t>(FieldComponent::bz)].ny,
                     "PIC previous Bz"),
                 "previous_bz");
    const auto source_layout = source_extents(
        state.fields.global_nx, state.fields.global_ny,
        is_cylindrical(global_config_.geometry));
    require_size(state.sources.jx,
                 checked_product(source_layout[0].nx, source_layout[0].ny,
                                 "PIC checkpoint Jx"),
                 "checkpoint jx");
    require_size(state.sources.jy,
                 checked_product(source_layout[1].nx, source_layout[1].ny,
                                 "PIC checkpoint Jy"),
                 "checkpoint jy");
    require_size(state.sources.jz,
                 checked_product(source_layout[2].nx, source_layout[2].ny,
                                 "PIC checkpoint Jz"),
                 "checkpoint jz");
    require_size(state.sources.charge,
                 checked_product(state.fields.global_nx,
                                 state.fields.global_ny,
                                 "PIC checkpoint charge"),
                 "checkpoint charge");
    if (state.sources.global_nx != state.fields.global_nx ||
        state.sources.global_ny != state.fields.global_ny) {
      throw std::invalid_argument{
          "distributed PIC checkpoint source and field meshes differ"};
    }
    if (state.has_previous_dt &&
        !(std::isfinite(state.previous_dt) && state.previous_dt > Real{0})) {
      throw std::invalid_argument{
          "distributed PIC checkpoint previous timestep is invalid"};
    }
    if (!std::isfinite(state.background_charge_density)) {
      throw std::invalid_argument{
          "distributed PIC checkpoint background density is invalid"};
    }
    for (int side = 0; side < 4; ++side) {
      const bool outflow =
          global_config_.boundary.field[side] == "outflow";
      const std::size_t expected = outflow
          ? 4 * mur_global_stride(topology_, side)
          : 0;
      if (state.boundary.mur_history[side].size() != expected ||
          state.boundary.mur_primed[side] > 1 ||
          (!outflow && state.boundary.mur_primed[side] != 0)) {
        throw std::invalid_argument{
            "distributed PIC checkpoint Mur history is incompatible"};
      }
      require_finite(std::span<const Real>{state.boundary.mur_history[side]},
                     "PIC Mur checkpoint history");
    }
    const unsigned int corner_mask =
        global_outflow_corner_mask(global_config_);
    const std::size_t expected_corner_size = corner_mask == 0u ? 0 : 8;
    if (state.boundary.outflow_corner_history.size() !=
            expected_corner_size ||
        (corner_mask == 0u && state.boundary.outflow_corners_primed)) {
      throw std::invalid_argument{
          "distributed PIC checkpoint corner history is incompatible"};
    }
    require_finite(
        std::span<const Real>{state.boundary.outflow_corner_history},
                   "PIC corner checkpoint history");
    std::unordered_set<std::uint64_t> particle_ids;
    for (const auto& species : state.species) {
      validate_pic_species_state(species);
      for (const std::uint64_t id : species.particles.id) {
        if (!particle_ids.insert(id).second) {
          throw std::invalid_argument{
              "distributed PIC checkpoint particle IDs must be globally unique across species"};
        }
      }
    }
      });

  std::uint64_t local_hash = 1469598103934665603ULL;
  hash_span(local_hash, std::span<const Real>{state.previous_bx});
  hash_span(local_hash, std::span<const Real>{state.previous_by});
  hash_span(local_hash, std::span<const Real>{state.previous_bz});
  hash_span(local_hash, std::span<const Real>{state.sources.jx});
  hash_span(local_hash, std::span<const Real>{state.sources.jy});
  hash_span(local_hash, std::span<const Real>{state.sources.jz});
  hash_span(local_hash, std::span<const Real>{state.sources.charge});
  hash_scalar(local_hash, state.step_count);
  hash_scalar(local_hash, state.previous_dt);
  hash_scalar(local_hash, state.has_previous_dt);
  hash_scalar(local_hash, state.background_initialized);
  hash_scalar(local_hash, state.background_charge_density);
  for (int side = 0; side < 4; ++side) {
    hash_span(local_hash,
              std::span<const Real>{state.boundary.mur_history[side]});
    hash_scalar(local_hash, state.boundary.mur_primed[side]);
  }
  hash_span(local_hash, std::span<const Real>{
                            state.boundary.outflow_corner_history});
  hash_scalar(local_hash, state.boundary.outflow_corners_primed);
  std::uint64_t minimum_hash = 0;
  std::uint64_t maximum_hash = 0;
  check_mpi(MPI_Allreduce(&local_hash, &minimum_hash, 1, MPI_UINT64_T, MPI_MIN,
                          detail::MpiRuntimeNativeAccess::world(*runtime_)),
            "MPI_Allreduce(distributed PIC restore minimum hash)");
  check_mpi(MPI_Allreduce(&local_hash, &maximum_hash, 1, MPI_UINT64_T, MPI_MAX,
                          detail::MpiRuntimeNativeAccess::world(*runtime_)),
            "MPI_Allreduce(distributed PIC restore maximum hash)");
  collective_require(
      *runtime_, worker_epoch_, minimum_hash == maximum_hash,
      "pic-restore-consistency", "PIC restart state differs between MPI ranks",
      -1);

  std::vector<PicSpeciesState> empty_species;
  DenseFields previous;
  DenseSources sources;
  std::vector<WorkerTask> publish;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-restore-preparation",
      "distributed PIC restore allocation failed", -1, [&] {
    if (particles_partitioned) {
      empty_species.resize(state.species.size());
      for (std::size_t kind = 0; kind < state.species.size(); ++kind) {
        empty_species[kind].config = state.species[kind].config;
        empty_species[kind].config.capacity = 0;
      }
    }
    previous.fields.global_nx = state.fields.global_nx;
    previous.fields.global_ny = state.fields.global_ny;
    previous.fields.bx = state.previous_bx;
    previous.fields.by = state.previous_by;
    previous.fields.bz = state.previous_bz;
    sources.sources = state.sources;
    rebuild_periodic_source_duplicates(sources);
    publish = local_indexed_tasks(
        solvers_.size(), [this, &state](std::size_t local, WorkerContext&) {
          auto& solver = *solvers_[local];
          PicTileAccess::restore_runtime_state(
              solver, static_cast<std::size_t>(state.step_count),
              state.previous_dt, state.has_previous_dt,
              state.background_initialized, state.background_charge_density);
        });
      });

  bool mutation_started = false;
  try {
    if (particles_partitioned) {
      seed(state.fields, &state.external_fields, empty_species);
      mutation_started = true;
      replace_local_species_particles(state.species);
    } else {
      seed(state.fields, &state.external_fields, state.species);
      mutation_started = true;
    }
    apply_fields(previous, false, true, "pic-restore-previous-b");
    apply_sources(sources, false, "pic-restore-sources");
    require_worker_success(*workers_, *runtime_, worker_epoch_,
                           "pic-restore-metadata", publish);
    apply_boundary_state(state.boundary);
    step_count_ = state.step_count;
    previous_dt_ = state.previous_dt;
    has_previous_dt_ = state.has_previous_dt;
  } catch (const std::exception& error) {
    if (lifecycle_.poisoned) throw;
    if (mutation_started) {
      poison_collectively("pic-restore-failure", error.what());
    }
    throw;
  } catch (...) {
    if (lifecycle_.poisoned) throw;
    if (mutation_started) {
      poison_collectively("pic-restore-failure",
                          "non-standard PIC restore failure");
    }
    throw;
  }
}

PicGlobalState PicTileRuntime::gather_state() {
  require_usable();
  reconcile_fields("pic-gather-live-fields");
  PicGlobalState result;
  result.fields = collect_fields(false, false,
                                 "pic-gather-live-fields-final").fields;
  result.external_fields = collect_fields(
      true, false, "pic-gather-external-fields").fields;
  const DenseFields previous = collect_fields(
      false, true, "pic-gather-previous-b");
  result.previous_bx = previous.fields.bx;
  result.previous_by = previous.fields.by;
  result.previous_bz = previous.fields.bz;
  result.sources = collect_sources_owned(
      false, "pic-gather-sources").sources;
  result.species = gather_particles();
  result.step_count = step_count_;
  result.previous_dt = previous_dt_;
  result.has_previous_dt = has_previous_dt_;
  result.background_initialized =
      PicTileAccess::background_initialized(*solvers_.front());
  result.background_charge_density =
      PicTileAccess::background_density(*solvers_.front());
  result.boundary = gather_boundary_state();
  ++telemetry_.global_state_gathers;
  return result;
}

std::vector<PicOwnedShard> PicTileRuntime::local_owned_shards(
    bool include_particles) {
  require_usable();
  const auto extents = field_extents(
      topology_.global_nx(), topology_.global_ny(),
      is_cylindrical(global_config_.geometry));
  const std::size_t first_endpoint =
      static_cast<std::size_t>(runtime_->rank()) *
      mapping_.devices_per_rank();
  std::vector<PicOwnedShard> result(solvers_.size());

  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(),
      [this, include_particles, first_endpoint, &extents, &result]
      (std::size_t local, WorkerContext&) {
        auto& solver = *solvers_[local];
        const std::size_t endpoint = first_endpoint + local;
        const TileExtent& tile = topology_.tile(endpoint);
        const Grid2D& grid = PicTileAccess::grid(solver);
        PicOwnedShard& shard = result[local];
        shard.endpoint = endpoint;
        shard.tile_x = tile.coordinate.x;
        shard.tile_y = tile.coordinate.y;
        shard.offset_x = tile.x.begin;
        shard.offset_y = tile.y.begin;
        shard.owned_nx = tile.x.size();
        shard.owned_ny = tile.y.size();

        const auto extract_fields =
            [&grid, &tile, &extents](YeeField2D<Real>& source,
                                     PicOwnedFields& destination) {
          for (std::size_t component = 0; component < extents.size();
               ++component) {
            const auto kind = static_cast<FieldComponent>(component);
            const ComponentExtent extent = extents[component];
            const std::size_t skip_x =
                extent.face_x && tile.coordinate.x != 0 ? 1 : 0;
            const std::size_t skip_y =
                extent.face_y && tile.coordinate.y != 0 ? 1 : 0;
            PicOwnedArray& owned = owned_field_array(destination, kind);
            owned.offset_x = tile.x.begin + skip_x;
            owned.offset_y = tile.y.begin + skip_y;
            owned.nx = tile.x.size() + (extent.face_x ? 1 : 0) - skip_x;
            owned.ny = tile.y.size() + (extent.face_y ? 1 : 0) - skip_y;
            owned.values.resize(checked_product(
                owned.nx, owned.ny, "PIC owned field component"));

            std::vector<Real> padded(grid.storage_size());
            field_buffer(source, kind).copy_to_host(
                padded.data(), padded.size());
            for (std::size_t y = 0; y < owned.ny; ++y) {
              const int local_y = static_cast<int>(y + skip_y);
              for (std::size_t x = 0; x < owned.nx; ++x) {
                const int local_x = static_cast<int>(x + skip_x);
                owned.values[row_major(x, y, owned.nx)] =
                    padded[grid.index(local_x, local_y)];
              }
            }
          }
        };
        extract_fields(PicTileAccess::fields(solver), shard.fields);
        extract_fields(PicTileAccess::external_fields(solver),
                       shard.external_fields);

        if (include_particles) {
          const auto& species = PicTileAccess::species(solver);
          shard.species.reserve(species.size());
          for (const auto& local_species : species) {
            PicSpeciesState state;
            state.config = pic::SpeciesConfig{
                local_species.name(), local_species.charge(),
                local_species.mass(), 0};
            state.particles = local_species.to_host();
            sort_particles(state.particles);
            state.config.capacity = state.particles.x.size();
            shard.species.push_back(std::move(state));
          }
        }
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-local-owned-shards", tasks);
  ++telemetry_.local_shard_extractions;
  return result;
}

std::vector<std::uint64_t> PicTileRuntime::alive_counts() {
  require_usable();
  const std::size_t species_count =
      PicTileAccess::species(*solvers_.front()).size();
  std::vector<std::vector<std::uint64_t>> endpoint_counts(
      solvers_.size(), std::vector<std::uint64_t>(species_count, 0));
  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(),
      [this, &endpoint_counts](std::size_t local, WorkerContext&) {
        const auto& species = PicTileAccess::species(*solvers_[local]);
        for (std::size_t kind = 0; kind < species.size(); ++kind) {
          endpoint_counts[local][kind] = static_cast<std::uint64_t>(
              pic::alive_count(species[kind]));
        }
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-alive-counts-local", tasks);

  std::vector<std::uint64_t> result(species_count, 0);
  for (const auto& endpoint : endpoint_counts) {
    for (std::size_t kind = 0; kind < species_count; ++kind) {
      result[kind] += endpoint[kind];
    }
  }
  allreduce_sum_in_place(*runtime_, result, MPI_UINT64_T,
                         "MPI_Allreduce(distributed PIC alive counts)");
  telemetry_.collective_bytes += static_cast<std::uint64_t>(
      result.size() * sizeof(std::uint64_t));
  return result;
}

std::vector<Real> PicTileRuntime::kinetic_energies() {
  require_usable();
  const std::size_t species_count =
      PicTileAccess::species(*solvers_.front()).size();
  std::vector<std::vector<long double>> endpoint_energies(
      solvers_.size(), std::vector<long double>(species_count, 0.0L));
  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(),
      [this, &endpoint_energies](std::size_t local, WorkerContext&) {
        const auto& species = PicTileAccess::species(*solvers_[local]);
        for (std::size_t kind = 0; kind < species.size(); ++kind) {
          endpoint_energies[local][kind] = static_cast<long double>(
              pic::total_kinetic_energy(species[kind]));
        }
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-kinetic-energies-local", tasks);

  std::vector<long double> global(species_count, 0.0L);
  for (const auto& endpoint : endpoint_energies) {
    for (std::size_t kind = 0; kind < species_count; ++kind) {
      global[kind] += endpoint[kind];
    }
  }
  allreduce_sum_in_place(*runtime_, global, MPI_LONG_DOUBLE,
                         "MPI_Allreduce(distributed PIC kinetic energies)");
  telemetry_.collective_bytes += static_cast<std::uint64_t>(
      global.size() * sizeof(long double));

  std::vector<Real> result(species_count, Real{0});
  for (std::size_t kind = 0; kind < species_count; ++kind) {
    if (!(std::isfinite(global[kind]) && global[kind] >= 0.0L) ||
        global[kind] >
            static_cast<long double>(std::numeric_limits<Real>::max())) {
      throw std::overflow_error{
          "distributed PIC kinetic energy is not representable"};
    }
    result[kind] = static_cast<Real>(global[kind]);
    if (global[kind] != 0.0L && result[kind] == Real{0}) {
      throw std::underflow_error{
          "distributed PIC kinetic energy is not representable"};
    }
  }
  return result;
}

Real PicTileRuntime::total_em_energy() {
  require_usable();
  std::vector<Real> endpoint_energy(solvers_.size(), Real{0});
  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(),
      [this, &endpoint_energy](std::size_t local, WorkerContext&) {
        endpoint_energy[local] = pic::total_em_energy(*solvers_[local]);
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-total-em-energy", tasks);

  long double rank_energy = 0.0L;
  for (const Real value : endpoint_energy) {
    rank_energy += static_cast<long double>(value);
  }
  long double global_energy = 0.0L;
  check_mpi(MPI_Allreduce(&rank_energy, &global_energy, 1, MPI_LONG_DOUBLE,
                          MPI_SUM,
                          detail::MpiRuntimeNativeAccess::world(*runtime_)),
            "MPI_Allreduce(distributed PIC EM energy)");
  telemetry_.collective_bytes += sizeof(long double);
  if (!(std::isfinite(global_energy) && global_energy >= 0.0L) ||
      global_energy >
          static_cast<long double>(std::numeric_limits<Real>::max())) {
    throw std::overflow_error{
        "distributed PIC electromagnetic energy is not representable"};
  }
  const Real result = static_cast<Real>(global_energy);
  if (global_energy != 0.0L && result == Real{0}) {
    throw std::underflow_error{
        "distributed PIC electromagnetic energy is not representable"};
  }
  return result;
}

Real PicTileRuntime::gauss_residual() {
  require_usable();
  std::vector<Real> endpoint_residual(solvers_.size(), Real{0});
  std::vector<long double> endpoint_volume(solvers_.size(), 0.0L);
  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_,
      solvers_.size(),
      [this, &endpoint_residual, &endpoint_volume]
      (std::size_t local, WorkerContext&) {
        auto& solver = *solvers_[local];
        endpoint_residual[local] = pic::gauss_residual(solver);
        const Grid2D grid = solver.grid();
        const long double lx = static_cast<long double>(grid.lx);
        const long double ly = static_cast<long double>(grid.ly);
        if (is_cylindrical(global_config_.geometry)) {
          const long double radial_mid =
              static_cast<long double>(grid.origin_x) + 0.5L * lx;
          endpoint_volume[local] =
              2.0L * static_cast<long double>(pi_v<Real>) * lx *
              radial_mid * ly;
        } else {
          endpoint_volume[local] = lx * ly;
        }
        if (!(std::isfinite(endpoint_volume[local]) &&
              endpoint_volume[local] > 0.0L)) {
          throw std::overflow_error{
              "distributed PIC diagnostic tile volume is not representable"};
        }
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-gauss-residual", tasks);

  std::array<long double, 2> rank_totals{0.0L, 0.0L};
  for (std::size_t local = 0; local < endpoint_residual.size(); ++local) {
    const long double residual = endpoint_residual[local];
    rank_totals[0] += residual * residual * endpoint_volume[local];
    rank_totals[1] += endpoint_volume[local];
  }
  std::array<long double, 2> global_totals{0.0L, 0.0L};
  check_mpi(MPI_Allreduce(rank_totals.data(), global_totals.data(), 2,
                          MPI_LONG_DOUBLE, MPI_SUM,
                          detail::MpiRuntimeNativeAccess::world(*runtime_)),
            "MPI_Allreduce(distributed PIC Gauss residual)");
  telemetry_.collective_bytes += 2 * sizeof(long double);
  if (!(std::isfinite(global_totals[0]) && global_totals[0] >= 0.0L &&
        std::isfinite(global_totals[1]) && global_totals[1] > 0.0L)) {
    throw std::overflow_error{
        "distributed PIC Gauss residual is not representable"};
  }
  const long double rms = std::sqrt(global_totals[0] / global_totals[1]);
  if (!std::isfinite(rms) ||
      rms > static_cast<long double>(std::numeric_limits<Real>::max())) {
    throw std::overflow_error{
        "distributed PIC Gauss residual is not representable"};
  }
  const Real result = static_cast<Real>(rms);
  if (rms != 0.0L && result == Real{0}) {
    throw std::underflow_error{
        "distributed PIC Gauss residual is not representable"};
  }
  return result;
}

CheckpointMetadata PicTileRuntime::checkpoint_metadata(
    std::uint64_t step, double time, std::string_view unit_system,
    std::span<const pic::SpeciesConfig> species) const {
  for (const auto& entry : species) {
    if (entry.name.empty() || !std::isfinite(entry.charge) ||
        !(std::isfinite(entry.mass) && entry.mass > Real{0})) {
      throw std::invalid_argument{
          "PIC checkpoint species configuration is invalid"};
    }
  }
  CheckpointMetadata metadata{
      .schema = std::string{checkpoint_schema},
      .physics = "pic",
      .precision = "float64",
      .geometry = global_config_.geometry,
      .unit_system = std::string{unit_system},
      .global_nx = static_cast<std::uint64_t>(topology_.global_nx()),
      .global_ny = static_cast<std::uint64_t>(topology_.global_ny()),
      .boundary_signature = pic_boundary_signature(global_config_),
      .species_signature = pic_species_signature(species),
      .background_signature = pic_background_signature(global_config_),
      .numerics_signature = pic_numerics_signature(global_config_),
      .step = step,
      .time = time,
  };
  validate_checkpoint_metadata(metadata);
  return metadata;
}

void PicTileRuntime::write_checkpoint(
    const std::filesystem::path& path, std::uint64_t step, double time,
    std::string_view unit_system,
    std::span<const std::uint8_t> diagnostic_state) {
  require_usable();
  try {
    write_checkpoint_impl(path, step, time, unit_system, diagnostic_state);
  } catch (const std::exception& error) {
    poison_collectively("pic-checkpoint-failure", error.what());
  } catch (...) {
    poison_collectively("pic-checkpoint-failure",
                        "non-standard PIC checkpoint failure");
  }
}

CheckpointMetadata PicTileRuntime::prepare_checkpoint_metadata(
    std::uint64_t step, double time, std::string_view unit_system,
    std::span<const std::uint8_t> diagnostic_state,
    std::vector<std::uint8_t>& encoded_diagnostic_state) {
  std::vector<pic::SpeciesConfig> configurations;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-checkpoint-species-storage",
      "PIC checkpoint species configuration allocation failed", -1, [&] {
    const auto& species_list = PicTileAccess::species(*solvers_.front());
    configurations.reserve(species_list.size());
    for (const auto& species : species_list) {
      configurations.push_back(pic::SpeciesConfig{
          species.name(), species.charge(), species.mass(), 0});
    }
  });

  encoded_diagnostic_state =
      collect_checkpoint_diagnostic_state(*runtime_, diagnostic_state);
  std::optional<CheckpointMetadata> metadata;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-checkpoint-metadata",
      "PIC checkpoint metadata preparation failed", -1, [&] {
    metadata.emplace(
        checkpoint_metadata(step, time, unit_system, configurations));
    metadata->diagnostic_state_bytes = encoded_diagnostic_state.size();
    validate_checkpoint_metadata(*metadata);
  });
  return std::move(*metadata);
}

PicCheckpointSnapshot PicTileRuntime::capture_checkpoint_snapshot(
    std::uint64_t step) {
  reconcile_fields("pic-checkpoint-live-fields");
  PicCheckpointSnapshot snapshot;
  std::vector<WorkerTask> stage;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-checkpoint-stage-storage",
      "PIC checkpoint tile staging allocation failed", -1, [&] {
    snapshot.local_tiles.resize(solvers_.size());
    const std::size_t first_endpoint =
        static_cast<std::size_t>(runtime_->rank()) *
        mapping_.devices_per_rank();
    stage = local_indexed_tasks(
        solvers_.size(), [this, first_endpoint, &snapshot]
        (std::size_t local, WorkerContext&) {
          auto& solver = *solvers_[local];
          auto& checkpoint = snapshot.local_tiles[local];
          checkpoint.tile = topology_.tile(first_endpoint + local);
          checkpoint.grid = PicTileAccess::grid(solver);
          const std::size_t size = checkpoint.grid.storage_size();
          auto& fields = PicTileAccess::fields(solver);
          auto& external_fields = PicTileAccess::external_fields(solver);
          for (std::size_t component = 0; component < 6; ++component) {
            checkpoint.fields[component].resize(size);
            checkpoint.external_fields[component].resize(size);
            field_buffer(fields, static_cast<FieldComponent>(component))
                .copy_to_host(checkpoint.fields[component].data(), size);
            field_buffer(external_fields,
                         static_cast<FieldComponent>(component))
                .copy_to_host(checkpoint.external_fields[component].data(),
                              size);
          }
          auto& previous_b = PicTileAccess::previous_b(solver);
          for (std::size_t component = 0; component < 3; ++component) {
            checkpoint.previous_b[component].resize(size);
            magnetic_buffer(
                previous_b, static_cast<FieldComponent>(component + 3))
                .copy_to_host(checkpoint.previous_b[component].data(), size);
          }
          for (auto& source : checkpoint.sources) source.resize(size);
          const auto& current = PicTileAccess::current(solver);
          current.jx.copy_to_host(checkpoint.sources[0].data(), size);
          current.jy.copy_to_host(checkpoint.sources[1].data(), size);
          current.jz.copy_to_host(checkpoint.sources[2].data(), size);
          PicTileAccess::charge(solver, false).values.copy_to_host(
              checkpoint.sources[3].data(), size);
        });
  });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-checkpoint-local-lattice-stage", stage);

  snapshot.step = step_count_;
  snapshot.previous_dt = previous_dt_;
  snapshot.has_previous_dt = has_previous_dt_;
  snapshot.background_initialized =
      PicTileAccess::background_initialized(*solvers_.front());
  snapshot.background_charge_density =
      PicTileAccess::background_density(*solvers_.front());
  snapshot.boundary = gather_boundary_state();
  snapshot.local_species = snapshot_local_particles();
  collective_require(
      *runtime_, worker_epoch_, snapshot.step == step, "pic-checkpoint-step",
      "checkpoint step differs from the committed PIC step");
  return snapshot;
}

void PicTileRuntime::write_checkpoint_lattices(
    ParallelCheckpointWriter& writer,
    const PicCheckpointSnapshot& snapshot) {
  const bool cylindrical = is_cylindrical(global_config_.geometry);
  const auto fields = field_extents(
      topology_.global_nx(), topology_.global_ny(), cylindrical);
  const auto sources = source_extents(
      topology_.global_nx(), topology_.global_ny(), cylindrical);
  const auto write_lattice =
      [this, &writer, &snapshot]
      (std::string_view name, const ComponentExtent& extent,
       auto&& select) {
        const auto shape = pic_checkpoint_shape(extent);
        std::vector<DatasetHyperslab> slabs;
        std::vector<Real> packed;
        collective_try_with_fallback(
            *runtime_, worker_epoch_,
            "pic-checkpoint-local-lattice-pack",
            "PIC checkpoint lattice packing failed", -1, [&] {
          slabs = pic_checkpoint_slabs(
              mapping_, topology_, runtime_->rank(), extent);
          packed = pack_local_pic_checkpoint_values(
              snapshot.local_tiles, extent, slabs, select);
        });
        writer.write_dataset(name, pic_checkpoint_real_type(), shape, slabs,
                             packed.data(), packed.size());
        ++telemetry_.checkpoint_local_lattice_writes;
      };

  for (std::size_t component = 0; component < fields.size(); ++component) {
    write_lattice(
        pic_field_dataset_names[component], fields[component],
        [component](const PicCheckpointTile& local)
            -> const std::vector<Real>& {
          return local.fields[component];
        });
    write_lattice(
        pic_external_field_dataset_names[component], fields[component],
        [component](const PicCheckpointTile& local)
            -> const std::vector<Real>& {
          return local.external_fields[component];
        });
  }
  for (std::size_t component = 0; component < 3; ++component) {
    write_lattice(
        pic_previous_b_dataset_names[component], fields[component + 3],
        [component](const PicCheckpointTile& local)
            -> const std::vector<Real>& {
          return local.previous_b[component];
        });
    write_lattice(
        pic_source_dataset_names[component], sources[component],
        [component](const PicCheckpointTile& local)
            -> const std::vector<Real>& {
          return local.sources[component];
        });
  }
  write_lattice(
      pic_source_dataset_names[3],
      ComponentExtent{topology_.global_nx(), topology_.global_ny(), false,
                      false},
      [](const PicCheckpointTile& local) -> const std::vector<Real>& {
        return local.sources[3];
      });
}

void PicTileRuntime::write_checkpoint_runtime(
    ParallelCheckpointWriter& writer,
    const PicCheckpointSnapshot& snapshot) {
  const std::array<Real, 2> runtime_reals{
      snapshot.previous_dt, snapshot.background_charge_density};
  const std::array<std::uint64_t, 1> runtime_integers{snapshot.step};
  const std::array<std::uint8_t, 7> runtime_flags{
      static_cast<std::uint8_t>(snapshot.has_previous_dt),
      static_cast<std::uint8_t>(snapshot.background_initialized),
      snapshot.boundary.mur_primed[0], snapshot.boundary.mur_primed[1],
      snapshot.boundary.mur_primed[2], snapshot.boundary.mur_primed[3],
      static_cast<std::uint8_t>(
          snapshot.boundary.outflow_corners_primed)};
  write_pic_checkpoint_1d(
      writer, *runtime_, worker_epoch_, "pic/runtime/reals",
      pic_checkpoint_real_type(), std::span<const Real>{runtime_reals}, false);
  write_pic_checkpoint_1d(
      writer, *runtime_, worker_epoch_, "pic/runtime/integers",
      CheckpointValueType::uint64,
      std::span<const std::uint64_t>{runtime_integers}, false);
  write_pic_checkpoint_1d(
      writer, *runtime_, worker_epoch_, "pic/runtime/flags",
      CheckpointValueType::uint8,
      std::span<const std::uint8_t>{runtime_flags}, false);
  for (int side = 0; side < 4; ++side) {
    if (global_config_.boundary.field[side] != "outflow") continue;
    write_pic_checkpoint_1d(
        writer, *runtime_, worker_epoch_,
        pic_mur_dataset_names[static_cast<std::size_t>(side)],
        pic_checkpoint_real_type(),
        std::span<const Real>{snapshot.boundary.mur_history[side]}, false);
  }
  if (!snapshot.boundary.outflow_corner_history.empty()) {
    write_pic_checkpoint_1d(
        writer, *runtime_, worker_epoch_, "pic/boundary/outflow_corners",
        pic_checkpoint_real_type(),
        std::span<const Real>{snapshot.boundary.outflow_corner_history},
        false);
  }
}

void PicTileRuntime::write_checkpoint_particles(
    ParallelCheckpointWriter& writer,
    const PicCheckpointSnapshot& snapshot) {
  std::vector<std::uint64_t> particle_counts;
  std::vector<std::uint64_t> particle_offsets;
  std::vector<std::array<std::string, 11>> particle_dataset_names;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-checkpoint-particle-storage",
      "PIC checkpoint particle metadata allocation failed", -1, [&] {
    particle_counts.resize(snapshot.local_species.size());
    particle_offsets.resize(snapshot.local_species.size());
    particle_dataset_names.resize(snapshot.local_species.size());
    for (std::size_t kind = 0;
         kind < snapshot.local_species.size(); ++kind) {
      const std::string prefix =
          "pic/particles/" + std::to_string(kind) + '/';
      for (std::size_t field = 0;
           field < pic_particle_dataset_suffixes.size(); ++field) {
        particle_dataset_names[kind][field] =
            prefix + std::string{pic_particle_dataset_suffixes[field]};
      }
    }
  });

  for (std::size_t kind = 0;
       kind < snapshot.local_species.size(); ++kind) {
    const std::uint64_t local_count =
        snapshot.local_species[kind].particles.id.size();
    check_mpi(MPI_Allreduce(
                  &local_count, &particle_counts[kind], 1, MPI_UINT64_T,
                  MPI_SUM,
                  detail::MpiRuntimeNativeAccess::world(*runtime_)),
              "MPI_Allreduce(PIC checkpoint particle count)");
    std::uint64_t offset = 0;
    check_mpi(MPI_Exscan(
                  &local_count, &offset, 1, MPI_UINT64_T, MPI_SUM,
                  detail::MpiRuntimeNativeAccess::world(*runtime_)),
              "MPI_Exscan(PIC checkpoint particle offset)");
    particle_offsets[kind] = runtime_->rank() == 0 ? 0 : offset;
  }
  write_pic_checkpoint_1d(
      writer, *runtime_, worker_epoch_, "pic/particles/counts",
      CheckpointValueType::uint64,
      std::span<const std::uint64_t>{particle_counts}, false);

  for (std::size_t kind = 0;
       kind < snapshot.local_species.size(); ++kind) {
    const auto& particles = snapshot.local_species[kind].particles;
    std::size_t field = 0;
    const auto write_real = [&](std::span<const Real> values) {
      const std::string_view name = particle_dataset_names[kind][field++];
      write_pic_checkpoint_local_1d(
          writer, *runtime_, worker_epoch_, name,
          pic_checkpoint_real_type(), particle_counts[kind],
          particle_offsets[kind], values);
    };
    write_real(particles.x);
    write_real(particles.y);
    write_real(particles.x_prev);
    write_real(particles.y_prev);
    write_real(particles.vx);
    write_real(particles.vy);
    write_real(particles.vz);
    write_real(particles.vphi_deposit);
    write_real(particles.weight);
    write_pic_checkpoint_local_1d(
        writer, *runtime_, worker_epoch_, particle_dataset_names[kind][9],
        CheckpointValueType::uint8, particle_counts[kind],
        particle_offsets[kind],
        std::span<const std::uint8_t>{particles.alive});
    write_pic_checkpoint_local_1d(
        writer, *runtime_, worker_epoch_, particle_dataset_names[kind][10],
        CheckpointValueType::uint64, particle_counts[kind],
        particle_offsets[kind],
        std::span<const std::uint64_t>{particles.id});
  }
}

void PicTileRuntime::write_checkpoint_impl(
    const std::filesystem::path& path, std::uint64_t step, double time,
    std::string_view unit_system,
    std::span<const std::uint8_t> diagnostic_state) {
  runtime_->require_orchestration_thread();
  const bool locally_ready =
      !lifecycle_.closed && lifecycle_.seeded && !lifecycle_.poisoned;
  collective_require(
      *runtime_, worker_epoch_, locally_ready, "pic-checkpoint-ready",
      lifecycle_.closed
          ? "distributed PIC runtime is closed"
          : lifecycle_.poisoned
                ? "distributed PIC runtime is poisoned"
                : "distributed PIC runtime is not seeded");

  std::vector<std::uint8_t> encoded_diagnostic_state;
  CheckpointMetadata metadata = prepare_checkpoint_metadata(
      step, time, unit_system, diagnostic_state, encoded_diagnostic_state);
  ParallelCheckpointWriter writer{
      *runtime_, path, std::move(metadata)};
  try {
    const PicCheckpointSnapshot snapshot =
        capture_checkpoint_snapshot(step);
    write_checkpoint_lattices(writer, snapshot);
    write_checkpoint_runtime(writer, snapshot);
    write_checkpoint_particles(writer, snapshot);
    writer.write_diagnostic_state(encoded_diagnostic_state);
    writer.commit();
  } catch (...) {
    try {
      writer.close();
    } catch (...) {
    }
    throw;
  }
}

void PicTileRuntime::read_restart_lattices(
    ParallelCheckpointReader& reader, PicRestartPayload& payload) {
  const bool cylindrical = is_cylindrical(global_config_.geometry);
  const auto fields = field_extents(
      topology_.global_nx(), topology_.global_ny(), cylindrical);
  const auto sources = source_extents(
      topology_.global_nx(), topology_.global_ny(), cylindrical);
  const std::size_t first_endpoint =
      static_cast<std::size_t>(runtime_->rank()) *
      mapping_.devices_per_rank();

  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-restart-local-lattice-allocation",
      "PIC restart local lattice allocation failed", -1, [&] {
    payload.local_tiles.resize(solvers_.size());
    for (std::size_t local = 0; local < solvers_.size(); ++local) {
      auto& checkpoint = payload.local_tiles[local];
      checkpoint.tile = topology_.tile(first_endpoint + local);
      checkpoint.grid = PicTileAccess::grid(*solvers_[local]);
      const std::size_t size = checkpoint.grid.storage_size();
      for (auto& values : checkpoint.fields) values.assign(size, Real{0});
      for (auto& values : checkpoint.external_fields) {
        values.assign(size, Real{0});
      }
      for (auto& values : checkpoint.previous_b) {
        values.assign(size, Real{0});
      }
      for (auto& values : checkpoint.sources) values.assign(size, Real{0});
    }
  });

  const auto read_lattice =
      [this, &reader, &payload]
      (std::string_view name, const ComponentExtent& extent,
       auto&& select) {
        const auto shape = pic_checkpoint_shape(extent);
        std::vector<DatasetHyperslab> slabs;
        std::size_t elements = 0;
        std::vector<Real> packed;
        collective_try_with_fallback(
            *runtime_, worker_epoch_, "pic-restart-local-lattice-buffer",
            "PIC restart local lattice allocation failed", -1, [&] {
          slabs = pic_checkpoint_slabs(
              mapping_, topology_, runtime_->rank(), extent);
          for (const auto& slab : slabs) {
            const std::size_t count = pic_slab_elements(slab);
            if (count > std::numeric_limits<std::size_t>::max() - elements) {
              throw std::length_error{
                  "PIC restart packed selection overflows"};
            }
            elements += count;
          }
          packed.resize(elements);
        });
        reader.read_dataset(name, pic_checkpoint_real_type(), shape, slabs,
                            packed.data(), packed.size());
        collective_try_with_fallback(
            *runtime_, worker_epoch_, "pic-restart-local-lattice-unpack",
            "PIC restart local lattice unpack failed", -1, [&] {
          unpack_local_pic_checkpoint_values(
              payload.local_tiles, extent, slabs, packed, select);
        });
        ++telemetry_.checkpoint_local_lattice_reads;
      };

  for (std::size_t component = 0; component < fields.size(); ++component) {
    read_lattice(
        pic_field_dataset_names[component], fields[component],
        [component](PicCheckpointTile& local) -> std::vector<Real>& {
          return local.fields[component];
        });
    read_lattice(
        pic_external_field_dataset_names[component], fields[component],
        [component](PicCheckpointTile& local) -> std::vector<Real>& {
          return local.external_fields[component];
        });
  }
  for (std::size_t component = 0; component < 3; ++component) {
    read_lattice(
        pic_previous_b_dataset_names[component], fields[component + 3],
        [component](PicCheckpointTile& local) -> std::vector<Real>& {
          return local.previous_b[component];
        });
    read_lattice(
        pic_source_dataset_names[component], sources[component],
        [component](PicCheckpointTile& local) -> std::vector<Real>& {
          return local.sources[component];
        });
  }
  read_lattice(
      pic_source_dataset_names[3],
      ComponentExtent{topology_.global_nx(), topology_.global_ny(), false,
                      false},
      [](PicCheckpointTile& local) -> std::vector<Real>& {
        return local.sources[3];
      });
}

void PicTileRuntime::read_restart_runtime(
    ParallelCheckpointReader& reader, PicRestartPayload& payload) {
  auto& state = payload.state;
  const auto runtime_reals = read_pic_checkpoint_1d<Real>(
      reader, *runtime_, "pic/runtime/reals", pic_checkpoint_real_type(), 2,
      MPI_DOUBLE, false, worker_epoch_);
  const auto runtime_integers = read_pic_checkpoint_1d<std::uint64_t>(
      reader, *runtime_, "pic/runtime/integers",
      CheckpointValueType::uint64, 1, MPI_UINT64_T, false, worker_epoch_);
  const auto runtime_flags = read_pic_checkpoint_1d<std::uint8_t>(
      reader, *runtime_, "pic/runtime/flags", CheckpointValueType::uint8, 7,
      MPI_UINT8_T, false, worker_epoch_);
  std::copy(runtime_flags.begin(), runtime_flags.end(),
            payload.runtime_flags.begin());

  state.previous_dt = runtime_reals[0];
  state.background_charge_density = runtime_reals[1];
  state.step_count = runtime_integers[0];
  state.has_previous_dt = payload.runtime_flags[0] != 0;
  state.background_initialized = payload.runtime_flags[1] != 0;
  for (int side = 0; side < 4; ++side) {
    state.boundary.mur_primed[side] = payload.runtime_flags[2 + side];
    if (global_config_.boundary.field[side] != "outflow") continue;
    const std::size_t elements = 4 * mur_global_stride(topology_, side);
    state.boundary.mur_history[side] = read_pic_checkpoint_1d<Real>(
        reader, *runtime_,
        pic_mur_dataset_names[static_cast<std::size_t>(side)],
        pic_checkpoint_real_type(), elements, MPI_DOUBLE, false,
        worker_epoch_);
  }
  state.boundary.outflow_corners_primed =
      payload.runtime_flags[6] != 0;
  if (global_outflow_corner_mask(global_config_) != 0u) {
    state.boundary.outflow_corner_history =
        read_pic_checkpoint_1d<Real>(
            reader, *runtime_, "pic/boundary/outflow_corners",
            pic_checkpoint_real_type(), 8, MPI_DOUBLE, false,
            worker_epoch_);
  }
}

void PicTileRuntime::read_restart_particles(
    ParallelCheckpointReader& reader,
    std::span<const pic::SpeciesConfig> expected_species,
    PicRestartPayload& payload) {
  auto& state = payload.state;
  std::vector<std::uint64_t> particle_counts;
  if (!expected_species.empty()) {
    particle_counts = read_pic_checkpoint_1d<std::uint64_t>(
        reader, *runtime_, "pic/particles/counts",
        CheckpointValueType::uint64, expected_species.size(), MPI_UINT64_T,
        false, worker_epoch_);
  }

  collective_try(
      *runtime_, worker_epoch_, "pic-restart-particle-counts",
      "PIC checkpoint particle count validation failed", -1, [&] {
    for (std::size_t kind = 0; kind < expected_species.size(); ++kind) {
      if (particle_counts[kind] >
          static_cast<std::uint64_t>(expected_species[kind].capacity)) {
        throw std::length_error{
            "PIC checkpoint particle count exceeds the configured species "
            "capacity"};
      }
      if (particle_counts[kind] > static_cast<std::uint64_t>(
                                      std::numeric_limits<unsigned int>::max())) {
        throw std::length_error{
            "PIC checkpoint particle count exceeds the device-counter range"};
      }
    }
  });

  std::vector<std::array<std::string, 11>> particle_dataset_names;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-restart-particle-storage",
      "PIC restart particle metadata allocation failed", -1, [&] {
    state.species.resize(expected_species.size());
    particle_dataset_names.resize(expected_species.size());
    for (std::size_t kind = 0; kind < expected_species.size(); ++kind) {
      state.species[kind].config = expected_species[kind];
      state.species[kind].config.capacity = 0;
      const std::string prefix =
          "pic/particles/" + std::to_string(kind) + '/';
      for (std::size_t field = 0;
           field < pic_particle_dataset_suffixes.size(); ++field) {
        particle_dataset_names[kind][field] =
            prefix + std::string{pic_particle_dataset_suffixes[field]};
      }
    }
  });

  for (std::size_t kind = 0; kind < expected_species.size(); ++kind) {
    const std::size_t count =
        static_cast<std::size_t>(particle_counts[kind]);
    PicSpeciesState chunk;
    chunk.config = std::move(state.species[kind].config);
    if (count == 0) {
      state.species[kind] = std::move(chunk);
      continue;
    }
    std::size_t field = 0;
    const auto read_real = [&]() {
      return read_pic_checkpoint_chunk<Real>(
          reader, *runtime_, particle_dataset_names[kind][field++],
          pic_checkpoint_real_type(), count, worker_epoch_);
    };
    auto& particles = chunk.particles;
    particles.x = read_real();
    particles.y = read_real();
    particles.x_prev = read_real();
    particles.y_prev = read_real();
    particles.vx = read_real();
    particles.vy = read_real();
    particles.vz = read_real();
    particles.vphi_deposit = read_real();
    particles.weight = read_real();
    particles.alive = read_pic_checkpoint_chunk<std::uint8_t>(
        reader, *runtime_, particle_dataset_names[kind][9],
        CheckpointValueType::uint8, count, worker_epoch_);
    particles.id = read_pic_checkpoint_chunk<std::uint64_t>(
        reader, *runtime_, particle_dataset_names[kind][10],
        CheckpointValueType::uint64, count, worker_epoch_);
    state.species[kind] = route_checkpoint_species(
        std::move(chunk), particle_counts[kind]);
  }
  validate_distributed_particle_ids(
      state.species, "pic-restart-global-particle-ids");
}

void PicTileRuntime::validate_restart_payload(
    const CheckpointMetadata& stored, PicRestartPayload& payload) {
  const bool cylindrical = is_cylindrical(global_config_.geometry);
  const auto fields = field_extents(
      topology_.global_nx(), topology_.global_ny(), cylindrical);
  const auto sources = source_extents(
      topology_.global_nx(), topology_.global_ny(), cylindrical);

  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-restart-data-validation",
      "PIC checkpoint data validation failed", -1, [&] {
    if (payload.state.step_count != stored.step) {
      throw std::runtime_error{
          "PIC checkpoint metadata and runtime step disagree"};
    }
    for (const std::uint8_t flag : payload.runtime_flags) {
      if (flag > 1) {
        throw std::runtime_error{
            "PIC checkpoint contains an invalid boolean flag"};
      }
    }
    for (std::size_t component = 0; component < fields.size(); ++component) {
      for (const auto& local : payload.local_tiles) {
        require_finite(std::span<const Real>{local.fields[component]},
                       "PIC checkpoint live field tile");
        require_finite(
            std::span<const Real>{local.external_fields[component]},
            "PIC checkpoint external field tile");
      }
    }
    for (std::size_t component = 0; component < 3; ++component) {
      for (const auto& local : payload.local_tiles) {
        require_finite(std::span<const Real>{local.previous_b[component]},
                       "PIC checkpoint previous-B tile");
        require_finite(std::span<const Real>{local.sources[component]},
                       "PIC checkpoint source tile");
      }
    }
    for (const auto& local : payload.local_tiles) {
      require_finite(std::span<const Real>{local.sources[3]},
                     "PIC checkpoint charge tile");
    }
    for (const auto& species : payload.state.species) {
      validate_pic_species_state(species);
    }
  });

  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-restart-periodic-validation",
      "PIC checkpoint periodic validation failed", -1, [&] {
    for (std::size_t component = 0; component < fields.size(); ++component) {
      verify_and_rebuild_local_periodic_duplicates(
          *runtime_, payload.local_tiles, fields[component], global_config_,
          "PIC checkpoint live field",
          [component](PicCheckpointTile& local) -> std::vector<Real>& {
            return local.fields[component];
          });
      verify_and_rebuild_local_periodic_duplicates(
          *runtime_, payload.local_tiles, fields[component], global_config_,
          "PIC checkpoint external field",
          [component](PicCheckpointTile& local) -> std::vector<Real>& {
            return local.external_fields[component];
          });
    }
    for (std::size_t component = 0; component < 3; ++component) {
      verify_and_rebuild_local_periodic_duplicates(
          *runtime_, payload.local_tiles, fields[component + 3],
          global_config_, "PIC checkpoint previous B",
          [component](PicCheckpointTile& local) -> std::vector<Real>& {
            return local.previous_b[component];
          });
      verify_and_rebuild_local_periodic_duplicates(
          *runtime_, payload.local_tiles, sources[component], global_config_,
          "PIC checkpoint source",
          [component](PicCheckpointTile& local) -> std::vector<Real>& {
            return local.sources[component];
          });
    }
  });
}

void PicTileRuntime::commit_restart_payload(
    PicRestartPayload& payload, bool& mutation_started) {
  auto& state = payload.state;
  LocalSources restored_sources;
  collective_try_with_fallback(
      *runtime_, worker_epoch_, "pic-restart-source-setup",
      "PIC restart source allocation failed", -1, [&] {
    restored_sources.endpoint.resize(solvers_.size());
    for (std::size_t local = 0; local < solvers_.size(); ++local) {
      restored_sources.endpoint[local] =
          payload.local_tiles[local].sources;
    }
  });

  auto publish_lattices = pic_worker_tasks(
      *runtime_, worker_epoch_, solvers_.size(),
      [this, &payload](std::size_t local, WorkerContext&) {
        auto& solver = *solvers_[local];
        const auto& checkpoint = payload.local_tiles[local];
        const std::size_t size = checkpoint.grid.storage_size();
        auto& fields = PicTileAccess::fields(solver);
        auto& external_fields = PicTileAccess::external_fields(solver);
        for (std::size_t component = 0; component < 6; ++component) {
          field_buffer(fields, static_cast<FieldComponent>(component))
              .copy_from_host(checkpoint.fields[component].data(), size);
          field_buffer(external_fields,
                       static_cast<FieldComponent>(component))
              .copy_from_host(
                  checkpoint.external_fields[component].data(), size);
        }
        auto& previous_b = PicTileAccess::previous_b(solver);
        for (std::size_t component = 0; component < 3; ++component) {
          magnetic_buffer(
              previous_b, static_cast<FieldComponent>(component + 3))
              .copy_from_host(
                  checkpoint.previous_b[component].data(), size);
        }
        backend::device_synchronize(nullptr);
      });
  mutation_started = true;
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-restart-local-lattice-publish",
                         publish_lattices);

  seed_local_species(state.species);
  exchange_field_halos("pic-restart-live-field-halos");
  exchange_field_halos("pic-restart-external-field-halos", true, false);
  exchange_field_halos("pic-restart-previous-b-halos", false, true);
  const std::array<std::size_t, 4> all_source_components{0, 1, 2, 3};
  exchange_source_halos(restored_sources, all_source_components,
                        "pic-restart-source-halos");
  apply_sources(restored_sources, false, "pic-restart-source-publish");

  auto publish_metadata = pic_worker_tasks(
      *runtime_, worker_epoch_, solvers_.size(),
      [this, &state](std::size_t local, WorkerContext&) {
        auto& solver = *solvers_[local];
        PicTileAccess::fill_field_ghosts(solver);
        PicTileAccess::restore_runtime_state(
            solver, static_cast<std::size_t>(state.step_count),
            state.previous_dt, state.has_previous_dt,
            state.background_initialized,
            state.background_charge_density);
        backend::device_synchronize(nullptr);
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-restart-metadata-publish",
                         publish_metadata);
  apply_boundary_state(state.boundary);
  step_count_ = state.step_count;
  previous_dt_ = state.previous_dt;
  has_previous_dt_ = state.has_previous_dt;
  lifecycle_.seeded = true;
}

CheckpointMetadata PicTileRuntime::restart_from_checkpoint(
    const std::filesystem::path& path, std::string_view unit_system,
    std::span<const pic::SpeciesConfig> expected_species,
    std::vector<std::vector<std::uint8_t>>* diagnostic_state) {
  runtime_->require_orchestration_thread();
  const bool locally_ready =
      !lifecycle_.closed && !lifecycle_.seeded && !lifecycle_.poisoned;
  collective_require(
      *runtime_, worker_epoch_, locally_ready, "pic-restart-ready",
      lifecycle_.closed
          ? "distributed PIC runtime is closed"
          : lifecycle_.poisoned
                ? "distributed PIC runtime is poisoned"
                : "distributed PIC runtime is already seeded");

  ParallelCheckpointReader reader{*runtime_, path};
  std::optional<CheckpointMetadata> stored_storage;
  try {
    collective_try_with_fallback(
        *runtime_, worker_epoch_, "pic-restart-metadata-copy",
        "PIC checkpoint metadata copy failed", -1, [&] {
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
          if (std::exchange(inject_pic_checkpoint_metadata_copy_failure,
                            false)) {
            throw std::bad_alloc{};
          }
#endif
          stored_storage.emplace(reader.metadata());
        });
  } catch (...) {
    try {
      reader.close();
    } catch (...) {
    }
    throw;
  }

  CheckpointMetadata& stored = *stored_storage;
  PicRestartPayload payload;
  bool mutation_started = false;
  try {
    collective_try_with_fallback(
        *runtime_, worker_epoch_, "pic-restart-compatibility",
        "PIC checkpoint is incompatible with this runtime", -1, [&] {
      const CheckpointMetadata requested = checkpoint_metadata(
          stored.step, stored.time, unit_system, expected_species);
      validate_restart_compatibility(stored, requested);
    });

    read_restart_lattices(reader, payload);
    read_restart_runtime(reader, payload);
    read_restart_particles(reader, expected_species, payload);
    payload.diagnostic_state = reader.read_diagnostic_state();
    validate_restart_payload(stored, payload);
    reader.close();

    commit_restart_payload(payload, mutation_started);
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
    const std::uint64_t post_reconcile_epoch = worker_epoch_++;
    collective_require_at_epoch(
        *runtime_, post_reconcile_epoch,
        !inject_restart_post_reconcile_failure_,
        "pic-restart-post-reconcile",
        "injected post-reconcile PIC restart failure");
#endif
    if (diagnostic_state != nullptr) {
      *diagnostic_state = std::move(payload.diagnostic_state);
    }
  } catch (const std::exception& error) {
    try {
      reader.close();
    } catch (...) {
    }
    if (mutation_started) {
      poison_collectively("pic-restart-failure", error.what());
    }
    throw;
  } catch (...) {
    try {
      reader.close();
    } catch (...) {
    }
    if (mutation_started) {
      poison_collectively("pic-restart-failure",
                          "non-standard PIC restart failure");
    }
    throw;
  }
  return std::move(*stored_storage);
}

void PicTileRuntime::close() {
  if (lifecycle_.closed) return;
  runtime_->require_orchestration_thread();
  if (transport_) {
    transport_->close();
    update_transport_telemetry();
  }
  auto tasks = pic_worker_tasks(*runtime_, worker_epoch_, solvers_.size(),
      [this](std::size_t local, WorkerContext&) {
        halo_buffers_[local].reset();
        solvers_[local].reset();
      });
  require_worker_success(*workers_, *runtime_, worker_epoch_,
                         "pic-runtime-close", tasks);
  workers_->close();
  lifecycle_.closed = true;
}

}  // namespace quasar::distributed
