// Device-side routing, ordering and uniqueness checking for distributed PIC
// particle migration.
//
// A particle that leaves its tile must be told which tile owns it next. That
// decision is floating-point work -- a periodic coordinate is wrapped back into
// the global domain, then reduced to a global cell index and mapped through the
// balanced tile partition -- and it used to run on the host, once per departing
// particle, over a snapshot downloaded for the purpose. It runs here now. What
// crosses to the host is a routed, ordered array of wire records.
//
// Ordering. The records are grouped by destination rank and, inside a group,
// sorted by ascending stable id. The id order is the point: without it the
// arrival order of a species would depend on the order the partition kernel
// happened to emit departures and on the order MPI delivered peers, so two
// runs of the same input could deposit in different orders. The sort primitive
// below is a stable counting sort, applied eight times over the bytes of the id
// and once more over the destination rank. Because every pass is stable, the
// final rank pass leaves ties in id order.
//
// That sort is deliberately the simple one: one thread per key value, each
// thread walking the whole array in index order and emitting its own bucket.
// It is O(n) wall clock with a small constant per pass and stable by
// construction, at the cost of reading the key array once per pass per thread.
// Migration batches are the particles that crossed a tile boundary in one step
// -- hundreds to a few thousand -- not the resident population, so the trade
// is the right one here and would not be for a full-population sort.

#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/physics/pic/kernels.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace quasar::pic {
namespace {

constexpr int kSortBlock = 256;
constexpr int kRadixDigits = 256;

// Same balanced split as quasar::distributed::VirtualTopology: the first
// `global_size % partitions` tiles get one extra cell. Pure integer arithmetic,
// so it is transcribed rather than approximated.
__device__ inline std::uint64_t owner_coordinate(std::uint64_t index,
                                                 std::uint64_t global_size,
                                                 std::uint64_t partitions) {
  const std::uint64_t base = global_size / partitions;
  const std::uint64_t remainder = global_size % partitions;
  const std::uint64_t large_region = (base + 1) * remainder;
  if (index < large_region) return index / (base + 1);
  return remainder + (index - large_region) / base;
}

// Host wrap_periodic, transcribed. fmod then a single conditional add, and the
// closing guard that a value rounding up to exactly `high` is folded to `low`
// so the domain stays half-open.
__device__ inline double wrap_periodic(double value, double low, double length) {
  double reduced = ::fmod(value - low, length);
  if (reduced < 0.0) reduced += length;
  double wrapped = low + reduced;
  const double high = low + length;
  if (!(wrapped < high)) wrapped = low;
  return wrapped;
}

// Host quasar::detail::scaled_difference_quotient, in the one form this needs.
__device__ inline double cell_coordinate(double value, double origin,
                                         double spacing) {
  return (value - origin) / spacing;
}

// Position to owning endpoint. Mirrors PicTileRuntime::owner_for_position,
// including its treatment of the closed upper edge: a particle sitting exactly
// on the high face belongs to the last tile rather than to a cell one past the
// end.
__device__ inline bool owner_for_position(const PicMigrationTopology& t,
                                          double x, double y,
                                          std::uint64_t* owner) {
  const double x_hi = t.grid.origin_x + t.grid.lx;
  const double y_hi = t.grid.origin_y + t.grid.ly;
  std::uint64_t ix = 0;
  std::uint64_t iy = 0;

  if (x == x_hi) {
    ix = t.global_nx - 1;
  } else {
    const double c = cell_coordinate(x, t.grid.origin_x, t.grid.dx());
    if (!(::isfinite(c) && c >= 0.0)) return false;
    const double floored = ::floor(c);
    if (floored >= static_cast<double>(t.global_nx)) return false;
    ix = static_cast<std::uint64_t>(floored);
  }

  if (y == y_hi) {
    iy = t.global_ny - 1;
  } else {
    const double c = cell_coordinate(y, t.grid.origin_y, t.grid.dy());
    if (!(::isfinite(c) && c >= 0.0)) return false;
    const double floored = ::floor(c);
    if (floored >= static_cast<double>(t.global_ny)) return false;
    iy = static_cast<std::uint64_t>(floored);
  }

  if (ix >= t.global_nx || iy >= t.global_ny) return false;
  const std::uint64_t tx = owner_coordinate(ix, t.global_nx, t.px);
  const std::uint64_t ty = owner_coordinate(iy, t.global_ny, t.py);
  if (tx >= t.px || ty >= t.py) return false;
  *owner = ty * t.px + tx;
  return true;
}

__global__ void route_kernel(std::size_t n, PicMigrationTopology topology,
                             std::uint64_t source_endpoint,
                             std::uint64_t species_index,
                             const double* __restrict__ x,
                             const double* __restrict__ y,
                             const double* __restrict__ x_prev,
                             const double* __restrict__ y_prev,
                             const double* __restrict__ vx,
                             const double* __restrict__ vy,
                             const double* __restrict__ vz,
                             const double* __restrict__ vphi_deposit,
                             const double* __restrict__ weight,
                             const std::uint8_t* __restrict__ alive,
                             const std::uint64_t* __restrict__ id,
                             const std::uint64_t* __restrict__ endpoint_rank,
                             PicParticleMigrationRecord* __restrict__ records,
                             std::uint64_t* __restrict__ destination_rank,
                             unsigned long long* __restrict__ migrated,
                             int* __restrict__ status) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n) return;

  PicParticleRecord record;
  record.x = x[p];
  record.y = y[p];
  record.x_prev = x_prev[p];
  record.y_prev = y_prev[p];
  record.vx = vx[p];
  record.vy = vy[p];
  record.vz = vz[p];
  record.vphi_deposit = vphi_deposit[p];
  record.weight = weight[p];
  record.id = id[p];
  record.source_endpoint = source_endpoint;
  record.alive = alive[p];

  std::uint64_t owner = source_endpoint;
  if (record.alive != 0) {
    // A dead particle keeps its source endpoint and is never wrapped: it
    // carries no position worth resolving and the receiving tile drops it.
    if (topology.periodic_x != 0) {
      const double wrapped = wrap_periodic(record.x, topology.grid.origin_x,
                                           topology.grid.lx);
      const double shift = wrapped - record.x;
      record.x = wrapped;
      record.x_prev += shift;
    }
    if (topology.periodic_y != 0) {
      const double wrapped = wrap_periodic(record.y, topology.grid.origin_y,
                                           topology.grid.ly);
      const double shift = wrapped - record.y;
      record.y = wrapped;
      record.y_prev += shift;
    }
    if (!owner_for_position(topology, record.x, record.y, &owner)) {
      atomicOr(status, kPicMigrationCoordinateOutsideMesh);
      owner = source_endpoint;
    }
  }

  if (owner >= topology.endpoint_count) {
    atomicOr(status, kPicMigrationOwnerOutOfRange);
    owner = source_endpoint;
  }
  if (owner != source_endpoint) {
    atomicAdd(migrated, 1ULL);
  }

  records[p].particle = record;
  records[p].destination_endpoint = owner;
  records[p].species = species_index;
  destination_rank[p] = endpoint_rank[owner];
}

// -- Stable counting sort ----------------------------------------------------
//
// One thread per bucket. Each thread makes two passes over the key array in
// index order: the first counts its bucket's members, the second -- after an
// exclusive scan of the counts -- writes their indices out consecutively.
// Walking in index order is what makes it stable.

__global__ void count_keys_kernel(std::size_t n,
                                  const std::uint64_t* __restrict__ keys,
                                  std::uint64_t bucket_count,
                                  std::uint64_t* __restrict__ counts) {
  const std::uint64_t bucket = blockIdx.x * blockDim.x + threadIdx.x;
  if (bucket >= bucket_count) return;
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (keys[i] == bucket) ++total;
  }
  counts[bucket] = total;
}

// Exclusive scan of the bucket counts. Buckets number 256 for a radix digit and
// the rank count for the grouping pass -- small enough that one serial thread
// is cheaper than a parallel scan's launch.
__global__ void scan_counts_kernel(std::uint64_t bucket_count,
                                   std::uint64_t* __restrict__ counts,
                                   std::uint64_t* __restrict__ offsets) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  std::uint64_t running = 0;
  for (std::uint64_t b = 0; b < bucket_count; ++b) {
    offsets[b] = running;
    running += counts[b];
  }
  offsets[bucket_count] = running;
}

__global__ void scatter_keys_kernel(std::size_t n,
                                    const std::uint64_t* __restrict__ keys,
                                    const std::uint32_t* __restrict__ order_in,
                                    std::uint64_t bucket_count,
                                    const std::uint64_t* __restrict__ offsets,
                                    std::uint32_t* __restrict__ order_out) {
  const std::uint64_t bucket = blockIdx.x * blockDim.x + threadIdx.x;
  if (bucket >= bucket_count) return;
  std::uint64_t cursor = offsets[bucket];
  for (std::size_t i = 0; i < n; ++i) {
    if (keys[i] == bucket) {
      order_out[cursor] = order_in[i];
      ++cursor;
    }
  }
}

// Extract one radix digit of the sort key of the element currently at position
// `i` of the permutation, so the counting sort can be driven by the permutation
// without ever moving the records themselves.
__global__ void radix_digit_kernel(std::size_t n,
                                   const std::uint64_t* __restrict__ ids,
                                   const std::uint32_t* __restrict__ order,
                                   int shift,
                                   std::uint64_t* __restrict__ digits) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  digits[i] = (ids[order[i]] >> shift) & 0xFFull;
}

__global__ void rank_key_kernel(std::size_t n,
                                const std::uint64_t* __restrict__ ranks,
                                const std::uint32_t* __restrict__ order,
                                std::uint64_t* __restrict__ keys) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  keys[i] = ranks[order[i]];
}

__global__ void iota_kernel(std::size_t n, std::uint32_t* __restrict__ order) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  order[i] = static_cast<std::uint32_t>(i);
}

__global__ void gather_records_kernel(
    std::size_t n, const PicParticleMigrationRecord* __restrict__ source,
    const std::uint32_t* __restrict__ order,
    PicParticleMigrationRecord* __restrict__ destination) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  destination[i] = source[order[i]];
}

__global__ void gather_ids_kernel(std::size_t n,
                                  const std::uint64_t* __restrict__ source,
                                  const std::uint32_t* __restrict__ order,
                                  std::uint64_t* __restrict__ destination) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  destination[i] = source[order[i]];
}

__global__ void adjacent_duplicate_kernel(
    std::size_t n, const std::uint64_t* __restrict__ sorted,
    int* __restrict__ status) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0 || i >= n) return;
  if (sorted[i] == sorted[i - 1]) atomicOr(status, kPicMigrationDuplicateId);
}

// Expand sorted wire records into the species' SoA planes at `base`.
__global__ void expand_records_kernel(
    std::size_t n, std::size_t base,
    const PicParticleMigrationRecord* __restrict__ records,
    double* __restrict__ x, double* __restrict__ y,
    double* __restrict__ x_prev, double* __restrict__ y_prev,
    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
    double* __restrict__ vphi_deposit, double* __restrict__ weight,
    std::uint8_t* __restrict__ alive, std::uint64_t* __restrict__ id) {
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const PicParticleRecord& r = records[i].particle;
  const std::size_t slot = base + i;
  x[slot] = r.x;
  y[slot] = r.y;
  x_prev[slot] = r.x_prev;
  y_prev[slot] = r.y_prev;
  vx[slot] = r.vx;
  vy[slot] = r.vy;
  vz[slot] = r.vz;
  vphi_deposit[slot] = r.vphi_deposit;
  weight[slot] = r.weight;
  alive[slot] = r.alive;
  id[slot] = r.id;
}

dim3 blocks_for(std::size_t n, int block) {
  return dim3(static_cast<unsigned int>((n + block - 1) / block));
}

// Apply one stable counting-sort pass keyed on `keys` (values in
// [0, bucket_count)), permuting `order` in place. Scratch is supplied by the
// caller so a multi-pass sort allocates once.
struct SortScratch {
  backend::DeviceBuffer<std::uint64_t> keys;
  backend::DeviceBuffer<std::uint64_t> counts;
  backend::DeviceBuffer<std::uint64_t> offsets;
  backend::DeviceBuffer<std::uint32_t> order_alt;
};

void counting_sort_pass(std::size_t n, std::uint64_t bucket_count,
                        SortScratch& scratch,
                        backend::DeviceBuffer<std::uint32_t>& order,
                        hipStream_t stream) {
  const dim3 block(kSortBlock);
  const dim3 bucket_blocks(
      static_cast<unsigned int>((bucket_count + kSortBlock - 1) / kSortBlock));
  hipLaunchKernelGGL(count_keys_kernel, bucket_blocks, block, 0, stream, n,
                     scratch.keys.device_ptr(), bucket_count,
                     scratch.counts.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  hipLaunchKernelGGL(scan_counts_kernel, dim3(1), dim3(1), 0, stream,
                     bucket_count, scratch.counts.device_ptr(),
                     scratch.offsets.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  hipLaunchKernelGGL(scatter_keys_kernel, bucket_blocks, block, 0, stream, n,
                     scratch.keys.device_ptr(), order.device_ptr(),
                     bucket_count, scratch.offsets.device_ptr(),
                     scratch.order_alt.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  std::swap(order, scratch.order_alt);
}

// Sort `order` (a permutation of [0, n)) so that ids[order[i]] ascends.
void sort_order_by_id(std::size_t n, const std::uint64_t* ids,
                      SortScratch& scratch,
                      backend::DeviceBuffer<std::uint32_t>& order,
                      hipStream_t stream) {
  const dim3 block(kSortBlock);
  for (int shift = 0; shift < 64; shift += 8) {
    hipLaunchKernelGGL(radix_digit_kernel, blocks_for(n, kSortBlock), block, 0,
                       stream, n, ids, order.device_ptr(), shift,
                       scratch.keys.device_ptr());
    QUASAR_HIP_CHECK(::hipGetLastError());
    counting_sort_pass(n, kRadixDigits, scratch, order, stream);
  }
}

SortScratch make_scratch(std::size_t n, std::uint64_t max_buckets) {
  SortScratch scratch;
  scratch.keys = backend::DeviceBuffer<std::uint64_t>(n, backend::uninitialized);
  scratch.counts = backend::DeviceBuffer<std::uint64_t>(max_buckets);
  scratch.offsets = backend::DeviceBuffer<std::uint64_t>(max_buckets + 1);
  scratch.order_alt =
      backend::DeviceBuffer<std::uint32_t>(n, backend::uninitialized);
  return scratch;
}

}  // namespace

PicMigrationRouting launch_pic_route_departing_particles(
    const PicDepartingParticles& departing, PicMigrationTopology topology,
    std::uint64_t source_endpoint, std::uint64_t species_index,
    const backend::DeviceBuffer<std::uint64_t>& endpoint_rank,
    std::size_t rank_count, int* status, backend::stream_t stream_in) {
  const auto stream = static_cast<hipStream_t>(stream_in);
  PicMigrationRouting routing;
  routing.rank_offsets.assign(rank_count + 1, 0);
  const std::size_t n = departing.count;
  if (n == 0) return routing;
  if (rank_count == 0) {
    throw std::invalid_argument{
        "PIC migration routing requires at least one destination rank"};
  }
  if (n > static_cast<std::size_t>(0xFFFFFFFFu)) {
    // The permutation is 32-bit. A migration batch this large would mean the
    // whole population crossed a tile boundary in one step.
    throw std::runtime_error{
        "PIC migration batch exceeds the routable particle count"};
  }

  backend::DeviceBuffer<PicParticleMigrationRecord> records(
      n, backend::uninitialized);
  backend::DeviceBuffer<std::uint64_t> destination_rank(n,
                                                        backend::uninitialized);
  backend::DeviceBuffer<unsigned long long> migrated(1);

  const dim3 block(kSortBlock);
  hipLaunchKernelGGL(
      route_kernel, blocks_for(n, kSortBlock), block, 0, stream, n, topology,
      source_endpoint, species_index, departing.x.device_ptr(),
      departing.y.device_ptr(), departing.x_prev.device_ptr(),
      departing.y_prev.device_ptr(), departing.vx.device_ptr(),
      departing.vy.device_ptr(), departing.vz.device_ptr(),
      departing.vphi_deposit.device_ptr(), departing.weight.device_ptr(),
      departing.alive.device_ptr(), departing.id.device_ptr(),
      endpoint_rank.device_ptr(), records.device_ptr(),
      destination_rank.device_ptr(), migrated.device_ptr(), status);
  QUASAR_HIP_CHECK(::hipGetLastError());

  const std::uint64_t max_buckets =
      rank_count > static_cast<std::size_t>(kRadixDigits)
          ? static_cast<std::uint64_t>(rank_count)
          : static_cast<std::uint64_t>(kRadixDigits);
  SortScratch scratch = make_scratch(n, max_buckets);
  backend::DeviceBuffer<std::uint32_t> order(n, backend::uninitialized);
  hipLaunchKernelGGL(iota_kernel, blocks_for(n, kSortBlock), block, 0, stream,
                     n, order.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());

  // Order by id first, then group by destination rank. The grouping pass is
  // stable, so records within a rank stay in id order.
  sort_order_by_id(n, departing.id.device_ptr(), scratch, order, stream);
  hipLaunchKernelGGL(rank_key_kernel, blocks_for(n, kSortBlock), block, 0,
                     stream, n, destination_rank.device_ptr(),
                     order.device_ptr(), scratch.keys.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  counting_sort_pass(n, static_cast<std::uint64_t>(rank_count), scratch, order,
                     stream);

  backend::DeviceBuffer<PicParticleMigrationRecord> grouped(
      n, backend::uninitialized);
  hipLaunchKernelGGL(gather_records_kernel, blocks_for(n, kSortBlock), block, 0,
                     stream, n, records.device_ptr(), order.device_ptr(),
                     grouped.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());

  routing.records.resize(n);
  grouped.copy_to_host(routing.records.data(), n);
  // The offsets the grouping pass just scanned are exactly the per-rank
  // boundaries; read them rather than recounting on the host.
  scratch.offsets.copy_to_host(routing.rank_offsets.data(), rank_count + 1);
  unsigned long long migrated_host = 0;
  migrated.copy_to_host(&migrated_host, 1);
  QUASAR_HIP_CHECK(::hipStreamSynchronize(stream));
  routing.migrated = static_cast<std::uint64_t>(migrated_host);
  return routing;
}

void launch_pic_append_migrated_records(
    ParticleSpecies& species,
    const std::vector<PicParticleMigrationRecord>& records,
    backend::stream_t stream_in) {
  const auto stream = static_cast<hipStream_t>(stream_in);
  const std::size_t n = records.size();
  if (n == 0) return;
  if (n > static_cast<std::size_t>(0xFFFFFFFFu)) {
    throw std::runtime_error{
        "PIC migration arrival batch exceeds the sortable particle count"};
  }
  const std::size_t base = species.size();
  if (base + n > species.capacity()) species.reserve(base + n);

  backend::DeviceBuffer<PicParticleMigrationRecord> device_records(
      n, backend::uninitialized);
  device_records.copy_from_host(records.data(), n);

  backend::DeviceBuffer<std::uint64_t> ids(n, backend::uninitialized);
  {
    std::vector<std::uint64_t> host_ids(n);
    for (std::size_t i = 0; i < n; ++i) host_ids[i] = records[i].particle.id;
    ids.copy_from_host(host_ids.data(), n);
  }

  const dim3 block(kSortBlock);
  SortScratch scratch = make_scratch(n, kRadixDigits);
  backend::DeviceBuffer<std::uint32_t> order(n, backend::uninitialized);
  hipLaunchKernelGGL(iota_kernel, blocks_for(n, kSortBlock), block, 0, stream,
                     n, order.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  sort_order_by_id(n, ids.device_ptr(), scratch, order, stream);

  backend::DeviceBuffer<PicParticleMigrationRecord> sorted(
      n, backend::uninitialized);
  hipLaunchKernelGGL(gather_records_kernel, blocks_for(n, kSortBlock), block, 0,
                     stream, n, device_records.device_ptr(),
                     order.device_ptr(), sorted.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());

  hipLaunchKernelGGL(expand_records_kernel, blocks_for(n, kSortBlock), block, 0,
                     stream, n, base, sorted.device_ptr(), species.x(),
                     species.y(), species.x_prev(), species.y_prev(),
                     species.vx(), species.vy(), species.vz(),
                     species.vphi_deposit(), species.weight(), species.alive(),
                     species.id());
  QUASAR_HIP_CHECK(::hipGetLastError());
  QUASAR_HIP_CHECK(::hipStreamSynchronize(stream));
  species.set_count(base + n);
}

bool launch_pic_ids_have_duplicate(const std::vector<std::uint64_t>& ids,
                                   backend::stream_t stream_in) {
  const auto stream = static_cast<hipStream_t>(stream_in);
  const std::size_t n = ids.size();
  if (n < 2) return false;
  if (n > static_cast<std::size_t>(0xFFFFFFFFu)) {
    throw std::runtime_error{
        "PIC particle id uniqueness check exceeds the sortable count"};
  }

  backend::DeviceBuffer<std::uint64_t> device_ids(n, backend::uninitialized);
  device_ids.copy_from_host(ids.data(), n);

  const dim3 block(kSortBlock);
  SortScratch scratch = make_scratch(n, kRadixDigits);
  backend::DeviceBuffer<std::uint32_t> order(n, backend::uninitialized);
  hipLaunchKernelGGL(iota_kernel, blocks_for(n, kSortBlock), block, 0, stream,
                     n, order.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  sort_order_by_id(n, device_ids.device_ptr(), scratch, order, stream);

  backend::DeviceBuffer<std::uint64_t> sorted(n, backend::uninitialized);
  hipLaunchKernelGGL(gather_ids_kernel, blocks_for(n, kSortBlock), block, 0,
                     stream, n, device_ids.device_ptr(), order.device_ptr(),
                     sorted.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());

  backend::DeviceBuffer<int> status(1);
  hipLaunchKernelGGL(adjacent_duplicate_kernel, blocks_for(n, kSortBlock),
                     block, 0, stream, n, sorted.device_ptr(),
                     status.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  int host_status = 0;
  status.copy_to_host(&host_status, 1);
  QUASAR_HIP_CHECK(::hipStreamSynchronize(stream));
  return (host_status & kPicMigrationDuplicateId) != 0;
}

}  // namespace quasar::pic
