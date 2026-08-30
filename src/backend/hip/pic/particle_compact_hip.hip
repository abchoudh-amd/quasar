#include "quasar/backend/device.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

// Stream-compaction of the particle arrays: alive particles are scattered to the
// front of scratch buffers via an atomic running counter (order is not
// preserved, which is fine for PIC), then copied back and the active count is
// shrunk. This reclaims slots left dead by absorbing boundaries so the push /
// deposit / gather kernels stop iterating over tombstones.

namespace {

// One output slot per alive particle, claimed atomically. Dead particles are
// never written, so the front [0, *out_count) range is densely alive.
__global__ void compact_scatter_kernel(std::size_t n,
                                       const double* __restrict__ x,
                                       const double* __restrict__ y,
                                       const double* __restrict__ x_prev,
                                       const double* __restrict__ y_prev,
                                       const double* __restrict__ vx,
                                       const double* __restrict__ vy,
                                       const double* __restrict__ vz,
                                       const double* __restrict__ vphi_deposit,
                                       const double* __restrict__ weight,
                                       const unsigned char* __restrict__ alive,
                                       const std::uint64_t* __restrict__ id,
                                       double* __restrict__ ox, double* __restrict__ oy,
                                       double* __restrict__ ox_prev, double* __restrict__ oy_prev,
                                       double* __restrict__ ovx, double* __restrict__ ovy,
                                       double* __restrict__ ovz, double* __restrict__ oweight,
                                       double* __restrict__ ovphi_deposit,
                                       unsigned char* __restrict__ oalive,
                                       std::uint64_t* __restrict__ oid,
                                       unsigned int* __restrict__ out_count) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;
  const unsigned int dst = atomicAdd(out_count, 1u);
  ox[dst] = x[p];
  oy[dst] = y[p];
  ox_prev[dst] = x_prev[p];
  oy_prev[dst] = y_prev[p];
  ovx[dst] = vx[p];
  ovy[dst] = vy[p];
  ovz[dst] = vz[p];
  ovphi_deposit[dst] = vphi_deposit[p];
  oweight[dst] = weight[p];
  oalive[dst] = 1;
  oid[dst] = id[p];
}

// Reduction of alive flags into a single device counter: each block sums its
// alive flags in shared memory, then one thread folds the block total into the
// global counter. The caller zeroes the counter before launch.
__global__ void alive_count_kernel(std::size_t n, const unsigned char* __restrict__ alive,
                                   unsigned int* __restrict__ total) {
  __shared__ unsigned int s_count;
  if (threadIdx.x == 0) s_count = 0;
  __syncthreads();
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < n && alive[p] != 0) atomicAdd(&s_count, 1u);
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(total, s_count);
}

__global__ void departure_count_kernel(
    std::size_t n, quasar::Grid2D grid,
    const double* __restrict__ x, const double* __restrict__ y,
    const unsigned char* __restrict__ alive, bool include_x_high,
    bool include_y_high, unsigned int* __restrict__ total) {
  __shared__ unsigned int s_count;
  if (threadIdx.x == 0) s_count = 0;
  __syncthreads();
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < n && alive[p] != 0) {
    const double x_hi = grid.origin_x + grid.lx;
    const double y_hi = grid.origin_y + grid.ly;
    const bool outside_x = !isfinite(x[p]) || x[p] < grid.origin_x ||
        x[p] > x_hi || (!include_x_high && x[p] == x_hi);
    const bool outside_y = !isfinite(y[p]) || y[p] < grid.origin_y ||
        y[p] > y_hi || (!include_y_high && y[p] == y_hi);
    if (outside_x || outside_y) atomicAdd(&s_count, 1u);
  }
  __syncthreads();
  if (threadIdx.x == 0) atomicAdd(total, s_count);
}

__device__ bool departing_particle(quasar::Grid2D grid, double x, double y,
                                   bool include_x_high,
                                   bool include_y_high) {
  const double x_hi = grid.origin_x + grid.lx;
  const double y_hi = grid.origin_y + grid.ly;
  const bool outside_x = !isfinite(x) || x < grid.origin_x || x > x_hi ||
      (!include_x_high && x == x_hi);
  const bool outside_y = !isfinite(y) || y < grid.origin_y || y > y_hi ||
      (!include_y_high && y == y_hi);
  return outside_x || outside_y;
}

__global__ void partition_departing_kernel(
    std::size_t n, quasar::Grid2D grid,
    const double* __restrict__ x, const double* __restrict__ y,
    const double* __restrict__ x_prev, const double* __restrict__ y_prev,
    const double* __restrict__ vx, const double* __restrict__ vy,
    const double* __restrict__ vz,
    const double* __restrict__ vphi_deposit,
    const double* __restrict__ weight,
    const unsigned char* __restrict__ alive,
    const std::uint64_t* __restrict__ id,
    double* __restrict__ ox, double* __restrict__ oy,
    double* __restrict__ ox_prev, double* __restrict__ oy_prev,
    double* __restrict__ ovx, double* __restrict__ ovy,
    double* __restrict__ ovz, double* __restrict__ ovphi_deposit,
    double* __restrict__ oweight, unsigned char* __restrict__ oalive,
    std::uint64_t* __restrict__ oid, bool include_x_high,
    bool include_y_high, unsigned int* __restrict__ resident_count,
    unsigned int* __restrict__ departure_count) {
  const std::size_t particle =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (particle >= n) return;
  const bool departing = alive[particle] != 0 &&
      departing_particle(grid, x[particle], y[particle], include_x_high,
                         include_y_high);
  const unsigned int destination = departing
      ? static_cast<unsigned int>(n - 1) - atomicAdd(departure_count, 1u)
      : atomicAdd(resident_count, 1u);
  ox[destination] = x[particle];
  oy[destination] = y[particle];
  ox_prev[destination] = x_prev[particle];
  oy_prev[destination] = y_prev[particle];
  ovx[destination] = vx[particle];
  ovy[destination] = vy[particle];
  ovz[destination] = vz[particle];
  ovphi_deposit[destination] = vphi_deposit[particle];
  oweight[destination] = weight[particle];
  oalive[destination] = alive[particle];
  oid[destination] = id[particle];
}

}  // namespace

extern "C" void launch_pic_particle_compact(quasar::pic::ParticleSpecies& s,
                                            quasar_stream_t stream_in) {
  const auto stream = static_cast<hipStream_t>(stream_in);
  const std::size_t n = s.size();
  if (n == 0) return;

  // Scratch is owned by the species (capacity-sized, allocated once). Reset only
  // the running counter rather than reallocating the complete scratch set per
  // call.
  QUASAR_HIP_CHECK(::hipMemsetAsync(s.compact_counter(), 0, sizeof(unsigned int), stream));

  const dim3 block(256);
  const dim3 grid((n + block.x - 1) / block.x);
  hipLaunchKernelGGL(compact_scatter_kernel, grid, block, 0, stream, n,
                     s.x(), s.y(), s.x_prev(), s.y_prev(), s.vx(), s.vy(), s.vz(),
                     s.vphi_deposit(),
                     s.weight(), s.alive(), s.id(),
                     s.compact_x(), s.compact_y(), s.compact_x_prev(),
                     s.compact_y_prev(), s.compact_vx(), s.compact_vy(),
                     s.compact_vz(), s.compact_weight(), s.compact_vphi_deposit(),
                     s.compact_alive(), s.compact_id(),
                     s.compact_counter());
  QUASAR_HIP_CHECK(::hipGetLastError());

  unsigned int alive_n = 0;
  QUASAR_HIP_CHECK(::hipMemcpyAsync(&alive_n, s.compact_counter(), sizeof(unsigned int),
                                    ::hipMemcpyDeviceToHost, stream));
  QUASAR_HIP_CHECK(::hipStreamSynchronize(stream));

  if (alive_n > 0) {
    const std::size_t bytes_d = static_cast<std::size_t>(alive_n) * sizeof(double);
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.x(), s.compact_x(), bytes_d, ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.y(), s.compact_y(), bytes_d, ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.x_prev(), s.compact_x_prev(), bytes_d, ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.y_prev(), s.compact_y_prev(), bytes_d, ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.vx(), s.compact_vx(), bytes_d, ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.vy(), s.compact_vy(), bytes_d, ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.vz(), s.compact_vz(), bytes_d, ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.vphi_deposit(), s.compact_vphi_deposit(),
                                      bytes_d, ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.weight(), s.compact_weight(), bytes_d, ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.alive(), s.compact_alive(),
                                      static_cast<std::size_t>(alive_n) * sizeof(std::uint8_t),
                                      ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.id(), s.compact_id(),
                                      static_cast<std::size_t>(alive_n)
                                          * sizeof(std::uint64_t),
                                      ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipStreamSynchronize(stream));
  }
  s.set_count(alive_n);
}

extern "C" std::size_t launch_pic_alive_count(const quasar::pic::ParticleSpecies& s,
                                              quasar_stream_t stream_in) {
  const auto stream = static_cast<hipStream_t>(stream_in);
  const std::size_t n = s.size();
  if (n == 0) return 0;
  const dim3 block(256);
  const dim3 grid((n + block.x - 1) / block.x);
  // Reuse the species-owned compaction counter for the reduction instead of
  // allocating (and zero-filling) a per-block partials buffer every call.
  unsigned int* counter = s.compact_counter();
  QUASAR_HIP_CHECK(::hipMemsetAsync(counter, 0, sizeof(unsigned int), stream));
  hipLaunchKernelGGL(alive_count_kernel, grid, block, 0, stream, n, s.alive(), counter);
  QUASAR_HIP_CHECK(::hipGetLastError());
  unsigned int total = 0;
  QUASAR_HIP_CHECK(::hipMemcpyAsync(&total, counter, sizeof(unsigned int),
                                    ::hipMemcpyDeviceToHost, stream));
  QUASAR_HIP_CHECK(::hipStreamSynchronize(stream));
  return total;
}

extern "C" std::size_t launch_pic_particle_departure_count(
    const quasar::pic::ParticleSpecies& s, int include_x_high,
    int include_y_high, quasar_stream_t stream_in) {
  const auto stream = static_cast<hipStream_t>(stream_in);
  const std::size_t n = s.size();
  if (n == 0) return 0;
  const dim3 block(256);
  const dim3 grid((n + block.x - 1) / block.x);
  unsigned int* counter = s.compact_counter();
  QUASAR_HIP_CHECK(
      ::hipMemsetAsync(counter, 0, sizeof(unsigned int), stream));
  hipLaunchKernelGGL(departure_count_kernel, grid, block, 0, stream, n,
                     s.grid(), s.x(), s.y(), s.alive(),
                     include_x_high != 0, include_y_high != 0, counter);
  QUASAR_HIP_CHECK(::hipGetLastError());
  unsigned int total = 0;
  QUASAR_HIP_CHECK(::hipMemcpyAsync(&total, counter, sizeof(unsigned int),
                                    ::hipMemcpyDeviceToHost, stream));
  QUASAR_HIP_CHECK(::hipStreamSynchronize(stream));
  return total;
}

namespace quasar::pic {

ParticleSpecies::HostSnapshot extract_pic_departing_particles(
    ParticleSpecies& species, int include_x_high, int include_y_high,
    backend::stream_t stream_in) {
  const auto stream = static_cast<hipStream_t>(stream_in);
  const std::size_t count = species.size();
  ParticleSpecies::HostSnapshot departing;
  if (count == 0) return departing;

  QUASAR_HIP_CHECK(::hipMemsetAsync(species.compact_counter(), 0,
                                    2 * sizeof(unsigned int), stream));
  const dim3 block(256);
  const dim3 blocks((count + block.x - 1) / block.x);
  hipLaunchKernelGGL(
      partition_departing_kernel, blocks, block, 0, stream, count,
      species.grid(), species.x(), species.y(), species.x_prev(),
      species.y_prev(), species.vx(), species.vy(), species.vz(),
      species.vphi_deposit(), species.weight(), species.alive(), species.id(),
      species.compact_x(), species.compact_y(), species.compact_x_prev(),
      species.compact_y_prev(), species.compact_vx(), species.compact_vy(),
      species.compact_vz(), species.compact_vphi_deposit(),
      species.compact_weight(), species.compact_alive(), species.compact_id(),
      include_x_high != 0, include_y_high != 0,
      species.compact_counter(), species.departure_counter());
  QUASAR_HIP_CHECK(::hipGetLastError());

  std::array<unsigned int, 2> counts{};
  QUASAR_HIP_CHECK(::hipMemcpyAsync(
      counts.data(), species.compact_counter(), 2 * sizeof(unsigned int),
      ::hipMemcpyDeviceToHost, stream));
  QUASAR_HIP_CHECK(::hipStreamSynchronize(stream));
  const std::size_t resident_count = counts[0];
  const std::size_t departure_count = counts[1];
  if (resident_count + departure_count != count) {
    throw std::runtime_error{
        "PIC device departure partition produced an invalid count"};
  }
  if (departure_count == 0) return departing;

  departing.x.resize(departure_count);
  departing.y.resize(departure_count);
  departing.x_prev.resize(departure_count);
  departing.y_prev.resize(departure_count);
  departing.vx.resize(departure_count);
  departing.vy.resize(departure_count);
  departing.vz.resize(departure_count);
  departing.vphi_deposit.resize(departure_count);
  departing.weight.resize(departure_count);
  departing.alive.resize(departure_count);
  departing.id.resize(departure_count);

  const auto copy_departing =
      [stream, resident_count, departure_count](void* destination,
                                                const void* source,
                                                std::size_t element_size) {
        const auto* bytes = static_cast<const std::byte*>(source);
        QUASAR_HIP_CHECK(::hipMemcpyAsync(
            destination, bytes + resident_count * element_size,
            departure_count * element_size, ::hipMemcpyDeviceToHost,
            stream));
      };
  copy_departing(departing.x.data(), species.compact_x(), sizeof(Real));
  copy_departing(departing.y.data(), species.compact_y(), sizeof(Real));
  copy_departing(departing.x_prev.data(), species.compact_x_prev(),
                 sizeof(Real));
  copy_departing(departing.y_prev.data(), species.compact_y_prev(),
                 sizeof(Real));
  copy_departing(departing.vx.data(), species.compact_vx(), sizeof(Real));
  copy_departing(departing.vy.data(), species.compact_vy(), sizeof(Real));
  copy_departing(departing.vz.data(), species.compact_vz(), sizeof(Real));
  copy_departing(departing.vphi_deposit.data(),
                 species.compact_vphi_deposit(), sizeof(Real));
  copy_departing(departing.weight.data(), species.compact_weight(),
                 sizeof(Real));
  copy_departing(departing.alive.data(), species.compact_alive(),
                 sizeof(std::uint8_t));
  copy_departing(departing.id.data(), species.compact_id(),
                 sizeof(std::uint64_t));

  const auto copy_residents =
      [stream, resident_count](void* destination, const void* source,
                               std::size_t element_size) {
        if (resident_count == 0) return;
        QUASAR_HIP_CHECK(::hipMemcpyAsync(
            destination, source, resident_count * element_size,
            ::hipMemcpyDeviceToDevice, stream));
      };
  copy_residents(species.x(), species.compact_x(), sizeof(Real));
  copy_residents(species.y(), species.compact_y(), sizeof(Real));
  copy_residents(species.x_prev(), species.compact_x_prev(), sizeof(Real));
  copy_residents(species.y_prev(), species.compact_y_prev(), sizeof(Real));
  copy_residents(species.vx(), species.compact_vx(), sizeof(Real));
  copy_residents(species.vy(), species.compact_vy(), sizeof(Real));
  copy_residents(species.vz(), species.compact_vz(), sizeof(Real));
  copy_residents(species.vphi_deposit(), species.compact_vphi_deposit(),
                 sizeof(Real));
  copy_residents(species.weight(), species.compact_weight(), sizeof(Real));
  copy_residents(species.alive(), species.compact_alive(),
                 sizeof(std::uint8_t));
  copy_residents(species.id(), species.compact_id(),
                 sizeof(std::uint64_t));
  QUASAR_HIP_CHECK(::hipStreamSynchronize(stream));
  species.set_count(resident_count);
  return departing;
}

PicDepartingParticles extract_pic_departing_particles_device(
    ParticleSpecies& species, int include_x_high, int include_y_high,
    backend::stream_t stream_in) {
  // Same partition as the host-snapshot extraction above -- deliberately the
  // same kernel, so the rule for what counts as departing exists once -- but
  // the departing block is copied device-to-device instead of downloaded. The
  // distributed runtime routes it here and never sees the coordinates.
  const auto stream = static_cast<hipStream_t>(stream_in);
  const std::size_t count = species.size();
  PicDepartingParticles departing;
  if (count == 0) return departing;

  QUASAR_HIP_CHECK(::hipMemsetAsync(species.compact_counter(), 0,
                                    2 * sizeof(unsigned int), stream));
  const dim3 block(256);
  const dim3 blocks((count + block.x - 1) / block.x);
  hipLaunchKernelGGL(
      partition_departing_kernel, blocks, block, 0, stream, count,
      species.grid(), species.x(), species.y(), species.x_prev(),
      species.y_prev(), species.vx(), species.vy(), species.vz(),
      species.vphi_deposit(), species.weight(), species.alive(), species.id(),
      species.compact_x(), species.compact_y(), species.compact_x_prev(),
      species.compact_y_prev(), species.compact_vx(), species.compact_vy(),
      species.compact_vz(), species.compact_vphi_deposit(),
      species.compact_weight(), species.compact_alive(), species.compact_id(),
      include_x_high != 0, include_y_high != 0,
      species.compact_counter(), species.departure_counter());
  QUASAR_HIP_CHECK(::hipGetLastError());

  std::array<unsigned int, 2> counts{};
  QUASAR_HIP_CHECK(::hipMemcpyAsync(
      counts.data(), species.compact_counter(), 2 * sizeof(unsigned int),
      ::hipMemcpyDeviceToHost, stream));
  QUASAR_HIP_CHECK(::hipStreamSynchronize(stream));
  const std::size_t resident_count = counts[0];
  const std::size_t departure_count = counts[1];
  if (resident_count + departure_count != count) {
    throw std::runtime_error{
        "PIC device departure partition produced an invalid count"};
  }

  if (departure_count != 0) {
    departing.count = departure_count;
    const auto claim_real = [&](const Real* compact) {
      backend::DeviceBuffer<Real> buffer(departure_count,
                                         backend::uninitialized);
      QUASAR_HIP_CHECK(::hipMemcpyAsync(
          buffer.device_ptr(), compact + resident_count,
          departure_count * sizeof(Real), ::hipMemcpyDeviceToDevice, stream));
      return buffer;
    };
    departing.x = claim_real(species.compact_x());
    departing.y = claim_real(species.compact_y());
    departing.x_prev = claim_real(species.compact_x_prev());
    departing.y_prev = claim_real(species.compact_y_prev());
    departing.vx = claim_real(species.compact_vx());
    departing.vy = claim_real(species.compact_vy());
    departing.vz = claim_real(species.compact_vz());
    departing.vphi_deposit = claim_real(species.compact_vphi_deposit());
    departing.weight = claim_real(species.compact_weight());
    departing.alive = backend::DeviceBuffer<std::uint8_t>(
        departure_count, backend::uninitialized);
    QUASAR_HIP_CHECK(::hipMemcpyAsync(
        departing.alive.device_ptr(),
        species.compact_alive() + resident_count,
        departure_count * sizeof(std::uint8_t), ::hipMemcpyDeviceToDevice,
        stream));
    departing.id = backend::DeviceBuffer<std::uint64_t>(
        departure_count, backend::uninitialized);
    QUASAR_HIP_CHECK(::hipMemcpyAsync(
        departing.id.device_ptr(), species.compact_id() + resident_count,
        departure_count * sizeof(std::uint64_t), ::hipMemcpyDeviceToDevice,
        stream));
  }

  const auto copy_residents =
      [stream, resident_count](void* destination, const void* source,
                               std::size_t element_size) {
        if (resident_count == 0) return;
        QUASAR_HIP_CHECK(::hipMemcpyAsync(
            destination, source, resident_count * element_size,
            ::hipMemcpyDeviceToDevice, stream));
      };
  copy_residents(species.x(), species.compact_x(), sizeof(Real));
  copy_residents(species.y(), species.compact_y(), sizeof(Real));
  copy_residents(species.x_prev(), species.compact_x_prev(), sizeof(Real));
  copy_residents(species.y_prev(), species.compact_y_prev(), sizeof(Real));
  copy_residents(species.vx(), species.compact_vx(), sizeof(Real));
  copy_residents(species.vy(), species.compact_vy(), sizeof(Real));
  copy_residents(species.vz(), species.compact_vz(), sizeof(Real));
  copy_residents(species.vphi_deposit(), species.compact_vphi_deposit(),
                 sizeof(Real));
  copy_residents(species.weight(), species.compact_weight(), sizeof(Real));
  copy_residents(species.alive(), species.compact_alive(),
                 sizeof(std::uint8_t));
  copy_residents(species.id(), species.compact_id(), sizeof(std::uint64_t));
  QUASAR_HIP_CHECK(::hipStreamSynchronize(stream));
  species.set_count(resident_count);
  return departing;
}

}  // namespace quasar::pic
