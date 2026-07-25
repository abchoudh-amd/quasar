#include "quasar/backend/device.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

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
                                       double* __restrict__ ox, double* __restrict__ oy,
                                       double* __restrict__ ox_prev, double* __restrict__ oy_prev,
                                       double* __restrict__ ovx, double* __restrict__ ovy,
                                       double* __restrict__ ovz, double* __restrict__ oweight,
                                       double* __restrict__ ovphi_deposit,
                                       unsigned char* __restrict__ oalive,
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

}  // namespace

extern "C" void launch_pic_particle_compact(quasar::pic::ParticleSpecies& s,
                                            quasar_stream_t stream_in) {
  const auto stream = static_cast<hipStream_t>(stream_in);
  const std::size_t n = s.size();
  if (n == 0) return;

  // Scratch is owned by the species (capacity-sized, allocated once). Reset only
  // the running counter rather than reallocating ten buffers per call.
  QUASAR_HIP_CHECK(::hipMemsetAsync(s.compact_counter(), 0, sizeof(unsigned int), stream));

  const dim3 block(256);
  const dim3 grid((n + block.x - 1) / block.x);
  hipLaunchKernelGGL(compact_scatter_kernel, grid, block, 0, stream, n,
                     s.x(), s.y(), s.x_prev(), s.y_prev(), s.vx(), s.vy(), s.vz(),
                     s.vphi_deposit(),
                     s.weight(), s.alive(),
                     s.compact_x(), s.compact_y(), s.compact_x_prev(),
                     s.compact_y_prev(), s.compact_vx(), s.compact_vy(),
                     s.compact_vz(), s.compact_weight(), s.compact_vphi_deposit(),
                     s.compact_alive(),
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
