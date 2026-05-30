#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/pic/launch.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

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
                                       const double* __restrict__ weight,
                                       const unsigned char* __restrict__ alive,
                                       double* __restrict__ ox, double* __restrict__ oy,
                                       double* __restrict__ ox_prev, double* __restrict__ oy_prev,
                                       double* __restrict__ ovx, double* __restrict__ ovy,
                                       double* __restrict__ ovz, double* __restrict__ oweight,
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
  oweight[dst] = weight[p];
  oalive[dst] = 1;
}

// Block-wise reduction of alive flags into per-block partials.
__global__ void alive_count_kernel(std::size_t n, const unsigned char* __restrict__ alive,
                                   unsigned int* __restrict__ partials) {
  __shared__ unsigned int s_count;
  if (threadIdx.x == 0) s_count = 0;
  __syncthreads();
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p < n && alive[p] != 0) atomicAdd(&s_count, 1u);
  __syncthreads();
  if (threadIdx.x == 0) partials[blockIdx.x] = s_count;
}

}  // namespace

extern "C" void launch_pic_particle_compact(quasar::pic::ParticleSpecies& s,
                                            hipStream_t stream) {
  const std::size_t n = s.size();
  if (n == 0) return;

  quasar::backend::DeviceBuffer<double> ox{n}, oy{n}, ox_prev{n}, oy_prev{n};
  quasar::backend::DeviceBuffer<double> ovx{n}, ovy{n}, ovz{n}, oweight{n};
  quasar::backend::DeviceBuffer<std::uint8_t> oalive{n};
  quasar::backend::DeviceBuffer<unsigned int> counter{1};  // zero-init by ctor

  const dim3 block(256);
  const dim3 grid((n + block.x - 1) / block.x);
  hipLaunchKernelGGL(compact_scatter_kernel, grid, block, 0, stream, n,
                     s.x(), s.y(), s.x_prev(), s.y_prev(), s.vx(), s.vy(), s.vz(),
                     s.weight(), s.alive(),
                     ox.device_ptr(), oy.device_ptr(), ox_prev.device_ptr(),
                     oy_prev.device_ptr(), ovx.device_ptr(), ovy.device_ptr(),
                     ovz.device_ptr(), oweight.device_ptr(), oalive.device_ptr(),
                     counter.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());

  unsigned int alive_n = 0;
  counter.copy_to_host(&alive_n, 1);

  if (alive_n > 0) {
    const std::size_t bytes_d = static_cast<std::size_t>(alive_n) * sizeof(double);
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.x(), ox.device_ptr(), bytes_d, ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.y(), oy.device_ptr(), bytes_d, ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.x_prev(), ox_prev.device_ptr(), bytes_d, ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.y_prev(), oy_prev.device_ptr(), bytes_d, ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.vx(), ovx.device_ptr(), bytes_d, ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.vy(), ovy.device_ptr(), bytes_d, ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.vz(), ovz.device_ptr(), bytes_d, ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.weight(), oweight.device_ptr(), bytes_d, ::hipMemcpyDeviceToDevice, stream));
    QUASAR_HIP_CHECK(::hipMemcpyAsync(s.alive(), oalive.device_ptr(),
                                      static_cast<std::size_t>(alive_n) * sizeof(std::uint8_t),
                                      ::hipMemcpyDeviceToDevice, stream));
  }
  QUASAR_HIP_CHECK(::hipStreamSynchronize(stream));
  s.set_count(alive_n);
}

extern "C" std::size_t launch_pic_alive_count(const quasar::pic::ParticleSpecies& s,
                                              hipStream_t stream) {
  const std::size_t n = s.size();
  if (n == 0) return 0;
  const dim3 block(256);
  const dim3 grid((n + block.x - 1) / block.x);
  quasar::backend::DeviceBuffer<unsigned int> partials{grid.x};
  hipLaunchKernelGGL(alive_count_kernel, grid, block, 0, stream, n, s.alive(),
                     partials.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  std::vector<unsigned int> host(grid.x);
  partials.copy_to_host(host.data(), grid.x);
  std::size_t total = 0;
  for (const auto v : host) total += v;
  return total;
}
