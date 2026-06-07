// Device reduction kernels for the ideal-MHD solver: the CFL maximum signal
// speed and the constrained-transport div(B) L-infinity diagnostic.
//
// Both quantities are a MAX over all interior cells. HIP has no native
// atomicMax for double, so each launcher uses a TWO-PASS block reduction: a
// kernel computes the per-cell value, reduces it within each block via a
// shared-memory tree to one partial max per block (written to a scratch
// DeviceBuffer), and the host finishes the reduction over the (small)
// block-partials array. The final scalar is then written to the caller's host
// pointer before the launcher returns (device_synchronize'd).

#include "quasar/backend/memory.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <vector>

namespace {

using quasar::Real;
using quasar::numerics::fast_magnetosonic_speed;
using quasar::numerics::MhdBackground;
using quasar::numerics::MhdState;

// 16x16 = 256 threads per block, matching the other MHD update kernels.
inline constexpr unsigned kBlock2D = 16;
inline constexpr unsigned kThreadsPerBlock = kBlock2D * kBlock2D;

inline dim3 mhd_grid_2d(const quasar::Grid2D& g, dim3 block) {
  return dim3((static_cast<unsigned>(g.nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(g.ny) + block.y - 1) / block.y);
}

// Shared-memory tree reduction of `sdata` (length kThreadsPerBlock) to its max,
// left in sdata[0]. tid is the linear thread index within the block.
__device__ inline void block_max_reduce(double* sdata, unsigned tid) {
  for (unsigned stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const double other = sdata[tid + stride];
      if (other > sdata[tid]) sdata[tid] = other;
    }
    __syncthreads();
  }
}

// Per-cell CFL signal speed max(|vx| + c_fast_x, |vy| + c_fast_y), reduced to one
// partial max per block. Cells with non-positive / non-finite rho contribute 0
// (parity with the host scan), as do out-of-range threads.
__global__ void cfl_max_speed_kernel(
    quasar::Grid2D g, int b0_active, Real gamma,
    const double* __restrict__ rho, const double* __restrict__ mx,
    const double* __restrict__ my, const double* __restrict__ mz,
    const double* __restrict__ energy, const double* __restrict__ bx,
    const double* __restrict__ by, const double* __restrict__ bz,
    const double* __restrict__ b0x, const double* __restrict__ b0y,
    const double* __restrict__ b0z,
    double* __restrict__ block_partials) {
  __shared__ double sdata[kThreadsPerBlock];
  const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  double local = 0.0;
  if (i < g.nx && j < g.ny) {
    const std::size_t k = g.index(i, j);
    const Real r = rho[k];
    if ((r > Real{0}) && std::isfinite(r)) {
      MhdState s;
      s.rho = r; s.mx = mx[k]; s.my = my[k]; s.mz = mz[k];
      s.energy = energy[k]; s.bx = bx[k]; s.by = by[k]; s.bz = bz[k];
      MhdBackground bg;  // zero by default => zero-background fast path
      if (b0_active) {
        bg.b0x = b0x[k];
        bg.b0y = b0y[k];
        bg.b0z = b0z[k];
      }
      const Real inv_rho = Real{1} / s.rho;
      const Real vx = std::abs(s.mx * inv_rho);
      const Real vy = std::abs(s.my * inv_rho);
      const Real cfx = fast_magnetosonic_speed(s, bg, 0, gamma);
      const Real cfy = fast_magnetosonic_speed(s, bg, 1, gamma);
      const Real sx = vx + cfx;
      const Real sy = vy + cfy;
      local = (sx > sy) ? sx : sy;
    }
  }

  sdata[tid] = local;
  __syncthreads();
  block_max_reduce(sdata, tid);
  if (tid == 0) {
    block_partials[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
  }
}

// Per-cell |div B| from the face-staggered in-plane field, reduced to one partial
// max per block. Same stencil as the host ChristliebFdCt::divergence_b_linf.
__global__ void ct_divb_linf_kernel(
    quasar::Grid2D g, Real inv_dx, Real inv_dy,
    const double* __restrict__ bx, const double* __restrict__ by,
    double* __restrict__ block_partials) {
  __shared__ double sdata[kThreadsPerBlock];
  const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  double local = 0.0;
  if (i < g.nx && j < g.ny) {
    const Real div =
        (bx[g.index(i + 1, j)] - bx[g.index(i, j)]) * inv_dx +
        (by[g.index(i, j + 1)] - by[g.index(i, j)]) * inv_dy;
    local = std::abs(div);
  }

  sdata[tid] = local;
  __syncthreads();
  block_max_reduce(sdata, tid);
  if (tid == 0) {
    block_partials[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
  }
}

// Finish a block-partials max on the host. Copies the partials to host and folds
// them into a single scalar.
inline Real finish_host_max(const quasar::backend::DeviceBuffer<Real>& partials,
                            std::size_t n_blocks) {
  std::vector<Real> host(n_blocks);
  partials.copy_to_host(host.data(), n_blocks);
  Real m = Real{0};
  for (std::size_t b = 0; b < n_blocks; ++b) {
    if (host[b] > m) m = host[b];
  }
  return m;
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_cfl_max_speed(const MhdField2D<Real>& u, const MhdBackgroundField<Real>& b0,
                              Real gamma, Real* host_max_speed, stream_t stream) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  const std::size_t n_blocks = static_cast<std::size_t>(grid.x) * grid.y;

  backend::DeviceBuffer<Real> block_partials{n_blocks, backend::uninitialized};

  const int b0_active = b0.active ? 1 : 0;
  const Real* b0x = b0.active ? b0.b0x_face.device_ptr() : nullptr;
  const Real* b0y = b0.active ? b0.b0y_face.device_ptr() : nullptr;
  const Real* b0z = b0.active ? b0.b0z_cell.device_ptr() : nullptr;

  hipLaunchKernelGGL(cfl_max_speed_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, b0_active, gamma,
                     u.rho.device_ptr(), u.mx.device_ptr(), u.my.device_ptr(),
                     u.mz.device_ptr(), u.energy.device_ptr(), u.bx_face.device_ptr(),
                     u.by_face.device_ptr(), u.bz_cell.device_ptr(),
                     b0x, b0y, b0z, block_partials.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  backend::device_synchronize(stream);

  *host_max_speed = finish_host_max(block_partials, n_blocks);
}

void launch_mhd_ct_divb_linf(const MhdField2D<Real>& u, Real* host_linf, stream_t stream) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  const std::size_t n_blocks = static_cast<std::size_t>(grid.x) * grid.y;

  backend::DeviceBuffer<Real> block_partials{n_blocks, backend::uninitialized};

  const Real inv_dx = Real{1} / g.dx();
  const Real inv_dy = Real{1} / g.dy();

  hipLaunchKernelGGL(ct_divb_linf_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, inv_dx, inv_dy,
                     u.bx_face.device_ptr(), u.by_face.device_ptr(),
                     block_partials.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  backend::device_synchronize(stream);

  *host_linf = finish_host_max(block_partials, n_blocks);
}

}  // namespace quasar::mhd
