// Deterministic device reductions for the Grad-Shafranov solver.
//
// Two reductions, held to two different standards, for a reason worth stating
// rather than burying.
//
// -- Max-norm: bit-exact ------------------------------------------------------
// Maximum is associative and introduces no rounding, so the order a tree
// reduction visits elements in cannot change the answer. The device result
// equals the host's sequential result exactly, and the test asserts equality.
//
// -- Current integral: deterministic, and MORE accurate than the host ---------
// Summation is not associative. The host reference
// (free_boundary.hpp::total_plasma_current) is a naive sequential sum in
// row-major order; a parallel tree sums in a different order and therefore
// lands on a different double. No amount of care makes those bit-equal without
// serializing, which would defeat the port.
//
// The honest resolution is to make the device version strictly better rather
// than merely different: double-double (two-sum) compensated accumulation, in a
// tree whose shape depends only on the grid size. That is
//   * reproducible run-to-run and launch-to-launch (the determinism the port
//     requires), and
//   * closer to the exactly-rounded sum than the host's naive accumulation.
// The equivalence test measures both against a long-double reference and
// asserts the device error does not exceed the host error.
//
// -- No atomics ---------------------------------------------------------------
// Neither reduction uses a floating-point atomic. Both finish on device via a
// second single-block pass over the block partials, so the result can stay
// resident for device-side control flow without a host round trip.
//
// The primitives all three properties rest on (two_sum, the double-double
// accumulator, the block folds) now live in backend/hip/reduction_detail.hpp so
// the PIC and MHD reductions share exactly this machinery rather than growing
// near-copies of it. The -ffp-contract=off requirement travels with them.

#include "quasar/backend/device.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/reduction_detail.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>

namespace quasar::equilibrium {
namespace {

namespace hd = ::quasar::backend::hip_detail;

using hd::Acc;
using hd::acc_add;
using hd::acc_merge;
using hd::acc_value;
using hd::max_propagating_nan;

inline constexpr unsigned kThreads = hd::kReduceThreads;
inline constexpr unsigned kMaxBlocks = hd::kReduceMaxBlocks;

// Interior index decomposition. The reductions run over interior nodes only, so
// the flat reduction index `t` in [0, n_int) maps to grid node
// (1 + t % (nr-2), 1 + t / (nr-2)).
__device__ inline std::size_t interior_index(std::size_t t, int nr) {
  const std::size_t w = static_cast<std::size_t>(nr - 2);
  const std::size_t i = 1 + t % w;
  const std::size_t j = 1 + t / w;
  return i + static_cast<std::size_t>(nr) * j;
}

__device__ inline void block_max_reduce(Real* sdata, unsigned tid) {
  hd::block_max_reduce<kThreads>(sdata, tid);
}

__global__ void max_abs_pass1(const Real* __restrict__ f,
                              Real* __restrict__ partial, std::size_t n_int,
                              int nr) {
  __shared__ Real sdata[kThreads];
  const unsigned tid = threadIdx.x;

  Real acc = Real{0};
  for (std::size_t t = blockIdx.x * blockDim.x + tid; t < n_int;
       t += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    acc = max_propagating_nan(acc, fabs(f[interior_index(t, nr)]));
  }
  sdata[tid] = acc;
  __syncthreads();

  block_max_reduce(sdata, tid);
  if (tid == 0) partial[blockIdx.x] = sdata[0];
}

__global__ void max_abs_pass2(const Real* __restrict__ partial,
                              Real* __restrict__ out, unsigned n_partial) {
  __shared__ Real sdata[kThreads];
  const unsigned tid = threadIdx.x;

  Real acc = Real{0};
  for (unsigned k = tid; k < n_partial; k += kThreads) {
    acc = max_propagating_nan(acc, partial[k]);
  }
  sdata[tid] = acc;
  __syncthreads();

  block_max_reduce(sdata, tid);
  if (tid == 0) out[0] = sdata[0];
}

// Fixed-order tree sum. Each thread accumulates a grid-strided slice with
// double-double compensation, then the block folds thread accumulators
// pairwise. Both the stride pattern and the fold order depend only on the grid
// dimensions, so the result is reproducible.
__global__ void sum_pass1(const Real* __restrict__ f, Real* __restrict__ p_hi,
                          Real* __restrict__ p_lo, std::size_t n_int, int nr) {
  __shared__ Real s_hi[kThreads];
  __shared__ Real s_lo[kThreads];
  const unsigned tid = threadIdx.x;

  Acc a{Real{0}, Real{0}};
  for (std::size_t t = blockIdx.x * blockDim.x + tid; t < n_int;
       t += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    acc_add(a, f[interior_index(t, nr)]);
  }
  s_hi[tid] = a.hi;
  s_lo[tid] = a.lo;
  __syncthreads();

  hd::block_acc_reduce<kThreads>(s_hi, s_lo, tid);
  if (tid == 0) {
    p_hi[blockIdx.x] = s_hi[0];
    p_lo[blockIdx.x] = s_lo[0];
  }
}

__global__ void sum_pass2(const Real* __restrict__ p_hi,
                          const Real* __restrict__ p_lo,
                          Real* __restrict__ out, unsigned n_partial,
                          Real scale) {
  __shared__ Real s_hi[kThreads];
  __shared__ Real s_lo[kThreads];
  const unsigned tid = threadIdx.x;

  Acc a{Real{0}, Real{0}};
  for (unsigned k = tid; k < n_partial; k += kThreads) {
    const Acc y{p_hi[k], p_lo[k]};
    acc_merge(a, y);
  }
  s_hi[tid] = a.hi;
  s_lo[tid] = a.lo;
  __syncthreads();

  hd::block_acc_reduce<kThreads>(s_hi, s_lo, tid);
  // The cell-area weight is applied once at the very end rather than per
  // element, matching the host (which sums bare values then multiplies) and
  // avoiding nr*nz extra roundings.
  if (tid == 0) out[0] = acc_value(Acc{s_hi[0], s_lo[0]}) * scale;
}

unsigned block_count(std::size_t n_int) {
  return hd::reduce_block_count(n_int);
}

}  // namespace

void GsReduceScratch::resize(const EllipticGrid& g) {
  (void)g;
  partial_hi = backend::DeviceBuffer<Real>{kMaxBlocks};
  partial_lo = backend::DeviceBuffer<Real>{kMaxBlocks};
  result     = backend::DeviceBuffer<Real>{1};
}

Real copy_scalar_to_host(const GsReduceScratch& scratch, stream_t stream) {
  backend::device_synchronize(stream);
  Real value{0};
  scratch.result.copy_to_host(&value, 1);
  return value;
}

void launch_gs_interior_max_norm(const EllipticGrid& g, const Real* d_f,
                                 GsReduceScratch& scratch, stream_t stream) {
  auto* s = static_cast<hipStream_t>(stream);
  const std::size_t n_int = static_cast<std::size_t>(g.nr - 2)
                          * static_cast<std::size_t>(g.nz - 2);
  const unsigned blocks = block_count(n_int);

  hipLaunchKernelGGL(max_abs_pass1, dim3{blocks}, dim3{kThreads}, 0, s, d_f,
                     scratch.partial_hi.device_ptr(), n_int, g.nr);
  QUASAR_HIP_CHECK(hipGetLastError());

  hipLaunchKernelGGL(max_abs_pass2, dim3{1}, dim3{kThreads}, 0, s,
                     scratch.partial_hi.device_ptr(),
                     scratch.result.device_ptr(), blocks);
  QUASAR_HIP_CHECK(hipGetLastError());
}

void launch_gs_total_plasma_current(const EllipticGrid& g, const Real* d_j_phi,
                                    GsReduceScratch& scratch, stream_t stream) {
  auto* s = static_cast<hipStream_t>(stream);
  const std::size_t n_int = static_cast<std::size_t>(g.nr - 2)
                          * static_cast<std::size_t>(g.nz - 2);
  const unsigned blocks = block_count(n_int);

  hipLaunchKernelGGL(sum_pass1, dim3{blocks}, dim3{kThreads}, 0, s, d_j_phi,
                     scratch.partial_hi.device_ptr(),
                     scratch.partial_lo.device_ptr(), n_int, g.nr);
  QUASAR_HIP_CHECK(hipGetLastError());

  hipLaunchKernelGGL(sum_pass2, dim3{1}, dim3{kThreads}, 0, s,
                     scratch.partial_hi.device_ptr(),
                     scratch.partial_lo.device_ptr(),
                     scratch.result.device_ptr(), blocks, g.dr() * g.dz());
  QUASAR_HIP_CHECK(hipGetLastError());
}

}  // namespace quasar::equilibrium
