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

#include "quasar/backend/device.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace quasar::equilibrium {
namespace {

inline constexpr unsigned kThreads = 256;
// Cap on the first-pass block count so the second pass is always a single
// block. 256 threads folding 1024 partials is four elements each.
inline constexpr unsigned kMaxBlocks = 1024;

// Error-free transformation: s = fl(a+b) and e = the exact rounding error, so
// that a + b == s + e exactly in real arithmetic. Knuth's two-sum; it needs no
// assumption about the relative magnitudes of a and b.
//
// This depends on -ffp-contract=off (set for the whole equilibrium HIP module).
// The identity holds only if each operation below rounds separately: if the
// compiler contracts any of these into an FMA the intermediate is computed at
// higher precision, `e` stops being the exact residual, and the compensation
// silently degrades to noise. The flag is documented in the module CMakeLists
// as protecting the bit-exactness of the line solve; it is protecting this too.
__device__ inline void two_sum(Real a, Real b, Real& s, Real& e) {
  s = a + b;
  const Real bb = s - a;
  e = (a - (s - bb)) + (b - bb);
}

// Double-double accumulator: `hi` carries the running sum, `lo` the accumulated
// rounding error.
struct Acc {
  Real hi;
  Real lo;
};

__device__ inline void acc_add(Acc& a, Real x) {
  Real s, e;
  two_sum(a.hi, x, s, e);
  a.hi = s;
  a.lo += e;
}

__device__ inline void acc_merge(Acc& a, const Acc& b) {
  Real s, e;
  two_sum(a.hi, b.hi, s, e);
  a.hi = s;
  a.lo += e + b.lo;
}

__device__ inline Real acc_value(const Acc& a) { return a.hi + a.lo; }

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
  for (unsigned s = kThreads / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
}

__global__ void max_abs_pass1(const Real* __restrict__ f,
                              Real* __restrict__ partial, std::size_t n_int,
                              int nr) {
  __shared__ Real sdata[kThreads];
  const unsigned tid = threadIdx.x;

  Real acc = Real{0};
  for (std::size_t t = blockIdx.x * blockDim.x + tid; t < n_int;
       t += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    acc = fmax(acc, fabs(f[interior_index(t, nr)]));
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
    acc = fmax(acc, partial[k]);
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

  for (unsigned s = kThreads / 2; s > 0; s >>= 1) {
    if (tid < s) {
      Acc x{s_hi[tid], s_lo[tid]};
      const Acc y{s_hi[tid + s], s_lo[tid + s]};
      acc_merge(x, y);
      s_hi[tid] = x.hi;
      s_lo[tid] = x.lo;
    }
    __syncthreads();
  }
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

  for (unsigned s = kThreads / 2; s > 0; s >>= 1) {
    if (tid < s) {
      Acc x{s_hi[tid], s_lo[tid]};
      const Acc y{s_hi[tid + s], s_lo[tid + s]};
      acc_merge(x, y);
      s_hi[tid] = x.hi;
      s_lo[tid] = x.lo;
    }
    __syncthreads();
  }
  // The cell-area weight is applied once at the very end rather than per
  // element, matching the host (which sums bare values then multiplies) and
  // avoiding nr*nz extra roundings.
  if (tid == 0) out[0] = acc_value(Acc{s_hi[0], s_lo[0]}) * scale;
}

unsigned block_count(std::size_t n_int) {
  const std::size_t want = (n_int + kThreads - 1) / kThreads;
  if (want == 0) return 1u;
  return want > kMaxBlocks ? kMaxBlocks : static_cast<unsigned>(want);
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
