#pragma once

// Shared deterministic device reduction primitives.
//
// These were written for the Grad-Shafranov port (gs_reduce.hip) and are now
// the common floor for every reduction in the tree. The properties they exist
// to guarantee are worth restating, because a reduction that merely produces a
// plausible number is easy and none of the callers want that:
//
//   * Deterministic. The block count is a pure function of the problem size
//     (reduce_block_count), and the fold order within a block is fixed, so the
//     tree shape depends only on the problem size and never on scheduling.
//     Repeated runs on the same input are bit-identical. Note this is
//     determinism, not block-count independence: reducing the same data under a
//     deliberately different launch geometry may land on a different last ulp.
//   * Compensated. Summation accumulates a double-double (Knuth two-sum) error
//     term, so the result is closer to the exactly-rounded sum than a naive
//     sequential host loop is. This is what lets a port DELETE its host
//     reference and still claim it did not lose accuracy.
//   * Atomic-free. Both passes finish on device via a single-block second pass
//     over the block partials, so a result can stay device-resident for
//     device-side control flow without a host round trip. Floating-point
//     atomics would also destroy determinism.
//
// -ffp-contract=off IS REQUIRED in any module that includes this header.
// The two-sum identity holds only if each operation rounds separately. If the
// compiler contracts the sequence into an FMA, the intermediate is computed at
// higher precision, `e` stops being the exact residual, and the compensation
// silently degrades into a naive sum. This does not fail loudly; it just makes
// the accuracy claim false. See src/backend/hip/equilibrium/CMakeLists.txt for
// the flag and the measurement behind it.

#include "quasar/core/types.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <limits>

namespace quasar::backend::hip_detail {

// Every reduction in the tree launches at this block size, and the templated
// block folds below are unrolled against it. The cap on first-pass blocks is
// what makes the second pass a single block: 256 threads folding 1024 partials
// is four elements each.
inline constexpr unsigned kReduceThreads = 256;
inline constexpr unsigned kReduceMaxBlocks = 1024;

// Error-free transformation: s = fl(a+b) and e = the exact rounding error, so
// that a + b == s + e exactly in real arithmetic. Knuth's two-sum; it needs no
// assumption about the relative magnitudes of a and b.
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

// fmax() returns the non-NaN operand, which would let a single poisoned cell
// vanish from a max-norm. Diagnostics use these norms to decide a field is
// invalid, so a NaN anywhere must reach the result.
__device__ inline Real max_propagating_nan(Real a, Real b) {
  if (isnan(a) || isnan(b)) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  return fmax(a, b);
}

// Fold `sdata[0..Threads)` to sdata[0] by maximum. Caller must have written
// every lane and synchronized. Launch geometry must be exactly Threads.
template <unsigned Threads>
__device__ inline void block_max_reduce(Real* sdata, unsigned tid) {
  for (unsigned s = Threads / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] = max_propagating_nan(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }
}

// Fold the paired hi/lo planes to lane 0 by compensated addition. Same
// preconditions as block_max_reduce.
template <unsigned Threads>
__device__ inline void block_acc_reduce(Real* s_hi, Real* s_lo, unsigned tid) {
  for (unsigned s = Threads / 2; s > 0; s >>= 1) {
    if (tid < s) {
      Acc x{s_hi[tid], s_lo[tid]};
      const Acc y{s_hi[tid + s], s_lo[tid + s]};
      acc_merge(x, y);
      s_hi[tid] = x.hi;
      s_lo[tid] = x.lo;
    }
    __syncthreads();
  }
}

// First-pass block count. Clamped so the second pass is always one block.
inline unsigned reduce_block_count(std::size_t n) {
  const std::size_t want = (n + kReduceThreads - 1) / kReduceThreads;
  if (want == 0) return 1u;
  return want > kReduceMaxBlocks ? kReduceMaxBlocks
                                 : static_cast<unsigned>(want);
}

}  // namespace quasar::backend::hip_detail
