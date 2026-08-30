#pragma once

// Device accumulator for sums of strictly positive terms whose FACTORS may be
// individually extreme.
//
// Why this exists rather than plain double-double (reduction_detail.hpp's Acc):
// the PIC energy and Gauss-residual diagnostics sum terms like
// 0.5*m*v^2*w and r^2 * (2*pi*dx*dy*r0), where the mass is ~1e-31, the weight
// ~1e10 and the volume factor ~1e-6. Forming that product directly underflows
// to zero long before the sum is taken, and the sum itself can then be a
// meaningless zero rather than a small number. The host code this replaces
// (pic/diagnostics.cpp::ScaledPositiveSum) solved that by keeping every term in
// a (mantissa, exponent) frame via frexp and never materializing the product.
// This is the device form of the same idea, with the same term semantics:
//
//   * a zero factor contributes nothing and is skipped, not treated as an error
//   * a negative or non-finite factor poisons the whole sum (`invalid`)
//   * the accumulator anchors on the largest exponent seen and rescales
//     upward only, so it is monotone and never loses the leading term
//
// Determinism: a merge rescales into max(exponent) of the two operands, which
// depends only on the values, and the tree shape is fixed by the problem size.
// Two runs and two block counts therefore fold identically.
//
// The final conversion to a Real -- including the overflow/underflow checks
// that must throw -- deliberately stays on the HOST. The kernel returns three
// scalars (sum, correction, exponent) and the host normalizes them. That keeps
// the exception behaviour of the original byte for byte while moving all O(N)
// work to the device.
//
// Requires -ffp-contract=off, as does reduction_detail.hpp.

#include "quasar/core/types.hpp"

#include <hip/hip_runtime.h>

namespace quasar::backend::hip_detail {

// One term, already decomposed. `zero` means "skip me"; `invalid` means the
// whole reduction is poisoned.
struct ScaledTerm {
  Real mantissa{Real{0}};
  int exponent{0};
  bool zero{true};
  bool invalid{false};
};

// Accumulator state.
//
// Deliberately NO default member initializers: this type is declared as a
// __shared__ array in every reduction kernel, and HIP rejects shared variables
// with initializers (a default member initializer makes the default constructor
// non-trivial). Value-initializing with `ScaledAcc a{}` still zeroes every
// field, which is the intended empty state -- sum and correction zero,
// exponent zero, and all three flags false. Shared arrays are written by every
// lane before they are read, so their indeterminate initial contents are never
// observed.
struct ScaledAcc {
  Real sum;
  Real correction;
  int exponent;
  bool initialized;
  bool invalid;
  // Distinct from `invalid` because the callers raise different exceptions: a
  // non-finite FIELD SAMPLE is a std::domain_error naming the field, while a
  // degenerate factor is a std::overflow_error about representability.
  bool nonfinite_input;
};

// Product of `n` factors, kept in (mantissa, exponent) form. Mirrors
// ScaledPositiveSum::add_product's decomposition exactly.
__device__ inline ScaledTerm scaled_product(const Real* factors, int n) {
  Real mantissa = Real{1};
  int exponent = 0;
  for (int k = 0; k < n; ++k) {
    const Real factor = factors[k];
    if (factor == Real{0}) return ScaledTerm{};
    if (!(factor > Real{0}) || !isfinite(factor)) {
      ScaledTerm t{};
      t.invalid = true;
      return t;
    }
    int factor_exponent = 0;
    mantissa *= frexp(factor, &factor_exponent);
    exponent += factor_exponent;
    int adjustment = 0;
    mantissa = frexp(mantissa, &adjustment);
    exponent += adjustment;
  }
  if (!(mantissa > Real{0})) {
    ScaledTerm t{};
    t.invalid = true;
    return t;
  }
  return ScaledTerm{mantissa, exponent, false, false};
}

// Product of `n` factors of ANY sign, kept in (mantissa, exponent) form with
// the sign carried in the mantissa. Used for net-charge sums, where the whole
// point is that positive and negative species cancel; scaled_product would
// reject the negative factor as poison.
__device__ inline ScaledTerm scaled_signed_product(const Real* factors, int n) {
  Real mantissa = Real{1};
  int exponent = 0;
  for (int k = 0; k < n; ++k) {
    const Real factor = factors[k];
    if (factor == Real{0}) return ScaledTerm{};
    if (!isfinite(factor)) {
      ScaledTerm t{};
      t.invalid = true;
      return t;
    }
    int factor_exponent = 0;
    mantissa *= frexp(factor, &factor_exponent);
    exponent += factor_exponent;
    int adjustment = 0;
    mantissa = frexp(mantissa, &adjustment);
    exponent += adjustment;
  }
  if (mantissa == Real{0}) return ScaledTerm{};
  return ScaledTerm{mantissa, exponent, false, false};
}

// value^2 times the product of `n` positive factors. Mirrors
// ScaledPositiveSum::add_square_times. `value` may be negative; it is squared.
__device__ inline ScaledTerm scaled_square_times(Real value,
                                                 const Real* factors, int n) {
  if (value == Real{0}) return ScaledTerm{};
  if (!isfinite(value)) {
    ScaledTerm t{};
    t.invalid = true;
    return t;
  }
  int value_exponent = 0;
  const Real value_mantissa = frexp(fabs(value), &value_exponent);
  Real mantissa = value_mantissa * value_mantissa;
  int exponent = 2 * value_exponent;
  int adjustment = 0;
  mantissa = frexp(mantissa, &adjustment);
  exponent += adjustment;
  for (int k = 0; k < n; ++k) {
    const Real factor = factors[k];
    if (factor == Real{0}) return ScaledTerm{};
    if (!(factor > Real{0}) || !isfinite(factor)) {
      ScaledTerm t{};
      t.invalid = true;
      return t;
    }
    int factor_exponent = 0;
    mantissa *= frexp(factor, &factor_exponent);
    exponent += factor_exponent;
    mantissa = frexp(mantissa, &adjustment);
    exponent += adjustment;
  }
  return ScaledTerm{mantissa, exponent, false, false};
}

// Re-anchor `a` to `exponent`, which must be >= a.exponent. Scaling only ever
// shrinks the stored values, so this cannot overflow.
__device__ inline void scaled_acc_rescale(ScaledAcc& a, int exponent) {
  if (!a.initialized || exponent == a.exponent) return;
  const int shift = a.exponent - exponent;
  a.sum = scalbn(a.sum, shift);
  a.correction = scalbn(a.correction, shift);
  a.exponent = exponent;
}

__device__ inline void scaled_acc_add(ScaledAcc& a, ScaledTerm t) {
  if (t.invalid) {
    a.invalid = true;
    return;
  }
  if (t.zero) return;
  if (!a.initialized) {
    a.exponent = t.exponent;
    a.initialized = true;
  } else if (t.exponent > a.exponent) {
    scaled_acc_rescale(a, t.exponent);
  }
  const Real term = scalbn(t.mantissa, t.exponent - a.exponent);
  const Real next = a.sum + term;
  // Kahan compensation, branch chosen by magnitude so the smaller operand is
  // the one whose lost bits are recovered.
  if (fabs(a.sum) >= fabs(term)) {
    a.correction += (a.sum - next) + term;
  } else {
    a.correction += (term - next) + a.sum;
  }
  a.sum = next;
}

__device__ inline void scaled_acc_merge(ScaledAcc& a, const ScaledAcc& b) {
  a.invalid = a.invalid || b.invalid;
  a.nonfinite_input = a.nonfinite_input || b.nonfinite_input;
  if (!b.initialized) return;
  if (!a.initialized) {
    const bool invalid = a.invalid;
    const bool nonfinite = a.nonfinite_input;
    a = b;
    a.invalid = invalid || b.invalid;
    a.nonfinite_input = nonfinite || b.nonfinite_input;
    return;
  }
  const int exponent = a.exponent > b.exponent ? a.exponent : b.exponent;
  scaled_acc_rescale(a, exponent);
  ScaledAcc rhs = b;
  scaled_acc_rescale(rhs, exponent);
  // Fold the partner's sum and its carried correction as two Kahan terms
  // rather than pre-collapsing them, so the compensation survives the merge.
  for (const Real term : {rhs.sum, rhs.correction}) {
    if (term == Real{0}) continue;
    const Real next = a.sum + term;
    if (fabs(a.sum) >= fabs(term)) {
      a.correction += (a.sum - next) + term;
    } else {
      a.correction += (term - next) + a.sum;
    }
    a.sum = next;
  }
}

// Block fold to lane 0. Caller writes every lane and syncs first.
template <unsigned Threads>
__device__ inline void block_scaled_acc_reduce(ScaledAcc* sdata, unsigned tid) {
  for (unsigned s = Threads / 2; s > 0; s >>= 1) {
    if (tid < s) {
      scaled_acc_merge(sdata[tid], sdata[tid + s]);
    }
    __syncthreads();
  }
}

}  // namespace quasar::backend::hip_detail
