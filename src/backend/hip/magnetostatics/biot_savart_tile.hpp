#pragma once

#include "quasar/core/types.hpp"

#include "backend/hip/magnetostatics/biot_savart_segment.hpp"

#include <hip/hip_runtime.h>

#include <limits>

namespace quasar::magnetostatics::detail {

// Error-free addition of two exponent-scaled floating-point numbers.  When the
// exponents are too far apart to overlap, the exact expansion is simply the two
// inputs.  Otherwise Knuth's TwoSum is evaluated in the larger exponent's safe
// mantissa domain, producing an exactly equivalent high/low pair.
template <class T>
__device__ __forceinline__ void scaled_two_sum(
    ScaledNumber<T> a, ScaledNumber<T> b,
    ScaledNumber<T>& high, ScaledNumber<T>& low) {
  if (a.mantissa == T{0}) {
    high = b;
    low = {};
    return;
  }
  if (b.mantissa == T{0}) {
    high = a;
    low = {};
    return;
  }

  const bool a_larger = a.exponent > b.exponent
      || (a.exponent == b.exponent && fabs(a.mantissa) >= fabs(b.mantissa));
  const ScaledNumber<T> larger = a_larger ? a : b;
  const ScaledNumber<T> smaller = a_larger ? b : a;
  const int gap = larger.exponent - smaller.exponent;
  if (gap > std::numeric_limits<T>::digits + 2) {
    high = larger;
    low = smaller;
    return;
  }

  const int common_exponent = larger.exponent;
  const T x = scalbn(a.mantissa, a.exponent - common_exponent);
  const T y = scalbn(b.mantissa, b.exponent - common_exponent);
  const T sum = x + y;
  const T virtual_y = sum - x;
  const T error = (x - (sum - virtual_y)) + (y - virtual_y);
  high = normalize_scaled(sum, common_exponent);
  low = normalize_scaled(error, common_exponent);
}

// A short floating-point expansion in an explicit power-of-two scale. Unlike
// one Kahan pair, the expansion retains non-overlapping magnitude bands, so an
// ordered [large, small, -large] source sequence leaves `small` intact. It also
// accepts per-segment ScaledNumber values outside T's material exponent range;
// only the final reduced field is rounded/materialized.
//
// Twelve components cover more than six hundred binary exponent bits in fp64
// (and over two hundred in fp32).  A long, otherwise benign sum can nevertheless
// create more than twelve exact round-off limbs.  In that case the two least
// significant limbs are rounded to one and the discarded residual is recorded
// as a conservative absolute-error bound.  This is a bounded-precision
// accumulator, not an unbounded exact summation: value() reports a numerical
// failure if later cancellation makes the discarded bound material at the
// precision of the returned T.  Ordinary long reductions therefore do not
// fail merely because they generated harmless sub-ulp history, while adversarial
// cancellation is never silently presented as a trustworthy field.
template <class T>
struct ScaledAccumulator {
  static constexpr int kCapacity = 12;
  ScaledNumber<T> terms[kCapacity]{};  // increasing, non-overlapping magnitude
  int count{};

  // Every discarded normalized residual has magnitude < 2^exponent.  The sum
  // of all discarded magnitudes is therefore bounded by
  //
  //   discarded_count * 2^largest_discarded_exponent.
  //
  // Keeping only a monotone maximum and a saturating count makes the error
  // certificate independent of T's material exponent range and cannot itself
  // overflow during a reduction whose segment count is an int.
  int largest_discarded_exponent{};
  int discarded_count{};

  __device__ __forceinline__ void record_discarded(
      ScaledNumber<T> residual) {
    if (residual.mantissa == T{0}) return;
    if (discarded_count == 0
        || residual.exponent > largest_discarded_exponent) {
      largest_discarded_exponent = residual.exponent;
    }
    if (discarded_count < std::numeric_limits<int>::max()) {
      ++discarded_count;
    }
  }

  __device__ __forceinline__ void compact_least_significant(int& size) {
    ScaledNumber<T> high{}, low{};
    scaled_two_sum(terms[0], terms[1], high, low);
    record_discarded(low);

    int output = 0;
    if (high.mantissa != T{0}) terms[output++] = high;
    for (int i = 2; i < size; ++i) terms[output++] = terms[i];
    size = output;
  }

  __device__ __forceinline__ bool discarded_error_is_material(
      ScaledNumber<T> rounded) const {
    if (discarded_count == 0) return false;
    if (rounded.mantissa == T{0}) return true;

    // ceil(log2(discarded_count)) without floating-point conversion.  The
    // saturating count is positive here, so (count - 1) is well-defined.
    int count_exponent = 0;
    unsigned int remaining =
        static_cast<unsigned int>(discarded_count - 1);
    while (remaining != 0u) {
      ++count_exponent;
      remaining >>= 1u;
    }

    const long long bound_exponent =
        static_cast<long long>(largest_discarded_exponent) + count_exponent;
    // Normal values have ulp 2^(exponent-digits).  Subnormals all share the
    // absolute ulp 2^(min_exponent-digits), so a relative-only comparison would
    // spuriously reject well-resolved subnormal results.  Two guard bits keep
    // the conservative uncertainty below one quarter of the output ulp.
    const long long normal_ulp_exponent =
        static_cast<long long>(rounded.exponent)
        - std::numeric_limits<T>::digits;
    const long long subnormal_ulp_exponent =
        static_cast<long long>(std::numeric_limits<T>::min_exponent)
        - std::numeric_limits<T>::digits;
    const long long ulp_exponent = normal_ulp_exponent > subnormal_ulp_exponent
        ? normal_ulp_exponent : subnormal_ulp_exponent;
    return bound_exponent > ulp_exponent - 2;
  }

  __device__ __forceinline__ void add(
      ScaledNumber<T> value, bool& numeric_failure) {
    if (value.mantissa == T{0}) return;
    if (!isfinite(value.mantissa)) {
      numeric_failure = true;
      return;
    }

    // Grow the existing non-overlapping expansion in place. The output index
    // never overtakes the input index, so each old term is read before its slot
    // is reused for the new low component.
    ScaledNumber<T> carry = value;
    int output_count = 0;
    const int old_count = count;
    for (int i = 0; i < old_count; ++i) {
      const ScaledNumber<T> old = terms[i];
      ScaledNumber<T> high{}, low{};
      scaled_two_sum(carry, old, high, low);
      if (low.mantissa != T{0}) terms[output_count++] = low;
      carry = high;
    }
    if (carry.mantissa != T{0}) {
      if (output_count >= kCapacity) {
        compact_least_significant(output_count);
      }
      terms[output_count++] = carry;
    }
    count = output_count;
  }

  __device__ __forceinline__ T value(bool& numeric_failure) const {
    if (count == 0) {
      // A retained zero is trustworthy only when no earlier compaction lost a
      // nonzero residual.  This is the strongest possible cancellation case.
      if (discarded_count != 0) numeric_failure = true;
      return T{0};
    }

    // The expansion is ordered from its least to most significant component.
    // Summing in that order preserves every component capable of affecting the
    // rounded leading term; scaled_two_sum keeps this collapse outside T's
    // material exponent range until the last operation.
    ScaledNumber<T> rounded{};
    for (int i = 0; i < count; ++i) {
      ScaledNumber<T> high{}, low{};
      scaled_two_sum(rounded, terms[i], high, low);
      rounded = high;
      // `low` is below the ulp of `high` by construction.  Later expansion
      // components are larger, so it cannot be promoted again during this
      // final correctly-directed rounding pass.
    }
    if (discarded_error_is_material(rounded)) numeric_failure = true;
    return materialize_scaled(rounded, numeric_failure);
  }
};

// Shared scaffold for the Biot-Savart device kernels (field and field-gradient).
// Both stream the conductor segments through a shared-memory tile and accumulate
// a per-observation-point quantity; they differ only in the accumulator and the
// per-segment body. This helper owns the tile load + cooperative copy + sync;
// the caller supplies `accum(a_seg, b_seg, I)`, invoked once per segment for the
// thread's own observation point `p` (captured by the functor).
template <class T, class SegFn>
__device__ inline void tiled_segment_reduce(
    const T* __restrict__ ax, const T* __restrict__ ay, const T* __restrict__ az,
    const T* __restrict__ bx, const T* __restrict__ by, const T* __restrict__ bz,
    const T* __restrict__ I_, int N, bool active, SegFn accum) {
  __shared__ T s_ax[kTileSegments];
  __shared__ T s_ay[kTileSegments];
  __shared__ T s_az[kTileSegments];
  __shared__ T s_bx[kTileSegments];
  __shared__ T s_by[kTileSegments];
  __shared__ T s_bz[kTileSegments];
  __shared__ T s_I [kTileSegments];

  for (int base = 0; base < N; base += kTileSegments) {
    const int lim = (N - base < kTileSegments) ? (N - base) : kTileSegments;

    for (int t = static_cast<int>(threadIdx.x); t < lim;
         t += static_cast<int>(blockDim.x)) {
      const int g = base + t;
      s_ax[t] = ax[g];
      s_ay[t] = ay[g];
      s_az[t] = az[g];
      s_bx[t] = bx[g];
      s_by[t] = by[g];
      s_bz[t] = bz[g];
      s_I [t] = I_[g];
    }
    __syncthreads();

    if (active) {
      for (int s = 0; s < lim; ++s) {
        accum(::quasar::Vec3T<T>{s_ax[s], s_ay[s], s_az[s]},
              ::quasar::Vec3T<T>{s_bx[s], s_by[s], s_bz[s]}, s_I[s]);
      }
    }

    __syncthreads();
  }
}

}  // namespace quasar::magnetostatics::detail
