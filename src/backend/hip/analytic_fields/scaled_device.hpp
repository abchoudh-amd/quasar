// Device helpers for carrying an analytic-field intermediate in
// (mantissa, exponent) form.
//
// The host evaluators these kernels replace did their scaling in `long double`:
// an 80-bit type with a wider exponent than binary64, which let them form
// products such as `moment_scale * mu0_over_4pi * inv_r^3` without the
// intermediate overflowing or underflowing on the way to a perfectly
// representable answer. A device has no `long double`, so the extended range
// has to be carried explicitly. numerics::ScaledValue
// (include/quasar/numerics/scaled_arithmetic.hpp) is exactly that carrier, and
// these are the three operations the field evaluators need on top of it:
// lift a Real into the frame, multiply within the frame, and collapse back with
// the representability check the host `checked_real` performed.
//
// Note this buys *range*, not the extra mantissa bits of x87 long double. Range
// is what the host code was actually relying on -- every quantity it scaled was
// a product or quotient of physically-scaled factors, not a cancelling sum. The
// cancelling sums (the gradient's B0-plus-displacement, the file grid's
// eight-node stencil) go through the exact expansion reducers in
// scaled_arithmetic.hpp instead, which are strictly better than what the host
// did.
//
// Requires -ffp-contract=off, like every other user of that header.
#pragma once

#include "quasar/core/types.hpp"
#include "quasar/numerics/scaled_arithmetic.hpp"

#include <hip/hip_runtime.h>

namespace quasar::backend::analytic_fields_detail {

using ::quasar::Real;
using ::quasar::numerics::ScaledValue;

// Status bits, shared with core::throw_on_evaluator_status.
inline constexpr int kStatusSingular       = 1;
inline constexpr int kStatusNotRepresentable = 2;
inline constexpr int kStatusNonFinitePoint = 4;
inline constexpr int kStatusOutsideGrid    = 8;

inline constexpr unsigned kBlock = 256;

// Lift a Real into the scaled frame. A non-finite input sets `invalid`; zero is
// the empty value, which every consumer treats as "contributes nothing".
__device__ inline ScaledValue sv_from(Real value, bool& invalid) {
  if (value == Real{0}) return ScaledValue{};
  if (!isfinite(value)) {
    invalid = true;
    return ScaledValue{};
  }
  int exponent = 0;
  const Real mantissa = frexp(value, &exponent);
  return ScaledValue{mantissa, exponent};
}

// Multiply within the frame. Only normalized mantissas are ever multiplied, so
// no intermediate can leave binary64's range however extreme the factors are.
__device__ inline ScaledValue sv_times(ScaledValue value, Real factor,
                                       bool& invalid) {
  if (value.mantissa == Real{0} || factor == Real{0}) return ScaledValue{};
  if (!isfinite(factor) || !isfinite(value.mantissa)) {
    invalid = true;
    return ScaledValue{};
  }
  int factor_exponent = 0;
  const Real factor_mantissa = frexp(factor, &factor_exponent);
  int shift = 0;
  const Real mantissa = frexp(value.mantissa * factor_mantissa, &shift);
  return ScaledValue{mantissa, value.exponent + factor_exponent + shift};
}

__device__ inline ScaledValue sv_times(ScaledValue lhs, ScaledValue rhs,
                                       bool& invalid) {
  if (lhs.mantissa == Real{0} || rhs.mantissa == Real{0}) return ScaledValue{};
  if (!isfinite(lhs.mantissa) || !isfinite(rhs.mantissa)) {
    invalid = true;
    return ScaledValue{};
  }
  int shift = 0;
  const Real mantissa = frexp(lhs.mantissa * rhs.mantissa, &shift);
  return ScaledValue{mantissa, lhs.exponent + rhs.exponent + shift};
}

// Collapse to a Real, flagging the value that is finite in the frame but has no
// binary64 representation. This is the device form of the host `checked_real`
// helpers, which threw std::overflow_error at exactly this point; the caller
// turns the flag into that same exception after the launch.
__device__ inline Real sv_to_real(ScaledValue value, bool& overflow) {
  if (value.mantissa == Real{0}) return Real{0};
  if (!isfinite(value.mantissa)) {
    overflow = true;
    return value.mantissa;
  }
  const Real result = scalbn(value.mantissa, value.exponent);
  if (!isfinite(result)) {
    overflow = true;
  }
  // A flush to zero is deliberately NOT an error. The host `checked_real`
  // helpers this replaces bounded only the upper end, and the physics agrees:
  // a dipole field at 2*DBL_MAX metres really is zero to binary64, whereas a
  // field too large to represent is a genuine loss of the answer.
  return result;
}

inline constexpr int kStatusNonFiniteNode  = 16;
inline constexpr int kStatusNotSolenoidal  = 32;

// Exact sum of up to four scaled values, optionally of their magnitudes. The
// reducer grows a non-overlapping expansion, so terms that cancel exactly
// collapse to exactly zero rather than to one ulp of an intermediate -- which
// is the whole point when the question being asked is "is this divergence
// zero?".
__device__ inline ScaledValue sv_sum(const ScaledValue* values, int count,
                                     bool absolute) {
  Real mantissa[4];
  int exponent[4];
  int active = 0;
  for (int k = 0; k < count && active < 4; ++k) {
    if (values[k].mantissa == Real{0}) continue;
    mantissa[active] = absolute ? fabs(values[k].mantissa) : values[k].mantissa;
    exponent[active] = values[k].exponent;
    ++active;
  }
  return ::quasar::numerics::reduce_scaled_terms_to_value(mantissa, exponent,
                                                          active);
}

// Magnitude comparison in the frame, so neither side is materialized first.
__device__ inline bool sv_magnitude_greater(ScaledValue lhs, ScaledValue rhs) {
  if (lhs.mantissa == Real{0}) return false;
  if (rhs.mantissa == Real{0}) return true;
  if (lhs.exponent != rhs.exponent) return lhs.exponent > rhs.exponent;
  return fabs(lhs.mantissa) > fabs(rhs.mantissa);
}

// Publish a thread-local status into the shared word. Integer atomicOr is
// exact and order-independent, so the reported status does not depend on the
// launch geometry.
__device__ inline void publish_status(int* status, int local) {
  if (local != 0) atomicOr(status, local);
}

inline dim3 grid_for(int n) {
  return dim3(static_cast<unsigned>(1 + (n - 1) / static_cast<int>(kBlock)));
}

}  // namespace quasar::backend::analytic_fields_detail
