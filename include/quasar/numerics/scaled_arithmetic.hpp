#pragma once

// Extended-exponent scalar arithmetic, shared by every axis that must keep an
// intermediate whose binary exponent leaves the binary64 output range.
//
// The core idea is a `ScaledValue`: a normalized mantissa paired with an
// unbounded integer exponent. Products, quotients and sums are accumulated in
// that frame and collapsed to a `Real` only at the end, so a calculation whose
// answer is representable stays exact even when its intermediates overflow or
// underflow binary64. The sums are exact expansions (Knuth two-sum grown into a
// non-overlapping expansion), not compensated approximations, so distributed
// cancellation collapses to exactly zero rather than to one ulp of an
// intermediate.
//
// Everything here is `QUASAR_HOST_DEVICE` and physics-neutral. It was factored
// out of `mhd_state.hpp`, which still owns the MHD equation-of-state helpers and
// includes this header; the analytic field evaluators
// (src/backend/hip/analytic_fields/) are the second consumer, replacing the
// host-only `long double` scaled products those evaluators used before they
// moved to the device -- a device has no `long double`, so the extended range
// has to be carried explicitly.
//
// Modules using the two-sum must be compiled with `-ffp-contract=off`.
// Contraction silently degrades an exact two-sum into a naive one.

#include "quasar/core/types.hpp"

#include <cmath>
#include <limits>

namespace quasar::numerics {

// Keep the variable-length exponent reducers as device call boundaries.  When
// HIP inlines their runtime-indexed local arrays into a large caller, LLVM can
// duplicate those arrays at every algebraic branch and promote them to LDS.
// A direct device call leaves the small workspaces in private storage once and
// preserves exactly the same reduction order.  Ordinary host builds remain
// header-inline.
#if defined(__HIP_DEVICE_COMPILE__)
#define QUASAR_MHD_NUMERICS_NOINLINE __attribute__((noinline))
#else
#define QUASAR_MHD_NUMERICS_NOINLINE
#endif

// A finite value represented without constraining its binary exponent to the
// binary64 output range.  This is used for intermediate moments which may be
// individually larger than DBL_MAX but cancel to a representable final flux or
// source when appended to a shared exponent accumulator.
struct ScaledValue {
  Real mantissa{};
  int exponent{};
};

// Exact two-sum for independently scaled finite binary64 values.  `high+low`
// equals lhs+rhs exactly.  When the exponent gap is too wide for the smaller
// mantissa to enter the larger one's arithmetic bin, retaining the two inputs
// unchanged is already an exact non-overlapping expansion.
QUASAR_HOST_DEVICE inline void scaled_two_sum(
    Real lhs_mantissa, int lhs_exponent,
    Real rhs_mantissa, int rhs_exponent,
    ScaledValue& high, ScaledValue& low) {
  if (lhs_mantissa == Real{0}) {
    high = ScaledValue{rhs_mantissa, rhs_exponent};
    low = {};
    return;
  }
  if (rhs_mantissa == Real{0}) {
    high = ScaledValue{lhs_mantissa, lhs_exponent};
    low = {};
    return;
  }
  const bool rhs_larger =
      rhs_exponent > lhs_exponent ||
      (rhs_exponent == lhs_exponent &&
       std::fabs(rhs_mantissa) > std::fabs(lhs_mantissa));
  if (rhs_larger) {
    const Real temporary_mantissa = lhs_mantissa;
    const int temporary_exponent = lhs_exponent;
    lhs_mantissa = rhs_mantissa;
    lhs_exponent = rhs_exponent;
    rhs_mantissa = temporary_mantissa;
    rhs_exponent = temporary_exponent;
  }

  const int gap = lhs_exponent - rhs_exponent;
  if (gap > 54) {
    high = ScaledValue{lhs_mantissa, lhs_exponent};
    low = ScaledValue{rhs_mantissa, rhs_exponent};
    return;
  }

  const Real x = lhs_mantissa;
  const Real y = scalbn(rhs_mantissa, -gap);
  const Real rounded = x + y;
  const Real virtual_y = rounded - x;
  const Real error = (x - (rounded - virtual_y)) + (y - virtual_y);
  if (rounded == Real{0}) {
    high = {};
  } else {
    int shift = 0;
    high.mantissa = frexp(rounded, &shift);
    high.exponent = lhs_exponent + shift;
  }
  if (error == Real{0}) {
    low = {};
  } else {
    int shift = 0;
    low.mantissa = frexp(error, &shift);
    low.exponent = lhs_exponent + shift;
  }
}

// Reduce normalized mantissa*2^exponent terms through an exact floating-point
// expansion. Each nonzero finite mantissa must satisfy 0.5 <= |mantissa| < 1;
// all producers in this header establish that invariant with frexp.
// A greedy largest-opposite pairing is insufficient: distributed cancellation
// can leave one ulp of an intermediate (and after scaling, a huge false
// residual) even when the input sum is exactly zero.  Grow-expansion retains
// every two-sum roundoff component, so no information is discarded before the
// final correctly-rounded collapse.
QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline ScaledValue
reduce_scaled_terms_to_value(
    Real* mantissa, int* exponent, int count) {
  int active = 0;
  for (int k = 0; k < count; ++k) {
    if (mantissa[k] == Real{0}) continue;
    mantissa[active] = mantissa[k];
    exponent[active] = exponent[k];
    ++active;
  }
  if (active == 0) return {};

  // Ascending magnitude is the canonical expansion order.  Insertion sort is
  // efficient for these short fixed arrays and device-portable.
  for (int k = 1; k < active; ++k) {
    const Real key_mantissa = mantissa[k];
    const int key_exponent = exponent[k];
    int p = k;
    while (p > 0 &&
           (exponent[p - 1] > key_exponent ||
            (exponent[p - 1] == key_exponent &&
             std::fabs(mantissa[p - 1]) > std::fabs(key_mantissa)))) {
      mantissa[p] = mantissa[p - 1];
      exponent[p] = exponent[p - 1];
      --p;
    }
    mantissa[p] = key_mantissa;
    exponent[p] = key_exponent;
  }

  // Grow one exact, non-overlapping expansion in place.  Before source `s`,
  // the expansion occupies only [0,s), so its writes cannot clobber a future
  // sorted source term.
  int expansion_count = 0;
  for (int source = 0; source < active; ++source) {
    ScaledValue carry{mantissa[source], exponent[source]};
    int write = 0;
    for (int k = 0; k < expansion_count; ++k) {
      const ScaledValue component{mantissa[k], exponent[k]};
      ScaledValue high, low;
      scaled_two_sum(carry.mantissa, carry.exponent,
                     component.mantissa, component.exponent, high, low);
      if (low.mantissa != Real{0}) {
        mantissa[write] = low.mantissa;
        exponent[write] = low.exponent;
        ++write;
      }
      carry = high;
    }
    if (carry.mantissa != Real{0}) {
      mantissa[write] = carry.mantissa;
      exponent[write] = carry.exponent;
      ++write;
    }
    expansion_count = write;
  }
  if (expansion_count == 0) return {};

  // Finalize like a correctly-rounded expansion sum (the same half-even fix
  // used by robust fsum implementations), after normalizing to the largest
  // remaining exponent.  Normalization prevents physical-range overflow; any
  // component that underflows at this relative scale is far below one ulp and
  // cannot affect the rounded leading component.
  const int scale = exponent[expansion_count - 1];
  int n = expansion_count - 1;
  Real high = scalbn(mantissa[n], exponent[n] - scale);
  Real low = Real{0};
  while (n > 0) {
    const Real x = high;
    --n;
    const Real y = scalbn(mantissa[n], exponent[n] - scale);
    high = x + y;
    const Real virtual_y = high - x;
    low = y - virtual_y;
    if (low != Real{0}) break;
  }
  if (n > 0) {
    const Real next = scalbn(mantissa[n - 1], exponent[n - 1] - scale);
    if ((low < Real{0} && next < Real{0}) ||
        (low > Real{0} && next > Real{0})) {
      const Real doubled_low = Real{2} * low;
      const Real adjusted = high + doubled_low;
      if (adjusted - high == doubled_low) high = adjusted;
    }
  }
  if (high == Real{0}) return {};
  int shift = 0;
  const Real result_mantissa = frexp(high, &shift);
  return ScaledValue{result_mantissa, scale + shift};
}

// Return the correctly rounded difference of two finite binary64 values while
// retaining an exponent outside binary64's materialized range.  This is the
// range-safe counterpart of `minuend - subtrahend`: opposite-sign operands
// near DBL_MAX may have a finite scaled difference even though ordinary
// subtraction overflows before a later small multiplier is applied.
QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline ScaledValue
scaled_difference_to_value(Real minuend, Real subtrahend) {
  if (!std::isfinite(minuend) || !std::isfinite(subtrahend)) {
    return ScaledValue{minuend - subtrahend, 0};
  }
  Real mantissa[2]{};
  int exponent[2]{};
  if (minuend != Real{0}) {
    mantissa[0] = frexp(minuend, &exponent[0]);
  }
  if (subtrahend != Real{0}) {
    mantissa[1] = frexp(-subtrahend, &exponent[1]);
  }
  return reduce_scaled_terms_to_value(mantissa, exponent, 2);
}

QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline Real reduce_scaled_terms(
    Real* mantissa, int* exponent, int count) {
  const ScaledValue value =
      reduce_scaled_terms_to_value(mantissa, exponent, count);
  return scalbn(value.mantissa, value.exponent);
}

// Sum a small number of signed products without forming any product at its
// physical exponent first. This preserves a finite cancellation-reduced result
// even when one or more individual products overflow binary64.
QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline Real
scaled_product_sum(const Real* a, const Real* b, int count) {
  // Eight terms are sufficient for every local ideal-MHD stress component,
  // including the field-split normal stress
  //
  //   rho*v_n^2 + p + B0.b + |b|^2/2
  //       - (B0_n*b + b_n*B0 + b_n*b),
  //
  // without first rounding any of those differently-scaled contributions into
  // a single pressure-like scalar.  Keep this fixed-size for HIP device code.
  constexpr int kMaxTerms = 8;
  if (count < 0 || count > kMaxTerms) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  Real mantissa[kMaxTerms]{};
  int exponent[kMaxTerms]{};
  Real nonfinite_sum = Real{0};
  bool has_nonfinite = false;
  for (int k = 0; k < count; ++k) {
    // Form a product only when an operand is already non-finite. This preserves
    // the ordinary IEEE propagation rules (including 0*inf -> NaN and
    // inf+(-inf) -> NaN) without overflowing any all-finite product.
    if (!std::isfinite(a[k]) || !std::isfinite(b[k])) {
      const Real product = a[k] * b[k];
      nonfinite_sum = has_nonfinite ? nonfinite_sum + product : product;
      has_nonfinite = true;
      continue;
    }
    if (a[k] == Real{0} || b[k] == Real{0}) continue;
    int ea = 0;
    int eb = 0;
    // Use the C/HIP overloads here. ROCm's device math surface provides
    // frexp/scalbn directly; std::frexp/std::ldexp is not portable across all
    // supported hipcc standard-library combinations.
    const Real ma = frexp(a[k], &ea);
    const Real mb = frexp(b[k], &eb);
    const Real m = ma * mb;
    int shift = 0;
    mantissa[k] = frexp(m, &shift);
    exponent[k] = ea + eb + shift;
  }
  const Real finite_sum = reduce_scaled_terms(mantissa, exponent, count);
  return has_nonfinite ? nonfinite_sum + finite_sum : finite_sum;
}

QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline Real product_sum2(
    Real a0, Real b0, Real a1, Real b1) {
  const Real a[2] = {a0, a1};
  const Real b[2] = {b0, b1};
  return scaled_product_sum(a, b, 2);
}

QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline Real product_sum3(
    Real a0, Real b0, Real a1, Real b1, Real a2, Real b2) {
  const Real a[3] = {a0, a1, a2};
  const Real b[3] = {b0, b1, b2};
  return scaled_product_sum(a, b, 3);
}

QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline Real product_sum4(
    Real a0, Real b0, Real a1, Real b1, Real a2, Real b2,
    Real a3, Real b3) {
  const Real a[4] = {a0, a1, a2, a3};
  const Real b[4] = {b0, b1, b2, b3};
  return scaled_product_sum(a, b, 4);
}

// Streaming four-factor accumulator for the cancellation-heavy field-split
// energy algebra.  Callers append each term directly, so the hot HLLD path keeps
// only these mantissa/exponent arrays (~300 bytes) instead of four 24-double
// factor arrays plus a second internal reduction buffer (>1 KiB per GPU thread).
struct ScaledQuaternaryAccumulator {
  static constexpr int kMaxTerms = 24;
  Real mantissa[kMaxTerms]{};
  int exponent[kMaxTerms]{};
  Real nonfinite_sum{};
  int count{};
  bool has_nonfinite{};
};

QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline void
append_scaled_quaternary_product(
    ScaledQuaternaryAccumulator& sum, Real a, Real b, Real c, Real d) {
  if (sum.count >= ScaledQuaternaryAccumulator::kMaxTerms) {
    sum.nonfinite_sum = std::numeric_limits<Real>::quiet_NaN();
    sum.has_nonfinite = true;
    return;
  }
  const int k = sum.count++;
  if (!std::isfinite(a) || !std::isfinite(b) ||
      !std::isfinite(c) || !std::isfinite(d)) {
    const Real product = ((a * b) * c) * d;
    sum.nonfinite_sum = sum.has_nonfinite
                            ? sum.nonfinite_sum + product
                            : product;
    sum.has_nonfinite = true;
    return;
  }
  if (a == Real{0} || b == Real{0} || c == Real{0} || d == Real{0}) return;

  int ea = 0, eb = 0, ec = 0, ed = 0;
  const Real ma = frexp(a, &ea);
  const Real mb = frexp(b, &eb);
  const Real mc = frexp(c, &ec);
  const Real md = frexp(d, &ed);
  const Real m = ((ma * mb) * mc) * md;
  int shift = 0;
  sum.mantissa[k] = frexp(m, &shift);
  sum.exponent[k] = ea + eb + ec + ed + shift;
}

QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline Real
finish_scaled_quaternary_sum(
    ScaledQuaternaryAccumulator& sum) {
  const Real finite_sum =
      reduce_scaled_terms(sum.mantissa, sum.exponent, sum.count);
  return sum.has_nonfinite ? sum.nonfinite_sum + finite_sum : finite_sum;
}

QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline ScaledValue
finish_scaled_quaternary_sum_to_value(
    ScaledQuaternaryAccumulator& sum) {
  if (sum.has_nonfinite) {
    const Real finite_sum =
        reduce_scaled_terms(sum.mantissa, sum.exponent, sum.count);
    return ScaledValue{sum.nonfinite_sum + finite_sum, 0};
  }
  return reduce_scaled_terms_to_value(
      sum.mantissa, sum.exponent, sum.count);
}

// Append (value.mantissa*2^value.exponent)*b*c*d to a quaternary accumulator
// without constraining the intermediate value to binary64's output exponent.
QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline void
append_scaled_value_product(
    ScaledQuaternaryAccumulator& sum, const ScaledValue& value,
    Real b, Real c, Real d) {
  if (sum.count >= ScaledQuaternaryAccumulator::kMaxTerms) {
    sum.nonfinite_sum = std::numeric_limits<Real>::quiet_NaN();
    sum.has_nonfinite = true;
    return;
  }
  const int k = sum.count++;
  if (!std::isfinite(value.mantissa) || !std::isfinite(b) ||
      !std::isfinite(c) || !std::isfinite(d)) {
    const Real product = ((scalbn(value.mantissa, value.exponent) * b) * c) * d;
    sum.nonfinite_sum = sum.has_nonfinite
                            ? sum.nonfinite_sum + product
                            : product;
    sum.has_nonfinite = true;
    return;
  }
  if (value.mantissa == Real{0} || b == Real{0} ||
      c == Real{0} || d == Real{0}) return;

  int ev = 0, eb = 0, ec = 0, ed = 0;
  const Real mv = frexp(value.mantissa, &ev);
  const Real mb = frexp(b, &eb);
  const Real mc = frexp(c, &ec);
  const Real md = frexp(d, &ed);
  const Real m = ((mv * mb) * mc) * md;
  int shift = 0;
  sum.mantissa[k] = frexp(m, &shift);
  sum.exponent[k] = value.exponent + ev + eb + ec + ed + shift;
}

// Sum up to 24 signed products of four factors without forming any product at
// its physical exponent first.  Shorter products use unit factors.  Field-split
// energy fluxes need this form because terms such as v_n*B0_t*b_t must cancel
// B0_n*v_t*b_t before a much smaller thermal/kinetic remainder is rounded; a
// two-factor accumulator cannot safely pre-form either triple product.
QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline Real
scaled_quaternary_product_sum(
    const Real* a, const Real* b, const Real* c, const Real* d, int count) {
  if (count < 0 || count > ScaledQuaternaryAccumulator::kMaxTerms) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  ScaledQuaternaryAccumulator sum;
  for (int k = 0; k < count; ++k) {
    append_scaled_quaternary_product(sum, a[k], b[k], c[k], d[k]);
  }
  return finish_scaled_quaternary_sum(sum);
}

// Sum terms of the form
//
//                 a[k] * b[k]
//   result = sum -----------------
//                d0[k] * d1[k]
//
// without first forming either product or either quotient at its physical
// exponent.  This is the range-safe primitive needed by cylindrical metric
// terms: a face difference divided by a very large spacing can be finite even
// when the difference itself overflows, and two individually overflowing
// stresses can cancel to a finite curvature source.  Every denominator is kept
// as two factors so expressions such as m_phi^2/(rho*r) do not overflow or
// underflow while first forming rho*r or 1/rho.
//
// The bound is a template parameter so each caller reserves exactly the
// fixed-size thread-local storage its fused operator requires. The ordinary
// standalone stencil below remains capped at six signed terms; larger
// multidimensional residuals instantiate correspondingly larger capacities.
template <int MaxTerms>
struct ScaledProductQuotientAccumulator {
  Real mantissa[MaxTerms]{};
  int exponent[MaxTerms]{};
  Real nonfinite_sum{};
  int count{};
  bool has_nonfinite{};
};

// Append the exact binary product a*b as a two-component scaled expansion.
// Multiplying the normalized mantissas keeps the leading product in range;
// FMA recovers its exact roundoff component.  Integer-coefficient rational
// stencils need both pieces: reducing only rounded coefficient*sample products
// can leave a large false residual even when their exact dyadic numerator is
// zero.
template <int MaxTerms>
QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline void
append_scaled_exact_product(
    ScaledProductQuotientAccumulator<MaxTerms>& sum, Real a, Real b) {
  if (sum.count + 2 > MaxTerms) {
    sum.nonfinite_sum = std::numeric_limits<Real>::quiet_NaN();
    sum.has_nonfinite = true;
    return;
  }
  if (!std::isfinite(a) || !std::isfinite(b)) {
    const Real product = a * b;
    sum.nonfinite_sum = sum.has_nonfinite
                            ? sum.nonfinite_sum + product
                            : product;
    sum.has_nonfinite = true;
    return;
  }
  if (a == Real{0} || b == Real{0}) return;

  int ea = 0, eb = 0;
  const Real ma = frexp(a, &ea);
  const Real mb = frexp(b, &eb);
  const Real high = ma * mb;
  const Real low = fma(ma, mb, -high);
  const auto append_component = [&](Real component) {
    if (component == Real{0}) return;
    const int k = sum.count++;
    int shift = 0;
    sum.mantissa[k] = frexp(component, &shift);
    sum.exponent[k] = ea + eb + shift;
  };
  append_component(high);
  append_component(low);
}

template <int MaxTerms>
QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline ScaledValue
finish_scaled_exact_product_sum_to_value(
    ScaledProductQuotientAccumulator<MaxTerms>& sum) {
  if (sum.has_nonfinite) {
    const Real finite_sum =
        reduce_scaled_terms(sum.mantissa, sum.exponent, sum.count);
    return ScaledValue{sum.nonfinite_sum + finite_sum, 0};
  }
  return reduce_scaled_terms_to_value(
      sum.mantissa, sum.exponent, sum.count);
}

// Divide a retained scaled value without first materializing its possibly
// out-of-range numerator. The denominator mantissa is in [0.5,1), so the
// intermediate quotient remains representable and is scaled only at the end.
// Callers that began with a multi-component exact numerator have already
// rounded that numerator once to ScaledValue; this range-safe division does not
// promise correctly rounded exact-rational output in the rare double-rounding
// case.
QUASAR_HOST_DEVICE inline Real scaled_value_divide(
    const ScaledValue& numerator, Real denominator) {
  if (!std::isfinite(numerator.mantissa) ||
      !std::isfinite(denominator) || denominator == Real{0}) {
    return scalbn(numerator.mantissa, numerator.exponent) / denominator;
  }
  if (numerator.mantissa == Real{0}) {
    return numerator.mantissa / denominator;
  }
  int denominator_exponent = 0;
  const Real denominator_mantissa =
      frexp(denominator, &denominator_exponent);
  int quotient_shift = 0;
  const Real quotient_mantissa = frexp(
      numerator.mantissa / denominator_mantissa, &quotient_shift);
  return scalbn(quotient_mantissa,
                numerator.exponent - denominator_exponent + quotient_shift);
}

template <int MaxTerms>
QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline void
append_scaled_product_quotient(
    ScaledProductQuotientAccumulator<MaxTerms>& sum,
    Real a, Real b, Real d0, Real d1) {
  if (sum.count >= MaxTerms) {
    sum.nonfinite_sum = std::numeric_limits<Real>::quiet_NaN();
    sum.has_nonfinite = true;
    return;
  }
  const int k = sum.count++;
  // Invalid/non-finite inputs retain ordinary IEEE behavior.  A non-finite
  // denominator can still produce a finite zero, so preserve that zero as an
  // ordinary (empty) scaled term rather than poisoning the whole reduction.
  if (!std::isfinite(a) || !std::isfinite(b) ||
      !std::isfinite(d0) || !std::isfinite(d1) ||
      d0 == Real{0} || d1 == Real{0}) {
    const Real term = ((a * b) / d0) / d1;
    if (!std::isfinite(term)) {
      sum.nonfinite_sum = sum.has_nonfinite
                              ? sum.nonfinite_sum + term
                              : term;
      sum.has_nonfinite = true;
    } else if (term != Real{0}) {
      sum.mantissa[k] = frexp(term, &sum.exponent[k]);
    }
    return;
  }
  if (a == Real{0} || b == Real{0}) return;

  int ea = 0, eb = 0, ed0 = 0, ed1 = 0;
  const Real ma = frexp(a, &ea);
  const Real mb = frexp(b, &eb);
  const Real md0 = frexp(d0, &ed0);
  const Real md1 = frexp(d1, &ed1);
  const Real m = ((ma * mb) / md0) / md1;
  int shift = 0;
  sum.mantissa[k] = frexp(m, &shift);
  sum.exponent[k] = ea + eb - ed0 - ed1 + shift;
}

// Append a*b*c/(d0*d1) while retaining all three numerator factors until
// their binary exponents have been separated.  Transverse MHD face
// quadrature uses this to place w_q*B0(q)*b(q) directly into the final cell
// reduction: reducing the quadrature product to one rounded face scalar first
// can erase a small survivor beside cancelling out-of-range products.
template <int MaxTerms>
QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline void
append_scaled_triple_product_quotient(
    ScaledProductQuotientAccumulator<MaxTerms>& sum,
    Real a, Real b, Real c, Real d0, Real d1) {
  if (sum.count >= MaxTerms) {
    sum.nonfinite_sum = std::numeric_limits<Real>::quiet_NaN();
    sum.has_nonfinite = true;
    return;
  }
  const int k = sum.count++;
  if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) ||
      !std::isfinite(d0) || !std::isfinite(d1) ||
      d0 == Real{0} || d1 == Real{0}) {
    const Real term = (((a * b) * c) / d0) / d1;
    if (!std::isfinite(term)) {
      sum.nonfinite_sum = sum.has_nonfinite
                              ? sum.nonfinite_sum + term
                              : term;
      sum.has_nonfinite = true;
    } else if (term != Real{0}) {
      sum.mantissa[k] = frexp(term, &sum.exponent[k]);
    }
    return;
  }
  if (a == Real{0} || b == Real{0} || c == Real{0}) return;

  int ea = 0, eb = 0, ec = 0, ed0 = 0, ed1 = 0;
  const Real ma = frexp(a, &ea);
  const Real mb = frexp(b, &eb);
  const Real mc = frexp(c, &ec);
  const Real md0 = frexp(d0, &ed0);
  const Real md1 = frexp(d1, &ed1);
  const Real m = (((ma * mb) * mc) / md0) / md1;
  int shift = 0;
  sum.mantissa[k] = frexp(m, &shift);
  sum.exponent[k] = ea + eb + ec - ed0 - ed1 + shift;
}

// Append (value.mantissa*2^value.exponent)*coefficient/(d0*d1) without
// materializing `value`.  In particular, an out-of-range transverse covariance
// can cancel its factorized mean contribution inside the final cell reduction.
template <int MaxTerms>
QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline void
append_scaled_value_quotient(
    ScaledProductQuotientAccumulator<MaxTerms>& sum,
    const ScaledValue& value, Real coefficient, Real d0, Real d1) {
  if (sum.count >= MaxTerms) {
    sum.nonfinite_sum = std::numeric_limits<Real>::quiet_NaN();
    sum.has_nonfinite = true;
    return;
  }
  const int k = sum.count++;
  if (!std::isfinite(value.mantissa) || !std::isfinite(coefficient) ||
      !std::isfinite(d0) || !std::isfinite(d1) ||
      d0 == Real{0} || d1 == Real{0}) {
    const Real term = ((scalbn(value.mantissa, value.exponent) * coefficient) /
                       d0) / d1;
    if (!std::isfinite(term)) {
      sum.nonfinite_sum = sum.has_nonfinite
                              ? sum.nonfinite_sum + term
                              : term;
      sum.has_nonfinite = true;
    } else if (term != Real{0}) {
      sum.mantissa[k] = frexp(term, &sum.exponent[k]);
    }
    return;
  }
  if (value.mantissa == Real{0} || coefficient == Real{0}) return;

  int ev = 0, ec = 0, ed0 = 0, ed1 = 0;
  const Real mv = frexp(value.mantissa, &ev);
  const Real mc = frexp(coefficient, &ec);
  const Real md0 = frexp(d0, &ed0);
  const Real md1 = frexp(d1, &ed1);
  const Real m = ((mv * mc) / md0) / md1;
  int shift = 0;
  sum.mantissa[k] = frexp(m, &shift);
  sum.exponent[k] =
      value.exponent + ev + ec - ed0 - ed1 + shift;
}

// Append (value.mantissa*2^value.exponent)*b*c/(d0*d1) without first
// materializing `value`.  The extra numerator factor is kept separate so a
// large scaled difference may be multiplied by a tiny flux without either an
// intermediate overflow or underflow.
template <int MaxTerms>
QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline void
append_scaled_value_product_quotient(
    ScaledProductQuotientAccumulator<MaxTerms>& sum,
    const ScaledValue& value, Real b, Real c, Real d0, Real d1) {
  if (sum.count >= MaxTerms) {
    sum.nonfinite_sum = std::numeric_limits<Real>::quiet_NaN();
    sum.has_nonfinite = true;
    return;
  }
  const int k = sum.count++;
  if (!std::isfinite(value.mantissa) || !std::isfinite(b) ||
      !std::isfinite(c) || !std::isfinite(d0) || !std::isfinite(d1) ||
      d0 == Real{0} || d1 == Real{0}) {
    const Real term = ((((scalbn(value.mantissa, value.exponent) * b) * c) /
                        d0) /
                       d1);
    if (!std::isfinite(term)) {
      sum.nonfinite_sum = sum.has_nonfinite
                              ? sum.nonfinite_sum + term
                              : term;
      sum.has_nonfinite = true;
    } else if (term != Real{0}) {
      sum.mantissa[k] = frexp(term, &sum.exponent[k]);
    }
    return;
  }
  if (value.mantissa == Real{0} || b == Real{0} || c == Real{0}) return;

  int ev = 0, eb = 0, ec = 0, ed0 = 0, ed1 = 0;
  const Real mv = frexp(value.mantissa, &ev);
  const Real mb = frexp(b, &eb);
  const Real mc = frexp(c, &ec);
  const Real md0 = frexp(d0, &ed0);
  const Real md1 = frexp(d1, &ed1);
  const Real m = (((mv * mb) * mc) / md0) / md1;
  int shift = 0;
  sum.mantissa[k] = frexp(m, &shift);
  sum.exponent[k] =
      value.exponent + ev + eb + ec - ed0 - ed1 + shift;
}

// Append (minuend-subtrahend)*multiplier*coefficient/(d0*d1) while
// retaining the difference in scaled form through the multiplication.
template <int MaxTerms>
QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline void
append_scaled_difference_product_quotient(
    ScaledProductQuotientAccumulator<MaxTerms>& sum,
    Real minuend, Real subtrahend, Real multiplier, Real coefficient,
    Real d0, Real d1) {
  const ScaledValue difference =
      scaled_difference_to_value(minuend, subtrahend);
  append_scaled_value_product_quotient(
      sum, difference, multiplier, coefficient, d0, d1);
}

template <int MaxTerms>
QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline Real
finish_scaled_product_quotient_sum(
    ScaledProductQuotientAccumulator<MaxTerms>& sum) {
  const Real finite_sum =
      reduce_scaled_terms(sum.mantissa, sum.exponent, sum.count);
  return sum.has_nonfinite ? sum.nonfinite_sum + finite_sum : finite_sum;
}

template <int MaxTerms>
QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline ScaledValue
finish_scaled_product_quotient_sum_to_value(
    ScaledProductQuotientAccumulator<MaxTerms>& sum) {
  if (sum.has_nonfinite) {
    const Real finite_sum =
        reduce_scaled_terms(sum.mantissa, sum.exponent, sum.count);
    return ScaledValue{sum.nonfinite_sum + finite_sum, 0};
  }
  return reduce_scaled_terms_to_value(
      sum.mantissa, sum.exponent, sum.count);
}

template <int MaxTerms>
QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline Real
scaled_product_quotient_sum_impl(
    const Real* a, const Real* b, const Real* d0, const Real* d1,
    int count) {
  if (count < 0 || count > MaxTerms) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  ScaledProductQuotientAccumulator<MaxTerms> sum;
  for (int k = 0; k < count; ++k) {
    append_scaled_product_quotient(sum, a[k], b[k], d0[k], d1[k]);
  }
  return finish_scaled_product_quotient_sum(sum);
}

QUASAR_HOST_DEVICE inline Real scaled_product_quotient_sum(
    const Real* a, const Real* b, const Real* d0, const Real* d1,
    int count) {
  return scaled_product_quotient_sum_impl<6>(a, b, d0, d1, count);
}

// Legacy extended standalone reduction for cancellation spanning more than six
// signed terms. Fused multidimensional kernels instantiate the accumulator
// directly with their operator-specific capacities.
QUASAR_HOST_DEVICE inline Real scaled_product_quotient_sum_extended(
    const Real* a, const Real* b, const Real* d0, const Real* d1,
    int count) {
  return scaled_product_quotient_sum_impl<20>(a, b, d0, d1, count);
}
#undef QUASAR_MHD_NUMERICS_NOINLINE

}  // namespace quasar::numerics
