// Cylindrical radial-momentum and standalone geometric-source kernels.

#include "mhd_update_detail.hpp"

#include "backend/hip/hip_check.hpp"

#include <cstdint>
#include <type_traits>

namespace {

using quasar::Real;
using quasar::numerics::MhdBackground;
using quasar::numerics::MhdState;
using quasar::numerics::pressure;

using namespace quasar::mhd::update_detail;

inline constexpr int kRadialMomentumMaxTerms = 308;
inline constexpr int kRadialMomentumCapacity = 320;
static_assert(kRadialMomentumCapacity >= kRadialMomentumMaxTerms);

// An exact online expansion for the fused radial residual.  Retaining one
// (double,int) pair per contribution made the maximum MP7 path reserve almost
// 2 KiB before any recovery helper was called.  Here every finite scaled term
// is decoded into its exact 53 significand bits and accumulated in balanced
// base-2^31 limbs.  Carries are propagated on every insertion, so each limb
// remains an int32_t and the storage is independent of the number of terms.
// No face or tensor point is rounded to an intermediate scalar: cancellation
// may span the complete cell exactly, just as in the former 320-slot logical
// expansion.
//
// For a finite binary64 factor frexp returns an exponent in [-1073,1024].
// The widest lower bound used below is a*b*w/(d0*d1): the two arbitrary
// denominators and the smallest MP7 tensor weight give normalized exponent
// -4201, hence least significand bit -4254.  The two-factor upper bound is
// 4196, and 320 same-sign terms need nine carry bits above exponent 4195.
// The explicit guard range [-4288,4267] leaves margin on both ends and is
// checked at run time before a term is inserted.
struct ExactRadialMomentumAccumulator {
  static constexpr int kRadixBits = 31;
  static constexpr std::int64_t kRadix = std::int64_t{1} << kRadixBits;
  static constexpr std::int64_t kHalfRadix = kRadix >> 1;
  static constexpr std::uint64_t kRadixMask =
      (std::uint64_t{1} << kRadixBits) - 1;
  static constexpr int kMinimumBit = -4288;
  static constexpr int kLimbCount = 276;
  static constexpr int kMaximumBit =
      kMinimumBit + kRadixBits * kLimbCount - 1;

  std::int32_t limb[kLimbCount]{};
  Real nonfinite_sum{};
  int count{};
  bool has_nonfinite{};
  bool out_of_range{};
};

static_assert(ExactRadialMomentumAccumulator::kMaximumBit >= 4204);
static_assert(ExactRadialMomentumAccumulator::kMinimumBit <= -4254);

// Common cells do not need the full fixed-range radix accumulator.  This
// short grow-expansion retains every emitted dyadic term exactly while it fits
// in four non-overlapping components.  Requiring the exact radix fallback as
// soon as a fifth component would be needed is a certificate, rather than a
// cancellation heuristic: an accepted expansion has discarded no bits even
// when its final sum is exactly zero after severe cancellation.
struct FastRadialMomentumAccumulator {
  static constexpr int kMaxComponents = 4;

  Real mantissa[kMaxComponents]{};
  int exponent[kMaxComponents]{};
  int components{};
  int count{};
  bool fallback{};
};

__device__ __attribute__((noinline)) void add_radial_limb(
    ExactRadialMomentumAccumulator& sum, int index, std::int64_t value) {
  while (value != 0) {
    if (index < 0 || index >= ExactRadialMomentumAccumulator::kLimbCount) {
      sum.out_of_range = true;
      return;
    }
    const std::int64_t total =
        static_cast<std::int64_t>(sum.limb[index]) + value;
    std::int64_t carry = total / ExactRadialMomentumAccumulator::kRadix;
    std::int64_t remainder = total % ExactRadialMomentumAccumulator::kRadix;
    if (remainder >= ExactRadialMomentumAccumulator::kHalfRadix) {
      remainder -= ExactRadialMomentumAccumulator::kRadix;
      ++carry;
    } else if (remainder < -ExactRadialMomentumAccumulator::kHalfRadix) {
      remainder += ExactRadialMomentumAccumulator::kRadix;
      --carry;
    }
    sum.limb[index] = static_cast<std::int32_t>(remainder);
    value = carry;
    ++index;
  }
}

__device__ __attribute__((noinline)) void append_radial_scaled_term(
    ExactRadialMomentumAccumulator& sum, Real mantissa, int exponent) {
  if (mantissa == Real{0}) return;

  union RealBits {
    Real value;
    std::uint64_t bits;
  } decoded{};
  decoded.value = std::fabs(mantissa);
  std::uint64_t significand =
      (std::uint64_t{1} << 52) |
      (decoded.bits & ((std::uint64_t{1} << 52) - 1));
  const int least_bit = exponent - 53;
  const int offset =
      least_bit - ExactRadialMomentumAccumulator::kMinimumBit;
  if (offset < 0) {
    sum.out_of_range = true;
    return;
  }
  int index = offset / ExactRadialMomentumAccumulator::kRadixBits;
  const int bit = offset % ExactRadialMomentumAccumulator::kRadixBits;
  const std::int64_t sign = std::signbit(mantissa) ? -1 : 1;

  const int first_bits = ExactRadialMomentumAccumulator::kRadixBits - bit;
  const std::uint64_t first_mask =
      (std::uint64_t{1} << first_bits) - 1;
  add_radial_limb(
      sum, index,
      sign * static_cast<std::int64_t>(
          (significand & first_mask) << bit));
  significand >>= first_bits;
  ++index;
  while (significand != 0) {
    add_radial_limb(
        sum, index,
        sign * static_cast<std::int64_t>(
            significand & ExactRadialMomentumAccumulator::kRadixMask));
    significand >>= ExactRadialMomentumAccumulator::kRadixBits;
    ++index;
  }
}

__device__ inline void append_radial_scaled_term(
    FastRadialMomentumAccumulator& sum, Real mantissa, int exponent) {
  if (sum.fallback || mantissa == Real{0}) return;
  const int least_bit = exponent - 53;
  const int highest_bit = exponent - 1;
  if (least_bit < ExactRadialMomentumAccumulator::kMinimumBit ||
      highest_bit > ExactRadialMomentumAccumulator::kMaximumBit) {
    sum.fallback = true;
    return;
  }

  quasar::numerics::ScaledValue carry{mantissa, exponent};
  int write = 0;
#pragma unroll
  for (int source = 0;
       source < FastRadialMomentumAccumulator::kMaxComponents; ++source) {
    if (source >= sum.components) break;
    quasar::numerics::ScaledValue high, low;
    quasar::numerics::scaled_two_sum(
        carry.mantissa, carry.exponent,
        sum.mantissa[source], sum.exponent[source], high, low);
    if (low.mantissa != Real{0}) {
      if (write >= FastRadialMomentumAccumulator::kMaxComponents) {
        sum.fallback = true;
        return;
      }
      sum.mantissa[write] = low.mantissa;
      sum.exponent[write] = low.exponent;
      ++write;
    }
    carry = high;
  }
  if (carry.mantissa != Real{0}) {
    if (write >= FastRadialMomentumAccumulator::kMaxComponents) {
      sum.fallback = true;
      return;
    }
    sum.mantissa[write] = carry.mantissa;
    sum.exponent[write] = carry.exponent;
    ++write;
  }
  sum.components = write;
}

__device__ inline bool reserve_radial_term(
    ExactRadialMomentumAccumulator& sum) {
  if (sum.count >= kRadialMomentumCapacity) {
    sum.out_of_range = true;
    return false;
  }
  ++sum.count;
  return true;
}

__device__ inline bool reserve_radial_term(
    FastRadialMomentumAccumulator& sum) {
  if (sum.fallback) return false;
  if (sum.count >= kRadialMomentumCapacity) {
    sum.fallback = true;
    return false;
  }
  ++sum.count;
  return true;
}

template <typename Accumulator>
__device__ __attribute__((noinline)) void append_radial_product_quotient(
    Accumulator& sum,
    Real a, Real b, Real d0, Real d1) {
  if (!reserve_radial_term(sum)) return;
  if (!std::isfinite(a) || !std::isfinite(b) ||
      !std::isfinite(d0) || !std::isfinite(d1) ||
      d0 == Real{0} || d1 == Real{0}) {
    if constexpr (std::is_same_v<Accumulator,
                                 FastRadialMomentumAccumulator>) {
      sum.fallback = true;
    } else {
      const Real term = ((a * b) / d0) / d1;
      if (!std::isfinite(term)) {
        sum.nonfinite_sum = sum.has_nonfinite
                                ? sum.nonfinite_sum + term
                                : term;
        sum.has_nonfinite = true;
      } else if (term != Real{0}) {
        int exponent = 0;
        append_radial_scaled_term(sum, frexp(term, &exponent), exponent);
      }
    }
    return;
  }
  if (a == Real{0} || b == Real{0}) return;

  int ea = 0, eb = 0, ed0 = 0, ed1 = 0;
  const Real ma = frexp(a, &ea);
  const Real mb = frexp(b, &eb);
  const Real md0 = frexp(d0, &ed0);
  const Real md1 = frexp(d1, &ed1);
  const Real value = ((ma * mb) / md0) / md1;
  int shift = 0;
  const Real mantissa = frexp(value, &shift);
  append_radial_scaled_term(
      sum, mantissa, ea + eb - ed0 - ed1 + shift);
}

template <typename Accumulator>
__device__ __attribute__((noinline)) void
append_radial_triple_product_quotient(
    Accumulator& sum,
    Real a, Real b, Real c, Real d0, Real d1) {
  if (!reserve_radial_term(sum)) return;
  if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) ||
      !std::isfinite(d0) || !std::isfinite(d1) ||
      d0 == Real{0} || d1 == Real{0}) {
    if constexpr (std::is_same_v<Accumulator,
                                 FastRadialMomentumAccumulator>) {
      sum.fallback = true;
    } else {
      const Real term = (((a * b) * c) / d0) / d1;
      if (!std::isfinite(term)) {
        sum.nonfinite_sum = sum.has_nonfinite
                                ? sum.nonfinite_sum + term
                                : term;
        sum.has_nonfinite = true;
      } else if (term != Real{0}) {
        int exponent = 0;
        append_radial_scaled_term(sum, frexp(term, &exponent), exponent);
      }
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
  const Real value = (((ma * mb) * mc) / md0) / md1;
  int shift = 0;
  const Real mantissa = frexp(value, &shift);
  append_radial_scaled_term(
      sum, mantissa, ea + eb + ec - ed0 - ed1 + shift);
}

__device__ inline bool radial_limb_bit(
    const ExactRadialMomentumAccumulator& sum, int absolute_bit) {
  const int offset =
      absolute_bit - ExactRadialMomentumAccumulator::kMinimumBit;
  if (offset < 0) return false;
  const int index = offset / ExactRadialMomentumAccumulator::kRadixBits;
  if (index >= ExactRadialMomentumAccumulator::kLimbCount) return false;
  const int bit = offset % ExactRadialMomentumAccumulator::kRadixBits;
  return (static_cast<std::uint32_t>(sum.limb[index]) &
          (std::uint32_t{1} << bit)) != 0;
}

__device__ __attribute__((noinline)) bool radial_limb_bits_below(
    const ExactRadialMomentumAccumulator& sum, int absolute_bit) {
  const int offset =
      absolute_bit - ExactRadialMomentumAccumulator::kMinimumBit;
  if (offset <= 0) return false;
  const int limit_limb =
      offset / ExactRadialMomentumAccumulator::kRadixBits;
  const int limit_bit =
      offset % ExactRadialMomentumAccumulator::kRadixBits;
  for (int index = 0; index < limit_limb; ++index) {
    if (sum.limb[index] != 0) return true;
  }
  if (limit_bit == 0 ||
      limit_limb >= ExactRadialMomentumAccumulator::kLimbCount) {
    return false;
  }
  const std::uint32_t mask =
      (std::uint32_t{1} << limit_bit) - 1;
  return (static_cast<std::uint32_t>(sum.limb[limit_limb]) & mask) != 0;
}

__device__ __attribute__((noinline)) Real finish_exact_radial_momentum_sum(
    ExactRadialMomentumAccumulator& sum) {
  if (sum.out_of_range) {
    return std::numeric_limits<Real>::quiet_NaN();
  }

  const auto normalize_nonnegative = [&]() {
    for (int index = 0;
         index + 1 < ExactRadialMomentumAccumulator::kLimbCount; ++index) {
      const std::int64_t value = sum.limb[index];
      std::int64_t carry = value / ExactRadialMomentumAccumulator::kRadix;
      std::int64_t remainder = value % ExactRadialMomentumAccumulator::kRadix;
      if (remainder < 0) {
        remainder += ExactRadialMomentumAccumulator::kRadix;
        --carry;
      }
      sum.limb[index] = static_cast<std::int32_t>(remainder);
      sum.limb[index + 1] = static_cast<std::int32_t>(
          static_cast<std::int64_t>(sum.limb[index + 1]) + carry);
    }
  };

  normalize_nonnegative();
  const bool negative =
      sum.limb[ExactRadialMomentumAccumulator::kLimbCount - 1] < 0;
  if (negative) {
    for (int index = 0;
         index < ExactRadialMomentumAccumulator::kLimbCount; ++index) {
      sum.limb[index] = -sum.limb[index];
    }
    normalize_nonnegative();
  }

  int highest_limb = ExactRadialMomentumAccumulator::kLimbCount - 1;
  while (highest_limb >= 0 && sum.limb[highest_limb] == 0) {
    --highest_limb;
  }
  Real finite_sum = Real{0};
  if (highest_limb >= 0) {
    const std::uint32_t highest_digit =
        static_cast<std::uint32_t>(sum.limb[highest_limb]);
    const int highest_digit_bit = 31 - __builtin_clz(highest_digit);
    int highest_bit = ExactRadialMomentumAccumulator::kMinimumBit +
        highest_limb * ExactRadialMomentumAccumulator::kRadixBits +
        highest_digit_bit;

    if (highest_bit < -1022) {
      // Subnormal results must be rounded directly at their fixed 2^-1074
      // quantum.  Rounding first to 53 relative bits and then calling scalbn
      // would double-round values just above a subnormal midpoint.
      std::uint64_t subnormal = 0;
      for (int absolute_bit = -1023; absolute_bit >= -1074;
           --absolute_bit) {
        subnormal = (subnormal << 1) |
            static_cast<std::uint64_t>(
                radial_limb_bit(sum, absolute_bit));
      }
      const bool round_bit = radial_limb_bit(sum, -1075);
      const bool sticky = radial_limb_bits_below(sum, -1075);
      if (round_bit && (sticky || (subnormal & 1) != 0)) ++subnormal;
      finite_sum = scalbn(
          negative ? -static_cast<Real>(subnormal)
                   : static_cast<Real>(subnormal),
          -1074);
    } else {
      std::uint64_t significand = 0;
      for (int bit = 0; bit < 53; ++bit) {
        significand = (significand << 1) |
            static_cast<std::uint64_t>(
                radial_limb_bit(sum, highest_bit - bit));
      }
      const int round_position = highest_bit - 53;
      const bool round_bit = radial_limb_bit(sum, round_position);
      const bool sticky = radial_limb_bits_below(sum, round_position);
      if (round_bit && (sticky || (significand & 1) != 0)) {
        ++significand;
        if (significand == (std::uint64_t{1} << 53)) {
          significand >>= 1;
          ++highest_bit;
        }
      }
      finite_sum = scalbn(
          negative ? -static_cast<Real>(significand)
                   : static_cast<Real>(significand),
          highest_bit - 52);
    }
  }
  return sum.has_nonfinite ? sum.nonfinite_sum + finite_sum : finite_sum;
}

__device__ __attribute__((noinline)) bool finish_fast_radial_momentum_sum(
    FastRadialMomentumAccumulator& sum, Real& result) {
  if (sum.fallback) return false;
  if (sum.components == 0) {
    result = Real{0};
    return true;
  }

  // append_radial_scaled_term maintains the same ascending, exact grow-
  // expansion invariant used by reduce_scaled_terms_to_value.  Collapse that
  // already-ordered four-component expansion directly so this hot kernel does
  // not share the variable-length reducer's call frame with geometric_source.
  const int scale = sum.exponent[sum.components - 1];
  int n = sum.components - 1;
  Real high = scalbn(sum.mantissa[n], sum.exponent[n] - scale);
  Real low = Real{0};
  while (n > 0) {
    const Real x = high;
    --n;
    const Real y = scalbn(sum.mantissa[n], sum.exponent[n] - scale);
    high = x + y;
    const Real virtual_y = high - x;
    low = y - virtual_y;
    if (low != Real{0}) break;
  }
  if (n > 0) {
    const Real next =
        scalbn(sum.mantissa[n - 1], sum.exponent[n - 1] - scale);
    if ((low < Real{0} && next < Real{0}) ||
        (low > Real{0} && next > Real{0})) {
      const Real doubled_low = Real{2} * low;
      const Real adjusted = high + doubled_low;
      if (adjusted - high == doubled_low) high = adjusted;
    }
  }
  if (high == Real{0}) {
    result = Real{0};
    return true;
  }
  int shift = 0;
  const Real result_mantissa = frexp(high, &shift);
  const int result_exponent = scale + shift;
  if (!std::isfinite(result_mantissa) ||
      result_exponent < std::numeric_limits<Real>::min_exponent) {
    return false;
  }
  const Real materialized = scalbn(result_mantissa, result_exponent);
  if (!std::isfinite(materialized) || materialized == Real{0} ||
      std::fabs(materialized) < std::numeric_limits<Real>::min()) {
    return false;
  }
  result = materialized;
  return true;
}

__device__ inline Real uniform_tensor_weight(int order, int qx, int qy) {
  const Real wx = order == 5
      ? quasar::numerics::kMp5TransverseGaussWeights[qx]
      : quasar::numerics::kMp7TransverseGaussWeights[qx];
  const Real wy = order == 5
      ? quasar::numerics::kMp5TransverseGaussWeights[qy]
      : quasar::numerics::kMp7TransverseGaussWeights[qy];
  return wx * wy;
}

__device__ inline bool finite_admissible_state(
    const MhdState& state, Real gamma) {
  if (!(isfinite(state.rho) && state.rho > Real{0} &&
        isfinite(state.mx) && isfinite(state.my) && isfinite(state.mz) &&
        isfinite(state.energy) && isfinite(state.bx) &&
        isfinite(state.by) && isfinite(state.bz))) {
    return false;
  }
  const Real gas_pressure = pressure(state, gamma);
  return isfinite(gas_pressure) && gas_pressure > Real{0};
}

__device__ inline bool finite_background(const MhdBackground& background) {
  return isfinite(background.b0x) && isfinite(background.b0y) &&
         isfinite(background.b0z);
}

// Stateless tensor-point recovery avoids a runtime-indexed 16-element cache
// (1408 bytes per thread) that HIP placed in scratch.  The retained object is
// only a view; individual point values live in registers for the duration of a
// correction evaluation.
struct DynamicTensorPointSampler {
  const quasar::Grid2D* g{};
  int i{};
  int j{};
  int scheme_order{};
  int collocation_order{};
  quasar::mhd::detail::CellStateView state_view{};
  const double* b0r{};
  const double* b0z{};
  const double* b0phi{};
  const quasar::numerics::RadialTablesView* radial_tables{};

  template <int Component>
  __device__ __attribute__((noinline)) Real state_component(
      int qx, int qy) const {
    return quasar::mhd::detail::point_cell_state_component(
        *g, i, j, scheme_order, qx, qy, state_view, Component,
        *radial_tables, collocation_order);
  }

  __device__ __attribute__((noinline)) MhdState state(
      int qx, int qy) const {
    return MhdState{
        state_component<0>(qx, qy), state_component<1>(qx, qy),
        state_component<2>(qx, qy), state_component<3>(qx, qy),
        state_component<4>(qx, qy), state_component<5>(qx, qy),
        state_component<6>(qx, qy), state_component<7>(qx, qy)};
  }

  template <int Component>
  __device__ __attribute__((noinline)) Real background_component(
      int qx, int qy) const {
    return quasar::mhd::detail::point_cell_background_component(
        *g, i, j, scheme_order, qx, qy, b0r, b0z, b0phi, Component,
        *radial_tables);
  }

  __device__ __attribute__((noinline)) MhdBackground background(
      int qx, int qy) const {
    return MhdBackground{
        background_component<0>(qx, qy),
        background_component<1>(qx, qy),
        background_component<2>(qx, qy)};
  }

};

struct BackgroundFieldView {
  const double* r{};
  const double* z{};
  const double* phi{};
};

// A tensor polynomial has no invariant-domain guarantee. Match the transverse
// HLLD policy: use the MP volume rule only when every recovered point state is
// admissible, otherwise retain the cell-centred tensor-difference operator.
__device__ __attribute__((noinline)) bool dynamic_tensor_points_admissible(
    const DynamicTensorPointSampler& points, Real gamma,
    int background_active) {
  const int count = points.scheme_order == 5 ? 3 : 4;
  for (int qy = 0; qy < count; ++qy) {
    for (int qx = 0; qx < count; ++qx) {
      if (!finite_admissible_state(points.state(qx, qy), gamma)) return false;
      if (background_active != 0 &&
          !finite_background(points.background(qx, qy))) {
        return false;
      }
    }
  }
  return true;
}

// Keep the admissibility probe's complete cell state, background, and sampler
// out of the residual kernel's long-lived register set. The accepted state is
// reloaded only after all face terms have been reduced.
__device__ __attribute__((noinline)) bool dynamic_tensor_cell_admissible(
    const quasar::Grid2D& g, int i, int j, int scheme_order,
    int collocation_order, Real gamma,
    const quasar::mhd::detail::CellStateView& state,
    int background_active, const BackgroundFieldView& background,
    const quasar::numerics::RadialTablesView& radial_tables) {
  const MhdState centre = quasar::mhd::load_cell_state(
      g, state.rho, state.mx, state.my, state.mz, state.energy,
      state.bx, state.by, state.bz, i, j,
      radial_tables, collocation_order);
  MhdBackground centre_background{};
  if (background_active != 0) {
    centre_background = quasar::mhd::load_cell_background(
        g, background.r, background.z, background.phi,
        i, j, radial_tables, collocation_order);
  }
  if (!finite_admissible_state(centre, gamma) ||
      (background_active != 0 &&
       !finite_background(centre_background))) {
    return false;
  }
  const DynamicTensorPointSampler points{
      &g, i, j, scheme_order, collocation_order,
      state, background.r, background.z, background.phi, &radial_tables};
  return dynamic_tensor_points_admissible(
      points, gamma, background_active);
}

__device__ inline bool factorized_product_is_constant(
    bool a_constant, Real reference_a,
    bool b_constant, Real reference_b) {
  return (a_constant && b_constant) ||
      (a_constant && reference_a == Real{0}) ||
      (b_constant && reference_b == Real{0});
}

template <typename Accumulator>
__device__ inline void append_radial_cross_family(
    Accumulator& sum,
    const MhdBackground& background, const MhdBackground& cross_b,
    int dir, int family, Real coefficient, Real weight,
    Real d0, Real d1) {
  if (dir == 0) {
    const Real sign = family == 0 ? Real{-1} : Real{1};
    append_radial_triple_product_quotient(
        sum, background_component(background, family),
        background_component(cross_b, family),
        coefficient * sign * weight, d0, d1);
    return;
  }
  if (family == 0) {
    append_radial_triple_product_quotient(
        sum, background.b0y, cross_b.b0x,
        -coefficient * weight, d0, d1);
  } else {
    append_radial_triple_product_quotient(
        sum, background.b0x, cross_b.b0y,
        -coefficient * weight, d0, d1);
  }
}

template <typename Accumulator>
__device__ inline void append_radial_cross_face_family(
    Accumulator& sum,
    const MhdBackground& background, const MhdBackground& cross_b,
    int dir, int family, Real coefficient, Real weight,
    Real spacing, bool append_dynamic_metric, Real radius) {
  append_radial_cross_family(
      sum, background, cross_b, dir, family,
      coefficient, weight, spacing, Real{1});
  if (append_dynamic_metric) {
    append_radial_cross_family(
        sum, background, cross_b, dir, family,
        Real{-1}, weight, radius, Real{2});
  }
}

template <typename Accumulator>
__device__ inline void append_radial_cross_average(
    Accumulator& sum,
    const MhdBackground& background, const MhdBackground& cross_b,
    int dir, Real coefficient, Real d0, Real d1) {
  if (dir == 0) {
    for (int component = 0; component < 3; ++component) {
      const Real sign = component == 0 ? Real{-1} : Real{1};
      append_radial_product_quotient(
          sum, background_component(background, component),
          coefficient * sign * background_component(cross_b, component),
          d0, d1);
    }
    return;
  }
  append_radial_product_quotient(
      sum, background.b0y, -coefficient * cross_b.b0x, d0, d1);
  append_radial_product_quotient(
      sum, background.b0x, -coefficient * cross_b.b0y, d0, d1);
}

template <typename Accumulator>
__device__ inline void append_radial_static_family(
    Accumulator& sum,
    const MhdBackground& background, int dir, int family,
    Real coefficient, Real weight, Real d0, Real d1) {
  if (dir == 0) {
    const Real sign = family == 0 ? Real{-0.5} : Real{0.5};
    const Real value = background_component(background, family);
    append_radial_triple_product_quotient(
        sum, value, value, coefficient * sign * weight, d0, d1);
    return;
  }
  append_radial_triple_product_quotient(
      sum, background.b0y, background.b0x,
      -coefficient * weight, d0, d1);
}

template <typename Accumulator>
__device__ inline void append_radial_static_face_family(
    Accumulator& sum,
    const MhdBackground& background, int dir, int family,
    Real coefficient, Real weight, Real spacing,
    bool append_static_metric, Real radius) {
  append_radial_static_family(
      sum, background, dir, family,
      coefficient, weight, spacing, Real{1});
  if (append_static_metric) {
    append_radial_static_family(
        sum, background, dir, family,
        Real{-1}, weight, radius, Real{2});
  }
}

template <typename Accumulator>
__device__ inline void append_radial_static_average(
    Accumulator& sum,
    const MhdBackground& background, int dir,
    Real coefficient, Real d0, Real d1) {
  if (dir == 0) {
    for (int component = 0; component < 3; ++component) {
      const Real sign = component == 0 ? Real{-0.5} : Real{0.5};
      const Real value = background_component(background, component);
      append_radial_product_quotient(
          sum, value, coefficient * sign * value, d0, d1);
    }
    return;
  }
  append_radial_product_quotient(
      sum, background.b0y, -coefficient * background.b0x, d0, d1);
}

// Stream one radial-momentum face directly into the cell-wide exact expansion.
// A valid MP cross face shares each recovered background point with the static
// stress. When cross quadrature is invalid, the cross term keeps its
// conservative average fallback while a non-curl-free static field still
// recovers and integrates its independently valid points. The first point pass
// detects factorized constant families; the second emits varying quadrature
// terms or one unit-weight reference for each constant family. This deliberate
// second recovery pass avoids a runtime-indexed point cache while preventing
// the exact accumulator from exposing a stored Gauss row's binary normalization
// residue against the matching tensor shortcut.
template <typename Accumulator>
__device__ __attribute__((noinline)) void append_radial_momentum_face(
    Accumulator& sum,
    const quasar::Grid2D& g, const quasar::mhd::BoundaryFlags4& flags,
    int dir, const BackgroundFieldView& background,
    int i, int j, int collocation_order, int scheme_order,
    const CrossBPointView& cross_view, std::size_t k,
    int decomposed_flux, bool append_static, Real coefficient,
    Real spacing, bool append_dynamic_metric, Real radius,
    const quasar::numerics::RadialTablesView& radial_tables) {
  const MhdBackground average_background = mapped_face_background<true>(
      g, flags, dir, background.r, background.z, background.phi, i, j,
      collocation_order, radial_tables);
  MhdBackground average_cross{};
  if (decomposed_flux != 0) {
    average_cross = load_cross_b_point(cross_view, 0, k);
  }
  const bool mp_face = scheme_order == 5 || scheme_order == 7;
  const bool valid_cross_points =
      decomposed_flux != 0 && mp_face && cross_view.quadrature_valid[k] != 0;

  if (mp_face && (valid_cross_points || append_static)) {
    const int point_count = scheme_order == 5 ? 3 : 4;
    const Real* point_weights = face_quadrature_weight_row<true>(
        dir, i, point_count, radial_tables);
    const MhdBackground reference_background =
        mapped_point_face_background<true>(
            g, flags, dir, background.r, background.z, background.phi,
            i, j, scheme_order, /*point=*/0, radial_tables);
    MhdBackground reference_cross{};
    if (valid_cross_points) {
      reference_cross = load_cross_b_point(cross_view, 0, k);
    }

    bool b0r_constant = true;
    bool b0z_constant = true;
    bool b0phi_constant = true;
    bool cross_r_constant = true;
    bool cross_z_constant = true;
    bool cross_phi_constant = true;
    for (int q = 1; q < point_count; ++q) {
      const MhdBackground point_background =
          mapped_point_face_background<true>(
              g, flags, dir, background.r, background.z, background.phi,
              i, j, scheme_order, q, radial_tables);
      b0r_constant = b0r_constant &&
          point_background.b0x == reference_background.b0x;
      b0z_constant = b0z_constant &&
          point_background.b0y == reference_background.b0y;
      b0phi_constant = b0phi_constant &&
          point_background.b0z == reference_background.b0z;
      if (valid_cross_points) {
        const MhdBackground point_cross =
            load_cross_b_point(cross_view, q, k);
        cross_r_constant = cross_r_constant &&
            point_cross.b0x == reference_cross.b0x;
        cross_z_constant = cross_z_constant &&
            point_cross.b0y == reference_cross.b0y;
        cross_phi_constant = cross_phi_constant &&
            point_cross.b0z == reference_cross.b0z;
      }
    }

    const bool cross_rr_constant = factorized_product_is_constant(
        b0r_constant, reference_background.b0x,
        cross_r_constant, reference_cross.b0x);
    const bool cross_zz_constant = factorized_product_is_constant(
        b0z_constant, reference_background.b0y,
        cross_z_constant, reference_cross.b0y);
    const bool cross_phiphi_constant = factorized_product_is_constant(
        b0phi_constant, reference_background.b0z,
        cross_phi_constant, reference_cross.b0z);
    const bool cross_zr_constant = factorized_product_is_constant(
        b0z_constant, reference_background.b0y,
        cross_r_constant, reference_cross.b0x);
    const bool cross_rz_constant = factorized_product_is_constant(
        b0r_constant, reference_background.b0x,
        cross_z_constant, reference_cross.b0y);
    const bool static_zr_constant = factorized_product_is_constant(
        b0z_constant, reference_background.b0y,
        b0r_constant, reference_background.b0x);

    for (int q = 0; q < point_count; ++q) {
      const MhdBackground point_background = q == 0
          ? reference_background
          : mapped_point_face_background<true>(
                g, flags, dir, background.r, background.z, background.phi,
                i, j, scheme_order, q, radial_tables);
      if (valid_cross_points) {
        const MhdBackground point_cross = q == 0
            ? reference_cross : load_cross_b_point(cross_view, q, k);
        if (dir == 0) {
          if (!cross_rr_constant) {
            append_radial_cross_face_family(
                sum, point_background, point_cross, dir, 0,
                coefficient, point_weights[q], spacing,
                append_dynamic_metric, radius);
          }
          if (!cross_zz_constant) {
            append_radial_cross_face_family(
                sum, point_background, point_cross, dir, 1,
                coefficient, point_weights[q], spacing,
                append_dynamic_metric, radius);
          }
          if (!cross_phiphi_constant) {
            append_radial_cross_face_family(
                sum, point_background, point_cross, dir, 2,
                coefficient, point_weights[q], spacing,
                append_dynamic_metric, radius);
          }
        } else {
          if (!cross_zr_constant) {
            append_radial_cross_face_family(
                sum, point_background, point_cross, dir, 0,
                coefficient, point_weights[q], spacing,
                /*append_dynamic_metric=*/false, radius);
          }
          if (!cross_rz_constant) {
            append_radial_cross_face_family(
                sum, point_background, point_cross, dir, 1,
                coefficient, point_weights[q], spacing,
                /*append_dynamic_metric=*/false, radius);
          }
        }
      }
      if (append_static) {
        if (dir == 0) {
          if (!b0r_constant) {
            append_radial_static_face_family(
                sum, point_background, dir, 0, coefficient,
                point_weights[q], spacing,
                /*append_static_metric=*/true, radius);
          }
          if (!b0z_constant) {
            append_radial_static_face_family(
                sum, point_background, dir, 1, coefficient,
                point_weights[q], spacing,
                /*append_static_metric=*/true, radius);
          }
          if (!b0phi_constant) {
            append_radial_static_face_family(
                sum, point_background, dir, 2, coefficient,
                point_weights[q], spacing,
                /*append_static_metric=*/true, radius);
          }
        } else if (!static_zr_constant) {
          append_radial_static_face_family(
              sum, point_background, dir, 0, coefficient,
              point_weights[q], spacing,
              /*append_static_metric=*/false, radius);
        }
      }
    }

    if (valid_cross_points) {
      if (dir == 0) {
        if (cross_rr_constant) {
          append_radial_cross_face_family(
              sum, reference_background, reference_cross, dir, 0,
              coefficient, Real{1}, spacing, append_dynamic_metric, radius);
        }
        if (cross_zz_constant) {
          append_radial_cross_face_family(
              sum, reference_background, reference_cross, dir, 1,
              coefficient, Real{1}, spacing, append_dynamic_metric, radius);
        }
        if (cross_phiphi_constant) {
          append_radial_cross_face_family(
              sum, reference_background, reference_cross, dir, 2,
              coefficient, Real{1}, spacing, append_dynamic_metric, radius);
        }
      } else {
        if (cross_zr_constant) {
          append_radial_cross_face_family(
              sum, reference_background, reference_cross, dir, 0,
              coefficient, Real{1}, spacing,
              /*append_dynamic_metric=*/false, radius);
        }
        if (cross_rz_constant) {
          append_radial_cross_face_family(
              sum, reference_background, reference_cross, dir, 1,
              coefficient, Real{1}, spacing,
              /*append_dynamic_metric=*/false, radius);
        }
      }
    }
    if (append_static) {
      if (dir == 0) {
        if (b0r_constant) {
          append_radial_static_face_family(
              sum, reference_background, dir, 0, coefficient,
              Real{1}, spacing, /*append_static_metric=*/true, radius);
        }
        if (b0z_constant) {
          append_radial_static_face_family(
              sum, reference_background, dir, 1, coefficient,
              Real{1}, spacing, /*append_static_metric=*/true, radius);
        }
        if (b0phi_constant) {
          append_radial_static_face_family(
              sum, reference_background, dir, 2, coefficient,
              Real{1}, spacing, /*append_static_metric=*/true, radius);
        }
      } else if (static_zr_constant) {
        append_radial_static_face_family(
            sum, reference_background, dir, 0, coefficient,
            Real{1}, spacing, /*append_static_metric=*/false, radius);
      }
    }
  }

  if (decomposed_flux != 0 && !valid_cross_points) {
    append_radial_cross_average(
        sum, average_background, average_cross, dir,
        coefficient, spacing, Real{1});
    if (append_dynamic_metric) {
      append_radial_cross_average(
          sum, average_background, average_cross, dir,
          Real{-1}, radius, Real{2});
    }
  }

  if (append_static && !mp_face) {
    append_radial_static_average(
        sum, average_background, dir,
        coefficient, spacing, Real{1});
  }
}

// Append the uniform-cell dynamic T_phiphi quadrature directly to the same
// exact expansion as the metric face stresses.  Algebraically this is the
// former cell T_rr reference plus the two tensor corrections plus the reduced
// cell tensor difference; their three cell-centred pieces cancel exactly.
// Expanding the surviving point quadrature here both avoids that artificial
// work and keeps cancellation available between arbitrary tensor points and
// faces.
template <typename Accumulator>
__device__ __attribute__((noinline)) void
append_dynamic_azimuthal_tensor(
    Accumulator& sum, const quasar::Grid2D& g, int i, int j,
    int scheme_order, int collocation_order, Real gamma,
    const quasar::mhd::detail::CellStateView& state_view,
    int background_active, const BackgroundFieldView& background_view,
    Real radius,
    const quasar::numerics::RadialTablesView& radial_tables) {
  const int count = scheme_order == 5 ? 3 : 4;
  const DynamicTensorPointSampler points{
      &g, i, j, scheme_order, collocation_order,
      state_view, background_view.r, background_view.z,
      background_view.phi, &radial_tables};
  const int centre = count / 2;
  const MhdState reference = points.state(centre, centre);
  const Real reference_pressure = pressure(reference, gamma);
  MhdBackground reference_background{};
  if (background_active != 0) {
    reference_background = points.background(centre, centre);
  }
  bool rho_constant = true;
  bool mphi_constant = true;
  bool pressure_constant = true;
  bool br_constant = true;
  bool bz_constant = true;
  bool bphi_constant = true;
  bool b0r_constant = true;
  bool b0z_constant = true;
  bool b0phi_constant = true;
  for (int qy = 0; qy < count; ++qy) {
    for (int qx = 0; qx < count; ++qx) {
      const MhdState state = qx == centre && qy == centre
          ? reference : points.state(qx, qy);
      rho_constant = rho_constant && state.rho == reference.rho;
      mphi_constant = mphi_constant && state.mz == reference.mz;
      pressure_constant = pressure_constant &&
          pressure(state, gamma) == reference_pressure;
      br_constant = br_constant && state.bx == reference.bx;
      bz_constant = bz_constant && state.by == reference.by;
      bphi_constant = bphi_constant && state.bz == reference.bz;
      if (background_active != 0) {
        const MhdBackground background = qx == centre && qy == centre
            ? reference_background : points.background(qx, qy);
        b0r_constant = b0r_constant &&
            background.b0x == reference_background.b0x;
        b0z_constant = b0z_constant &&
            background.b0y == reference_background.b0y;
        b0phi_constant = b0phi_constant &&
            background.b0z == reference_background.b0z;
      }
    }
  }
  const bool rotation_constant =
      (mphi_constant && rho_constant) ||
      (mphi_constant && reference.mz == Real{0});
  const bool radial_cross_constant =
      (b0r_constant && br_constant) ||
      (b0r_constant && reference_background.b0x == Real{0}) ||
      (br_constant && reference.bx == Real{0});
  const bool axial_cross_constant =
      (b0z_constant && bz_constant) ||
      (b0z_constant && reference_background.b0y == Real{0}) ||
      (bz_constant && reference.by == Real{0});
  const bool azimuthal_cross_constant =
      (b0phi_constant && bphi_constant) ||
      (b0phi_constant && reference_background.b0z == Real{0}) ||
      (bphi_constant && reference.bz == Real{0});

  for (int qy = 0; qy < count; ++qy) {
    for (int qx = 0; qx < count; ++qx) {
      const MhdState state = qx == centre && qy == centre
          ? reference : points.state(qx, qy);
      const Real weight = uniform_tensor_weight(scheme_order, qx, qy);
      if (!rotation_constant) {
        append_radial_triple_product_quotient(
            sum, state.mz, state.mz, weight, state.rho, radius);
      }
      if (!pressure_constant) {
        append_radial_product_quotient(
            sum, pressure(state, gamma), weight, radius, Real{1});
      }
      if (!br_constant) {
        append_radial_triple_product_quotient(
            sum, state.bx, state.bx, Real{0.5} * weight,
            radius, Real{1});
      }
      if (!bz_constant) {
        append_radial_triple_product_quotient(
            sum, state.by, state.by, Real{0.5} * weight,
            radius, Real{1});
      }
      if (!bphi_constant) {
        append_radial_triple_product_quotient(
            sum, state.bz, state.bz, Real{-0.5} * weight,
            radius, Real{1});
      }
      if (background_active != 0) {
        const MhdBackground background = qx == centre && qy == centre
            ? reference_background : points.background(qx, qy);
        if (!radial_cross_constant) {
          append_radial_triple_product_quotient(
              sum, background.b0x, state.bx, weight, radius, Real{1});
        }
        if (!axial_cross_constant) {
          append_radial_triple_product_quotient(
              sum, background.b0y, state.by, weight, radius, Real{1});
        }
        if (!azimuthal_cross_constant) {
          append_radial_triple_product_quotient(
              sum, background.b0z, state.bz, -weight, radius, Real{1});
        }
      }
    }
  }
  if (rotation_constant) {
    append_radial_product_quotient(
        sum, reference.mz, reference.mz, reference.rho, radius);
  }
  if (pressure_constant) {
    append_radial_product_quotient(
        sum, reference_pressure, Real{1}, radius, Real{1});
  }
  if (br_constant) {
    append_radial_product_quotient(
        sum, reference.bx, Real{0.5} * reference.bx,
        radius, Real{1});
  }
  if (bz_constant) {
    append_radial_product_quotient(
        sum, reference.by, Real{0.5} * reference.by,
        radius, Real{1});
  }
  if (bphi_constant) {
    append_radial_product_quotient(
        sum, reference.bz, Real{-0.5} * reference.bz,
        radius, Real{1});
  }
  if (background_active != 0) {
    if (radial_cross_constant) {
      append_radial_product_quotient(
          sum, reference_background.b0x, reference.bx,
          radius, Real{1});
    }
    if (axial_cross_constant) {
      append_radial_product_quotient(
          sum, reference_background.b0y, reference.by,
          radius, Real{1});
    }
    if (azimuthal_cross_constant) {
      append_radial_product_quotient(
          sum, reference_background.b0z, -reference.bz,
          radius, Real{1});
    }
  }
}

// Append <M0_phiphi>_uniform/r without first reducing each field component to
// a face-sized scalar.  This is the static counterpart of the dynamic helper
// above and preserves cancellation with all four metric stress faces.
template <typename Accumulator>
__device__ __attribute__((noinline)) void append_static_azimuthal_tensor(
    Accumulator& sum, const quasar::Grid2D& g, int i, int j,
    int scheme_order, const BackgroundFieldView& background_view,
    Real radius,
    const quasar::numerics::RadialTablesView& radial_tables) {
  const int count = scheme_order == 5 ? 3 : 4;
  const DynamicTensorPointSampler points{
      &g, i, j, scheme_order, /*collocation_order=*/0,
      quasar::mhd::detail::CellStateView{},
      background_view.r, background_view.z, background_view.phi,
      &radial_tables};
  const int centre = count / 2;
  const MhdBackground reference = points.background(centre, centre);
  bool radial_constant = true;
  bool axial_constant = true;
  bool azimuthal_constant = true;
  for (int qy = 0; qy < count; ++qy) {
    for (int qx = 0; qx < count; ++qx) {
      const MhdBackground background = qx == centre && qy == centre
          ? reference : points.background(qx, qy);
      radial_constant = radial_constant && background.b0x == reference.b0x;
      axial_constant = axial_constant && background.b0y == reference.b0y;
      azimuthal_constant = azimuthal_constant &&
          background.b0z == reference.b0z;
    }
  }
  for (int qy = 0; qy < count; ++qy) {
    for (int qx = 0; qx < count; ++qx) {
      const MhdBackground background = qx == centre && qy == centre
          ? reference : points.background(qx, qy);
      const Real weight = uniform_tensor_weight(scheme_order, qx, qy);
      if (!radial_constant) {
        append_radial_triple_product_quotient(
            sum, background.b0x, background.b0x, Real{0.5} * weight,
            radius, Real{1});
      }
      if (!axial_constant) {
        append_radial_triple_product_quotient(
            sum, background.b0y, background.b0y, Real{0.5} * weight,
            radius, Real{1});
      }
      if (!azimuthal_constant) {
        append_radial_triple_product_quotient(
            sum, background.b0z, background.b0z, Real{-0.5} * weight,
            radius, Real{1});
      }
    }
  }
  if (radial_constant) {
    append_radial_product_quotient(
        sum, reference.b0x, Real{0.5} * reference.b0x,
        radius, Real{1});
  }
  if (axial_constant) {
    append_radial_product_quotient(
        sum, reference.b0y, Real{0.5} * reference.b0y,
        radius, Real{1});
  }
  if (azimuthal_constant) {
    append_radial_product_quotient(
        sum, reference.b0z, Real{-0.5} * reference.b0z,
        radius, Real{1});
  }
}

// Complete cylindrical radial-momentum residual in the tensor-difference form
//
//   R_mr = -d_r F_rr - d_z F_zr + (T_phiphi-T_rr)/r.
//
// Unlike an annular aggregate F_rr plus a separately expanded T_phiphi/r,
// this form cancels gas pressure and axial-field terms analytically before
// rounding. That matters when the already-rounded face flux cannot retain a
// small pressure beside a dominant toroidal stress. The split curvature is
//
//   rho(vphi^2-vr^2) + 2 B0r br + br^2
//                         - 2 B0phi bphi - bphi^2,
//
// and the low-order static curvature is B0r^2-B0phi^2. MP5/MP7 evaluate the
// static part in its equivalent annular form: metric-weighted radial face
// stresses plus a uniform-cell M0_phiphi tensor quadrature. The dynamic MP path
// adds the matching radial metric faces and the uniform-cell T_phiphi tensor
// quadrature. Every point and face product enters one exact cell-wide sum: the
// common path uses a certified four-component grow-expansion, and cells that
// require a fifth component or boundary-range rounding are recomputed by the
// full radix kernel. Cancellation can therefore span different points, faces,
// and binary scales without charging every cell for the radix workspace. The
// worst fused active MP7 path appends 308 terms: 12 material/wave terms,
// 64 cross-stress products, 128 dynamic tensor products, 56 static face
// products, and 48 static tensor products. Keep a small explicit capacity
// margin above that proven bound.
struct RadialMomentumKernelArgs {
  quasar::Grid2D g;
  quasar::mhd::BoundaryFlags4 flags;
  int background_active;
  int globally_curl_free;
  int decomposed_flux;
  int collocation_order;
  int scheme_order;
  Real gamma;
  quasar::numerics::RadialTablesView radial_tables;
  const double* rho;
  const double* mr;
  const double* mz;
  const double* mphi;
  const double* energy;
  const double* br;
  const double* bz;
  const double* bphi;
  const double* b0r;
  const double* b0z;
  const double* b0phi;
  const double* radial_momentum_flux;
  const double* axial_momentum_flux;
  const double* radial_wave;
  const double* axial_wave;
  CrossBPointView radial_cross;
  CrossBPointView axial_cross;
  double* radial_momentum_rate;
};

template <typename Accumulator>
__device__ __attribute__((noinline)) void accumulate_radial_momentum_cell(
    const RadialMomentumKernelArgs& args, int i, int j,
    Accumulator& sum) {
  const quasar::Grid2D& g = args.g;
  const quasar::mhd::BoundaryFlags4& flags = args.flags;
  const int background_active = args.background_active;
  const int globally_curl_free = args.globally_curl_free;
  const int decomposed_flux = args.decomposed_flux;
  const int collocation_order = args.collocation_order;
  const int scheme_order = args.scheme_order;
  const Real gamma = args.gamma;
  const quasar::numerics::RadialTablesView& radial_tables =
      args.radial_tables;
  const double* const rho = args.rho;
  const double* const mr = args.mr;
  const double* const mz = args.mz;
  const double* const mphi = args.mphi;
  const double* const energy = args.energy;
  const double* const br = args.br;
  const double* const bz = args.bz;
  const double* const bphi = args.bphi;
  const double* const b0r = args.b0r;
  const double* const b0z = args.b0z;
  const double* const b0phi = args.b0phi;
  const double* const radial_momentum_flux = args.radial_momentum_flux;
  const double* const axial_momentum_flux = args.axial_momentum_flux;
  const double* const radial_wave = args.radial_wave;
  const double* const axial_wave = args.axial_wave;
  const CrossBPointView& radial_cross = args.radial_cross;
  const CrossBPointView& axial_cross = args.axial_cross;
  const quasar::mhd::detail::CellStateView state_view{
      rho, mr, mz, mphi, energy, br, bz, bphi};
  const BackgroundFieldView background_view{b0r, b0z, b0phi};

  const std::size_t k = g.index(i, j);
  const std::size_t kr = g.index(i + 1, j);
  const std::size_t kz = g.index(i, j + 1);
  const Real r = g.r_at_cell_center(i);

  const bool mp_candidate = scheme_order == 5 || scheme_order == 7;
  const bool high_order_dynamic =
      mp_candidate &&
      dynamic_tensor_cell_admissible(
          g, i, j, scheme_order, collocation_order, gamma,
          state_view, background_active, background_view, radial_tables);

  // Ordinary radial tensor derivative. Curvature is supplied explicitly by
  // T_phiphi-T_rr below; using the annular derivative here would double-count
  // F_rr/r and recreate the aggregate-pressure cancellation defect.
  append_radial_product_quotient(
      sum, Real{1}, -radial_momentum_flux[kr], g.dx(), Real{1});
  append_radial_product_quotient(
      sum, Real{1}, radial_momentum_flux[k], g.dx(), Real{1});
  if (high_order_dynamic) {
    // Convert the ordinary radial difference to the annular face divergence.
    // The matching uniform-cell T_rr term is appended below, retaining the
    // pressure-free tensor difference as the other half of the identity.
    append_radial_product_quotient(
        sum, Real{1}, -radial_momentum_flux[kr], r, Real{2});
    append_radial_product_quotient(
        sum, Real{1}, -radial_momentum_flux[k], r, Real{2});
  }

  // -d_z F_{z,mr}.
  append_radial_product_quotient(
      sum, Real{1}, -axial_momentum_flux[kz], g.dy(), Real{1});
  append_radial_product_quotient(
      sum, Real{1}, axial_momentum_flux[k], g.dy(), Real{1});

  if (decomposed_flux != 0) {
    append_radial_product_quotient(
        sum, radial_wave[kr], Real{-1}, g.dx(), Real{1});
    append_radial_product_quotient(
        sum, radial_wave[k], Real{1}, g.dx(), Real{1});
    if (high_order_dynamic) {
      append_radial_product_quotient(
          sum, radial_wave[kr], Real{-1}, r, Real{2});
      append_radial_product_quotient(
          sum, radial_wave[k], Real{-1}, r, Real{2});
    }
    append_radial_product_quotient(
        sum, axial_wave[kz], Real{-1}, g.dy(), Real{1});
    append_radial_product_quotient(
        sum, axial_wave[k], Real{1}, g.dy(), Real{1});
  }

  const bool append_static =
      background_active != 0 && globally_curl_free == 0;
  if (decomposed_flux != 0 || append_static) {
    append_radial_momentum_face(
        sum, g, flags, 0, background_view, i, j,
        collocation_order, scheme_order, radial_cross, k,
        decomposed_flux, append_static, Real{1}, g.dx(),
        high_order_dynamic, r, radial_tables);
    append_radial_momentum_face(
        sum, g, flags, 0, background_view, i + 1, j,
        collocation_order, scheme_order, radial_cross, kr,
        decomposed_flux, append_static, Real{-1}, g.dx(),
        high_order_dynamic, r, radial_tables);
    append_radial_momentum_face(
        sum, g, flags, 1, background_view, i, j,
        collocation_order, scheme_order, axial_cross, k,
        decomposed_flux, append_static, Real{1}, g.dy(),
        /*append_dynamic_metric=*/false, r, radial_tables);
    append_radial_momentum_face(
        sum, g, flags, 1, background_view, i, j + 1,
        collocation_order, scheme_order, axial_cross, kz,
        decomposed_flux, append_static, Real{-1}, g.dy(),
        /*append_dynamic_metric=*/false, r, radial_tables);
  }

  if (high_order_dynamic) {
    append_dynamic_azimuthal_tensor(
        sum, g, i, j, scheme_order, collocation_order, gamma,
        state_view, background_active, background_view, r, radial_tables);
    if (append_static) {
      // -[r_hi M0_rr(hi)-r_lo M0_rr(lo)]/(r_c dr)
      // - d_z M0_zr + <M0_phiphi>_uniform/r_c.
      append_static_azimuthal_tensor(
          sum, g, i, j, scheme_order, background_view, r, radial_tables);
    }
  } else {
    MhdState centre{};
    if (mp_candidate) {
      centre = quasar::mhd::load_cell_state(
          g, rho, mr, mz, mphi, energy, br, bz, bphi, i, j,
          radial_tables, collocation_order);
    } else {
      // Preserve the order-1/2 API's historical read set: standalone
      // low-order callers need not initialize components absent from the
      // pressure-free tensor difference.
      centre.rho = rho[k];
      centre.mx = mr[k];
      centre.mz = mphi[k];
      centre.bx = quasar::mhd::cell_bx(
          g, br, i, j, radial_tables, collocation_order);
      centre.bz = bphi[k];
    }
    MhdBackground centre_background{};
    if (background_active != 0) {
      centre_background = quasar::mhd::load_cell_background(
          g, b0r, b0z, b0phi, i, j, radial_tables, collocation_order);
    }
    const Real br_pert = centre.bx;
    const Real bphi_pert = centre.bz;
    // Reduced tensor difference. Keep each quotient/product separate so
    // individually overflowing rotation and toroidal terms can cancel before
    // a much smaller radial-field survivor is rounded.
    append_radial_product_quotient(
        sum, mphi[k], mphi[k], rho[k], r);
    append_radial_product_quotient(
        sum, mr[k], -mr[k], rho[k], r);
    append_radial_product_quotient(
        sum, centre_background.b0x, br_pert, Real{0.5}, r);
    append_radial_product_quotient(
        sum, br_pert, br_pert, Real{1}, r);
    append_radial_product_quotient(
        sum, centre_background.b0z, -bphi_pert, Real{0.5}, r);
    append_radial_product_quotient(
        sum, bphi_pert, -bphi_pert, Real{1}, r);
    if (append_static && mp_candidate) {
      append_static_azimuthal_tensor(
          sum, g, i, j, scheme_order, background_view, r, radial_tables);
    } else if (append_static) {
      // Pressure-free order-1/2 tensor difference.
      append_radial_product_quotient(
          sum, centre_background.b0x, centre_background.b0x,
          Real{1}, r);
      append_radial_product_quotient(
          sum, centre_background.b0z, -centre_background.b0z,
          Real{1}, r);
    }
  }

}

__device__ __attribute__((noinline)) Real exact_radial_momentum_cell(
    const RadialMomentumKernelArgs& args, int i, int j) {
  ExactRadialMomentumAccumulator sum;
  accumulate_radial_momentum_cell(args, i, j, sum);
  return finish_exact_radial_momentum_sum(sum);
}

__global__ void __launch_bounds__(256, 2)
cylindrical_radial_momentum_fast_kernel(
    const RadialMomentumKernelArgs args) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= args.g.nx || j >= args.g.ny) return;

  FastRadialMomentumAccumulator sum;
  accumulate_radial_momentum_cell(args, i, j, sum);
  Real result{};
  if (!finish_fast_radial_momentum_sum(sum, result)) {
    result = std::numeric_limits<Real>::quiet_NaN();
  }
  args.radial_momentum_rate[args.g.index(i, j)] = result;
}

__global__ void __launch_bounds__(256, 2)
cylindrical_radial_momentum_exact_fallback_kernel(
    const RadialMomentumKernelArgs args) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= args.g.nx || j >= args.g.ny) return;

  const std::size_t k = args.g.index(i, j);
  if (!std::isnan(args.radial_momentum_rate[k])) return;
  args.radial_momentum_rate[k] = exact_radial_momentum_cell(args, i, j);
}

// Axisymmetric (r,z) geometric source for the m=0 cylindrical update.
//
// In the conventional annular formulation, the only terms left outside the
// radial face/volume-weighted momentum divergence are
//
//   S_{m_r} = [rho v_phi^2 + p + |B0+b|^2/2
//                                  -(B0_phi+b_phi)^2] / r.
//
// The expression below expands every static, cross, and perturbation product
// into one common-exponent sum. This standalone API is the full geometric
// source; the solver's flux-difference calls deliberately leave radial momentum
// untouched and its cancellation-safe fused residual instead uses the
// analytically pressure-free tensor-difference kernel above. Continuity, axial
// momentum, and energy have no pointwise source. Angular momentum is handled
// conservatively inside flux_difference_kernel with the exact int(r^2 dr) cell
// moment, so no pointwise m_phi source remains. Toroidal induction remains
// metric-free, while B_r/B_z are advanced by the matching annular CT curl.
//
// On-axis handling: the radius is read at the CELL CENTER, r = (i+0.5)*dr (for a
// domain that starts on the axis, origin_x = 0), so the innermost cell i=0 has
// r = 0.5*dr > 0 and the 1/r factor is finite (= 2/dr) -- the geometric source is
// NOT singular at any cell center and the i=0 column must receive it like every
// other column. Only a non-positive or non-finite radius (which cannot occur for
// a cell center on a valid grid, but is guarded defensively) is skipped to avoid
// a division by zero.
inline constexpr int kGeometricSourceTermsPerPoint = 11;
inline constexpr int kMp7GeometricNodesPerAxis = 4;
inline constexpr int kMp7GeometricPointCount =
    kMp7GeometricNodesPerAxis * kMp7GeometricNodesPerAxis;
inline constexpr int kHighOrderGeometricSourceCapacity = 176;
static_assert(kHighOrderGeometricSourceCapacity >=
              kGeometricSourceTermsPerPoint * kMp7GeometricPointCount);

template <bool Weighted, int Capacity>
__device__ inline void append_geometric_source_terms(
    quasar::numerics::ScaledProductQuotientAccumulator<Capacity>& source_sum,
    const MhdState& state, const MhdBackground& background, Real gamma,
    Real weight, Real radius) {
  if constexpr (!Weighted) (void)weight;
  const auto append = [&](Real a, Real b, Real d0, Real d1) {
    if constexpr (Weighted) {
      quasar::numerics::append_scaled_triple_product_quotient(
          source_sum, weight, a, b, d0, d1);
    } else {
      quasar::numerics::append_scaled_product_quotient(
          source_sum, a, b, d0, d1);
    }
  };
  const Real gas_pressure = pressure(state, background, gamma);
  append(state.mz, state.mz, state.rho, radius);
  append(gas_pressure, Real{1}, Real{1}, radius);
  append(background.b0x, state.bx, Real{1}, radius);
  append(background.b0y, state.by, Real{1}, radius);
  append(background.b0z, -state.bz, Real{1}, radius);
  append(background.b0x, Real{0.5} * background.b0x, Real{1}, radius);
  append(background.b0y, Real{0.5} * background.b0y, Real{1}, radius);
  append(background.b0z, Real{-0.5} * background.b0z, Real{1}, radius);
  append(state.bx, Real{0.5} * state.bx, Real{1}, radius);
  append(state.by, Real{0.5} * state.by, Real{1}, radius);
  append(state.bz, Real{-0.5} * state.bz, Real{1}, radius);
}

// Recover the complete equation-native state/background at every MP tensor
// point. If any point is inadmissible, return false so the caller retains the
// bounded cell-centred fallback. Otherwise append all eleven signed terms from
// every point directly to one cell-wide reduction. MP7 has 4*4 points, hence
// the proven 11*16 = 176 entry bound above; no point is rounded to an
// intermediate scalar before cancellation across the cell.
__device__ __attribute__((noinline)) bool try_high_order_geometric_source(
    quasar::Grid2D g, int i, int j, int b0_active, Real gamma,
    int collocation_order,
    quasar::numerics::RadialTablesView radial_tables,
    const double* rho, const double* mx, const double* my, const double* mz,
    const double* energy, const double* bx, const double* by, const double* bz,
    const double* b0x, const double* b0y, const double* b0z,
    Real radius, Real& source) {
  const int order = radial_tables.scheme_order;
  const int count = order == 7 ? 4 : 3;
  const quasar::mhd::detail::CellStateView state_view{
      rho, mx, my, mz, energy, bx, by, bz};

  for (int qy = 0; qy < count; ++qy) {
    for (int qx = 0; qx < count; ++qx) {
      const MhdState state = quasar::mhd::detail::point_cell_state(
          g, i, j, order, qx, qy, state_view, radial_tables,
          collocation_order);
      if (!finite_admissible_state(state, gamma)) return false;
      if (b0_active != 0 &&
          !finite_background(quasar::mhd::detail::point_cell_background(
              g, i, j, order, qx, qy, b0x, b0y, b0z,
              radial_tables))) {
        return false;
      }
    }
  }

  quasar::numerics::ScaledProductQuotientAccumulator<
      kHighOrderGeometricSourceCapacity> source_sum;
  for (int qy = 0; qy < count; ++qy) {
    for (int qx = 0; qx < count; ++qx) {
      const MhdState state = quasar::mhd::detail::point_cell_state(
          g, i, j, order, qx, qy, state_view, radial_tables,
          collocation_order);
      MhdBackground background{};
      if (b0_active != 0) {
        background = quasar::mhd::detail::point_cell_background(
            g, i, j, order, qx, qy, b0x, b0y, b0z, radial_tables);
      }
      append_geometric_source_terms<true>(
          source_sum, state, background, gamma,
          uniform_tensor_weight(order, qx, qy), radius);
    }
  }
  source = quasar::numerics::finish_scaled_product_quotient_sum(source_sum);
  return true;
}

template <bool RadialCollocation>
__global__ void QUASAR_LAUNCH_BOUNDS_256 geometric_source_kernel(
    quasar::Grid2D g, int b0_active, Real gamma, int collocation_order,
    quasar::numerics::RadialTablesView radial_tables,
    const double* __restrict__ rho, const double* __restrict__ mx,
    const double* __restrict__ my, const double* __restrict__ mz,
    const double* __restrict__ energy, const double* __restrict__ bx,
    const double* __restrict__ by, const double* __restrict__ bz,
    const double* __restrict__ b0x, const double* __restrict__ b0y,
    const double* __restrict__ b0z,
    double* __restrict__ drho, double* __restrict__ dmr,
    double* __restrict__ dmz_axial, double* __restrict__ dmphi,
    double* __restrict__ denergy) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const Real r = g.r_at_cell_center(i);
  if (!(r > Real{0})) return;  // defensive: skip only a non-positive cell-center radius
  const std::size_t k = g.index(i, j);
  const quasar::numerics::RadialTablesView tables = RadialCollocation
      ? radial_tables : quasar::numerics::RadialTablesView{};
  if constexpr (RadialCollocation) {
    const bool high_order = collocation_order == 0 &&
        tables.active != 0 &&
        (tables.scheme_order == 5 || tables.scheme_order == 7) &&
        tables.quadrature_nodes > 0;
    Real tensor_source{};
    if (high_order && try_high_order_geometric_source(
            g, i, j, b0_active, gamma, collocation_order, tables,
            rho, mx, my, mz, energy, bx, by, bz, b0x, b0y, b0z,
            r, tensor_source)) {
      dmr[k] += tensor_source;
      return;
    }
  }
  // Slot mapping: mx = m_r, my = m_z (axial), mz = m_phi (azimuthal);
  // bx = B_r, by = B_z, bz = B_phi.
  const Real br_pert = quasar::mhd::cell_bx(
      g, bx, i, j, tables, collocation_order);
  const Real bz_pert = quasar::mhd::cell_by(
      g, by, i, j, tables, collocation_order);
  const Real bphi_pert = bz[k];
  MhdBackground background{};
  if (b0_active != 0) {
    background = quasar::mhd::load_cell_background(
        g, b0x, b0y, b0z, i, j, tables, collocation_order);
  }
  MhdState s;
  s.rho = rho[k]; s.mx = mx[k]; s.my = my[k]; s.mz = mz[k];
  s.energy = energy[k]; s.bx = br_pert; s.by = bz_pert; s.bz = bphi_pert;
  // Fluid rotation/pressure and all total-field magnetic curvature pieces form
  // one physical source. Accumulate all eleven signed quotients before rounding
  // so dominant centrifugal/static/toroidal terms can cancel without erasing a
  // representable survivor.
  quasar::numerics::ScaledProductQuotientAccumulator<11> source_sum;
  append_geometric_source_terms<false>(
      source_sum, s, background, gamma, Real{1}, r);
  dmr[k] +=
      quasar::numerics::finish_scaled_product_quotient_sum(source_sum);

  (void)drho;
  (void)dmz_axial;
  (void)dmphi;
  (void)denergy;
  // S_{B_phi} = 0: toroidal induction is metric-free, handled by the Cartesian
  // flux difference; the poloidal faces B_r/B_z are advanced by CT.
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_cylindrical_radial_momentum_residual(
    const MhdField2D<Real>& u, const MhdBackgroundField<Real>& b0,
    const MhdField2D<Real>& flux_r, const MhdField2D<Real>& flux_z,
    MhdField2D<Real>& dudt, BoundaryFlags4 flags, stream_t stream,
    Real gamma, int collocation_order, int scheme_order,
    const MhdMomentumFluxParts2D<Real>* parts_r,
    const MhdMomentumFluxParts2D<Real>* parts_z,
    quasar::numerics::RadialTablesView radial_tables) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  const int decomposed_flux =
      (b0.active && parts_r != nullptr && parts_z != nullptr) ? 1 : 0;
  const auto point_view = [](const MhdMomentumFluxParts2D<Real>* parts) {
    CrossBPointView view;
    if (parts == nullptr) return view;
    for (int q = 0; q < 4; ++q) {
      view.x[q] = parts->cross_b_point[q].x.device_ptr();
      view.y[q] = parts->cross_b_point[q].y.device_ptr();
      view.z[q] = parts->cross_b_point[q].z.device_ptr();
    }
    view.quadrature_valid = parts->quadrature_valid.device_ptr();
    return view;
  };
  const CrossBPointView radial_cross = point_view(parts_r);
  const CrossBPointView axial_cross = point_view(parts_z);
  const RadialMomentumKernelArgs kernel_args{
      g, flags, b0.active ? 1 : 0,
      b0.globally_curl_free ? 1 : 0, decomposed_flux,
      collocation_order, scheme_order, gamma, radial_tables,
      u.rho.device_ptr(), u.mx.device_ptr(), u.my.device_ptr(),
      u.mz.device_ptr(), u.energy.device_ptr(),
      u.bx_face.device_ptr(), u.by_face.device_ptr(), u.bz_cell.device_ptr(),
      b0.active ? b0.b0x_face.device_ptr() : nullptr,
      b0.active ? b0.b0y_face.device_ptr() : nullptr,
      b0.active ? b0.b0z_cell.device_ptr() : nullptr,
      flux_r.mx.device_ptr(), flux_z.mx.device_ptr(),
      decomposed_flux != 0 ? parts_r->wave_x.device_ptr() : nullptr,
      decomposed_flux != 0 ? parts_z->wave_x.device_ptr() : nullptr,
      radial_cross, axial_cross, dudt.mx.device_ptr()};
  hipLaunchKernelGGL(
      cylindrical_radial_momentum_fast_kernel, grid, block, 0,
      static_cast<hipStream_t>(stream), kernel_args);
  QUASAR_HIP_CHECK(::hipGetLastError());
  hipLaunchKernelGGL(
      cylindrical_radial_momentum_exact_fallback_kernel, grid, block, 0,
      static_cast<hipStream_t>(stream), kernel_args);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_geometric_source(const MhdField2D<Real>& u, MhdField2D<Real>& dudt,
                                 const MhdBackgroundField<Real>& b0,
                                 Grid2D grid, Real gamma, stream_t stream,
                                 int collocation_order,
                                 quasar::numerics::RadialTablesView radial_tables) {
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 launch_grid = mhd_grid_2d(grid, block);
  const auto launch = [&](auto radial_tag) {
    constexpr bool kRadialCollocation = decltype(radial_tag)::value;
    const auto tables = kRadialCollocation
        ? radial_tables : quasar::numerics::RadialTablesView{};
    hipLaunchKernelGGL(
        (geometric_source_kernel<kRadialCollocation>), launch_grid, block, 0,
        static_cast<hipStream_t>(stream), grid, b0.active ? 1 : 0, gamma,
        collocation_order, tables, u.rho.device_ptr(), u.mx.device_ptr(),
        u.my.device_ptr(), u.mz.device_ptr(), u.energy.device_ptr(),
        u.bx_face.device_ptr(), u.by_face.device_ptr(), u.bz_cell.device_ptr(),
        b0.active ? b0.b0x_face.device_ptr() : nullptr,
        b0.active ? b0.b0y_face.device_ptr() : nullptr,
        b0.active ? b0.b0z_cell.device_ptr() : nullptr,
        dudt.rho.device_ptr(), dudt.mx.device_ptr(), dudt.my.device_ptr(),
        dudt.mz.device_ptr(), dudt.energy.device_ptr());
  };
  if (radial_tables.active != 0) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
