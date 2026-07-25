#pragma once

// Ideal-MHD state, primitive, and flux types plus the host/device-callable
// equation-of-state helpers shared by the MHD numerics axis (reconstruction,
// Riemann solver, eigensystem). All math here is `inline QUASAR_HOST_DEVICE`
// so kernels can call it directly; src/numerics/mhd_state.cpp provides the
// translation-unit anchor for the HIP-tagged module object.

#include "quasar/core/types.hpp"

#include <cmath>
#include <limits>

namespace quasar::numerics {

// Keep the variable-length exponent reducers as device call boundaries.  When
// HIP inlines their runtime-indexed local arrays into a large HLLD caller, LLVM
// can duplicate those arrays at every algebraic branch and promote them to LDS.
// A direct device call leaves the small workspaces in private storage once and
// preserves exactly the same reduction order.  Ordinary host builds remain
// header-inline.
#if defined(__HIP_DEVICE_COMPILE__)
#define QUASAR_MHD_NUMERICS_NOINLINE __attribute__((noinline))
#else
#define QUASAR_MHD_NUMERICS_NOINLINE
#endif

// Conserved-variable storage. Component order is the canonical 8-tuple
// (rho, mx, my, mz, energy, bx, by, bz) used everywhere downstream. In an
// unsplit state `energy` is the ordinary total ideal-MHD energy. In the active
// field split B=B0+b, the same slot stores
//   E' = rho*e + |m|^2/(2*rho) + |b|^2/2,
// while bx/by/bz store b; B0 is carried separately by MhdBackground.
struct MhdState {
  Real rho{};
  Real mx{};
  Real my{};
  Real mz{};
  Real energy{};
  Real bx{};
  Real by{};
  Real bz{};
};

// Primitive variables (velocity + gas pressure).
struct MhdPrim {
  Real rho{};
  Real vx{};
  Real vy{};
  Real vz{};
  Real p{};
  Real bx{};
  Real by{};
  Real bz{};
};

// Physical flux in the canonical conserved-variable order.
struct MhdFlux {
  Real rho{};
  Real mx{};
  Real my{};
  Real mz{};
  Real energy{};
  Real bx{};
  Real by{};
  Real bz{};
};

// Static background magnetic field B0 for the field-split formulation
// B = B0 + b. The stored conserved state carries the perturbation b in
// u.bx/u.by/u.bz and the perturbation-only magnetic energy 0.5|b|^2; this POD
// carries the (spatially sampled) constant background components passed to the
// background-aware EOS overloads below. It is a trivial value type so it can be
// passed by value into HIP device code.
struct MhdBackground {
  Real b0x{};
  Real b0y{};
  Real b0z{};
};

// Reduce terms already represented as mantissa*2^exponent.  Always pair the
// largest term with the largest (therefore closest-magnitude) opposite-sign
// term when one exists.  This preserves a small survivor in expressions such
// as A+B-A-B+c: combining A+B first can round once at O(A) and leave an ulp(A)
// residue that overwhelms c even though the exact large terms cancel.
QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline Real reduce_scaled_terms(
    Real* mantissa, int* exponent, int count) {
  auto larger_magnitude = [&](int lhs, int rhs) {
    return exponent[lhs] > exponent[rhs] ||
           (exponent[lhs] == exponent[rhs] &&
            std::fabs(mantissa[lhs]) > std::fabs(mantissa[rhs]));
  };

  while (true) {
    int largest = -1;
    for (int k = 0; k < count; ++k) {
      if (mantissa[k] == Real{0}) continue;
      if (largest < 0 || larger_magnitude(k, largest)) largest = k;
    }
    if (largest < 0) return Real{0};

    int opposite = -1;
    int same_sign = -1;
    const bool largest_negative = mantissa[largest] < Real{0};
    for (int k = 0; k < count; ++k) {
      if (k == largest || mantissa[k] == Real{0}) continue;
      const bool opposite_sign =
          (mantissa[k] < Real{0}) != largest_negative;
      int& candidate = opposite_sign ? opposite : same_sign;
      if (candidate < 0 || larger_magnitude(k, candidate)) candidate = k;
    }

    const int partner = (opposite >= 0) ? opposite : same_sign;
    if (partner < 0) {
      return scalbn(mantissa[largest], exponent[largest]);
    }

    const int common = exponent[largest];
    const Real sum = mantissa[largest] +
        scalbn(mantissa[partner], exponent[partner] - common);
    mantissa[partner] = Real{0};
    if (sum == Real{0}) {
      mantissa[largest] = Real{0};
      continue;
    }
    int shift = 0;
    mantissa[largest] = frexp(sum, &shift);
    exponent[largest] = common + shift;
  }
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
// The six-term bound is deliberate: all current MHD uses fit in fixed-size
// thread-local storage (the cylindrical div(B) stencil is the largest, with six
// signed terms), avoiding dynamic storage in device code.
template <int MaxTerms>
struct ScaledProductQuotientAccumulator {
  Real mantissa[MaxTerms]{};
  int exponent[MaxTerms]{};
  Real nonfinite_sum{};
  int count{};
  bool has_nonfinite{};
};

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

template <int MaxTerms>
QUASAR_HOST_DEVICE QUASAR_MHD_NUMERICS_NOINLINE inline Real
finish_scaled_product_quotient_sum(
    ScaledProductQuotientAccumulator<MaxTerms>& sum) {
  const Real finite_sum =
      reduce_scaled_terms(sum.mantissa, sum.exponent, sum.count);
  return sum.has_nonfinite ? sum.nonfinite_sum + finite_sum : finite_sum;
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

// Extended reduction used only where a cylindrical background-stress/source
// cancellation spans more than six signed terms.  Twenty terms cover the fused
// radial split-stress operator (currently 17) with room for its paired metric
// corrections, while keeping the hotter ordinary stencil path at six terms.
QUASAR_HOST_DEVICE inline Real scaled_product_quotient_sum_extended(
    const Real* a, const Real* b, const Real* d0, const Real* d1,
    int count) {
  return scaled_product_quotient_sum_impl<20>(a, b, d0, d1, count);
}

// Scale-safe Euclidean and quadratic forms used throughout the MHD numerics.
// Forming x*x + y*y + z*z directly can overflow (or underflow) even when the
// quantity ultimately consumed by the algorithm is representable.  Scaling by
// the largest component keeps the dimensionless sum in [1,3].
QUASAR_HOST_DEVICE inline Real scaled_norm3(Real x, Real y, Real z) {
  const Real scale = std::fmax(std::fabs(x), std::fmax(std::fabs(y), std::fabs(z)));
  if (scale == Real{0}) return Real{0};
  const Real xs = x / scale;
  const Real ys = y / scale;
  const Real zs = z / scale;
  return scale * std::sqrt(xs * xs + ys * ys + zs * zs);
}

// 0.5*(x^2+y^2+z^2), evaluated without first forming an overflowing square.
// The half factor is applied before the first product so every mathematically
// representable half-norm remains representable during the calculation.
QUASAR_HOST_DEVICE inline Real half_squared_norm3(Real x, Real y, Real z) {
  const Real scale = std::fmax(std::fabs(x), std::fmax(std::fabs(y), std::fabs(z)));
  if (scale == Real{0}) return Real{0};
  const Real xs = x / scale;
  const Real ys = y / scale;
  const Real zs = z / scale;
  const Real normalized_square = xs * xs + ys * ys + zs * zs;
  return (Real{0.5} * scale) * (scale * normalized_square);
}

// K = |m|^2/(2 rho).  Dividing the scaled momentum by sqrt(rho) before
// squaring avoids both m*m overflow and 1/rho overflow for tiny densities.
QUASAR_HOST_DEVICE inline Real kinetic_from_momentum(Real mx, Real my, Real mz,
                                                     Real rho) {
  const Real scale = std::fmax(std::fabs(mx), std::fmax(std::fabs(my), std::fabs(mz)));
  if (scale == Real{0}) return Real{0};
  const Real xs = mx / scale;
  const Real ys = my / scale;
  const Real zs = mz / scale;
  const Real q = scale / std::sqrt(rho);
  const Real normalized_square = xs * xs + ys * ys + zs * zs;
  return (Real{0.5} * q) * (q * normalized_square);
}

// K = rho |v|^2/2, paired with kinetic_from_momentum for primitive input.
QUASAR_HOST_DEVICE inline Real kinetic_from_velocity(Real rho, Real vx, Real vy,
                                                     Real vz) {
  const Real scale = std::fmax(std::fabs(vx), std::fmax(std::fabs(vy), std::fabs(vz)));
  if (scale == Real{0}) return Real{0};
  const Real xs = vx / scale;
  const Real ys = vy / scale;
  const Real zs = vz / scale;
  const Real q = std::sqrt(rho) * scale;
  const Real normalized_square = xs * xs + ys * ys + zs * zs;
  return (Real{0.5} * q) * (q * normalized_square);
}

// Fast/slow magnetosonic roots from primitive thermodynamic quantities.  The
// quadratic is solved after normalizing all characteristic speeds by their
// maximum, so neither (a^2+v_A^2)^2 nor B^2 is ever formed at physical scale.
// The slow root uses the product relation cf^2 cs^2 = a^2 c_an^2, avoiding the
// catastrophic subtraction 0.5*(sum-sqrt(discriminant)).
QUASAR_HOST_DEVICE inline void magnetosonic_speeds(
    Real rho, Real p, Real bx, Real by, Real bz, int dir, Real gamma,
    Real& fast, Real& slow, Real& sound, Real& alfven_normal,
    Real& alfven_total) {
  const Real sqrt_rho = std::sqrt(rho);
  sound = (p > Real{0} && gamma > Real{0})
      ? std::sqrt(gamma) * (std::sqrt(p) / sqrt_rho)
      : Real{0};

  const Real ax = bx / sqrt_rho;
  const Real ay = by / sqrt_rho;
  const Real az = bz / sqrt_rho;
  alfven_total = scaled_norm3(ax, ay, az);
  alfven_normal = std::fabs((dir == 0) ? ax : ay);

  const Real scale = std::fmax(sound, alfven_total);
  if (!(scale > Real{0})) {
    fast = Real{0};
    slow = Real{0};
    return;
  }

  const Real an = sound / scale;
  const Real bn = alfven_total / scale;
  const Real can = alfven_normal / scale;
  const Real a2n = an * an;
  const Real b2n = bn * bn;
  const Real can2 = can * can;
  const Real sum = a2n + b2n;
  Real disc = sum * sum - Real{4} * a2n * can2;
  if (disc < Real{0}) disc = Real{0};
  const Real fast2n = Real{0.5} * (sum + std::sqrt(disc));
  const Real slow2n = (fast2n > Real{0})
      ? (a2n * can2) / fast2n
      : Real{0};
  fast = scale * std::sqrt(fast2n);
  slow = scale * std::sqrt(slow2n);
}

// p = (gamma-1)*(E - 0.5*rho|v|^2 - 0.5|B|^2). Total energy convention.
QUASAR_HOST_DEVICE inline Real pressure(const MhdState& u, Real gamma) {
  const Real kinetic = kinetic_from_momentum(u.mx, u.my, u.mz, u.rho);
  const Real magnetic = half_squared_norm3(u.bx, u.by, u.bz);
  // Subtract sequentially: kinetic+magnetic may overflow even though the
  // diagnostically useful (negative) pressure residual is representable.
  return (gamma - Real{1}) * ((u.energy - kinetic) - magnetic);
}

// Field-split gas pressure. By construction the stored E' carries only the
// perturbation magnetic energy 0.5|b|^2 (b = u.bx/by/bz), so the gas pressure
// has NO B0 cross term and does not depend on the background at all:
//   p = (gamma-1)*(E - 0.5*rho|v|^2 - 0.5|b|^2).
// The MhdBackground argument exists only to give the EOS a uniform call surface
// shared with fast_magnetosonic_speed(); it is intentionally unread here so that
// pressure(u, {0,0,0}, gamma) is bit-for-bit identical to pressure(u, gamma).
QUASAR_HOST_DEVICE inline Real pressure(const MhdState& u, const MhdBackground& /*b0*/,
                                        Real gamma) {
  return pressure(u, gamma);
}

QUASAR_HOST_DEVICE inline bool background_is_zero(const MhdBackground& b0) {
  return b0.b0x == Real{0} && b0.b0y == Real{0} && b0.b0z == Real{0};
}

// Total pressure with the pure-background magnetic pressure removed:
//
//   q = p + |B0+b|^2/2 - |B0|^2/2
//     = p + B0.b + |b|^2/2.
//
// q is the pressure variable that appears in the cancellation-free field-split
// Riemann problem.  In particular, it contains no O(|B0|^2) quantity.  Evaluate
// all seven signed products in one exponent-scaled accumulation so opposing
// B0.b components can cancel before a finite gas-pressure remainder is rounded
// away (or before either cross product overflows).
QUASAR_HOST_DEVICE inline Real split_total_pressure(const MhdState& u,
                                                    const MhdBackground& b0,
                                                    Real gamma) {
  const Real p = pressure(u, gamma);
  const Real a[7] = {p, b0.b0x, b0.b0y, b0.b0z, u.bx, u.by, u.bz};
  const Real b[7] = {Real{1}, u.bx, u.by, u.bz,
                     Real{0.5} * u.bx, Real{0.5} * u.by,
                     Real{0.5} * u.bz};
  return scaled_product_sum(a, b, 7);
}

// Conserved -> primitive. v = m/rho; p from the gamma law above. Magnetic
// components pass through unchanged. Exact inverse of to_conserved().
QUASAR_HOST_DEVICE inline MhdPrim to_primitive(const MhdState& u, Real gamma) {
  MhdPrim w;
  w.rho = u.rho;
  w.vx  = u.mx / u.rho;
  w.vy  = u.my / u.rho;
  w.vz  = u.mz / u.rho;
  w.bx  = u.bx;
  w.by  = u.by;
  w.bz  = u.bz;
  w.p = pressure(u, gamma);
  return w;
}

// Primitive -> conserved. m = rho*v; E = p/(gamma-1) + 0.5*rho|v|^2 + 0.5|B|^2.
// Exact inverse of to_primitive().
QUASAR_HOST_DEVICE inline MhdState to_conserved(const MhdPrim& w, Real gamma) {
  MhdState u;
  u.rho = w.rho;
  u.mx  = w.rho * w.vx;
  u.my  = w.rho * w.vy;
  u.mz  = w.rho * w.vz;
  u.bx  = w.bx;
  u.by  = w.by;
  u.bz  = w.bz;
  const Real kinetic  = kinetic_from_velocity(w.rho, w.vx, w.vy, w.vz);
  const Real magnetic = half_squared_norm3(w.bx, w.by, w.bz);
  u.energy = w.p / (gamma - Real{1}) + kinetic + magnetic;
  return u;
}

// Fast magnetosonic speed along the normal `dir` (0=x, 1=y).
//   a^2  = gamma*p/rho                       (sound speed squared)
//   ca^2 = |B|^2/rho                         (total Alfven speed squared)
//   cax  = B_dir/sqrt(rho)                   (directional Alfven speed)
//   c_f^2 = 0.5*[ (a^2+ca^2) + sqrt((a^2+ca^2)^2 - 4 a^2 cax^2) ]
// c_f >= a, c_f >= |cax|, and reduces to a when B=0.
//
// Field-split overload: the wave speeds see the TOTAL magnetic field
// B = b + B0, where b = u.bx/by/bz is the stored perturbation and B0 is the
// background. The gas pressure p still comes from the perturbation-only EOS
// above (no B0 cross term). With b0 == {0,0,0} the total field equals b and this
// is bit-for-bit identical to the zero-background overload below.
QUASAR_HOST_DEVICE inline Real fast_magnetosonic_speed(const MhdState& u,
                                                       const MhdBackground& b0,
                                                       int dir, Real gamma) {
  const Real p  = pressure(u, b0, gamma);
  const Real Bx = u.bx + b0.b0x;
  const Real By = u.by + b0.b0y;
  const Real Bz = u.bz + b0.b0z;
  Real fast, slow, sound, alfven_normal, alfven_total;
  magnetosonic_speeds(u.rho, p, Bx, By, Bz, dir, gamma,
                      fast, slow, sound, alfven_normal, alfven_total);
  (void)slow;
  (void)sound;
  (void)alfven_normal;
  (void)alfven_total;
  return fast;
}

// Zero-background overload. Delegates to the field-split overload with
// B0 == {0,0,0}: the total field reduces to b = u.bx/by/bz and p reduces to
// pressure(u, gamma), so the result is unchanged from the original body.
QUASAR_HOST_DEVICE inline Real fast_magnetosonic_speed(const MhdState& u, int dir, Real gamma) {
  return fast_magnetosonic_speed(u, MhdBackground{}, dir, gamma);
}

// Translation-unit anchor (defined in src/numerics/mhd_state.cpp) so the
// HIP-tagged module object is non-trivial even though all math is header-inline.
const char* mhd_state_version() noexcept;

#undef QUASAR_MHD_NUMERICS_NOINLINE

}  // namespace quasar::numerics
