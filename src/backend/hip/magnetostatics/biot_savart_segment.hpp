#pragma once

// Per-segment Biot-Savart device kernel. Included only from .hip translation
// units; guarded so accidental inclusion from plain C++ TUs is a no-op.
#ifdef __HIPCC__

#include <hip/hip_runtime.h>

#include "quasar/core/types.hpp"

#include <limits>

namespace quasar::magnetostatics::detail {

// A scaled Euclidean norm avoids overflow/underflow in dot(v,v). This matters
// for the fp32 evaluator: segment and observation vectors are normalized by
// their respective component scales, and valid SI geometries span many decades.
template <class T>
__device__ __forceinline__ T stable_length(Vec3T<T> v) {
  const T ax = fabs(v.x);
  const T ay = fabs(v.y);
  const T az = fabs(v.z);
  const T scale = fmax(ax, fmax(ay, az));
  if (scale == T{0}) return T{0};
  const Vec3T<T> w = v / scale;
  return scale * sqrt(dot(w, w));
}

// Compare a*b and c*d as exact real products of the input floating-point
// values.  Direct products can overflow, underflow, or round two unequal
// values together.  frexp first moves every nonzero operand into a safe
// mantissa range; fma then exposes the exact product round-off, so equality of
// both the rounded product and its residual is equality of the real products.
// This is the small exact predicate needed by the filament-collinearity test.
template <class T>
__device__ __forceinline__ bool exact_product_equal(T a, T b, T c, T d) {
  const bool left_zero = a == T{0} || b == T{0};
  const bool right_zero = c == T{0} || d == T{0};
  if (left_zero || right_zero) return left_zero && right_zero;

  int ea = 0, eb = 0, ec = 0, ed = 0;
  T ma = frexp(a, &ea);
  const T mb = frexp(b, &eb);
  T mc = frexp(c, &ec);
  const T md = frexp(d, &ed);
  const int left_exponent = ea + eb;
  const int right_exponent = ec + ed;
  const int difference = left_exponent - right_exponent;
  // Products of two normalized mantissas have magnitudes in [1/4, 1), so
  // exponents separated by two or more cannot denote the same nonzero value.
  if (difference >= 2 || difference <= -2) return false;
  if (difference > 0) {
    ma = scalbn(ma, difference);
  } else if (difference < 0) {
    mc = scalbn(mc, -difference);
  }

  const T left = ma * mb;
  const T right = mc * md;
  if (left != right) return false;
  return fma(ma, mb, -left) == fma(mc, md, -right);
}

template <class T>
struct ScaledNumber {
  T mantissa{};
  int exponent{};
};

// Keep per-segment results exponent-scaled until the reduction has combined
// every source.  Materializing each segment first is not a linear operation at
// the ends of the floating-point range: two half-subnormal contributions can
// add to a representable value, and individually overflowing contributions can
// cancel.  These small internal vector/matrix types carry the scaled values
// across the segment/reduction seam without changing the public kernel ABI.
template <class T>
struct ScaledVec3 {
  ScaledNumber<T> x{}, y{}, z{};
};

template <class T>
struct ScaledMat3x3 {
  ScaledVec3<T> r0{}, r1{}, r2{};
};

template <class T>
__device__ __forceinline__ ScaledNumber<T>
normalize_scaled(T mantissa, int exponent) {
  if (mantissa == T{0}) return {};
  int adjustment = 0;
  mantissa = frexp(mantissa, &adjustment);
  return ScaledNumber<T>{mantissa, exponent + adjustment};
}

template <class T>
__device__ __forceinline__ ScaledNumber<T> scaled_from(T value) {
  if (value == T{0}) return {};
  int exponent = 0;
  return ScaledNumber<T>{frexp(value, &exponent), exponent};
}

template <class T>
__device__ __forceinline__ ScaledNumber<T>
scaled_multiply(ScaledNumber<T> a, T b) {
  if (a.mantissa == T{0} || b == T{0}) return {};
  int exponent = 0;
  const T mantissa = frexp(b, &exponent);
  return normalize_scaled(a.mantissa * mantissa, a.exponent + exponent);
}

template <class T>
__device__ __forceinline__ ScaledNumber<T>
scaled_multiply(ScaledNumber<T> a, ScaledNumber<T> b) {
  if (a.mantissa == T{0} || b.mantissa == T{0}) return {};
  return normalize_scaled(a.mantissa * b.mantissa, a.exponent + b.exponent);
}

template <class T>
__device__ __forceinline__ ScaledNumber<T>
scaled_divide(ScaledNumber<T> a, T b) {
  if (a.mantissa == T{0}) return {};
  int exponent = 0;
  const T mantissa = frexp(b, &exponent);
  return normalize_scaled(a.mantissa / mantissa, a.exponent - exponent);
}

template <class T>
__device__ __forceinline__ ScaledNumber<T>
scaled_divide(ScaledNumber<T> a, ScaledNumber<T> b) {
  if (a.mantissa == T{0}) return {};
  return normalize_scaled(a.mantissa / b.mantissa,
                          a.exponent - b.exponent);
}

template <class T>
__device__ __forceinline__ ScaledNumber<T>
scaled_negate(ScaledNumber<T> value) {
  value.mantissa = -value.mantissa;
  return value;
}

template <class T>
__device__ __forceinline__ ScaledNumber<T>
scaled_pair_sum(ScaledNumber<T> a, ScaledNumber<T> b) {
  if (a.mantissa == T{0}) return b;
  if (b.mantissa == T{0}) return a;
  const int common_exponent = a.exponent > b.exponent
      ? a.exponent : b.exponent;
  const int a_shift = a.exponent - common_exponent;
  const int b_shift = b.exponent - common_exponent;
  // If the smaller term cannot overlap the larger significand, it cannot
  // change the rounded pair unless the larger later cancels. Callers combine
  // the two largest exponents first, so any such cancellation is handled
  // before a still smaller band is considered.
  if (a_shift < -std::numeric_limits<T>::digits - 2) return b;
  if (b_shift < -std::numeric_limits<T>::digits - 2) return a;
  return normalize_scaled(
      scalbn(a.mantissa, a_shift) + scalbn(b.mantissa, b_shift),
      common_exponent);
}

template <class T, int Capacity>
__device__ __forceinline__ ScaledNumber<T>
scaled_largest_pair_sum(ScaledNumber<T> (&input)[Capacity], int count) {
  while (count > 1) {
    int first = 0;
    for (int i = 1; i < count; ++i) {
      if (input[i].exponent > input[first].exponent) first = i;
    }
    int second = first == 0 ? 1 : 0;
    for (int i = 0; i < count; ++i) {
      if (i != first && input[i].exponent > input[second].exponent) second = i;
    }
    const ScaledNumber<T> sum = scaled_pair_sum(input[first], input[second]);
    ScaledNumber<T> remaining[Capacity]{};
    int next_count = 0;
    for (int i = 0; i < count; ++i) {
      if (i != first && i != second) remaining[next_count++] = input[i];
    }
    if (sum.mantissa != T{0}) remaining[next_count++] = sum;
    for (int i = 0; i < next_count; ++i) input[i] = remaining[i];
    count = next_count;
  }
  return count == 0 ? ScaledNumber<T>{} : input[0];
}

template <class T>
__device__ __forceinline__ T
materialize_scaled(ScaledNumber<T> value, bool& numeric_failure) {
  if (value.mantissa == T{0}) return T{0};
  const T result = scalbn(value.mantissa, value.exponent);
  if (!isfinite(result)) numeric_failure = true;
  return result;
}

// Return log(1+q) without first materializing q.  This matters in the far field:
// q is proportional to segment_length / observation_distance and may be below
// the smallest T even though multiplication by a very large current brings the
// final vector potential back into range.
template <class T>
__device__ __forceinline__ ScaledNumber<T>
scaled_log1p(ScaledNumber<T> q) {
  if (q.mantissa == T{0}) return {};
  if (q.exponent <= -1) {
    const T value = scalbn(q.mantissa, q.exponent);
    const T correction = value == T{0} ? T{1} : log1p(value) / value;
    return scaled_multiply(q, correction);
  }
  const T value = scalbn(q.mantissa, q.exponent);
  if (isfinite(value)) return scaled_from(log1p(value));

  // q itself exceeds the working range, but log(1+q) does not: its magnitude is
  // only O(exponent).  The omitted log1p(1/q) correction is below one ulp here.
  const T log_q = log(fabs(q.mantissa))
                + static_cast<T>(q.exponent) * log(T{2});
  return scaled_from(log_q);
}

// Segment-local geometry normalized by a common physical distance scale. This
// keeps the field Jacobian free of the length^8 denominator-squared intermediate
// in the traditional quotient-rule expression without ever forming
// distance/length, which itself may lie outside T's exponent range. `gamma` is
// the cancellation-prone factor |x||y| + x.y. For points whose endpoint vectors
// oppose each other, use the exact identity
//
//   gamma = |x cross y|^2 / (|x||y| - x.y)
//         = rho^2 / (alpha - beta),
//
// where rho is the perpendicular displacement from the segment line. This
// remains accurate arbitrarily close to, but off, a filament.
template <class T>
struct SegmentGeometry {
  Vec3T<T> t{};             // unit tangent from a to b
  Vec3T<T> x{};             // (p-a) / distance_scale
  Vec3T<T> y{};             // (p-b) / distance_scale
  Vec3T<T> rho{};           // perpendicular component of x
  Vec3T<T> unit_a{};        // (p-a) / |p-a|, independently scaled
  Vec3T<T> unit_b{};        // (p-b) / |p-b|, independently scaled
  ScaledNumber<T> length{}; // |b-a|, kept outside T's exponent range
  ScaledNumber<T> distance_a{};
  ScaledNumber<T> distance_b{};
  ScaledNumber<T> length_over_distance{};
  T distance_scale{};       // physical scale removed from endpoint vectors
  T A{};                    // |scaled x|
  T B{};                    // |scaled y|
  T alpha{};                // A*B in scaled coordinates
  T beta{};                 // x.y in scaled coordinates
  ScaledNumber<T> gamma{};  // alpha+beta in scaled coordinates
  bool collinear{};         // exact line test, including exterior points
};

template <class T>
__device__ __forceinline__ bool endpoint_direction(
    Vec3T<T> direct, Vec3T<T> point, Vec3T<T> endpoint,
    Vec3T<T>& unit, ScaledNumber<T>& distance) {
  T scale = fmax(fabs(direct.x), fmax(fabs(direct.y), fabs(direct.z)));
  Vec3T<T> normalized = direct;
  if (!(scale > T{0}) || !isfinite(scale)
      || !isfinite(direct.x) || !isfinite(direct.y) || !isfinite(direct.z)) {
    scale = fmax(
        fmax(fabs(point.x), fmax(fabs(point.y), fabs(point.z))),
        fmax(fabs(endpoint.x), fmax(fabs(endpoint.y), fabs(endpoint.z))));
    if (!(scale > T{0}) || !isfinite(scale)) return false;
    normalized = point / scale - endpoint / scale;
  } else {
    normalized = direct / scale;
  }
  const T norm = stable_length(normalized);
  if (!(norm > T{0}) || !isfinite(norm)) return false;
  unit = normalized / norm;
  distance = scaled_multiply(scaled_from(scale), norm);
  return true;
}

template <class T>
__device__ __forceinline__ SegmentGeometry<T>
segment_geometry(Vec3T<T> a, Vec3T<T> b, Vec3T<T> p,
                 bool& singular, bool& numeric_failure) {
  SegmentGeometry<T> g;
  const Vec3T<T> L = b - a;
  const T segment_scale = fmax(
      fabs(L.x), fmax(fabs(L.y), fabs(L.z)));
  if (!(segment_scale > T{0}) || !isfinite(segment_scale)) {
    numeric_failure = true;  // unresolved segment in this working precision
    return g;
  }
  const Vec3T<T> scaled_L = L / segment_scale;
  const T scaled_length = stable_length(scaled_L);
  if (!(scaled_length > T{0}) || !isfinite(scaled_length)) {
    numeric_failure = true;
    return g;
  }
  g.t = scaled_L / scaled_length;
  g.length = scaled_multiply(scaled_from(segment_scale), scaled_length);

  Vec3T<T> ra = p - a;
  Vec3T<T> rb = p - b;
  if (!endpoint_direction(ra, p, a, g.unit_a, g.distance_a)
      || !endpoint_direction(rb, p, b, g.unit_b, g.distance_b)) {
    // Exact endpoints are classified as physical singularities below. Other
    // failures mean the endpoint displacement was lost in this precision.
    const bool at_a = p.x == a.x && p.y == a.y && p.z == a.z;
    const bool at_b = p.x == b.x && p.y == b.y && p.z == b.z;
    if (at_a || at_b) singular = true;
    else numeric_failure = true;
    return g;
  }
  const bool direct_displacements =
      isfinite(ra.x) && isfinite(ra.y) && isfinite(ra.z)
      && isfinite(rb.x) && isfinite(rb.y) && isfinite(rb.z);
  if (direct_displacements) {
    g.distance_scale = fmax(
        fmax(fabs(ra.x), fmax(fabs(ra.y), fabs(ra.z))),
        fmax(fabs(rb.x), fmax(fabs(rb.y), fabs(rb.z))));
    if (g.distance_scale > T{0} && isfinite(g.distance_scale)) {
      g.x = ra / g.distance_scale;
      g.y = rb / g.distance_scale;
    }
  } else {
    // Opposite-sign finite coordinates can overflow on subtraction.  Scale the
    // coordinates first in just that case; direct subtraction above is retained
    // for nearby large coordinates so their small separation is not rounded away.
    g.distance_scale = fmax(
        fmax(fabs(p.x), fmax(fabs(p.y), fabs(p.z))),
        fmax(fmax(fabs(a.x), fmax(fabs(a.y), fabs(a.z))),
             fmax(fabs(b.x), fmax(fabs(b.y), fabs(b.z)))));
    if (g.distance_scale > T{0} && isfinite(g.distance_scale)) {
      g.x = p / g.distance_scale - a / g.distance_scale;
      g.y = p / g.distance_scale - b / g.distance_scale;
    }
  }
  if (!(g.distance_scale > T{0}) || !isfinite(g.distance_scale)) {
    numeric_failure = true;
    return g;
  }
  g.length_over_distance = scaled_divide(g.length, g.distance_scale);

  const T z_scaled = dot(g.x, g.t);
  g.rho = g.x - z_scaled * g.t;
  const T rho_mag = stable_length(g.rho);

  // Projection onto a normalized diagonal can leave a round-off-sized rho even
  // for an exactly representable point on the segment.  Classify collinearity
  // independently by reconstructing p from the dominant segment component with
  // fused multiply-add.  Exact equality deliberately supplies no distance
  // tolerance: a representably distinct near-line point must remain finite.
  int dominant = 0;
  if (fabs(L.y) > fabs(L.x)) dominant = 1;
  if (fabs(L.z) > fabs(dominant == 0 ? L.x : L.y)) dominant = 2;
  const T l_component = dominant == 0 ? L.x : dominant == 1 ? L.y : L.z;
  const T p_component = dominant == 0 ? p.x : dominant == 1 ? p.y : p.z;
  const T a_component = dominant == 0 ? a.x : dominant == 1 ? a.y : a.z;
  T displacement = p_component - a_component;
  T parameter = displacement / l_component;
  if (!isfinite(displacement) || !isfinite(parameter)) {
    parameter = (p_component / g.distance_scale
                 - a_component / g.distance_scale)
              / (l_component / g.distance_scale);
  }
  const bool at_a = p.x == a.x && p.y == a.y && p.z == a.z;
  const bool at_b = p.x == b.x && p.y == b.y && p.z == b.z;
  // When the endpoint displacements are representable, test exact real
  // collinearity rather than reconstructing p with a rounded parameter.  The
  // latter misses genuine points such as (27,63) on (0,0)->(33,77), because
  // 9/11 is not representable and fma((9/11),77,0) rounds above 63.
  bool on_line = false;
  if (direct_displacements) {
    on_line = exact_product_equal(L.x, displacement, l_component, ra.x)
           && exact_product_equal(L.y, displacement, l_component, ra.y)
           && exact_product_equal(L.z, displacement, l_component, ra.z);
  } else {
    // A point inside a validated segment always has representable endpoint
    // displacements.  This fallback is therefore only needed to recognize
    // collinear points beyond an endpoint whose subtraction overflowed, so the
    // finite outside-filament field is not mislabeled as unresolved geometry.
    on_line = isfinite(parameter)
        && fma(parameter, L.x, a.x) == p.x
        && fma(parameter, L.y, a.y) == p.y
        && fma(parameter, L.z, a.z) == p.z;
  }
  g.collinear = on_line;
  // Once exact collinearity is known, decide segment membership without the
  // rounded projection quotient.  Immediately outside an endpoint, forming
  // (p-a)/(b-a) can round to exactly 0 or 1 (for example p=nextafter(b,+inf)
  // on [-1,1]) and falsely report a filament collision.  The dominant segment
  // coordinate is strictly monotone, so ordered endpoint comparisons classify
  // the closed segment exactly and work for either endpoint orientation.
  const T b_component = dominant == 0 ? b.x : dominant == 1 ? b.y : b.z;
  const bool within_dominant_extent =
      (a_component <= p_component && p_component <= b_component)
      || (b_component <= p_component && p_component <= a_component);
  if (at_a || at_b
      || (on_line && within_dominant_extent)) {
    singular = true;
    return g;
  }

  if (rho_mag == T{0}) {
    // A nonzero transverse component disappeared while scaling.  The exact
    // reconstruction above ruled out a filament collision, so report an
    // unresolved numerical geometry rather than a physical singularity.
    if (!on_line) {
      numeric_failure = true;
      return g;
    }
  }

  g.A = stable_length(g.x);
  g.B = stable_length(g.y);
  g.alpha = g.A * g.B;
  g.beta = dot(g.x, g.y);

  if (g.beta < T{0}) {
    const T delta = g.alpha - g.beta;
    // |x cross y| = (length/distance_scale)*|rho|.  Keep its square in scaled
    // form so a resolvable off-filament displacement is not mistaken for the
    // exact singularity merely because the squared quantity underflows.
    const ScaledNumber<T> cross_mag =
        scaled_multiply(g.length_over_distance, rho_mag);
    g.gamma = scaled_divide(scaled_multiply(cross_mag, cross_mag), delta);
  } else {
    // Endpoint vectors point in the same direction, so this sum does not cancel.
    g.gamma = scaled_from(g.alpha + g.beta);
  }

  if (!(g.A > T{0}) || !(g.B > T{0}) || !(g.alpha > T{0})
      || !(g.gamma.mantissa > T{0}) || !isfinite(g.gamma.mantissa)) {
    numeric_failure = true;
  }
  return g;
}

// Closed-form field of one straight filamentary segment. With endpoint
// displacements divided by D, the dimensional prefactor is K*I*L/D^2.
template <class T>
__device__ __forceinline__
ScaledVec3<T> segment_B(Vec3T<T> a, Vec3T<T> b, Vec3T<T> p, T I,
                        bool& singular, bool& numeric_failure) {
  if (I == T{0}) return {};
  const SegmentGeometry<T> g =
      segment_geometry(a, b, p, singular, numeric_failure);
  if (singular || numeric_failure) return {};

  ScaledNumber<T> f = scaled_divide(
      scaled_divide(scaled_from(g.A + g.B), g.alpha), g.gamma);
  ScaledNumber<T> prefactor = scaled_multiply(
      scaled_multiply(scaled_from(mu0_over_4pi_v<T>), I), g.length);
  prefactor = scaled_divide(
      scaled_divide(prefactor, g.distance_scale), g.distance_scale);
  prefactor = scaled_multiply(prefactor, f);
  // An exactly collinear point outside the finite segment has B=0.  Repeating
  // the cross product with rounded unit vectors can manufacture a tiny
  // transverse component, which a large current then amplifies into a huge
  // false field.  The exact predicate used for singularity classification is
  // also authoritative for this exterior zero.
  const Vec3T<T> u = g.collinear ? Vec3T<T>{} : cross(g.t, g.x);
  return ScaledVec3<T>{scaled_multiply(prefactor, u.x),
                       scaled_multiply(prefactor, u.y),
                       scaled_multiply(prefactor, u.z)};
}

// Magnetic vector potential of one straight segment. `log1p` avoids loss of
// all significant digits far from the segment. The identity
//
//   (A+B-q)(A+B+q) = 2 gamma,  q = segment_length / distance_scale
//
// turns the cancellation-prone logarithm into
// log1p(q*(A+B+q)/gamma). For an open segment this potential still satisfies
// curl(A)=B, but it is Coulomb gauge only after endpoint terms cancel in a
// closed/current-continuous conductor network.
template <class T>
__device__ __forceinline__
ScaledVec3<T> segment_A(Vec3T<T> a, Vec3T<T> b, Vec3T<T> p, T I,
                        bool& singular, bool& numeric_failure) {
  if (I == T{0}) return {};
  const SegmentGeometry<T> g =
      segment_geometry(a, b, p, singular, numeric_failure);
  if (singular || numeric_failure) return {};

  const T length_over_distance =
      materialize_scaled(g.length_over_distance, numeric_failure);
  if (numeric_failure) return {};
  ScaledNumber<T> ratio = scaled_multiply(
      g.length_over_distance, g.A + g.B + length_over_distance);
  ratio = scaled_divide(ratio, g.gamma);
  const ScaledNumber<T> log_ratio = scaled_log1p(ratio);
  ScaledNumber<T> prefactor = scaled_multiply(
      scaled_multiply(scaled_from(mu0_over_4pi_v<T>), I), log_ratio);
  return ScaledVec3<T>{scaled_multiply(prefactor, g.t.x),
                       scaled_multiply(prefactor, g.t.y),
                       scaled_multiply(prefactor, g.t.z)};
}

// Closed-form Jacobian dB_i/dp_j. Differentiation is performed in the same
// dimensionless coordinates as segment_B. Writing grad(f) through logarithmic
// derivatives removes the D^2 intermediate that made the old fp32 expression
// overflow/underflow under a simple similarity scaling.
template <class T>
__device__ __forceinline__ ScaledMat3x3<T>
segment_gradB_endpoint_form(const SegmentGeometry<T>& g, T I,
                            bool& numeric_failure) {
  const T cos_a = dot(g.t, g.unit_a);
  const T cos_b = dot(g.t, g.unit_b);
  const T h = cos_a - cos_b;

  const bool use_a = g.distance_a.exponent < g.distance_b.exponent
      || (g.distance_a.exponent == g.distance_b.exponent
          && fabs(g.distance_a.mantissa) <= fabs(g.distance_b.mantissa));
  const Vec3T<T> reference_unit = use_a ? g.unit_a : g.unit_b;
  const ScaledNumber<T> reference_distance =
      use_a ? g.distance_a : g.distance_b;
  const T reference_cos = use_a ? cos_a : cos_b;
  const Vec3T<T> rho_unit = reference_unit - reference_cos * g.t;
  const Vec3T<T> cross_unit = cross(g.t, reference_unit);
  const T sin2 = dot(cross_unit, cross_unit);
  if (!(sin2 > T{0}) || !isfinite(sin2)) {
    numeric_failure = true;
    return {};
  }

  const ScaledNumber<T> radius_squared = scaled_multiply(
      scaled_multiply(reference_distance, reference_distance), sin2);
  const ScaledNumber<T> radius_fourth =
      scaled_multiply(radius_squared, radius_squared);
  const ScaledNumber<T> w =
      scaled_divide(scaled_from(h), radius_squared);

  const auto endpoint_grad_h = [&](Vec3T<T> unit, T cosine,
                                   ScaledNumber<T> distance) {
    const Vec3T<T> rho = unit - cosine * g.t;
    const Vec3T<T> cross_value = cross(g.t, unit);
    const T local_sin2 = dot(cross_value, cross_value);
    const Vec3T<T> numerator = local_sin2 * g.t - cosine * rho;
    return ScaledVec3<T>{
        scaled_divide(scaled_from(numerator.x), distance),
        scaled_divide(scaled_from(numerator.y), distance),
        scaled_divide(scaled_from(numerator.z), distance)};
  };
  const ScaledVec3<T> grad_a =
      endpoint_grad_h(g.unit_a, cos_a, g.distance_a);
  const ScaledVec3<T> grad_b =
      endpoint_grad_h(g.unit_b, cos_b, g.distance_b);
  const auto grad_h_component = [](ScaledNumber<T> a, ScaledNumber<T> b) {
    return scaled_pair_sum(a, scaled_negate(b));
  };
  const ScaledVec3<T> grad_h{
      grad_h_component(grad_a.x, grad_b.x),
      grad_h_component(grad_a.y, grad_b.y),
      grad_h_component(grad_a.z, grad_b.z)};

  const auto grad_w_component = [&](ScaledNumber<T> grad_h_value,
                                    T rho_component) {
    const ScaledNumber<T> first =
        scaled_divide(grad_h_value, radius_squared);
    ScaledNumber<T> second = scaled_multiply(
        scaled_multiply(reference_distance, rho_component), T{-2} * h);
    second = scaled_divide(second, radius_fourth);
    return scaled_pair_sum(first, second);
  };
  const ScaledVec3<T> grad_w{
      grad_w_component(grad_h.x, rho_unit.x),
      grad_w_component(grad_h.y, rho_unit.y),
      grad_w_component(grad_h.z, rho_unit.z)};

  const Vec3T<T> tx_row0{T{0}, -g.t.z,  g.t.y};
  const Vec3T<T> tx_row1{g.t.z, T{0},  -g.t.x};
  const Vec3T<T> tx_row2{-g.t.y, g.t.x, T{0}};
  const ScaledNumber<T> source =
      scaled_multiply(scaled_from(mu0_over_4pi_v<T>), I);
  const auto row = [&](Vec3T<T> tx, T cross_component) {
    const ScaledNumber<T> physical_cross =
        scaled_multiply(reference_distance, cross_component);
    const auto entry = [&](T tx_component, ScaledNumber<T> grad_w_value) {
      const ScaledNumber<T> curl_term = scaled_multiply(w, tx_component);
      const ScaledNumber<T> scale_term =
          scaled_multiply(physical_cross, grad_w_value);
      return scaled_multiply(source, scaled_pair_sum(curl_term, scale_term));
    };
    return ScaledVec3<T>{entry(tx.x, grad_w.x), entry(tx.y, grad_w.y),
                         entry(tx.z, grad_w.z)};
  };
  return ScaledMat3x3<T>{row(tx_row0, cross_unit.x),
                          row(tx_row1, cross_unit.y),
                          row(tx_row2, cross_unit.z)};
}

template <class T>
__device__ __forceinline__
ScaledMat3x3<T> segment_gradB(Vec3T<T> a, Vec3T<T> b, Vec3T<T> p, T I,
                              bool& singular, bool& numeric_failure) {
  if (I == T{0}) return {};
  const SegmentGeometry<T> g =
      segment_geometry(a, b, p, singular, numeric_failure);
  if (singular || numeric_failure) return {};
  if (!g.collinear
      && (g.A < std::numeric_limits<T>::min()
          || g.B < std::numeric_limits<T>::min())) {
    return segment_gradB_endpoint_form(g, I, numeric_failure);
  }

  // Divide components by their norms directly. Forming 1/A first can overflow
  // for a subnormal but nonzero endpoint distance, after which zero vector
  // components become 0*inf = NaN even though x/A is a finite unit vector.
  const Vec3T<T> x_over_A{g.x.x / g.A, g.x.y / g.A, g.x.z / g.A};
  const Vec3T<T> y_over_B{g.y.x / g.B, g.y.y / g.B, g.y.z / g.B};
  const Vec3T<T> grad_s = x_over_A + y_over_B;
  const Vec3T<T> grad_alpha = g.B * x_over_A + g.A * y_over_B;
  const Vec3T<T> grad_beta = g.x + g.y;

  ScaledVec3<T> grad_gamma;
  if (g.beta < T{0}) {
    const T delta = g.alpha - g.beta;
    const ScaledNumber<T> length_squared = scaled_multiply(
        g.length_over_distance, g.length_over_distance);
    const auto entry = [&](T rho, T alpha_minus_beta) {
      ScaledNumber<T> terms[2]{
          scaled_multiply(length_squared, T{2} * rho),
          scaled_negate(scaled_multiply(g.gamma, alpha_minus_beta))};
      return scaled_divide(scaled_largest_pair_sum(terms, 2), delta);
    };
    grad_gamma = ScaledVec3<T>{
        entry(g.rho.x, grad_alpha.x - grad_beta.x),
        entry(g.rho.y, grad_alpha.y - grad_beta.y),
        entry(g.rho.z, grad_alpha.z - grad_beta.z)};
  } else {
    const Vec3T<T> sum = grad_alpha + grad_beta;
    grad_gamma = ScaledVec3<T>{scaled_from(sum.x), scaled_from(sum.y),
                               scaled_from(sum.z)};
  }

  const T s = g.A + g.B;
  const ScaledNumber<T> f = scaled_divide(
      scaled_divide(scaled_from(s), g.alpha), g.gamma);
  const Vec3T<T> u = g.collinear ? Vec3T<T>{} : cross(g.t, g.x);

  const Vec3T<T> tx_row0{T{0}, -g.t.z,  g.t.y};
  const Vec3T<T> tx_row1{g.t.z, T{0},  -g.t.x};
  const Vec3T<T> tx_row2{-g.t.y, g.t.x, T{0}};

  // Differentiating the D-normalized field adds one more 1/D, giving the
  // dimensional prefactor K*I*L/D^3. Keep it and f exponent-scaled until the
  // final matrix component is known.
  ScaledNumber<T> prefactor = scaled_multiply(
      scaled_multiply(scaled_from(mu0_over_4pi_v<T>), I), g.length);
  prefactor = scaled_divide(
      scaled_divide(
          scaled_divide(prefactor, g.distance_scale), g.distance_scale),
      g.distance_scale);
  prefactor = scaled_multiply(prefactor, f);

  const auto row = [&](Vec3T<T> tx, T u_component) {
    const auto entry = [&](T tx_component, T grad_s_component,
                           T grad_alpha_component,
                           ScaledNumber<T> grad_gamma_component) {
      // Evaluate
      //   tx + u*(grad(s)/s - grad(alpha)/alpha - grad(gamma)/gamma)
      // as four scaled terms.  Every quotient is finite as a scaled number even
      // when its reciprocal alone is outside T, and largest-pair summation lets
      // the O(1) cancellations occur before a subnormal residual is rounded.
      ScaledNumber<T> terms[4]{
          scaled_from(tx_component),
          scaled_divide(
              scaled_multiply(scaled_from(u_component), grad_s_component), s),
          scaled_negate(scaled_divide(
              scaled_multiply(scaled_from(u_component), grad_alpha_component),
              g.alpha)),
          scaled_negate(scaled_divide(
              scaled_multiply(scaled_from(u_component), grad_gamma_component),
              g.gamma))};
      return scaled_multiply(prefactor,
                             scaled_largest_pair_sum(terms, 4));
    };
    return ScaledVec3<T>{
        entry(tx.x, grad_s.x, grad_alpha.x, grad_gamma.x),
        entry(tx.y, grad_s.y, grad_alpha.y, grad_gamma.y),
        entry(tx.z, grad_s.z, grad_alpha.z, grad_gamma.z)};
  };
  return ScaledMat3x3<T>{row(tx_row0, u.x), row(tx_row1, u.y),
                          row(tx_row2, u.z)};
}

}  // namespace quasar::magnetostatics::detail

#endif  // __HIPCC__
