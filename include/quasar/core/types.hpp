#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#if defined(__HIPCC__) || defined(__CUDACC__)
  #define QUASAR_HOST_DEVICE __host__ __device__
#else
  #define QUASAR_HOST_DEVICE
#endif

namespace quasar {

// `Real` is the default precision for host-side code that does not need
// fp32 / fp64 distinction. The Biot-Savart kernels are now templated on a
// precision parameter T (see Vec3T<T> below); Real is preserved here so the
// existing test/example code keeps compiling unchanged.
using Real = double;
using Size = std::size_t;

// -- Variable-template constants ------------------------------------------
//
// All physical constants are available both as a precision-specific
// variable template (e.g. `mu0_v<float>`) and as a plain `double` alias
// (e.g. `mu0`). The aliases preserve the default double-precision API.

template <class T>
inline constexpr T pi_v = static_cast<T>(3.141592653589793238462643383279502884L);

template <class T>
inline constexpr T mu0_v = static_cast<T>(4) * pi_v<T> * static_cast<T>(1e-7);

template <class T>
inline constexpr T mu0_over_4pi_v = static_cast<T>(1e-7);

template <class T>
inline constexpr T kEps_v = static_cast<T>(1e-30);

// Relative tolerance for geometry-scaled singularity guards: a few ulp of the
// working precision, so the cutoff tracks the magnitude of the quantity being
// tested rather than a fixed absolute floor (see segment_B / segment_gradB).
template <class T>
inline constexpr T kRelEps_v = static_cast<T>(8) * std::numeric_limits<T>::epsilon();

inline constexpr Real pi           = pi_v<Real>;
inline constexpr Real mu0          = mu0_v<Real>;
inline constexpr Real mu0_over_4pi = mu0_over_4pi_v<Real>;
inline constexpr Real kEps         = kEps_v<Real>;
inline constexpr Real kRelEps      = kRelEps_v<Real>;

// -- Vec3T -----------------------------------------------------------------

template <class T>
struct Vec3T {
  T x{};
  T y{};
  T z{};

  QUASAR_HOST_DEVICE constexpr Vec3T() = default;
  QUASAR_HOST_DEVICE constexpr Vec3T(T X, T Y, T Z) : x{X}, y{Y}, z{Z} {}
};

template <class T>
QUASAR_HOST_DEVICE constexpr inline Vec3T<T>
operator+(Vec3T<T> a, Vec3T<T> b) noexcept {
  return Vec3T<T>{a.x + b.x, a.y + b.y, a.z + b.z};
}
template <class T>
QUASAR_HOST_DEVICE constexpr inline Vec3T<T>
operator-(Vec3T<T> a, Vec3T<T> b) noexcept {
  return Vec3T<T>{a.x - b.x, a.y - b.y, a.z - b.z};
}
template <class T>
QUASAR_HOST_DEVICE constexpr inline Vec3T<T> operator-(Vec3T<T> a) noexcept {
  return Vec3T<T>{-a.x, -a.y, -a.z};
}
template <class T>
QUASAR_HOST_DEVICE constexpr inline Vec3T<T>
operator*(T s, Vec3T<T> a) noexcept {
  return Vec3T<T>{s * a.x, s * a.y, s * a.z};
}
template <class T>
QUASAR_HOST_DEVICE constexpr inline Vec3T<T>
operator*(Vec3T<T> a, T s) noexcept {
  return s * a;
}
template <class T>
QUASAR_HOST_DEVICE constexpr inline Vec3T<T>
operator/(Vec3T<T> a, T s) noexcept {
  return Vec3T<T>{a.x / s, a.y / s, a.z / s};
}

template <class T>
QUASAR_HOST_DEVICE constexpr inline T dot(Vec3T<T> a, Vec3T<T> b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
template <class T>
QUASAR_HOST_DEVICE constexpr inline Vec3T<T>
cross(Vec3T<T> a, Vec3T<T> b) noexcept {
  return Vec3T<T>{a.y * b.z - a.z * b.y,
                  a.z * b.x - a.x * b.z,
                  a.x * b.y - a.y * b.x};
}
template <class T>
QUASAR_HOST_DEVICE constexpr inline T length_squared(Vec3T<T> a) noexcept {
  return dot(a, a);
}
template <class T>
QUASAR_HOST_DEVICE inline T length(Vec3T<T> a) noexcept {
  // Scale before squaring so a finite vector does not spuriously underflow or
  // overflow merely because its norm is far from unity.
  const T scale = fmax(fabs(a.x), fmax(fabs(a.y), fabs(a.z)));
  // Match hypot's IEEE special-value semantics: an infinite component
  // dominates a NaN, while a NaN with no infinite component propagates.  fmax
  // intentionally suppresses a lone NaN, so test for it after the infinity
  // case and before treating the resulting zero scale as a zero vector.
  if (scale == std::numeric_limits<T>::infinity()) return scale;
  if (std::isnan(a.x) || std::isnan(a.y) || std::isnan(a.z)) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  if (scale == T{0}) return T{0};
  const Vec3T<T> scaled = a / scale;
  return scale * std::sqrt(dot(scaled, scaled));
}
template <class T>
QUASAR_HOST_DEVICE inline Vec3T<T> normalized(Vec3T<T> a) noexcept {
  // Normalize in scaled coordinates. Dividing by length(a) would turn a
  // perfectly finite vector such as (DBL_MAX, DBL_MAX, 0) into zero when its
  // Euclidean norm itself overflows, even though the unit direction is fully
  // representable. Preserve the historical NaN result for the zero vector.
  const T scale = fmax(fabs(a.x), fmax(fabs(a.y), fabs(a.z)));
  if (scale == T{0}) return a / scale;
  const Vec3T<T> scaled = a / scale;
  return scaled / std::sqrt(dot(scaled, scaled));
}

// -- Mat3x3T ---------------------------------------------------------------

template <class T>
struct Mat3x3T {
  Vec3T<T> r0{};
  Vec3T<T> r1{};
  Vec3T<T> r2{};

  QUASAR_HOST_DEVICE constexpr Mat3x3T() = default;
  QUASAR_HOST_DEVICE constexpr Mat3x3T(Vec3T<T> row0, Vec3T<T> row1, Vec3T<T> row2)
    : r0{row0}, r1{row1}, r2{row2} {}
};

template <class T>
QUASAR_HOST_DEVICE constexpr inline Vec3T<T>
operator*(const Mat3x3T<T>& m, Vec3T<T> v) noexcept {
  return Vec3T<T>{dot(m.r0, v), dot(m.r1, v), dot(m.r2, v)};
}

// -- Backward-compat aliases -----------------------------------------------
//
// Existing call sites use `Vec3` and `Mat3x3` and expect them to be the
// double-precision instantiation. `Vec3f` / `Mat3x3f` are the fp32 siblings.

using Vec3    = Vec3T<Real>;
using Vec3f   = Vec3T<float>;
using Mat3x3  = Mat3x3T<Real>;
using Mat3x3f = Mat3x3T<float>;

}  // namespace quasar
