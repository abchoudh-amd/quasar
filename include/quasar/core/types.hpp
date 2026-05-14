#pragma once

#include <cmath>
#include <cstddef>

#if defined(__HIPCC__) || defined(__CUDACC__)
  #define QUASAR_HOST_DEVICE __host__ __device__
#else
  #define QUASAR_HOST_DEVICE
#endif

namespace quasar {

using Real = double;
using Size = std::size_t;

inline constexpr Real pi           = static_cast<Real>(3.141592653589793238462643383279502884L);
inline constexpr Real mu0          = static_cast<Real>(4) * pi * static_cast<Real>(1e-7);
inline constexpr Real mu0_over_4pi = static_cast<Real>(1e-7);
inline constexpr Real kEps         = static_cast<Real>(1e-30);

struct Vec3 {
  Real x{};
  Real y{};
  Real z{};

  QUASAR_HOST_DEVICE constexpr Vec3() = default;
  QUASAR_HOST_DEVICE constexpr Vec3(Real X, Real Y, Real Z) : x{X}, y{Y}, z{Z} {}
};

QUASAR_HOST_DEVICE constexpr inline Vec3 operator+(Vec3 a, Vec3 b) noexcept {
  return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}
QUASAR_HOST_DEVICE constexpr inline Vec3 operator-(Vec3 a, Vec3 b) noexcept {
  return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}
QUASAR_HOST_DEVICE constexpr inline Vec3 operator-(Vec3 a) noexcept {
  return Vec3{-a.x, -a.y, -a.z};
}
QUASAR_HOST_DEVICE constexpr inline Vec3 operator*(Real s, Vec3 a) noexcept {
  return Vec3{s * a.x, s * a.y, s * a.z};
}
QUASAR_HOST_DEVICE constexpr inline Vec3 operator*(Vec3 a, Real s) noexcept {
  return s * a;
}
QUASAR_HOST_DEVICE constexpr inline Vec3 operator/(Vec3 a, Real s) noexcept {
  return Vec3{a.x / s, a.y / s, a.z / s};
}

QUASAR_HOST_DEVICE constexpr inline Real dot(Vec3 a, Vec3 b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
QUASAR_HOST_DEVICE constexpr inline Vec3 cross(Vec3 a, Vec3 b) noexcept {
  return Vec3{a.y * b.z - a.z * b.y,
              a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x};
}
QUASAR_HOST_DEVICE constexpr inline Real length_squared(Vec3 a) noexcept {
  return dot(a, a);
}
QUASAR_HOST_DEVICE inline Real length(Vec3 a) noexcept {
  return std::sqrt(length_squared(a));
}
QUASAR_HOST_DEVICE inline Vec3 normalized(Vec3 a) noexcept {
  return a / length(a);
}

struct Mat3x3 {
  Vec3 r0{};
  Vec3 r1{};
  Vec3 r2{};

  QUASAR_HOST_DEVICE constexpr Mat3x3() = default;
  QUASAR_HOST_DEVICE constexpr Mat3x3(Vec3 row0, Vec3 row1, Vec3 row2)
    : r0{row0}, r1{row1}, r2{row2} {}
};

QUASAR_HOST_DEVICE constexpr inline Vec3 operator*(const Mat3x3& m, Vec3 v) noexcept {
  return Vec3{dot(m.r0, v), dot(m.r1, v), dot(m.r2, v)};
}

}  // namespace quasar
