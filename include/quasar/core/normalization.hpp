#pragma once

#include "quasar/core/types.hpp"

#include <cmath>
#include <stdexcept>

namespace quasar {

namespace constants {
inline constexpr Real c0     = Real{299792458.0};
inline constexpr Real eps0   = Real{8.8541878128e-12};
inline constexpr Real qe_abs = Real{1.602176634e-19};
inline constexpr Real me     = Real{9.1093837015e-31};
}  // namespace constants

enum class UnitTag {
  time,
  length,
  velocity,
  e_field,
  b_field,
  density,
  charge,
  mass,
  temperature_eV
};

struct Normalization {
  Real n_ref{Real{1}};
  Real q_ref{Real{1}};
  Real m_ref{Real{1}};
  Real omega_p_ref{Real{1}};

  // The all-ones state is reserved for the dimensionless identity mapping.
  // Encoding the mode in the existing public state preserves this type's
  // layout while distinguishing it from every value produced by plasma().
  static constexpr Normalization identity() noexcept { return {}; }
  constexpr bool is_identity() const noexcept {
    return n_ref == Real{1} && q_ref == Real{1} && m_ref == Real{1}
        && omega_p_ref == Real{1};
  }

  static Normalization plasma(Real n_ref, Real q_ref, Real m_ref);

  Real to_internal(Real value, UnitTag tag) const;
  Real to_si(Real value, UnitTag tag) const;

  // These products/ratios use exponent-scaled implementations in the .cpp so
  // they remain robust on platforms where long double has no wider exponent
  // range than Real.
  Real length_scale() const noexcept;
  Real time_scale() const noexcept;
  Real e_field_scale() const noexcept;
  Real b_field_scale() const noexcept;
  Real temperature_eV_scale() const noexcept;
};

inline constexpr Normalization identity_normalization() noexcept {
  return Normalization::identity();
}

}  // namespace quasar
