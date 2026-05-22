#include "quasar/core/normalization.hpp"

#include <cmath>

namespace quasar {

Normalization Normalization::plasma(Real n, Real q, Real m) {
  if (!(std::isfinite(n) && std::isfinite(q) && std::isfinite(m))
      || n <= Real{0} || q == Real{0} || m <= Real{0}) {
    throw std::invalid_argument{"Normalization::plasma: invalid reference values"};
  }
  Normalization out;
  out.n_ref = n;
  out.q_ref = q;
  out.m_ref = m;
  out.omega_p_ref = std::sqrt(n * q * q / (constants::eps0 * m));
  return out;
}

Real Normalization::to_internal(Real value, UnitTag tag) const {
  switch (tag) {
    case UnitTag::time:          return value * omega_p_ref;
    case UnitTag::length:        return value / length_scale();
    case UnitTag::velocity:      return value / constants::c0;
    case UnitTag::e_field:       return value / e_field_scale();
    case UnitTag::b_field:       return value / b_field_scale();
    case UnitTag::density:       return value / n_ref;
    case UnitTag::charge:        return value / q_ref;
    case UnitTag::mass:          return value / m_ref;
    case UnitTag::temperature_eV:return value;
  }
  throw std::invalid_argument{"Normalization::to_internal: unknown unit tag"};
}

Real Normalization::to_si(Real value, UnitTag tag) const {
  switch (tag) {
    case UnitTag::time:          return value / omega_p_ref;
    case UnitTag::length:        return value * length_scale();
    case UnitTag::velocity:      return value * constants::c0;
    case UnitTag::e_field:       return value * e_field_scale();
    case UnitTag::b_field:       return value * b_field_scale();
    case UnitTag::density:       return value * n_ref;
    case UnitTag::charge:        return value * q_ref;
    case UnitTag::mass:          return value * m_ref;
    case UnitTag::temperature_eV:return value;
  }
  throw std::invalid_argument{"Normalization::to_si: unknown unit tag"};
}

}  // namespace quasar
