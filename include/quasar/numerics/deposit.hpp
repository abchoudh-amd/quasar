#pragma once

#include "quasar/core/yee_field.hpp"

namespace quasar::pic {
class ParticleSpecies;
}

namespace quasar::numerics {

class IDepositScheme {
 public:
  virtual ~IDepositScheme() = default;
  virtual void deposit(const pic::ParticleSpecies& species, JField2D<Real>& current,
                       Real dt) const = 0;
};

template <int ShapeOrder>
class Esirkepov2D final : public IDepositScheme {
 public:
  void deposit(const pic::ParticleSpecies& species, JField2D<Real>& current,
               Real dt) const override;
};

}  // namespace quasar::numerics
