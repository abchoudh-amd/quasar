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
  // Selects per-axis node indexing in the deposit: a periodic axis wraps, a wall
  // axis deposits into ghost cells (later folded back by the specular BC). An axis
  // is periodic only when BOTH of its sides are periodic.
  virtual void set_periodic_axes(bool periodic_x, bool periodic_y) = 0;
};

template <int ShapeOrder>
class Esirkepov2D final : public IDepositScheme {
 public:
  void deposit(const pic::ParticleSpecies& species, JField2D<Real>& current,
               Real dt) const override;
  void set_periodic_axes(bool periodic_x, bool periodic_y) override {
    periodic_x_ = periodic_x;
    periodic_y_ = periodic_y;
  }

 private:
  bool periodic_x_{true};
  bool periodic_y_{true};
};

}  // namespace quasar::numerics
