#pragma once

#include "quasar/core/registry.hpp"
#include "quasar/core/yee_field.hpp"

namespace quasar::pic {
class ParticleSpecies;
}

namespace quasar::numerics {

class IParticlePusher {
 public:
  virtual ~IParticlePusher() = default;
  virtual void push(pic::ParticleSpecies& species,
                    const YeeField2D<Real>& self_fields,
                    const YeeField2D<Real>& external_fields,
                    Real dt) const = 0;
  // Selects per-axis field-gather indexing: a periodic axis wraps, a wall axis
  // clamps the interpolation stencil into the ghost layer (reading the field-BC
  // closure) instead of wrapping. An axis is periodic only when BOTH sides are.
  virtual void set_periodic_axes(bool periodic_x, bool periodic_y) = 0;
};

template <int ShapeOrder>
class BorisPusher final : public IParticlePusher {
 public:
  void push(pic::ParticleSpecies& species,
            const YeeField2D<Real>& self_fields,
            const YeeField2D<Real>& external_fields,
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

// Registers a concrete pusher under a deck-facing name (e.g. "boris_cic").
#define QUASAR_REGISTER_PUSHER(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::numerics::IParticlePusher, Name, Class)
