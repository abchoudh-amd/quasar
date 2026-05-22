#pragma once

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
};

template <int ShapeOrder>
class BorisPusher final : public IParticlePusher {
 public:
  void push(pic::ParticleSpecies& species,
            const YeeField2D<Real>& self_fields,
            const YeeField2D<Real>& external_fields,
            Real dt) const override;
};

}  // namespace quasar::numerics
