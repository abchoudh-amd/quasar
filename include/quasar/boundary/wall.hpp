#pragma once

#include "quasar/boundary/boundary_condition.hpp"

namespace quasar::boundary {

class PecFieldBC final : public IFieldBoundary {
 public:
  void fill_ghosts(YeeField2D<Real>& field, Side side) const override;
};

class SpecularParticleBC final : public IParticleBoundary {
 public:
  void apply(pic::ParticleSpecies& species, Side side) const override;
};

class AbsorbingParticleBC final : public IParticleBoundary {
 public:
  void apply(pic::ParticleSpecies& species, Side side) const override;
};

}  // namespace quasar::boundary
