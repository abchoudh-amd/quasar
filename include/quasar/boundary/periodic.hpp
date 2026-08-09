#pragma once

#include "quasar/boundary/boundary_condition.hpp"

namespace quasar::boundary {

class PeriodicFieldBC final : public IFieldBoundary {
 public:
  int ghost_continuation_mode() const noexcept override { return 0; }
  void fill_ghosts(YeeField2D<Real>& field, Side side) const override;
};

class PeriodicParticleBC final : public IParticleBoundary {
 public:
  void apply(pic::ParticleSpecies& species, Side side) const override;
};
}  // namespace quasar::boundary
