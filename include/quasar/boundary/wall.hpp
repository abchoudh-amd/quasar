#pragma once

#include "quasar/boundary/boundary_condition.hpp"

namespace quasar::boundary {

// Perfect-electric-conductor (reflecting) field wall. At 2nd order it uses the
// one-sided characteristic closure (no ghost fill; boundary nodes corrected
// after each curl). At 4th order it currently falls back to the mirror ghost
// fill until the 4th-order closure lands.
class PecFieldBC final : public IFieldBoundary {
 public:
  void configure(int fdtd_order) override { order_ = fdtd_order; }
  void fill_ghosts(YeeField2D<Real>& field, Side side) const override;
  void correct_after_b(YeeField2D<Real>& field, Side side, Real dt) const override;
  void correct_after_e(YeeField2D<Real>& field, Side side, Real dt) const override;

 private:
  int order_{2};
};

class SpecularParticleBC final : public IParticleBoundary {
 public:
  void apply(pic::ParticleSpecies& species, Side side) const override;
  void fold_current(JField2D<Real>& current, Side side) const override;
};

class AbsorbingParticleBC final : public IParticleBoundary {
 public:
  void apply(pic::ParticleSpecies& species, Side side) const override;
};

}  // namespace quasar::boundary
