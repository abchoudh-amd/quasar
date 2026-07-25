#pragma once

#include "quasar/boundary/boundary_condition.hpp"

namespace quasar::boundary {

// Perfect-electric-conductor (reflecting) field wall. Every curl is preceded
// and followed by a stagger-aware even/odd parity continuation into the halo;
// tangential electric samples on a physical face are pinned by the same fill.
class PecFieldBC final : public IFieldBoundary {
 public:
  void configure_geometry(bool cylindrical) override { cylindrical_ = cylindrical; }
  void fill_ghosts(YeeField2D<Real>& field, Side side) const override;
  void correct_after_b(YeeField2D<Real>& field, Side side, Real dt) const override;
  void correct_after_e(YeeField2D<Real>& field, Side side, Real dt) const override;

 private:
  bool cylindrical_{false};
};

class SpecularParticleBC final : public IParticleBoundary {
 public:
  void configure_geometry(bool cylindrical) override {
    cylindrical_ = cylindrical;
  }
  void apply(pic::ParticleSpecies& species, Side side) const override;
  void fold_current(JField2D<Real>& current, Side side) const override;
  void fold_charge(ScalarGrid2D<Real>& charge, Side side) const override;

 private:
  bool cylindrical_{false};
};

class AbsorbingParticleBC final : public IParticleBoundary {
 public:
  void apply(pic::ParticleSpecies& species, Side side) const override;
  void prepare_deposit(pic::ParticleSpecies& species, Side side,
                       int shape_order) const override;
};

}  // namespace quasar::boundary
