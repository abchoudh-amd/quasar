#pragma once

#include "quasar/boundary/boundary_condition.hpp"

namespace quasar::boundary {

// A distributed subdomain interface is neither a physical wall nor a local
// periodic wrap.  The coordinator supplies field halos and migrates particles;
// these no-op closures expose that topology through the normal registry seam.
class InternalFieldBC final : public IFieldBoundary {
 public:
  int ghost_continuation_mode() const noexcept override { return 4; }
  bool is_internal_cut() const noexcept override { return true; }
};

class InternalParticleBC final : public IParticleBoundary {
 public:
  bool is_internal_cut() const noexcept override { return true; }
  void apply(pic::ParticleSpecies&, Side) const override {}
};

}  // namespace quasar::boundary
