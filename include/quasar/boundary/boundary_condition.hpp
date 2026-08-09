#pragma once

#include "quasar/core/grid.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/core/yee_field.hpp"

#include <array>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace quasar::pic {
class ParticleSpecies;
}

namespace quasar::boundary {

// Per-side boundary selection by registry name (order: x_lo, x_hi, y_lo, y_hi).
// The names map directly to the QUASAR_REGISTER_*_BOUNDARY keys, so adding a
// boundary is a single registration with no enum / switch / binding to update.
// Public/deck field names are "periodic" | "pec" | "outflow" and particle
// names are "periodic" | "specular" | "absorbing".  The reserved
// "internal" name is installed only by the distributed tile builder; physics
// YAML must never use it.
struct BoundarySpec {
  std::array<std::string, 4> field{"periodic", "periodic", "periodic", "periodic"};
  std::array<std::string, 4> particle{"periodic", "periodic", "periodic", "periodic"};
};

class IFieldBoundary {
 public:
  virtual ~IFieldBoundary() = default;

  // Compact fourth-order current correction needs the ghost continuation
  // selected by the boundary implementation.  Keeping this discriminator on
  // the plug-in avoids string dispatch in the solver and lets registered
  // boundaries describe their own closure.  A negative value means that the
  // boundary does not support the fourth-order compact solve.
  virtual int ghost_continuation_mode() const noexcept { return -1; }
  virtual bool is_internal_cut() const noexcept { return false; }

  // Pre-curl ghost fill. Periodic copies the opposite edge, PEC applies the
  // component's exact even/odd wall parity, and outflow linearly continues the
  // live characteristic boundary state. All closures include corner halos.
  virtual void fill_ghosts(YeeField2D<Real>&, Side) const {}

  // Post-update refresh/correction. correct_after_b restores B ghost parity or
  // continuation after Faraday. correct_after_e does the same after Ampere; an
  // outflow additionally advances tangential E with its Mur characteristic.
  virtual void correct_after_b(YeeField2D<Real>&, Side, Real) const {}
  virtual void correct_after_e(YeeField2D<Real>&, Side, Real) const {}

  // Lets a BC pick an order-dependent kernel once at construction (the FDTD
  // order fixes how many boundary layers the closure must rewrite).
  virtual void configure(int) {}
  virtual void configure_geometry(bool) {}

  // For an outflow face: whether its low/high transverse ends meet another
  // outflow. The diagonal characteristic owns the doubly-tangential component
  // there, so both face updates skip it. Other component corners have a unique
  // owning face from their staggering and are still advanced.
  virtual void set_corner_skip(bool, bool) {}

  // Committed-step restart seam. Stateless boundaries keep the empty default;
  // an outflow boundary returns its Mur strip in device layout and whether the
  // strip has been primed. The distributed runtime assembles/repartitions the
  // strips, so concrete boundaries remain topology-agnostic.
  virtual std::vector<Real> checkpoint_history() const { return {}; }
  virtual bool checkpoint_history_primed() const noexcept { return false; }
  virtual void restore_checkpoint_history(std::span<const Real> history,
                                          bool primed) {
    if (!history.empty() || primed) {
      throw std::invalid_argument{
          "field boundary does not accept checkpoint history"};
    }
  }
};

class IParticleBoundary {
 public:
  virtual ~IParticleBoundary() = default;
  virtual bool is_internal_cut() const noexcept { return false; }
  virtual void configure_geometry(bool) {}
  virtual void apply(pic::ParticleSpecies& species, Side side) const = 0;

  // Optional pre-deposit trajectory closure.  An absorbing wall uses this hook
  // after the particle push but before current deposition to extend a crossing
  // macro-particle's finite shape just past the wall.  Its endpoint then carries
  // zero charge in every physical cell, so the deposited boundary current
  // removes the complete shape rather than only the portion whose centre left.
  virtual void prepare_deposit(pic::ParticleSpecies&, Side, int) const {}

  // Post-deposit current closure. A reflecting BC (specular) deposits boundary-
  // crossing current into the ghost cells; this hook folds it back into the
  // interior as image current before the filter / E-update read J. Non-reflecting
  // BCs (periodic, absorbing) leave this a no-op.
  virtual void fold_current(JField2D<Real>&, Side) const {}

  // Scalar counterpart of fold_current.  A reflecting wall mirrors charge
  // deposited in its ghost cells back into the physical cells, preserving the
  // macro-particle's total charge as its shape overlaps the wall.
  virtual void fold_charge(ScalarGrid2D<Real>&, Side) const {}
};

}  // namespace quasar::boundary

#define QUASAR_REGISTER_FIELD_BOUNDARY(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::boundary::IFieldBoundary, Name, Class)

#define QUASAR_REGISTER_PARTICLE_BOUNDARY(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::boundary::IParticleBoundary, Name, Class)
