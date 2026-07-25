#pragma once

#include "quasar/core/grid.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/core/yee_field.hpp"

#include <array>
#include <string>

namespace quasar::pic {
class ParticleSpecies;
}

namespace quasar::boundary {

// Per-side boundary selection by registry name (order: x_lo, x_hi, y_lo, y_hi).
// The names map directly to the QUASAR_REGISTER_*_BOUNDARY keys, so adding a
// boundary is a single registration with no enum / switch / binding to update.
// Field names: "periodic" | "pec" | "outflow". Particle names: "periodic" |
// "specular" | "absorbing".
struct BoundarySpec {
  std::array<std::string, 4> field{"periodic", "periodic", "periodic", "periodic"};
  std::array<std::string, 4> particle{"periodic", "periodic", "periodic", "periodic"};
};

class IFieldBoundary {
 public:
  virtual ~IFieldBoundary() = default;

  // Pre-curl ghost fill. Periodic copies the opposite edge, PEC applies the
  // component's exact even/odd wall parity, and outflow linearly continues the
  // live characteristic boundary state. All closures include corner halos.
  virtual void fill_ghosts(YeeField2D<Real>& field, Side side) const {}

  // Post-update refresh/correction. correct_after_b restores B ghost parity or
  // continuation after Faraday. correct_after_e does the same after Ampere; an
  // outflow additionally advances tangential E with its Mur characteristic.
  virtual void correct_after_b(YeeField2D<Real>& field, Side side, Real dt) const {}
  virtual void correct_after_e(YeeField2D<Real>& field, Side side, Real dt) const {}

  // Lets a BC pick an order-dependent kernel once at construction (the FDTD
  // order fixes how many boundary layers the closure must rewrite).
  virtual void configure(int fdtd_order) {}
  virtual void configure_geometry(bool cylindrical) {}

  // For an outflow face: whether its low/high transverse ends meet another
  // outflow. The diagonal characteristic owns the doubly-tangential component
  // there, so both face updates skip it. Other component corners have a unique
  // owning face from their staggering and are still advanced.
  virtual void set_corner_skip(bool skip_lo, bool skip_hi) {}
};

class IParticleBoundary {
 public:
  virtual ~IParticleBoundary() = default;
  virtual void configure_geometry(bool cylindrical) {}
  virtual void apply(pic::ParticleSpecies& species, Side side) const = 0;

  // Optional pre-deposit trajectory closure.  An absorbing wall uses this hook
  // after the particle push but before current deposition to extend a crossing
  // macro-particle's finite shape just past the wall.  Its endpoint then carries
  // zero charge in every physical cell, so the deposited boundary current
  // removes the complete shape rather than only the portion whose centre left.
  virtual void prepare_deposit(pic::ParticleSpecies& species, Side side,
                               int shape_order) const {}

  // Post-deposit current closure. A reflecting BC (specular) deposits boundary-
  // crossing current into the ghost cells; this hook folds it back into the
  // interior as image current before the filter / E-update read J. Non-reflecting
  // BCs (periodic, absorbing) leave this a no-op.
  virtual void fold_current(JField2D<Real>& current, Side side) const {}

  // Scalar counterpart of fold_current.  A reflecting wall mirrors charge
  // deposited in its ghost cells back into the physical cells, preserving the
  // macro-particle's total charge as its shape overlaps the wall.
  virtual void fold_charge(ScalarGrid2D<Real>& charge, Side side) const {}
};

}  // namespace quasar::boundary

#define QUASAR_REGISTER_FIELD_BOUNDARY(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::boundary::IFieldBoundary, Name, Class)

#define QUASAR_REGISTER_PARTICLE_BOUNDARY(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::boundary::IParticleBoundary, Name, Class)
