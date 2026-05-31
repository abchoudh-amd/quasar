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

enum class FieldBoundaryKind { periodic, pec, outflow };
enum class ParticleBoundaryKind { periodic, specular, absorbing };

struct BoundarySpec {
  std::array<FieldBoundaryKind, 4> field{
      FieldBoundaryKind::periodic, FieldBoundaryKind::periodic,
      FieldBoundaryKind::periodic, FieldBoundaryKind::periodic};
  std::array<ParticleBoundaryKind, 4> particle{
      ParticleBoundaryKind::periodic, ParticleBoundaryKind::periodic,
      ParticleBoundaryKind::periodic, ParticleBoundaryKind::periodic};
};

class IFieldBoundary {
 public:
  virtual ~IFieldBoundary() = default;

  // Pre-curl ghost fill. Used by the periodic BC (copy opposite edge) so the
  // ghost-aware stencil wraps; the one-sided/characteristic BCs (pec, outflow)
  // leave this a no-op and instead correct the boundary nodes after each curl.
  virtual void fill_ghosts(YeeField2D<Real>& field, Side side) const {}

  // Post-update boundary-node corrections. A one-sided / characteristic closure
  // overwrites the boundary row the interior curl just computed (which read
  // stale ghosts) with the correct one-sided stencil + closure. correct_after_b
  // runs right after advance_b; correct_after_e right after advance_e (so the
  // outflow Mur update can read the just-updated adjacent interior node).
  virtual void correct_after_b(YeeField2D<Real>& field, Side side, Real dt) const {}
  virtual void correct_after_e(YeeField2D<Real>& field, Side side, Real dt) const {}

  // Lets a BC pick an order-dependent kernel once at construction (the FDTD
  // order fixes how many boundary layers the closure must rewrite).
  virtual void configure(int fdtd_order) {}
};

class IParticleBoundary {
 public:
  virtual ~IParticleBoundary() = default;
  virtual void apply(pic::ParticleSpecies& species, Side side) const = 0;
};

}  // namespace quasar::boundary

#define QUASAR_REGISTER_FIELD_BOUNDARY(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::boundary::IFieldBoundary, Name, Class)

#define QUASAR_REGISTER_PARTICLE_BOUNDARY(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::boundary::IParticleBoundary, Name, Class)
