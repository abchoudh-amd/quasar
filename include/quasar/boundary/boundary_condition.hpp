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

enum class FieldBoundaryKind { periodic, pec };
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
  virtual void fill_ghosts(YeeField2D<Real>& field, Side side) const = 0;
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
