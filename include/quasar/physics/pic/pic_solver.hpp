#pragma once

#include "quasar/boundary/boundary_condition.hpp"
#include "quasar/core/normalization.hpp"
#include "quasar/core/yee_field.hpp"
#include "quasar/numerics/deposit.hpp"
#include "quasar/numerics/field_evaluator.hpp"
#include "quasar/numerics/field_solver.hpp"
#include "quasar/numerics/filter.hpp"
#include "quasar/numerics/particle_pusher.hpp"
#include "quasar/physics/pic/species.hpp"

#include <array>
#include <memory>
#include <vector>

namespace quasar::magnetostatics {
class ConductorSystem;
}

namespace quasar::pic {

struct EmPicConfig {
  Grid2D grid{};
  int fdtd_order{2};
  int shape_order{1};
  boundary::BoundarySpec boundary{};
  Normalization normalization{};
};

class EmPic2D3V {
 public:
  explicit EmPic2D3V(EmPicConfig cfg);

  Grid2D grid() const noexcept { return grid_; }
  YeeField2D<Real>& fields() noexcept { return fields_; }
  YeeField2D<Real>& external_fields() noexcept { return external_fields_; }
  JField2D<Real>& current() noexcept { return current_; }
  const std::vector<ParticleSpecies>& species() const noexcept { return species_; }
  std::vector<ParticleSpecies>& species() noexcept { return species_; }

  void add_species(ParticleSpecies s);
  void step(Real dt);
  void advance(Real t_end, Real dt);

  const EmPicConfig& config() const noexcept { return cfg_; }

 private:
  void fill_field_ghosts();
  void apply_particle_bcs(ParticleSpecies& s);
  bool has_absorbing_boundary() const noexcept;

  EmPicConfig cfg_{};
  // Steps taken so far; drives the particle-compaction cadence.
  std::size_t step_count_{0};
  Grid2D grid_{};
  YeeField2D<Real> fields_{};
  YeeField2D<Real> external_fields_{};
  JField2D<Real> current_{};
  std::vector<ParticleSpecies> species_{};
  std::unique_ptr<numerics::IFieldSolver> field_solver_{};
  std::unique_ptr<numerics::IParticlePusher> pusher_{};
  std::unique_ptr<numerics::IDepositScheme> deposit_{};
  numerics::FilterPipeline filters_{};
  // Per-side boundary conditions, indexed by Side (x_lo, x_hi, y_lo, y_hi),
  // constructed from cfg_.boundary through the plugin registry.
  std::array<std::unique_ptr<boundary::IParticleBoundary>, 4> particle_bcs_{};
  std::array<std::unique_ptr<boundary::IFieldBoundary>, 4> field_bcs_{};
};

// Samples the evaluator's E/B at the Yee grid's cell-staggered points and stores
// them in external_fields. The grid coordinates are multiplied by `length_scale`
// (internal -> SI metres) before the SI evaluator is called, and the returned SI
// field is divided by `e_field_scale`/`b_field_scale` to land in the solver's
// internal units. All scales default to 1, i.e. a pure SI pass-through that leaves
// the existing (SI) behaviour unchanged; pass a plasma normalization's scales to
// drive an SI field source from a normalized solver.
void sample_external_field(numerics::IFieldEvaluator& evaluator,
                           const magnetostatics::ConductorSystem& conductors,
                           YeeField2D<Real>& external_fields,
                           Real length_scale = Real{1},
                           Real e_field_scale = Real{1},
                           Real b_field_scale = Real{1});

}  // namespace quasar::pic
