#pragma once

#include "quasar/boundary/boundary_condition.hpp"
#include "quasar/core/field_source.hpp"
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
#include <string>
#include <string_view>
#include <vector>

namespace quasar::pic {

// One current-smoothing filter in the deck-configured pipeline: a registry name
// ("binomial" / "compensated_binomial") and its pass count.
struct FilterSpec {
  std::string name{};
  int passes{1};
};

struct EmPicConfig {
  Grid2D grid{};
  // FDTD order (2 or 4): numeric because it also sets the ghost-halo width and the
  // boundary closures' one-sided layer count. The field-solver registry name is
  // derived from it ("yee_o{order}").
  int fdtd_order{2};
  // Particle shape, carried as the deck vocabulary ("cic" / "tsc"). The pusher and
  // deposit registry names are derived from it ("boris_{shape}", "esirkepov_{shape}").
  std::string shape{"cic"};
  // Which lab plane the 2D grid represents, used only by sample_external_field to
  // place sample points and map the returned 3-vector into the PIC frame:
  // "xy" (default) = lab z=0 slice, grid axes (x,y); "xz" = lab y=0 meridional
  // slice, grid axes (x,z) with the out-of-plane component along lab y.
  std::string plane{"xy"};
  // Coordinate geometry: "cartesian" (default) keeps the existing (x,y) Yee
  // scheme; "cylindrical" selects the axisymmetric m=0 (r,z) schemes
  // (yee_cyl_o2 / boris_cyl_{shape} / esirkepov_cyl_{shape}) and auto-wires the
  // r=0 (x_lo) side to the on-axis BC. In cylindrical mode the x-axis is the
  // radius r and the y-axis is the axial coordinate z.
  std::string geometry{"cartesian"};
  boundary::BoundarySpec boundary{};
  Normalization normalization{};
  std::vector<FilterSpec> filters{};
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
  // The CFL stability limit (internal units, c = 1) for the scheme actually
  // running: the cylindrical 2nd-order limit in cylindrical mode, else the
  // Cartesian limit for cfg_.fdtd_order. step()/advance() reject any dt above it.
  Real cfl_limit() const;
  // Drains each species' persistent deposit-overflow flag and throws if any
  // deposit spilled outside the deposition window. step() runs this on a cadence;
  // advance() runs it after the final step. A driver that calls step() directly
  // (e.g. the Python run loop) must call finalize() once after its last step so a
  // late-run overflow is not missed.
  void finalize();

  const EmPicConfig& config() const noexcept { return cfg_; }

 private:
  void fill_field_ghosts();
  void correct_field_boundaries_b(Real dt);
  void correct_field_boundaries_e(Real dt);
  void apply_particle_bcs(ParticleSpecies& s);
  bool has_absorbing_boundary() const noexcept;
  void check_deposit_overflow();
  // Throws if dt exceeds cfl_limit(); used by step()/advance().
  void check_cfl(Real dt) const;

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

// Samples the evaluator's E/B at the Yee grid's cell-node points (matching the
// node-collocated convention the particle gather assumes) and stores them in
// external_fields. The grid coordinates are multiplied by `length_scale`
// (internal -> SI metres) before the SI evaluator is called, and the returned SI
// field is divided by `e_field_scale`/`b_field_scale` to land in the solver's
// internal units. All scales default to 1, i.e. a pure SI pass-through that leaves
// the existing (SI) behaviour unchanged; pass a plasma normalization's scales to
// drive an SI field source from a normalized solver.
//
// The source is the axis-neutral core::IFieldSource the evaluator contract
// already consumes: a magnetostatics::ConductorSystem for Biot-Savart, or an
// ignored placeholder for the analytic evaluators. This keeps the PIC external-
// field path off any concrete magnetostatics type.
//
// `plane` selects which lab plane the grid represents. "xy" (default) samples at
// lab (x, y, 0) and maps components identity. "xz" samples at lab (x, 0, z) and
// maps the returned vector through a right-handed 90-degree rotation about lab x
// so the grid frame (i=x, j=z, out-of-plane=-y) stays right-handed for the Boris
// cross products: bx<-B.x, by<-B.z, bz<--B.y (and likewise for E).
void sample_external_field(numerics::IFieldEvaluator& evaluator,
                           const core::IFieldSource& source,
                           YeeField2D<Real>& external_fields,
                           Real length_scale = Real{1},
                           Real e_field_scale = Real{1},
                           Real b_field_scale = Real{1},
                           std::string_view plane = "xy");

}  // namespace quasar::pic
