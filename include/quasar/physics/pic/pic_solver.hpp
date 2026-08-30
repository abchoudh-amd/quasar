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
#include "quasar/physics/pic/particle_sampling.hpp"
#include "quasar/physics/pic/species.hpp"

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace quasar::distributed {
class PicTileAccess;
}

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
  // (yee_cyl_o2 or yee_cyl_o4 / boris_cyl_{shape} / esirkepov_cyl_{shape}) and
  // auto-wires the r=0 (x_lo) side to the on-axis BC. In cylindrical mode the
  // x-axis is the radius r and the y-axis is the axial coordinate z.
  std::string geometry{"cartesian"};
  boundary::BoundarySpec boundary{};
  Normalization normalization{};
  std::vector<FilterSpec> filters{};
  // When enabled, add a fixed uniform background whose charge exactly cancels
  // the particles' initial total charge. The background remains fixed as
  // particles leave, so boundary loss produces the physically expected net
  // charge. This must be explicit; the solver never silently assumes ions.
  bool neutralizing_background{false};
  // Topology-independent identity of the resolved prescribed external-field
  // source (evaluator kind, parameters, conductors, plane, and unit scales).
  // The serial solver does not interpret this value; distributed checkpoints
  // use it only to reject a restart against a different physical background.
  std::string external_field_signature{};
  // Distributed checkpoint compatibility identity for the run-loop timestep
  // policy and nominal internal dt.  The solver itself continues to accept the
  // dt passed to step(); termination targets are deliberately excluded.
  std::string timestep_signature{};
};

class EmPic2D3V {
 public:
  explicit EmPic2D3V(EmPicConfig cfg);

  Grid2D grid() const noexcept { return grid_; }
  YeeField2D<Real>& fields() noexcept { return fields_; }
  const YeeField2D<Real>& fields() const noexcept { return fields_; }
  YeeField2D<Real>& external_fields() noexcept { return external_fields_; }
  const YeeField2D<Real>& external_fields() const noexcept { return external_fields_; }
  JField2D<Real>& current() noexcept { return current_; }
  const JField2D<Real>& current() const noexcept { return current_; }
  // Materializes the particle/background charge on first access so diagnostics
  // before the first timestep cannot silently observe a vacuum array.
  const ScalarGrid2D<Real>& charge_density();
  // A const solver cannot perform the required deposit. Make an uninitialized
  // access explicit rather than returning a plausible but wrong zero field.
  const ScalarGrid2D<Real>& charge_density() const;
  const std::vector<ParticleSpecies>& species() const noexcept { return species_; }

  void add_species(ParticleSpecies s);
  // Solver-owned host upload for a species that has already been inserted.
  // Initial positions are checked against this solver's physical domain. The
  // uploaded velocity components are physical values at t=0, not a pre-staggered
  // v^-1/2; step() advances them over dt/2 before the first drift. All
  // charge/background caches are invalidated. Uploads are forbidden after the
  // first evolution step begins because velocity then lives on leapfrog half
  // steps.
  void set_species_particles(std::size_t index,
                             const std::vector<Real>& x,
                             const std::vector<Real>& y,
                             const std::vector<Real>& vx,
                             const std::vector<Real>& vy,
                             const std::vector<Real>& vz,
                             const std::vector<Real>& weight);
  // Sample this species' initial particles directly into device memory. Same
  // preconditions and cache invalidation as set_species_particles, and the
  // same physical-domain requirement -- checked in the kernel that produces
  // the coordinates rather than on an uploaded array.
  void sample_species_particles(std::size_t index,
                                ParticleSampleConfig config);
  void step(Real dt);
  void advance(Real t_end, Real dt);
  // The CFL stability limit (internal units, c = 1) for the scheme actually
  // running: the selected cylindrical order in cylindrical mode, otherwise the
  // Cartesian limit for cfg_.fdtd_order. step()/advance() reject any dt above it.
  Real cfl_limit() const;
  // Drains each species' persistent deposit-error flag and throws if any
  // coordinate/value was nonrepresentable or a trajectory spilled outside the
  // deposition window. step() drains before the separate post-processing source
  // finiteness gate; finalize() remains a defensive public drain.
  void finalize();

  const EmPicConfig& config() const noexcept { return cfg_; }

 private:
  friend class ::quasar::distributed::PicTileAccess;

  void fill_field_ghosts();
  void correct_field_boundaries_b(Real dt);
  void correct_field_boundaries_e(Real dt);
  void prime_outflow_corners();
  void correct_outflow_corners(Real dt);
  void apply_particle_bcs_before_deposit(ParticleSpecies& s);
  void prepare_absorbing_bcs_for_deposit(ParticleSpecies& s);
  void apply_absorbing_bcs_after_deposit(ParticleSpecies& s);
  bool has_absorbing_boundary() const noexcept;
  void check_deposit_overflow();
  void deposit_charge_density(ScalarGrid2D<Real>& charge);
  void ensure_charge_density();
  void initialize_neutralizing_background();
  // Throws if dt exceeds cfl_limit(); used by step()/advance().
  void check_cfl(Real dt) const;

  EmPicConfig cfg_{};
  // Steps taken so far; drives the particle-compaction cadence.
  std::size_t step_count_{0};
  Grid2D grid_{};
  YeeField2D<Real> fields_{};
  YeeField2D<Real> external_fields_{};
  BField2D<Real> previous_b_{};
  JField2D<Real> current_{};
  ScalarGrid2D<Real> charge_{};
  ScalarGrid2D<Real> next_charge_{};
  // Shared sticky flag for the synchronous post-processing source scan. Unlike
  // the per-species deposit flags, this also covers wall foldback, filters,
  // order-four correction, periodic restoration, and background addition.
  backend::DeviceBuffer<unsigned int> source_finite_error_{1};
  bool charge_valid_{false};
  bool evolution_started_{false};
  bool background_initialized_{false};
  Real background_charge_density_{Real{0}};
  // Work arrays for the order-four current compatibility solve D4+=D2+S.
  // Empty for order two; allocated once by the constructor for order four.
  backend::DeviceBuffer<Real> current_rhs_x_{};
  backend::DeviceBuffer<Real> current_rhs_y_{};
  backend::DeviceBuffer<Real> current_iter_x_{};
  backend::DeviceBuffer<Real> current_iter_y_{};
  bool periodic_x_{true};
  bool periodic_y_{true};
  // Width of the preceding position step. After the special startup kick from
  // uploaded v(t=0) to v^1/2, B and particle velocity live at half steps, so a
  // changed dt advances them over (dt_prev + dt)/2 and uses unequal
  // interpolation weights to recover B at the integer force time.
  Real previous_dt_{Real{0}};
  bool has_previous_dt_{false};
  std::vector<ParticleSpecies> species_{};
  std::unique_ptr<numerics::IFieldSolver> field_solver_{};
  std::unique_ptr<numerics::IParticlePusher> pusher_{};
  std::unique_ptr<numerics::IDepositScheme> deposit_{};
  numerics::FilterPipeline filters_{};
  // Per-side boundary conditions, indexed by Side (x_lo, x_hi, y_lo, y_hi),
  // constructed from cfg_.boundary through the plugin registry.
  std::array<std::unique_ptr<boundary::IParticleBoundary>, 4> particle_bcs_{};
  std::array<std::unique_ptr<boundary::IFieldBoundary>, 4> field_bcs_{};
  backend::DeviceBuffer<Real> outflow_corner_history_{};
  unsigned int outflow_corner_mask_{0};
  bool outflow_corners_primed_{false};
};

// Samples each E/B component at its own Yee sub-lattice and stores it in
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
// so a Cartesian grid frame is (i=x, j=z, out-of-plane=-y).  Cylindrical mode
// stores physical (r,z,phi), for which +phi=+lab-y on the sampled phi=0 plane;
// its pusher performs the separate (r,z,phi)->(r,phi,z) permutation.
// Every padded Yee node is evaluated, including true boundary ghosts. This is
// required for finite-size wall gathers of a nonuniform prescribed field; a
// file-backed evaluator must therefore cover the solver's ghost-extended sample
// coordinates as well as the physical domain.
//
// Cylindrical sampling is a physical contract, not a reinterpretation of an
// arbitrary Cartesian slice.  The evaluator must be rotationally covariant
// about the selected symmetry axis for both E and B.  A nonzero magnetic field
// must also advertise a trustworthy evaluate_grad_B implementation, whose trace
// is checked for the continuous Maxwell constraint div(B)=0.  The prescribed
// field is gathered only by the particle pusher and is not advanced by the FDTD
// curl, so a smooth nonlinear solenoidal field is not required to cancel under
// the finite-difference divergence to roundoff.  `fdtd_order` must still match
// the owning field solver so its public order/halo contract is validated.
void sample_external_field(numerics::IFieldEvaluator& evaluator,
                           const core::IFieldSource& source,
                           YeeField2D<Real>& external_fields,
                           Real length_scale = Real{1},
                           Real e_field_scale = Real{1},
                           Real b_field_scale = Real{1},
                           std::string_view plane = "xy",
                           std::string_view geometry = "cartesian",
                           int fdtd_order = 2);

}  // namespace quasar::pic
