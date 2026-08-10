#pragma once

// Ideal-MHD solver driver for the 2D finite-volume + CT vertical
// slice. This is the MHD analogue of pic::EmPic2D3V: a thin driver that resolves
// every pluggable scheme (flux reconstruction, Riemann solver, CT scheme, SSP-RK
// integrator, positivity limiter, per-side fluid/field boundaries) by deck-facing
// string name through the plugin registry, owns ALL device scratch (the live
// state, the SSP-RK stage registers, the residual register, the per-direction
// interface states, a flux scratch field, the corner EMF, and the CFL max-rate
// reduction scratch), and exposes the residual/stage seam the SSP-RK integrator
// drives. (The CT scheme owns its own div(B) reduction scratch, the only device
// buffer not held here.)
//
// The integrator (numerics::ISsprkIntegrator) is intentionally state-free: it
// only sequences calls to compute_residual / combine_stage / rk_register /
// residual_register on this solver, so the register routing and all buffer
// ownership live here. See docs/dev-guide and the kernels.hpp launch ABI for the
// device seam this driver calls.

#include "quasar/boundary/mhd_boundary.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/ct_scheme.hpp"
#include "quasar/numerics/flux_reconstruction.hpp"
#include "quasar/numerics/interface_states.hpp"
#include "quasar/numerics/positivity_limiter.hpp"
#include "quasar/numerics/radial_tables.hpp"
#include "quasar/numerics/riemann_solver.hpp"
#include "quasar/numerics/ssprk_integrator.hpp"
#include "quasar/physics/mhd/kernels.hpp"  // BoundaryFlags4
#include "quasar/physics/mhd/mhd_background.hpp"  // MhdBackgroundField, MhdBackgroundSpec
#include "quasar/physics/mhd/mhd_field.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace quasar::distributed {
class MhdTileAccess;
}

namespace quasar::mhd {

// Deck-facing configuration for the MHD solver. Every scheme axis is a registry
// name so a new scheme is selectable without touching this struct or the driver.
struct MhdConfig {
  Grid2D grid{};
  Real gamma{Real{5} / Real{3}};
  std::string geometry{"cartesian"};
  std::string reconstruction{"mp7"};
  std::string riemann{"hlld"};
  std::string integrator{"ssprk3"};
  std::string ct{"fd_ct_christlieb"};
  std::string positivity{"troubled_cell"};
  // Positive thresholds retained for the explicit repair kernel and deck/API
  // compatibility. Neither automatic initial-state construction nor
  // conservative evolution clamps to them. A successful step accepts only
  // rho>0 and internal energy>0; a request that cannot make representable
  // positive progress is restored and reported. An arbitrary nonzero floor
  // cannot be invariant without adding/removing conserved quantities.
  Real rho_floor{Real{1e-8}};
  Real p_floor{Real{1e-9}};
  // CFL safety factor (deck: numerics.cfl). The C++ contract carried no cfl
  // field, but the Python deck passes one; added here so cfl_limit() scales the
  // stable step by it. Default 0.4 is a conservative MHD multidimensional value.
  Real cfl{Real{0.4}};
  // Distributed checkpoint compatibility identity for the run-loop timestep
  // policy.  The solver does not interpret this value: the Python distributed
  // runner records "auto" or the exact fixed dt, while native orchestration may
  // supply an equivalent stable identity.  Absolute termination targets are
  // intentionally not part of this signature.
  std::string timestep_signature{};
  boundary::MhdBoundarySpec boundary{};
  // Static background magnetic field B0 for the field-split formulation
  // B = B0 + b. Disabled by default (enabled=false => zero-B0 fast path,
  // bit-identical to the no-background solver). When enabled the solver
  // allocates b0_ at the working grid; the B0 values are seeded from Python.
  MhdBackgroundSpec background{};
};

class MhdSolver2D {
 public:
  explicit MhdSolver2D(MhdConfig cfg);

  Grid2D grid() const noexcept { return grid_; }
  // A mutable view may be retained and written after this call returns. Once
  // exposed, live-state solenoidality therefore stays on the strict external-
  // data predicate for the lifetime of this solver instance.
  MhdField2D<Real>& state() noexcept;
  const MhdField2D<Real>& state() const noexcept { return rk_[0]; }
  const MhdConfig& config() const noexcept { return cfg_; }
  // Number of conservative substeps accepted while advancing the most recent
  // requested interval. Exposed so regression tests can prove that conservation
  // holds across actual positivity subcycling.
  int last_positivity_substeps() const noexcept {
    return last_positivity_substeps_;
  }

  // Stage a host buffer (sized grid.storage_size()) into the named live-state
  // component. Magnetic spellings accept both the staggered name and the short
  // name: "bx"/"bx_face", "by"/"by_face", "bz"/"bz_cell".
  void seed_state(std::string_view component, const std::vector<Real>& host_buf);

  // Stage a host buffer (sized grid.storage_size()) into a named static
  // background-field component for the field-split formulation B = B0 + b.
  // Spellings mirror seed_state's magnetic aliases: "b0x"/"b0x_face",
  // "b0y"/"b0y_face", "b0z"/"b0z_cell". Requires the background to be enabled
  // (b0_ allocated); throws std::invalid_argument on a size mismatch or unknown
  // component, and std::logic_error if the background is inactive. Components
  // may be staged in any order; the completed field is checked for finite,
  // discretely solenoidal samples before the next solver operation consumes it.
  void seed_background(std::string_view component, const std::vector<Real>& host_buf);

  // True iff the field-split background B0 is enabled (cfg_.background.enabled).
  bool has_background() const noexcept;

  // One SSP-RK3 step; rejects a dt above cfl_limit() (an over-CFL step diverges).
  void step(Real dt);
  // One SSP-RK3 step WITHOUT the CFL re-check. For the auto-dt run loop, which
  // obtains dt from cfl_limit() immediately before stepping: step() would
  // recompute the identical full-grid device reduction inside check_cfl(), so
  // the caller that already holds a freshly-computed limit uses this to avoid
  // the redundant reduction. dt must still be positive.
  void step_unchecked(Real dt);
  // Loop step() to t_end, CFL-checked each step.
  void advance(Real t_end, Real dt);

  // Additive finite-volume Courant limit from the four incident reconstructed
  // faces. Configured HLLD uses alpha=max_side|v_n|+max_side(c_fast,n), matching
  // its common outer fast speed; the piecewise-constant LF retry uses its own
  // alpha=max_side(|v_n|+c_fast,n). Cylindrical applies the matching annular,
  // angular-momentum, and metric-free radial self-weights.
  Real cfl_limit() const;

  // Max |div B| over the interior; delegates to the CT scheme diagnostic.
  Real divergence_b_max() const;

  // Read a state component back to host (sized grid.storage_size()). The "bx"/
  // "by" spellings use the same face-to-cell collocation as the EOS: Cartesian
  // and axial rows are selected from the working halo, while cylindrical
  // radial bx uses the reconstruction-order row in radial_tables_.
  // "bx_face"/"by_face" return the raw CT arrays and "bz" is already
  // cell-centred.
  std::vector<Real> state_component_to_host(std::string_view component) const;

  // -- SSP-RK integrator seam (the integrator only calls these) ---------------
  // dudt := L(u): the conservative residual (-div F + geometric source).
  void compute_residual(const MhdField2D<Real>& u, MhdField2D<Real>& dudt);
  // Apply the Shu-Osher combine (including CT face-B rate) and verify positivity
  // for `stage`; an inadmissible candidate triggers a conservative retry.
  void combine_stage(int stage, Real dt);
  MhdField2D<Real>& rk_register(int k);
  MhdField2D<Real>& residual_register() noexcept;
  int n_rk_registers() const noexcept { return kNumRkRegisters; }

 private:
  friend class ::quasar::distributed::MhdTileAccess;

  struct PositivityRetry {
    Real theta{Real{0}};
  };

  // Map a component spelling to its DeviceBuffer in a field (throws on unknown).
  static backend::DeviceBuffer<Real>& component_buffer(MhdField2D<Real>& f,
                                                       std::string_view component);
  void check_cfl(Real dt) const;
  Real cfl_limit_for_collocation(int collocation_order) const;
  void ensure_background_solenoidal() const;
  void ensure_live_state_solenoidal() const;
  void ensure_live_state_admissible(int collocation_order = 0) const;
  bool is_cylindrical() const noexcept { return cfg_.geometry == "cylindrical"; }
  void fill_ghosts(MhdField2D<Real>& u) const;
  void copy_state(const MhdField2D<Real>& src, MhdField2D<Real>& dst);
  // Distributed stepping stops after derived-EMF preparation to exchange the
  // cell/face tables. It also stops after the HLLD face records are produced,
  // before any flux divergence consumes them, so the runtime can broadcast
  // one canonical complete record at every shared face. The ordinary serial
  // residual calls these phases back-to-back.
  void prepare_residual_face_records(const MhdField2D<Real>& u,
                                     MhdField2D<Real>& dudt,
                                     FaceOwnershipFlags4 ownership = {});
  void consume_residual_face_records(const MhdField2D<Real>& u,
                                     MhdField2D<Real>& dudt);
  void compute_residual_flux_and_emf(const MhdField2D<Real>& u,
                                     MhdField2D<Real>& dudt);
  void finish_ct_emf();
  void compute_ct_rate_from_emf(MhdField2D<Real>& dudt);
  void finish_split_energy(MhdField2D<Real>& dudt);
  // The distributed driver must reconcile the just-written face fields before
  // positivity samples the high-order cell-collocated magnetic field.  Keep the
  // ordinary serial seam as one call, but expose its two internal phases to the
  // friend runtime so an internal-tile halo exchange can sit between them.
  int apply_stage_update(int stage, Real dt);
  Real stage_admissible_fraction(int stage);
  Real combine_stage_fraction(int stage, Real dt);
  void advance_positive(Real dt);
  void note_external_mutable_state_access() noexcept;
  int reconstruction_order() const;
  // Per-side one-sided boundary flags ([x_lo, x_hi, y_lo, y_hi]) derived from
  // cfg_.boundary.field via boundary::mhd_boundary_is_periodic (0 = periodic,
  // 1 = non-periodic). All-zero on an all-periodic deck (fast path).
  BoundaryFlags4 boundary_flags() const;

  static constexpr int kNumRkRegisters = 3;  // live state + 2 SSP-RK3 stage regs

  MhdConfig cfg_{};
  // The reconstruction scheme is declared (and built) FIRST: its required_nghost()
  // fixes the working grid, which in turn sizes every field/register below. C++
  // initializes members in declaration order, so this must precede grid_.
  std::unique_ptr<numerics::IFluxReconstruction> reconstruction_{};
  Grid2D grid_{};
  // Radius-dependent coefficient ownership is created only for cylindrical
  // geometry and only after grid_ carries the reconstruction-resolved halo.
  // Its view is immutable for the lifetime of the solver.
  numerics::RadialTables radial_tables_{};
  // rk_[0] is the live state U; rk_[1], rk_[2] are the SSP-RK3 stage registers.
  std::array<MhdField2D<Real>, kNumRkRegisters> rk_{};
  // Snapshot of U^n for conservative positivity retries. A rejected SSP-RK
  // candidate is rolled back exactly, then retried with a smaller substep.
  MhdField2D<Real> step_backup_{};
  // Snapshot of the complete public step request. Internal positivity
  // substeps may commit one by one, but any later exception restores this
  // interval-start state so a failed step never advances unreported time.
  MhdField2D<Real> request_backup_{};
  MhdField2D<Real> residual_{};  // dudt accumulator (the L(u) register)
  // Retain both directional fluxes through the CT phase so an active B0 can
  // assemble its complete energy invariant in one common-exponent reduction.
  MhdField2D<Real> flux_x_{};
  MhdField2D<Real> flux_y_{};
  // Active-background momentum fluxes remain decomposed as material stress H
  // (in flux_x_/flux_y_), Riemann wave correction W, and a factored effective
  // perturbation field for C(B0,b) until the two-direction fused divergence.
  MhdMomentumFluxParts2D<Real> momentum_flux_x_{};
  MhdMomentumFluxParts2D<Real> momentum_flux_y_{};
  numerics::MhdInterfaceStates<Real> ifx_;  // dir=0 reconstructed interface states
  numerics::MhdInterfaceStates<Real> ify_;  // dir=1 reconstructed interface states

  // Reuse of the CFL reconstruction by the first residual stage.
  //
  // cfl_limit() reconstructs both directions of the live register to obtain one
  // scalar dt, and the auto-dt loop then immediately calls compute_residual on
  // that same unchanged register, which reconstructs it again into the same two
  // buffers. The two launches are the same computation: the only parameter that
  // differs is rate_only, which selects reconstructed_rate_state_admissible --
  // a direct forward to reconstructed_state_admissible. So the second launch
  // can be skipped when the buffers still describe the state it would read.
  //
  // Validity is tracked with a monotonic generation counter rather than a bare
  // bool: EVERY write to any register (combine_stage, copy_state and its
  // rollbacks, seed_state, floors, ghost refills) bumps state_generation_, so a
  // cache entry is only honoured when its recorded generation, register address,
  // and reconstruction order all still match. Anything not explicitly accounted
  // for invalidates by default, because a missed bump is the one failure mode
  // that would silently corrupt a run.
  std::uint64_t state_generation_{1};
  const MhdField2D<Real>* interface_cache_source_{nullptr};
  std::uint64_t interface_cache_generation_{0};
  int interface_cache_order_{-1};

  // Called by every state mutation. Invalidating on ghost refills too is
  // deliberate: reconstruction reads ghost cells, so a refill changes its input
  // even when no interior cell moved.
  void invalidate_interface_cache() noexcept {
    ++state_generation_;
    interface_cache_source_ = nullptr;
    interface_cache_order_ = -1;
  }

  // True when ifx_/ify_ already hold the reconstruction of `u` at `order`.
  bool interface_cache_valid(const MhdField2D<Real>& u, int order) const noexcept {
    return interface_cache_source_ == &u
        && interface_cache_generation_ == state_generation_
        && interface_cache_order_ == order;
  }

  void note_interface_cache(const MhdField2D<Real>& u, int order) noexcept {
    interface_cache_source_ = &u;
    interface_cache_generation_ = state_generation_;
    interface_cache_order_ = order;
  }
  EmfField2D<Real> emf_{};       // corner-staggered CT EMF
  // Static background field B0 (field split B = B0 + b). active iff the deck
  // enabled the background; inactive (default) => zero B0 fast path.
  MhdBackgroundField<Real> b0_{};
  // Solver-owned block-partials scratch for the CFL max-rate device reduction,
  // reused across steps so cfl_limit() does not hipMalloc/hipFree per call.
  // mutable because cfl_limit() is const (it only reads the state).
  mutable backend::DeviceBuffer<Real> cfl_scratch_{};
  mutable backend::DeviceBuffer<Real> divb_scratch_{};
  // A successfully accepted internal CT/RK update has stronger provenance than
  // an arbitrary seed: its exact-arithmetic divergence is inherited from the
  // preflighted input, so the live check may account for independent face-
  // storage rounding even when one directional contribution rounds to zero.
  // The privilege is never inferred from values. Any public mutable view is
  // conservatively treated as retainable and permanently disables it.
  bool live_state_solver_owned_{false};
  bool external_mutable_state_exposed_{false};
  bool internal_integrator_access_{false};
  // 0 selects the configured high-order reconstruction; 1 is the conservative
  // piecewise-constant retry path for a positivity-troubled substep.
  int positivity_reconstruction_order_{0};
  // True only while advance_positive() owns a rollback snapshot. Direct calls
  // through the public integrator registry exercise the RK method without the
  // solver-level retry controller and therefore must not compare against an
  // uninitialized step_backup_.
  bool positivity_control_active_{false};
  // True when the rollback base is admissible under adjacent-face magnetic
  // collocation. While true, a completed high-order substep must preserve that
  // fallback domain so a later rejection always has a valid low-order anchor.
  bool positivity_low_order_anchor_available_{false};
  int last_positivity_substeps_{0};
  // Analytic constructor samples are validated once. Explicit component seeds
  // invalidate that proof; the next operation that consumes or diagnoses B0
  // validates the complete three-component field after callers have staged all
  // components.
  mutable bool background_validated_{true};

  std::unique_ptr<numerics::IRiemannSolver> riemann_{};
  std::unique_ptr<numerics::ICtScheme> ct_{};
  std::unique_ptr<numerics::ISsprkIntegrator> integrator_{};
  std::unique_ptr<numerics::IPositivityLimiter> positivity_{};
  std::array<std::unique_ptr<boundary::IMhdFluidBoundary>, 4> fluid_bcs_{};
  std::array<std::unique_ptr<boundary::IMhdFieldBoundary>, 4> field_bcs_{};
};

}  // namespace quasar::mhd
