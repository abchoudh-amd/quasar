#pragma once

// Fixed-boundary, non-axisymmetric ideal-MHD stability orchestration.
//
// This is the first end-to-end consumer of GsDeviceResult.  It keeps the
// equilibrium and all field/matrix arrays on the active HIP device while it
// traces the requested annular flux region, constructs the Chebyshev x Fourier
// discretization, assembles the plasma energy and inertia, and invokes the
// dense generalized eigensolver.  Only compact status/diagnostic values and
// the extremal eigenvalues are copied to the host.
//
// The currently supported domain excludes the magnetic axis and applies the
// fixed conducting condition xi^lambda=0 only at lambda_outer.  The truncated
// inner flux surface receives the natural weak-form condition.  This is an
// explicitly annular model; it must not be presented as a full-axis tokamak
// result until harmonic/component-dependent axis regularity is implemented.

#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/condition_estimator.hpp"
#include "quasar/numerics/generalized_eigensolver.hpp"
#include "quasar/numerics/shift_invert_lanczos.hpp"
#include "quasar/physics/equilibrium/gs_solver.hpp"
#include "quasar/physics/stability/kernels.hpp"
#include "quasar/physics/stability/spectral_blocks.hpp"
#include "quasar/physics/stability/toroidal_energy.hpp"
#include "quasar/physics/stability/toroidal_equilibrium.hpp"

#include <cstddef>
#include <limits>
#include <optional>
#include <vector>

namespace quasar::stability {

enum class StabilityClassification {
  unstable,
  no_instability_detected,
  unresolved,
};

enum class ModeSolveStatus {
  solved,
  equilibrium_not_converged,
  topology_overflow,
  unsupported_rational_topology,
  surface_trace_failed,
  equilibrium_fields_failed,
  geometry_validation_failed,
  assembly_failed,
  condition_estimate_failed,
  eigensolver_failed,
};

enum class StabilityStatus {
  complete,
  equilibrium_not_converged,
  incomplete_scan,
};

enum class RadialBoundaryModel {
  outer_fixed_inner_natural,
};

// How the assembled pencil K x = omega^2 M x is solved.
enum class EigenMethod {
  // Full spectrum via hipSOLVER sygvd: O(real_order^3), returns every
  // eigenvalue, and is the reference the shift-invert path is validated
  // against.  This is the default because the spectral extremes it reports are
  // what the classification and the resolution threshold are defined in terms
  // of.
  dense,

  // Shift-invert Lanczos on the block-tridiagonal factorization of
  // (K - shift*M).  Returns only the eigenvalues nearest `eigen_shift`, so it
  // is the right choice when a scan needs the growth rate of the worst mode and
  // not the whole spectrum.  Falls back to `dense` and records the reason in
  // `ModeSummary::shift_invert_fallback` when the discretization has no exact
  // block-tridiagonal structure -- currently any rational-surface topology.
  shift_invert,
};

// Why a requested shift-invert solve did not run.
enum class ShiftInvertFallback {
  not_requested,
  // The solve ran; no fallback occurred.
  none,
  // Harmonic-specific radial splits make the radial-major reordering
  // non-rectangular, so no exact block-tridiagonal partition exists.
  unsupported_topology,
  // The reordered matrix had nonzeros outside the block-tridiagonal pattern,
  // so extracting it would have silently discarded real couplings.
  structure_not_exact,
  // (K - shift*M) could not be factored, or the iteration did not converge.
  iteration_failed,
};

struct StabilityProfiles {
  DensityProfileCoefficients density{};
  Real f_vacuum{0};
};

struct StabilityConfig {
  std::vector<int> toroidal_modes{1};

  // The magnetic axis is deliberately excluded until its component- and
  // harmonic-specific regularity conditions are available.
  Real lambda_inner{Real{0.1}};
  Real lambda_outer{Real{0.9}};

  int q_probe_count{32};
  int contour_points{256};
  int minimum_radial_domains{2};
  int chebyshev_order{6};
  int m_max{4};
  int n_theta{32};
  int f_profile_samples{257};
  Real minimum_domain_width{Real{1e-3}};

  ToroidalAssemblyConfig assembly{};
  Real condition_relative_floor{0};
  Real eigenvalue_absolute_tolerance{0};
  Real eigenvalue_relative_tolerance{Real{1e-10}};
  Real maximum_mass_digits_lost{Real{12}};

  // Eigenvalue method and its shift-invert parameters.
  //
  // The shift must lie below the eigenvalue of interest for the lowest mode to
  // be the one nearest it; a shift inside the spectrum converges to whatever
  // surrounds it instead.  The default of a small negative value targets the
  // marginally unstable band, which is where the classification decision is
  // actually made.
  EigenMethod eigen_method{EigenMethod::dense};
  Real eigen_shift{Real{-1}};
  int eigen_wanted{4};
  int eigen_maximum_iterations{0};
  Real eigen_residual_tolerance{Real{1e-10}};

  // Largest magnitude tolerated outside the block-tridiagonal pattern before
  // the shift-invert path refuses to use it.  The couplings are structurally
  // zero, so the default admits nothing but exact zeros.
  Real block_structure_tolerance{0};
};

struct ModeSummary {
  ModeSolveStatus status{ModeSolveStatus::surface_trace_failed};
  StabilityClassification classification{
      StabilityClassification::unresolved};
  int n_toroidal{0};
  Real lambda_inner{0};
  Real lambda_outer{0};
  RadialBoundaryModel radial_boundary_model{
      RadialBoundaryModel::outer_fixed_inner_natural};
  RationalSurfaces rational_surfaces{};
  RadialDomains radial_domains{};
  int real_order{0};
  Real minimum_omega_squared{std::numeric_limits<Real>::quiet_NaN()};
  Real maximum_absolute_omega_squared{0};
  Real growth_rate{std::numeric_limits<Real>::quiet_NaN()};
  Real eigenvalue_resolution_threshold{
      std::numeric_limits<Real>::quiet_NaN()};
  ToroidalGeometryValidationSummary geometry{};
  std::optional<ToroidalAssemblyDiagnostics> assembly{};
  std::optional<numerics::SymmetricConditionEstimate> stiffness_condition{};
  std::optional<numerics::SymmetricConditionEstimate> inertia_condition{};
  std::optional<numerics::GeneralizedEigenStatus> eigen_status{};
  int eigen_solver_info{0};

  // Shift-invert provenance.  `shift_invert` is populated only when the
  // shift-invert path actually ran; `shift_invert_fallback` says why it did not
  // when the method was requested.  A fallback is not an error: the dense
  // result is returned and the classification is unaffected.
  EigenMethod eigen_method{EigenMethod::dense};
  ShiftInvertFallback shift_invert_fallback{
      ShiftInvertFallback::not_requested};
  std::optional<numerics::ShiftInvertLanczosResult> shift_invert{};
  Real block_structure_residual{0};

  [[nodiscard]] bool solved() const noexcept {
    return status == ModeSolveStatus::solved;
  }
};

struct ModeEigenResult {
  ModeSummary summary{};
  int chebyshev_order{0};
  int m_max{0};
  int n_theta{0};
  std::vector<std::size_t> free_to_full_complex{};
  backend::DeviceBuffer<Real> radial_nodes{};
  numerics::GeneralizedEigenResult eigensystem{};
};

struct StabilityResult {
  StabilityStatus status{StabilityStatus::incomplete_scan};
  equilibrium::GsStatus equilibrium_status{
      equilibrium::GsStatus::iteration_limit};
  std::vector<int> scanned_n{};
  std::vector<ModeSummary> modes{};
  StabilityClassification classification{
      StabilityClassification::unresolved};

  // The aggregates below are reduced over the TRUSTED modes only -- those with
  // `status == solved` and a classification other than `unresolved`.  They are
  // NaN only when no mode in the scan was trusted.
  //
  // `complete_mode_coverage` is false when at least one requested mode failed
  // to solve or came back unresolved.  A false value does not invalidate the
  // aggregates; it says the scan did not cover every requested n, so the true
  // worst case may lie in a mode that was not resolved.  Read it before quoting
  // `aggregate_margin` or `maximum_growth_rate` as a property of the
  // equilibrium rather than of the modes that converged.
  bool complete_mode_coverage{false};
  Real aggregate_margin{std::numeric_limits<Real>::quiet_NaN()};
  Real maximum_growth_rate{std::numeric_limits<Real>::quiet_NaN()};
  int worst_n{0};
  Real lambda_inner{0};
  Real lambda_outer{0};
  RadialBoundaryModel radial_boundary_model{
      RadialBoundaryModel::outer_fixed_inner_natural};

  [[nodiscard]] bool ok() const noexcept {
    return status == StabilityStatus::complete
        && classification != StabilityClassification::unresolved;
  }
};

class StabilitySolver {
 public:
  StabilitySolver(StabilityConfig config, StabilityProfiles profiles);

  // Solve one nonzero toroidal mode and retain its complete dense eigensystem.
  // A non-null stream must belong to the device that owns equilibrium.psi and
  // equilibrium.j_phi.
  [[nodiscard]] ModeEigenResult solve_mode(
      const equilibrium::GsDeviceResult& equilibrium, int n_toroidal,
      stream_t stream = nullptr) const;

  // Scan StabilityConfig::toroidal_modes in caller order.  Only compact mode
  // summaries are retained so dense eigenvector storage is released between
  // modes. A non-null stream has the same-device precondition as solve_mode().
  [[nodiscard]] StabilityResult solve(
      const equilibrium::GsDeviceResult& equilibrium,
      stream_t stream = nullptr) const;

  [[nodiscard]] const StabilityConfig& config() const noexcept {
    return config_;
  }
  [[nodiscard]] const StabilityProfiles& profiles() const noexcept {
    return profiles_;
  }

 private:
  StabilityConfig config_{};
  StabilityProfiles profiles_{};
};

}  // namespace quasar::stability
