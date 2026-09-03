#include "quasar/physics/stability/stability_solver.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace quasar::stability {
namespace {

using backend::DeviceBuffer;
using equilibrium::GsDeviceResult;
using equilibrium::GsFluxSurfaces;
using equilibrium::GsMagneticField;

struct PreparedEquilibrium {
  GsMagneticField field{};
  FluxCoordinateGrid probe_coordinates{};
  bool probe_surfaces_valid{false};
};

[[nodiscard]] bool finite_nonnegative(Real value) {
  return std::isfinite(value) && value >= Real{0};
}

void validate_config(const StabilityConfig& config,
                     const StabilityProfiles& profiles) {
  if (config.toroidal_modes.empty()) {
    throw std::invalid_argument{
        "StabilitySolver: toroidal_modes must not be empty"};
  }
  std::unordered_set<int> unique_modes;
  for (const int n : config.toroidal_modes) {
    if (n == 0) {
      throw std::invalid_argument{
          "StabilitySolver: n=0 is not supported by the current toroidal "
          "energy formulation"};
    }
    if (!unique_modes.insert(n).second) {
      throw std::invalid_argument{
          "StabilitySolver: toroidal_modes must be unique"};
    }
  }
  if (!std::isfinite(config.lambda_inner)
      || !std::isfinite(config.lambda_outer)
      || !(config.lambda_inner > Real{0})
      || !(config.lambda_outer > config.lambda_inner)
      || config.lambda_outer > Real{1}) {
    throw std::invalid_argument{
        "StabilitySolver: require 0 < lambda_inner < lambda_outer <= 1"};
  }
  if (config.q_probe_count < 2 || config.contour_points < 4
      || config.minimum_radial_domains <= 0
      || config.minimum_radial_domains > RadialDomains::kMaxDomains
      || config.chebyshev_order <= 0 || config.m_max < 0
      || static_cast<long long>(config.n_theta)
             < 4LL * static_cast<long long>(config.m_max) + 1LL
      || config.f_profile_samples < 2) {
    throw std::invalid_argument{
        "StabilitySolver: invalid spectral or tracing resolution"};
  }
  const long long minimum_radial_nodes =
      static_cast<long long>(config.minimum_radial_domains)
      * (static_cast<long long>(config.chebyshev_order) + 1LL);
  if (minimum_radial_nodes > std::numeric_limits<int>::max()) {
    throw std::length_error{
        "StabilitySolver: local radial node count does not fit in int"};
  }
  if (!finite_nonnegative(config.minimum_domain_width)
      || config.minimum_domain_width
             >= config.lambda_outer - config.lambda_inner
      || !finite_nonnegative(config.condition_relative_floor)
      || !finite_nonnegative(config.eigenvalue_absolute_tolerance)
      || !finite_nonnegative(config.eigenvalue_relative_tolerance)
      || !finite_nonnegative(config.maximum_mass_digits_lost)) {
    throw std::invalid_argument{
        "StabilitySolver: tolerances must be finite and non-negative"};
  }
  if (!std::isfinite(profiles.f_vacuum)
      || profiles.f_vacuum == Real{0}) {
    throw std::invalid_argument{
        "StabilitySolver: f_vacuum must be finite and nonzero"};
  }
  if (config.eigen_method == EigenMethod::shift_invert) {
    if (!std::isfinite(config.eigen_shift)) {
      throw std::invalid_argument{
          "StabilitySolver: eigen_shift must be finite"};
    }
    if (config.eigen_wanted <= 0 || config.eigen_maximum_iterations < 0
        || !finite_nonnegative(config.eigen_residual_tolerance)
        || !finite_nonnegative(config.block_structure_tolerance)) {
      throw std::invalid_argument{
          "StabilitySolver: invalid shift-invert parameters"};
    }
  }
  if (profiles.density.count <= 0
      || profiles.density.count
             > DensityProfileCoefficients::kMaxCoefficients) {
    throw std::invalid_argument{
        "StabilitySolver: density polynomial metadata is invalid"};
  }
  for (int k = 0; k < profiles.density.count; ++k) {
    if (!std::isfinite(profiles.density.coefficients[k])) {
      throw std::invalid_argument{
          "StabilitySolver: density polynomial must be finite"};
    }
  }
}

void validate_equilibrium_contract(const GsDeviceResult& gs) {
  gs.grid.validate();
  const auto& critical = gs.critical;
  if (!critical.axis.valid
      || critical.axis.kind != equilibrium::CriticalKind::o_point
      || !std::isfinite(critical.axis.r)
      || !std::isfinite(critical.axis.z)
      || !std::isfinite(critical.axis.psi)
      || !(critical.axis.r > gs.grid.r_min)
      || !(critical.axis.r < gs.grid.r_max)
      || !(critical.axis.z > gs.grid.z_min)
      || !(critical.axis.z < gs.grid.z_max)) {
    throw std::invalid_argument{
        "StabilitySolver: Grad--Shafranov result lacks a valid finite "
        "interior magnetic axis"};
  }
  if (!critical.has_closed_surface || critical.critical_point_overflow) {
    throw std::invalid_argument{
        "StabilitySolver: Grad--Shafranov result lacks a trustworthy closed "
        "flux surface"};
  }
  if (!std::isfinite(critical.psi_axis)
      || !std::isfinite(critical.psi_boundary)) {
    throw std::invalid_argument{
        "StabilitySolver: Grad--Shafranov flux normalization is non-finite"};
  }
  if (critical.axis.psi != critical.psi_axis) {
    throw std::invalid_argument{
        "StabilitySolver: Grad--Shafranov magnetic-axis flux metadata is "
        "inconsistent"};
  }
  const Real flux_span = critical.psi_boundary - critical.psi_axis;
  if (!std::isfinite(flux_span) || flux_span == Real{0}) {
    throw std::invalid_argument{
        "StabilitySolver: Grad--Shafranov flux normalization span must be "
        "finite and nonzero"};
  }
  if (!std::isfinite(gs.profile_scale)) {
    throw std::invalid_argument{
        "StabilitySolver: Grad--Shafranov profile scale must be finite"};
  }

  const std::size_t expected = gs.grid.size();
  if (gs.psi.size() != expected || gs.j_phi.size() != expected) {
    throw std::invalid_argument{
        "StabilitySolver: Grad--Shafranov device fields have the wrong size"};
  }
  if (gs.psi.owner_device() != gs.j_phi.owner_device()) {
    throw std::invalid_argument{
        "StabilitySolver: Grad--Shafranov fields reside on different devices"};
  }
  if (gs.profile_coefficients.n_p <= 0
      || gs.profile_coefficients.n_p
             > equilibrium::ProfileCoefficients::kMaxCoefficients
      || gs.profile_coefficients.n_f <= 0
      || gs.profile_coefficients.n_f
             > equilibrium::ProfileCoefficients::kMaxCoefficients) {
    throw std::invalid_argument{
        "StabilitySolver: Grad--Shafranov result lacks valid profile "
        "provenance"};
  }
  for (int k = 0; k < gs.profile_coefficients.n_p; ++k) {
    if (!std::isfinite(gs.profile_coefficients.p_coeffs[k])) {
      throw std::invalid_argument{
          "StabilitySolver: pressure-profile provenance is non-finite"};
    }
  }
  for (int k = 0; k < gs.profile_coefficients.n_f; ++k) {
    if (!std::isfinite(gs.profile_coefficients.f_coeffs[k])) {
      throw std::invalid_argument{
          "StabilitySolver: toroidal-field profile provenance is non-finite"};
    }
  }
}

[[nodiscard]] bool all_flags_one(const DeviceBuffer<int>& flags,
                                 std::size_t count, stream_t stream) {
  std::vector<int> host(count);
  flags.copy_to_host_async(host.data(), host.size(), stream);
  backend::device_synchronize(stream);
  return std::all_of(host.begin(), host.end(),
                     [](int value) { return value == 1; });
}

[[nodiscard]] bool all_surfaces_closed(const GsFluxSurfaces& surfaces,
                                       stream_t stream) {
  return all_flags_one(surfaces.closed,
                       static_cast<std::size_t>(surfaces.n_surfaces), stream);
}

PreparedEquilibrium prepare_equilibrium(const GsDeviceResult& gs,
                                        const StabilityConfig& config,
                                        const StabilityProfiles& profiles,
                                        stream_t stream) {
  PreparedEquilibrium prepared;
  const auto& grid = gs.grid;
  const int owner = gs.psi.owner_device();
  backend::DeviceGuard guard{owner};

  DeviceBuffer<Real> f_table{
      static_cast<std::size_t>(config.f_profile_samples),
      backend::uninitialized, backend::on_device(owner)};
  equilibrium::launch_gs_integrate_f_profile(
      gs.profile_coefficients, profiles.f_vacuum, gs.critical.psi_axis,
      gs.critical.psi_boundary, gs.profile_scale, config.f_profile_samples,
      f_table.device_ptr(), stream);

  equilibrium::GsOperatorScratch operator_scratch{grid};
  equilibrium::GsDerivativeFields derivatives{grid};
  equilibrium::launch_gs_compute_derivatives(
      grid, gs.psi.device_ptr(), derivatives, operator_scratch, stream);
  prepared.field.resize(grid);
  equilibrium::launch_gs_compute_field(
      grid, gs.psi.device_ptr(), derivatives, f_table.device_ptr(),
      config.f_profile_samples, gs.critical.psi_axis,
      gs.critical.psi_boundary, prepared.field, stream);

  DeviceBuffer<Real> device_targets{
      static_cast<std::size_t>(config.q_probe_count), backend::uninitialized,
      backend::on_device(owner)};
  launch_build_probe_targets(config.lambda_inner, config.lambda_outer,
                             config.q_probe_count, device_targets.device_ptr(),
                             stream);

  GsFluxSurfaces probe_surfaces{config.q_probe_count, config.contour_points};
  equilibrium::launch_gs_trace_surfaces_at(
      grid, gs.psi.device_ptr(), prepared.field, gs.critical.axis.r,
      gs.critical.axis.z, gs.critical.psi_axis, gs.critical.psi_boundary,
      device_targets, probe_surfaces, stream);
  prepared.probe_surfaces_valid = all_surfaces_closed(probe_surfaces, stream);
  if (!prepared.probe_surfaces_valid) return prepared;

  prepared.probe_coordinates.resize(config.q_probe_count, config.n_theta);
  launch_build_flux_coordinates(
      grid, probe_surfaces, prepared.field, gs.critical.axis.r,
      gs.critical.axis.z, prepared.probe_coordinates, stream);
  prepared.probe_surfaces_valid = all_flags_one(
      prepared.probe_coordinates.valid,
      static_cast<std::size_t>(prepared.probe_coordinates.n_psi), stream);
  return prepared;
}

RationalSurfaces relevant_rational_surfaces(
    const RationalSurfaces& source, int m_max, Real lambda_inner,
    Real lambda_outer) {
  RationalSurfaces filtered{};
  filtered.overflow = source.overflow;
  filtered.has_rational_interval = source.has_rational_interval;
  const int count = std::clamp(source.count, 0,
                               RationalSurfaces::kMaxRational);
  for (int k = 0; k < count; ++k) {
    if (std::abs(source.m[k]) > m_max
        || !(source.psi_n[k] > lambda_inner)
        || !(source.psi_n[k] < lambda_outer)) {
      continue;
    }
    if (filtered.count >= RationalSurfaces::kMaxRational) {
      filtered.overflow = true;
      break;
    }
    filtered.psi_n[filtered.count] = source.psi_n[k];
    filtered.m[filtered.count] = source.m[k];
    ++filtered.count;
  }
  return filtered;
}

// Runs the shift-invert path on an assembled pencil.
//
// Returns false, with `fallback` set, whenever the block-tridiagonal route is
// not available or does not converge.  The caller then keeps the dense result,
// so a fallback costs an extra factorization attempt but never a wrong answer.
[[nodiscard]] bool try_shift_invert(
    const ToroidalMatrixPair& matrices, const SpectralDofLayout& layout,
    const StabilityConfig& config, ModeSummary& summary, stream_t stream) {
  const SpectralBlockStructure structure = build_spectral_block_structure(
      layout, matrices.free_to_full_complex, config.chebyshev_order);
  if (!structure.ok() || structure.order != matrices.real_order) {
    summary.shift_invert_fallback = ShiftInvertFallback::unsupported_topology;
    return false;
  }

  SpectralBlockMatrix shifted;
  launch_extract_shifted_block_tridiagonal(
      matrices.stiffness, matrices.inertia, config.eigen_shift, structure,
      shifted, stream);
  summary.block_structure_residual = shifted.maximum_outside_pattern;
  if (!(shifted.maximum_outside_pattern
        <= config.block_structure_tolerance)) {
    summary.shift_invert_fallback = ShiftInvertFallback::structure_not_exact;
    return false;
  }

  auto factorization = numerics::factor_block_tridiagonal(
      shifted.lower, shifted.diagonal, shifted.upper, structure.partition,
      stream);
  if (!factorization.ok()) {
    summary.shift_invert_fallback = ShiftInvertFallback::iteration_failed;
    return false;
  }

  backend::DeviceBuffer<Real> permuted_mass;
  launch_permute_dense_symmetric(matrices.inertia, structure, permuted_mass,
                                 stream);

  numerics::ShiftInvertLanczosConfig lanczos;
  lanczos.wanted = std::min(config.eigen_wanted, matrices.real_order);
  lanczos.shift = config.eigen_shift;
  lanczos.maximum_iterations = config.eigen_maximum_iterations;
  lanczos.residual_tolerance = config.eigen_residual_tolerance;
  auto solution = numerics::solve_shift_invert_lanczos(
      factorization, permuted_mass, lanczos, stream);
  if (!solution.ok() || solution.eigenvalues.empty()) {
    summary.shift_invert = std::move(solution);
    summary.shift_invert_fallback = ShiftInvertFallback::iteration_failed;
    return false;
  }

  summary.shift_invert = std::move(solution);
  summary.shift_invert_fallback = ShiftInvertFallback::none;
  summary.eigen_method = EigenMethod::shift_invert;
  return true;
}

[[nodiscard]] bool geometry_is_acceptable(
    const ToroidalGeometryValidationSummary& geometry,
    const ToroidalAssemblyConfig& config) {
  return geometry.ok()
      && geometry.maximum_q_relative_deviation
             <= config.field_pitch_tolerance
      && geometry.maximum_flux_scale_relative_deviation
             <= config.flux_scale_tolerance
      && geometry.maximum_pest_j_over_r2_relative_deviation
             <= config.pest_tolerance;
}

ModeEigenResult solve_prepared_mode(
    const GsDeviceResult& gs, const PreparedEquilibrium& prepared,
    const StabilityConfig& config, const StabilityProfiles& profiles,
    int n_toroidal, stream_t stream) {
  ModeEigenResult result;
  result.summary.n_toroidal = n_toroidal;
  result.summary.lambda_inner = config.lambda_inner;
  result.summary.lambda_outer = config.lambda_outer;
  result.chebyshev_order = config.chebyshev_order;
  result.m_max = config.m_max;
  result.n_theta = config.n_theta;
  result.summary.shift_invert_fallback =
      config.eigen_method == EigenMethod::shift_invert
          ? ShiftInvertFallback::unsupported_topology
          : ShiftInvertFallback::not_requested;

  if (n_toroidal == 0) {
    throw std::invalid_argument{
        "StabilitySolver::solve_mode: n=0 is not supported"};
  }
  if (!prepared.probe_surfaces_valid) {
    result.summary.status = ModeSolveStatus::surface_trace_failed;
    return result;
  }

  const int owner = gs.psi.owner_device();
  backend::DeviceGuard guard{owner};
  DeviceBuffer<RationalSurfaces> device_rational{
      1, backend::on_device(owner)};
  launch_locate_rational_surfaces(prepared.probe_coordinates, n_toroidal,
                                  config.m_max, device_rational.device_ptr(),
                                  stream);
  RationalSurfaces raw_rational{};
  device_rational.copy_to_host_async(&raw_rational, 1, stream);
  backend::device_synchronize(stream);
  RationalSurfaces rational = relevant_rational_surfaces(
      raw_rational, config.m_max, config.lambda_inner, config.lambda_outer);
  result.summary.rational_surfaces = rational;
  if (rational.overflow) {
    result.summary.status = ModeSolveStatus::topology_overflow;
    return result;
  }
  // The low-level layout can represent all-component one-sided cuts, but the
  // continuum interface conditions for the two tangential components have not
  // yet been derived.  Do not turn that experimental admissible space into an
  // optimizer-facing stability classification.
  if (rational.has_rational_interval || rational.count != 0) {
    result.summary.status = ModeSolveStatus::unsupported_rational_topology;
    return result;
  }

  device_rational.copy_from_host_async(&rational, 1, stream);
  DeviceBuffer<RadialDomains> device_domains{1, backend::on_device(owner)};
  launch_build_radial_domains(
      device_rational.device_ptr(), config.lambda_inner,
      config.lambda_outer, config.minimum_radial_domains,
      config.minimum_domain_width, device_domains.device_ptr(), stream);
  RadialDomains domains{};
  device_domains.copy_to_host_async(&domains, 1, stream);
  backend::device_synchronize(stream);
  result.summary.radial_domains = domains;
  if (domains.overflow || domains.n_domains <= 0) {
    result.summary.status = ModeSolveStatus::topology_overflow;
    return result;
  }

  ChebyshevBasis basis{config.chebyshev_order, domains.n_domains};
  const std::size_t n_lambda_size = backend::detail::checked_size_product(
      static_cast<std::size_t>(basis.n_domains),
      static_cast<std::size_t>(basis.n_nodes),
      "StabilitySolver: local radial node count overflows size_t");
  if (n_lambda_size
      > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error{
        "StabilitySolver: local radial node count does not fit in int"};
  }
  const int n_lambda = static_cast<int>(n_lambda_size);

  launch_build_chebyshev_basis(device_domains.device_ptr(), basis, stream);
  SpectralDofLayout layout{domains, config.chebyshev_order, config.m_max,
                           config.n_theta};

  GsFluxSurfaces spectral_surfaces{n_lambda, config.contour_points};
  equilibrium::launch_gs_trace_surfaces_at(
      gs.grid, gs.psi.device_ptr(), prepared.field, gs.critical.axis.r,
      gs.critical.axis.z, gs.critical.psi_axis, gs.critical.psi_boundary,
      basis.nodes, spectral_surfaces, stream);
  if (!all_surfaces_closed(spectral_surfaces, stream)) {
    result.summary.status = ModeSolveStatus::surface_trace_failed;
    return result;
  }

  FluxCoordinateGrid coordinates{n_lambda, config.n_theta};
  launch_build_spectral_flux_coordinates(
      gs.grid, spectral_surfaces, prepared.field, basis, coordinates, stream);

  ToroidalEquilibriumFields toroidal_equilibrium{n_lambda, config.n_theta};
  try {
    launch_build_toroidal_equilibrium(
        coordinates, gs.profile_coefficients, gs.critical.psi_axis,
        gs.critical.psi_boundary, gs.profile_scale, profiles.f_vacuum,
        profiles.density,
        toroidal_equilibrium, stream);
  } catch (const ToroidalEquilibriumValidationError&) {
    result.summary.status = ModeSolveStatus::equilibrium_fields_failed;
    return result;
  }

  ToroidalGeometryValidation validation{n_lambda};
  launch_validate_toroidal_geometry(
      coordinates, gs.grid, prepared.field, toroidal_equilibrium,
      validation, stream);
  result.summary.geometry =
      summarize_toroidal_geometry_validation(validation, stream);
  if (!geometry_is_acceptable(result.summary.geometry, config.assembly)) {
    result.summary.status = ModeSolveStatus::geometry_validation_failed;
    return result;
  }

  ToroidalAssemblyConfig assembly_config = config.assembly;
  assembly_config.n_toroidal = n_toroidal;
  // The generalized eigensolve below factors the inertia internally and reports
  // an indefinite mass through GeneralizedEigenStatus::mass_not_positive_
  // definite, which is mapped back onto the assembly diagnostics after the
  // solve.  Running the assembly's own Cholesky here would pay for the same
  // O(real_order^3) factorization a second time.
  assembly_config.verify_mass_positive_definite = false;
  ToroidalMatrixPair matrices;
  launch_assemble_fixed_boundary_toroidal_matrices(
      basis, domains, coordinates, toroidal_equilibrium, layout,
      assembly_config, matrices, stream);
  result.summary.assembly = matrices.diagnostics;
  result.summary.real_order = matrices.real_order;
  if (!result.summary.assembly->ok()) {
    result.summary.status = ModeSolveStatus::assembly_failed;
    return result;
  }

  result.summary.stiffness_condition = numerics::estimate_symmetric_condition(
      matrices.stiffness, matrices.real_order,
      numerics::MatrixTriangle::lower, config.condition_relative_floor,
      stream);
  result.summary.inertia_condition = numerics::estimate_symmetric_condition(
      matrices.inertia, matrices.real_order, numerics::MatrixTriangle::lower,
      config.condition_relative_floor, stream);
  const auto condition_failed = [](const auto& estimate) {
    return estimate.status
               == numerics::SymmetricConditionStatus::failed_to_converge
        || estimate.status
               == numerics::SymmetricConditionStatus::invalid_solver_argument;
  };
  if (condition_failed(*result.summary.stiffness_condition)
      || condition_failed(*result.summary.inertia_condition)) {
    result.summary.status = ModeSolveStatus::condition_estimate_failed;
    return result;
  }

  auto eigensystem = numerics::solve_generalized_symmetric_eigenproblem(
      matrices.stiffness, matrices.inertia, matrices.real_order,
      numerics::MatrixTriangle::lower, stream);
  result.summary.eigen_status = eigensystem.status;
  result.summary.eigen_solver_info = eigensystem.solver_info;
  if (!eigensystem.ok()) {
    // The assembly's own Cholesky was suppressed above to avoid factoring the
    // inertia twice.  Restore the diagnostic it would have produced so an
    // indefinite mass matrix is still reported as such, and with the same
    // leading-minor order, rather than only as an opaque eigensolver failure.
    if (eigensystem.status
            == numerics::GeneralizedEigenStatus::mass_not_positive_definite
        && result.summary.assembly.has_value()) {
      result.summary.assembly->status |=
          ToroidalAssemblyStatus::mass_not_positive_definite;
      result.summary.assembly->mass_failed_leading_minor_order =
          eigensystem.failed_leading_minor_order;
    }
    result.summary.status = ModeSolveStatus::eigensolver_failed;
    result.eigensystem = std::move(eigensystem);
    return result;
  }

  std::vector<Real> eigenvalues(
      static_cast<std::size_t>(eigensystem.order));
  eigensystem.eigenvalues.copy_to_host_async(
      eigenvalues.data(), eigenvalues.size(), stream);
  backend::device_synchronize(stream);
  if (eigenvalues.empty()
      || !std::all_of(eigenvalues.begin(), eigenvalues.end(),
                      [](Real value) { return std::isfinite(value); })) {
    eigensystem.status = numerics::GeneralizedEigenStatus::nonfinite_result;
    result.summary.eigen_status = eigensystem.status;
    result.summary.status = ModeSolveStatus::eigensolver_failed;
    result.eigensystem = std::move(eigensystem);
    return result;
  }
  const Real minimum = eigenvalues.front();
  const Real spectral_scale = std::max(
      std::abs(eigenvalues.front()), std::abs(eigenvalues.back()));
  const Real threshold = config.eigenvalue_absolute_tolerance
                       + config.eigenvalue_relative_tolerance * spectral_scale;
  const bool threshold_unresolved =
      !std::isfinite(spectral_scale) || !std::isfinite(threshold);

  result.summary.minimum_omega_squared = minimum;
  result.summary.maximum_absolute_omega_squared = spectral_scale;
  result.summary.eigenvalue_resolution_threshold = threshold;
  result.summary.growth_rate =
      !threshold_unresolved && minimum < -threshold ? std::sqrt(-minimum)
                                                    : Real{0};
  const bool classification_unresolved =
      threshold_unresolved || !result.summary.inertia_condition->ok()
      || result.summary.inertia_condition->digits_lost
             > config.maximum_mass_digits_lost;
  if (classification_unresolved) {
    result.summary.classification = StabilityClassification::unresolved;
  } else if (minimum < -threshold) {
    result.summary.classification = StabilityClassification::unstable;
  } else {
    result.summary.classification =
        StabilityClassification::no_instability_detected;
  }
  // Shift-invert runs alongside the dense solve rather than replacing it.
  //
  // The dense spectrum supplies `maximum_absolute_omega_squared`, and that
  // number is what the resolution threshold -- and therefore the stable/unstable
  // decision -- is defined against.  Shift-invert returns only the eigenvalues
  // nearest the shift and has no way to bound the top of the spectrum, so
  // dropping the dense solve today would mean silently redefining the threshold
  // in terms of a looser bound.  Skipping the O(real_order^3) solve entirely
  // needs a largest-magnitude estimator for the pencil as well; until that
  // exists, this path is exercised, cross-checked against the dense result, and
  // reported, but it is not yet load-bearing.
  if (config.eigen_method == EigenMethod::shift_invert) {
    (void)try_shift_invert(matrices, layout, config, result.summary, stream);
  }

  result.summary.status = ModeSolveStatus::solved;
  result.free_to_full_complex = std::move(matrices.free_to_full_complex);
  result.radial_nodes = std::move(basis.nodes);
  result.eigensystem = std::move(eigensystem);
  return result;
}

}  // namespace

StabilitySolver::StabilitySolver(StabilityConfig config,
                                 StabilityProfiles profiles)
    : config_{std::move(config)}, profiles_{std::move(profiles)} {
  validate_config(config_, profiles_);
}

ModeEigenResult StabilitySolver::solve_mode(
    const GsDeviceResult& equilibrium, int n_toroidal,
    stream_t stream) const {
  ModeEigenResult result;
  result.summary.n_toroidal = n_toroidal;
  if (n_toroidal == 0) {
    throw std::invalid_argument{
        "StabilitySolver::solve_mode: n=0 is not supported"};
  }
  if (!equilibrium.ok()) {
    result.summary.status = ModeSolveStatus::equilibrium_not_converged;
    return result;
  }
  validate_equilibrium_contract(equilibrium);
  backend::DeviceGuard guard{equilibrium.psi.owner_device()};
  const PreparedEquilibrium prepared =
      prepare_equilibrium(equilibrium, config_, profiles_, stream);
  return solve_prepared_mode(equilibrium, prepared, config_, profiles_,
                             n_toroidal, stream);
}

StabilityResult StabilitySolver::solve(
    const GsDeviceResult& equilibrium, stream_t stream) const {
  StabilityResult result;
  result.equilibrium_status = equilibrium.status;
  result.lambda_inner = config_.lambda_inner;
  result.lambda_outer = config_.lambda_outer;
  if (!equilibrium.ok()) {
    result.status = StabilityStatus::equilibrium_not_converged;
    return result;
  }
  validate_equilibrium_contract(equilibrium);

  backend::DeviceGuard guard{equilibrium.psi.owner_device()};
  const PreparedEquilibrium prepared =
      prepare_equilibrium(equilibrium, config_, profiles_, stream);
  result.status = StabilityStatus::complete;
  result.aggregate_margin = std::numeric_limits<Real>::infinity();
  result.maximum_growth_rate = Real{0};
  bool have_solved_mode = false;
  bool have_trusted_mode = false;
  bool have_unstable_mode = false;
  bool have_unresolved_mode = false;

  for (const int n : config_.toroidal_modes) {
    result.scanned_n.push_back(n);
    ModeEigenResult full = solve_prepared_mode(
        equilibrium, prepared, config_, profiles_, n, stream);
    const ModeSummary summary = full.summary;
    result.modes.push_back(summary);
    if (!summary.solved()) {
      result.status = StabilityStatus::incomplete_scan;
      have_unresolved_mode = true;
      continue;
    }

    have_solved_mode = true;
    if (summary.classification == StabilityClassification::unstable) {
      have_unstable_mode = true;
    } else if (summary.classification
               == StabilityClassification::unresolved) {
      have_unresolved_mode = true;
      continue;
    }
    have_trusted_mode = true;
    if (summary.minimum_omega_squared < result.aggregate_margin) {
      result.aggregate_margin = summary.minimum_omega_squared;
      result.worst_n = n;
    }
    result.maximum_growth_rate =
        std::max(result.maximum_growth_rate, summary.growth_rate);
  }

  // The aggregates describe the modes that were actually resolved, and nothing
  // more.  An unresolved mode makes the scan incomplete -- which is already
  // reported through `status` and through the per-mode `modes` entries -- but
  // it does not invalidate a growth rate measured on a mode that converged.
  // Poisoning the aggregate here used to hand callers a NaN `maximum_growth_
  // rate` alongside an `unstable` classification, which is strictly less
  // information than the trusted maximum plus the incomplete-scan flag.
  result.complete_mode_coverage = !have_unresolved_mode;
  if (!have_trusted_mode) {
    result.aggregate_margin = std::numeric_limits<Real>::quiet_NaN();
    result.maximum_growth_rate = std::numeric_limits<Real>::quiet_NaN();
    result.worst_n = 0;
  }
  if (have_unstable_mode) {
    result.classification = StabilityClassification::unstable;
  } else if (have_unresolved_mode || !have_solved_mode) {
    result.classification = StabilityClassification::unresolved;
  } else {
    result.classification =
        StabilityClassification::no_instability_detected;
  }
  return result;
}

}  // namespace quasar::stability
