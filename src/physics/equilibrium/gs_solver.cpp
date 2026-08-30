// Free-boundary Grad-Shafranov driver.
//
// This is the orchestration layer: it owns the device buffers and sequences the
// kernel launches in GsSolver::solve_device. The legacy GsSolver::solve entry
// point is only the explicit download adapter. There is no arithmetic here
// beyond the two scalar decisions the loop cannot avoid making on the host.
//
// The structure deliberately mirrors the host solver line for line rather than
// being reorganized around what would be natural for a GPU. During a
// hard-replace port, a restructured loop and a ported loop fail differently and
// only one of them is diagnosable: if the device result diverges, the question
// has to be "which kernel" and not "which of two algorithms".

#include "quasar/physics/equilibrium/gs_solver.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"
#include "quasar/numerics/gs_operator_l6.hpp"  // l6_is_applicable

#include <cmath>
#include <stdexcept>
#include <utility>

namespace quasar::equilibrium {

namespace {

using backend::DeviceBuffer;

// The host solver's seed-default rules, applied verbatim.
//
// Note the asymmetry between r_center and z_center: r_center falls back when it
// is non-positive, z_center when it is exactly zero. That is not a tidy rule,
// but it is the host's rule, and a "cleaner" version would move the seed on any
// deck that sets z_center to something the host would have overridden.
struct ResolvedSeed {
  Real r_center;
  Real z_center;
  Real minor_radius;
  Real depth;
  Real sign;
};

ResolvedSeed resolve_seed(const GsConfig& cfg) {
  const numerics::EllipticGrid& g = cfg.grid;
  ResolvedSeed s;
  s.r_center = cfg.seed.r_center > Real{0}
                   ? cfg.seed.r_center
                   : g.r_min + Real{0.5} * (g.r_max - g.r_min);
  s.z_center = cfg.seed.z_center != Real{0}
                   ? cfg.seed.z_center
                   : g.z_min + Real{0.5} * (g.z_max - g.z_min);
  s.minor_radius = cfg.seed.minor_radius > Real{0}
                       ? cfg.seed.minor_radius
                       : Real{0.25} * std::min(g.r_max - g.r_min,
                                               g.z_max - g.z_min);
  s.depth = cfg.seed.depth * kMu0 * std::abs(cfg.plasma_current);
  s.sign = cfg.plasma_current >= Real{0} ? Real{1} : Real{-1};
  return s;
}

bool valid_resolved_seed(const GsConfig& cfg, const ResolvedSeed& seed) {
  return std::isfinite(seed.r_center) && std::isfinite(seed.z_center)
      && std::isfinite(seed.minor_radius) && seed.minor_radius > Real{0}
      && std::isfinite(seed.depth) && seed.depth > Real{0}
      && std::isfinite(seed.sign)
      && seed.r_center > cfg.grid.r_min
      && seed.r_center < cfg.grid.r_max
      && seed.z_center > cfg.grid.z_min
      && seed.z_center < cfg.grid.z_max;
}

CriticalPointSet to_host_set(const GsCriticalResult& r) {
  CriticalPointSet out;
  out.axis = r.axis;
  out.psi_axis = r.psi_axis;
  out.psi_boundary = r.psi_boundary;
  out.has_closed_surface = r.has_closed_surface;
  out.critical_point_overflow = r.x_point_overflow;
  out.x_points.assign(r.x_points, r.x_points + r.n_x);
  return out;
}

void validate_config(const GsConfig& cfg) {
  if (!std::isfinite(cfg.plasma_current) || cfg.plasma_current == Real{0}) {
    throw std::invalid_argument{
        "GsSolver: plasma_current must be finite and nonzero"};
  }
  if (cfg.max_iterations <= 0) {
    throw std::invalid_argument{
        "GsSolver: max_iterations must be positive"};
  }
  if (!std::isfinite(cfg.tolerance) || !(cfg.tolerance > Real{0})) {
    throw std::invalid_argument{
        "GsSolver: tolerance must be finite and positive"};
  }
  if (!std::isfinite(cfg.picard_relaxation)
      || !(cfg.picard_relaxation > Real{0})
      || cfg.picard_relaxation > Real{1}) {
    throw std::invalid_argument{
        "GsSolver: picard_relaxation must be finite and in (0, 1]"};
  }
  if (!std::isfinite(cfg.newton_residual_threshold)
      || cfg.newton_residual_threshold < Real{0}
      || !std::isfinite(cfg.newton_geometry_tolerance)
      || cfg.newton_geometry_tolerance < Real{0}) {
    throw std::invalid_argument{
        "GsSolver: Newton thresholds must be finite and non-negative"};
  }

  const PlasmaSeed& seed = cfg.seed;
  if (!std::isfinite(seed.r_center) || !std::isfinite(seed.z_center)
      || !std::isfinite(seed.minor_radius) || !std::isfinite(seed.depth)
      || seed.r_center < Real{0} || seed.minor_radius < Real{0}
      || !(seed.depth > Real{0})) {
    throw std::invalid_argument{
        "GsSolver: plasma seed parameters are malformed"};
  }
  if (seed.r_center > Real{0}
      && !(seed.r_center > cfg.grid.r_min
           && seed.r_center < cfg.grid.r_max)) {
    throw std::invalid_argument{
        "GsSolver: explicit seed r_center must lie inside the grid"};
  }
  if (seed.z_center != Real{0}
      && !(seed.z_center > cfg.grid.z_min
           && seed.z_center < cfg.grid.z_max)) {
    throw std::invalid_argument{
        "GsSolver: explicit seed z_center must lie inside the grid"};
  }

  for (const CoilFilament& coil : cfg.coils) {
    if (!std::isfinite(coil.r) || !std::isfinite(coil.z)
        || !std::isfinite(coil.current) || !(coil.r > Real{0})) {
      throw std::invalid_argument{
          "GsSolver: coil coordinates/current must be finite and coil radius "
          "must be positive"};
    }
  }
}

}  // namespace

GsResult GsDeviceResult::copy_to_host(stream_t stream) const {
  if (psi.owner_device() != j_phi.owner_device()) {
    throw std::invalid_argument{
        "GsDeviceResult::copy_to_host: psi and j_phi reside on different "
        "devices"};
  }

  GsResult host;
  host.status = status;
  host.iterations = iterations;
  host.residual = residual;
  host.residual_history = residual_history;
  host.psi.resize(psi.size());
  host.j_phi.resize(j_phi.size());
  host.critical = critical;
  host.achieved_current = achieved_current;
  host.profile_scale = profile_scale;
  host.profile_coefficients = profile_coefficients;
  host.newton_steps = newton_steps;

  if (!psi.empty() || !j_phi.empty()) {
    const int owner = !psi.empty() ? psi.owner_device() : j_phi.owner_device();
    backend::DeviceGuard guard{owner};
    psi.copy_to_host_async(host.psi.data(), host.psi.size(), stream);
    j_phi.copy_to_host_async(host.j_phi.data(), host.j_phi.size(), stream);
    backend::device_synchronize(stream);
  }
  return host;
}

GsSolver::GsSolver(GsConfig cfg,
                               std::shared_ptr<IEquilibriumProfile> profile)
  : cfg_{std::move(cfg)} {
  cfg_.grid.validate();
  validate_config(cfg_);
  if (!profile) {
    throw std::invalid_argument{"GsSolver: profile must not be null"};
  }
  if (!numerics::l6_is_applicable(cfg_.grid)) {
    throw std::invalid_argument{
        "GsSolver: grid too small for the sixth-order operator"};
  }
  const auto* poly = dynamic_cast<const PolynomialProfile*>(profile.get());
  if (poly == nullptr) {
    throw std::invalid_argument{
        "GsSolver: only PolynomialProfile can be lowered to the device; "
        "a non-polynomial profile needs its own device evaluator"};
  }
  profile_ = to_coefficients(*poly);
}

GsResult GsSolver::solve() {
  return solve_device().copy_to_host();
}

GsDeviceResult GsSolver::solve_device(stream_t stream) {
  const numerics::EllipticGrid& g = cfg_.grid;
  const std::size_t n = g.size();

  GsDeviceResult res;
  res.grid = g;
  res.profile_coefficients = profile_;

  DeviceBuffer<Real> d_psi{n};
  DeviceBuffer<Real> d_psi_next{n};
  // Snapshot of the last state whose current, boundary, residual, and
  // critical-point metadata were all validated. d_psi_next is reused as the
  // next candidate and may become non-finite before that validation completes.
  DeviceBuffer<Real> d_psi_good{n};
  DeviceBuffer<Real> d_solved{n};
  DeviceBuffer<Real> d_j_phi{n};
  DeviceBuffer<Real> d_j_candidate{n};
  DeviceBuffer<Real> d_rhs{n};
  DeviceBuffer<Real> d_residual{n};
  DeviceBuffer<Real> d_jac{n};
  DeviceBuffer<Real> d_delta{n};
  // Separate from d_solved: the line search overwrites its trial buffer on
  // every backtracking step, and d_solved holds the Picard fallback that is
  // still needed if no step length is accepted.
  DeviceBuffer<Real> d_trial{n};

  GsOperatorScratch op{g};
  GsReduceScratch reduce{g};
  GsDerivativeFields deriv{g};
  GsCriticalScratch critical{g};
  GsPlasmaMaskScratch plasma_mask{g};
  GsDeviceMultigrid mg{g};
  GsCoilSet coils{cfg_.coils};

  Real prev_axis_r = Real{0};
  Real prev_axis_z = Real{0};
  bool have_prev_axis = false;
  bool have_coherent_state = false;
  Real first_residual = Real{0};

  // Every exit path -- converged or not -- retains BOTH partial fields. Once a
  // state has been evaluated, d_psi_good is that exact Picard state: its
  // interior is the iterate from which critical points and current were built,
  // its boundary was refreshed from that current, and its residual was measured
  // before any subsequent interior update. Candidate construction may overwrite
  // d_psi_next before discovering non-finite arithmetic, so it is never used as
  // the retained result until swapped into d_psi_good after full validation.
  // The synchronization is required before the local scratch buffers are
  // destroyed; it does not download either full field.
  const auto finish = [&](GsStatus status) -> GsDeviceResult {
    res.status = status;
    backend::device_synchronize(stream);
    res.psi = have_coherent_state ? std::move(d_psi_good) : std::move(d_psi);
    res.j_phi = std::move(d_j_phi);
    return std::move(res);
  };

  // Vacuum coil field plus the seeded plasma column. The column is essential:
  // a pure coil field satisfies Delta* psi = 0, which admits no interior
  // extremum, so without a seed there is no O-point and the loop cannot start.
  launch_gs_evaluate_coil_field(g, coils, d_psi.device_ptr(), stream);
  const ResolvedSeed seed = resolve_seed(cfg_);
  if (!valid_resolved_seed(cfg_, seed)) {
    return finish(GsStatus::numerical_failure);
  }
  launch_gs_add_plasma_seed(g, seed.r_center, seed.z_center, seed.minor_radius,
                            seed.depth, seed.sign, d_psi.device_ptr(), stream);

  for (int it = 1; it <= cfg_.max_iterations; ++it) {
    res.iterations = it;

    // -- 1. critical points --------------------------------------------------
    launch_gs_compute_derivatives(g, d_psi.device_ptr(), deriv, op, stream);
    launch_gs_find_critical_points(g, d_psi.device_ptr(), deriv, critical,
                                   stream);
    const GsCriticalResult cps = copy_critical_to_host(critical, stream);
    CriticalPointSet candidate_critical = to_host_set(cps);

    if (cps.numerical_failure) {
      if (!have_coherent_state) res.critical = std::move(candidate_critical);
      return finish(GsStatus::numerical_failure);
    }

    if (cps.x_point_overflow) {
      // Preserve the last coherent physical state, but expose why the next
      // topology evaluation could not be trusted.
      res.critical.critical_point_overflow = true;
      if (!have_coherent_state) res.critical = std::move(candidate_critical);
      return finish(GsStatus::critical_point_overflow);
    }

    if (!cps.axis.valid) {
      if (!have_coherent_state) res.critical = std::move(candidate_critical);
      return finish(have_prev_axis ? GsStatus::axis_lost
                                   : GsStatus::no_closed_surface);
    }
    if (!cps.has_closed_surface) {
      if (!have_coherent_state) res.critical = std::move(candidate_critical);
      return finish(GsStatus::no_closed_surface);
    }

    const bool geometry_stable =
        have_prev_axis
        && std::abs(cps.axis.r - prev_axis_r)
               < cfg_.newton_geometry_tolerance * g.dr()
        && std::abs(cps.axis.z - prev_axis_z)
               < cfg_.newton_geometry_tolerance * g.dz();
    prev_axis_r = cps.axis.r;
    prev_axis_z = cps.axis.z;
    have_prev_axis = true;

    // -- 2. current density, normalized to the requested I_p -----------------
    launch_gs_build_plasma_mask(
        g, d_psi.device_ptr(), deriv, cps.axis.r, cps.axis.z, cps.psi_axis,
        cps.psi_boundary, plasma_mask, stream);
    launch_gs_build_current(g, d_psi.device_ptr(), profile_, cps.psi_axis,
                            cps.psi_boundary, plasma_mask.mask.device_ptr(),
                            d_j_candidate.device_ptr(), stream);
    launch_gs_total_plasma_current(g, d_j_candidate.device_ptr(), reduce,
                                   stream);
    const Real raw_current = copy_scalar_to_host(reduce, stream);

    if (!std::isfinite(raw_current)) {
      if (!have_coherent_state) res.critical = std::move(candidate_critical);
      return finish(GsStatus::numerical_failure);
    }
    if (raw_current == Real{0}) {
      if (!have_coherent_state) res.critical = std::move(candidate_critical);
      return finish(GsStatus::no_closed_surface);
    }
    const Real scale = cfg_.plasma_current / raw_current;
    if (!std::isfinite(scale) || scale == Real{0}) {
      if (!have_coherent_state) res.critical = std::move(candidate_critical);
      return finish(GsStatus::numerical_failure);
    }
    launch_gs_scale_field(g, scale, d_j_candidate.device_ptr(), stream);
    launch_gs_total_plasma_current(g, d_j_candidate.device_ptr(), reduce,
                                   stream);
    const Real achieved_current = copy_scalar_to_host(reduce, stream);
    if (!std::isfinite(achieved_current) || achieved_current == Real{0}) {
      if (!have_coherent_state) res.critical = std::move(candidate_critical);
      return finish(GsStatus::numerical_failure);
    }

    // -- 3. boundary condition ------------------------------------------------
    // Only boundary nodes change here. Thus candidate_critical and the current
    // remain tied to the unchanged interior while d_psi_next becomes the exact
    // free-boundary candidate whose residual is evaluated.
    backend::device_memcpy_d2d_async(d_psi_next.device_ptr(),
                                     d_psi.device_ptr(), n * sizeof(Real),
                                     stream);
    launch_gs_apply_coil_boundary(g, coils, d_psi_next.device_ptr(), stream);
    launch_gs_add_plasma_boundary(g, d_j_candidate.device_ptr(),
                                  d_psi_next.device_ptr(), stream);

    // -- 4. interior source and linear solve ----------------------------------
    launch_gs_build_rhs(g, d_j_candidate.device_ptr(), d_rhs.device_ptr(),
                        stream);

    launch_gs_residual_l6(g, d_psi_next.device_ptr(), d_rhs.device_ptr(),
                          d_residual.device_ptr(), op, stream);
    launch_gs_interior_max_norm(g, d_residual.device_ptr(), reduce, stream);
    const Real rnorm = copy_scalar_to_host(reduce, stream);
    if (!std::isfinite(rnorm) || rnorm < Real{0}) {
      if (!have_coherent_state) res.critical = std::move(candidate_critical);
      return finish(GsStatus::numerical_failure);
    }

    if (it == 1) first_residual = rnorm > Real{0} ? rnorm : Real{1};
    const Real relative_residual = rnorm / first_residual;
    if (!std::isfinite(relative_residual)
        || relative_residual < Real{0}) {
      if (!have_coherent_state) res.critical = std::move(candidate_critical);
      return finish(GsStatus::numerical_failure);
    }
    res.critical = std::move(candidate_critical);
    res.profile_scale = scale;
    res.achieved_current = achieved_current;
    res.residual = relative_residual;
    res.residual_history.push_back(res.residual);
    std::swap(d_j_phi, d_j_candidate);
    std::swap(d_psi_good, d_psi_next);
    have_coherent_state = true;

    if (res.residual <= cfg_.tolerance) {
      return finish(GsStatus::converged);
    }

    // Stall and iteration-limit exits return the just-evaluated state. Applying
    // another nonlinear update first would pair the new psi with the old
    // current, critical points, residual, and profile scale.
    if (res.residual_history.size() >= 12) {
      const std::size_t k = res.residual_history.size();
      const Real then = res.residual_history[k - 12];
      if (res.residual > Real{0.98} * then) {
        return finish(GsStatus::residual_stalled);
      }
    }
    if (it == cfg_.max_iterations) {
      return finish(GsStatus::iteration_limit);
    }

    const bool try_newton = cfg_.enable_newton && geometry_stable
                         && res.residual < cfg_.newton_residual_threshold;

    backend::device_memcpy_d2d_async(d_solved.device_ptr(),
                                     d_psi_good.device_ptr(), n * sizeof(Real),
                                     stream);
    GsDefectCorrectionConfig dc;
    dc.max_iterations = 40;
    dc.tolerance = Real{1e-8};
    const GsDefectCorrectionReport dc_report =
        launch_gs_solve_defect_corrected(g, d_solved.device_ptr(),
                                         d_rhs.device_ptr(), mg, op, reduce,
                                         dc, stream);
    if (!std::isfinite(dc_report.initial_residual)
        || dc_report.initial_residual < Real{0}
        || !std::isfinite(dc_report.final_residual)
        || dc_report.final_residual < Real{0}) {
      return finish(GsStatus::numerical_failure);
    }

    // The accepted state includes the freshly recomputed free-boundary values.
    // Start every nonlinear update from that coherent state: the compact L6
    // derivatives used by Newton depend on the boundary, even though the
    // correction itself is applied only to interior nodes.
    backend::device_memcpy_d2d_async(d_psi.device_ptr(),
                                     d_psi_good.device_ptr(), n * sizeof(Real),
                                     stream);

    if (try_newton) {
      launch_gs_build_jacobian_diagonal(g, d_psi.device_ptr(), profile_,
                                        cps.psi_axis, cps.psi_boundary, scale,
                                        plasma_mask.mask.device_ptr(),
                                        d_jac.device_ptr(), stream);
      launch_gs_residual_l6(g, d_psi.device_ptr(), d_rhs.device_ptr(),
                            d_residual.device_ptr(), op, stream);
      launch_gs_newton_correction(g, d_jac.device_ptr(),
                                  d_residual.device_ptr(),
                                  d_delta.device_ptr(), mg, stream);

      const Real accepted = launch_gs_newton_line_search(
          g, d_psi.device_ptr(), d_delta.device_ptr(), d_rhs.device_ptr(),
          d_trial.device_ptr(), op, reduce, stream);

      if (accepted > Real{0}) {
        // The line search has already formed and validated this exact trial.
        // Copy it instead of rebuilding the update from a potentially
        // different base state.
        backend::device_memcpy_d2d_async(d_psi.device_ptr(),
                                         d_trial.device_ptr(),
                                         n * sizeof(Real), stream);
        ++res.newton_steps;
      } else {
        launch_gs_blend(g, d_psi.device_ptr(), d_solved.device_ptr(),
                        cfg_.picard_relaxation, stream);
      }
    } else {
      launch_gs_blend(g, d_psi.device_ptr(), d_solved.device_ptr(),
                      cfg_.picard_relaxation, stream);
    }

    // Restore exact boundary data: the blend interpolates it too.
    launch_gs_restore_boundary(g, d_psi_good.device_ptr(), d_psi.device_ptr(),
                               stream);

  }

  return finish(GsStatus::iteration_limit);
}

}  // namespace quasar::equilibrium
