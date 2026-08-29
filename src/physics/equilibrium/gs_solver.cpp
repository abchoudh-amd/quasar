// Free-boundary Grad-Shafranov driver.
//
// This is the orchestration layer: it owns the device buffers, sequences the
// kernel launches, and reproduces GsSolver::solve step for step. There is no
// arithmetic here beyond the two scalar decisions the loop cannot avoid making
// on the host.
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
                   : Real{0.5} * (g.r_min + g.r_max);
  s.z_center = cfg.seed.z_center != Real{0}
                   ? cfg.seed.z_center
                   : Real{0.5} * (g.z_min + g.z_max);
  s.minor_radius = cfg.seed.minor_radius > Real{0}
                       ? cfg.seed.minor_radius
                       : Real{0.25} * std::min(g.r_max - g.r_min,
                                               g.z_max - g.z_min);
  s.depth = cfg.seed.depth * kMu0 * std::abs(cfg.plasma_current);
  s.sign = cfg.plasma_current >= Real{0} ? Real{1} : Real{-1};
  return s;
}

CriticalPointSet to_host_set(const GsCriticalResult& r) {
  CriticalPointSet out;
  out.axis = r.axis;
  out.psi_axis = r.psi_axis;
  out.psi_boundary = r.psi_boundary;
  out.has_closed_surface = r.has_closed_surface;
  out.x_points.assign(r.x_points, r.x_points + r.n_x);
  return out;
}

}  // namespace

GsSolver::GsSolver(GsConfig cfg,
                               std::shared_ptr<IEquilibriumProfile> profile)
  : cfg_{std::move(cfg)} {
  cfg_.grid.validate();
  if (!profile) {
    throw std::invalid_argument{"GsSolver: profile must not be null"};
  }
  if (!numerics::l6_is_applicable(cfg_.grid)) {
    throw std::invalid_argument{
        "GsSolver: grid too small for the sixth-order operator"};
  }
  if (!(cfg_.plasma_current != Real{0})) {
    throw std::invalid_argument{"GsSolver: plasma_current must be nonzero"};
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
  const numerics::EllipticGrid& g = cfg_.grid;
  const std::size_t n = g.size();
  constexpr stream_t stream = nullptr;

  GsResult res;
  res.psi.assign(n, Real{0});
  res.j_phi.assign(n, Real{0});

  DeviceBuffer<Real> d_psi{n};
  DeviceBuffer<Real> d_psi_next{n};
  DeviceBuffer<Real> d_solved{n};
  DeviceBuffer<Real> d_j_phi{n};
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
  GsDeviceMultigrid mg{g};
  GsCoilSet coils{cfg_.coils};

  // Vacuum coil field plus the seeded plasma column. The column is essential:
  // a pure coil field satisfies Delta* psi = 0, which admits no interior
  // extremum, so without a seed there is no O-point and the loop cannot start.
  launch_gs_evaluate_coil_field(g, coils, d_psi.device_ptr(), stream);
  const ResolvedSeed seed = resolve_seed(cfg_);
  launch_gs_add_plasma_seed(g, seed.r_center, seed.z_center, seed.minor_radius,
                            seed.depth, seed.sign, d_psi.device_ptr(), stream);

  Real prev_axis_r = Real{0};
  Real prev_axis_z = Real{0};
  bool have_prev_axis = false;
  Real first_residual = Real{0};

  // Every exit path -- converged or not -- must retain BOTH partial fields.
  // GsResult documents that a failed solve keeps its best-effort solution so an
  // optimizer can score the configuration and continue, and j_phi is half of
  // that. Reading back only psi on the failure paths left j_phi as the zeros it
  // was constructed with, which reads as "no current anywhere" rather than "the
  // current that was there when the plasma was lost".
  const auto finish = [&](GsStatus status) -> GsResult& {
    res.status = status;
    backend::device_synchronize(stream);
    d_psi.copy_to_host(res.psi.data(), res.psi.size());
    d_j_phi.copy_to_host(res.j_phi.data(), res.j_phi.size());
    return res;
  };

  for (int it = 1; it <= cfg_.max_iterations; ++it) {
    res.iterations = it;

    // -- 1. critical points --------------------------------------------------
    launch_gs_compute_derivatives(g, d_psi.device_ptr(), deriv, op, stream);
    launch_gs_find_critical_points(g, d_psi.device_ptr(), deriv, critical,
                                   stream);
    const GsCriticalResult cps = copy_critical_to_host(critical, stream);
    res.critical = to_host_set(cps);

    if (!cps.axis.valid || !cps.has_closed_surface) {
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
    launch_gs_build_current(g, d_psi.device_ptr(), profile_, cps.psi_axis,
                            cps.psi_boundary, d_j_phi.device_ptr(), stream);
    launch_gs_total_plasma_current(g, d_j_phi.device_ptr(), reduce, stream);
    const Real raw_current = copy_scalar_to_host(reduce, stream);

    if (raw_current == Real{0} || !std::isfinite(raw_current)) {
      return finish(GsStatus::no_closed_surface);
    }
    const Real scale = cfg_.plasma_current / raw_current;
    res.profile_scale = scale;
    launch_gs_scale_field(g, scale, d_j_phi.device_ptr(), stream);
    launch_gs_total_plasma_current(g, d_j_phi.device_ptr(), reduce, stream);
    res.achieved_current = copy_scalar_to_host(reduce, stream);

    // -- 3. boundary condition ------------------------------------------------
    backend::device_memcpy_d2d_async(d_psi_next.device_ptr(),
                                     d_psi.device_ptr(), n * sizeof(Real),
                                     stream);
    launch_gs_apply_coil_boundary(g, coils, d_psi_next.device_ptr(), stream);
    launch_gs_add_plasma_boundary(g, d_j_phi.device_ptr(),
                                  d_psi_next.device_ptr(), stream);

    // -- 4. interior source and linear solve ----------------------------------
    launch_gs_build_rhs(g, d_j_phi.device_ptr(), d_rhs.device_ptr(), stream);

    launch_gs_residual_l6(g, d_psi_next.device_ptr(), d_rhs.device_ptr(),
                          d_residual.device_ptr(), op, stream);
    launch_gs_interior_max_norm(g, d_residual.device_ptr(), reduce, stream);
    const Real rnorm = copy_scalar_to_host(reduce, stream);

    if (it == 1) first_residual = rnorm > Real{0} ? rnorm : Real{1};
    res.residual = rnorm / first_residual;
    res.residual_history.push_back(res.residual);

    if (res.residual <= cfg_.tolerance) {
      backend::device_memcpy_d2d_async(d_psi.device_ptr(),
                                       d_psi_next.device_ptr(),
                                       n * sizeof(Real), stream);
      return finish(GsStatus::converged);
    }

    const bool try_newton = cfg_.enable_newton && geometry_stable
                         && res.residual < cfg_.newton_residual_threshold;

    backend::device_memcpy_d2d_async(d_solved.device_ptr(),
                                     d_psi_next.device_ptr(), n * sizeof(Real),
                                     stream);
    GsDefectCorrectionConfig dc;
    dc.max_iterations = 40;
    dc.tolerance = Real{1e-8};
    (void)launch_gs_solve_defect_corrected(g, d_solved.device_ptr(),
                                           d_rhs.device_ptr(), mg, op, reduce,
                                           dc, stream);

    if (try_newton) {
      launch_gs_build_jacobian_diagonal(g, d_psi.device_ptr(), profile_,
                                        cps.psi_axis, cps.psi_boundary, scale,
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
        launch_gs_axpy_interior(g, d_psi.device_ptr(), d_delta.device_ptr(),
                                accepted, stream);
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
    launch_gs_restore_boundary(g, d_psi_next.device_ptr(), d_psi.device_ptr(),
                               stream);

    // Stall detection: no meaningful progress over a window of iterations.
    if (res.residual_history.size() >= 12) {
      const std::size_t k = res.residual_history.size();
      const Real then = res.residual_history[k - 12];
      if (res.residual > Real{0.98} * then) {
        return finish(GsStatus::residual_stalled);
      }
    }
  }

  return finish(GsStatus::iteration_limit);
}

}  // namespace quasar::equilibrium
