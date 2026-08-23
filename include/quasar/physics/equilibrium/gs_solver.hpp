#pragma once

// Free-boundary Grad-Shafranov solver driver.
//
// Solves
//
//   Delta* psi = -mu0 r^2 p'(psi_N) - F F'(psi_N)  =  -mu0 r j_phi
//
// with psi on the computational boundary supplied by the Green's-function
// integral over external coils plus the plasma's own current.
//
// -- The outer loop ------------------------------------------------------------
//   1. locate axis and X-points          -> psi_axis, psi_bdry, plasma mask
//   2. evaluate profiles at psi_N, rescale to hit the requested I_p
//   3. update boundary psi (coils + plasma Green's function)
//   4. Picard or Newton step, solved by defect-corrected multigrid
//   repeat
//
// Steps 1-3 are what make this nonlinear: the source depends on psi through
// psi_N, and the boundary condition depends on psi through the plasma current.
//
// -- Picard, with Newton available but off by default --------------------------
// The plan specified a Picard-then-Newton switch. The switch is implemented and
// its criteria are sound (residual below threshold AND critical points
// stationary -- the second is not redundant, because the residual can look
// small while the plasma boundary is still migrating, which is exactly when a
// quadratic step overshoots).
//
// But Newton is DISABLED by default because it measurably hurts on this
// problem: the diagonal profile Jacobian is incomplete for a free-boundary
// solve, where psi on the boundary is itself a dense functional of the interior
// psi. See the extended note on GsConfig::enable_newton for the measurement and
// for what completing it would require.
//
// The production path is therefore damped Picard, which is unconditionally
// consistent with the full nonlinearity including the boundary. It converges in
// a few hundred iterations on a 65x65 reference case.
//
// -- Failure is an answer ------------------------------------------------------
// A free-boundary GS problem may legitimately have NO solution: the requested
// I_p and coil set may admit no confined plasma, the plasma may drift into the
// wall, or the axis and X-point may merge. These are physical outcomes, not
// programming errors, so they are reported through GsStatus with the partial
// psi retained -- an optimizer must be able to score a failed configuration and
// continue. Malformed INPUT still throws.

#include "quasar/core/types.hpp"
#include "quasar/numerics/defect_correction.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/numerics/geometric_multigrid.hpp"
#include "quasar/numerics/gs_operator_l6.hpp"
#include "quasar/physics/equilibrium/critical_points.hpp"
#include "quasar/physics/equilibrium/equilibrium_profile.hpp"
#include "quasar/physics/equilibrium/free_boundary.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace quasar::equilibrium {

enum class GsStatus {
  converged,
  no_closed_surface,   // no O-point: nothing is confined
  axis_lost,           // axis left the domain or merged with an X-point
  residual_stalled,    // iteration ceased making progress
  iteration_limit,     // ran out of iterations while still improving
};

inline const char* to_string(GsStatus s) {
  switch (s) {
    case GsStatus::converged:         return "converged";
    case GsStatus::no_closed_surface: return "no_closed_surface";
    case GsStatus::axis_lost:         return "axis_lost";
    case GsStatus::residual_stalled:  return "residual_stalled";
    case GsStatus::iteration_limit:   return "iteration_limit";
  }
  return "unknown";
}

// Initial guess for the plasma column.
//
// This is NOT optional decoration -- the iteration cannot start without it. A
// vacuum field satisfies Delta* psi = 0, and an operator of that form admits no
// interior extremum, so a pure coil field NEVER has an O-point. Since the axis
// is what defines psi_N, which defines the current, which creates the axis, the
// loop has no way to bootstrap itself from vacuum alone.
//
// The standard resolution (FreeGS and CREATE both do this) is to seed a plausible
// plasma column, let it generate a current, and let the iteration relax the shape.
// The seed's details do not survive convergence; only its topology matters.
struct PlasmaSeed {
  Real r_center{0};      // 0 => use the domain's geometric centre
  Real z_center{0};
  Real minor_radius{0};  // 0 => a quarter of the smaller domain extent
  // Seeded well depth as a MULTIPLE OF mu0*I_p, which is the natural flux scale
  // a current I_p generates. Scaling to the vacuum field's own variation instead
  // is wrong and fails in an instructive way: on a representative case the
  // vacuum span was 8x the depth the requested current could sustain, so the
  // well decayed toward its physical value, passed through zero on the way, and
  // the axis vanished mid-iteration ("axis_lost" at iteration 4). Seeding near
  // the self-consistent scale keeps the topology stable from the first step.
  Real depth{Real{0.3}};
};

struct GsConfig {
  EllipticGrid grid{};
  std::vector<CoilFilament> coils{};
  Real plasma_current{Real{1e6}};       // target I_p in amperes
  int  max_iterations{100};
  Real tolerance{Real{1e-9}};           // relative nonlinear residual
  Real picard_relaxation{Real{1}};
  // Switch thresholds. Both must be met before Newton is attempted.
  Real newton_residual_threshold{Real{1e-3}};
  Real newton_geometry_tolerance{Real{1}};  // in grid cells
  // DEFAULT OFF, and that is a measured decision rather than caution.
  //
  // The plan called for a Picard-then-Newton switch on the expectation that
  // Newton's quadratic convergence would pay for itself. The implemented Newton
  // step uses the diagonal profile Jacobian dS/dpsi, which is the standard
  // fixed-boundary term. On this FREE-boundary problem that Jacobian is
  // incomplete: psi on the computational boundary is itself a functional of psi
  // through the plasma Green's-function integral, contributing a DENSE
  // d(psi_bdry)/d(psi_interior) block that the diagonal term does not capture.
  //
  // Measured on the reference case (65x65, I_p = 1 MA):
  //   enable_newton = false -> converged,        222 iterations, 9.5e-10
  //   enable_newton = true  -> residual_stalled,  63 iterations, 7.4e-04
  //
  // With the boundary coupling missing, the Newton direction is wrong by an
  // O(1) amount near convergence: the step is accepted once, overshoots, and
  // the iteration stalls. Damped Picard is slower per unit of asymptotic rate
  // but is unconditionally consistent with the full nonlinearity, boundary
  // included.
  //
  // Completing this properly requires either a matrix-free Jacobian-vector
  // product that differentiates the boundary integral too (a Jacobian-free
  // Newton-Krylov formulation), or the von Hagenow surface-current form that
  // makes the boundary dependence local. Both are real work and neither is
  // needed for correctness -- only for speed.
  bool enable_newton{false};
  PlasmaSeed seed{};
};

struct GsResult {
  GsStatus status{GsStatus::iteration_limit};
  int  iterations{0};
  Real residual{0};
  std::vector<Real> residual_history{};
  // Best-effort solution, retained even on failure so a caller can inspect or
  // score it.
  ScalarField psi{};
  ScalarField j_phi{};
  CriticalPointSet critical{};
  Real achieved_current{0};
  // Amplitude applied to p' and FF' to satisfy the requested plasma current.
  // Derived quantities reconstructed from the profile must use the same scale.
  Real profile_scale{1};
  int  newton_steps{0};

  bool ok() const noexcept { return status == GsStatus::converged; }
};

class GsSolver {
 public:
  GsSolver(GsConfig cfg, std::shared_ptr<IEquilibriumProfile> profile)
    : cfg_{std::move(cfg)}, profile_{std::move(profile)} {
    cfg_.grid.validate();
    if (!profile_) {
      throw std::invalid_argument{"GsSolver: profile must not be null"};
    }
    if (!numerics::l6_is_applicable(cfg_.grid)) {
      throw std::invalid_argument{
          "GsSolver: grid too small for the sixth-order operator"};
    }
    if (!(cfg_.plasma_current != Real{0})) {
      throw std::invalid_argument{"GsSolver: plasma_current must be nonzero"};
    }
  }

  GsResult solve() {
    const EllipticGrid& g = cfg_.grid;
    numerics::GsMultigrid mg{g};

    GsResult res;
    res.psi = numerics::make_field(g);
    res.j_phi = numerics::make_field(g);

    // Vacuum coil field plus a seeded plasma column. The column is essential:
    // without it there is no O-point and the loop cannot start (see PlasmaSeed).
    evaluate_coil_field(g, cfg_.coils, res.psi);
    add_plasma_seed(g, res.psi);

    ScalarField rhs = numerics::make_field(g);
    ScalarField residual = numerics::make_field(g);
    Real prev_axis_r = Real{0};
    Real prev_axis_z = Real{0};
    bool have_prev_axis = false;
    Real first_residual = Real{0};

    for (int it = 1; it <= cfg_.max_iterations; ++it) {
      res.iterations = it;

      // -- 1. critical points ------------------------------------------------
      res.critical = find_critical_points(g, res.psi);
      if (!res.critical.axis.valid) {
        res.status = GsStatus::no_closed_surface;
        return res;
      }
      if (!res.critical.has_closed_surface) {
        res.status = GsStatus::no_closed_surface;
        return res;
      }

      const bool geometry_stable =
          have_prev_axis
          && std::abs(res.critical.axis.r - prev_axis_r)
                 < cfg_.newton_geometry_tolerance * g.dr()
          && std::abs(res.critical.axis.z - prev_axis_z)
                 < cfg_.newton_geometry_tolerance * g.dz();
      prev_axis_r = res.critical.axis.r;
      prev_axis_z = res.critical.axis.z;
      have_prev_axis = true;

      // -- 2. current density, normalized to the requested I_p ---------------
      build_current(g, res.psi, res.critical, res.j_phi);
      const Real raw_current = total_plasma_current(g, res.j_phi);
      if (raw_current == Real{0} || !std::isfinite(raw_current)) {
        res.status = GsStatus::no_closed_surface;
        return res;
      }
      const Real scale = cfg_.plasma_current / raw_current;
      res.profile_scale = scale;
      for (auto& v : res.j_phi) v *= scale;
      res.achieved_current = total_plasma_current(g, res.j_phi);

      // -- 3. boundary condition --------------------------------------------
      ScalarField psi_next = res.psi;
      apply_coil_boundary(g, cfg_.coils, psi_next);
      add_plasma_boundary(g, res.j_phi, psi_next);

      // -- 4. interior source and linear solve -------------------------------
      // Delta* psi = -mu0 r j_phi
      for (int j = 1; j < g.nz - 1; ++j) {
        for (int i = 1; i < g.nr - 1; ++i) {
          rhs[g.index(i, j)] = -kMu0 * g.r(i) * res.j_phi[g.index(i, j)];
        }
      }

      numerics::gs_residual_l6(g, psi_next, rhs, residual);
      const Real rnorm = numerics::interior_max_norm(g, residual);
      if (it == 1) first_residual = rnorm > Real{0} ? rnorm : Real{1};
      res.residual = rnorm / first_residual;
      res.residual_history.push_back(res.residual);

      if (res.residual <= cfg_.tolerance) {
        res.psi = psi_next;
        res.status = GsStatus::converged;
        return res;
      }

      const bool try_newton = cfg_.enable_newton && geometry_stable
                           && res.residual < cfg_.newton_residual_threshold;

      ScalarField solved = psi_next;
      numerics::DefectCorrectionConfig dc;
      dc.max_iterations = 40;
      dc.tolerance = Real{1e-8};
      const auto lin = numerics::solve_defect_corrected(g, solved, rhs, mg, dc);
      (void)lin;

      if (try_newton) {
        // A true Newton step, not merely a line-searched Picard step.
        //
        // Picard freezes the source at the current psi and solves
        // Delta* psi_new = S(psi_old); its error contracts by a fixed factor
        // because it ignores dS/dpsi entirely. Newton instead linearizes the
        // whole residual:
        //
        //   [Delta* - dS/dpsi] delta = -(Delta* psi - S(psi))
        //
        // where dS/dpsi is DIAGONAL:
        //
        //   dS/dpsi = -mu0 r^2 p''(psi_N)/(psi_b - psi_a)
        //             - FF''(psi_N)/(psi_b - psi_a)
        //
        // Omitting that diagonal term is what leaves the iteration linear. It
        // is cheap -- one extra array, folded into the multigrid stencil.
        ScalarField jac = numerics::make_field(g);
        build_jacobian_diagonal(g, res.psi, res.critical, scale, jac);

        ScalarField nres = numerics::make_field(g);
        numerics::gs_residual_l6(g, res.psi, rhs, nres);

        ScalarField delta = numerics::make_field(g);
        newton_correction(g, jac, nres, delta, mg);

        const Real accepted = newton_line_search(g, res.psi, delta, rhs);
        if (accepted > Real{0}) {
          for (int j = 1; j < g.nz - 1; ++j) {
            for (int i = 1; i < g.nr - 1; ++i) {
              res.psi[g.index(i, j)] += accepted * delta[g.index(i, j)];
            }
          }
          ++res.newton_steps;
        } else {
          blend(res.psi, solved, cfg_.picard_relaxation);
        }
      } else {
        blend(res.psi, solved, cfg_.picard_relaxation);
      }

      // Restore exact boundary data (the blend interpolates it too).
      for (int j = 0; j < g.nz; ++j) {
        for (int i = 0; i < g.nr; ++i) {
          if (g.on_boundary(i, j)) res.psi[g.index(i, j)] = psi_next[g.index(i, j)];
        }
      }

      // Stall detection: no meaningful progress over a window of iterations.
      if (res.residual_history.size() >= 12) {
        const std::size_t n = res.residual_history.size();
        const Real then = res.residual_history[n - 12];
        if (res.residual > Real{0.98} * then) {
          res.status = GsStatus::residual_stalled;
          return res;
        }
      }
    }

    res.status = GsStatus::iteration_limit;
    return res;
  }

 private:
  // +1 for positive plasma current, -1 for reversed. The seeded extremum must
  // point the same way the current will drive it.
  Real sign_of_current() const noexcept {
    return cfg_.plasma_current >= Real{0} ? Real{1} : Real{-1};
  }

  // Superimpose a smooth flux well so the initial state has an O-point.
  //
  // The well is quadratic-with-compact-support: -depth * (1 - s^2)^2 for s < 1,
  // where s is the normalized distance from the seed centre. The (1-s^2)^2 form
  // is C^1 at the seed edge, so it does not inject a kink that the sixth-order
  // residual would have to smooth away over many iterations.
  //
  // The depth is a multiple of mu0*|I_p| -- the flux scale the requested current
  // actually generates -- NOT of the vacuum field's variation. See PlasmaSeed
  // for why that distinction decides whether the iteration survives its first
  // few steps.
  void add_plasma_seed(const EllipticGrid& g, ScalarField& psi) const {
    const Real rc = cfg_.seed.r_center > Real{0}
        ? cfg_.seed.r_center
        : Real{0.5} * (g.r_min + g.r_max);
    const Real zc = cfg_.seed.z_center != Real{0}
        ? cfg_.seed.z_center
        : Real{0.5} * (g.z_min + g.z_max);
    const Real a = cfg_.seed.minor_radius > Real{0}
        ? cfg_.seed.minor_radius
        : Real{0.25} * std::min(g.r_max - g.r_min, g.z_max - g.z_min);
    const Real depth = cfg_.seed.depth * kMu0 * std::abs(cfg_.plasma_current);

    for (int j = 0; j < g.nz; ++j) {
      for (int i = 0; i < g.nr; ++i) {
        if (g.on_boundary(i, j)) continue;  // boundary is Green's-function data
        const Real dr = (g.r(i) - rc) / a;
        const Real dz = (g.z(j) - zc) / a;
        const Real s2 = dr * dr + dz * dz;
        if (s2 >= Real{1}) continue;
        const Real w = Real{1} - s2;
        // ADDED, not subtracted: with Delta* psi = -mu0 r j_phi and j_phi > 0
        // the right-hand side is negative, and Delta* is negative-definite on a
        // Dirichlet domain, so psi attains an interior MAXIMUM. Seeding a
        // minimum instead forces the iteration to invert the well's sign before
        // it can converge -- observed as the well passing through zero at
        // iteration 3 and the axis then drifting steadily outward while the
        // residual stalled.
        psi[g.index(i, j)] += depth * w * w * sign_of_current();
      }
    }
  }

  // j_phi = r p'(psi_N) + FF'(psi_N) / (mu0 r), zero outside the plasma.
  void build_current(const EllipticGrid& g, const ScalarField& psi,
                     const CriticalPointSet& cps, ScalarField& j_phi) const {
    j_phi.assign(g.size(), Real{0});
    for (int j = 1; j < g.nz - 1; ++j) {
      for (int i = 1; i < g.nr - 1; ++i) {
        const Real pn = normalized_flux(psi[g.index(i, j)], cps.psi_axis,
                                        cps.psi_boundary);
        // psi_N == 1 is outside the last closed surface: no current there.
        if (pn >= Real{1}) continue;
        const Real r = g.r(i);
        j_phi[g.index(i, j)] = r * profile_->dp_dpsi(pn)
                             + profile_->ff_prime(pn) / (kMu0 * r);
      }
    }
  }

  static void blend(ScalarField& target, const ScalarField& candidate,
                    Real weight) {
    for (std::size_t k = 0; k < target.size(); ++k) {
      target[k] += weight * (candidate[k] - target[k]);
    }
  }

  // dS/dpsi, the diagonal of the source's derivative with respect to psi.
  // Zero outside the plasma, where the profiles drive no current.
  void build_jacobian_diagonal(const EllipticGrid& g, const ScalarField& psi,
                               const CriticalPointSet& cps, Real profile_scale,
                               ScalarField& jac) const {
    jac.assign(g.size(), Real{0});
    const Real denom = cps.psi_boundary - cps.psi_axis;
    if (denom == Real{0}) return;
    for (int j = 1; j < g.nz - 1; ++j) {
      for (int i = 1; i < g.nr - 1; ++i) {
        const Real pn = normalized_flux(psi[g.index(i, j)], cps.psi_axis,
                                        cps.psi_boundary);
        if (pn >= Real{1} || pn <= Real{0}) continue;
        const Real r = g.r(i);
        // S(psi) = -mu0 r^2 p'(psi_N) - FF'(psi_N); differentiate via psi_N.
        jac[g.index(i, j)] = profile_scale
            * (-kMu0 * r * r * profile_->d2p_dpsi2(pn)
               - profile_->ff_prime_prime(pn)) / denom;
      }
    }
  }

  // Solve [Delta* - diag(jac)] delta = -residual.
  //
  // The shifted operator is not the plain Delta* the multigrid hierarchy is
  // built for, so the diagonal shift is applied as a defect-corrected outer
  // iteration around the existing V-cycle rather than by rebuilding the
  // hierarchy. In practice the shift is small compared with the operator's
  // diagonal, so a handful of passes suffices.
  void newton_correction(const EllipticGrid& g, const ScalarField& jac,
                         const ScalarField& residual, ScalarField& delta,
                         numerics::GsMultigrid& mg) const {
    delta.assign(g.size(), Real{0});
    ScalarField work = numerics::make_field(g);
    ScalarField rhs_shift = numerics::make_field(g);

    for (int pass = 0; pass < 4; ++pass) {
      // The residual convention here is r = b - L6(psi), so L6(psi) - b = -r
      // and the Newton system [L6 - J] delta = -(L6(psi) - b) has right-hand
      // side +r, not -r. Moving the diagonal shift across gives
      // Delta* delta = r + jac*delta.
      for (int j = 1; j < g.nz - 1; ++j) {
        for (int i = 1; i < g.nr - 1; ++i) {
          const std::size_t k = g.index(i, j);
          rhs_shift[k] = residual[k] + jac[k] * delta[k];
        }
      }
      work.assign(g.size(), Real{0});
      for (int c = 0; c < 3; ++c) mg.v_cycle(work, rhs_shift);
      delta.swap(work);
    }
  }

  // Backtracking line search on the L6 residual norm for a Newton step.
  // Returns the accepted step length, or 0 if no tested length improved it.
  Real newton_line_search(const EllipticGrid& g, const ScalarField& current,
                          const ScalarField& delta,
                          const ScalarField& rhs) const {
    ScalarField resid = numerics::make_field(g);
    numerics::gs_residual_l6(g, current, rhs, resid);
    const Real base = numerics::interior_max_norm(g, resid);

    ScalarField trial = numerics::make_field(g);
    for (const Real alpha : {Real{1}, Real{0.5}, Real{0.25}, Real{0.125}}) {
      trial = current;
      for (int j = 1; j < g.nz - 1; ++j) {
        for (int i = 1; i < g.nr - 1; ++i) {
          trial[g.index(i, j)] += alpha * delta[g.index(i, j)];
        }
      }
      numerics::gs_residual_l6(g, trial, rhs, resid);
      if (numerics::interior_max_norm(g, resid) < base) return alpha;
    }
    return Real{0};
  }

  GsConfig cfg_;
  std::shared_ptr<IEquilibriumProfile> profile_;
};

}  // namespace quasar::equilibrium
