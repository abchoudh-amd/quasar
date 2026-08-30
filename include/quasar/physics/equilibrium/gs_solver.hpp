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
// -- Execution -----------------------------------------------------------------
// Every arithmetic step runs on the GPU, through the launch ABI in
// physics/equilibrium/kernels.hpp. What remains here is control flow: the
// iteration counter, the status transitions, and the stall window. Two scalars
// are read back per iteration -- the raw plasma current, which sets the I_p
// normalization, and the residual norm, which drives the convergence and stall
// tests. Both feed branches that decide what the next iteration does, so those
// synchronizations are load-bearing rather than incidental; eliminating them
// would change the iteration count.
//
// There is no host fallback. The host implementation this replaced was retained
// only long enough to validate the port kernel by kernel, and those equivalence
// tests -- most of them asserting bitwise equality -- are what licensed its
// removal. Correctness is now pinned by three things: the manufactured-solution
// order study in tests/unit/numerics/, which never depended on this driver; the
// per-kernel tests, which still compare against the host reference
// implementations retained under numerics/ for exactly that purpose; and the
// port gate in tests/unit/physics/equilibrium/test_gs_port_gate.cpp, which
// checks the assembled solver against an equilibrium recorded before any of the
// port existed.
//
// -- Picard, with Newton available but off by default --------------------------
// The switch is implemented and its criteria are sound (residual below
// threshold AND critical points stationary -- the second is not redundant,
// because the residual can look small while the plasma boundary is still
// migrating, which is exactly when a quadratic step overshoots).
//
// But Newton is DISABLED by default because it measurably hurts on this
// problem: the diagonal profile Jacobian is incomplete for a free-boundary
// solve, where psi on the boundary is itself a dense functional of the interior
// psi. See the extended note on GsConfig::enable_newton.
//
// The production path is therefore damped Picard, which is unconditionally
// consistent with the full nonlinearity including the boundary.
//
// -- Profiles ------------------------------------------------------------------
// The profile must be a PolynomialProfile. IEquilibriumProfile is virtual and a
// vtable cannot cross to the device, so the selected profile is lowered to a
// ProfileCoefficients POD at construction and evaluated on device with no
// indirection. A non-polynomial profile is a construction-time error rather
// than a silent fallback; adding one means giving it a device evaluator.
//
// -- Failure is an answer ------------------------------------------------------
// A free-boundary GS problem may legitimately have NO solution: the requested
// I_p and coil set may admit no confined plasma, the plasma may drift into the
// wall, or the axis and X-point may merge. These are physical outcomes, not
// programming errors, so they are reported through GsStatus with the partial
// psi AND j_phi retained -- an optimizer must be able to score a failed
// configuration and continue. Malformed INPUT still throws.

#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/critical_points.hpp"
#include "quasar/physics/equilibrium/equilibrium_profile.hpp"
#include "quasar/physics/equilibrium/free_boundary.hpp"

#include <memory>
#include <string>
#include <vector>

namespace quasar::equilibrium {

using stream_t = ::quasar::backend::stream_t;

enum class GsStatus {
  converged,
  no_closed_surface,   // no O-point: nothing is confined
  axis_lost,           // axis left the domain or merged with an X-point
  critical_point_overflow,  // fixed device result capacity was exhausted
  numerical_failure,   // derived arithmetic became non-finite
  residual_stalled,    // iteration ceased making progress
  iteration_limit,     // ran out of iterations while still improving
};

inline const char* to_string(GsStatus s) {
  switch (s) {
    case GsStatus::converged:         return "converged";
    case GsStatus::no_closed_surface: return "no_closed_surface";
    case GsStatus::axis_lost:         return "axis_lost";
    case GsStatus::critical_point_overflow:
      return "critical_point_overflow";
    case GsStatus::numerical_failure: return "numerical_failure";
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
  numerics::EllipticGrid grid{};
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
  // The implemented Newton step uses the diagonal profile Jacobian dS/dpsi,
  // which is the standard fixed-boundary term. On this FREE-boundary problem
  // that Jacobian is incomplete: psi on the computational boundary is itself a
  // functional of psi through the plasma Green's-function integral,
  // contributing a DENSE d(psi_bdry)/d(psi_interior) block that the diagonal
  // term does not capture.
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
  // score it. BOTH fields are populated on every exit path; a failed solve
  // returning zeros for j_phi would read as "no current anywhere" rather than
  // "the current present when confinement failed".
  numerics::ScalarField psi{};
  numerics::ScalarField j_phi{};
  CriticalPointSet critical{};
  Real achieved_current{0};
  // Amplitude applied to p' and FF' to satisfy the requested plasma current.
  // Derived quantities reconstructed from the profile must use the same scale.
  Real profile_scale{1};
  // Exact device-lowered profile used by the solve.  Downstream physics must
  // use these coefficients rather than accepting a second, potentially
  // inconsistent profile from the caller.
  ProfileCoefficients profile_coefficients{};
  int  newton_steps{0};

  bool ok() const noexcept { return status == GsStatus::converged; }
};

// Device-resident counterpart to GsResult. Scalar metadata remains on the host
// because it already drives the nonlinear iteration, while the two full grid
// fields stay on the device for downstream equilibrium consumers.
struct GsDeviceResult {
  GsDeviceResult() = default;
  GsDeviceResult(const GsDeviceResult&) = delete;
  GsDeviceResult& operator=(const GsDeviceResult&) = delete;
  GsDeviceResult(GsDeviceResult&&) noexcept = default;
  GsDeviceResult& operator=(GsDeviceResult&&) noexcept = default;

  GsStatus status{GsStatus::iteration_limit};
  int  iterations{0};
  Real residual{0};
  std::vector<Real> residual_history{};
  numerics::EllipticGrid grid{};
  backend::DeviceBuffer<Real> psi{};
  backend::DeviceBuffer<Real> j_phi{};
  CriticalPointSet critical{};
  Real achieved_current{0};
  Real profile_scale{1};
  // Exact device-lowered profile used by the solve; retained so downstream
  // device consumers (notably ideal-MHD stability) cannot drift from the
  // Grad--Shafranov source that produced psi and j_phi.
  ProfileCoefficients profile_coefficients{};
  int  newton_steps{0};

  bool ok() const noexcept { return status == GsStatus::converged; }

  // Download both fields after all preceding work on `stream` completes and
  // reconstruct the existing host-side result contract. Both fields must have
  // the same device owner. A non-null stream must have been created on that
  // same device; raw stream handles do not carry portable ownership metadata.
  GsResult copy_to_host(stream_t stream = nullptr) const;
};

class GsSolver {
 public:
  GsSolver(GsConfig cfg, std::shared_ptr<IEquilibriumProfile> profile);

  GsResult solve();
  // A non-null stream must belong to the current device at call entry; all
  // returned buffers are allocated on that device.
  GsDeviceResult solve_device(stream_t stream = nullptr);

 private:
  GsConfig cfg_;
  ProfileCoefficients profile_{};
};

}  // namespace quasar::equilibrium
