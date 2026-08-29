#pragma once

// Source-convention equilibrium fields for the toroidal ideal-MHD energy
// principle.
//
// The spectral FluxCoordinateGrid stores the counter-clockwise straight-field
// angle constructed by the equilibrium pipeline.  The PEST/Glasser convention
// used by the stability derivation instead takes
//
//   lambda       = psi_N,
//   theta_source = -theta_stored,
//   S            = psi_axis - psi_boundary.
//
// Consequently the mixed covariant metric changes sign while the diagonal
// components do not.  With J = R (R_lambda Z_thetaStored -
// R_thetaStored Z_lambda), the source-convention magnetic field is
//
//   B^theta = S/J,       B^phi = q S/J.
//
// Glasser, Phys. Plasmas 23, 072505 (2016), Eq. (7), writes curl(B) as C.
// Restored to physical SI current density j = C/mu0,
//
//   j^theta = -F_lambda/(mu0 J),
//   j^phi   = q j^theta - P_lambda/S,
//
// where F = R B_phi.  These formulas use only analytic equilibrium profiles;
// they never differentiate sampled grid fields or consult the legacy nearest-
// neighbour F table.

#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/equilibrium_profile.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"
#include "quasar/physics/stability/kernels.hpp"

#include <stdexcept>

namespace quasar::stability {

// Host-visible classification for physical validation failures detected while
// constructing the analytic equilibrium fields.  Contract defects such as
// mismatched shapes or device ownership continue to throw std::invalid_argument
// directly so orchestration code cannot mistake a programming error for an
// unsupported equilibrium.
enum class ToroidalEquilibriumValidationStatus {
  invalid_lambda,
  nonpositive_f_squared,
  nonpositive_density,
  nonfinite_profile,
  invalid_geometry,
};

class ToroidalEquilibriumValidationError final
    : public std::invalid_argument {
 public:
  ToroidalEquilibriumValidationError(
      ToroidalEquilibriumValidationStatus status, const char* message)
      : std::invalid_argument{message}, status_{status} {}

  [[nodiscard]] ToroidalEquilibriumValidationStatus status() const noexcept {
    return status_;
  }

 private:
  ToroidalEquilibriumValidationStatus status_;
};

// Mandatory polynomial mass-density profile rho(lambda).  It is a separate
// input from the Grad-Shafranov free functions because density does not enter
// that equilibrium equation, but it is required by the stability mass matrix.
// Positivity is checked at every supplied spectral radial node on device.
struct DensityProfileCoefficients {
  static constexpr int kMaxCoefficients =
      equilibrium::ProfileCoefficients::kMaxCoefficients;

  Real coefficients[kMaxCoefficients]{};
  int count{0};
};

// Device-resident equilibrium quantities in the PEST/Glasser convention.
// Flux functions have one value per local radial node.  Geometry, magnetic
// field, and current components are surface-major over (lambda, theta).
struct ToroidalEquilibriumFields {
  ToroidalEquilibriumFields() = default;
  ToroidalEquilibriumFields(int n_lambda, int n_theta) {
    resize(n_lambda, n_theta);
  }

  void resize(int n_lambda, int n_theta);

  // Flux functions.
  backend::DeviceBuffer<Real> pressure{};
  backend::DeviceBuffer<Real> pressure_lambda{};
  backend::DeviceBuffer<Real> f{};
  backend::DeviceBuffer<Real> f_squared{};
  backend::DeviceBuffer<Real> f_lambda{};
  backend::DeviceBuffer<Real> ff_lambda{};
  backend::DeviceBuffer<Real> density{};

  // Source-convention covariant metric and positive-or-signed Jacobian as
  // supplied by the coordinate orientation.  No absolute value is applied.
  backend::DeviceBuffer<Real> g_lambda_lambda{};
  backend::DeviceBuffer<Real> g_lambda_theta{};
  backend::DeviceBuffer<Real> g_theta_theta{};
  backend::DeviceBuffer<Real> g_phi_phi{};
  backend::DeviceBuffer<Real> jacobian{};

  // Contravariant source-coordinate components.
  backend::DeviceBuffer<Real> b_theta{};
  backend::DeviceBuffer<Real> b_phi{};
  backend::DeviceBuffer<Real> j_theta{};
  backend::DeviceBuffer<Real> j_phi{};

  int n_lambda{0};
  int n_theta{0};
  Real signed_flux_scale{0};  // S = psi_axis - psi_boundary
};

// Evaluate P, P_lambda, F, F_lambda, FF_lambda, and rho analytically at the
// exact radial nodes in `coords`, then form source-convention metric, magnetic,
// and physical-current components at every tensor-product point.
//
// The pressure datum is P(1)=0 and the toroidal-field datum is F(1)=f_vacuum:
//
//   P(lambda)   = -integral_lambda^1 P_s ds,
//   F^2(lambda) = f_vacuum^2
//                 - 2 (psi_boundary-psi_axis) profile_scale
//                   integral_lambda^1 ff_prime(s) ds.
//
// A compact device status word is the only data copied to the host.  Physical
// validation failures (F^2 <= 0, rho <= 0, an invalid normalized flux label,
// or degenerate/non-finite geometry) throw
// ToroidalEquilibriumValidationError. Shape, ownership, and other API contract
// defects throw their ordinary standard exceptions.
void launch_build_toroidal_equilibrium(
    const FluxCoordinateGrid& coords,
    const equilibrium::ProfileCoefficients& profile, Real psi_axis,
    Real psi_boundary, Real profile_scale, Real f_vacuum,
    const DensityProfileCoefficients& density,
    ToroidalEquilibriumFields& out, stream_t stream);

// Per-surface signed validation of the coordinate/field handoff.  B_R, B_Z,
// and B_phi are sampled from the equilibrium grid with tensor-product cubic
// interpolation, matching the spectral geometry path.  The checks are
//
//   q = B^phi/B^theta,       S = J B^theta,
//   J/R^2 = a flux function,
//
// with
//
//   B^theta_source = (B_R Z_lambda - B_Z R_lambda)
//                    /(R_lambda Z_thetaStored
//                      - R_thetaStored Z_lambda),
//   B^phi = B_phi_physical/R.
//
// Deviations compare signed values; a reversed field therefore reports an
// O(2) error rather than passing through absolute values.  Malformed samples
// are represented by finite max-Real deviations so diagnostic reductions do
// not silently turn into NaNs.
struct ToroidalGeometryValidation {
  ToroidalGeometryValidation() = default;
  explicit ToroidalGeometryValidation(int n_surfaces) { resize(n_surfaces); }

  void resize(int n_surfaces);

  backend::DeviceBuffer<Real> q_relative_deviation{};
  backend::DeviceBuffer<Real> flux_scale_relative_deviation{};
  backend::DeviceBuffer<Real> pest_j_over_r2_relative_deviation{};
  int n_surfaces{0};
};

void launch_validate_toroidal_geometry(
    const FluxCoordinateGrid& coords, const numerics::EllipticGrid& grid,
    const equilibrium::GsMagneticField& field,
    const ToroidalEquilibriumFields& equilibrium,
    ToroidalGeometryValidation& out, stream_t stream);

// Compact host-visible reduction of the per-surface validation arrays.  A
// surface is invalid when any deviation is negative, non-finite, or the
// max-Real sentinel emitted for malformed geometry.  Maxima retain every valid
// metric value even when a different metric marks the same surface invalid.
struct ToroidalGeometryValidationSummary {
  Real maximum_q_relative_deviation{0};
  Real maximum_flux_scale_relative_deviation{0};
  Real maximum_pest_j_over_r2_relative_deviation{0};
  int invalid_surface_count{0};
  int first_invalid_surface{-1};

  [[nodiscard]] bool ok() const noexcept {
    return invalid_surface_count == 0;
  }
};

// Scan surfaces once in ascending index order on device, then transfer one
// summary object to the host.  The ordered single-thread reduction is
// deterministic and uses no atomics.
[[nodiscard]] ToroidalGeometryValidationSummary
summarize_toroidal_geometry_validation(
    const ToroidalGeometryValidation& validation, stream_t stream);

}  // namespace quasar::stability
