#include "quasar/physics/mhd/background_builder.hpp"

#include "quasar/core/observations.hpp"
#include "quasar/numerics/mhd_background_metrics.hpp"
#include "quasar/physics/mhd/kernels.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace quasar::mhd {
namespace {

// The algebraic stopping tolerance of the opt-in vacuum projection. It is a
// curl/vacuum-solve criterion and is deliberately independent of the strict,
// scale-free solenoidality proof applied to the resulting staggered field.
constexpr Real kVacuumProjectionRelativeTolerance = Real{5.0e-11};

struct StatusWord {
  backend::DeviceBuffer<int> device{1};

  StatusWord() {
    const int zero = 0;
    device.copy_from_host(&zero, 1);
  }

  int read(backend::stream_t stream) {
    int host = 0;
    device.copy_to_host(&host, 1);
    backend::device_synchronize(stream);
    return host;
  }
};

void throw_on_background_status(int status) {
  if ((status & kMhdBackgroundCoordinateNotRepresentable) != 0) {
    throw std::invalid_argument{
        "padded grid coordinates are not representable in float64"};
  }
  if ((status & kMhdBackgroundPotentialNotRepresentable) != 0) {
    throw std::invalid_argument{
        "scaled background vector potential is not representable"};
  }
  if ((status & kMhdBackgroundRadialMeasureInvalid) != 0) {
    throw std::invalid_argument{
        "background_field cylindrical curl encountered an invalid annular cell "
        "measure"};
  }
  if ((status & kMhdBackgroundVacuumCoefficientsInvalid) != 0) {
    throw std::invalid_argument{
        "cylindrical vacuum projection coefficients are not representable"};
  }
  if ((status & kMhdBackgroundSampleNotFinite) != 0) {
    throw std::invalid_argument{
        "background field is not representable in float64"};
  }
}

// The boundary rules the padded samples must already satisfy, in the order the
// sequential sweep would have reported them.
void throw_on_boundary_violation(unsigned violated) {
  struct Rule {
    unsigned bit;
    const char* message;
  };
  static constexpr Rule kRules[] = {
      {mhd_background_rule_periodic_x,
       "background_field is not periodic across the x boundary"},
      {mhd_background_rule_axis_parity,
       "background_field violates the r=0 axis parity closure"},
      {mhd_background_rule_axis_constraint,
       "background_field violates the r=0 axis constraint"},
      {mhd_background_rule_x_wall_parity,
       "background_field violates the x-wall mirror parity"},
      {mhd_background_rule_x_wall_normal,
       "background_field has a nonzero normal component at an x wall"},
      {mhd_background_rule_periodic_y,
       "background_field is not periodic across the y boundary"},
      {mhd_background_rule_y_wall_parity,
       "background_field violates the y-wall mirror parity"},
      {mhd_background_rule_y_wall_normal,
       "background_field has a nonzero normal component at a y wall"},
  };
  for (const Rule& rule : kRules) {
    if ((violated & rule.bit) != 0) {
      throw std::invalid_argument{std::string{rule.message} +
                                  "; the configured field boundary requires it"};
    }
  }
}

}  // namespace

MhdBackgroundAffineProfile lower_affine_background_profile(
    const numerics::IMhdBackgroundProfile& profile, const std::string& name) {
  MhdBackgroundAffineProfile lowered{};
  for (int comp = 0; comp < 3; ++comp) {
    const Real c = profile.sample(comp, Real{0}, Real{0});
    const Real sx = profile.sample(comp, Real{1}, Real{0}) - c;
    const Real sy = profile.sample(comp, Real{0}, Real{1}) - c;
    if (!(std::isfinite(c) && std::isfinite(sx) && std::isfinite(sy))) {
      throw std::invalid_argument{
          "background profile '" + name + "' produced a non-finite sample"};
    }
    // Two probes off the axes. Affine is not an assumption we are entitled to
    // make -- the sample() contract only says an affine profile's centre value
    // IS its element moment -- so a profile with curvature is refused by name
    // rather than silently linearized.
    static constexpr Real kProbes[2][2] = {{Real{2}, Real{-3}},
                                           {Real{-0.5}, Real{0.25}}};
    for (const auto& probe : kProbes) {
      const Real expected = c + (sx * probe[0] + sy * probe[1]);
      if (profile.sample(comp, probe[0], probe[1]) != expected) {
        throw std::invalid_argument{
            "background profile '" + name +
            "' is not affine over an element and therefore has no device "
            "sampling path; a nonlinear profile must supply its own element "
            "moment (see IMhdBackgroundProfile::sample)"};
      }
    }
    lowered.constant[comp] = c;
    lowered.slope_x[comp] = sx;
    lowered.slope_y[comp] = sy;
  }
  return lowered;
}

void build_background_from_profile(const MhdBackgroundBuildSpec& spec,
                                   MhdBackgroundField<Real>& out,
                                   backend::stream_t stream) {
  StatusWord status;
  launch_mhd_sample_background_profile(spec.grid, spec.profile,
                                       spec.magnetic_scale, out,
                                       status.device.device_ptr(), stream);
  throw_on_background_status(status.read(stream));
}

void scale_explicit_background(const MhdBackgroundBuildSpec& spec,
                               MhdBackgroundField<Real>& out,
                               backend::stream_t stream) {
  StatusWord status;
  launch_mhd_scale_background(out, spec.magnetic_scale,
                              status.device.device_ptr(), stream);
  throw_on_background_status(status.read(stream));
}

void build_background_from_corner_potential(
    const MhdBackgroundBuildSpec& spec, backend::DeviceBuffer<Real>& a_corners,
    MhdBackgroundField<Real>& out, backend::stream_t stream) {
  const std::size_t expected =
      static_cast<std::size_t>(spec.grid.pitch() + 1) *
      static_cast<std::size_t>(spec.grid.height() + 1);
  if (a_corners.size() != expected) {
    throw std::invalid_argument{
        "corner vector potential must cover the padded (pitch+1)x(height+1) "
        "corner grid"};
  }
  if (!std::isfinite(spec.b_scale)) {
    throw std::invalid_argument{
        "background_field.params.b_scale must be finite"};
  }

  StatusWord status;
  launch_mhd_scale_corner_potential(a_corners, spec.b_scale,
                                    status.device.device_ptr(), stream);
  throw_on_background_status(status.read(stream));

  if (spec.vacuum_project != 0) {
    // The projection runs in deck units, before B is converted to the solver's
    // variable: the linear solve is scale invariant, and the derived field
    // still undergoes the standard divergence validation after conversion.
    //
    // A strictly positive padded radial interval is required. The r=0 parity
    // closure needs a separate axis row in this elliptic operator and is
    // therefore deliberately not inferred.
    const Real inner = spec.grid.origin_x -
                       static_cast<Real>(spec.grid.nghost) * spec.grid.dx();
    if (!(std::isfinite(inner) && inner > Real{0})) {
      throw std::invalid_argument{
          "background_field.params.vacuum_project requires the full padded "
          "corner grid to lie at r > 0 (annular geometry)"};
    }
    const MhdVacuumProjectionReport report =
        launch_mhd_project_vacuum_potential(
            spec.grid, a_corners, kVacuumProjectionRelativeTolerance,
            status.device.device_ptr(), stream);
    throw_on_background_status(status.read(stream));
    if (report.field_scale_not_representable != 0) {
      throw std::invalid_argument{
          "projected field derivative scale is not representable"};
    }
    if (report.lost_definiteness != 0) {
      throw std::invalid_argument{
          "cylindrical vacuum projection failed: discrete operator lost "
          "positive definiteness"};
    }
    if (report.nonfinite_residual != 0) {
      throw std::invalid_argument{
          "cylindrical vacuum projection failed with a non-finite residual"};
    }
    if (!report.converged) {
      std::ostringstream message;
      message.setf(std::ios::scientific);
      message.precision(3);
      message << "cylindrical vacuum projection did not converge within "
              << report.max_iterations << " iterations (residual "
              << report.residual << ", target " << report.target << ")";
      throw std::invalid_argument{message.str()};
    }
  }

  launch_mhd_curl_corner_potential(spec.grid, spec.cylindrical, a_corners,
                                   spec.magnetic_scale, spec.bz0, out,
                                   status.device.device_ptr(), stream);
  throw_on_background_status(status.read(stream));
}

void build_background_from_conductors(const MhdBackgroundBuildSpec& spec,
                                      const core::IFieldSource& conductors,
                                      const numerics::IFieldEvaluator& evaluator,
                                      MhdBackgroundField<Real>& out,
                                      backend::stream_t stream) {
  if (!evaluator.provides_vector_potential()) {
    throw std::invalid_argument{
        "background_field.conductors requires an evaluator that models the "
        "vector potential A"};
  }
  // MHD-x is lab-X and MHD-y is lab-Z, so the out-of-plane potential is the
  // lab-Y component on the Y=0 slice. The corner grid covers the PADDED domain:
  // its low corner is one full halo below the interior origin.
  core::ObservationGrid corners;
  corners.origin = Vec3{
      spec.grid.origin_x - static_cast<Real>(spec.grid.nghost) * spec.grid.dx(),
      Real{0},
      spec.grid.origin_y - static_cast<Real>(spec.grid.nghost) * spec.grid.dy()};
  corners.spacing = Vec3{spec.grid.dx(), Real{0}, spec.grid.dy()};
  corners.dims = {spec.grid.pitch() + 1, 1, spec.grid.height() + 1};

  const core::DeviceVectorField potential =
      evaluator.evaluate_A(conductors, corners.to_device_point_cloud());
  const std::size_t expected =
      static_cast<std::size_t>(spec.grid.pitch() + 1) *
      static_cast<std::size_t>(spec.grid.height() + 1);
  if (potential.size() != expected) {
    throw std::invalid_argument{
        "background_field.conductors evaluator returned a vector potential of "
        "the wrong length for the padded corner grid"};
  }

  backend::DeviceBuffer<Real> a_corners(expected);
  StatusWord status;
  launch_mhd_extract_corner_potential(potential.y(), a_corners,
                                      status.device.device_ptr(), stream);
  const int extract_status = status.read(stream);
  if ((extract_status & kMhdBackgroundPotentialNotRepresentable) != 0) {
    throw std::invalid_argument{
        "background_field.conductors produced a non-finite vector potential"};
  }
  throw_on_background_status(extract_status);

  build_background_from_corner_potential(spec, a_corners, out, stream);
}

void validate_deck_background(const MhdBackgroundBuildSpec& spec,
                              const MhdBackgroundField<Real>& b0,
                              backend::stream_t stream) {
  const MhdBackgroundSolenoidalResult solenoidal =
      launch_mhd_validate_background_solenoidal(b0, spec.grid, spec.cylindrical,
                                                stream);
  if (solenoidal.nonfinite_sample != 0) {
    throw std::invalid_argument{
        "background_field must contain only finite values"};
  }
  const Real relative_linf = numerics::normalized_scaled_ratio(
      solenoidal.residual_linf, solenoidal.directional_scale_linf);
  if (!(std::isfinite(relative_linf) &&
        relative_linf <= numerics::kDiscreteSolenoidalTolerance)) {
    std::ostringstream message;
    message.setf(std::ios::scientific);
    message.precision(3);
    message << "background_field is not discretely divergence-free: maximum "
               "relative stencil defect "
            << relative_linf << " exceeds "
            << numerics::kDiscreteSolenoidalTolerance << ".";
    throw std::invalid_argument{message.str()};
  }
  throw_on_boundary_violation(
      launch_mhd_validate_background_boundaries(b0, spec.grid, spec.field_modes,
                                                stream)
          .violated_rules);
}

}  // namespace quasar::mhd
