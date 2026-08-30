#include "quasar/physics/pic/pic_solver.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/core/device_observations.hpp"
#include "quasar/core/observations.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace quasar::pic {

namespace {

enum class AxisParity { none, center_even, face_odd };

struct CylindricalRotation {
  Real cosine{};
  Real sine{};
  bool exact_quarter_turn{false};
};

constexpr Real kCovarianceTolerance =
    Real{4096} * std::numeric_limits<Real>::epsilon();

// Apply the covariance tolerance at the smallest normal scale as an absolute
// floor.  This accepts a few thousand representational ulps around exact zero
// without introducing a field-unit-dependent floor such as 1e-12 tesla.
constexpr Real kCovarianceAbsoluteFloor =
    kCovarianceTolerance * std::numeric_limits<Real>::min();

bool parity_close(Real actual, Real expected) {
  if (actual == expected) return true;
  const Real scale = std::max(std::abs(actual), std::abs(expected));
  if (scale == Real{0}) return true;
  const Real normalized_error = std::abs(actual / scale - expected / scale);
  return normalized_error
      <= Real{128} * std::numeric_limits<Real>::epsilon();
}

bool rotation_component_close(Real actual, Real expected,
                              Real operation_scale) {
  if (actual == expected) return true;
  const Real direct_error = std::abs(actual - expected);
  if (std::isfinite(direct_error)
      && direct_error <= kCovarianceAbsoluteFloor) return true;
  const Real scale = std::max(
      {std::abs(actual), std::abs(expected), operation_scale});
  if (scale == Real{0}) return true;
  return std::abs(actual / scale - expected / scale)
      <= kCovarianceTolerance;
}

Vec3 cylindrical_rotate(Vec3 value, std::string_view plane,
                        CylindricalRotation rotation) {
  // Preserve the quarter-turn as an exact signed permutation.  Besides avoiding
  // trigonometric noise, this makes each output component depend on exactly one
  // input component, so an unrelated large axial field cannot hide a bad small
  // transverse component.
  if (rotation.exact_quarter_turn) {
    if (plane == "xz") return Vec3{-value.y, value.x, value.z};
    return Vec3{value.z, value.y, -value.x};
  }
  const Real c = rotation.cosine;
  const Real s = rotation.sine;
  if (plane == "xz") {
    // Active right-handed rotation about lab z.
    return Vec3{std::fma(-s, value.y, c * value.x),
                std::fma(s, value.x, c * value.y), value.z};
  }
  // Active right-handed rotation about lab y.  At phi=0 this takes +lab x
  // toward -lab z, the cylindrical +phi convention for the lab x-y slice.
  return Vec3{std::fma(s, value.z, c * value.x), value.y,
              std::fma(-s, value.x, c * value.z)};
}

Vec3 cylindrical_rotation_scales(Vec3 value, std::string_view plane,
                                 CylindricalRotation rotation) {
  if (rotation.exact_quarter_turn) {
    if (plane == "xz") {
      return Vec3{std::abs(value.y), std::abs(value.x), std::abs(value.z)};
    }
    return Vec3{std::abs(value.z), std::abs(value.y), std::abs(value.x)};
  }
  // A general rotated component is a two-term sum.  Its roundoff floor may
  // scale with either contributing component, but never with the unrelated
  // symmetry-axis component.
  if (plane == "xz") {
    const Real transverse = std::max(std::abs(value.x), std::abs(value.y));
    return Vec3{transverse, transverse, std::abs(value.z)};
  }
  const Real transverse = std::max(std::abs(value.x), std::abs(value.z));
  return Vec3{transverse, std::abs(value.y), transverse};
}

CylindricalRotation noncommensurate_probe_rotation() {
  // pi/sqrt(2) is not a rational fraction of a full turn.  The exact quarter
  // turn catches signed-axis mistakes; this second probe rejects fields with
  // fourfold covariance (for example an m=4 mode) that are not axisymmetric.
  const Real angle = pi_v<Real> / std::sqrt(Real{2});
  return CylindricalRotation{std::cos(angle), std::sin(angle), false};
}

core::PointCloud rotated_cylindrical_points(
    const core::PointCloud& points, std::string_view plane,
    CylindricalRotation rotation) {
  core::PointCloud rotated;
  for (const Vec3 point : points.points()) {
    const Vec3 result = cylindrical_rotate(point, plane, rotation);
    if (!(std::isfinite(result.x) && std::isfinite(result.y)
          && std::isfinite(result.z))) {
      throw std::overflow_error{
          "sample_external_field: a rotated sample coordinate is not finite"};
    }
    rotated.add(result);
  }
  return rotated;
}

void enforce_cylindrical_covariance(const Field<Vec3>& meridional,
                                    const Field<Vec3>& rotated,
                                    std::string_view plane,
                                    std::string_view field_name,
                                    CylindricalRotation rotation) {
  if (meridional.size() != rotated.size()) {
    throw std::runtime_error{
        "sample_external_field: evaluator returned the wrong number of values"};
  }
  for (std::size_t sample = 0; sample < meridional.size(); ++sample) {
    const Vec3 expected = cylindrical_rotate(
        meridional[sample], plane, rotation);
    const Vec3 operation_scale = cylindrical_rotation_scales(
        meridional[sample], plane, rotation);
    const Vec3 actual = rotated[sample];
    if (!(std::isfinite(actual.x) && std::isfinite(actual.y)
          && std::isfinite(actual.z) && std::isfinite(expected.x)
          && std::isfinite(expected.y) && std::isfinite(expected.z))) {
      throw std::runtime_error{
          "sample_external_field: evaluator returned a non-finite field"};
    }
    if (!(rotation_component_close(actual.x, expected.x, operation_scale.x)
          && rotation_component_close(actual.y, expected.y, operation_scale.y)
          && rotation_component_close(actual.z, expected.z, operation_scale.z))) {
      throw std::invalid_argument{
        "sample_external_field: cylindrical " + std::string{field_name}
          + " is not rotationally covariant about the symmetry axis"};
    }
  }
}

void enforce_cylindrical_axis_parity(const Grid2D& g,
                                     std::vector<Real>& host,
                                     AxisParity parity,
                                     std::string_view component) {
  if (parity == AxisParity::none || g.origin_x != Real{0}) return;

  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    if (parity == AxisParity::face_odd) {
      if (!parity_close(host[g.index(0, j)], Real{0})) {
        throw std::invalid_argument{
            "sample_external_field: cylindrical " + std::string{component}
            + " must vanish at r=0 for a regular axisymmetric field"};
      }
      host[g.index(0, j)] = Real{0};
    }
    for (int gh = 1; gh <= g.nghost; ++gh) {
      const int mirror = parity == AxisParity::face_odd ? gh : gh - 1;
      const Real expected = parity == AxisParity::face_odd
                          ? -host[g.index(mirror, j)]
                          : host[g.index(mirror, j)];
      if (!parity_close(host[g.index(-gh, j)], expected)) {
        throw std::invalid_argument{
            "sample_external_field: evaluator is not regular and axisymmetric; "
            "cylindrical " + std::string{component}
            + (parity == AxisParity::face_odd
                   ? " violates odd radial parity"
                   : " violates even radial parity")};
      }
      // Make the prescribed ghosts bit-for-bit consistent with the closure used
      // by the evolved field after accepting evaluator roundoff at the tolerance
      // above.
      host[g.index(-gh, j)] = expected;
    }
  }
}

// Builds one component's Yee-lattice sample points, already scaled from internal
// length units to SI. Offsets are in cells from the domain origin (0=face/node,
// 1/2=centre).
// Folding the scale in here avoids a second full PointCloud copy (the former
// to_si_points pass).
//
core::PointCloud yee_points(const Grid2D& g, Real offset_x, Real offset_y,
                            Real length_scale, std::string_view plane) {
  core::PointCloud pts;
  // Evaluate the complete padded lattice, not just its physical subset. A
  // wall-adjacent finite-size gather legitimately reads the boundary-filled
  // ghosts of the evolved field. The prescribed field must therefore provide
  // values at those same coordinates; edge replication would reduce a
  // nonuniform analytic/file field to a first-order constant continuation.
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      // Keep the affine coordinate as one rounded operation.  On a translated
      // grid, the product can lie outside Real even though cancellation with the
      // origin leaves a representable coordinate (for example
      // -DBL_MAX + 1.5*DBL_MAX).  A multiply followed by an add spuriously turns
      // that valid Yee point into infinity.
      const Real internal_a = std::fma(
          static_cast<Real>(i) + offset_x, g.dx(), g.origin_x);
      const Real internal_b = std::fma(
          static_cast<Real>(j) + offset_y, g.dy(), g.origin_y);
      const Real a = internal_a * length_scale;
      const Real b = internal_b * length_scale;
      if (!(std::isfinite(a) && std::isfinite(b))) {
        throw std::overflow_error{
            "sample_external_field: a scaled sample coordinate is not finite"};
      }
      if ((internal_a != Real{0} && a == Real{0})
          || (internal_b != Real{0} && b == Real{0})) {
        throw std::underflow_error{
            "sample_external_field: a nonzero sample coordinate underflows "
            "after length scaling"};
      }
      // "xy": sample the lab z=0 slice at (x, y, 0).
      // "xz": sample the lab y=0 meridional slice at (x, 0, z).
      if (plane == "xz") {
        pts.add(Vec3{a, Real{0}, b});
      } else {
        pts.add(Vec3{a, b, Real{0}});
      }
    }
  }
  return pts;
}

// Materializes one PIC-frame component, reading lab axis `axis` (0=x,1=y,2=z)
// and scaling by `sign` (+1 / -1) for the right-handed frame map.  Sampling and
// validation complete for all six components before any device buffer is
// changed, so a rejected evaluator cannot leave a partially updated prescribed
// field behind.
std::vector<Real> materialize_component(
    const Grid2D& g, const Field<Vec3>& values, int axis,
    Real sign, Real field_scale, AxisParity parity = AxisParity::none,
    std::string_view component = {}) {
  if (values.size() != g.storage_size()) {
    throw std::runtime_error{
        "sample_external_field: evaluator returned the wrong number of values"};
  }
  std::vector<Real> host(g.storage_size(), Real{0});
  std::size_t sample = 0;
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i, ++sample) {
      const Vec3 v = values[sample];
      if (!(std::isfinite(v.x) && std::isfinite(v.y)
            && std::isfinite(v.z))) {
        throw std::runtime_error{
            "sample_external_field: evaluator returned a non-finite field"};
      }
      const Real c = axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
      const Real mapped = sign * c / field_scale;
      if (!std::isfinite(mapped)) {
        throw std::overflow_error{
            "sample_external_field: scaled field value is not finite"};
      }
      if (c != Real{0} && mapped == Real{0}) {
        throw std::underflow_error{
            "sample_external_field: a nonzero field value underflows solver "
            "units"};
      }
      host[g.index(i, j)] = mapped;
    }
  }
  enforce_cylindrical_axis_parity(g, host, parity, component);
  return host;
}

bool finite_matrix(const Mat3x3& gradient) {
  const Vec3 rows[] = {gradient.r0, gradient.r1, gradient.r2};
  for (const Vec3 row : rows) {
    if (!(std::isfinite(row.x) && std::isfinite(row.y)
          && std::isfinite(row.z))) return false;
  }
  return true;
}

void enforce_continuous_magnetic_solenoidality(
    const Field<Mat3x3>& gradients, std::size_t expected_size) {
  if (gradients.size() != expected_size) {
    throw std::runtime_error{
        "sample_external_field: evaluator returned the wrong number of gradients"};
  }
  constexpr long double tolerance = Real{4096}
      * static_cast<long double>(std::numeric_limits<Real>::epsilon());
  for (std::size_t sample = 0; sample < gradients.size(); ++sample) {
    const Mat3x3 gradient = gradients[sample];
    if (!finite_matrix(gradient)) {
      throw std::runtime_error{
          "sample_external_field: evaluator returned a non-finite magnetic-field "
          "gradient"};
    }
    const Real scale = std::max(
        {std::abs(gradient.r0.x), std::abs(gradient.r1.y),
         std::abs(gradient.r2.z)});
    if (scale == Real{0}) continue;
    const long double dxx = static_cast<long double>(gradient.r0.x / scale);
    const long double dyy = static_cast<long double>(gradient.r1.y / scale);
    const long double dzz = static_cast<long double>(gradient.r2.z / scale);
    const long double residual = std::abs(dxx + dyy + dzz);
    const long double magnitude =
        std::abs(dxx) + std::abs(dyy) + std::abs(dzz);
    if (residual > tolerance * magnitude) {
      throw std::invalid_argument{
          "sample_external_field: cylindrical magnetic field is not "
          "solenoidal: trace(grad B) is nonzero at sample "
          + std::to_string(sample)};
    }
  }
}

bool identically_zero(const Field<Vec3>& field) {
  for (const Vec3 value : field) {
    if (value.x != Real{0} || value.y != Real{0} || value.z != Real{0}) {
      return false;
    }
  }
  return true;
}

}  // namespace

void sample_external_field(numerics::IFieldEvaluator& evaluator,
                           const core::IFieldSource& source,
                           YeeField2D<Real>& external_fields,
                           Real length_scale, Real e_field_scale,
                           Real b_field_scale, std::string_view plane,
                           std::string_view geometry, int fdtd_order) {
  const Grid2D g = external_fields.grid;
  g.validate();
  if (plane != "xy" && plane != "xz") {
    throw std::invalid_argument{
        "sample_external_field: plane must be 'xy' or 'xz'"};
  }
  if (geometry != "cartesian" && geometry != "cylindrical") {
    throw std::invalid_argument{
        "sample_external_field: geometry must be 'cartesian' or 'cylindrical'"};
  }
  // Validate the public order even for Cartesian calls.  It is part of the
  // sampler contract and silently accepting a nonsensical order here would make
  // a later switch to cylindrical geometry change validation semantics.
  const int required_halo = required_nghost(fdtd_order);
  const bool cylindrical = geometry == "cylindrical";
  if (cylindrical && g.origin_x < Real{0}) {
    throw std::invalid_argument{
        "sample_external_field: cylindrical radial origin must be non-negative"};
  }
  if (cylindrical && g.nghost < required_halo) {
    throw std::invalid_argument{
        "sample_external_field: grid halo is too small for fdtd_order="
        + std::to_string(fdtd_order)};
  }
  if (!(std::isfinite(length_scale) && length_scale > Real{0})
      || !(std::isfinite(e_field_scale) && e_field_scale != Real{0})
      || !(std::isfinite(b_field_scale) && b_field_scale != Real{0})) {
    throw std::invalid_argument{
        "sample_external_field: scales must be finite, length_scale must be "
        "positive, and field scales must be nonzero"};
  }

  // Map each PIC-frame component (ex/ey/ez, bx/by/bz) to a lab axis and sign.
  // "xy": identity.  Cartesian "xz" uses the right-handed grid triad
  // (x,z,-y).  Cylindrical storage is instead the physical (r,z,phi) component
  // order and the pusher explicitly permutes it to the right-handed
  // (r,phi,z) basis, so at the phi=0 sampling plane +phi is +lab-y.
  struct Map { int axis; Real sign; };
  Map mx, my, mz;
  if (plane == "xz") {
    mx = {0, Real{+1}};   // pic-x  <- lab x
    my = {2, Real{+1}};   // pic-y  <- lab z
    mz = {1, geometry == "cylindrical" ? Real{+1} : Real{-1}};
  } else {
    mx = {0, Real{+1}};
    my = {1, Real{+1}};
    // For cylindrical (r,z) mapped to lab (x,y), +phi is -lab-z so that
    // e_r x e_phi = e_z.  Cartesian xy keeps the ordinary +lab-z component.
    mz = {2, geometry == "cylindrical" ? Real{-1} : Real{+1}};
  }

  struct Offset { Real x; Real y; };
  const Offset ex_o{Real{0}, Real{0.5}};
  const Offset ey_o{Real{0.5}, Real{0}};
  const Offset ez_o = cylindrical ? ex_o : Offset{Real{0.5}, Real{0.5}};
  const Offset bx_o = cylindrical ? Offset{Real{0}, Real{0}} : ey_o;
  const Offset by_o = cylindrical ? Offset{Real{0.5}, Real{0.5}} : ex_o;
  const Offset bz_o{Real{0}, Real{0}};
  const CylindricalRotation quarter_turn{Real{0}, Real{1}, true};
  const CylindricalRotation noncommensurate =
      noncommensurate_probe_rotation();
  const bool provides_grad_b = evaluator.provides_grad_B();

  // The evaluator interface is device-resident: it takes a DevicePointCloud and
  // returns SoA planes. The sample-point construction and the covariance /
  // parity / solenoidality checks below are still host loops, so each result is
  // brought back with .to_host() here. Moving those loops onto the device (and
  // deleting these downloads) is the remainder of this port.
  const auto upload = [](const core::PointCloud& pts) {
    return core::DevicePointCloud::upload(pts);
  };
  const auto eval_e = [&](Offset o) {
    const auto points = yee_points(g, o.x, o.y, length_scale, plane);
    auto values = evaluator.evaluate_E(source, upload(points)).to_host();
    if (cylindrical) {
      const auto quarter_turn_points =
          rotated_cylindrical_points(points, plane, quarter_turn);
      const auto quarter_turn_values =
          evaluator.evaluate_E(source, upload(quarter_turn_points)).to_host();
      enforce_cylindrical_covariance(
          values, quarter_turn_values, plane, "electric field", quarter_turn);
      const auto noncommensurate_points =
          rotated_cylindrical_points(points, plane, noncommensurate);
      const auto noncommensurate_values =
          evaluator.evaluate_E(source, upload(noncommensurate_points)).to_host();
      enforce_cylindrical_covariance(
          values, noncommensurate_values, plane, "electric field",
          noncommensurate);
    }
    return values;
  };
  const auto eval_b = [&](Offset o) {
    const auto points = yee_points(g, o.x, o.y, length_scale, plane);
    const auto device_points = upload(points);
    auto values = evaluator.evaluate_B(source, device_points).to_host();
    if (cylindrical) {
      if (provides_grad_b) {
        const auto gradients =
            evaluator.evaluate_grad_B(source, device_points).to_host();
        enforce_continuous_magnetic_solenoidality(
            gradients, points.size());
      }
      const auto quarter_turn_points =
          rotated_cylindrical_points(points, plane, quarter_turn);
      const auto quarter_turn_values =
          evaluator.evaluate_B(source, upload(quarter_turn_points)).to_host();
      enforce_cylindrical_covariance(
          values, quarter_turn_values, plane, "magnetic field", quarter_turn);
      const auto noncommensurate_points =
          rotated_cylindrical_points(points, plane, noncommensurate);
      const auto noncommensurate_values =
          evaluator.evaluate_B(source, upload(noncommensurate_points)).to_host();
      enforce_cylindrical_covariance(
          values, noncommensurate_values, plane, "magnetic field",
          noncommensurate);
    }
    return values;
  };
  const auto e_on_ex = eval_e(ex_o);
  const auto e_on_ey = eval_e(ey_o);
  const auto ex = materialize_component(
      g, e_on_ex, mx.axis, mx.sign, e_field_scale,
      cylindrical ? AxisParity::face_odd : AxisParity::none, "Er");
  const auto ey = materialize_component(
      g, e_on_ey, my.axis, my.sign, e_field_scale,
      cylindrical ? AxisParity::center_even : AxisParity::none, "Ez");
  std::vector<Real> ez;
  if (cylindrical) {
    ez = materialize_component(
        g, e_on_ex, mz.axis, mz.sign, e_field_scale,
        AxisParity::face_odd, "Ephi");
  } else {
    const auto e_on_ez = eval_e(ez_o);
    ez = materialize_component(
        g, e_on_ez, mz.axis, mz.sign, e_field_scale);
  }

  const auto b_on_bx = eval_b(bx_o);
  const auto b_on_by = eval_b(by_o);
  if (cylindrical && !provides_grad_b
      && !(identically_zero(b_on_bx) && identically_zero(b_on_by))) {
    throw std::invalid_argument{
        "sample_external_field: a nonzero cylindrical magnetic field requires "
        "an evaluator with a trustworthy gradient (provides_grad_B() must "
        "return true)"};
  }
  const auto br = materialize_component(
      g, b_on_bx, mx.axis, mx.sign, b_field_scale,
      cylindrical ? AxisParity::face_odd : AxisParity::none, "Br");
  const auto bz = materialize_component(
      g, b_on_by, my.axis, my.sign, b_field_scale,
      cylindrical ? AxisParity::center_even : AxisParity::none, "Bz");
  std::vector<Real> bphi;
  if (cylindrical) {
    bphi = materialize_component(
        g, b_on_bx, mz.axis, mz.sign, b_field_scale,
        AxisParity::face_odd, "Bphi");
  } else {
    const auto b_on_bz = eval_b(bz_o);
    bphi = materialize_component(
        g, b_on_bz, mz.axis, mz.sign, b_field_scale);
  }

  external_fields.ex.copy_from_host(ex.data(), ex.size());
  external_fields.ey.copy_from_host(ey.data(), ey.size());
  external_fields.ez.copy_from_host(ez.data(), ez.size());
  external_fields.bx.copy_from_host(br.data(), br.size());
  external_fields.by.copy_from_host(bz.data(), bz.size());
  external_fields.bz.copy_from_host(bphi.data(), bphi.size());
}

}  // namespace quasar::pic
