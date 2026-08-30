#include "quasar/physics/pic/pic_solver.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/core/device_observations.hpp"
#include "quasar/core/observations.hpp"
#include "quasar/physics/pic/kernels.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace quasar::pic {

namespace {

using ::quasar::backend::DeviceBuffer;
using ::quasar::core::DevicePointCloud;
using ::quasar::core::DeviceTensorField;
using ::quasar::core::DeviceVectorField;

enum class AxisParity { none, center_even, face_odd };

struct CylindricalRotation {
  Real cosine{};
  Real sine{};
  bool exact_quarter_turn{false};
};

// Threshold configuration. These are scalars, independent of the grid and the
// particle count, so they stay on the host and are handed to the kernels that
// apply them.
constexpr Real kCovarianceTolerance =
    Real{4096} * std::numeric_limits<Real>::epsilon();

// Apply the covariance tolerance at the smallest normal scale as an absolute
// floor.  This accepts a few thousand representational ulps around exact zero
// without introducing a field-unit-dependent floor such as 1e-12 tesla.
constexpr Real kCovarianceAbsoluteFloor =
    kCovarianceTolerance * std::numeric_limits<Real>::min();

constexpr Real kParityTolerance =
    Real{128} * std::numeric_limits<Real>::epsilon();

constexpr Real kSolenoidalTolerance =
    Real{4096} * std::numeric_limits<Real>::epsilon();

CylindricalRotation noncommensurate_probe_rotation() {
  // pi/sqrt(2) is not a rational fraction of a full turn.  The exact quarter
  // turn catches signed-axis mistakes; this second probe rejects fields with
  // fourfold covariance (for example an m=4 mode) that are not axisymmetric.
  const Real angle = pi_v<Real> / std::sqrt(Real{2});
  return CylindricalRotation{std::cos(angle), std::sin(angle), false};
}

int checked_sample_count(std::size_t n) {
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error{
        "sample_external_field: sample count exceeds the signed kernel-index "
        "limit"};
  }
  return static_cast<int>(n);
}

DeviceBuffer<int> fresh_status() { return DeviceBuffer<int>(1); }

// Reading a status word back is the only place this file synchronizes, and it
// is where the exception the host loops used to throw inline is raised instead.
int read_status(const DeviceBuffer<int>& status) {
  int value = 0;
  status.copy_to_host(&value, 1);
  return value;
}

// One component's Yee-lattice sample points over the whole padded lattice,
// already scaled from internal length units to SI. Offsets are in cells from
// the domain origin (0 = face/node, 1/2 = centre).
DevicePointCloud yee_points(const Grid2D& g, Real offset_x, Real offset_y,
                            Real length_scale, bool plane_is_xz) {
  DevicePointCloud points(g.storage_size());
  auto status = fresh_status();
  ::launch_pic_yee_points(g, offset_x, offset_y, length_scale,
                          plane_is_xz ? 1 : 0, points.x(), points.y(),
                          points.z(), status.device_ptr(), nullptr);
  const int flags = read_status(status);
  if ((flags & 1) != 0) {
    throw std::overflow_error{
        "sample_external_field: a scaled sample coordinate is not finite"};
  }
  if ((flags & 2) != 0) {
    throw std::underflow_error{
        "sample_external_field: a nonzero sample coordinate underflows "
        "after length scaling"};
  }
  return points;
}

DevicePointCloud rotated_points(const DevicePointCloud& points,
                                bool plane_is_xz,
                                CylindricalRotation rotation) {
  const int M = checked_sample_count(points.size());
  DevicePointCloud rotated(points.size());
  auto status = fresh_status();
  ::launch_pic_rotate_points(plane_is_xz ? 1 : 0, rotation.cosine,
                             rotation.sine,
                             rotation.exact_quarter_turn ? 1 : 0, points.x(),
                             points.y(), points.z(), M, rotated.x(),
                             rotated.y(), rotated.z(), status.device_ptr(),
                             nullptr);
  if ((read_status(status) & 1) != 0) {
    throw std::overflow_error{
        "sample_external_field: a rotated sample coordinate is not finite"};
  }
  return rotated;
}

void enforce_cylindrical_covariance(const DeviceVectorField& meridional,
                                    const DeviceVectorField& rotated,
                                    bool plane_is_xz,
                                    std::string_view field_name,
                                    CylindricalRotation rotation) {
  if (meridional.size() != rotated.size()) {
    throw std::runtime_error{
        "sample_external_field: evaluator returned the wrong number of values"};
  }
  const int M = checked_sample_count(meridional.size());
  auto status = fresh_status();
  ::launch_pic_check_rotational_covariance(
      plane_is_xz ? 1 : 0, rotation.cosine, rotation.sine,
      rotation.exact_quarter_turn ? 1 : 0, meridional.x(), meridional.y(),
      meridional.z(), rotated.x(), rotated.y(), rotated.z(), M,
      kCovarianceTolerance, kCovarianceAbsoluteFloor, status.device_ptr(),
      nullptr);
  const int flags = read_status(status);
  if ((flags & 1) != 0) {
    throw std::runtime_error{
        "sample_external_field: evaluator returned a non-finite field"};
  }
  if ((flags & 2) != 0) {
    throw std::invalid_argument{
        "sample_external_field: cylindrical " + std::string{field_name}
        + " is not rotationally covariant about the symmetry axis"};
  }
}

void enforce_continuous_magnetic_solenoidality(const DeviceTensorField& grad,
                                               std::size_t expected_size) {
  if (grad.size() != expected_size) {
    throw std::runtime_error{
        "sample_external_field: evaluator returned the wrong number of gradients"};
  }
  const int M = checked_sample_count(grad.size());
  auto status = fresh_status();
  ::launch_pic_check_continuous_solenoidality(
      grad.data(), M, kSolenoidalTolerance, status.device_ptr(), nullptr);
  const int flags = read_status(status);
  if ((flags & 1) != 0) {
    throw std::runtime_error{
        "sample_external_field: evaluator returned a non-finite magnetic-field "
        "gradient"};
  }
  if ((flags & 2) != 0) {
    // The host version named the offending sample index. A predicate
    // OR-reduced across all points cannot, and recovering it would need a
    // second pass to locate the first failure -- which is diagnosis, not
    // validation. The message keeps the physics and drops the index.
    throw std::invalid_argument{
        "sample_external_field: cylindrical magnetic field is not "
        "solenoidal: trace(grad B) is nonzero"};
  }
}

bool identically_zero(const DeviceVectorField& field) {
  const int M = checked_sample_count(field.size());
  auto flag = fresh_status();
  ::launch_pic_field_has_nonzero(field.x(), field.y(), field.z(), M,
                                 flag.device_ptr(), nullptr);
  return read_status(flag) == 0;
}

// Materializes one PIC-frame component into `out`, reading lab axis `axis`
// (0=x, 1=y, 2=z) and scaling by `sign` (+1 / -1) for the right-handed frame
// map.
//
// Sampling and validation complete for all six components before any field
// buffer is changed, so a rejected evaluator cannot leave a partially updated
// prescribed field behind. That is why `out` is scratch rather than the
// destination YeeField2D plane.
void materialize_component(const Grid2D& g, const DeviceVectorField& values,
                           int axis, Real sign, Real field_scale,
                           DeviceBuffer<Real>& out,
                           AxisParity parity = AxisParity::none,
                           std::string_view component = {}) {
  if (values.size() != g.storage_size()) {
    throw std::runtime_error{
        "sample_external_field: evaluator returned the wrong number of values"};
  }
  const int total = checked_sample_count(g.storage_size());
  auto status = fresh_status();
  ::launch_pic_materialize_external_component(
      values.x(), values.y(), values.z(), total, axis, sign, field_scale,
      out.device_ptr(), status.device_ptr(), nullptr);
  const int flags = read_status(status);
  if ((flags & 1) != 0) {
    throw std::runtime_error{
        "sample_external_field: evaluator returned a non-finite field"};
  }
  if ((flags & 2) != 0) {
    throw std::overflow_error{
        "sample_external_field: scaled field value is not finite"};
  }
  if ((flags & 4) != 0) {
    throw std::underflow_error{
        "sample_external_field: a nonzero field value underflows solver units"};
  }

  if (parity == AxisParity::none || g.origin_x != Real{0}) return;

  auto parity_status = fresh_status();
  const bool face_odd = parity == AxisParity::face_odd;
  ::launch_pic_enforce_axis_parity(g, out.device_ptr(), face_odd ? 1 : 0,
                                   kParityTolerance,
                                   parity_status.device_ptr(), nullptr);
  const int parity_flags = read_status(parity_status);
  if ((parity_flags & 1) != 0) {
    throw std::invalid_argument{
        "sample_external_field: cylindrical " + std::string{component}
        + " must vanish at r=0 for a regular axisymmetric field"};
  }
  if ((parity_flags & 2) != 0) {
    throw std::invalid_argument{
        "sample_external_field: evaluator is not regular and axisymmetric; "
        "cylindrical " + std::string{component}
        + (face_odd ? " violates odd radial parity"
                    : " violates even radial parity")};
  }
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
  const bool plane_is_xz = plane == "xz";
  if (plane_is_xz) {
    mx = {0, Real{+1}};   // pic-x  <- lab x
    my = {2, Real{+1}};   // pic-y  <- lab z
    mz = {1, cylindrical ? Real{+1} : Real{-1}};
  } else {
    mx = {0, Real{+1}};
    my = {1, Real{+1}};
    // For cylindrical (r,z) mapped to lab (x,y), +phi is -lab-z so that
    // e_r x e_phi = e_z.  Cartesian xy keeps the ordinary +lab-z component.
    mz = {2, cylindrical ? Real{-1} : Real{+1}};
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

  const auto eval_e = [&](Offset o) {
    const auto points = yee_points(g, o.x, o.y, length_scale, plane_is_xz);
    auto values = evaluator.evaluate_E(source, points);
    if (cylindrical) {
      for (const CylindricalRotation rotation :
           {quarter_turn, noncommensurate}) {
        const auto probe_points = rotated_points(points, plane_is_xz, rotation);
        const auto probe_values = evaluator.evaluate_E(source, probe_points);
        enforce_cylindrical_covariance(values, probe_values, plane_is_xz,
                                       "electric field", rotation);
      }
    }
    return values;
  };
  const auto eval_b = [&](Offset o) {
    const auto points = yee_points(g, o.x, o.y, length_scale, plane_is_xz);
    auto values = evaluator.evaluate_B(source, points);
    if (cylindrical) {
      if (provides_grad_b) {
        enforce_continuous_magnetic_solenoidality(
            evaluator.evaluate_grad_B(source, points), points.size());
      }
      for (const CylindricalRotation rotation :
           {quarter_turn, noncommensurate}) {
        const auto probe_points = rotated_points(points, plane_is_xz, rotation);
        const auto probe_values = evaluator.evaluate_B(source, probe_points);
        enforce_cylindrical_covariance(values, probe_values, plane_is_xz,
                                       "magnetic field", rotation);
      }
    }
    return values;
  };

  const std::size_t storage = g.storage_size();
  DeviceBuffer<Real> ex(storage), ey(storage), ez(storage);
  DeviceBuffer<Real> br(storage), bz(storage), bphi(storage);

  const auto e_on_ex = eval_e(ex_o);
  const auto e_on_ey = eval_e(ey_o);
  materialize_component(g, e_on_ex, mx.axis, mx.sign, e_field_scale, ex,
                        cylindrical ? AxisParity::face_odd : AxisParity::none,
                        "Er");
  materialize_component(g, e_on_ey, my.axis, my.sign, e_field_scale, ey,
                        cylindrical ? AxisParity::center_even
                                    : AxisParity::none,
                        "Ez");
  if (cylindrical) {
    materialize_component(g, e_on_ex, mz.axis, mz.sign, e_field_scale, ez,
                          AxisParity::face_odd, "Ephi");
  } else {
    materialize_component(g, eval_e(ez_o), mz.axis, mz.sign, e_field_scale, ez);
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
  materialize_component(g, b_on_bx, mx.axis, mx.sign, b_field_scale, br,
                        cylindrical ? AxisParity::face_odd : AxisParity::none,
                        "Br");
  materialize_component(g, b_on_by, my.axis, my.sign, b_field_scale, bz,
                        cylindrical ? AxisParity::center_even
                                    : AxisParity::none,
                        "Bz");
  if (cylindrical) {
    materialize_component(g, b_on_bx, mz.axis, mz.sign, b_field_scale, bphi,
                          AxisParity::face_odd, "Bphi");
  } else {
    materialize_component(g, eval_b(bz_o), mz.axis, mz.sign, b_field_scale,
                          bphi);
  }

  // Publish only now that all six components have validated. Device to device:
  // the sampled field never touches host memory at any point above.
  const std::size_t bytes = storage * sizeof(Real);
  const auto publish = [&](DeviceBuffer<Real>& dst,
                           const DeviceBuffer<Real>& src) {
    backend::device_memcpy_d2d_async(dst.device_ptr(), src.device_ptr(), bytes,
                                     nullptr);
  };
  publish(external_fields.ex, ex);
  publish(external_fields.ey, ey);
  publish(external_fields.ez, ez);
  publish(external_fields.bx, br);
  publish(external_fields.by, bz);
  publish(external_fields.bz, bphi);
  backend::device_synchronize(nullptr);
}

}  // namespace quasar::pic
