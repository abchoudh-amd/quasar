#pragma once

// Base class for test-double field evaluators written in host code.
//
// IFieldEvaluator is device-resident, but a test double exists to supply a
// specific analytic field to the code under test -- the double's own arithmetic
// is never what is being tested. Writing each one as a HIP kernel would add a
// kernel per fake for no coverage. This base implements the device interface by
// moving the points to the host, calling a host hook, and moving the answer
// back, so a fake stays a dozen readable lines.
//
// Production evaluators must NOT do this: for them the round trip is the cost
// the device interface exists to eliminate. That is why this lives under tests/
// and not on the interface.

#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/core/device_observations.hpp"
#include "quasar/core/field.hpp"
#include "quasar/core/field_source.hpp"
#include "quasar/core/observations.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/field_evaluator.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quasar::test {

namespace detail {

inline core::PointCloud to_host_cloud(const core::DevicePointCloud& points) {
  const core::PointSoA soa = points.to_host();
  core::PointCloud out;
  for (std::size_t i = 0; i < soa.n_points(); ++i) {
    out.add(Vec3{soa.px[i], soa.py[i], soa.pz[i]});
  }
  return out;
}

inline core::DeviceVectorField to_device(const Field<Vec3>& values) {
  const std::size_t n = values.size();
  core::DeviceVectorField out(n, backend::uninitialized);
  if (n == 0) return out;
  std::vector<Real> plane(n);
  for (std::size_t i = 0; i < n; ++i) plane[i] = values[i].x;
  backend::device_memcpy_h2d(out.x(), plane.data(), n * sizeof(Real));
  for (std::size_t i = 0; i < n; ++i) plane[i] = values[i].y;
  backend::device_memcpy_h2d(out.y(), plane.data(), n * sizeof(Real));
  for (std::size_t i = 0; i < n; ++i) plane[i] = values[i].z;
  backend::device_memcpy_h2d(out.z(), plane.data(), n * sizeof(Real));
  return out;
}

inline core::DeviceTensorField to_device(const Field<Mat3x3>& values) {
  const std::size_t n = values.size();
  core::DeviceTensorField out(n, backend::uninitialized);
  if (n == 0) return out;
  // Component-major, matching DeviceTensorField's documented layout.
  std::vector<Real> staged(core::checked_tensor_values(n));
  for (std::size_t p = 0; p < n; ++p) {
    const Vec3 rows[3] = {values[p].r0, values[p].r1, values[p].r2};
    for (int row = 0; row < 3; ++row) {
      const Real entries[3] = {rows[row].x, rows[row].y, rows[row].z};
      for (int col = 0; col < 3; ++col) {
        staged[static_cast<std::size_t>(3 * row + col) * n + p] = entries[col];
      }
    }
  }
  backend::device_memcpy_h2d(out.data(), staged.data(),
                             staged.size() * sizeof(Real));
  return out;
}

}  // namespace detail

class HostFieldEvaluator : public numerics::IFieldEvaluator {
 public:
  core::DeviceVectorField evaluate_B(
      const core::IFieldSource& source,
      const core::DevicePointCloud& points) const final {
    return detail::to_device(host_B(source, detail::to_host_cloud(points)));
  }

  core::DeviceVectorField evaluate_E(
      const core::IFieldSource& source,
      const core::DevicePointCloud& points) const final {
    return detail::to_device(host_E(source, detail::to_host_cloud(points)));
  }

  core::DeviceVectorField evaluate_A(
      const core::IFieldSource& source,
      const core::DevicePointCloud& points) const final {
    return detail::to_device(host_A(source, detail::to_host_cloud(points)));
  }

  core::DeviceTensorField evaluate_grad_B(
      const core::IFieldSource& source,
      const core::DevicePointCloud& points) const final {
    return detail::to_device(
        host_grad_B(source, detail::to_host_cloud(points)));
  }

 protected:
  virtual Field<Vec3> host_B(const core::IFieldSource&,
                             const core::PointCloud&) const = 0;

  // The defaults mirror IFieldEvaluator's: zero E, zero Jacobian, no A.
  virtual Field<Vec3> host_E(const core::IFieldSource&,
                             const core::PointCloud& points) const {
    return Field<Vec3>(points.size());
  }
  virtual Field<Mat3x3> host_grad_B(const core::IFieldSource&,
                                    const core::PointCloud& points) const {
    return Field<Mat3x3>(points.size());
  }
  virtual Field<Vec3> host_A(const core::IFieldSource&,
                             const core::PointCloud&) const {
    throw std::runtime_error{
        "this field evaluator does not provide a vector potential A"};
  }
};

}  // namespace quasar::test
