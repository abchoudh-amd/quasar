#include "quasar/core/device_observations.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace quasar::core {

namespace {

// Every plane of every container here is sized from one count. Guard the
// products that turn that count into a byte size before any allocation, so a
// caller cannot silently wrap around into a short buffer that a kernel then
// writes past.
std::size_t checked_scale(std::size_t n_points, std::size_t components,
                          const char* what) {
  if (n_points != 0
      && components > std::numeric_limits<std::size_t>::max() / n_points) {
    throw std::length_error{std::string{"quasar::core::"} + what
                            + ": value count is not representable"};
  }
  return n_points * components;
}

}  // namespace

std::size_t checked_tensor_values(std::size_t n_points) {
  return checked_scale(n_points, 9, "DeviceTensorField");
}

// -- DevicePointCloud -------------------------------------------------------

DevicePointCloud::DevicePointCloud(std::size_t n_points)
    : px_(n_points), py_(n_points), pz_(n_points), n_points_{n_points} {}

DevicePointCloud DevicePointCloud::upload(const PointSoA& points,
                                          backend::stream_t stream) {
  // Reject mismatched planes and non-finite coordinates before sizing anything
  // from px alone; otherwise a shorter plane is read past its host allocation
  // by the copy below.
  points.validate();
  const std::size_t n = points.n_points();

  DevicePointCloud out;
  out.px_ = backend::DeviceBuffer<Real>(n, backend::uninitialized);
  out.py_ = backend::DeviceBuffer<Real>(n, backend::uninitialized);
  out.pz_ = backend::DeviceBuffer<Real>(n, backend::uninitialized);
  out.n_points_ = n;
  if (n == 0) return out;

  out.px_.copy_from_host_async(points.px.data(), n, stream);
  out.py_.copy_from_host_async(points.py.data(), n, stream);
  out.pz_.copy_from_host_async(points.pz.data(), n, stream);
  // The caller owns `points` and is entitled to drop it the moment this
  // returns, so the asynchronous copies must be complete, not merely queued.
  backend::device_synchronize(stream);
  return out;
}

DevicePointCloud DevicePointCloud::upload(const PointCloud& points,
                                          backend::stream_t stream) {
  return upload(points.to_point_soa(), stream);
}

PointSoA DevicePointCloud::to_host(backend::stream_t stream) const {
  PointSoA out;
  out.px.resize(n_points_);
  out.py.resize(n_points_);
  out.pz.resize(n_points_);
  if (n_points_ == 0) return out;
  px_.copy_to_host_async(out.px.data(), n_points_, stream);
  py_.copy_to_host_async(out.py.data(), n_points_, stream);
  pz_.copy_to_host_async(out.pz.data(), n_points_, stream);
  backend::device_synchronize(stream);
  return out;
}

// -- DeviceVectorField ------------------------------------------------------

DeviceVectorField::DeviceVectorField(std::size_t n_points)
    : x_(n_points), y_(n_points), z_(n_points), n_points_{n_points} {}

DeviceVectorField::DeviceVectorField(std::size_t n_points,
                                     backend::uninitialized_t)
    : x_(n_points, backend::uninitialized),
      y_(n_points, backend::uninitialized),
      z_(n_points, backend::uninitialized),
      n_points_{n_points} {}

Field<Vec3> DeviceVectorField::to_host(backend::stream_t stream) const {
  Field<Vec3> out(n_points_);
  if (n_points_ == 0) return out;

  // One staging allocation holding three contiguous planes, then a single
  // transpose into the AoS Field. Three separate host vectors would cost three
  // allocations for the same traffic.
  std::vector<Real> staged(checked_scale(n_points_, 3, "DeviceVectorField"));
  const std::size_t n = n_points_;
  x_.copy_to_host_async(staged.data(),         n, stream);
  y_.copy_to_host_async(staged.data() + n,     n, stream);
  z_.copy_to_host_async(staged.data() + 2 * n, n, stream);
  backend::device_synchronize(stream);

  for (std::size_t i = 0; i < n; ++i) {
    out[i] = Vec3{staged[i], staged[n + i], staged[2 * n + i]};
  }
  return out;
}

// -- DeviceTensorField ------------------------------------------------------

DeviceTensorField::DeviceTensorField(std::size_t n_points)
    : g_(checked_tensor_values(n_points)), n_points_{n_points} {}

DeviceTensorField::DeviceTensorField(std::size_t n_points,
                                     backend::uninitialized_t)
    : g_(checked_tensor_values(n_points), backend::uninitialized),
      n_points_{n_points} {}

Field<Mat3x3> DeviceTensorField::to_host(backend::stream_t stream) const {
  Field<Mat3x3> out(n_points_);
  if (n_points_ == 0) return out;

  const std::size_t n = n_points_;
  std::vector<Real> staged(checked_tensor_values(n));
  g_.copy_to_host_async(staged.data(), staged.size(), stream);
  backend::device_synchronize(stream);

  for (std::size_t p = 0; p < n; ++p) {
    Mat3x3 m;
    m.r0 = Vec3{staged[0 * n + p], staged[1 * n + p], staged[2 * n + p]};
    m.r1 = Vec3{staged[3 * n + p], staged[4 * n + p], staged[5 * n + p]};
    m.r2 = Vec3{staged[6 * n + p], staged[7 * n + p], staged[8 * n + p]};
    out[p] = m;
  }
  return out;
}

// -- Status decoding --------------------------------------------------------

void throw_on_evaluator_status(int status, const char* who, const char* what) {
  // Singularity is reported first. When a point is genuinely singular the
  // overflow bit is usually set too (the intermediate blew past the working
  // range on the way), and "singular" is the more specific diagnosis.
  if ((status & 1) != 0) {
    throw std::domain_error{std::string{who} + ": " + what
                            + " is singular at an observation point"};
  }
  if ((status & 2) != 0) {
    throw std::overflow_error{std::string{who} + ": " + what
                              + " is not representable in the working precision"};
  }
  if ((status & 4) != 0) {
    throw std::invalid_argument{
        std::string{who} + ": an observation coordinate is not finite"};
  }
  if ((status & 8) != 0) {
    throw std::out_of_range{std::string{who}
                            + ": an observation point lies outside the field grid"};
  }
}

}  // namespace quasar::core
