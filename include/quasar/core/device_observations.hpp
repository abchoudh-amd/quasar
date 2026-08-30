#pragma once

// Device-resident structure-of-arrays containers for an observation-point set
// and the fields sampled on it.
//
// These are the currency of the numerics field-evaluator interface. An
// evaluator takes a DevicePointCloud and returns DeviceVectorField /
// DeviceTensorField: every plane stays on the device, so a chain of evaluations
// (Biot-Savart into a PIC external-field sampler, say) never round-trips
// through host memory. The host staging types core::PointSoA and core::Field
// remain, but only at an *output* boundary -- a CLI writing an .npz, a Python
// binding building a NumPy array, or a test asserting a value.
//
// SoA rather than AoS is not a preference. The kernels behind every evaluator
// write one component plane per output pointer because that is what coalesces;
// the old host interface forced a transpose into Field<Vec3> immediately after
// each kernel purely to satisfy its own signature. Keeping the planes is what
// deletes that transpose.
//
// These live in core, next to the host observation types they mirror, so the
// numerics axis can name them without depending on any physics module.

#include "quasar/backend/memory.hpp"
#include "quasar/core/field.hpp"
#include "quasar/core/observations.hpp"
#include "quasar/core/types.hpp"

#include <cstddef>

namespace quasar::core {

// Device counterpart of PointSoA: three coordinate planes of equal length.
class DevicePointCloud {
 public:
  DevicePointCloud() = default;

  // Allocates zeroed planes for `n_points` observations. A kernel that fills
  // every entry (the WP4 point generators) writes straight into these.
  explicit DevicePointCloud(std::size_t n_points);

  // Validates the host planes (mismatched lengths and non-finite coordinates
  // are rejected before anything is sized from px alone) and uploads them.
  // Synchronous on return, so the caller's host staging may be released.
  static DevicePointCloud upload(const PointSoA& points,
                                 backend::stream_t stream = nullptr);
  static DevicePointCloud upload(const PointCloud& points,
                                 backend::stream_t stream = nullptr);

  std::size_t size()  const noexcept { return n_points_; }
  bool        empty() const noexcept { return n_points_ == 0; }

  Real* x() noexcept { return px_.device_ptr(); }
  Real* y() noexcept { return py_.device_ptr(); }
  Real* z() noexcept { return pz_.device_ptr(); }
  const Real* x() const noexcept { return px_.device_ptr(); }
  const Real* y() const noexcept { return py_.device_ptr(); }
  const Real* z() const noexcept { return pz_.device_ptr(); }

  // Output boundary only: downloads the three planes into host SoA.
  PointSoA to_host(backend::stream_t stream = nullptr) const;

 private:
  backend::DeviceBuffer<Real> px_{}, py_{}, pz_{};
  std::size_t n_points_{0};
};

// Three component planes of a vector field sampled at `size()` points.
class DeviceVectorField {
 public:
  DeviceVectorField() = default;

  // Zeroed planes. The zero-field defaults on IFieldEvaluator return this.
  explicit DeviceVectorField(std::size_t n_points);
  // Planes left uninitialized, for a kernel that writes every entry.
  DeviceVectorField(std::size_t n_points, backend::uninitialized_t);

  std::size_t size()  const noexcept { return n_points_; }
  bool        empty() const noexcept { return n_points_ == 0; }

  Real* x() noexcept { return x_.device_ptr(); }
  Real* y() noexcept { return y_.device_ptr(); }
  Real* z() noexcept { return z_.device_ptr(); }
  const Real* x() const noexcept { return x_.device_ptr(); }
  const Real* y() const noexcept { return y_.device_ptr(); }
  const Real* z() const noexcept { return z_.device_ptr(); }

  // Output boundary only. Downloads the three planes and transposes them into
  // the host AoS Field the CLIs, bindings and tests still speak.
  Field<Vec3> to_host(backend::stream_t stream = nullptr) const;

 private:
  backend::DeviceBuffer<Real> x_{}, y_{}, z_{};
  std::size_t n_points_{0};
};

// Nine component planes of a rank-2 tensor field (the field Jacobian
// (grad B)_{ij} = dB_i/dp_j), held in one contiguous buffer.
//
// Layout is component-major: entry (i, j) of point `p` lives at
// `data()[(3 * i + j) * size() + p]`. That is the layout the Biot-Savart
// gradient kernel already writes, and it keeps each of the nine planes
// individually coalesced.
class DeviceTensorField {
 public:
  DeviceTensorField() = default;

  explicit DeviceTensorField(std::size_t n_points);
  DeviceTensorField(std::size_t n_points, backend::uninitialized_t);

  std::size_t size()  const noexcept { return n_points_; }
  bool        empty() const noexcept { return n_points_ == 0; }

  Real*       data()       noexcept { return g_.device_ptr(); }
  const Real* data() const noexcept { return g_.device_ptr(); }

  // Output boundary only.
  Field<Mat3x3> to_host(backend::stream_t stream = nullptr) const;

 private:
  backend::DeviceBuffer<Real> g_{};
  std::size_t n_points_{0};
};

// Shared by every evaluator whose kernel reports failure through a status word
// instead of throwing on the device. Bit 0 means the requested quantity is
// singular at some observation point; bit 1 means it is finite mathematically
// but not representable in the working precision. `what` names the quantity in
// the message ("magnetic field", "magnetic-field gradient", ...) and `who`
// names the evaluator.
//
// Kernels set bits with an integer atomic, which is exact and
// order-independent, so the reported status does not depend on the launch
// geometry -- but it also does not identify *which* point failed. That matches
// the host behaviour it replaces: the host loops threw on the first offending
// point without naming its index either.
void throw_on_evaluator_status(int status, const char* who, const char* what);

// Number of Real values a DeviceTensorField of `n_points` holds, with the
// 9 * n_points product checked. Callers sizing host staging use this.
std::size_t checked_tensor_values(std::size_t n_points);

}  // namespace quasar::core
