#include "quasar/physics/analytic_fields/file_grid.hpp"

#include "quasar/core/device_observations.hpp"
#include "quasar/physics/analytic_fields/kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quasar::analytic_fields {

namespace {

bool finite(Vec3 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y)
      && std::isfinite(value.z);
}

std::size_t checked_product(std::array<std::size_t, 3> dims) {
  std::size_t product = 1;
  for (const std::size_t n : dims) {
    if (n == 0) {
      throw std::invalid_argument{"FileGridEvaluator: dimensions must be positive"};
    }
    if (product > std::numeric_limits<std::size_t>::max() / n) {
      throw std::overflow_error{"FileGridEvaluator: grid size is not representable"};
    }
    product *= n;
  }
  return product;
}

Real grid_coordinate(Real origin, Real spacing, std::size_t index) {
  const Real coordinate = std::fma(static_cast<Real>(index), spacing, origin);
  if (!std::isfinite(coordinate)) {
    throw std::overflow_error{
        "FileGridEvaluator: upper grid coordinate is not representable"};
  }
  return coordinate;
}

bool dimension_fits_size_t(Real value) {
  constexpr int size_bits = std::numeric_limits<std::size_t>::digits;
  constexpr int real_max_exponent = std::numeric_limits<Real>::max_exponent;
  if constexpr (size_bits < real_max_exponent) {
    // size_t is unsigned, hence its first unrepresentable positive integer is
    // exactly 2^digits.  Comparing against that exclusive bound avoids casting
    // SIZE_MAX to a floating type that may round it up to 2^digits.
    return value < std::ldexp(Real{1}, size_bits);
  }
  // Every finite non-negative Real then lies below size_t's upper bound.
  return true;
}

}  // namespace

FileGridEvaluator::FileGridEvaluator(std::string path) : path_{std::move(path)} {
  set_grid(load_text_grid(path_));
}

bool FileGridEvaluator::provides_grad_B() const noexcept {
  return configured_ && grid_.dims[0] > 1 && grid_.dims[1] > 1
      && grid_.dims[2] > 1;
}

namespace {

// Host-side, scalar-cost checks on the grid descriptor. These do not scale with
// the node count -- they are configuration validation, not field arithmetic --
// so they stay on the host. The two O(nodes) sweeps (value finiteness and
// solenoidality) run on the device in set_grid() once the map is uploaded.
void validate_grid_descriptor(const Vec3& origin, const Vec3& spacing,
                              const std::array<std::size_t, 3>& dims,
                              Real divergence_tolerance,
                              std::size_t value_count) {
  if (!finite(origin) || !finite(spacing)
      || !std::isfinite(divergence_tolerance)
      || divergence_tolerance < Real{0}) {
    throw std::invalid_argument{
        "FileGridEvaluator: origin, spacing, and divergence tolerance must be finite"};
  }
  const std::size_t expected = checked_product(dims);
  if (value_count != expected) {
    throw std::invalid_argument{
        "FileGridEvaluator: values length does not match nx*ny*nz"};
  }
  const Real origins[] = {origin.x, origin.y, origin.z};
  const Real spacings[] = {spacing.x, spacing.y, spacing.z};
  for (int axis = 0; axis < 3; ++axis) {
    if (spacings[axis] < Real{0}
        || (dims[axis] > 1 && spacings[axis] <= Real{0})) {
      throw std::invalid_argument{
          "FileGridEvaluator: spacing must be positive on non-singleton axes"};
    }
    if (dims[axis] > 1) {
      const Real first_next =
          grid_coordinate(origins[axis], spacings[axis], std::size_t{1});
      const Real upper =
          grid_coordinate(origins[axis], spacings[axis], dims[axis] - 1);
      const Real upper_previous =
          grid_coordinate(origins[axis], spacings[axis], dims[axis] - 2);
      if (first_next == origins[axis] || upper == upper_previous) {
        throw std::overflow_error{
            "FileGridEvaluator: adjacent grid coordinates collapse in host precision"};
      }
    }
  }
}

// Pack the descriptor into the kernel POD. `derivative` selects which axis (if
// any) contributes its derivative weights instead of its interpolation weights;
// nullptr means none, which is the plain field sample.
QuasarAfFileGridParams grid_params(const Vec3& origin, const Vec3& spacing,
                                   const std::array<std::size_t, 3>& dims,
                                   const int* derivative) {
  QuasarAfFileGridParams params{};
  const Real origins[3] = {origin.x, origin.y, origin.z};
  const Real spacings[3] = {spacing.x, spacing.y, spacing.z};
  for (int axis = 0; axis < 3; ++axis) {
    params.origin[axis] = origins[axis];
    params.spacing[axis] = spacings[axis];
    if (dims[axis] > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::length_error{
          "FileGridEvaluator: dimension exceeds the signed kernel-index limit"};
    }
    params.dims[axis] = static_cast<int>(dims[axis]);
    params.derivative_axis[axis] = derivative == nullptr ? 0 : derivative[axis];
  }
  return params;
}

int checked_point_count(std::size_t n) {
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error{
        "FileGridEvaluator: observation count exceeds the signed kernel-index limit"};
  }
  return static_cast<int>(n);
}

}  // namespace

void FileGridEvaluator::set_grid(GridData grid) {
  validate_grid_descriptor(grid.origin, grid.spacing, grid.dims,
                           grid.divergence_tolerance, grid.values.size());

  // Transpose the parsed AoS nodes into the three SoA planes the sampling
  // kernel reads, then upload. This is the map's only host residency: the
  // staging vector is released below.
  const std::size_t count = grid.values.size();
  std::vector<Real> plane(count);
  backend::DeviceBuffer<Real> vx(count, backend::uninitialized);
  backend::DeviceBuffer<Real> vy(count, backend::uninitialized);
  backend::DeviceBuffer<Real> vz(count, backend::uninitialized);
  if (count != 0) {
    for (std::size_t i = 0; i < count; ++i) plane[i] = grid.values[i].x;
    vx.copy_from_host(plane.data(), count);
    for (std::size_t i = 0; i < count; ++i) plane[i] = grid.values[i].y;
    vy.copy_from_host(plane.data(), count);
    for (std::size_t i = 0; i < count; ++i) plane[i] = grid.values[i].z;
    vz.copy_from_host(plane.data(), count);
  }

  // Node finiteness and trilinear solenoidality: both O(nodes), both on device.
  // The round-off allowance on the tolerance is part of the contract, not
  // tuning -- it keeps a mildly discretized but genuinely solenoidal map usable
  // while still rejecting a grossly non-physical one.
  const Real relative_tolerance =
      grid.divergence_tolerance
      + Real{256} * std::numeric_limits<Real>::epsilon();
  backend::DeviceBuffer<int> status(1);
  ::launch_analytic_file_grid_validate(
      grid_params(grid.origin, grid.spacing, grid.dims, nullptr),
      vx.device_ptr(), vy.device_ptr(), vz.device_ptr(), relative_tolerance,
      status.device_ptr(), nullptr);
  int host_status = 0;
  status.copy_to_host(&host_status, 1);
  if ((host_status & 16) != 0) {
    throw std::invalid_argument{
        "FileGridEvaluator: field values must have finite components"};
  }
  if ((host_status & 32) != 0) {
    throw std::invalid_argument{
        "FileGridEvaluator: trilinear field map is not solenoidal"};
  }

  grid.values.clear();
  grid.values.shrink_to_fit();
  grid_ = std::move(grid);
  vx_ = std::move(vx);
  vy_ = std::move(vy);
  vz_ = std::move(vz);
  configured_ = true;
}

FileGridEvaluator::GridData FileGridEvaluator::load_text_grid(
    const std::string& path) {
  if (path.empty()) {
    throw std::invalid_argument{"FileGridEvaluator: path must not be empty"};
  }
  std::ifstream input{path};
  if (!input) {
    throw std::runtime_error{"FileGridEvaluator: cannot open field grid '" + path + "'"};
  }

  std::string magic;
  int version = 0;
  if (!(input >> magic >> version) || magic != "QUASAR_FILE_GRID" || version != 1) {
    throw std::invalid_argument{
        "FileGridEvaluator: expected 'QUASAR_FILE_GRID 1' header"};
  }

  GridData grid;
  std::string keyword;
  long long nx = 0, ny = 0, nz = 0;
  if (!(input >> keyword >> nx >> ny >> nz) || keyword != "dims") {
    throw std::invalid_argument{"FileGridEvaluator: expected 'dims nx ny nz'"};
  }
  if (nx <= 0 || ny <= 0 || nz <= 0) {
    throw std::invalid_argument{"FileGridEvaluator: dimensions must be positive"};
  }
  const auto fits_size_t = [](long long value) {
    if constexpr (sizeof(std::size_t) < sizeof(long long)) {
      return static_cast<unsigned long long>(value)
          <= static_cast<unsigned long long>(
              std::numeric_limits<std::size_t>::max());
    }
    return true;
  };
  if (!fits_size_t(nx) || !fits_size_t(ny) || !fits_size_t(nz)) {
    throw std::overflow_error{
        "FileGridEvaluator: dimension is not representable as size_t"};
  }
  grid.dims = {static_cast<std::size_t>(nx), static_cast<std::size_t>(ny),
               static_cast<std::size_t>(nz)};
  if (!(input >> keyword >> grid.origin.x >> grid.origin.y >> grid.origin.z)
      || keyword != "origin") {
    throw std::invalid_argument{"FileGridEvaluator: expected 'origin x y z'"};
  }
  if (!(input >> keyword >> grid.spacing.x >> grid.spacing.y >> grid.spacing.z)
      || keyword != "spacing") {
    throw std::invalid_argument{"FileGridEvaluator: expected 'spacing dx dy dz'"};
  }
  if (!(input >> keyword) || keyword != "data") {
    throw std::invalid_argument{"FileGridEvaluator: expected 'data' marker"};
  }

  const std::size_t count = checked_product(grid.dims);
  grid.values.resize(count);
  for (std::size_t index = 0; index < count; ++index) {
    Vec3& value = grid.values[index];
    if (!(input >> value.x >> value.y >> value.z)) {
      throw std::invalid_argument{
          "FileGridEvaluator: field grid ended before nx*ny*nz vectors"};
    }
  }
  std::string extra;
  if (input >> extra) {
    throw std::invalid_argument{
        "FileGridEvaluator: unexpected trailing data after field vectors"};
  }
  // Descriptor-only here: the O(nodes) sweeps run on the device in set_grid(),
  // and every caller of load_text_grid() feeds it straight into set_grid().
  validate_grid_descriptor(grid.origin, grid.spacing, grid.dims,
                           grid.divergence_tolerance, grid.values.size());
  return grid;
}

void FileGridEvaluator::configure(const numerics::EvaluatorParams& params) {
  numerics::reject_unknown_params(
      params, {"origin", "spacing", "dims", "values"},
      "FileGridEvaluator");
  const auto origin_it = params.find("origin");
  const auto spacing_it = params.find("spacing");
  const auto dims_it = params.find("dims");
  const auto values_it = params.find("values");
  if (origin_it == params.end() || spacing_it == params.end()
      || dims_it == params.end() || values_it == params.end()) {
    throw std::invalid_argument{
        "FileGridEvaluator: configure requires origin, spacing, dims, and values"};
  }
  GridData grid;
  grid.origin = numerics::param_vec3(params, "origin");
  grid.spacing = numerics::param_vec3(params, "spacing");
  if (dims_it->second.size() != 3) {
    throw std::invalid_argument{"FileGridEvaluator: dims must have 3 elements"};
  }
  for (int axis = 0; axis < 3; ++axis) {
    const Real value = dims_it->second[axis];
    if (!std::isfinite(value) || value < Real{1}
        || std::floor(value) != value || !dimension_fits_size_t(value)) {
      throw std::invalid_argument{
          "FileGridEvaluator: dims must be positive exact integers"};
    }
    grid.dims[axis] = static_cast<std::size_t>(value);
  }
  const std::size_t count = checked_product(grid.dims);
  if (count > std::numeric_limits<std::size_t>::max() / 3) {
    throw std::overflow_error{
        "FileGridEvaluator: flattened vector count is not representable"};
  }
  if (values_it->second.size() != 3 * count) {
    throw std::invalid_argument{
        "FileGridEvaluator: values must contain 3*nx*ny*nz elements"};
  }
  grid.values.resize(count);
  for (std::size_t index = 0; index < count; ++index) {
    grid.values[index] = Vec3{values_it->second[3 * index],
                              values_it->second[3 * index + 1],
                              values_it->second[3 * index + 2]};
  }
  set_grid(std::move(grid));
  path_.clear();
}

core::DeviceVectorField FileGridEvaluator::evaluate_B(
    const core::IFieldSource&, const core::DevicePointCloud& observations) const {
  if (!configured_) {
    throw std::invalid_argument{"FileGridEvaluator: no field grid is configured"};
  }
  const int M = checked_point_count(observations.size());
  core::DeviceVectorField out(observations.size(), backend::uninitialized);
  backend::DeviceBuffer<int> status(1);

  ::launch_analytic_file_grid_sample(
      grid_params(grid_.origin, grid_.spacing, grid_.dims, nullptr),
      vx_.device_ptr(), vy_.device_ptr(), vz_.device_ptr(),
      observations.x(), observations.y(), observations.z(), M,
      out.x(), out.y(), out.z(), status.device_ptr(), nullptr);

  int host_status = 0;
  status.copy_to_host(&host_status, 1);
  core::throw_on_evaluator_status(host_status, "FileGridEvaluator",
                                  "interpolated magnetic field");
  return out;
}

core::DeviceTensorField FileGridEvaluator::evaluate_grad_B(
    const core::IFieldSource&, const core::DevicePointCloud& observations) const {
  if (!configured_) {
    throw std::invalid_argument{"FileGridEvaluator: no field grid is configured"};
  }
  if (!provides_grad_B()) {
    throw std::runtime_error{
        "FileGridEvaluator: a full magnetic-field gradient requires at least "
        "two grid nodes on every axis; singleton axes are geometric planes"};
  }
  const int M = checked_point_count(observations.size());
  core::DeviceTensorField out(observations.size(), backend::uninitialized);
  if (M == 0) return out;
  backend::DeviceBuffer<int> status(1);

  // One launch per gradient column. Column `axis` is dB/dcoordinate_axis, whose
  // three components are tensor entries (0,axis), (1,axis) and (2,axis) --
  // planes 0*3+axis, 1*3+axis and 2*3+axis of the component-major buffer. The
  // sampling kernel writes three arbitrary output pointers, so it can fill
  // those strided planes directly and the host never transposes anything.
  const std::size_t n = observations.size();
  for (int axis = 0; axis < 3; ++axis) {
    int derivative[3] = {0, 0, 0};
    derivative[axis] = 1;
    Real* base = out.data();
    ::launch_analytic_file_grid_sample(
        grid_params(grid_.origin, grid_.spacing, grid_.dims, derivative),
        vx_.device_ptr(), vy_.device_ptr(), vz_.device_ptr(),
        observations.x(), observations.y(), observations.z(), M,
        base + static_cast<std::size_t>(0 * 3 + axis) * n,
        base + static_cast<std::size_t>(1 * 3 + axis) * n,
        base + static_cast<std::size_t>(2 * 3 + axis) * n,
        status.device_ptr(), nullptr);
  }

  int host_status = 0;
  status.copy_to_host(&host_status, 1);
  core::throw_on_evaluator_status(host_status, "FileGridEvaluator",
                                  "magnetic-field gradient");
  return out;
}

QUASAR_REGISTER_FIELD_EVALUATOR("file_grid", FileGridEvaluator)

}  // namespace quasar::analytic_fields
