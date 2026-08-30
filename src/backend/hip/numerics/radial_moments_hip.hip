// Device assembly and batched solution of the cylindrical radial stencil rows.
//
// A radial table needs roughly twenty rows per radial index across the padded
// grid. Each row is an independent general system of width at most eight:
// either a moment matrix against monomial data (point-value target) or a
// Vandermonde matrix against moment data (cell-average target). The host used
// to walk them one at a time in `long double`, with its own partial-pivoted
// Gaussian elimination and one iterative-refinement step.
//
// Here the assembly, the refinement, the normalization and the residual are
// kernels, and the two eliminations are one batched rocSOLVER LU each. The
// refinement is a second, independent factorization rather than a reuse of the
// first one's factors, which is what the host did -- it is the cheaper of the
// two and reusing the factors would change what the step means.
//
// Precision: binary64 rather than the host's `long double`, which is the whole
// reason the residual gate downstream matters. Measured over the padded grid
// for every width and measure the tables use, the worst residual is 3.4e-12
// against a 2.0e-12 `long double` reference and a 1e-11 threshold. The
// refinement step is what buys that: it makes the result backward stable, so
// the residual tracks working precision times the matrix norm rather than the
// condition number of a Vandermonde system.

#include "quasar/numerics/radial_moments.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/batched_lu.hpp"
#include "quasar/numerics/radial_cell_moments.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace quasar::numerics {

namespace {

constexpr unsigned kBlock = 128;
constexpr int kOrder = kMaxRadialStencilWidth;

// Status bits, OR-reduced across the batch with an integer atomic, plus a
// per-row status word so a failure can name its row.
constexpr int kInvalidVolume = 1;
constexpr int kInvalidConstantMode = 2;
constexpr int kNotRepresentable = 4;
constexpr int kNoExactPartition = 8;

// Device-side mirror of RadialRowSpec. The enums are lowered to ints so the POD
// crosses the launch boundary without depending on their underlying type.
struct DeviceRowSpec {
  Real rho_anchor;
  Real node_xi;
  int width;
  int offset;
  int target;   // 0 = point_value, 1 = cell_average
  int measure;  // matches RadialCellMeasure
};

struct DeviceGaussSpec {
  Real rho_anchor;
  int count;
  int measure;
  Real node_xi[kOrder];
  Real cartesian_weights[kOrder];
};

__device__ inline RadialCellMeasure as_measure(int measure) {
  return static_cast<RadialCellMeasure>(measure);
}

dim3 grid_for(int n) {
  return dim3(static_cast<unsigned>(
      (n + static_cast<int>(kBlock) - 1) / static_cast<int>(kBlock)));
}

// Column-major offset of entry (row m, column k) within system `s`.
__device__ inline long long entry(long long s, int m, int k) {
  return s * (kOrder * kOrder) + m + static_cast<long long>(k) * kOrder;
}

// Builds the moment system for every row, padded to kOrder with an identity
// block so a single strided-batched factorization covers mixed widths.
//
// The padding cannot interfere with the real system: for a column below the
// row's width every padding row holds exactly zero, so partial pivoting never
// selects one unless the whole column is zero -- which is a singular system
// either way.
__global__ void assemble_kernel(const DeviceRowSpec* __restrict__ specs,
                                int count,
                                Real* __restrict__ matrices,
                                Real* __restrict__ rhs,
                                int* __restrict__ row_status) {
  const int s = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (s >= count) return;

  const DeviceRowSpec spec = specs[s];
  const RadialCellMeasure measure = as_measure(spec.measure);
  int local = 0;

  for (int m = 0; m < kOrder; ++m) {
    if (m >= spec.width) {
      // Identity padding: x[m] = 0 and no coupling to the real block.
      for (int k = 0; k < kOrder; ++k) {
        matrices[entry(s, m, k)] = (k == m) ? Real{1} : Real{0};
      }
      rhs[static_cast<long long>(s) * kOrder + m] = Real{0};
      continue;
    }

    if (spec.target == 0) {
      // point_value: monomial target, moment data.
      rhs[static_cast<long long>(s) * kOrder + m] =
          radial_integer_power(spec.node_xi, m);
      for (int k = 0; k < kOrder; ++k) {
        if (k >= spec.width) {
          matrices[entry(s, m, k)] = Real{0};
          continue;
        }
        const Real cell_rho =
            spec.rho_anchor + static_cast<Real>(spec.offset + k);
        const Real moment = shifted_normalized_cell_moment(
            cell_rho, spec.rho_anchor, m, measure);
        if (!isfinite(moment)) local |= kInvalidVolume;
        matrices[entry(s, m, k)] = moment;
      }
    } else {
      // cell_average: moment target, Vandermonde data.
      const Real moment = shifted_normalized_cell_moment(
          spec.rho_anchor, spec.rho_anchor, m, measure);
      if (!isfinite(moment)) local |= kInvalidVolume;
      rhs[static_cast<long long>(s) * kOrder + m] = moment;
      for (int k = 0; k < kOrder; ++k) {
        if (k >= spec.width) {
          matrices[entry(s, m, k)] = Real{0};
          continue;
        }
        const Real point =
            static_cast<Real>(spec.offset + k) + spec.node_xi;
        matrices[entry(s, m, k)] = radial_integer_power(point, m);
      }
    }
  }
  row_status[s] = local;
}

// defect = rhs - A * x, formed against the unfactored matrix.
__global__ void defect_kernel(const DeviceRowSpec* __restrict__ specs,
                              int count,
                              const Real* __restrict__ matrices,
                              const Real* __restrict__ rhs,
                              const Real* __restrict__ solution,
                              Real* __restrict__ defect) {
  const int s = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (s >= count) return;
  const int width = specs[s].width;

  for (int m = 0; m < kOrder; ++m) {
    const long long slot = static_cast<long long>(s) * kOrder + m;
    if (m >= width) {
      defect[slot] = Real{0};
      continue;
    }
    Real value = rhs[slot];
    for (int k = 0; k < width; ++k) {
      value = fma(-matrices[entry(s, m, k)],
                  solution[static_cast<long long>(s) * kOrder + k], value);
    }
    defect[slot] = value;
  }
}

// Applies the correction, normalizes the row to an exact binary64 partition of
// unity, and measures the residual against the unfactored system.
__global__ void finish_kernel(const DeviceRowSpec* __restrict__ specs,
                              int count,
                              const Real* __restrict__ matrices,
                              const Real* __restrict__ rhs,
                              const Real* __restrict__ solution,
                              const Real* __restrict__ correction,
                              Real* __restrict__ coefficients,
                              Real* __restrict__ residuals,
                              int* __restrict__ row_status) {
  const int s = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (s >= count) return;
  const int width = specs[s].width;
  const long long base = static_cast<long long>(s) * kOrder;

  Real x[kOrder];
  Real sum = Real{0};
  for (int k = 0; k < width; ++k) {
    x[k] = solution[base + k] + correction[base + k];
    sum += x[k];
  }

  int local = row_status[s];
  if (sum == Real{0} || !isfinite(sum)) {
    row_status[s] = local | kInvalidConstantMode;
    return;
  }

  Real c[kOrder];
  for (int k = 0; k < width; ++k) {
    c[k] = x[k] / sum;
    if (!isfinite(c[k])) local |= kNotRepresentable;
  }

  // Make the binary64 partition of unity an invariant, rather than merely a
  // tolerance. With the stencils supported here the prefix is close to one, so
  // Sterbenz-exact subtraction makes the final in-order addition exactly one.
  // The small correction is no larger than ordinary conversion error.
  Real prefix = Real{0};
  for (int k = 0; k + 1 < width; ++k) prefix += c[k];
  c[width - 1] = Real{1} - prefix;
  Real in_order = Real{0};
  for (int k = 0; k < width; ++k) in_order += c[k];
  if (in_order != Real{1}) local |= kNoExactPartition;

  Real residual = Real{0};
  for (int m = 0; m < width; ++m) {
    Real reconstructed = Real{0};
    for (int k = 0; k < width; ++k) {
      reconstructed = fma(c[k], matrices[entry(s, m, k)], reconstructed);
    }
    const Real defect = fabs(reconstructed - rhs[base + m]);
    residual = defect > residual ? defect : residual;
  }

  for (int k = 0; k < width; ++k) coefficients[base + k] = c[k];
  for (int k = width; k < kOrder; ++k) coefficients[base + k] = Real{0};
  residuals[s] = residual;
  row_status[s] = local;
}

// Gauss rules need no linear solve: fold the radial measure into the supplied
// Cartesian weights, renormalize, and measure monomial exactness.
__global__ void gauss_kernel(const DeviceGaussSpec* __restrict__ specs,
                             int count,
                             Real* __restrict__ coefficients,
                             Real* __restrict__ residuals,
                             int* __restrict__ row_status) {
  const int s = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (s >= count) return;

  const DeviceGaussSpec spec = specs[s];
  const RadialCellMeasure measure = as_measure(spec.measure);
  const long long base = static_cast<long long>(s) * kOrder;

  Real weighted[kOrder];
  Real normalization = Real{0};
  int local = 0;
  for (int q = 0; q < spec.count; ++q) {
    const Real node = spec.node_xi[q];
    const Real weight = spec.cartesian_weights[q];
    const Real radius = spec.rho_anchor + node;
    switch (measure) {
      case RadialCellMeasure::uniform:
        weighted[q] = weight;
        break;
      case RadialCellMeasure::annular:
        weighted[q] = weight * fabs(radius);
        break;
      case RadialCellMeasure::angular_momentum:
        weighted[q] = weight * radius * radius;
        break;
    }
    normalization += weighted[q];
  }
  if (!(normalization > Real{0}) || !isfinite(normalization)) {
    row_status[s] = kInvalidConstantMode;
    return;
  }

  Real c[kOrder];
  for (int q = 0; q < spec.count; ++q) {
    c[q] = weighted[q] / normalization;
    if (!isfinite(c[q])) local |= kNotRepresentable;
  }
  Real prefix = Real{0};
  for (int q = 0; q + 1 < spec.count; ++q) prefix += c[q];
  c[spec.count - 1] = Real{1} - prefix;
  Real in_order = Real{0};
  for (int q = 0; q < spec.count; ++q) in_order += c[q];
  if (in_order != Real{1}) local |= kNoExactPartition;

  // Exactness through the maximum degree retained after multiplying the
  // integrand by the radial measure.
  const int exact_degree =
      2 * spec.count - 1 - radial_measure_power(measure);
  Real residual = Real{0};
  for (int m = 0; m <= exact_degree; ++m) {
    Real quadrature = Real{0};
    for (int q = 0; q < spec.count; ++q) {
      quadrature = fma(c[q], radial_integer_power(spec.node_xi[q], m),
                       quadrature);
    }
    const Real exact = shifted_normalized_cell_moment(
        spec.rho_anchor, spec.rho_anchor, m, measure);
    if (!isfinite(exact)) local |= kInvalidVolume;
    const Real defect = fabs(quadrature - exact);
    residual = defect > residual ? defect : residual;
  }

  for (int q = 0; q < spec.count; ++q) coefficients[base + q] = c[q];
  for (int q = spec.count; q < kOrder; ++q) coefficients[base + q] = Real{0};
  residuals[s] = residual;
  row_status[s] = local;
}

struct DeviceExtrapolationSpec {
  Real rho_center;
  Real face_offset;
  int neighbor_offset;
  int measure;
};

__global__ void extrapolation_kernel(
    const DeviceExtrapolationSpec* __restrict__ specs, int count,
    Real* __restrict__ factors, int* __restrict__ row_status) {
  const int s = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (s >= count) return;

  const DeviceExtrapolationSpec spec = specs[s];
  const RadialCellMeasure measure = as_measure(spec.measure);
  const Real center =
      shifted_normalized_cell_moment(spec.rho_center, Real{0}, 1, measure);
  const Real neighbor = shifted_normalized_cell_moment(
      spec.rho_center + static_cast<Real>(spec.neighbor_offset), Real{0}, 1,
      measure);
  const Real denominator = center - neighbor;
  const Real beta =
      (spec.rho_center + spec.face_offset - center) / denominator;

  int local = 0;
  if (!isfinite(center) || !isfinite(neighbor)) local |= kInvalidVolume;
  if (denominator == Real{0} || !isfinite(beta)) local |= kNotRepresentable;
  factors[s] = beta;
  row_status[s] = local;
}

// -- Host helpers -----------------------------------------------------------

void validate_width(int width, const char* what) {
  if (width < 1 || width > kMaxRadialStencilWidth) {
    throw std::invalid_argument{std::string{what} + " must be in [1, 8]"};
  }
}

[[noreturn]] void throw_row_failure(int status, std::size_t index) {
  const std::string where = " (row " + std::to_string(index) + ")";
  if ((status & kInvalidVolume) != 0) {
    throw std::runtime_error{"radial cell has invalid weighted volume" + where};
  }
  if ((status & kInvalidConstantMode) != 0) {
    throw std::runtime_error{"radial moment row has invalid constant mode"
                             + where};
  }
  if ((status & kNotRepresentable) != 0) {
    throw std::runtime_error{"radial moment row is not representable" + where};
  }
  throw std::runtime_error{
      "radial moment row cannot represent an exact partition of unity" + where};
}

void check_row_status(const std::vector<int>& status) {
  for (std::size_t i = 0; i < status.size(); ++i) {
    if (status[i] != 0) throw_row_failure(status[i], i);
  }
}

// Packs the device results into the returned rows.
std::vector<RadialStencilRow> gather_rows(
    const std::vector<Real>& coefficients, const std::vector<Real>& residuals,
    const std::vector<int>& widths) {
  std::vector<RadialStencilRow> rows(widths.size());
  for (std::size_t i = 0; i < widths.size(); ++i) {
    rows[i].width = widths[i];
    for (int k = 0; k < widths[i]; ++k) {
      rows[i].c[k] = coefficients[i * kOrder + static_cast<std::size_t>(k)];
    }
    rows[i].residual = residuals[i];
  }
  return rows;
}

}  // namespace

Real normalized_cell_moment(Real rho, int m) {
  return normalized_cell_moment(rho, m, RadialCellMeasure::annular);
}

Real normalized_cell_moment(Real rho, int m, RadialCellMeasure measure) {
  if (!std::isfinite(rho)) {
    throw std::invalid_argument{"radial cell coordinate must be finite"};
  }
  if (m < 0) {
    throw std::invalid_argument{"radial moment degree must be non-negative"};
  }
  if (radial_measure_power(measure) < 0) {
    throw std::invalid_argument{"unknown radial cell measure"};
  }
  const Real moment =
      shifted_normalized_cell_moment(rho, Real{0}, m, measure);
  if (!std::isfinite(moment)) {
    throw std::runtime_error{"radial cell has invalid weighted volume"};
  }
  return moment;
}

std::vector<RadialStencilRow> solve_radial_rows(
    const std::vector<RadialRowSpec>& specs, backend::stream_t stream) {
  std::vector<DeviceRowSpec> host_specs;
  std::vector<int> widths;
  host_specs.reserve(specs.size());
  widths.reserve(specs.size());
  for (const RadialRowSpec& spec : specs) {
    if (!std::isfinite(spec.rho_anchor) || !std::isfinite(spec.node_xi)) {
      throw std::invalid_argument{"radial stencil coordinates must be finite"};
    }
    validate_width(spec.width, "radial stencil width");
    if (spec.target != RadialMomentTarget::point_value
        && spec.target != RadialMomentTarget::cell_average) {
      throw std::invalid_argument{"unknown radial moment target"};
    }
    if (radial_measure_power(spec.measure) < 0) {
      throw std::invalid_argument{"unknown radial cell measure"};
    }
    host_specs.push_back(DeviceRowSpec{
        spec.rho_anchor, spec.node_xi, spec.width, spec.offset,
        spec.target == RadialMomentTarget::point_value ? 0 : 1,
        static_cast<int>(spec.measure)});
    widths.push_back(spec.width);
  }

  const int count = static_cast<int>(specs.size());
  if (count == 0) return {};
  if (specs.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error{
        "solve_radial_rows: batch exceeds the signed kernel-index limit"};
  }

  const std::size_t n = static_cast<std::size_t>(count);
  const std::size_t matrix_values = n * kOrder * kOrder;
  const std::size_t vector_values = n * kOrder;

  backend::DeviceBuffer<DeviceRowSpec> d_specs(n, backend::uninitialized);
  d_specs.copy_from_host(host_specs.data(), n);

  backend::DeviceBuffer<Real> original(matrix_values, backend::uninitialized);
  backend::DeviceBuffer<Real> rhs(vector_values, backend::uninitialized);
  backend::DeviceBuffer<int> row_status(n);

  assemble_kernel<<<grid_for(count), dim3(kBlock), 0,
                    static_cast<hipStream_t>(stream)>>>(
      d_specs.device_ptr(), count, original.device_ptr(), rhs.device_ptr(),
      row_status.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());

  {
    std::vector<int> status(n, 0);
    row_status.copy_to_host(status.data(), n);
    check_row_status(status);
  }

  // getrf overwrites its matrix, and both the refinement and the residual need
  // the unfactored system, so each solve works on its own copy.
  const auto solve_against_copy = [&](backend::DeviceBuffer<Real>& right_side,
                                      const char* what) {
    backend::DeviceBuffer<Real> factors(matrix_values, backend::uninitialized);
    backend::device_memcpy_d2d_async(factors.device_ptr(),
                                     original.device_ptr(),
                                     matrix_values * sizeof(Real), stream);
    const BatchedLuResult result =
        solve_batched_dense_lu(factors, right_side, kOrder, count, stream);
    if (!result.ok()) {
      throw std::runtime_error{
          std::string{"singular radial moment system in the "} + what
          + " (row " + std::to_string(result.first_singular_system) + ")"};
    }
  };

  backend::DeviceBuffer<Real> solution(vector_values, backend::uninitialized);
  backend::device_memcpy_d2d_async(solution.device_ptr(), rhs.device_ptr(),
                                   vector_values * sizeof(Real), stream);
  solve_against_copy(solution, "initial solve");

  // One iterative-refinement step, using the unmodified moment system to
  // recover the defect left by elimination. This is what makes the result
  // backward stable in binary64, and hence what the residual gate relies on.
  backend::DeviceBuffer<Real> correction(vector_values,
                                         backend::uninitialized);
  defect_kernel<<<grid_for(count), dim3(kBlock), 0,
                  static_cast<hipStream_t>(stream)>>>(
      d_specs.device_ptr(), count, original.device_ptr(), rhs.device_ptr(),
      solution.device_ptr(), correction.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  solve_against_copy(correction, "refinement solve");

  backend::DeviceBuffer<Real> coefficients(vector_values,
                                           backend::uninitialized);
  backend::DeviceBuffer<Real> residuals(n, backend::uninitialized);
  finish_kernel<<<grid_for(count), dim3(kBlock), 0,
                  static_cast<hipStream_t>(stream)>>>(
      d_specs.device_ptr(), count, original.device_ptr(), rhs.device_ptr(),
      solution.device_ptr(), correction.device_ptr(),
      coefficients.device_ptr(), residuals.device_ptr(),
      row_status.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());

  std::vector<Real> host_coefficients(vector_values);
  std::vector<Real> host_residuals(n);
  std::vector<int> status(n, 0);
  coefficients.copy_to_host_async(host_coefficients.data(), vector_values,
                                  stream);
  residuals.copy_to_host_async(host_residuals.data(), n, stream);
  row_status.copy_to_host_async(status.data(), n, stream);
  backend::device_synchronize(stream);
  check_row_status(status);

  return gather_rows(host_coefficients, host_residuals, widths);
}

RadialStencilRow solve_radial_row(Real rho_anchor, int width, int offset,
                                  RadialMomentTarget target, Real node_xi) {
  return solve_radial_row(rho_anchor, width, offset, target,
                          RadialCellMeasure::annular, node_xi);
}

RadialStencilRow solve_radial_row(Real rho_anchor, int width, int offset,
                                  RadialMomentTarget target,
                                  RadialCellMeasure measure, Real node_xi) {
  const std::vector<RadialRowSpec> specs{
      RadialRowSpec{rho_anchor, node_xi, width, offset, target, measure}};
  return solve_radial_rows(specs).front();
}

std::vector<RadialStencilRow> radial_gauss_weight_rows(
    const std::vector<RadialGaussSpec>& specs, backend::stream_t stream) {
  std::vector<DeviceGaussSpec> host_specs;
  std::vector<int> widths;
  host_specs.reserve(specs.size());
  widths.reserve(specs.size());
  for (const RadialGaussSpec& spec : specs) {
    if (!std::isfinite(spec.rho_anchor)) {
      throw std::invalid_argument{"radial Gauss coordinate must be finite"};
    }
    validate_width(spec.count, "radial Gauss count");
    if (radial_measure_power(spec.measure) < 0) {
      throw std::invalid_argument{"unknown radial cell measure"};
    }
    DeviceGaussSpec device{};
    device.rho_anchor = spec.rho_anchor;
    device.count = spec.count;
    device.measure = static_cast<int>(spec.measure);
    for (int q = 0; q < spec.count; ++q) {
      if (!std::isfinite(spec.node_xi[q])
          || !std::isfinite(spec.cartesian_weights[q])) {
        throw std::invalid_argument{"radial Gauss rule must be finite"};
      }
      device.node_xi[q] = spec.node_xi[q];
      device.cartesian_weights[q] = spec.cartesian_weights[q];
    }
    host_specs.push_back(device);
    widths.push_back(spec.count);
  }

  const int count = static_cast<int>(specs.size());
  if (count == 0) return {};
  const std::size_t n = static_cast<std::size_t>(count);
  const std::size_t vector_values = n * kOrder;

  backend::DeviceBuffer<DeviceGaussSpec> d_specs(n, backend::uninitialized);
  d_specs.copy_from_host(host_specs.data(), n);
  backend::DeviceBuffer<Real> coefficients(vector_values,
                                           backend::uninitialized);
  backend::DeviceBuffer<Real> residuals(n, backend::uninitialized);
  backend::DeviceBuffer<int> row_status(n);

  gauss_kernel<<<grid_for(count), dim3(kBlock), 0,
                 static_cast<hipStream_t>(stream)>>>(
      d_specs.device_ptr(), count, coefficients.device_ptr(),
      residuals.device_ptr(), row_status.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());

  std::vector<Real> host_coefficients(vector_values);
  std::vector<Real> host_residuals(n);
  std::vector<int> status(n, 0);
  coefficients.copy_to_host_async(host_coefficients.data(), vector_values,
                                  stream);
  residuals.copy_to_host_async(host_residuals.data(), n, stream);
  row_status.copy_to_host_async(status.data(), n, stream);
  backend::device_synchronize(stream);
  for (std::size_t i = 0; i < status.size(); ++i) {
    if (status[i] == 0) continue;
    if ((status[i] & kInvalidConstantMode) != 0) {
      throw std::runtime_error{"radial Gauss rule has invalid normalization "
                               "(row " + std::to_string(i) + ")"};
    }
    throw_row_failure(status[i], i);
  }

  return gather_rows(host_coefficients, host_residuals, widths);
}

std::vector<Real> radial_face_extrapolation_factors(
    const std::vector<RadialExtrapolationSpec>& specs,
    backend::stream_t stream) {
  std::vector<DeviceExtrapolationSpec> host_specs;
  host_specs.reserve(specs.size());
  for (const RadialExtrapolationSpec& spec : specs) {
    if (!std::isfinite(spec.rho_center) || !std::isfinite(spec.face_offset)) {
      throw std::invalid_argument{
          "radial extrapolation coordinates must be finite"};
    }
    if (radial_measure_power(spec.measure) < 0) {
      throw std::invalid_argument{"unknown radial cell measure"};
    }
    host_specs.push_back(DeviceExtrapolationSpec{
        spec.rho_center, spec.face_offset, spec.neighbor_offset,
        static_cast<int>(spec.measure)});
  }

  const int count = static_cast<int>(specs.size());
  if (count == 0) return {};
  const std::size_t n = static_cast<std::size_t>(count);

  backend::DeviceBuffer<DeviceExtrapolationSpec> d_specs(
      n, backend::uninitialized);
  d_specs.copy_from_host(host_specs.data(), n);
  backend::DeviceBuffer<Real> factors(n, backend::uninitialized);
  backend::DeviceBuffer<int> row_status(n);

  extrapolation_kernel<<<grid_for(count), dim3(kBlock), 0,
                         static_cast<hipStream_t>(stream)>>>(
      d_specs.device_ptr(), count, factors.device_ptr(),
      row_status.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());

  std::vector<Real> host_factors(n);
  std::vector<int> status(n, 0);
  factors.copy_to_host_async(host_factors.data(), n, stream);
  row_status.copy_to_host_async(status.data(), n, stream);
  backend::device_synchronize(stream);
  for (std::size_t i = 0; i < status.size(); ++i) {
    if (status[i] == 0) continue;
    throw std::runtime_error{
        "ill-conditioned radial face extrapolation (row "
        + std::to_string(i) + ")"};
  }
  return host_factors;
}

RadialStencilRow radial_gauss_weights(Real rho_anchor, int count,
                                      const Real* node_xi,
                                      const Real* cartesian_weights) {
  return radial_gauss_weights(rho_anchor, count, node_xi, cartesian_weights,
                              RadialCellMeasure::annular);
}

RadialStencilRow radial_gauss_weights(Real rho_anchor, int count,
                                      const Real* node_xi,
                                      const Real* cartesian_weights,
                                      RadialCellMeasure measure) {
  if (node_xi == nullptr || cartesian_weights == nullptr) {
    throw std::invalid_argument{"radial Gauss arrays must not be null"};
  }
  validate_width(count, "radial Gauss count");
  RadialGaussSpec spec{};
  spec.rho_anchor = rho_anchor;
  spec.count = count;
  spec.measure = measure;
  for (int q = 0; q < count; ++q) {
    spec.node_xi[q] = node_xi[q];
    spec.cartesian_weights[q] = cartesian_weights[q];
  }
  return radial_gauss_weight_rows(std::vector<RadialGaussSpec>{spec}).front();
}

}  // namespace quasar::numerics
