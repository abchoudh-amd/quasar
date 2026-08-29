#include "quasar/numerics/condition_estimator.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/numerics/linear_algebra_detail.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace quasar::numerics {
namespace {

struct DeviceConditionSummary {
  Real smallest;
  Real largest;
  Real condition;
  Real digits;
  int singular;
};

__global__ void summarize_spectrum_kernel(const Real* eigenvalues, int order,
                                          Real relative_floor,
                                          DeviceConditionSummary* out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  Real smallest = std::numeric_limits<Real>::infinity();
  Real largest = Real{0};
  int non_finite = 0;
  for (int i = 0; i < order; ++i) {
    const Real magnitude = fabs(eigenvalues[i]);
    // C/C++ fmin and fmax intentionally return the non-NaN operand when only
    // one operand is NaN.  Track non-finite eigenvalues before the reduction
    // so they cannot be silently discarded and reported as a finite spectrum.
    if (!isfinite(magnitude)) {
      non_finite = 1;
      continue;
    }
    smallest = fmin(smallest, magnitude);
    largest = fmax(largest, magnitude);
  }

  DeviceConditionSummary summary{};
  summary.smallest = smallest;
  summary.largest = largest;
  const Real floor = relative_floor > Real{0}
                         ? relative_floor
                         : std::numeric_limits<Real>::epsilon()
                               * static_cast<Real>(order);
  summary.singular = non_finite != 0
                  || !(isfinite(smallest) && isfinite(largest))
                  || !(largest > Real{0})
                  || smallest <= floor * largest;
  if (summary.singular != 0) {
    summary.condition = std::numeric_limits<Real>::infinity();
    summary.digits = std::numeric_limits<Real>::infinity();
  } else {
    summary.condition = largest / smallest;
    summary.digits = log10(summary.condition);
  }
  *out = summary;
}

void validate_matrix(const backend::DeviceBuffer<Real>& matrix, int order,
                     Real relative_floor) {
  if (order <= 0) {
    throw std::invalid_argument{
        "estimate_symmetric_condition: order must be positive"};
  }
  const auto n = static_cast<std::size_t>(order);
  if (n > std::numeric_limits<std::size_t>::max() / n
      || matrix.size() != n * n) {
    throw std::invalid_argument{
        "estimate_symmetric_condition: matrix must contain exactly "
        "order*order column-major values"};
  }
  if (!std::isfinite(relative_floor) || relative_floor < Real{0}) {
    throw std::invalid_argument{
        "estimate_symmetric_condition: relative_floor must be finite and "
        "non-negative"};
  }
}

}  // namespace

SymmetricConditionEstimate estimate_symmetric_condition(
    const backend::DeviceBuffer<Real>& matrix, int order,
    MatrixTriangle triangle, Real relative_floor, backend::stream_t stream) {
  validate_matrix(matrix, order, relative_floor);

  const int device = matrix.owner_device();
  backend::DeviceGuard guard{device};
  const auto n = static_cast<std::size_t>(order);
  const auto matrix_size = n * n;

  backend::DeviceBuffer<Real> work_matrix{
      matrix_size, backend::uninitialized, backend::on_device(device)};
  backend::DeviceBuffer<Real> eigenvalues{
      n, backend::uninitialized, backend::on_device(device)};
  backend::DeviceBuffer<int> device_info{1, backend::on_device(device)};
  detail::copy_device_buffer(work_matrix, matrix, matrix_size, stream);

  detail::SolverHandle solver{stream};
  const hipblasFillMode_t fill =
      triangle == MatrixTriangle::lower ? HIPBLAS_FILL_MODE_LOWER
                                        : HIPBLAS_FILL_MODE_UPPER;
  int workspace_size = 0;
  detail::check_hipsolver(
      hipsolverDnDsyevd_bufferSize(
          solver.get(), HIPSOLVER_EIG_MODE_NOVECTOR, fill, order,
          work_matrix.device_ptr(), order, eigenvalues.device_ptr(),
          &workspace_size),
      "syevd condition-estimator workspace query");
  if (workspace_size < 0) {
    throw std::runtime_error{
        "hipSOLVER syevd condition-estimator workspace query returned a "
        "negative size"};
  }
  backend::DeviceBuffer<Real> workspace{
      static_cast<std::size_t>(workspace_size), backend::uninitialized,
      backend::on_device(device)};
  detail::check_hipsolver(
      hipsolverDnDsyevd(
          solver.get(), HIPSOLVER_EIG_MODE_NOVECTOR, fill, order,
          work_matrix.device_ptr(), order, eigenvalues.device_ptr(),
          workspace.device_ptr(), workspace_size, device_info.device_ptr()),
      "syevd condition estimator");

  SymmetricConditionEstimate result;
  result.order = order;
  result.solver_info = detail::read_info(device_info, stream);
  if (result.solver_info < 0) {
    result.status = SymmetricConditionStatus::invalid_solver_argument;
    result.invalid_argument_position = -result.solver_info;
    return result;
  }
  if (result.solver_info > 0) {
    result.status = SymmetricConditionStatus::failed_to_converge;
    // This solve requests eigenvalues only, for which LAPACK defines INFO as
    // the count of off-diagonal entries that did not converge to zero.
    result.unconverged_off_diagonal_count = result.solver_info;
    return result;
  }

  backend::DeviceBuffer<DeviceConditionSummary> device_summary{
      1, backend::on_device(device)};
  hipLaunchKernelGGL(summarize_spectrum_kernel, dim3{1}, dim3{1}, 0,
                     detail::as_hip_stream(stream), eigenvalues.device_ptr(),
                     order, relative_floor, device_summary.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());
  DeviceConditionSummary summary{};
  device_summary.copy_to_host_async(&summary, 1, stream);
  backend::device_synchronize(stream);

  result.smallest_absolute_eigenvalue = summary.smallest;
  result.largest_absolute_eigenvalue = summary.largest;
  result.condition_estimate = summary.condition;
  result.digits_lost = summary.digits;
  result.status = summary.singular != 0
                      ? SymmetricConditionStatus::numerically_singular
                      : SymmetricConditionStatus::success;
  return result;
}

}  // namespace quasar::numerics
