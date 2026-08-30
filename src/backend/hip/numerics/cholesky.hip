#include "quasar/numerics/cholesky.hpp"

#include "backend/hip/numerics/linear_algebra_detail.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace quasar::numerics {

CholeskyResult check_symmetric_positive_definite(
    const backend::DeviceBuffer<Real>& matrix,
    int order,
    MatrixTriangle triangle,
    backend::stream_t stream) {
  if (order <= 0) {
    throw std::invalid_argument{
        "check_symmetric_positive_definite: order must be positive"};
  }
  const auto n = static_cast<std::size_t>(order);
  if (n > std::numeric_limits<std::size_t>::max() / n
      || matrix.size() != n * n) {
    throw std::invalid_argument{
        "check_symmetric_positive_definite: matrix must contain exactly "
        "order*order column-major values"};
  }

  const int device = matrix.owner_device();
  backend::DeviceGuard guard{device};
  const auto matrix_size = n * n;

  // potrf overwrites its argument, so the public const input is copied.
  backend::DeviceBuffer<Real> factor{
      matrix_size, backend::uninitialized, backend::on_device(device)};
  backend::DeviceBuffer<int> device_info{1, backend::on_device(device)};
  detail::copy_device_buffer(factor, matrix, matrix_size, stream);

  const hipblasFillMode_t fill =
      triangle == MatrixTriangle::lower ? HIPBLAS_FILL_MODE_LOWER
                                        : HIPBLAS_FILL_MODE_UPPER;
  detail::SolverHandle solver{stream};
  int workspace_size = 0;
  detail::check_hipsolver(
      hipsolverDnDpotrf_bufferSize(solver.get(), fill, order,
                                   factor.device_ptr(), order,
                                   &workspace_size),
      "potrf workspace query");
  if (workspace_size < 0) {
    throw std::runtime_error{
        "hipSOLVER potrf workspace query returned a negative size"};
  }
  backend::DeviceBuffer<Real> workspace{
      static_cast<std::size_t>(workspace_size), backend::uninitialized,
      backend::on_device(device)};
  detail::check_hipsolver(
      hipsolverDnDpotrf(solver.get(), fill, order, factor.device_ptr(), order,
                        workspace.device_ptr(), workspace_size,
                        device_info.device_ptr()),
      "potrf");

  CholeskyResult result;
  result.order = order;
  result.solver_info = detail::read_info(device_info, stream);
  if (result.solver_info < 0) {
    result.status = CholeskyStatus::invalid_solver_argument;
    result.invalid_argument_position = -result.solver_info;
  } else if (result.solver_info > 0) {
    result.status = CholeskyStatus::not_positive_definite;
    result.failed_leading_minor_order = result.solver_info;
  }
  return result;
}

}  // namespace quasar::numerics
