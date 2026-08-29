#include "quasar/numerics/generalized_eigensolver.hpp"

#include "backend/hip/numerics/linear_algebra_detail.hpp"

#include <cstddef>
#include <stdexcept>

namespace quasar::numerics {
namespace {

void validate_problem(const backend::DeviceBuffer<Real>& a,
                      const backend::DeviceBuffer<Real>& b,
                      int order) {
  if (order <= 0) {
    throw std::invalid_argument{
        "solve_generalized_symmetric_eigenproblem: order must be positive"};
  }
  const auto n = static_cast<std::size_t>(order);
  const auto matrix_size = n * n;
  if (a.size() != matrix_size || b.size() != matrix_size) {
    throw std::invalid_argument{
        "solve_generalized_symmetric_eigenproblem: A and B must each contain "
        "exactly order*order column-major values"};
  }
  if (a.owner_device() != b.owner_device()) {
    throw std::invalid_argument{
        "solve_generalized_symmetric_eigenproblem: A and B reside on "
        "different devices"};
  }
}

}  // namespace

GeneralizedEigenResult solve_generalized_symmetric_eigenproblem(
    const backend::DeviceBuffer<Real>& a,
    const backend::DeviceBuffer<Real>& b,
    int order,
    MatrixTriangle triangle,
    backend::stream_t stream) {
  validate_problem(a, b, order);

  const auto n = static_cast<std::size_t>(order);
  const auto matrix_size = n * n;
  const int device = a.owner_device();
  backend::DeviceGuard device_guard{device};

  GeneralizedEigenResult result;
  result.order = order;
  result.eigenvalues = backend::DeviceBuffer<Real>{
      n, backend::uninitialized, backend::on_device(device)};
  result.eigenvectors = backend::DeviceBuffer<Real>{
      matrix_size, backend::uninitialized, backend::on_device(device)};
  backend::DeviceBuffer<Real> mass_copy{
      matrix_size, backend::uninitialized, backend::on_device(device)};
  backend::DeviceBuffer<int> device_info{1, backend::on_device(device)};

  // Both copies are intentionally explicit: sygvd destroys A and B even when
  // it reports a numerical failure, while the public inputs are const oracles.
  detail::copy_device_buffer(result.eigenvectors, a, matrix_size, stream);
  detail::copy_device_buffer(mass_copy, b, matrix_size, stream);

  detail::SolverHandle solver{stream};
  const hipblasFillMode_t fill =
      triangle == MatrixTriangle::lower ? HIPBLAS_FILL_MODE_LOWER
                                        : HIPBLAS_FILL_MODE_UPPER;

  int workspace_size = 0;
  detail::check_hipsolver(
      hipsolverDnDsygvd_bufferSize(
          solver.get(), HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_VECTOR,
          fill, order, result.eigenvectors.device_ptr(), order,
          mass_copy.device_ptr(), order, result.eigenvalues.device_ptr(),
          &workspace_size),
      "sygvd workspace query");
  if (workspace_size < 0) {
    throw std::runtime_error{
        "hipSOLVER sygvd workspace query returned a negative size"};
  }
  backend::DeviceBuffer<Real> workspace{
      static_cast<std::size_t>(workspace_size), backend::uninitialized,
      backend::on_device(device)};

  detail::check_hipsolver(
      hipsolverDnDsygvd(
          solver.get(), HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_VECTOR,
          fill, order, result.eigenvectors.device_ptr(), order,
          mass_copy.device_ptr(), order, result.eigenvalues.device_ptr(),
          workspace.device_ptr(), workspace_size, device_info.device_ptr()),
      "sygvd");

  result.solver_info = detail::read_info(device_info, stream);
  if (result.solver_info == 0) {
    result.status = GeneralizedEigenStatus::success;
  } else if (result.solver_info < 0) {
    result.status = GeneralizedEigenStatus::invalid_solver_argument;
    result.invalid_argument_position = -result.solver_info;
  } else if (result.solver_info <= order) {
    result.status = GeneralizedEigenStatus::failed_to_converge;
  } else {
    result.status = GeneralizedEigenStatus::mass_not_positive_definite;
    result.failed_leading_minor_order = result.solver_info - order;
  }
  return result;
}

}  // namespace quasar::numerics
