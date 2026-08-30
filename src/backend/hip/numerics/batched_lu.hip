// Batched dense LU on the active HIP device, over rocSOLVER's strided-batched
// getrf/getrs.
//
// rocSOLVER rather than hipSOLVER because hipSOLVER exposes only the
// single-matrix getrf/getrs; there is no batched entry point in it. Driving the
// single-matrix path once per system would mean thousands of library calls for
// the radial moment tables, whose systems are at most 8x8.

#include "quasar/numerics/batched_lu.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace quasar::numerics {

namespace {

static_assert(std::is_same_v<Real, double>,
              "solve_batched_dense_lu binds the double-precision rocSOLVER "
              "entry points by name; Real must be double.");

void check_rocblas(rocblas_status status, const char* operation) {
  if (status == rocblas_status_success) return;
  const char* description = rocblas_status_to_string(status);
  throw std::runtime_error{
      std::string{"rocSOLVER "} + operation + " failed with status "
      + std::to_string(static_cast<int>(status)) + " ("
      + (description != nullptr ? description : "unrecognized status") + ")"};
}

// Owns one rocBLAS handle bound to the caller's stream.
class RocblasHandle {
 public:
  explicit RocblasHandle(backend::stream_t stream) {
    check_rocblas(rocblas_create_handle(&handle_), "create handle");
    try {
      check_rocblas(
          rocblas_set_stream(handle_, static_cast<hipStream_t>(stream)),
          "set stream");
      // The factorizations below are per-system and reduce nothing across the
      // batch, but rocBLAS may still use atomics internally for other work
      // dispatched on this handle. Disallow them, matching the discipline the
      // hipSOLVER handle in linear_algebra_detail.hpp follows.
      check_rocblas(
          rocblas_set_atomics_mode(handle_, rocblas_atomics_not_allowed),
          "disable atomics");
    } catch (...) {
      rocblas_destroy_handle(handle_);
      throw;
    }
  }

  RocblasHandle(const RocblasHandle&) = delete;
  RocblasHandle& operator=(const RocblasHandle&) = delete;

  ~RocblasHandle() { rocblas_destroy_handle(handle_); }

  rocblas_handle get() const noexcept { return handle_; }

 private:
  rocblas_handle handle_{nullptr};
};

std::size_t checked_product(std::size_t a, std::size_t b, const char* what) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::length_error{std::string{"solve_batched_dense_lu: "} + what
                            + " is not representable"};
  }
  return a * b;
}

}  // namespace

BatchedLuResult solve_batched_dense_lu(backend::DeviceBuffer<Real>& matrices,
                                       backend::DeviceBuffer<Real>& rhs,
                                       int order, int count,
                                       backend::stream_t stream) {
  if (order <= 0 || count < 0) {
    throw std::invalid_argument{
        "solve_batched_dense_lu: order must be positive and count "
        "non-negative"};
  }
  const std::size_t n = static_cast<std::size_t>(order);
  const std::size_t batch = static_cast<std::size_t>(count);
  const std::size_t matrix_values =
      checked_product(checked_product(n, n, "matrix size"), batch,
                      "batch matrix size");
  const std::size_t rhs_values = checked_product(n, batch, "batch rhs size");
  if (matrices.size() != matrix_values || rhs.size() != rhs_values) {
    throw std::invalid_argument{
        "solve_batched_dense_lu: matrices must hold order*order*count values "
        "and rhs must hold order*count"};
  }

  BatchedLuResult result;
  result.order = order;
  result.count = count;
  if (count == 0) return result;

  const RocblasHandle handle{stream};
  backend::DeviceBuffer<int> pivots(rhs_values, backend::uninitialized);
  backend::DeviceBuffer<int> info(batch);  // zero-initialized

  const rocblas_int rocblas_order = static_cast<rocblas_int>(order);
  const rocblas_int rocblas_count = static_cast<rocblas_int>(count);
  const rocblas_stride matrix_stride = static_cast<rocblas_stride>(n * n);
  const rocblas_stride vector_stride = static_cast<rocblas_stride>(n);

  check_rocblas(
      rocsolver_dgetrf_strided_batched(
          handle.get(), rocblas_order, rocblas_order, matrices.device_ptr(),
          rocblas_order, matrix_stride, pivots.device_ptr(), vector_stride,
          info.device_ptr(), rocblas_count),
      "getrf_strided_batched");

  // Read the per-system info before solving. getrs on a singular factorization
  // is undefined, and reporting *which* system failed is what lets the caller
  // name the offending stencil row rather than the whole table.
  std::vector<int> host_info(batch, 0);
  info.copy_to_host_async(host_info.data(), batch, stream);
  backend::device_synchronize(stream);

  for (std::size_t s = 0; s < batch; ++s) {
    if (host_info[s] == 0) continue;
    result.status = host_info[s] < 0 ? BatchedLuStatus::invalid_solver_argument
                                     : BatchedLuStatus::singular;
    result.first_singular_system = static_cast<int>(s);
    result.failed_pivot_column = host_info[s];
    return result;
  }

  check_rocblas(
      rocsolver_dgetrs_strided_batched(
          handle.get(), rocblas_operation_none, rocblas_order,
          /*nrhs=*/1, matrices.device_ptr(), rocblas_order, matrix_stride,
          pivots.device_ptr(), vector_stride, rhs.device_ptr(), rocblas_order,
          vector_stride, rocblas_count),
      "getrs_strided_batched");
  backend::device_synchronize(stream);
  return result;
}

}  // namespace quasar::numerics
