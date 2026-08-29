#pragma once

#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>

#include <stdexcept>
#include <string>

namespace quasar::numerics::detail {

inline const char* hipsolver_status_name(hipsolverStatus_t status) noexcept {
  switch (status) {
    case HIPSOLVER_STATUS_SUCCESS: return "success";
    case HIPSOLVER_STATUS_NOT_INITIALIZED: return "not initialized";
    case HIPSOLVER_STATUS_ALLOC_FAILED: return "allocation failed";
    case HIPSOLVER_STATUS_INVALID_VALUE: return "invalid value";
    case HIPSOLVER_STATUS_MAPPING_ERROR: return "mapping error";
    case HIPSOLVER_STATUS_EXECUTION_FAILED: return "execution failed";
    case HIPSOLVER_STATUS_INTERNAL_ERROR: return "internal error";
    case HIPSOLVER_STATUS_NOT_SUPPORTED: return "not supported";
    case HIPSOLVER_STATUS_ARCH_MISMATCH: return "architecture mismatch";
    case HIPSOLVER_STATUS_HANDLE_IS_NULLPTR: return "null handle";
    case HIPSOLVER_STATUS_INVALID_ENUM: return "invalid enum";
    case HIPSOLVER_STATUS_UNKNOWN: return "unknown";
    case HIPSOLVER_STATUS_ZERO_PIVOT: return "zero pivot";
    case HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "matrix type not supported";
  }
  return "unrecognized status";
}

inline void check_hipsolver(hipsolverStatus_t status, const char* operation) {
  if (status == HIPSOLVER_STATUS_SUCCESS) return;
  throw std::runtime_error{
      std::string{"hipSOLVER "} + operation + " failed with status "
      + std::to_string(static_cast<int>(status)) + " ("
      + hipsolver_status_name(status) + ")"};
}

inline void check_hipblas(hipblasStatus_t status, const char* operation) {
  if (status == HIPBLAS_STATUS_SUCCESS) return;
  const char* description = hipblasStatusToString(status);
  throw std::runtime_error{
      std::string{"hipBLAS "} + operation + " failed with status "
      + std::to_string(static_cast<int>(status)) + " ("
      + (description != nullptr ? description : "unrecognized status") + ")"};
}

inline hipStream_t as_hip_stream(backend::stream_t stream) noexcept {
  return static_cast<hipStream_t>(stream);
}

class SolverHandle {
 public:
  explicit SolverHandle(backend::stream_t stream) {
    check_hipsolver(hipsolverDnCreate(&handle_), "create");
    try {
      check_hipsolver(
          hipsolverSetDeterministicMode(handle_,
                                        HIPSOLVER_DETERMINISTIC_RESULTS),
          "set deterministic mode");
      check_hipsolver(hipsolverDnSetStream(handle_, as_hip_stream(stream)),
                      "set stream");
    } catch (...) {
      (void)hipsolverDnDestroy(handle_);
      handle_ = nullptr;
      throw;
    }
  }

  SolverHandle(const SolverHandle&) = delete;
  SolverHandle& operator=(const SolverHandle&) = delete;

  ~SolverHandle() {
    if (handle_ != nullptr) (void)hipsolverDnDestroy(handle_);
  }

  [[nodiscard]] hipsolverHandle_t get() const noexcept { return handle_; }

 private:
  hipsolverHandle_t handle_{nullptr};
};

class BlasHandle {
 public:
  explicit BlasHandle(backend::stream_t stream) {
    check_hipblas(hipblasCreate(&handle_), "create");
    try {
      check_hipblas(hipblasSetStream(handle_, as_hip_stream(stream)),
                    "set stream");
      check_hipblas(
          hipblasSetPointerMode(handle_, HIPBLAS_POINTER_MODE_HOST),
          "set host scalar pointer mode");
      check_hipblas(
          hipblasSetAtomicsMode(handle_, HIPBLAS_ATOMICS_NOT_ALLOWED),
          "disable atomics");
    } catch (...) {
      (void)hipblasDestroy(handle_);
      handle_ = nullptr;
      throw;
    }
  }

  BlasHandle(const BlasHandle&) = delete;
  BlasHandle& operator=(const BlasHandle&) = delete;

  ~BlasHandle() {
    if (handle_ != nullptr) (void)hipblasDestroy(handle_);
  }

  [[nodiscard]] hipblasHandle_t get() const noexcept { return handle_; }

 private:
  hipblasHandle_t handle_{nullptr};
};

inline int read_info(const backend::DeviceBuffer<int>& device_info,
                     backend::stream_t stream) {
  int info = 0;
  device_info.copy_to_host_async(&info, 1, stream);
  backend::device_synchronize(stream);
  return info;
}

inline void copy_device_buffer(backend::DeviceBuffer<Real>& destination,
                               const backend::DeviceBuffer<Real>& source,
                               std::size_t count,
                               backend::stream_t stream) {
  if (count == 0) return;
  backend::device_memcpy_d2d_async(destination.device_ptr(), source.device_ptr(),
                                   count * sizeof(Real), stream);
}

}  // namespace quasar::numerics::detail
