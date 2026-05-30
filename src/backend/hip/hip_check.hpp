#pragma once

// Private backend-internal HIP error-check macro. Lives under src/backend/hip/
// so it is never installed and never leaks <hip/hip_runtime.h> into public
// headers. Throws quasar::backend::DeviceError (HIP-free type) on failure.

#include "quasar/backend/device.hpp"

#include <hip/hip_runtime.h>

#define QUASAR_HIP_CHECK(EXPR)                                                       \
  do {                                                                              \
    ::hipError_t _quasar_hip_check_err = (EXPR);                                    \
    if (_quasar_hip_check_err != ::hipSuccess) {                                    \
      throw ::quasar::backend::DeviceError{                                         \
          static_cast<int>(_quasar_hip_check_err),                                  \
          ::hipGetErrorString(_quasar_hip_check_err), __FILE__, __LINE__};          \
    }                                                                              \
  } while (0)
