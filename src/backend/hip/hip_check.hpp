#pragma once

// Private backend-internal HIP error-check macro. Lives under src/backend/hip/
// so it is never installed and never leaks <hip/hip_runtime.h> into public
// headers. Throws quasar::backend::DeviceError (HIP-free type) on failure.

#include "quasar/backend/device.hpp"

#include <hip/hip_runtime.h>

// Occupancy hint for kernels launched at the standard 256-thread block.
//
// Without an explicit bound the compiler must assume the HIP maximum of 1024
// threads per block, which on CDNA caps a kernel at 512/4 = 128 VGPRs. Every
// launcher in this backend uses 256 threads (16x16 for 2-D grids, 256x1 for
// 1-D), so declaring that lets the register allocator use the real budget --
// up to 512 VGPRs at one wave, 256 at two -- instead of spilling to scratch to
// stay under a limit no launch actually needs.
//
// This is an assertion about the launch geometry, not a request: launching a
// kernel with more than 256 threads per block after annotating it is undefined.
// Keep it in lockstep with kBlock2D/kLaunchBlock (16*16) and kBlock/
// kLaunchBlock (256). Kernels are annotated individually rather than globally
// so an intentionally larger launch simply omits the macro.
#define QUASAR_LAUNCH_BOUNDS_256 __launch_bounds__(256)

#define QUASAR_HIP_CHECK(EXPR)                                                       \
  do {                                                                              \
    ::hipError_t _quasar_hip_check_err = (EXPR);                                    \
    if (_quasar_hip_check_err != ::hipSuccess) {                                    \
      throw ::quasar::backend::DeviceError{                                         \
          static_cast<int>(_quasar_hip_check_err),                                  \
          ::hipGetErrorString(_quasar_hip_check_err), __FILE__, __LINE__};          \
    }                                                                              \
  } while (0)
