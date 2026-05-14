#pragma once

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace quasar::backend {

using stream_t = hipStream_t;

struct HipError : public std::runtime_error {
  hipError_t  code;
  const char* file;
  int         line;

  HipError(hipError_t c, const char* msg, const char* f, int ln)
    : std::runtime_error{std::string{"HIP error "}
                         + std::to_string(static_cast<int>(c))
                         + " (" + (msg ? msg : "?") + ") at "
                         + (f ? f : "?") + ":" + std::to_string(ln)},
      code{c}, file{f}, line{ln} {}
};

int  device_count();
bool has_hip_runtime();

}  // namespace quasar::backend

#define QUASAR_HIP_CHECK(EXPR)                                                          \
  do {                                                                                  \
    ::hipError_t _quasar_hip_check_err = (EXPR);                                        \
    if (_quasar_hip_check_err != ::hipSuccess) {                                        \
      throw ::quasar::backend::HipError{_quasar_hip_check_err,                          \
                                        ::hipGetErrorString(_quasar_hip_check_err),    \
                                        __FILE__,                                       \
                                        __LINE__};                                      \
    }                                                                                   \
  } while (0)
