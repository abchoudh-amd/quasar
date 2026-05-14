#include "quasar/backend/device.hpp"

#include <hip/hip_runtime.h>

namespace quasar::backend {

int device_count() {
  int n = 0;
  const ::hipError_t e = ::hipGetDeviceCount(&n);
  if (e != ::hipSuccess) {
    return 0;
  }
  return n;
}

bool has_hip_runtime() {
  static const bool cached = (device_count() > 0);
  return cached;
}

}  // namespace quasar::backend
