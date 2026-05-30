#include "quasar/backend/device.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

// HIP-backed implementations of the backend-neutral memory primitives declared
// in include/quasar/backend/device.hpp. All HIP types/calls are confined here.

namespace quasar::backend {

namespace {
::hipStream_t as_hip(stream_t s) noexcept { return static_cast<::hipStream_t>(s); }
}  // namespace

void* device_alloc(std::size_t bytes) {
  void* ptr = nullptr;
  QUASAR_HIP_CHECK(::hipMalloc(&ptr, bytes));
  // Zero-initialize so freshly allocated buffers never observe recycled memory.
  QUASAR_HIP_CHECK(::hipMemset(ptr, 0, bytes));
  return ptr;
}

void device_free(void* ptr) noexcept {
  // Must not throw (called from destructors); swallow free errors.
  (void)::hipFree(ptr);
}

void device_memset(void* ptr, int value, std::size_t bytes) {
  QUASAR_HIP_CHECK(::hipMemset(ptr, value, bytes));
}

void device_synchronize(stream_t stream) {
  QUASAR_HIP_CHECK(::hipStreamSynchronize(as_hip(stream)));
}

void device_memcpy_h2d(void* dst, const void* src, std::size_t bytes) {
  QUASAR_HIP_CHECK(::hipMemcpy(dst, src, bytes, ::hipMemcpyHostToDevice));
}

void device_memcpy_d2h(void* dst, const void* src, std::size_t bytes) {
  QUASAR_HIP_CHECK(::hipMemcpy(dst, src, bytes, ::hipMemcpyDeviceToHost));
}

void device_memcpy_h2d_async(void* dst, const void* src, std::size_t bytes, stream_t stream) {
  QUASAR_HIP_CHECK(::hipMemcpyAsync(dst, src, bytes, ::hipMemcpyHostToDevice, as_hip(stream)));
}

void device_memcpy_d2h_async(void* dst, const void* src, std::size_t bytes, stream_t stream) {
  QUASAR_HIP_CHECK(::hipMemcpyAsync(dst, src, bytes, ::hipMemcpyDeviceToHost, as_hip(stream)));
}

}  // namespace quasar::backend
