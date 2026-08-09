#include "quasar/backend/device.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

// HIP-backed implementations of the backend-neutral memory primitives declared
// in include/quasar/backend/device.hpp. All HIP types/calls are confined here.

namespace quasar::backend {

namespace {
::hipStream_t as_hip(stream_t s) noexcept { return static_cast<::hipStream_t>(s); }
}  // namespace

void* device_alloc_uninit(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  void* ptr = nullptr;
  QUASAR_HIP_CHECK(::hipMalloc(&ptr, bytes));
  return ptr;
}

void* device_alloc(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  void* ptr = device_alloc_uninit(bytes);
  // Zero-initialize so freshly allocated buffers never observe recycled memory.
  try {
    QUASAR_HIP_CHECK(::hipMemset(ptr, 0, bytes));
  } catch (...) {
    // Construction has not yet transferred ownership to a DeviceBuffer. Avoid
    // leaking the successful allocation if the subsequent initialization fails.
    (void)::hipFree(ptr);
    throw;
  }
  return ptr;
}

void device_free(void* ptr) noexcept {
  // Must not throw (called from destructors); swallow free errors.
  (void)::hipFree(ptr);
}

void device_free_on(void* ptr, int owner_device) noexcept {
  if (ptr == nullptr) return;
  int previous_device = -1;
  const bool have_previous = (::hipGetDevice(&previous_device) == ::hipSuccess);
  const bool changed = have_previous && previous_device != owner_device
                    && ::hipSetDevice(owner_device) == ::hipSuccess;
  (void)::hipFree(ptr);
  if (changed) (void)::hipSetDevice(previous_device);
}

void device_memset(void* ptr, int value, std::size_t bytes) {
  if (bytes == 0) return;
  QUASAR_HIP_CHECK(::hipMemset(ptr, value, bytes));
}

void device_memset_async(void* ptr, int value, std::size_t bytes, stream_t stream) {
  if (bytes == 0) return;
  QUASAR_HIP_CHECK(::hipMemsetAsync(ptr, value, bytes, as_hip(stream)));
}

void device_synchronize(stream_t stream) {
  QUASAR_HIP_CHECK(::hipStreamSynchronize(as_hip(stream)));
}

void device_memcpy_h2d(void* dst, const void* src, std::size_t bytes) {
  if (bytes == 0) return;
  QUASAR_HIP_CHECK(::hipMemcpy(dst, src, bytes, ::hipMemcpyHostToDevice));
}

void device_memcpy_d2h(void* dst, const void* src, std::size_t bytes) {
  if (bytes == 0) return;
  QUASAR_HIP_CHECK(::hipMemcpy(dst, src, bytes, ::hipMemcpyDeviceToHost));
}

void device_memcpy_h2d_async(void* dst, const void* src, std::size_t bytes, stream_t stream) {
  if (bytes == 0) return;
  QUASAR_HIP_CHECK(::hipMemcpyAsync(dst, src, bytes, ::hipMemcpyHostToDevice, as_hip(stream)));
}

void device_memcpy_d2h_async(void* dst, const void* src, std::size_t bytes, stream_t stream) {
  if (bytes == 0) return;
  QUASAR_HIP_CHECK(::hipMemcpyAsync(dst, src, bytes, ::hipMemcpyDeviceToHost, as_hip(stream)));
}

void device_memcpy_peer_async(void* dst, int dst_device,
                              const void* src, int src_device,
                              std::size_t bytes, stream_t stream) {
  if (bytes == 0) return;
  QUASAR_HIP_CHECK(::hipMemcpyPeerAsync(
      dst, dst_device, src, src_device, bytes, as_hip(stream)));
}

void* pinned_host_alloc(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  void* ptr = nullptr;
  QUASAR_HIP_CHECK(::hipHostMalloc(&ptr, bytes, hipHostMallocPortable));
  return ptr;
}

void pinned_host_free(void* ptr) noexcept {
  if (ptr != nullptr) (void)::hipHostFree(ptr);
}

}  // namespace quasar::backend
