#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

// Backend-neutral device interface. This public header is deliberately
// HIP-free: it exposes an opaque stream handle and free-function memory
// primitives whose definitions live in src/backend/hip/. Consumers (core,
// physics, numerics, boundary, bindings, tests) include this without pulling in
// <hip/hip_runtime.h>, so the public API does not require ROCm to compile.

namespace quasar::backend {

// Opaque device-stream handle. Backed by hipStream_t inside src/backend/hip/;
// nullptr denotes the default stream. Callers treat it as an opaque token.
using stream_t = void*;
using event_t  = void*;

// Thrown by the backend on a device-runtime failure. `code` is the underlying
// runtime error code (stored as int so this header stays HIP-free).
struct DeviceError : public std::runtime_error {
  int         code;
  const char* file;
  int         line;

  DeviceError(int c, const char* msg, const char* f, int ln)
    : std::runtime_error{std::string{"device error "}
                         + std::to_string(c)
                         + " (" + (msg ? msg : "?") + ") at "
                         + (f ? f : "?") + ":" + std::to_string(ln)},
      code{c}, file{f}, line{ln} {}
};

int  device_count();
bool has_hip_runtime();
int  current_device();
void set_device(int device);
bool try_set_device(int device) noexcept;

// Temporarily select a device on the calling host thread and restore the
// previous selection on scope exit. HIP's current device is thread-local, so
// this guard also lets owner-aware buffers remain safe when orchestration code
// briefly touches allocations belonging to different worker devices.
class DeviceGuard {
 public:
  explicit DeviceGuard(int device);
  DeviceGuard(const DeviceGuard&)            = delete;
  DeviceGuard& operator=(const DeviceGuard&) = delete;
  DeviceGuard(DeviceGuard&&)                 = delete;
  DeviceGuard& operator=(DeviceGuard&&)      = delete;
  ~DeviceGuard() noexcept;

 private:
  int  previous_device_{-1};
  bool restore_{false};
};

// Physical identity/discovery primitives used before endpoint ownership is
// established. UUID is returned as 32 lowercase hexadecimal digits.
std::string device_uuid(int device);
std::string device_pci_bus_id(int device);
bool device_can_access_peer(int device, int peer_device);
void device_enable_peer_access(int peer_device);

// Non-default streams and events. Existing nullptr/default-stream behavior is
// retained for all legacy callers.
stream_t stream_create();
void     stream_destroy(stream_t stream) noexcept;
event_t  event_create();
void     event_destroy(event_t event) noexcept;
void     event_record(event_t event, stream_t stream);
void     event_synchronize(event_t event);
void     stream_wait_event(stream_t stream, event_t event);

// Device-memory primitives (implemented over HIP in src/backend/hip/). Throws
// DeviceError on failure. device_alloc zero-initializes the allocation;
// device_alloc_uninit skips the zero-fill for buffers a kernel overwrites in
// full before any read (transient scratch / output buffers).
void* device_alloc(std::size_t bytes);
void* device_alloc_uninit(std::size_t bytes);
void  device_free(void* ptr) noexcept;
void  device_free_on(void* ptr, int owner_device) noexcept;
void  device_memset(void* ptr, int value, std::size_t bytes);
// Asynchronous fill queued on `stream` (nullptr = default stream). Unlike
// device_memset (synchronous hipMemset), this does not block the host, so a
// per-step buffer clear stays off the critical path.
void  device_memset_async(void* ptr, int value, std::size_t bytes, stream_t stream);
void  device_memcpy_h2d(void* dst, const void* src, std::size_t bytes);
void  device_memcpy_d2h(void* dst, const void* src, std::size_t bytes);
void  device_memcpy_h2d_async(void* dst, const void* src, std::size_t bytes, stream_t stream);
void  device_memcpy_d2h_async(void* dst, const void* src, std::size_t bytes, stream_t stream);
void  device_memcpy_peer_async(void* dst, int dst_device,
                               const void* src, int src_device,
                               std::size_t bytes, stream_t stream);

// Page-locked host storage for staged inter-process transport.
void* pinned_host_alloc(std::size_t bytes);
void  pinned_host_free(void* ptr) noexcept;

// Block until all work on `stream` (nullptr = default stream) completes.
void  device_synchronize(stream_t stream);

}  // namespace quasar::backend
