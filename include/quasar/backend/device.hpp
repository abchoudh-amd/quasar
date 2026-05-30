#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

// Backend-neutral device interface. This installed header is deliberately
// HIP-free: it exposes an opaque stream handle and free-function memory
// primitives whose definitions live in src/backend/hip/. Consumers (core,
// physics, numerics, boundary, bindings, tests) include this without pulling in
// <hip/hip_runtime.h>, so the public API does not require ROCm to compile.

namespace quasar::backend {

// Opaque device-stream handle. Backed by hipStream_t inside src/backend/hip/;
// nullptr denotes the default stream. Callers treat it as an opaque token.
using stream_t = void*;

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

// Device-memory primitives (implemented over HIP in src/backend/hip/). Returned
// allocations are zero-initialized. Throws DeviceError on failure.
void* device_alloc(std::size_t bytes);
void  device_free(void* ptr) noexcept;
void  device_memcpy_h2d(void* dst, const void* src, std::size_t bytes);
void  device_memcpy_d2h(void* dst, const void* src, std::size_t bytes);
void  device_memcpy_h2d_async(void* dst, const void* src, std::size_t bytes, stream_t stream);
void  device_memcpy_d2h_async(void* dst, const void* src, std::size_t bytes, stream_t stream);

}  // namespace quasar::backend
