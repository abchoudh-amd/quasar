#pragma once

#include "quasar/backend/device.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace quasar::backend {

// RAII owner of a device-side buffer. Move-only. Allocation/free goes through
// hipMalloc/hipFree wrapped in QUASAR_HIP_CHECK. Async copy variants accept a
// quasar::backend::stream_t.
template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() noexcept = default;

  explicit DeviceBuffer(std::size_t n) : size_{n}, bytes_{n * sizeof(T)} {
    if (n != 0) {
      QUASAR_HIP_CHECK(::hipMalloc(&ptr_, bytes_));
      // Zero-initialize so freshly constructed fields/particles never observe
      // recycled device memory from a prior allocation.
      QUASAR_HIP_CHECK(::hipMemset(ptr_, 0, bytes_));
    }
  }

  DeviceBuffer(const DeviceBuffer&)            = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_{other.ptr_}, size_{other.size_}, bytes_{other.bytes_} {
    other.ptr_   = nullptr;
    other.size_  = 0;
    other.bytes_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      ptr_         = other.ptr_;
      size_        = other.size_;
      bytes_       = other.bytes_;
      other.ptr_   = nullptr;
      other.size_  = 0;
      other.bytes_ = 0;
    }
    return *this;
  }

  ~DeviceBuffer() { release(); }

  T*          device_ptr()       noexcept { return static_cast<T*>(ptr_); }
  const T*    device_ptr() const noexcept { return static_cast<const T*>(ptr_); }
  std::size_t size()       const noexcept { return size_; }
  std::size_t bytes()      const noexcept { return bytes_; }
  bool        empty()      const noexcept { return size_ == 0; }

  void copy_from_host(const T* host_src, std::size_t n) {
    QUASAR_HIP_CHECK(::hipMemcpy(ptr_, host_src, n * sizeof(T),
                                 ::hipMemcpyHostToDevice));
  }
  void copy_from_host_async(const T* host_src, std::size_t n, stream_t stream) {
    QUASAR_HIP_CHECK(::hipMemcpyAsync(ptr_, host_src, n * sizeof(T),
                                      ::hipMemcpyHostToDevice, stream));
  }
  void copy_to_host(T* host_dst, std::size_t n) const {
    QUASAR_HIP_CHECK(::hipMemcpy(host_dst, ptr_, n * sizeof(T),
                                 ::hipMemcpyDeviceToHost));
  }
  void copy_to_host_async(T* host_dst, std::size_t n, stream_t stream) const {
    QUASAR_HIP_CHECK(::hipMemcpyAsync(host_dst, ptr_, n * sizeof(T),
                                      ::hipMemcpyDeviceToHost, stream));
  }

 private:
  void release() noexcept {
    if (ptr_ != nullptr) {
      // Destructor must not throw; swallow free errors.
      (void)::hipFree(ptr_);
      ptr_ = nullptr;
    }
    size_  = 0;
    bytes_ = 0;
  }

  void*       ptr_{nullptr};
  std::size_t size_{0};
  std::size_t bytes_{0};
};

// Paired host vector + device buffer with sync_to_device / sync_to_host helpers.
template <class T>
class mirror_view {
 public:
  mirror_view() = default;

  explicit mirror_view(std::size_t n) : host_(n), device_{n} {}

  std::size_t size()  const noexcept { return host_.size(); }
  bool        empty() const noexcept { return host_.empty(); }

  T*       host_data()        noexcept { return host_.data(); }
  const T* host_data()  const noexcept { return host_.data(); }
  T*       device_ptr()       noexcept { return device_.device_ptr(); }
  const T* device_ptr() const noexcept { return device_.device_ptr(); }

  std::vector<T>&        host()        noexcept { return host_; }
  const std::vector<T>&  host()  const noexcept { return host_; }
  DeviceBuffer<T>&       device()       noexcept { return device_; }
  const DeviceBuffer<T>& device() const noexcept { return device_; }

  void sync_to_device() {
    device_.copy_from_host(host_.data(), host_.size());
  }
  void sync_to_device_async(stream_t stream) {
    device_.copy_from_host_async(host_.data(), host_.size(), stream);
  }
  void sync_to_host() {
    device_.copy_to_host(host_.data(), host_.size());
  }
  void sync_to_host_async(stream_t stream) {
    device_.copy_to_host_async(host_.data(), host_.size(), stream);
  }

 private:
  std::vector<T>  host_{};
  DeviceBuffer<T> device_{};
};

}  // namespace quasar::backend
