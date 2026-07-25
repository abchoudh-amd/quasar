#pragma once

#include "quasar/backend/device.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quasar::backend {

namespace detail {

inline std::size_t checked_size_product(std::size_t a, std::size_t b,
                                        const char* message) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::length_error{message};
  }
  return a * b;
}

template <class T>
std::size_t checked_buffer_bytes(std::size_t n) {
  return checked_size_product(
      n, sizeof(T), "DeviceBuffer: element count overflows byte size");
}

}  // namespace detail

// Tag selecting an uninitialized device allocation (skips the zero-fill) for a
// buffer a kernel overwrites in full before any read.
struct uninitialized_t {};
inline constexpr uninitialized_t uninitialized{};

// RAII owner of a device-side buffer. Move-only. Allocation/free goes through
// hipMalloc/hipFree wrapped in QUASAR_HIP_CHECK. Async copy variants accept a
// quasar::backend::stream_t.
template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() noexcept = default;

  explicit DeviceBuffer(std::size_t n)
    : size_{n}, bytes_{detail::checked_buffer_bytes<T>(n)} {
    if (n != 0) {
      // device_alloc zero-initializes, so freshly constructed fields/particles
      // never observe recycled device memory from a prior allocation.
      ptr_ = device_alloc(bytes_);
    }
  }

  // Allocate without the zero-fill. Only valid when the caller writes every
  // element before reading (transient scratch / kernel-output buffers).
  DeviceBuffer(std::size_t n, uninitialized_t)
    : size_{n}, bytes_{detail::checked_buffer_bytes<T>(n)} {
    if (n != 0) {
      ptr_ = device_alloc_uninit(bytes_);
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
    validate_copy(host_src, n, "copy_from_host");
    if (n == 0) return;
    device_memcpy_h2d(ptr_, host_src, n * sizeof(T));
  }
  void copy_from_host_async(const T* host_src, std::size_t n, stream_t stream) {
    validate_copy(host_src, n, "copy_from_host_async");
    if (n == 0) return;
    device_memcpy_h2d_async(ptr_, host_src, n * sizeof(T), stream);
  }
  void copy_to_host(T* host_dst, std::size_t n) const {
    validate_copy(host_dst, n, "copy_to_host");
    if (n == 0) return;
    device_memcpy_d2h(host_dst, ptr_, n * sizeof(T));
  }
  void copy_to_host_async(T* host_dst, std::size_t n, stream_t stream) const {
    validate_copy(host_dst, n, "copy_to_host_async");
    if (n == 0) return;
    device_memcpy_d2h_async(host_dst, ptr_, n * sizeof(T), stream);
  }

 private:
  void validate_copy(const void* host, std::size_t n, const char* operation) const {
    if (n > size_) {
      throw std::out_of_range{std::string{"DeviceBuffer::"} + operation
                              + ": copy exceeds buffer size"};
    }
    if (n != 0 && host == nullptr) {
      throw std::invalid_argument{std::string{"DeviceBuffer::"} + operation
                                  + ": host pointer is null"};
    }
  }

  void release() noexcept {
    if (ptr_ != nullptr) {
      device_free(ptr_);  // noexcept; swallows free errors
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
