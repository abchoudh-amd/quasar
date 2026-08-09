#pragma once

#include "quasar/backend/device.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
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

struct device_ordinal_t {
  int value{-1};
};

inline device_ordinal_t on_device(int ordinal) {
  if (ordinal < 0) {
    throw std::invalid_argument{"device ordinal must be non-negative"};
  }
  return {ordinal};
}

// Move-only owner of a non-default HIP stream. Construction records the device
// that owns the stream; legacy code can continue using nullptr as the default
// stream without constructing this type.
class DeviceStream {
 public:
  DeviceStream() : owner_device_{current_device()}, stream_{stream_create()} {}

  explicit DeviceStream(device_ordinal_t device)
    : owner_device_{device.value} {
    DeviceGuard guard{owner_device_};
    stream_ = stream_create();
  }

  DeviceStream(const DeviceStream&)            = delete;
  DeviceStream& operator=(const DeviceStream&) = delete;

  DeviceStream(DeviceStream&& other) noexcept
    : owner_device_{other.owner_device_}, stream_{other.stream_} {
    other.owner_device_ = -1;
    other.stream_       = nullptr;
  }

  DeviceStream& operator=(DeviceStream&& other) noexcept {
    if (this != &other) {
      release();
      owner_device_       = other.owner_device_;
      stream_             = other.stream_;
      other.owner_device_ = -1;
      other.stream_       = nullptr;
    }
    return *this;
  }

  ~DeviceStream() { release(); }

  [[nodiscard]] stream_t get() const noexcept { return stream_; }
  [[nodiscard]] int owner_device() const noexcept { return owner_device_; }
  [[nodiscard]] explicit operator bool() const noexcept {
    return stream_ != nullptr;
  }

  void synchronize() const {
    DeviceGuard guard{owner_device_};
    device_synchronize(stream_);
  }

 private:
  void release() noexcept {
    if (stream_ != nullptr) {
      try {
        DeviceGuard guard{owner_device_};
        stream_destroy(stream_);
      } catch (...) {
        // Destructors cannot surface runtime errors. The backend destroy
        // primitive is itself best-effort and may still release the handle.
        stream_destroy(stream_);
      }
    }
    owner_device_ = -1;
    stream_       = nullptr;
  }

  int      owner_device_{-1};
  stream_t stream_{nullptr};
};

// Move-only event with the same device ownership discipline as DeviceStream.
class DeviceEvent {
 public:
  DeviceEvent() : owner_device_{current_device()}, event_{event_create()} {}

  explicit DeviceEvent(device_ordinal_t device)
    : owner_device_{device.value} {
    DeviceGuard guard{owner_device_};
    event_ = event_create();
  }

  DeviceEvent(const DeviceEvent&)            = delete;
  DeviceEvent& operator=(const DeviceEvent&) = delete;

  DeviceEvent(DeviceEvent&& other) noexcept
    : owner_device_{other.owner_device_}, event_{other.event_} {
    other.owner_device_ = -1;
    other.event_        = nullptr;
  }

  DeviceEvent& operator=(DeviceEvent&& other) noexcept {
    if (this != &other) {
      release();
      owner_device_       = other.owner_device_;
      event_              = other.event_;
      other.owner_device_ = -1;
      other.event_        = nullptr;
    }
    return *this;
  }

  ~DeviceEvent() { release(); }

  [[nodiscard]] event_t get() const noexcept { return event_; }
  [[nodiscard]] int owner_device() const noexcept { return owner_device_; }
  [[nodiscard]] explicit operator bool() const noexcept {
    return event_ != nullptr;
  }

  void record(stream_t stream) {
    DeviceGuard guard{owner_device_};
    event_record(event_, stream);
  }

  void record(const DeviceStream& stream) {
    if (stream.owner_device() != owner_device_) {
      throw std::invalid_argument{
          "DeviceEvent::record: stream and event owners differ"};
    }
    record(stream.get());
  }

  void synchronize() const {
    DeviceGuard guard{owner_device_};
    event_synchronize(event_);
  }

 private:
  void release() noexcept {
    if (event_ != nullptr) {
      try {
        DeviceGuard guard{owner_device_};
        event_destroy(event_);
      } catch (...) {
        event_destroy(event_);
      }
    }
    owner_device_ = -1;
    event_        = nullptr;
  }

  int     owner_device_{-1};
  event_t event_{nullptr};
};

// Page-locked host storage used by staged MPI transport. It is intentionally a
// buffer rather than a vector: pinned elements are raw storage and callers
// should use it only with trivially copyable transport payloads.
template <class T>
class PinnedHostBuffer {
 public:
  static_assert(std::is_trivially_copyable_v<T>,
                "PinnedHostBuffer requires a trivially copyable payload");
  PinnedHostBuffer() noexcept = default;

  explicit PinnedHostBuffer(std::size_t n)
    : size_{n}, bytes_{detail::checked_buffer_bytes<T>(n)} {
    if (bytes_ != 0) ptr_ = static_cast<T*>(pinned_host_alloc(bytes_));
  }

  PinnedHostBuffer(const PinnedHostBuffer&)            = delete;
  PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

  PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : ptr_{other.ptr_}, size_{other.size_}, bytes_{other.bytes_} {
    other.ptr_   = nullptr;
    other.size_  = 0;
    other.bytes_ = 0;
  }

  PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
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

  ~PinnedHostBuffer() { release(); }

  [[nodiscard]] T* data() noexcept { return ptr_; }
  [[nodiscard]] const T* data() const noexcept { return ptr_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

  T& operator[](std::size_t index) noexcept { return ptr_[index]; }
  const T& operator[](std::size_t index) const noexcept { return ptr_[index]; }

 private:
  void release() noexcept {
    pinned_host_free(ptr_);
    ptr_   = nullptr;
    size_  = 0;
    bytes_ = 0;
  }

  T*          ptr_{nullptr};
  std::size_t size_{0};
  std::size_t bytes_{0};
};

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
      owner_device_ = current_device();
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
      owner_device_ = current_device();
      ptr_ = device_alloc_uninit(bytes_);
    }
  }

  DeviceBuffer(std::size_t n, device_ordinal_t device)
    : size_{n}, bytes_{detail::checked_buffer_bytes<T>(n)},
      owner_device_{device.value} {
    if (n != 0) {
      DeviceGuard guard{owner_device_};
      ptr_ = device_alloc(bytes_);
    }
  }

  DeviceBuffer(std::size_t n, uninitialized_t, device_ordinal_t device)
    : size_{n}, bytes_{detail::checked_buffer_bytes<T>(n)},
      owner_device_{device.value} {
    if (n != 0) {
      DeviceGuard guard{owner_device_};
      ptr_ = device_alloc_uninit(bytes_);
    }
  }

  DeviceBuffer(const DeviceBuffer&)            = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_{other.ptr_}, size_{other.size_}, bytes_{other.bytes_},
      owner_device_{other.owner_device_} {
    other.ptr_          = nullptr;
    other.size_         = 0;
    other.bytes_        = 0;
    other.owner_device_ = -1;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      ptr_                = other.ptr_;
      size_               = other.size_;
      bytes_              = other.bytes_;
      owner_device_       = other.owner_device_;
      other.ptr_          = nullptr;
      other.size_         = 0;
      other.bytes_        = 0;
      other.owner_device_ = -1;
    }
    return *this;
  }

  ~DeviceBuffer() { release(); }

  T*          device_ptr()       noexcept { return static_cast<T*>(ptr_); }
  const T*    device_ptr() const noexcept { return static_cast<const T*>(ptr_); }
  std::size_t size()       const noexcept { return size_; }
  std::size_t bytes()      const noexcept { return bytes_; }
  bool        empty()      const noexcept { return size_ == 0; }
  int         owner_device() const noexcept { return owner_device_; }

  void copy_from_host(const T* host_src, std::size_t n) {
    validate_copy(host_src, n, "copy_from_host");
    if (n == 0) return;
    DeviceGuard guard{owner_device_};
    device_memcpy_h2d(ptr_, host_src, n * sizeof(T));
  }
  void copy_from_host_async(const T* host_src, std::size_t n, stream_t stream) {
    validate_copy(host_src, n, "copy_from_host_async");
    if (n == 0) return;
    DeviceGuard guard{owner_device_};
    device_memcpy_h2d_async(ptr_, host_src, n * sizeof(T), stream);
  }
  void copy_to_host(T* host_dst, std::size_t n) const {
    validate_copy(host_dst, n, "copy_to_host");
    if (n == 0) return;
    DeviceGuard guard{owner_device_};
    device_memcpy_d2h(host_dst, ptr_, n * sizeof(T));
  }
  void copy_to_host_async(T* host_dst, std::size_t n, stream_t stream) const {
    validate_copy(host_dst, n, "copy_to_host_async");
    if (n == 0) return;
    DeviceGuard guard{owner_device_};
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
      device_free_on(ptr_, owner_device_);  // noexcept; swallows free errors
      ptr_ = nullptr;
    }
    size_         = 0;
    bytes_        = 0;
    owner_device_ = -1;
  }

  void*       ptr_{nullptr};
  std::size_t size_{0};
  std::size_t bytes_{0};
  int         owner_device_{-1};
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
