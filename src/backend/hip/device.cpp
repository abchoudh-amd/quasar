#include "quasar/backend/device.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <iomanip>
#include <sstream>

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

int current_device() {
  int device = -1;
  QUASAR_HIP_CHECK(::hipGetDevice(&device));
  return device;
}

void set_device(int device) {
  QUASAR_HIP_CHECK(::hipSetDevice(device));
}

bool try_set_device(int device) noexcept {
  return ::hipSetDevice(device) == ::hipSuccess;
}

DeviceGuard::DeviceGuard(int device) : previous_device_{current_device()} {
  if (device != previous_device_) {
    set_device(device);
    restore_ = true;
  }
}

DeviceGuard::~DeviceGuard() noexcept {
  if (restore_) (void)try_set_device(previous_device_);
}

std::string device_uuid(int device) {
  ::hipUUID uuid{};
  QUASAR_HIP_CHECK(::hipDeviceGetUuid(&uuid, device));
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (const char byte : uuid.bytes) {
    result << std::setw(2)
           << static_cast<unsigned int>(static_cast<unsigned char>(byte));
  }
  return result.str();
}

std::string device_pci_bus_id(int device) {
  char value[32]{};
  QUASAR_HIP_CHECK(::hipDeviceGetPCIBusId(value, sizeof(value), device));
  return value;
}

bool device_can_access_peer(int device, int peer_device) {
  int can_access = 0;
  QUASAR_HIP_CHECK(
      ::hipDeviceCanAccessPeer(&can_access, device, peer_device));
  return can_access != 0;
}

void device_enable_peer_access(int peer_device) {
  const ::hipError_t error = ::hipDeviceEnablePeerAccess(peer_device, 0);
  if (error == ::hipErrorPeerAccessAlreadyEnabled) {
    // Clear HIP's per-thread last-error slot; already-enabled is the desired
    // idempotent state for persistent worker setup.
    (void)::hipGetLastError();
    return;
  }
  if (error != ::hipSuccess) {
    throw DeviceError{static_cast<int>(error), ::hipGetErrorString(error),
                      __FILE__, __LINE__};
  }
}

stream_t stream_create() {
  ::hipStream_t stream{};
  QUASAR_HIP_CHECK(
      ::hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  return static_cast<stream_t>(stream);
}

void stream_destroy(stream_t stream) noexcept {
  if (stream != nullptr) {
    (void)::hipStreamDestroy(static_cast<::hipStream_t>(stream));
  }
}

event_t event_create() {
  ::hipEvent_t event{};
  QUASAR_HIP_CHECK(
      ::hipEventCreateWithFlags(&event, hipEventDisableTiming));
  return static_cast<event_t>(event);
}

void event_destroy(event_t event) noexcept {
  if (event != nullptr) {
    (void)::hipEventDestroy(static_cast<::hipEvent_t>(event));
  }
}

void event_record(event_t event, stream_t stream) {
  QUASAR_HIP_CHECK(::hipEventRecord(static_cast<::hipEvent_t>(event),
                                    static_cast<::hipStream_t>(stream)));
}

void event_synchronize(event_t event) {
  QUASAR_HIP_CHECK(
      ::hipEventSynchronize(static_cast<::hipEvent_t>(event)));
}

void stream_wait_event(stream_t stream, event_t event) {
  QUASAR_HIP_CHECK(::hipStreamWaitEvent(
      static_cast<::hipStream_t>(stream),
      static_cast<::hipEvent_t>(event), 0));
}

}  // namespace quasar::backend
