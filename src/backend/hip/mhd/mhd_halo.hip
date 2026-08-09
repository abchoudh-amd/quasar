// Device-resident packing and unpacking for the fixed-width distributed MHD
// halo wire format. The index algebra intentionally mirrors mhd_halo.cpp; the
// host implementation remains the compact reference used by CPU-only tests.

#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <limits>
#include <stdexcept>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::mhd::MhdDeviceHaloComponent;
using quasar::mhd::MhdDeviceHaloComponents;
using quasar::mhd::MhdDeviceHaloConstComponent;
using quasar::mhd::MhdDeviceHaloConstComponents;
using quasar::mhd::MhdDeviceHaloDirection;
using quasar::mhd::MhdDeviceHaloLayout;
using quasar::mhd::MhdDeviceHaloValueKind;
using quasar::numerics::ScaledValue;

inline constexpr unsigned kBlock = 256;

constexpr bool valid_direction(MhdDeviceHaloDirection direction) noexcept {
  const int value = static_cast<int>(direction);
  return value >= static_cast<int>(MhdDeviceHaloDirection::x_low) &&
         value <= static_cast<int>(MhdDeviceHaloDirection::y_high);
}

__host__ __device__ constexpr bool is_x(
    MhdDeviceHaloDirection direction) noexcept {
  return direction == MhdDeviceHaloDirection::x_low ||
         direction == MhdDeviceHaloDirection::x_high;
}

__host__ __device__ constexpr bool is_low(
    MhdDeviceHaloDirection direction) noexcept {
  return direction == MhdDeviceHaloDirection::x_low ||
         direction == MhdDeviceHaloDirection::y_low;
}

constexpr bool valid_kind(MhdDeviceHaloValueKind kind) noexcept {
  const int value = static_cast<int>(kind);
  return value >= static_cast<int>(MhdDeviceHaloValueKind::absent) &&
         value <= static_cast<int>(MhdDeviceHaloValueKind::scaled_exponent);
}

constexpr bool valid_layout(MhdDeviceHaloLayout layout) noexcept {
  const int value = static_cast<int>(layout);
  return value >= static_cast<int>(MhdDeviceHaloLayout::cell) &&
         value <= static_cast<int>(MhdDeviceHaloLayout::node);
}

__host__ __device__ constexpr bool is_x_face(
    MhdDeviceHaloLayout layout) noexcept {
  return layout == MhdDeviceHaloLayout::x_face ||
         layout == MhdDeviceHaloLayout::x_face_extended_y ||
         layout == MhdDeviceHaloLayout::node;
}

__host__ __device__ constexpr bool is_y_face(
    MhdDeviceHaloLayout layout) noexcept {
  return layout == MhdDeviceHaloLayout::y_face ||
         layout == MhdDeviceHaloLayout::node;
}

__host__ __device__ constexpr bool includes_high_y(
    MhdDeviceHaloLayout layout) noexcept {
  return layout == MhdDeviceHaloLayout::cell_extended_y ||
         layout == MhdDeviceHaloLayout::x_face_extended_y ||
         layout == MhdDeviceHaloLayout::y_face ||
         layout == MhdDeviceHaloLayout::node;
}

std::size_t checked_product(std::size_t left, std::size_t right) {
  if (left != 0 &&
      right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error{"MHD device halo payload size overflows"};
  }
  return left * right;
}

std::size_t halo_scalar_count(const Grid2D& grid,
                              MhdDeviceHaloDirection direction) {
  if (!valid_direction(direction)) {
    throw std::invalid_argument{"MHD device halo direction is invalid"};
  }
  if (grid.nghost <= 0 || grid.nx < grid.nghost ||
      grid.ny < grid.nghost) {
    throw std::invalid_argument{
        "MHD device halo exchange requires non-empty tiles at least as wide as the halo"};
  }
  const std::size_t depth = static_cast<std::size_t>(grid.nghost) + 1;
  const std::size_t transverse = is_x(direction)
      ? static_cast<std::size_t>(grid.ny) + 1
      : static_cast<std::size_t>(grid.pitch());
  return checked_product(
      quasar::mhd::mhd_device_halo_component_count,
      checked_product(depth, transverse));
}

void validate_components(const MhdDeviceHaloConstComponents& components) {
  for (std::size_t index = 0;
       index < quasar::mhd::mhd_device_halo_component_count; ++index) {
    const auto& component = components.component[index];
    if (!valid_kind(component.kind) ||
        (component.kind != MhdDeviceHaloValueKind::absent &&
         component.values == nullptr)) {
      throw std::invalid_argument{"MHD device halo source lane is invalid"};
    }
  }
}

void validate_components(const MhdDeviceHaloComponents& components) {
  for (std::size_t index = 0;
       index < quasar::mhd::mhd_device_halo_component_count; ++index) {
    const auto& component = components.component[index];
    if (!valid_kind(component.kind) || !valid_layout(component.layout) ||
        (component.kind != MhdDeviceHaloValueKind::absent &&
         component.values == nullptr)) {
      throw std::invalid_argument{
          "MHD device halo destination lane is invalid"};
    }
  }
}

unsigned launch_blocks(std::size_t count) {
  const std::size_t blocks = (count + kBlock - 1) / kBlock;
  if (blocks > std::numeric_limits<unsigned>::max()) {
    throw std::overflow_error{"MHD device halo launch grid overflows"};
  }
  return static_cast<unsigned>(blocks);
}

__device__ Real read_component(MhdDeviceHaloConstComponent component,
                               std::size_t index) {
  switch (component.kind) {
    case MhdDeviceHaloValueKind::real:
      return static_cast<const Real*>(component.values)[index];
    case MhdDeviceHaloValueKind::int32:
      return static_cast<Real>(
          static_cast<const int*>(component.values)[index]);
    case MhdDeviceHaloValueKind::scaled_mantissa:
      return static_cast<const ScaledValue*>(component.values)[index].mantissa;
    case MhdDeviceHaloValueKind::scaled_exponent:
      return static_cast<Real>(
          static_cast<const ScaledValue*>(component.values)[index].exponent);
    case MhdDeviceHaloValueKind::absent:
      return Real{0};
  }
  return Real{0};
}

__device__ void write_component(MhdDeviceHaloComponent component,
                                std::size_t index, Real value) {
  switch (component.kind) {
    case MhdDeviceHaloValueKind::real:
      static_cast<Real*>(component.values)[index] = value;
      return;
    case MhdDeviceHaloValueKind::int32:
      static_cast<int*>(component.values)[index] = static_cast<int>(value);
      return;
    case MhdDeviceHaloValueKind::scaled_mantissa:
      static_cast<ScaledValue*>(component.values)[index].mantissa = value;
      return;
    case MhdDeviceHaloValueKind::scaled_exponent:
      static_cast<ScaledValue*>(component.values)[index].exponent =
          static_cast<int>(value);
      return;
    case MhdDeviceHaloValueKind::absent:
      return;
  }
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 pack_halo_kernel(
    Grid2D grid, MhdDeviceHaloDirection direction,
    MhdDeviceHaloConstComponents components, std::size_t block_size,
    Real* payload, std::size_t payload_size) {
  const std::size_t output =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output >= payload_size) return;

  const std::size_t component = output / block_size;
  const std::size_t position = output - component * block_size;
  std::size_t source = 0;
  if (is_x(direction)) {
    const int depth = grid.nghost + 1;
    const int j = static_cast<int>(position / static_cast<std::size_t>(depth));
    const int offset =
        static_cast<int>(position % static_cast<std::size_t>(depth));
    const int first_i = is_low(direction) ? 0 : grid.nx - grid.nghost;
    source = grid.index(first_i + offset, j);
  } else {
    const int pitch = grid.pitch();
    const int offset =
        static_cast<int>(position / static_cast<std::size_t>(pitch));
    const int i = -grid.nghost +
        static_cast<int>(position % static_cast<std::size_t>(pitch));
    const int first_j = is_low(direction) ? 0 : grid.ny - grid.nghost;
    source = grid.index(i, first_j + offset);
  }
  payload[output] = read_component(components.component[component], source);
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 unpack_halo_kernel(
    Grid2D grid, MhdDeviceHaloDirection direction, const Real* payload,
    std::size_t payload_size, MhdDeviceHaloComponents components,
    std::size_t block_size, int receive_shared_face) {
  const std::size_t input =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (input >= payload_size) return;

  const std::size_t component_index = input / block_size;
  const std::size_t position = input - component_index * block_size;
  const MhdDeviceHaloComponent component =
      components.component[component_index];
  if (component.kind == MhdDeviceHaloValueKind::absent) return;

  std::size_t destination = 0;
  bool target = false;
  if (is_x(direction)) {
    const int depth = grid.nghost + 1;
    const int j = static_cast<int>(position / static_cast<std::size_t>(depth));
    const int offset =
        static_cast<int>(position % static_cast<std::size_t>(depth));
    const int i = is_low(direction)
        ? -grid.nghost + offset
        : grid.nx + offset;
    const bool transverse_owned = includes_high_y(component.layout)
        ? j <= grid.ny
        : j < grid.ny;
    const bool normal_target = !is_x_face(component.layout)
        ? offset < grid.nghost
        : is_low(direction)
            ? (offset < grid.nghost ||
               (offset == grid.nghost && receive_shared_face != 0))
            : ((offset > 0 && offset < grid.nghost) ||
               (offset == 0 && receive_shared_face != 0));
    target = transverse_owned && normal_target;
    if (target) destination = grid.index(i, j);
  } else {
    const int pitch = grid.pitch();
    const int offset =
        static_cast<int>(position / static_cast<std::size_t>(pitch));
    const int i = -grid.nghost +
        static_cast<int>(position % static_cast<std::size_t>(pitch));
    const int j = is_low(direction)
        ? -grid.nghost + offset
        : grid.ny + offset;
    target = !is_y_face(component.layout)
        ? offset < grid.nghost
        : is_low(direction)
            ? (offset < grid.nghost ||
               (offset == grid.nghost && receive_shared_face != 0))
            : ((offset > 0 && offset < grid.nghost) ||
               (offset == 0 && receive_shared_face != 0));
    if (target) destination = grid.index(i, j);
  }
  if (target) write_component(component, destination, payload[input]);
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_device_halo_pack(
    Grid2D grid, MhdDeviceHaloDirection direction,
    const MhdDeviceHaloConstComponents& components,
    backend::DeviceBuffer<Real>& payload, stream_t stream) {
  const std::size_t count = halo_scalar_count(grid, direction);
  if (payload.size() != count) {
    throw std::invalid_argument{
        "MHD device halo payload does not match the tile and direction"};
  }
  validate_components(components);
  const std::size_t block_size =
      count / mhd_device_halo_component_count;
  hipLaunchKernelGGL(pack_halo_kernel, dim3(launch_blocks(count)),
                     dim3(kBlock), 0, static_cast<hipStream_t>(stream), grid,
                     direction, components, block_size, payload.device_ptr(),
                     count);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_device_halo_unpack(
    Grid2D grid, MhdDeviceHaloDirection direction,
    const backend::DeviceBuffer<Real>& payload,
    const MhdDeviceHaloComponents& components, bool receive_shared_face,
    stream_t stream) {
  const std::size_t count = halo_scalar_count(grid, direction);
  if (payload.size() != count) {
    throw std::invalid_argument{
        "MHD device halo payload does not match the tile and direction"};
  }
  validate_components(components);
  const std::size_t block_size =
      count / mhd_device_halo_component_count;
  hipLaunchKernelGGL(unpack_halo_kernel, dim3(launch_blocks(count)),
                     dim3(kBlock), 0, static_cast<hipStream_t>(stream), grid,
                     direction, payload.device_ptr(), count, components,
                     block_size, receive_shared_face ? 1 : 0);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
