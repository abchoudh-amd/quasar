#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/distributed/mhd_halo.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <vector>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::distributed::ConstMhdRegisterComponents;
using quasar::distributed::Direction;
using quasar::distributed::MhdHaloLayouts;
using quasar::distributed::MhdHaloStagger;
using quasar::distributed::MhdRegisterComponents;
using quasar::distributed::mhd_register_component_count;
using quasar::mhd::MhdDeviceHaloComponents;
using quasar::mhd::MhdDeviceHaloConstComponents;
using quasar::mhd::MhdDeviceHaloDirection;
using quasar::mhd::MhdDeviceHaloLayout;
using quasar::mhd::MhdDeviceHaloValueKind;
using quasar::numerics::ScaledValue;

struct HostRegister {
  explicit HostRegister(const Grid2D& grid, Real initial = Real{0}) {
    for (auto& component : storage) {
      component.assign(grid.storage_size(), initial);
    }
  }

  ConstMhdRegisterComponents const_views() const {
    ConstMhdRegisterComponents result;
    for (std::size_t component = 0;
         component < mhd_register_component_count; ++component) {
      result[component] = storage[component];
    }
    return result;
  }

  MhdRegisterComponents views() {
    MhdRegisterComponents result;
    for (std::size_t component = 0;
         component < mhd_register_component_count; ++component) {
      result[component] = storage[component];
    }
    return result;
  }

  std::array<std::vector<Real>, mhd_register_component_count> storage{};
};

struct DeviceRegister {
  explicit DeviceRegister(const Grid2D& grid) {
    for (auto& component : storage) {
      component = DeviceBuffer<Real>{grid.storage_size()};
    }
  }

  std::array<DeviceBuffer<Real>, mhd_register_component_count> storage{};
};

bool require_device() {
  if (!quasar::backend::has_hip_runtime() ||
      quasar::backend::device_count() == 0) {
    return false;
  }
  quasar::backend::set_device(0);
  return true;
}

MhdDeviceHaloDirection device_direction(Direction direction) {
  switch (direction) {
    case Direction::x_low: return MhdDeviceHaloDirection::x_low;
    case Direction::x_high: return MhdDeviceHaloDirection::x_high;
    case Direction::y_low: return MhdDeviceHaloDirection::y_low;
    case Direction::y_high: return MhdDeviceHaloDirection::y_high;
  }
  return MhdDeviceHaloDirection::x_low;
}

MhdDeviceHaloLayout device_layout(MhdHaloStagger layout) {
  switch (layout) {
    case MhdHaloStagger::cell: return MhdDeviceHaloLayout::cell;
    case MhdHaloStagger::cell_extended_y:
      return MhdDeviceHaloLayout::cell_extended_y;
    case MhdHaloStagger::x_face: return MhdDeviceHaloLayout::x_face;
    case MhdHaloStagger::x_face_extended_y:
      return MhdDeviceHaloLayout::x_face_extended_y;
    case MhdHaloStagger::y_face: return MhdDeviceHaloLayout::y_face;
    case MhdHaloStagger::node: return MhdDeviceHaloLayout::node;
  }
  return MhdDeviceHaloLayout::cell;
}

Real encoded(std::size_t component, std::size_t index) {
  return static_cast<Real>(100000 * component + index + 1);
}

void upload_register(const HostRegister& host, DeviceRegister& device) {
  for (std::size_t component = 0;
       component < mhd_register_component_count; ++component) {
    device.storage[component].copy_from_host(
        host.storage[component].data(), host.storage[component].size());
  }
}

HostRegister download_register(const Grid2D& grid,
                               const DeviceRegister& device) {
  HostRegister result{grid};
  for (std::size_t component = 0;
       component < mhd_register_component_count; ++component) {
    device.storage[component].copy_to_host(
        result.storage[component].data(), result.storage[component].size());
  }
  return result;
}

MhdDeviceHaloConstComponents real_sources(const DeviceRegister& device,
                                          std::size_t count = 8) {
  MhdDeviceHaloConstComponents result{};
  for (std::size_t component = 0; component < count; ++component) {
    result.component[component] = {
        device.storage[component].device_ptr(),
        MhdDeviceHaloValueKind::real};
  }
  return result;
}

MhdDeviceHaloComponents real_destinations(
    DeviceRegister& device, const MhdHaloLayouts& layouts) {
  MhdDeviceHaloComponents result{};
  for (std::size_t component = 0;
       component < mhd_register_component_count; ++component) {
    result.component[component] = {
        device.storage[component].device_ptr(),
        MhdDeviceHaloValueKind::real, device_layout(layouts[component])};
  }
  return result;
}

constexpr std::array<Direction, 4> directions{
    Direction::x_low, Direction::x_high,
    Direction::y_low, Direction::y_high};

}  // namespace

TEST(MhdDeviceHalo, PackMatchesHostWireLayoutForEveryDirection) {
  if (!require_device()) GTEST_SKIP() << "HIP device is unavailable";
  const Grid2D grid{5, 4, 5.0, 4.0, 0.0, 0.0, 2};
  HostRegister host{grid};
  for (std::size_t component = 0;
       component < mhd_register_component_count; ++component) {
    for (std::size_t index = 0; index < grid.storage_size(); ++index) {
      host.storage[component][index] = encoded(component, index);
    }
  }
  // An absent lane must still occupy its fixed wire block as exact zeroes.
  std::fill(host.storage[7].begin(), host.storage[7].end(), Real{0});
  DeviceRegister device{grid};
  upload_register(host, device);
  const auto components = real_sources(device, 7);

  for (const Direction direction : directions) {
    SCOPED_TRACE(static_cast<int>(direction));
    const std::size_t count =
        quasar::distributed::mhd_register_halo_scalar_count(grid, direction);
    std::vector<Real> expected(count);
    quasar::distributed::pack_mhd_register_halo(
        grid, direction, host.const_views(), expected);
    DeviceBuffer<Real> payload{count};
    quasar::mhd::launch_mhd_device_halo_pack(
        grid, device_direction(direction), components, payload, nullptr);
    quasar::backend::device_synchronize(nullptr);
    std::vector<Real> actual(count);
    payload.copy_to_host(actual.data(), actual.size());
    EXPECT_EQ(actual, expected);
  }
}

TEST(MhdDeviceHalo, UnpackMatchesAllHostStaggersAndOwnershipModes) {
  if (!require_device()) GTEST_SKIP() << "HIP device is unavailable";
  const Grid2D grid{5, 4, 5.0, 4.0, 0.0, 0.0, 2};
  MhdHaloLayouts layouts{
      MhdHaloStagger::cell,
      MhdHaloStagger::cell_extended_y,
      MhdHaloStagger::x_face,
      MhdHaloStagger::x_face_extended_y,
      MhdHaloStagger::y_face,
      MhdHaloStagger::node,
      MhdHaloStagger::cell,
      MhdHaloStagger::x_face};

  for (const Direction direction : directions) {
    const std::size_t count =
        quasar::distributed::mhd_register_halo_scalar_count(grid, direction);
    const std::size_t block = count / mhd_register_component_count;
    std::vector<Real> payload(count);
    for (std::size_t component = 0;
         component < mhd_register_component_count; ++component) {
      for (std::size_t position = 0; position < block; ++position) {
        payload[component * block + position] = encoded(component, position);
      }
    }
    DeviceBuffer<Real> device_payload{count};
    device_payload.copy_from_host(payload.data(), payload.size());

    for (const bool receive_shared_face : {false, true}) {
      SCOPED_TRACE(static_cast<int>(direction));
      SCOPED_TRACE(receive_shared_face);
      HostRegister expected{grid, Real{-1}};
      quasar::distributed::unpack_mhd_register_halo(
          grid, direction, payload, expected.views(), receive_shared_face,
          layouts);
      HostRegister initial{grid, Real{-1}};
      DeviceRegister device{grid};
      upload_register(initial, device);
      const auto components = real_destinations(device, layouts);
      quasar::mhd::launch_mhd_device_halo_unpack(
          grid, device_direction(direction), device_payload, components,
          receive_shared_face, nullptr);
      quasar::backend::device_synchronize(nullptr);
      const HostRegister actual = download_register(grid, device);
      EXPECT_EQ(actual.storage, expected.storage);
    }
  }
}

TEST(MhdDeviceHalo, ConvertsIntegerAndScaledValueLanesWithoutHostStaging) {
  if (!require_device()) GTEST_SKIP() << "HIP device is unavailable";
  const Grid2D grid{4, 5, 4.0, 5.0, 0.0, 0.0, 2};
  const std::size_t size = grid.storage_size();
  std::vector<int> integers(size);
  std::vector<ScaledValue> scaled(size);
  HostRegister wire_source{grid};
  for (std::size_t index = 0; index < size; ++index) {
    integers[index] = static_cast<int>(index % 2);
    scaled[index] = ScaledValue{
        Real{0.5} + static_cast<Real>(index) / Real{1024},
        static_cast<int>(index % 31) - 15};
    wire_source.storage[0][index] = static_cast<Real>(integers[index]);
    wire_source.storage[1][index] = scaled[index].mantissa;
    wire_source.storage[2][index] = static_cast<Real>(scaled[index].exponent);
  }
  DeviceBuffer<int> device_integers{size};
  DeviceBuffer<ScaledValue> device_scaled{size};
  device_integers.copy_from_host(integers.data(), integers.size());
  device_scaled.copy_from_host(scaled.data(), scaled.size());
  MhdDeviceHaloConstComponents sources{};
  sources.component[0] = {
      device_integers.device_ptr(), MhdDeviceHaloValueKind::int32};
  sources.component[1] = {
      device_scaled.device_ptr(), MhdDeviceHaloValueKind::scaled_mantissa};
  sources.component[2] = {
      device_scaled.device_ptr(), MhdDeviceHaloValueKind::scaled_exponent};

  const Direction direction = Direction::y_high;
  const std::size_t count =
      quasar::distributed::mhd_register_halo_scalar_count(grid, direction);
  std::vector<Real> expected_payload(count);
  quasar::distributed::pack_mhd_register_halo(
      grid, direction, wire_source.const_views(), expected_payload);
  DeviceBuffer<Real> payload{count};
  quasar::mhd::launch_mhd_device_halo_pack(
      grid, device_direction(direction), sources, payload, nullptr);
  quasar::backend::device_synchronize(nullptr);
  std::vector<Real> actual_payload(count);
  payload.copy_to_host(actual_payload.data(), actual_payload.size());
  EXPECT_EQ(actual_payload, expected_payload);

  HostRegister expected_wire{grid};
  std::fill(expected_wire.storage[0].begin(), expected_wire.storage[0].end(),
            Real{-7});
  std::fill(expected_wire.storage[1].begin(), expected_wire.storage[1].end(),
            Real{-9});
  std::fill(expected_wire.storage[2].begin(), expected_wire.storage[2].end(),
            Real{-99});
  MhdHaloLayouts layouts{};
  layouts.fill(MhdHaloStagger::cell);
  quasar::distributed::unpack_mhd_register_halo(
      grid, direction, expected_payload, expected_wire.views(), false, layouts);

  std::vector<int> integer_destination(size, -7);
  std::vector<ScaledValue> scaled_destination(
      size, ScaledValue{Real{-9}, -99});
  DeviceBuffer<int> device_integer_destination{size};
  DeviceBuffer<ScaledValue> device_scaled_destination{size};
  device_integer_destination.copy_from_host(
      integer_destination.data(), integer_destination.size());
  device_scaled_destination.copy_from_host(
      scaled_destination.data(), scaled_destination.size());
  MhdDeviceHaloComponents destinations{};
  destinations.component[0] = {
      device_integer_destination.device_ptr(), MhdDeviceHaloValueKind::int32,
      MhdDeviceHaloLayout::cell};
  destinations.component[1] = {
      device_scaled_destination.device_ptr(),
      MhdDeviceHaloValueKind::scaled_mantissa, MhdDeviceHaloLayout::cell};
  destinations.component[2] = {
      device_scaled_destination.device_ptr(),
      MhdDeviceHaloValueKind::scaled_exponent, MhdDeviceHaloLayout::cell};
  quasar::mhd::launch_mhd_device_halo_unpack(
      grid, device_direction(direction), payload, destinations, false, nullptr);
  quasar::backend::device_synchronize(nullptr);
  device_integer_destination.copy_to_host(
      integer_destination.data(), integer_destination.size());
  device_scaled_destination.copy_to_host(
      scaled_destination.data(), scaled_destination.size());

  for (std::size_t index = 0; index < size; ++index) {
    EXPECT_EQ(integer_destination[index],
              static_cast<int>(expected_wire.storage[0][index]));
    EXPECT_EQ(scaled_destination[index].mantissa,
              expected_wire.storage[1][index]);
    EXPECT_EQ(scaled_destination[index].exponent,
              static_cast<int>(expected_wire.storage[2][index]));
  }
}
