// Device-side execution of the immutable fixed PIC halo plan. Each thread
// moves one scalar, preserving the host plan's payload order exactly.

#include "quasar/physics/pic/kernels.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::pic::PicDeviceHaloComponents;
using quasar::pic::PicDeviceHaloConstComponents;
using quasar::pic::PicDeviceHaloEntry;
using quasar::pic::PicDeviceHaloUpdate;
using quasar::pic::PicDeviceTileExtent;

inline constexpr unsigned kBlock = 256;
inline constexpr std::uint32_t kAllComponents =
    (std::uint32_t{1} << quasar::pic::pic_device_halo_component_count) - 1;

unsigned launch_blocks(std::size_t count) {
  const std::size_t blocks = (count + kBlock - 1) / kBlock;
  if (blocks > std::numeric_limits<unsigned>::max()) {
    throw std::overflow_error{"PIC device halo launch grid overflows"};
  }
  return static_cast<unsigned>(blocks);
}

void validate_common(const Grid2D& grid, std::size_t payload_count,
                     std::size_t payload_capacity,
                     std::size_t entry_count,
                     std::uint32_t component_mask) {
  if (grid.nx <= 0 || grid.ny <= 0 || grid.nghost < 0) {
    throw std::invalid_argument{"PIC device halo grid is invalid"};
  }
  if (payload_count > payload_capacity || entry_count > payload_count) {
    throw std::invalid_argument{
        "PIC device halo payload is smaller than its exchange plan"};
  }
  if ((component_mask & ~kAllComponents) != 0) {
    throw std::invalid_argument{"PIC device halo component mask is invalid"};
  }
}

void validate_components(const PicDeviceHaloConstComponents& components,
                         std::uint32_t component_mask) {
  for (std::size_t component = 0;
       component < quasar::pic::pic_device_halo_component_count;
       ++component) {
    if ((component_mask & (std::uint32_t{1} << component)) != 0 &&
        components.component[component] == nullptr) {
      throw std::invalid_argument{
          "PIC device halo source component is null"};
    }
  }
}

void validate_components(const PicDeviceHaloComponents& components,
                         std::uint32_t component_mask) {
  for (std::size_t component = 0;
       component < quasar::pic::pic_device_halo_component_count;
       ++component) {
    if ((component_mask & (std::uint32_t{1} << component)) != 0 &&
        components.component[component] == nullptr) {
      throw std::invalid_argument{
          "PIC device halo destination component is null"};
    }
  }
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 pack_halo_kernel(
    Grid2D grid, PicDeviceHaloConstComponents components,
    const PicDeviceHaloEntry* entries, std::size_t entry_count,
    std::uint32_t component_mask, Real* payload) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= entry_count) return;
  const PicDeviceHaloEntry entry = entries[index];
  if (entry.component >= quasar::pic::pic_device_halo_component_count ||
      (component_mask & (std::uint32_t{1} << entry.component)) == 0) {
    return;
  }
  if (entry.source_x < -grid.nghost ||
      entry.source_x >= grid.nx + grid.nghost ||
      entry.source_y < -grid.nghost ||
      entry.source_y >= grid.ny + grid.nghost) {
    return;
  }
  payload[index] = components.component[entry.component]
      [grid.index(entry.source_x, entry.source_y)];
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 unpack_halo_kernel(
    Grid2D grid, const Real* payload, const PicDeviceHaloEntry* entries,
    std::size_t entry_count, PicDeviceHaloComponents components,
    std::uint32_t component_mask, PicDeviceHaloUpdate update) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= entry_count) return;
  const PicDeviceHaloEntry entry = entries[index];
  if (entry.component >= quasar::pic::pic_device_halo_component_count ||
      (component_mask & (std::uint32_t{1} << entry.component)) == 0) {
    return;
  }
  if (entry.destination_x < -grid.nghost ||
      entry.destination_x >= grid.nx + grid.nghost ||
      entry.destination_y < -grid.nghost ||
      entry.destination_y >= grid.ny + grid.nghost) {
    return;
  }
  Real* destination = components.component[entry.component] +
      grid.index(entry.destination_x, entry.destination_y);
  if (update == PicDeviceHaloUpdate::add) {
    atomicAdd(destination, payload[index]);
  } else {
    *destination = payload[index];
  }
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 accumulate_halo_kernel(
    Grid2D grid, PicDeviceHaloConstComponents sources,
    const PicDeviceHaloEntry* entries, std::size_t entry_count,
    PicDeviceHaloComponents destinations, std::uint32_t component_mask) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= entry_count) return;
  const PicDeviceHaloEntry entry = entries[index];
  if (entry.component >= quasar::pic::pic_device_halo_component_count ||
      (component_mask & (std::uint32_t{1} << entry.component)) == 0 ||
      entry.source_x < -grid.nghost ||
      entry.source_x >= grid.nx + grid.nghost ||
      entry.source_y < -grid.nghost ||
      entry.source_y >= grid.ny + grid.nghost ||
      entry.destination_x < -grid.nghost ||
      entry.destination_x >= grid.nx + grid.nghost ||
      entry.destination_y < -grid.nghost ||
      entry.destination_y >= grid.ny + grid.nghost) {
    return;
  }
  atomicAdd(destinations.component[entry.component] +
                grid.index(entry.destination_x, entry.destination_y),
            sources.component[entry.component]
                [grid.index(entry.source_x, entry.source_y)]);
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 copy_components_kernel(
    Grid2D grid, PicDeviceHaloConstComponents sources,
    PicDeviceHaloComponents destinations, std::uint32_t component_mask) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= grid.storage_size()) return;
  for (std::uint32_t component = 0;
       component < quasar::pic::pic_device_halo_component_count;
       ++component) {
    if ((component_mask & (std::uint32_t{1} << component)) != 0) {
      destinations.component[component][index] =
          sources.component[component][index];
    }
  }
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 add_background_kernel(
    Grid2D grid, Real* charge, Real density) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= grid.interior_size()) return;
  const int i = static_cast<int>(index % static_cast<std::size_t>(grid.nx));
  const int j = static_cast<int>(index / static_cast<std::size_t>(grid.nx));
  charge[grid.index(i, j)] += density;
}

__device__ inline bool owns_x_face(const Grid2D& grid,
                                   const PicDeviceTileExtent& tile,
                                   int i) {
  const int begin = tile.tile_x == 0 ? 0 : 1;
  // A shared face belongs to the tile on its low side.  Every internal tile
  // therefore owns its local high face, while the tile on the high side skips
  // local face zero.  The only high face without distinct ownership is the
  // periodic global-high duplicate of global face zero.
  const bool owns_high = tile.tile_x + 1 < tile.tiles_x ||
      tile.periodic_x == 0;
  const int end = grid.nx + (owns_high ? 1 : 0);
  return i >= begin && i < end;
}

__device__ inline bool owns_y_face(const Grid2D& grid,
                                   const PicDeviceTileExtent& tile,
                                   int j) {
  const int begin = tile.tile_y == 0 ? 0 : 1;
  const bool owns_high = tile.tile_y + 1 < tile.tiles_y ||
      tile.periodic_y == 0;
  const int end = grid.ny + (owns_high ? 1 : 0);
  return j >= begin && j < end;
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 distributed_filter_axis_kernel(
    Grid2D grid, PicDeviceTileExtent tile, const Real* input, Real* output,
    int axis, Real neighbor_weight, Real center_weight, bool cylindrical) {
  const std::size_t storage_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (storage_index >= grid.storage_size()) return;
  output[storage_index] = input[storage_index];

  const int padded_x = static_cast<int>(
      storage_index % static_cast<std::size_t>(grid.pitch()));
  const int padded_y = static_cast<int>(
      storage_index / static_cast<std::size_t>(grid.pitch()));
  const int i = padded_x - grid.nghost;
  const int j = padded_y - grid.nghost;
  const bool owned = cylindrical
      ? owns_x_face(grid, tile, i) && j >= 0 && j < grid.ny
      : i >= 0 && i < grid.nx && j >= 0 && j < grid.ny;
  if (!owned) return;

  const std::uint64_t global_x =
      tile.global_x_begin + static_cast<std::uint64_t>(i);
  const std::uint64_t global_y =
      tile.global_y_begin + static_cast<std::uint64_t>(j);
  const std::uint64_t active_x =
      tile.global_nx + (cylindrical ? std::uint64_t{1} : std::uint64_t{0});
  const std::uint64_t active_y = tile.global_ny;
  const std::size_t center = grid.index(i, j);
  if (cylindrical && grid.origin_x == Real{0} && global_x == 0) {
    output[center] = Real{0};
    return;
  }
  const bool low = axis == 0 ? global_x == 0 : global_y == 0;
  const bool high = axis == 0
      ? global_x + 1 == active_x
      : global_y + 1 == active_y;
  const bool periodic = axis == 0 ? tile.periodic_x != 0
                                  : tile.periodic_y != 0;
  const int di = axis == 0 ? 1 : 0;
  const int dj = axis == 0 ? 0 : 1;
  const Real previous = !periodic && low
      ? input[center] : input[grid.index(i - di, j - dj)];
  const Real next = !periodic && high
      ? input[center] : input[grid.index(i + di, j + dj)];
  output[center] = neighbor_weight * previous +
      center_weight * input[center] + neighbor_weight * next;
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 distributed_axis_values_kernel(
    Grid2D grid, const Real* radial_current, Real* axis_values,
    bool owns_axis) {
  const int j = static_cast<int>(
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x);
  if (!owns_axis || j >= grid.ny) return;
  axis_values[grid.index(0, j)] = radial_current[grid.index(1, j)];
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 distributed_order4_kernel(
    Grid2D grid, PicDeviceTileExtent tile, const Real* radial,
    const Real* axial, const Real* rhs_radial, const Real* rhs_axial,
    const Real* axis_values, Real* output_radial, Real* output_axial,
    int x_lo_mode, int x_hi_mode, int y_lo_mode, int y_hi_mode,
    bool cylindrical, bool on_axis) {
  const std::size_t storage_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (storage_index >= grid.storage_size()) return;
  output_radial[storage_index] = radial[storage_index];
  output_axial[storage_index] = axial[storage_index];

  const int padded_x = static_cast<int>(
      storage_index % static_cast<std::size_t>(grid.pitch()));
  const int padded_y = static_cast<int>(
      storage_index / static_cast<std::size_t>(grid.pitch()));
  const int i = padded_x - grid.nghost;
  const int j = padded_y - grid.nghost;

  if (owns_x_face(grid, tile, i) && j >= 0 && j < grid.ny) {
    const std::uint64_t global_x =
        tile.global_x_begin + static_cast<std::uint64_t>(i);
    const std::size_t center = grid.index(i, j);
    if (!cylindrical) {
      if (tile.periodic_x == 0 && global_x == 0) {
        output_radial[center] = x_lo_mode == 2
            ? rhs_radial[center]
            : Real{12} / Real{13} * rhs_radial[center] +
                  Real{1} / Real{13} * radial[grid.index(i + 1, j)];
      } else if (tile.periodic_x == 0 && global_x == tile.global_nx) {
        output_radial[center] = x_hi_mode == 2
            ? rhs_radial[center]
            : Real{12} / Real{13} * rhs_radial[center] +
                  Real{1} / Real{13} * radial[grid.index(i - 1, j)];
      } else {
        output_radial[center] =
            Real{12} / Real{13} * rhs_radial[center] +
            Real{1} / Real{26} *
                (radial[grid.index(i - 1, j)] +
                 radial[grid.index(i + 1, j)]);
      }
    } else if (on_axis && global_x == 0) {
      output_radial[center] = Real{0};
    } else if (on_axis && global_x == 1) {
      output_radial[center] = tile.global_nx == 1
          ? (x_hi_mode == 2 ? rhs_radial[center]
                            : Real{6} / Real{7} * rhs_radial[center])
          : Real{6} / Real{7} *
                (rhs_radial[center] +
                 radial[grid.index(i + 1, j)] / Real{12});
    } else {
      const Real radius = grid.r_at_edge(i);
      if (!on_axis && global_x == 0) {
        if (x_lo_mode == 2) {
          const Real delta = grid.dx() / radius;
          output_radial[center] =
              (Real{12} * rhs_radial[center] +
               delta * radial[grid.index(i + 1, j)]) /
              (Real{12} + delta);
        } else {
          output_radial[center] =
              Real{12} / Real{13} * rhs_radial[center] +
              Real{1} / Real{13} * radial[grid.index(i + 1, j)];
        }
      } else if (global_x == tile.global_nx) {
        const Real delta = grid.dx() / radius;
        const Real inner = radial[grid.index(i - 1, j)];
        const Real axis_term = on_axis
            ? grid.dx() / radius * axis_values[grid.index(0, j)]
            : Real{0};
        output_radial[center] = x_hi_mode == 2
            ? (Real{12} * rhs_radial[center] - delta * inner - axis_term) /
                  (Real{12} - delta)
            : Real{12} / Real{13} *
                  (rhs_radial[center] + inner / Real{12} -
                   axis_term / Real{12});
      } else {
        const Real axis_term = on_axis
            ? grid.dx() / radius * axis_values[grid.index(0, j)] / Real{12}
            : Real{0};
        output_radial[center] = Real{12} / Real{13} *
            (rhs_radial[center] +
             (grid.r_at_edge(i - 1) / radius *
                  radial[grid.index(i - 1, j)] +
              grid.r_at_edge(i + 1) / radius *
                  radial[grid.index(i + 1, j)]) /
                 Real{24} -
             axis_term);
      }
    }
  }

  if (i >= 0 && i < grid.nx && owns_y_face(grid, tile, j)) {
    const std::uint64_t global_y =
        tile.global_y_begin + static_cast<std::uint64_t>(j);
    const std::size_t center = grid.index(i, j);
    if (tile.periodic_y == 0 && global_y == 0) {
      output_axial[center] = y_lo_mode == 2
          ? rhs_axial[center]
          : Real{12} / Real{13} * rhs_axial[center] +
                Real{1} / Real{13} * axial[grid.index(i, j + 1)];
    } else if (tile.periodic_y == 0 && global_y == tile.global_ny) {
      output_axial[center] = y_hi_mode == 2
          ? rhs_axial[center]
          : Real{12} / Real{13} * rhs_axial[center] +
                Real{1} / Real{13} * axial[grid.index(i, j - 1)];
    } else {
      output_axial[center] =
          Real{12} / Real{13} * rhs_axial[center] +
          Real{1} / Real{26} *
              (axial[grid.index(i, j - 1)] +
               axial[grid.index(i, j + 1)]);
    }
  }
}

}  // namespace

namespace quasar::pic {

void launch_pic_device_halo_pack(
    Grid2D grid, const PicDeviceHaloConstComponents& components,
    const backend::DeviceBuffer<PicDeviceHaloEntry>& entries,
    std::size_t payload_count, std::uint32_t component_mask,
    backend::DeviceBuffer<Real>& payload, backend::stream_t stream) {
  validate_common(grid, payload_count, payload.size(), entries.size(),
                  component_mask);
  validate_components(components, component_mask);
  if (payload_count != 0) {
    backend::device_memset_async(payload.device_ptr(), 0,
                                 payload_count * sizeof(Real), stream);
  }
  if (entries.empty()) return;
  hipLaunchKernelGGL(pack_halo_kernel, dim3(launch_blocks(entries.size())),
                     dim3(kBlock), 0, static_cast<hipStream_t>(stream), grid,
                     components, entries.device_ptr(), entries.size(),
                     component_mask, payload.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_pic_device_halo_unpack(
    Grid2D grid, const backend::DeviceBuffer<Real>& payload,
    std::size_t payload_count,
    const backend::DeviceBuffer<PicDeviceHaloEntry>& entries,
    const PicDeviceHaloComponents& components,
    std::uint32_t component_mask, PicDeviceHaloUpdate update,
    backend::stream_t stream) {
  validate_common(grid, payload_count, payload.size(), entries.size(),
                  component_mask);
  validate_components(components, component_mask);
  if (update != PicDeviceHaloUpdate::assign &&
      update != PicDeviceHaloUpdate::add) {
    throw std::invalid_argument{"PIC device halo update mode is invalid"};
  }
  if (entries.empty()) return;
  hipLaunchKernelGGL(unpack_halo_kernel,
                     dim3(launch_blocks(entries.size())), dim3(kBlock), 0,
                     static_cast<hipStream_t>(stream), grid,
                     payload.device_ptr(), entries.device_ptr(), entries.size(),
                     components, component_mask, update);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_pic_device_halo_accumulate(
    Grid2D grid, const PicDeviceHaloConstComponents& sources,
    const backend::DeviceBuffer<PicDeviceHaloEntry>& entries,
    const PicDeviceHaloComponents& destinations,
    std::uint32_t component_mask, backend::stream_t stream) {
  validate_common(grid, entries.size(), entries.size(), entries.size(),
                  component_mask);
  validate_components(sources, component_mask);
  validate_components(destinations, component_mask);
  if (entries.empty()) return;
  hipLaunchKernelGGL(accumulate_halo_kernel,
                     dim3(launch_blocks(entries.size())), dim3(kBlock), 0,
                     static_cast<hipStream_t>(stream), grid, sources,
                     entries.device_ptr(), entries.size(), destinations,
                     component_mask);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_pic_device_components_copy(
    Grid2D grid, const PicDeviceHaloConstComponents& sources,
    const PicDeviceHaloComponents& destinations,
    std::uint32_t component_mask, backend::stream_t stream) {
  validate_common(grid, 0, 0, 0, component_mask);
  validate_components(sources, component_mask);
  validate_components(destinations, component_mask);
  hipLaunchKernelGGL(copy_components_kernel,
                     dim3(launch_blocks(grid.storage_size())), dim3(kBlock),
                     0, static_cast<hipStream_t>(stream), grid, sources,
                     destinations, component_mask);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_pic_distributed_source_background(
    Grid2D grid, Real* charge, Real density, backend::stream_t stream) {
  if (charge == nullptr || !std::isfinite(density)) {
    throw std::invalid_argument{"PIC distributed source background is invalid"};
  }
  hipLaunchKernelGGL(add_background_kernel,
                     dim3(launch_blocks(grid.interior_size())), dim3(kBlock),
                     0, static_cast<hipStream_t>(stream), grid, charge,
                     density);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_pic_distributed_filter_axis(
    Grid2D grid, PicDeviceTileExtent tile, const Real* input, Real* output,
    int axis, Real neighbor_weight, Real center_weight, int cylindrical,
    backend::stream_t stream) {
  if (input == nullptr || output == nullptr || input == output ||
      (axis != 0 && axis != 1) || !std::isfinite(neighbor_weight) ||
      !std::isfinite(center_weight)) {
    throw std::invalid_argument{"PIC distributed filter launch is invalid"};
  }
  hipLaunchKernelGGL(distributed_filter_axis_kernel,
                     dim3(launch_blocks(grid.storage_size())), dim3(kBlock),
                     0, static_cast<hipStream_t>(stream), grid, tile, input,
                     output, axis, neighbor_weight, center_weight,
                     cylindrical != 0);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_pic_distributed_axis_values(
    Grid2D grid, const Real* radial_current, Real* axis_values,
    int owns_axis, backend::stream_t stream) {
  if (radial_current == nullptr || axis_values == nullptr) {
    throw std::invalid_argument{"PIC distributed axis-value launch is invalid"};
  }
  hipLaunchKernelGGL(distributed_axis_values_kernel,
                     dim3(launch_blocks(static_cast<std::size_t>(grid.ny))),
                     dim3(kBlock), 0, static_cast<hipStream_t>(stream), grid,
                     radial_current, axis_values, owns_axis != 0);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_pic_distributed_current_correct_order4(
    Grid2D grid, PicDeviceTileExtent tile, const Real* radial_current,
    const Real* axial_current, const Real* rhs_radial,
    const Real* rhs_axial, const Real* axis_values, Real* output_radial,
    Real* output_axial, int x_lo_mode, int x_hi_mode, int y_lo_mode,
    int y_hi_mode, int cylindrical, int on_axis,
    backend::stream_t stream) {
  if (radial_current == nullptr || axial_current == nullptr ||
      rhs_radial == nullptr || rhs_axial == nullptr || axis_values == nullptr ||
      output_radial == nullptr || output_axial == nullptr ||
      radial_current == output_radial || axial_current == output_axial ||
      x_lo_mode < 0 || x_lo_mode > 4 || x_hi_mode < 0 || x_hi_mode > 4 ||
      y_lo_mode < 0 || y_lo_mode > 4 || y_hi_mode < 0 || y_hi_mode > 4) {
    throw std::invalid_argument{
        "PIC distributed fourth-order source launch is invalid"};
  }
  hipLaunchKernelGGL(distributed_order4_kernel,
                     dim3(launch_blocks(grid.storage_size())), dim3(kBlock),
                     0, static_cast<hipStream_t>(stream), grid, tile,
                     radial_current, axial_current, rhs_radial, rhs_axial,
                     axis_values, output_radial, output_axial, x_lo_mode,
                     x_hi_mode, y_lo_mode, y_hi_mode, cylindrical != 0,
                     on_axis != 0);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::pic
