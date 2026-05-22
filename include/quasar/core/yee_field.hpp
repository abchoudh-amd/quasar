#pragma once

#include "quasar/backend/memory.hpp"
#include "quasar/core/grid.hpp"

#include <cstddef>

namespace quasar {

template <class T>
struct ScalarGrid2D {
  Grid2D grid{};
  backend::DeviceBuffer<T> values{};

  ScalarGrid2D() = default;
  explicit ScalarGrid2D(Grid2D g) : grid{g}, values{g.storage_size()} {}

  std::size_t size() const noexcept { return values.size(); }
  bool empty() const noexcept { return values.empty(); }

  T*       device_ptr()       noexcept { return values.device_ptr(); }
  const T* device_ptr() const noexcept { return values.device_ptr(); }
};

template <class T>
struct YeeField2D {
  Grid2D grid{};
  backend::DeviceBuffer<T> ex{};
  backend::DeviceBuffer<T> ey{};
  backend::DeviceBuffer<T> ez{};
  backend::DeviceBuffer<T> bx{};
  backend::DeviceBuffer<T> by{};
  backend::DeviceBuffer<T> bz{};

  YeeField2D() = default;
  explicit YeeField2D(Grid2D g)
    : grid{g},
      ex{g.storage_size()}, ey{g.storage_size()}, ez{g.storage_size()},
      bx{g.storage_size()}, by{g.storage_size()}, bz{g.storage_size()} {}

  std::size_t component_size() const noexcept { return grid.storage_size(); }
  std::size_t total_values() const noexcept { return 6 * component_size(); }
};

template <class T>
struct JField2D {
  Grid2D grid{};
  backend::DeviceBuffer<T> jx{};
  backend::DeviceBuffer<T> jy{};
  backend::DeviceBuffer<T> jz{};

  JField2D() = default;
  explicit JField2D(Grid2D g)
    : grid{g}, jx{g.storage_size()}, jy{g.storage_size()}, jz{g.storage_size()} {}

  std::size_t component_size() const noexcept { return grid.storage_size(); }
  std::size_t total_values() const noexcept { return 3 * component_size(); }
};

template <class T>
struct HostYeeField2D {
  Grid2D grid{};
  backend::mirror_view<T> ex{};
  backend::mirror_view<T> ey{};
  backend::mirror_view<T> ez{};
  backend::mirror_view<T> bx{};
  backend::mirror_view<T> by{};
  backend::mirror_view<T> bz{};

  HostYeeField2D() = default;
  explicit HostYeeField2D(Grid2D g)
    : grid{g},
      ex{g.storage_size()}, ey{g.storage_size()}, ez{g.storage_size()},
      bx{g.storage_size()}, by{g.storage_size()}, bz{g.storage_size()} {}
};

}  // namespace quasar
