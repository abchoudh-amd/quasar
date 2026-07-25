#pragma once

#include "quasar/backend/memory.hpp"
#include "quasar/core/grid.hpp"

#include <cstddef>

namespace quasar {

namespace detail {

inline std::size_t validated_grid_storage_size(const Grid2D& grid) {
  grid.validate();
  return grid.storage_size();
}

template <class T>
inline std::size_t yee_component_storage_size(const Grid2D& grid,
                                              std::size_t components) {
  const std::size_t component_size = validated_grid_storage_size(grid);
  if (grid.nghost < 1) {
    throw std::invalid_argument{
        "Yee field layouts require at least one ghost cell for physical high faces"};
  }
  const std::size_t total_values = backend::detail::checked_size_product(
      component_size, components,
      "Yee field layout: aggregate element count overflows size_t");
  (void)backend::detail::checked_size_product(
      total_values, sizeof(T),
      "Yee field layout: aggregate byte count overflows size_t");
  return component_size;
}

}  // namespace detail

// Logical (unpadded) component extent.  YeeField2D keeps every component in a
// Grid2D-sized padded allocation so all six arrays share one pitch and can use
// Grid2D::index.  The physical component lattices are nevertheless different:
// the upper face/node at i=nx and/or j=ny is a real degree of freedom, stored in
// what is the first high halo slot for a cell-centred component.  Keeping the
// allocation uniform preserves the public/device ABI; these extents define which
// slots are physical and must be advanced rather than treated as ghosts.
struct YeeExtent2D {
  int nx{0};
  int ny{0};
};

struct CartesianYeeLayout2D {
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D ex(const Grid2D& g) noexcept {
    return {g.nx + 1, g.ny};
  }
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D ey(const Grid2D& g) noexcept {
    return {g.nx, g.ny + 1};
  }
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D ez(const Grid2D& g) noexcept {
    return {g.nx, g.ny};
  }
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D bx(const Grid2D& g) noexcept {
    return {g.nx, g.ny + 1};
  }
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D by(const Grid2D& g) noexcept {
    return {g.nx + 1, g.ny};
  }
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D bz(const Grid2D& g) noexcept {
    return {g.nx + 1, g.ny + 1};
  }
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D jx(const Grid2D& g) noexcept {
    return ex(g);
  }
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D jy(const Grid2D& g) noexcept {
    return ey(g);
  }
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D jz(const Grid2D& g) noexcept {
    return ez(g);
  }
};

// Axisymmetric (r,z) staggering follows field parity in radius: the odd
// components Er/Ephi/Br/Bphi live on radial faces (including r=0), while the
// even Ez/Bz components live at radial centres. Axial staggering remains Yee.
// Stored component names map (x,y,z) -> (r,z,phi).
struct CylindricalYeeLayout2D {
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D ex(const Grid2D& g) noexcept {
    return {g.nx + 1, g.ny};       // Er
  }
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D ey(const Grid2D& g) noexcept {
    return {g.nx, g.ny + 1};       // Ez
  }
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D ez(const Grid2D& g) noexcept {
    return {g.nx + 1, g.ny};       // Ephi
  }
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D bx(const Grid2D& g) noexcept {
    return {g.nx + 1, g.ny + 1};   // Br
  }
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D by(const Grid2D& g) noexcept {
    return {g.nx, g.ny};           // Bz
  }
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D bz(const Grid2D& g) noexcept {
    return {g.nx + 1, g.ny + 1};   // Bphi
  }
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D jx(const Grid2D& g) noexcept {
    return ex(g);
  }
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D jy(const Grid2D& g) noexcept {
    return ey(g);
  }
  QUASAR_HOST_DEVICE static constexpr YeeExtent2D jz(const Grid2D& g) noexcept {
    return ez(g);
  }
};

template <class T>
struct ScalarGrid2D {
  Grid2D grid{};
  backend::DeviceBuffer<T> values{};

  ScalarGrid2D() = default;
  explicit ScalarGrid2D(Grid2D g)
    : grid{g}, values{detail::validated_grid_storage_size(grid)} {}

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
      ex{detail::yee_component_storage_size<T>(grid, 6)},
      ey{grid.storage_size()}, ez{grid.storage_size()},
      bx{grid.storage_size()}, by{grid.storage_size()}, bz{grid.storage_size()} {}

  std::size_t component_size() const noexcept { return ex.size(); }
  std::size_t total_values() const {
    return backend::detail::checked_size_product(
        component_size(), 6,
        "YeeField2D::total_values: element count overflows size_t");
  }
};

template <class T>
struct JField2D {
  Grid2D grid{};
  backend::DeviceBuffer<T> jx{};
  backend::DeviceBuffer<T> jy{};
  backend::DeviceBuffer<T> jz{};

  JField2D() = default;
  explicit JField2D(Grid2D g)
    : grid{g},
      jx{detail::yee_component_storage_size<T>(grid, 3)},
      jy{grid.storage_size()}, jz{grid.storage_size()} {}

  std::size_t component_size() const noexcept { return jx.size(); }
  std::size_t total_values() const {
    return backend::detail::checked_size_product(
        component_size(), 3,
        "JField2D::total_values: element count overflows size_t");
  }
};

// Three-component magnetic snapshot used to time-centre the Lorentz force. The
// Yee solver stores B at half steps; retaining the pre-Faraday state lets the
// pusher gather B^n = (B^{n-1/2}+B^{n+1/2})/2 without duplicating E/J storage.
template <class T>
struct BField2D {
  Grid2D grid{};
  backend::DeviceBuffer<T> bx{};
  backend::DeviceBuffer<T> by{};
  backend::DeviceBuffer<T> bz{};

  BField2D() = default;
  explicit BField2D(Grid2D g)
    : grid{g},
      bx{detail::yee_component_storage_size<T>(grid, 3)},
      by{grid.storage_size()}, bz{grid.storage_size()} {}
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
      ex{detail::yee_component_storage_size<T>(grid, 6)},
      ey{grid.storage_size()}, ez{grid.storage_size()},
      bx{grid.storage_size()}, by{grid.storage_size()}, bz{grid.storage_size()} {}
};

}  // namespace quasar
