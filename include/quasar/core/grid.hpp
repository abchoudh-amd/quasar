#pragma once

#include "quasar/core/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace quasar {

enum class Side { x_lo, x_hi, y_lo, y_hi };

struct GhostHalo {
  int nghost{1};
};

struct Grid2D {
  int  nx{1};
  int  ny{1};
  Real lx{1};
  Real ly{1};
  Real origin_x{0};
  Real origin_y{0};
  int  nghost{1};

  constexpr Grid2D() = default;

  Grid2D(int nx_in, int ny_in, Real lx_in, Real ly_in,
         Real ox = Real{0}, Real oy = Real{0}, int halo = 1)
    : nx{nx_in}, ny{ny_in}, lx{lx_in}, ly{ly_in},
      origin_x{ox}, origin_y{oy}, nghost{halo} {
    validate();
  }

  QUASAR_HOST_DEVICE constexpr Real dx() const noexcept { return lx / static_cast<Real>(nx); }
  QUASAR_HOST_DEVICE constexpr Real dy() const noexcept { return ly / static_cast<Real>(ny); }
  QUASAR_HOST_DEVICE constexpr int  pitch() const noexcept { return nx + 2 * nghost; }
  QUASAR_HOST_DEVICE constexpr int  height() const noexcept { return ny + 2 * nghost; }
  QUASAR_HOST_DEVICE constexpr std::size_t interior_size() const noexcept {
    return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
  }
  QUASAR_HOST_DEVICE constexpr std::size_t storage_size() const noexcept {
    return static_cast<std::size_t>(pitch()) * static_cast<std::size_t>(height());
  }

  QUASAR_HOST_DEVICE constexpr int wrap_i(int i) const noexcept {
    int r = i % nx;
    return r < 0 ? r + nx : r;
  }

  QUASAR_HOST_DEVICE constexpr int wrap_j(int j) const noexcept {
    int r = j % ny;
    return r < 0 ? r + ny : r;
  }

  QUASAR_HOST_DEVICE constexpr std::size_t index(int i, int j) const noexcept {
    return static_cast<std::size_t>(i + nghost)
         + static_cast<std::size_t>(pitch()) * static_cast<std::size_t>(j + nghost);
  }

  QUASAR_HOST_DEVICE constexpr std::size_t periodic_index(int i, int j) const noexcept {
    return index(wrap_i(i), wrap_j(j));
  }

  QUASAR_HOST_DEVICE constexpr Real x_at_cell_center(int i) const noexcept {
    return origin_x + (static_cast<Real>(i) + Real{0.5}) * dx();
  }

  QUASAR_HOST_DEVICE constexpr Real y_at_cell_center(int j) const noexcept {
    return origin_y + (static_cast<Real>(j) + Real{0.5}) * dy();
  }

  void validate() const {
    if (nx <= 0 || ny <= 0) {
      throw std::invalid_argument{"Grid2D: nx and ny must be positive"};
    }
    if (!(std::isfinite(lx) && std::isfinite(ly)) || lx <= Real{0} || ly <= Real{0}) {
      throw std::invalid_argument{"Grid2D: lx and ly must be finite and positive"};
    }
    if (!(std::isfinite(origin_x) && std::isfinite(origin_y))) {
      throw std::invalid_argument{"Grid2D: origin must be finite"};
    }
    if (nghost < 0) {
      throw std::invalid_argument{"Grid2D: ghost halo must be non-negative"};
    }
  }
};

// Minimum ghost-cell halo for a given FDTD order: the 2nd-order curl reads one
// cell past the boundary, the 4th-order curl reads two.
constexpr int required_nghost(int fdtd_order) noexcept {
  return fdtd_order == 4 ? 2 : 1;
}

inline Real cfl_dt(const Grid2D& g, int fdtd_order, Real c = Real{1}) {
  g.validate();
  if (c <= Real{0} || !std::isfinite(c)) {
    throw std::invalid_argument{"cfl_dt: wave speed must be finite and positive"};
  }
  Real factor = Real{1};
  if (fdtd_order == 4) {
    factor = Real{7} / Real{6};
  } else if (fdtd_order != 2) {
    throw std::invalid_argument{"cfl_dt: supported FDTD orders are 2 and 4"};
  }
  const Real sx = Real{1} / (g.dx() * g.dx());
  const Real sy = Real{1} / (g.dy() * g.dy());
  return Real{1} / (c * factor * std::sqrt(sx + sy));
}

}  // namespace quasar
