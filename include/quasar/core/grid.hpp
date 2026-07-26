#pragma once

#include "quasar/core/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace quasar {

enum class Side { x_lo, x_hi, y_lo, y_hi };

struct GhostHalo {
  int nghost{1};
};

namespace detail {

// Evaluate short products and quotients without letting an intermediate leave
// Real's exponent range when the final result is representable.  Keeping the
// mantissas O(1) also works for subnormal inputs on both host and HIP device.
QUASAR_HOST_DEVICE inline Real scaled_product4(
    Real a, Real b, Real c, Real d) noexcept {
  if (a == Real{0} || b == Real{0} || c == Real{0} || d == Real{0}) {
    return Real{0};
  }
  int ea = 0, eb = 0, ec = 0, ed = 0;
  const Real ma = frexp(a, &ea);
  const Real mb = frexp(b, &eb);
  const Real mc = frexp(c, &ec);
  const Real md = frexp(d, &ed);
  Real mantissa = ma * mb * mc * md;
  int adjustment = 0;
  mantissa = frexp(mantissa, &adjustment);
  return scalbn(mantissa, ea + eb + ec + ed + adjustment);
}

QUASAR_HOST_DEVICE inline Real scaled_quotient3(
    Real numerator, Real denominator_a, Real denominator_b) noexcept {
  if (numerator == Real{0}) return Real{0};
  int en = 0, ea = 0, eb = 0;
  const Real mn = frexp(numerator, &en);
  const Real ma = frexp(denominator_a, &ea);
  const Real mb = frexp(denominator_b, &eb);
  Real mantissa = (mn / ma) / mb;
  int adjustment = 0;
  mantissa = frexp(mantissa, &adjustment);
  return scalbn(mantissa, en - ea - eb + adjustment);
}

// Evaluate (high-low)/denominator after scaling the subtraction.  Two finite
// coordinates can have an unrepresentable raw difference even though the
// dimensionless reduced coordinate is small (for example a translated domain
// spanning most of Real's range).  Shape, gather, deposit, and periodic-wrap
// indexing all rely on this quotient.
QUASAR_HOST_DEVICE inline Real scaled_difference_quotient(
    Real high, Real low, Real denominator) noexcept {
  if (high == low) return Real{0};
  // Same-sign nearby coordinates satisfy Sterbenz's lemma, so their direct
  // subtraction is both finite and exact.  Normalizing them separately first
  // can round away part of a local offset on a large translated domain.  Use
  // the scaled path only for the case it is meant to protect: a raw difference
  // that leaves the exponent range (typically opposite-sign extremes).
  const Real direct = high - low;
  if (std::isfinite(direct)) return direct / denominator;
  const Real scale = fmax(fabs(high), fabs(low));
  if (scale == Real{0}) return Real{0};
  const Real normalized = high / scale - low / scale;
  if (normalized == Real{0}) return Real{0};
  int es = 0, en = 0, ed = 0;
  const Real ms = frexp(scale, &es);
  const Real mn = frexp(normalized, &en);
  const Real md = frexp(denominator, &ed);
  Real mantissa = (ms * mn) / md;
  int adjustment = 0;
  mantissa = frexp(mantissa, &adjustment);
  return scalbn(mantissa, es + en - ed + adjustment);
}

}  // namespace detail

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

  QUASAR_HOST_DEVICE Real x_at_cell_center(int i) const noexcept {
    return fma(static_cast<Real>(i) + Real{0.5}, dx(), origin_x);
  }

  QUASAR_HOST_DEVICE Real y_at_cell_center(int j) const noexcept {
    return fma(static_cast<Real>(j) + Real{0.5}, dy(), origin_y);
  }

  // -- Cylindrical (r,z) accessors -------------------------------------------
  // In cylindrical mode the x-axis is the radius r (origin_x is r_min, dx() is
  // dr) and the y-axis is the axial coordinate z. These mirror the Cartesian
  // x_at_* helpers so device kernels read the radius with no new type; they are
  // meaningless (but harmless) on a Cartesian run, which never calls them.

  // Radius at the cell center of column i: r = origin_x + (i + 0.5)*dr.
  QUASAR_HOST_DEVICE Real r_at_cell_center(int i) const noexcept {
    return fma(static_cast<Real>(i) + Real{0.5}, dx(), origin_x);
  }

  // Radius at the cell edge (left face) of column i: r = origin_x + i*dr. The
  // i=0 edge sits at origin_x, i.e. r=0 when the domain starts on the axis.
  QUASAR_HOST_DEVICE Real r_at_edge(int i) const noexcept {
    return fma(static_cast<Real>(i), dx(), origin_x);
  }

  // Cell volume for column i under the azimuthal 2*pi convention (the axisymmetric
  // m=0 ring of one cell): V_i = 2*pi * r_at_cell_center(i) * dr * dz.
  QUASAR_HOST_DEVICE Real cell_volume(int i) const noexcept {
    return detail::scaled_product4(
        Real{2} * pi_v<Real>, r_at_cell_center(i), dx(), dy());
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
    // pitch()/height() are part of the device ABI and intentionally return int.
    // Reserve one representable count for the physical high face used by Yee
    // layouts, then reject halos whose arithmetic would overflow the ABI before
    // any allocation or index calculation is attempted.
    constexpr int imax = std::numeric_limits<int>::max();
    if (nx == imax || ny == imax) {
      throw std::overflow_error{
          "Grid2D: dimensions leave no representable physical high face"};
    }
    if (nghost > (imax - nx) / 2 || nghost > (imax - ny) / 2) {
      throw std::overflow_error{"Grid2D: dimensions plus ghost halo exceed int range"};
    }
    const Real dx_value = lx / static_cast<Real>(nx);
    const Real dy_value = ly / static_cast<Real>(ny);
    if (!(std::isfinite(dx_value) && dx_value > Real{0}
          && std::isfinite(dy_value) && dy_value > Real{0})) {
      throw std::overflow_error{"Grid2D: cell spacing is not representable"};
    }
    const Real upper_x = origin_x + lx;
    const Real upper_y = origin_y + ly;
    if (!(std::isfinite(upper_x) && std::isfinite(upper_y))) {
      throw std::overflow_error{"Grid2D: upper domain bound is not representable"};
    }
    // Coordinates are stored as Real. Distinct logical cells must therefore
    // remain distinct after rounding at both ends of the domain.
    // Use the same affine expression as the public coordinate accessors at
    // both ends.  Reassociating the high centre as upper - dx/2 can round to a
    // different value for subnormal spacings and miss a centre that actually
    // collapses onto the high face.
    if (x_at_cell_center(0) == origin_x
        || y_at_cell_center(0) == origin_y
        || x_at_cell_center(nx - 1) == upper_x
        || y_at_cell_center(ny - 1) == upper_y) {
      throw std::overflow_error{"Grid2D: cell coordinates collapse in host precision"};
    }
    const std::size_t p = static_cast<std::size_t>(nx + 2 * nghost);
    const std::size_t h = static_cast<std::size_t>(ny + 2 * nghost);
    if (h != 0 && p > std::numeric_limits<std::size_t>::max() / h) {
      throw std::overflow_error{"Grid2D: storage size is not representable"};
    }
  }
};

// Minimum ghost-cell halo for a given FDTD order: the 2nd-order curl reads one
// cell past the boundary, the 4th-order curl reads two.
inline int required_nghost(int fdtd_order) {
  if (fdtd_order == 2) return 1;
  if (fdtd_order == 4) return 2;
  throw std::invalid_argument{
      "required_nghost: supported FDTD orders are 2 and 4"};
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
  // Algebraically this is 1/(c*factor*hypot(1/dx,1/dy)).  Evaluate it in
  // scaled form so squaring a very small spacing cannot overflow and a large
  // aspect ratio does not discard the smaller contribution prematurely.
  const Real h_min = std::min(g.dx(), g.dy());
  const Real h_max = std::max(g.dx(), g.dy());
  const Real ratio = h_min / h_max;
  const Real spectral_factor = factor * std::sqrt(Real{1} + ratio * ratio);
  const Real dt = detail::scaled_quotient3(h_min, spectral_factor, c);
  if (!(std::isfinite(dt) && dt > Real{0})) {
    throw std::overflow_error{"cfl_dt: stable timestep is not representable"};
  }
  return dt;
}

// Cylindrical (r,z) CFL limit for the m=0 mimetic curl.  Order two has the
// Cartesian Yee bound.  At order four the regular-axis closure is not exactly
// Fourier-diagonal: the conservative all-grid induced-norm estimate is
//   rho(-A_r B_r) <= 35/(6 dr^2),
// while the axial Cartesian contribution is 49/(9 dz^2).  Leapfrog stability
// therefore follows from
//   dt <= 1 / (c*sqrt(35/(24 dr^2) + 49/(36 dz^2))).
// This is a proved sufficient bound, not an empirical safety margin.
inline Real cyl_cfl_dt(const Grid2D& g, int fdtd_order, Real c) {
  g.validate();
  if (c <= Real{0} || !std::isfinite(c)) {
    throw std::invalid_argument{
        "cyl_cfl_dt: wave speed must be finite and positive"};
  }
  if (fdtd_order == 2) return cfl_dt(g, fdtd_order, c);
  if (fdtd_order != 4) {
    throw std::invalid_argument{
        "cyl_cfl_dt: supported FDTD orders are 2 and 4"};
  }

  // Scale by the smaller spacing before squaring.  This is algebraically the
  // formula above, but remains finite for extreme aspect ratios and spacings.
  constexpr Real radial_squared = Real{35} / Real{24};
  constexpr Real axial_squared = Real{49} / Real{36};
  const Real dr = g.dx();
  const Real dz = g.dy();
  Real h_min = dr;
  Real ratio = dr / dz;
  Real spectral_squared = radial_squared + axial_squared * ratio * ratio;
  if (dz < dr) {
    h_min = dz;
    ratio = dz / dr;
    spectral_squared = axial_squared + radial_squared * ratio * ratio;
  }
  const Real spectral_factor = std::sqrt(spectral_squared);
  const Real dt = detail::scaled_quotient3(h_min, spectral_factor, c);
  if (!(std::isfinite(dt) && dt > Real{0})) {
    throw std::overflow_error{
        "cyl_cfl_dt: stable timestep is not representable"};
  }
  return dt;
}

// Backward-compatible order-two spelling.
inline Real cyl_cfl_dt(const Grid2D& g, Real c = Real{1}) {
  return cyl_cfl_dt(g, 2, c);
}

}  // namespace quasar
