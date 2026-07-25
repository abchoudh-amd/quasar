#include "quasar/physics/pic/diagnostics.hpp"

#include "quasar/numerics/stencil.hpp"
#include "quasar/physics/pic/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

namespace quasar::pic {

namespace {

// Snapshot one device field component into a host vector sized to the grid's
// padded storage. Diagnostics are host-side reductions; the per-step hot path
// does not call these.
std::vector<Real> component_to_host(const backend::DeviceBuffer<Real>& buf) {
  std::vector<Real> h(buf.size());
  if (!h.empty()) {
    buf.copy_to_host(h.data(), h.size());
  }
  return h;
}

struct ScaledLongValue {
  long double mantissa{0.0L};
  int exponent{0};
};

class ScaledPositiveSum {
 public:
  void add_product(std::initializer_list<long double> factors) {
    long double mantissa = 1.0L;
    int exponent = 0;
    for (const long double factor : factors) {
      if (factor == 0.0L) return;
      if (!(factor > 0.0L) || !std::isfinite(factor)) {
        invalid_ = true;
        return;
      }
      int factor_exponent = 0;
      mantissa *= std::frexp(factor, &factor_exponent);
      exponent += factor_exponent;
      int adjustment = 0;
      mantissa = std::frexp(mantissa, &adjustment);
      exponent += adjustment;
    }
    if (!(mantissa > 0.0L)) {
      invalid_ = true;
      return;
    }
    add({mantissa, exponent});
  }

  void add_square_times(
      long double value, std::initializer_list<long double> factors) {
    if (value == 0.0L) return;
    if (!std::isfinite(value)) {
      invalid_ = true;
      return;
    }
    int value_exponent = 0;
    const long double value_mantissa =
        std::frexp(std::fabs(value), &value_exponent);
    long double mantissa = value_mantissa * value_mantissa;
    int exponent = 2 * value_exponent;
    int adjustment = 0;
    mantissa = std::frexp(mantissa, &adjustment);
    exponent += adjustment;
    for (const long double factor : factors) {
      if (factor == 0.0L) return;
      if (!(factor > 0.0L) || !std::isfinite(factor)) {
        invalid_ = true;
        return;
      }
      int factor_exponent = 0;
      mantissa *= std::frexp(factor, &factor_exponent);
      exponent += factor_exponent;
      mantissa = std::frexp(mantissa, &adjustment);
      exponent += adjustment;
    }
    add({mantissa, exponent});
  }

  ScaledLongValue normalized() const {
    if (invalid_) {
      return {std::numeric_limits<long double>::infinity(), 0};
    }
    if (!initialized_) return {};
    const long double total = sum_ + correction_;
    if (!(total > 0.0L)) return {};
    int adjustment = 0;
    return {std::frexp(total, &adjustment), exponent_ + adjustment};
  }

 private:
  void add(ScaledLongValue value) {
    if (!initialized_) {
      exponent_ = value.exponent;
      initialized_ = true;
    } else if (value.exponent > exponent_) {
      const int shift = exponent_ - value.exponent;
      sum_ = std::scalbn(sum_, shift);
      correction_ = std::scalbn(correction_, shift);
      exponent_ = value.exponent;
    }
    const long double term = std::scalbn(
        value.mantissa, value.exponent - exponent_);
    const long double next = sum_ + term;
    if (std::fabs(sum_) >= std::fabs(term)) {
      correction_ += (sum_ - next) + term;
    } else {
      correction_ += (term - next) + sum_;
    }
    sum_ = next;
  }

  bool initialized_{false};
  bool invalid_{false};
  int exponent_{0};
  long double sum_{0.0L};
  long double correction_{0.0L};
};

Real scaled_sum_to_real(const ScaledPositiveSum& sum, long double multiplier,
                        const char* error) {
  const auto value = sum.normalized();
  if (value.mantissa == 0.0L) return Real{0};
  long double mantissa = value.mantissa * multiplier;
  int adjustment = 0;
  mantissa = std::frexp(mantissa, &adjustment);
  const long double result = std::scalbn(
      mantissa, value.exponent + adjustment);
  if (!std::isfinite(result)
      || result > static_cast<long double>(std::numeric_limits<Real>::max())) {
    throw std::overflow_error{error};
  }
  const Real converted = static_cast<Real>(result);
  if (result != 0.0L && converted == Real{0}) {
    throw std::underflow_error{error};
  }
  return converted;
}

Real scaled_rms(const ScaledPositiveSum& numerator,
                const ScaledPositiveSum& denominator,
                const char* error) {
  const auto num = numerator.normalized();
  const auto den = denominator.normalized();
  if (!(den.mantissa > 0.0L) || !std::isfinite(den.mantissa)
      || !std::isfinite(num.mantissa)) {
    throw std::overflow_error{error};
  }
  if (num.mantissa == 0.0L) return Real{0};
  int exponent = num.exponent - den.exponent;
  long double mantissa = num.mantissa / den.mantissa;
  if (exponent % 2 != 0) {
    mantissa *= 2.0L;
    --exponent;
  }
  const long double rms = std::scalbn(std::sqrt(mantissa), exponent / 2);
  if (!std::isfinite(rms)
      || rms > static_cast<long double>(std::numeric_limits<Real>::max())) {
    throw std::overflow_error{error};
  }
  const Real converted = static_cast<Real>(rms);
  if (rms != 0.0L && converted == Real{0}) {
    throw std::underflow_error{error};
  }
  return converted;
}

}  // namespace

std::size_t alive_count(const ParticleSpecies& species) {
  // Device-side reduction over the alive flags — avoids copying all seven
  // particle arrays to the host just to count survivors.
  return ::launch_pic_alive_count(species, nullptr);
}

Real total_kinetic_energy(const ParticleSpecies& species) {
  const auto snap = species.to_host();
  const long double m = static_cast<long double>(species.mass());
  ScaledPositiveSum ke;
  for (std::size_t p = 0; p < snap.x.size(); ++p) {
    if (snap.alive[p] == 0) continue;
    const long double speed = std::hypot(
        std::hypot(static_cast<long double>(snap.vx[p]),
                   static_cast<long double>(snap.vy[p])),
        static_cast<long double>(snap.vz[p]));
    ke.add_product({0.5L, m, speed, speed,
                    static_cast<long double>(snap.weight[p])});
  }
  return scaled_sum_to_real(
      ke, 1.0L, "total_kinetic_energy: energy is not representable");
}

Real total_em_energy(const YeeField2D<Real>& fields, const Grid2D& grid,
                     const boundary::BoundarySpec& boundary,
                     bool cylindrical) {
  // Normalized natural units (c=eps0=mu0=1). Each staggered sample owns its
  // primal/dual control volume. This is positive definite even at the Nyquist
  // mode; collocating by averaging before squaring would incorrectly erase a
  // checkerboard field and is not the Yee energy norm.
  if (fields.ex.size() < grid.storage_size()) return Real{0};

  const auto ex = component_to_host(fields.ex);
  const auto ey = component_to_host(fields.ey);
  const auto ez = component_to_host(fields.ez);
  const auto bx = component_to_host(fields.bx);
  const auto by = component_to_host(fields.by);
  const auto bz = component_to_host(fields.bz);

  const bool periodic_x = boundary.field[0] == "periodic" &&
                          boundary.field[1] == "periodic";
  const bool periodic_y = boundary.field[2] == "periodic" &&
                          boundary.field[3] == "periodic";
  ScaledPositiveSum twice_energy;
  const auto add = [&](
      Real value, std::initializer_list<long double> volume_factors) {
    if (!std::isfinite(value)) {
      throw std::domain_error{"total_em_energy: field contains a non-finite value"};
    }
    twice_energy.add_square_times(
        static_cast<long double>(value), volume_factors);
  };

  const long double dx = static_cast<long double>(grid.dx());
  const long double dy = static_cast<long double>(grid.dy());
  const auto cart_x = [&](int i) {
    if (periodic_x) return dx;
    return (i == 0 || i == grid.nx) ? 0.5L * dx : dx;
  };
  const auto cart_y = [&](int j) {
    if (periodic_y) return dy;
    return (j == 0 || j == grid.ny) ? 0.5L * dy : dy;
  };
  const int x_faces = periodic_x ? grid.nx : grid.nx + 1;
  const int y_faces = periodic_y ? grid.ny : grid.ny + 1;

  if (!cylindrical) {
    for (int j = 0; j < grid.ny; ++j) {
      for (int i = 0; i < x_faces; ++i) {
        add(ex[grid.index(i, j)], {cart_x(i), dy});
        add(by[grid.index(i, j)], {cart_x(i), dy});
      }
    }
    for (int j = 0; j < y_faces; ++j) {
      for (int i = 0; i < grid.nx; ++i) {
        add(ey[grid.index(i, j)], {dx, cart_y(j)});
        add(bx[grid.index(i, j)], {dx, cart_y(j)});
      }
    }
    for (int j = 0; j < grid.ny; ++j) {
      for (int i = 0; i < grid.nx; ++i) {
        add(ez[grid.index(i, j)], {dx, dy});
      }
    }
    for (int j = 0; j < y_faces; ++j) {
      for (int i = 0; i < x_faces; ++i) {
        add(bz[grid.index(i, j)], {cart_x(i), cart_y(j)});
      }
    }
  } else {
    // Radial face dual annulus [midpoint_left, midpoint_right], clipped to the
    // physical radial domain. Each square difference is split into positive
    // scaled products to retain a thin annulus at a large radius.
    const long double r0 = static_cast<long double>(grid.origin_x);
    const long double pi = static_cast<long double>(pi_v<Real>);
    const auto add_radial_face = [&](Real value, int i, long double dz) {
      const long double width =
          (i == 0 || i == grid.nx) ? 0.5L * dx : dx;
      const long double radial_offset =
          (i == 0) ? 0.5L * dx
                   : (i == grid.nx)
                         ? (2.0L * static_cast<long double>(grid.nx) - 0.5L)
                               * dx
                         : 2.0L * static_cast<long double>(i) * dx;
      // pi*width*(2*r0 + radial_offset), split into two positive
      // products so neither a large 2*r0 nor the full area is materialized.
      add(value, {pi, width, 2.0L, r0, dz});
      add(value, {pi, width, radial_offset, dz});
    };
    const auto add_radial_cell = [&](Real value, int i, long double dz) {
      const long double offset = static_cast<long double>(i) + 0.5L;
      // 2*pi*dx*(r0 + offset*dx), likewise kept as scaled products.
      add(value, {2.0L, pi, dx, r0, dz});
      add(value, {2.0L, pi, dx, offset, dx, dz});
    };
    // Cylindrical radial topology is never periodic: all nx+1 radial faces are
    // physical (the axis face is included and odd components vanish there).
    for (int j = 0; j < grid.ny; ++j) {
      for (int i = 0; i <= grid.nx; ++i) {
        add_radial_face(ex[grid.index(i, j)], i, dy);  // Er
        add_radial_face(ez[grid.index(i, j)], i, dy);  // Ephi
      }
    }
    for (int j = 0; j < y_faces; ++j) {
      for (int i = 0; i < grid.nx; ++i) {
        add_radial_cell(ey[grid.index(i, j)], i, cart_y(j));  // Ez
      }
      for (int i = 0; i <= grid.nx; ++i) {
        add_radial_face(bx[grid.index(i, j)], i, cart_y(j));  // Br
        add_radial_face(bz[grid.index(i, j)], i, cart_y(j));  // Bphi
      }
    }
    for (int j = 0; j < grid.ny; ++j) {
      for (int i = 0; i < grid.nx; ++i) {
        add_radial_cell(by[grid.index(i, j)], i, dy);  // Bz
      }
    }
  }

  return scaled_sum_to_real(
      twice_energy, 0.5L, "total_em_energy: energy is not representable");
}

Real total_em_energy(const YeeField2D<Real>& fields, const Grid2D& grid,
                     bool cylindrical) {
  return total_em_energy(fields, grid, boundary::BoundarySpec{}, cylindrical);
}

Real total_em_energy(const EmPic2D3V& solver) {
  return total_em_energy(solver.fields(), solver.grid(), solver.config().boundary,
                         solver.config().geometry == "cylindrical");
}

Real gauss_residual(const YeeField2D<Real>& fields,
                    const ScalarGrid2D<Real>& charge, int fdtd_order,
                    const boundary::BoundarySpec& boundary, bool cylindrical) {
  // Gauss's law residual ‖div(E)-rho‖_2 using the same forward staggered
  // derivative that annihilates Ampere's curl. Periodic axes wrap explicitly;
  // wall axes read their maintained high-face halo.
  const auto& grid = fields.grid;
  if (fields.ex.size() < grid.storage_size() ||
      charge.values.size() < grid.storage_size()) return Real{0};
  const auto ex = component_to_host(fields.ex);
  const auto ey = component_to_host(fields.ey);
  const auto rho = component_to_host(charge.values);

  const bool periodic_x = boundary.field[0] == "periodic" &&
                          boundary.field[1] == "periodic";
  const bool periodic_y = boundary.field[2] == "periodic" &&
                          boundary.field[3] == "periodic";
  const auto at = [&](const std::vector<Real>& f, int i, int j) -> Real {
    const int ii = periodic_x ? grid.wrap_i(i) :
                   std::max(-grid.nghost, std::min(grid.nx - 1 + grid.nghost, i));
    const int jj = periodic_y ? grid.wrap_j(j) :
                   std::max(-grid.nghost, std::min(grid.ny - 1 + grid.nghost, j));
    return f[grid.index(ii, jj)];
  };
  const auto dx_fwd = [&](const std::vector<Real>& f, int i, int j) {
    if (fdtd_order == 4) {
      return numerics::staggered_derivative_scaled_values<4>(
          at(f, i - 1, j), at(f, i, j), at(f, i + 1, j),
          at(f, i + 2, j), grid.dx());
    }
    return numerics::staggered_derivative_scaled_values<2>(
        Real{0}, at(f, i, j), at(f, i + 1, j), Real{0}, grid.dx());
  };
  const auto dy_fwd = [&](const std::vector<Real>& f, int i, int j) {
    if (fdtd_order == 4) {
      return numerics::staggered_derivative_scaled_values<4>(
          at(f, i, j - 1), at(f, i, j), at(f, i, j + 1),
          at(f, i, j + 2), grid.dy());
    }
    return numerics::staggered_derivative_scaled_values<2>(
        Real{0}, at(f, i, j), at(f, i, j + 1), Real{0}, grid.dy());
  };
  ScaledPositiveSum weighted_sum_sq;
  ScaledPositiveSum total_volume;
  const long double dx = static_cast<long double>(grid.dx());
  const long double dy = static_cast<long double>(grid.dy());
  const long double r0 = static_cast<long double>(grid.origin_x);
  const long double pi = static_cast<long double>(pi_v<Real>);
  for (int j = 0; j < grid.ny; ++j) {
    for (int i = 0; i < grid.nx; ++i) {
      Real residual;
      if (cylindrical) {
        numerics::detail::ScaledValue radial;
        if (fdtd_order == 4) {
          radial = numerics::cylindrical_radial_flux_scaled_values<4>(
              at(ex, i - 1, j), at(ex, i, j), at(ex, i + 1, j),
              at(ex, i + 2, j), grid.r_at_cell_center(i), grid.dx());
        } else {
          radial = numerics::cylindrical_radial_flux_scaled_values<2>(
              Real{0}, at(ex, i, j), at(ex, i + 1, j), Real{0},
              grid.r_at_cell_center(i), grid.dx());
        }
        residual = numerics::detail::scaled_signed_sum(
            radial, dy_fwd(ey, i, j),
            numerics::detail::negate(numerics::detail::scaled_value(
                rho[grid.index(i, j)])));
      } else {
        residual = numerics::detail::scaled_signed_sum(
            dx_fwd(ex, i, j), dy_fwd(ey, i, j),
            numerics::detail::negate(numerics::detail::scaled_value(
                rho[grid.index(i, j)])));
      }
      if (!std::isfinite(residual)) {
        throw std::domain_error{
            "gauss_residual: divergence or charge contains a non-finite value"};
      }
      const long double r = static_cast<long double>(residual);
      if (cylindrical) {
        const long double offset = static_cast<long double>(i) + 0.5L;
        // 2*pi*dx*dy*(r0 + offset*dx), split to retain a thin annulus
        // at a large radius without forming either radius or volume first.
        weighted_sum_sq.add_square_times(r, {2.0L, pi, dx, dy, r0});
        weighted_sum_sq.add_square_times(
            r, {2.0L, pi, dx, dy, offset, dx});
        total_volume.add_product({2.0L, pi, dx, dy, r0});
        total_volume.add_product({2.0L, pi, dx, dy, offset, dx});
      } else {
        weighted_sum_sq.add_square_times(r, {dx, dy});
        total_volume.add_product({dx, dy});
      }
    }
  }
  return scaled_rms(
      weighted_sum_sq, total_volume,
      "gauss_residual: residual norm or domain volume is not representable");
}

Real gauss_residual(EmPic2D3V& solver) {
  return gauss_residual(solver.fields(), solver.charge_density(),
                        solver.config().fdtd_order, solver.config().boundary,
                        solver.config().geometry == "cylindrical");
}

Real electric_divergence_norm(const YeeField2D<Real>& fields, int fdtd_order,
                              const boundary::BoundarySpec& boundary,
                              bool cylindrical) {
  ScalarGrid2D<Real> zero{fields.grid};
  return gauss_residual(fields, zero, fdtd_order, boundary, cylindrical);
}

}  // namespace quasar::pic
