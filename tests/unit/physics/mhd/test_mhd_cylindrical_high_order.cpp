#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/finite_volume_quadrature.hpp"
#include "quasar/numerics/radial_tables.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"
#include "quasar/physics/mhd/mhd_solver.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace {

using quasar::Grid2D;
using quasar::Real;

constexpr Real kGamma = Real{5} / Real{3};

void zero_field(quasar::mhd::MhdField2D<Real>& field,
                const std::vector<Real>& zero) {
  field.rho.copy_from_host(zero.data(), zero.size());
  field.mx.copy_from_host(zero.data(), zero.size());
  field.my.copy_from_host(zero.data(), zero.size());
  field.mz.copy_from_host(zero.data(), zero.size());
  field.energy.copy_from_host(zero.data(), zero.size());
  field.bx_face.copy_from_host(zero.data(), zero.size());
  field.by_face.copy_from_host(zero.data(), zero.size());
  field.bz_cell.copy_from_host(zero.data(), zero.size());
}

long double integer_power(long double value, int power) {
  long double result = 1.0L;
  for (int n = 0; n < power; ++n) result *= value;
  return result;
}

long double uniform_cell_average_power(long double lo, long double hi,
                                       int power) {
  return (integer_power(hi, power + 1) -
          integer_power(lo, power + 1)) /
         (static_cast<long double>(power + 1) * (hi - lo));
}

long double radial_cell_average_power(long double lo, long double hi,
                                      int power) {
  const long double numerator =
      (integer_power(hi, power + 2) -
       integer_power(lo, power + 2)) /
      static_cast<long double>(power + 2);
  const long double volume = (hi * hi - lo * lo) / 2.0L;
  return numerator / volume;
}

long double angular_cell_average_power(long double lo, long double hi,
                                       int power) {
  const long double numerator =
      (integer_power(hi, power + 3) -
       integer_power(lo, power + 3)) /
      static_cast<long double>(power + 3);
  const long double volume = (hi * hi * hi - lo * lo * lo) / 3.0L;
  return numerator / volume;
}

long double radial_cell_average_sine(long double lo, long double hi,
                                     long double wave) {
  const auto primitive = [wave](long double radius) {
    return -radius * std::cos(wave * radius) / wave +
           std::sin(wave * radius) / (wave * wave);
  };
  const long double volume = (hi * hi - lo * lo) / 2.0L;
  return (primitive(hi) - primitive(lo)) / volume;
}

long double radial_cell_average_cosine_derivative(
    long double lo, long double hi, long double wave) {
  // Ring average of d_r sin(wave*r) = wave*cos(wave*r).
  const auto primitive = [wave](long double radius) {
    return radius * std::sin(wave * radius) +
           std::cos(wave * radius) / wave;
  };
  const long double volume = (hi * hi - lo * lo) / 2.0L;
  return (primitive(hi) - primitive(lo)) / volume;
}

Real separable_monomial_average(const Grid2D& grid, int i, int j,
                                int radial_power, int axial_power,
                                long double radial_scale,
                                long double axial_scale) {
  const long double dr = static_cast<long double>(grid.dx());
  const long double dz = static_cast<long double>(grid.dy());
  const long double r_lo =
      static_cast<long double>(grid.origin_x) + i * dr;
  const long double z_lo =
      static_cast<long double>(grid.origin_y) + j * dz;
  const long double radial_average =
      radial_cell_average_power(r_lo, r_lo + dr, radial_power) /
      integer_power(radial_scale, radial_power);
  const long double axial_average =
      uniform_cell_average_power(z_lo, z_lo + dz, axial_power) /
      integer_power(axial_scale, axial_power);
  return static_cast<Real>(radial_average * axial_average);
}

Real separable_uniform_radial_monomial_average(
    const Grid2D& grid, int i, int j, int radial_power, int axial_power,
    long double radial_scale, long double axial_scale) {
  const long double dr = static_cast<long double>(grid.dx());
  const long double dz = static_cast<long double>(grid.dy());
  const long double r_lo =
      static_cast<long double>(grid.origin_x) + i * dr;
  const long double z_lo =
      static_cast<long double>(grid.origin_y) + j * dz;
  const long double radial_average =
      uniform_cell_average_power(r_lo, r_lo + dr, radial_power) /
      integer_power(radial_scale, radial_power);
  const long double axial_average =
      uniform_cell_average_power(z_lo, z_lo + dz, axial_power) /
      integer_power(axial_scale, axial_power);
  return static_cast<Real>(radial_average * axial_average);
}

Real split_energy_inner_product(
    const Grid2D& grid, int order, const std::vector<Real>& background_z,
    const std::vector<Real>& magnetic_rate_z, int target_i, int target_j) {
  const std::size_t size = grid.storage_size();
  const std::vector<Real> zero(size, Real{0});
  const std::vector<quasar::numerics::ScaledValue> zero_covariance(size);

  quasar::mhd::MhdBackgroundField<Real> background{grid};
  background.active = true;
  background.globally_curl_free = false;
  background.b0x_face.copy_from_host(zero.data(), size);
  background.b0y_face.copy_from_host(zero.data(), size);
  background.b0z_cell.copy_from_host(background_z.data(), size);

  quasar::mhd::MhdField2D<Real> flux_x{grid};
  quasar::mhd::MhdField2D<Real> flux_y{grid};
  quasar::mhd::MhdField2D<Real> rate{grid};
  zero_field(flux_x, zero);
  zero_field(flux_y, zero);
  zero_field(rate, zero);
  rate.bz_cell.copy_from_host(magnetic_rate_z.data(), size);

  quasar::mhd::MhdMomentumFluxParts2D<Real> parts_x{grid};
  quasar::mhd::MhdMomentumFluxParts2D<Real> parts_y{grid};
  parts_x.b0_induction_covariance.copy_from_host(zero_covariance.data(), size);
  parts_y.b0_induction_covariance.copy_from_host(zero_covariance.data(), size);

  quasar::mhd::BoundaryFlags4 outflow{};
  for (int side = 0; side < 4; ++side) outflow.side[side] = 1;
  quasar::numerics::RadialTables radial_tables{grid, order};
  quasar::mhd::launch_mhd_split_energy_residual(
      background, flux_x, parts_x, flux_y, parts_y, rate, outflow,
      /*stream=*/nullptr, /*cylindrical=*/true,
      /*collocation_order=*/0, order, radial_tables.view());
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> energy_rate(size);
  rate.energy.copy_to_host(energy_rate.data(), size);
  return energy_rate[grid.index(target_i, target_j)];
}

long double uniform_average_sine(long double lo, long double hi,
                                 long double wave) {
  return (std::cos(wave * lo) - std::cos(wave * hi)) /
         (wave * (hi - lo));
}

long double uniform_average_sine_squared(long double lo, long double hi,
                                         long double wave) {
  return 0.5L -
         (std::sin(2.0L * wave * hi) -
          std::sin(2.0L * wave * lo)) /
             (4.0L * wave * (hi - lo));
}

long double uniform_average_sine_cosine(long double lo, long double hi,
                                        long double wave) {
  return (std::cos(2.0L * wave * lo) -
          std::cos(2.0L * wave * hi)) /
         (4.0L * wave * (hi - lo));
}

void zero_momentum_flux_parts(
    quasar::mhd::MhdMomentumFluxParts2D<Real>& parts,
    const std::vector<Real>& zero, const std::vector<int>& zero_flag) {
  const std::size_t size = zero.size();
  parts.wave_x.copy_from_host(zero.data(), size);
  parts.wave_y.copy_from_host(zero.data(), size);
  parts.wave_z.copy_from_host(zero.data(), size);
  for (auto& point : parts.cross_b_point) {
    point.x.copy_from_host(zero.data(), size);
    point.y.copy_from_host(zero.data(), size);
    point.z.copy_from_host(zero.data(), size);
  }
  parts.quadrature_valid.copy_from_host(zero_flag.data(), size);
}

struct HostRadialState {
  explicit HostRadialState(std::size_t size)
      : rho(size, Real{0}), radial_momentum(size, Real{0}),
        axial_momentum(size, Real{0}),
        azimuthal_momentum(size, Real{0}), energy(size, Real{0}),
        radial_field(size, Real{0}), axial_field(size, Real{0}),
        toroidal_field(size, Real{0}) {}

  std::vector<Real> rho;
  std::vector<Real> radial_momentum;
  std::vector<Real> axial_momentum;
  std::vector<Real> azimuthal_momentum;
  std::vector<Real> energy;
  std::vector<Real> radial_field;
  std::vector<Real> axial_field;
  std::vector<Real> toroidal_field;
};

void copy_host_state(const HostRadialState& host,
                     quasar::mhd::MhdField2D<Real>& device) {
  const std::size_t size = host.rho.size();
  device.rho.copy_from_host(host.rho.data(), size);
  device.mx.copy_from_host(host.radial_momentum.data(), size);
  device.my.copy_from_host(host.axial_momentum.data(), size);
  device.mz.copy_from_host(host.azimuthal_momentum.data(), size);
  device.energy.copy_from_host(host.energy.data(), size);
  device.bx_face.copy_from_host(host.radial_field.data(), size);
  device.by_face.copy_from_host(host.axial_field.data(), size);
  device.bz_cell.copy_from_host(host.toroidal_field.data(), size);
}

std::vector<Real> inactive_radial_residual(
    const Grid2D& grid, int order, Real gamma,
    const HostRadialState& host_state,
    const std::vector<Real>& radial_flux,
    const std::vector<Real>& axial_flux) {
  const std::size_t size = grid.storage_size();
  const std::vector<Real> zero(size, Real{0});
  quasar::mhd::MhdField2D<Real> state{grid};
  quasar::mhd::MhdField2D<Real> flux_r{grid};
  quasar::mhd::MhdField2D<Real> flux_z{grid};
  quasar::mhd::MhdField2D<Real> residual{grid};
  copy_host_state(host_state, state);
  zero_field(flux_r, zero);
  zero_field(flux_z, zero);
  zero_field(residual, zero);
  flux_r.mx.copy_from_host(radial_flux.data(), size);
  flux_z.mx.copy_from_host(axial_flux.data(), size);

  quasar::mhd::MhdBackgroundField<Real> inactive_background{grid};
  quasar::numerics::RadialTables radial_tables{grid, order};
  quasar::mhd::BoundaryFlags4 outflow{};
  for (int side = 0; side < 4; ++side) outflow.side[side] = 1;
  quasar::mhd::launch_mhd_cylindrical_radial_momentum_residual(
      state, inactive_background, flux_r, flux_z, residual, outflow,
      /*stream=*/nullptr, gamma, /*collocation_order=*/0, order,
      /*parts_r=*/nullptr, /*parts_z=*/nullptr, radial_tables.view());
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> result(size);
  residual.mx.copy_to_host(result.data(), size);
  return result;
}

std::vector<Real> inactive_tensor_source(
    const Grid2D& grid, int order, Real gamma,
    const HostRadialState& host_state) {
  const std::size_t size = grid.storage_size();
  const std::vector<Real> zero(size, Real{0});
  quasar::mhd::MhdField2D<Real> state{grid};
  quasar::mhd::MhdField2D<Real> residual{grid};
  copy_host_state(host_state, state);
  zero_field(residual, zero);

  quasar::mhd::MhdBackgroundField<Real> inactive_background{grid};
  quasar::numerics::RadialTables radial_tables{grid, order};
  quasar::mhd::launch_mhd_geometric_source(
      state, residual, inactive_background, grid, gamma,
      /*stream=*/nullptr, /*collocation_order=*/0, radial_tables.view());
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> result(size);
  residual.mx.copy_to_host(result.data(), size);
  return result;
}

struct StaticToroidalForceError {
  Real radial{};
  Real axial{};
};

Real dynamic_radial_force_error(int order, int resolution) {
  const Grid2D grid{resolution, /*ny=*/4, Real{1}, Real{1},
                    Real{1}, Real{0}, /*nghost=*/4};
  const std::size_t size = grid.storage_size();
  const std::vector<Real> zero(size, Real{0});
  std::vector<Real> rho(size, Real{1});
  std::vector<Real> mphi(size);
  std::vector<Real> energy(size);
  std::vector<Real> radial_flux(size);
  constexpr long double pressure_base = 2.0L;
  constexpr long double pressure_amplitude = 0.2L;
  constexpr long double toroidal_coefficient = 0.15L;
  constexpr long double wave = 2.0L * quasar::pi;

  // rho=1, m_r=m_z=B=0, m_phi=beta*r^2, and
  // p=p0+A*sin(2*pi*r). Store exact ring averages, including the nonlinear
  // kinetic-energy moment, throughout the MP halo. The exact radial numerical
  // flux supplied below is T_rr=p; T_phiphi=p+m_phi^2 is genuinely distinct.
  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
      const long double r_lo =
          static_cast<long double>(grid.origin_x) +
          static_cast<long double>(i) * static_cast<long double>(grid.dx());
      const long double r_hi = r_lo + static_cast<long double>(grid.dx());
      const long double pressure_average =
          pressure_base + pressure_amplitude *
                              radial_cell_average_sine(r_lo, r_hi, wave);
      const std::size_t index = grid.index(i, j);
      mphi[index] = static_cast<Real>(
          toroidal_coefficient *
          angular_cell_average_power(r_lo, r_hi, /*power=*/2));
      energy[index] = static_cast<Real>(
          pressure_average / (static_cast<long double>(kGamma) - 1.0L) +
          0.5L * toroidal_coefficient * toroidal_coefficient *
              radial_cell_average_power(r_lo, r_hi, /*power=*/4));

      const long double face_radius = r_lo;
      radial_flux[index] = static_cast<Real>(
          pressure_base +
          pressure_amplitude * std::sin(wave * face_radius));
    }
  }

  quasar::mhd::MhdField2D<Real> state{grid};
  quasar::mhd::MhdField2D<Real> flux_r{grid};
  quasar::mhd::MhdField2D<Real> flux_z{grid};
  quasar::mhd::MhdField2D<Real> residual{grid};
  zero_field(state, zero);
  zero_field(flux_r, zero);
  zero_field(flux_z, zero);
  zero_field(residual, zero);
  state.rho.copy_from_host(rho.data(), size);
  state.mz.copy_from_host(mphi.data(), size);
  state.energy.copy_from_host(energy.data(), size);
  flux_r.mx.copy_from_host(radial_flux.data(), size);

  quasar::mhd::MhdBackgroundField<Real> background{grid};
  quasar::numerics::RadialTables radial_tables{grid, order};
  quasar::mhd::BoundaryFlags4 outflow{};
  for (int side = 0; side < 4; ++side) outflow.side[side] = 1;
  quasar::mhd::launch_mhd_cylindrical_radial_momentum_residual(
      state, background, flux_r, flux_z, residual, outflow,
      /*stream=*/nullptr, kGamma, /*collocation_order=*/0, order,
      /*parts_r=*/nullptr, /*parts_z=*/nullptr, radial_tables.view());
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> radial_rate(size);
  residual.mx.copy_to_host(radial_rate.data(), size);
  Real error = Real{0};
  for (int j = 0; j < grid.ny; ++j) {
    for (int i = 0; i < grid.nx; ++i) {
      const long double r_lo =
          static_cast<long double>(grid.origin_x) +
          static_cast<long double>(i) * static_cast<long double>(grid.dx());
      const long double r_hi = r_lo + static_cast<long double>(grid.dx());
      const long double volume = (r_hi * r_hi - r_lo * r_lo) / 2.0L;
      const long double pressure_force =
          -pressure_amplitude * radial_cell_average_cosine_derivative(
                                    r_lo, r_hi, wave);
      const long double rotation_force =
          toroidal_coefficient * toroidal_coefficient *
          (integer_power(r_hi, 5) - integer_power(r_lo, 5)) /
          (5.0L * volume);
      const Real exact =
          static_cast<Real>(pressure_force + rotation_force);
      error += std::abs(radial_rate[grid.index(i, j)] - exact);
    }
  }
  return error / static_cast<Real>(grid.nx * grid.ny);
}

StaticToroidalForceError static_toroidal_force_error(int order,
                                                     int resolution) {
  const Grid2D grid{resolution, resolution, Real{1}, Real{1},
                    Real{1}, Real{0}, /*nghost=*/4};
  const std::size_t size = grid.storage_size();
  const std::vector<Real> zero(size, Real{0});
  const std::vector<Real> one(size, Real{1});
  const std::vector<int> zero_flag(size, 0);
  std::vector<Real> background_phi(size);
  constexpr long double wave = 2.0L * quasar::pi;

  // B0 = r sin(2 pi z) e_phi is axisymmetrically divergence-free and carries
  // current. B0_phi uses the uniform-dr cell measure, so store that exact
  // average, including every analytic ghost needed by the MP stencil.
  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    const long double z_lo =
        static_cast<long double>(grid.origin_y) +
        static_cast<long double>(j) * static_cast<long double>(grid.dy());
    const long double z_hi = z_lo + static_cast<long double>(grid.dy());
    const long double z_average =
        uniform_average_sine(z_lo, z_hi, wave);
    for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
      const long double r_lo =
          static_cast<long double>(grid.origin_x) +
          static_cast<long double>(i) * static_cast<long double>(grid.dx());
      const long double r_hi = r_lo + static_cast<long double>(grid.dx());
      background_phi[grid.index(i, j)] = static_cast<Real>(
          uniform_cell_average_power(r_lo, r_hi, /*power=*/1) * z_average);
    }
  }

  quasar::mhd::MhdBackgroundField<Real> background{grid};
  background.active = true;
  background.globally_curl_free = false;
  background.b0x_face.copy_from_host(zero.data(), size);
  background.b0y_face.copy_from_host(zero.data(), size);
  background.b0z_cell.copy_from_host(background_phi.data(), size);

  quasar::mhd::MhdField2D<Real> flux_r{grid};
  quasar::mhd::MhdField2D<Real> flux_z{grid};
  quasar::mhd::MhdField2D<Real> residual{grid};
  zero_field(flux_r, zero);
  zero_field(flux_z, zero);
  zero_field(residual, zero);

  quasar::mhd::MhdMomentumFluxParts2D<Real> parts_r{grid};
  quasar::mhd::MhdMomentumFluxParts2D<Real> parts_z{grid};
  zero_momentum_flux_parts(parts_r, zero, zero_flag);
  zero_momentum_flux_parts(parts_z, zero, zero_flag);

  quasar::numerics::RadialTables radial_tables{grid, order};
  quasar::mhd::BoundaryFlags4 outflow{};
  for (int side = 0; side < 4; ++side) outflow.side[side] = 1;
  quasar::mhd::launch_mhd_split_momentum_residual(
      background, flux_r, parts_r, flux_z, parts_z, residual, outflow,
      /*stream=*/nullptr, /*cylindrical=*/true,
      /*collocation_order=*/0, order, radial_tables.view());

  // The radial tensor launcher owns both the radial face stress and the
  // cell-volume curvature term. A zero perturbation isolates the static
  // Maxwell tensor, while rho=1 keeps its reduced dynamic terms well-defined.
  quasar::mhd::MhdField2D<Real> state{grid};
  zero_field(state, zero);
  state.rho.copy_from_host(one.data(), size);
  quasar::mhd::launch_mhd_cylindrical_radial_momentum_residual(
      state, background, flux_r, flux_z, residual, outflow,
      /*stream=*/nullptr, kGamma, /*collocation_order=*/0, order,
      /*parts_r=*/nullptr, /*parts_z=*/nullptr, radial_tables.view());
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> radial_rate(size);
  std::vector<Real> axial_rate(size);
  residual.mx.copy_to_host(radial_rate.data(), size);
  residual.my.copy_to_host(axial_rate.data(), size);

  StaticToroidalForceError error;
  for (int j = 0; j < grid.ny; ++j) {
    const long double z_lo =
        static_cast<long double>(grid.origin_y) +
        static_cast<long double>(j) * static_cast<long double>(grid.dy());
    const long double z_hi = z_lo + static_cast<long double>(grid.dy());
    const long double sin_squared =
        uniform_average_sine_squared(z_lo, z_hi, wave);
    const long double sin_cos =
        uniform_average_sine_cosine(z_lo, z_hi, wave);
    for (int i = 0; i < grid.nx; ++i) {
      const long double r_lo =
          static_cast<long double>(grid.origin_x) +
          static_cast<long double>(i) * static_cast<long double>(grid.dx());
      const long double r_hi = r_lo + static_cast<long double>(grid.dx());
      // curl(B0) x B0 = (-2 r sin^2(2 pi z),
      //                  -2 pi r^2 sin(2 pi z) cos(2 pi z), 0).
      const Real exact_radial = static_cast<Real>(
          -2.0L * radial_cell_average_power(r_lo, r_hi, /*power=*/1) *
          sin_squared);
      const Real exact_axial = static_cast<Real>(
          -wave * radial_cell_average_power(r_lo, r_hi, /*power=*/2) *
          sin_cos);
      const std::size_t index = grid.index(i, j);
      error.radial += std::abs(radial_rate[index] - exact_radial);
      error.axial += std::abs(axial_rate[index] - exact_axial);
    }
  }
  const Real cell_count = static_cast<Real>(grid.nx * grid.ny);
  error.radial /= cell_count;
  error.axial /= cell_count;
  return error;
}

quasar::mhd::MhdConfig cylindrical_config(const std::string& reconstruction) {
  quasar::mhd::MhdConfig config;
  config.grid = Grid2D{16, 16, Real{1}, Real{1}, Real{0}, Real{0},
                       /*nghost=*/4};
  config.gamma = kGamma;
  config.geometry = "cylindrical";
  config.reconstruction = reconstruction;
  config.riemann = "hlld";
  config.integrator = "ssprk3";
  config.ct = "fd_ct_christlieb";
  config.positivity = "troubled_cell";
  config.cfl = Real{0.4};
  for (int side = 0; side < 4; ++side) {
    config.boundary.fluid[side] = "outflow";
    config.boundary.field[side] = "outflow";
  }
  config.boundary.fluid[0] = "axis";
  config.boundary.field[0] = "axis";
  config.boundary.fluid[2] = "periodic";
  config.boundary.field[2] = "periodic";
  config.boundary.fluid[3] = "periodic";
  config.boundary.field[3] = "periodic";
  return config;
}

void seed_cylindrical_entropy_wave(quasar::mhd::MhdSolver2D& solver,
                                   const Grid2D& grid) {
  const std::size_t size = grid.storage_size();
  std::vector<Real> rho(size), mx(size, Real{0}), my(size), mz(size, Real{0});
  std::vector<Real> energy(size), bx(size, Real{0}), by(size, Real{0.05});
  std::vector<Real> bz(size, Real{0});
  constexpr Real velocity_z = Real{0.1};
  constexpr Real pressure = Real{1};
  constexpr Real magnetic_energy = Real{0.5} * Real{0.05} * Real{0.05};

  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    const Real z = grid.y_at_cell_center(grid.wrap_j(j));
    const Real density =
        Real{1} + Real{0.05} * std::sin(Real{2} * quasar::pi * z);
    for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
      const std::size_t index = grid.index(i, j);
      rho[index] = density;
      my[index] = density * velocity_z;
      energy[index] = pressure / (kGamma - Real{1}) +
                      Real{0.5} * density * velocity_z * velocity_z +
                      magnetic_energy;
    }
  }

  solver.seed_state("rho", rho);
  solver.seed_state("mx", mx);
  solver.seed_state("my", my);
  solver.seed_state("mz", mz);
  solver.seed_state("energy", energy);
  solver.seed_state("bx", bx);
  solver.seed_state("by", by);
  solver.seed_state("bz", bz);
}

}  // namespace

TEST(MhdCylindricalHighOrder,
     SplitEnergyTensorGaussMatchesDesignDegreeIntegral) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D grid{12, 12, Real{1.5}, Real{1.5}, Real{2}, Real{1},
                    /*nghost=*/4};
  constexpr int target_i = 6;
  constexpr int target_j = 5;
  const long double radial_scale =
      static_cast<long double>(grid.r_at_cell_center(target_i));
  const long double axial_scale =
      static_cast<long double>(grid.y_at_cell_center(target_j));

  for (const int order : {5, 7}) {
    SCOPED_TRACE(order);
    const int factor_degree = (order - 1) / 2;
    const std::size_t size = grid.storage_size();
    std::vector<Real> factor(size);
    for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
      for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
        factor[grid.index(i, j)] =
            separable_uniform_radial_monomial_average(
                grid, i, j, factor_degree, factor_degree, radial_scale,
                axial_scale);
      }
    }

    const Real actual = split_energy_inner_product(
        grid, order, factor, factor, target_i, target_j);
    const Real expected = -separable_monomial_average(
        grid, target_i, target_j, 2 * factor_degree, 2 * factor_degree,
        radial_scale, axial_scale);
    const Real tolerance = Real{4096} * std::numeric_limits<Real>::epsilon() *
                           std::max(Real{1}, std::abs(expected));
    EXPECT_NEAR(actual, expected, tolerance)
        << "the degree-" << (order - 1)
        << " tensor product must be integrated under r dr dz";
  }
}

TEST(MhdCylindricalHighOrder,
     SplitEnergyCovarianceIsBitZeroForEitherUniformFactor) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D grid{12, 12, Real{1.5}, Real{1.5}, Real{2}, Real{1},
                    /*nghost=*/4};
  constexpr int target_i = 6;
  constexpr int target_j = 5;
  const std::size_t size = grid.storage_size();
  const std::vector<Real> uniform(size, Real{2});
  std::vector<Real> varying(size);
  const Real target_z = grid.y_at_cell_center(target_j);
  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    const Real value = Real{1} + grid.y_at_cell_center(j) - target_z;
    for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
      varying[grid.index(i, j)] = value;
    }
  }

  for (const int order : {5, 7}) {
    SCOPED_TRACE(order);
    EXPECT_EQ(split_energy_inner_product(
                  grid, order, uniform, varying, target_i, target_j),
              Real{-2});
    EXPECT_EQ(split_energy_inner_product(
                  grid, order, varying, uniform, target_i, target_j),
              Real{-2});
  }
}

TEST(MhdCylindricalHighOrder,
     SplitEnergyBphiProductUsesAnnularMomentOfFlatStoredFactors) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D grid{12, 12, Real{1.5}, Real{1.5}, Real{2}, Real{1},
                    /*nghost=*/4};
  constexpr int target_i = 6;
  constexpr int target_j = 5;
  const std::size_t size = grid.storage_size();
  const std::vector<Real> constant_b0phi(size, Real{2});
  std::vector<Real> varying_bphi_rate(size);
  const long double radial_scale =
      static_cast<long double>(grid.r_at_cell_center(target_i));
  const long double axial_scale =
      static_cast<long double>(grid.y_at_cell_center(target_j));
  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
      varying_bphi_rate[grid.index(i, j)] =
          separable_uniform_radial_monomial_average(
              grid, i, j, /*radial_power=*/1, /*axial_power=*/0,
              radial_scale, axial_scale);
    }
  }
  const Real expected = Real{-2} * separable_monomial_average(
      grid, target_i, target_j, /*radial_power=*/1, /*axial_power=*/0,
      radial_scale, axial_scale);
  const Real flat_means_product =
      Real{-2} * varying_bphi_rate[grid.index(target_i, target_j)];
  ASSERT_GT(std::abs(expected - flat_means_product),
            Real{1024} * std::numeric_limits<Real>::epsilon());

  for (const int order : {5, 7}) {
    SCOPED_TRACE(order);
    const Real constant_background = split_energy_inner_product(
        grid, order, constant_b0phi, varying_bphi_rate,
        target_i, target_j);
    const Real constant_rate = split_energy_inner_product(
        grid, order, varying_bphi_rate, constant_b0phi,
        target_i, target_j);
    EXPECT_NEAR(constant_background, expected,
                Real{256} * std::numeric_limits<Real>::epsilon());
    EXPECT_NEAR(constant_rate, expected,
                Real{256} * std::numeric_limits<Real>::epsilon());
    EXPECT_GT(std::abs(constant_background - flat_means_product),
              Real{0.5} * std::abs(expected - flat_means_product));
    EXPECT_GT(std::abs(constant_rate - flat_means_product),
              Real{0.5} * std::abs(expected - flat_means_product));
  }
}

TEST(MhdCylindricalHighOrder,
     CurrentCarryingStaticMaxwellStressKeepsMpOrder) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const StaticToroidalForceError mp5_coarse =
      static_toroidal_force_error(/*order=*/5, /*resolution=*/12);
  const StaticToroidalForceError mp5_fine =
      static_toroidal_force_error(/*order=*/5, /*resolution=*/24);
  const StaticToroidalForceError mp7_coarse =
      static_toroidal_force_error(/*order=*/7, /*resolution=*/10);
  const StaticToroidalForceError mp7_fine =
      static_toroidal_force_error(/*order=*/7, /*resolution=*/20);

  const auto slope = [](Real coarse, Real fine) {
    EXPECT_GT(coarse, Real{0});
    EXPECT_GT(fine, Real{0});
    return std::log2(coarse / fine);
  };
  EXPECT_GT(slope(mp5_coarse.radial, mp5_fine.radial), Real{4.0})
      << "radial errors " << mp5_coarse.radial << " -> "
      << mp5_fine.radial;
  EXPECT_GT(slope(mp5_coarse.axial, mp5_fine.axial), Real{4.0})
      << "axial errors " << mp5_coarse.axial << " -> "
      << mp5_fine.axial;
  EXPECT_GT(slope(mp7_coarse.radial, mp7_fine.radial), Real{5.8})
      << "radial errors " << mp7_coarse.radial << " -> "
      << mp7_fine.radial;
  EXPECT_GT(slope(mp7_coarse.axial, mp7_fine.axial), Real{5.8})
      << "axial errors " << mp7_coarse.axial << " -> "
      << mp7_fine.axial;
}

TEST(MhdCylindricalHighOrder, DynamicRadialTensorKeepsMpOrder) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Real mp5_coarse = dynamic_radial_force_error(5, 12);
  const Real mp5_fine = dynamic_radial_force_error(5, 24);
  const Real mp7_coarse = dynamic_radial_force_error(7, 10);
  const Real mp7_fine = dynamic_radial_force_error(7, 20);
  const auto slope = [](Real coarse, Real fine) {
    EXPECT_GT(coarse, Real{0});
    EXPECT_GT(fine, Real{0});
    return std::log2(coarse / fine);
  };
  const Real mp5_order = slope(mp5_coarse, mp5_fine);
  const Real mp7_order = slope(mp7_coarse, mp7_fine);
  ::testing::Test::RecordProperty("dynamic_mp5_coarse_error", mp5_coarse);
  ::testing::Test::RecordProperty("dynamic_mp5_fine_error", mp5_fine);
  ::testing::Test::RecordProperty("dynamic_mp5_observed_order", mp5_order);
  ::testing::Test::RecordProperty("dynamic_mp7_coarse_error", mp7_coarse);
  ::testing::Test::RecordProperty("dynamic_mp7_fine_error", mp7_fine);
  ::testing::Test::RecordProperty("dynamic_mp7_observed_order", mp7_order);
  EXPECT_GT(mp5_order, Real{4.0})
      << "radial errors " << mp5_coarse << " -> " << mp5_fine;
  EXPECT_GT(mp7_order, Real{5.8})
      << "radial errors " << mp7_coarse << " -> " << mp7_fine;
}

TEST(MhdCylindricalHighOrder,
     InactiveMpTensorExpansionAcceptsExactZero) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // On this annulus, a constant unit gas pressure gives F_rr=T_phiphi=1.
  // The ordinary face difference is zero and the two metric half-faces cancel
  // the tensor source exactly.  Exact zero is an important accepted fast-path
  // result: it must not be mistaken for an underflow requiring fallback.
  const Grid2D grid{/*nx=*/4, /*ny=*/3, /*lx=*/Real{8}, /*ly=*/Real{3},
                    /*origin_x=*/Real{15}, /*origin_y=*/Real{0},
                    /*nghost=*/4};
  const std::size_t size = grid.storage_size();
  HostRadialState state(size);
  std::fill(state.rho.begin(), state.rho.end(), Real{1});
  std::fill(state.energy.begin(), state.energy.end(), Real{1});
  const std::vector<Real> radial_flux(size, Real{1});
  const std::vector<Real> axial_flux(size, Real{0});

  for (const int order : {5, 7}) {
    SCOPED_TRACE(order);
    const std::vector<Real> residual = inactive_radial_residual(
        grid, order, /*gamma=*/Real{2}, state, radial_flux, axial_flux);
    for (int j = 0; j < grid.ny; ++j) {
      for (int i = 0; i < grid.nx; ++i) {
        EXPECT_EQ(residual[grid.index(i, j)], Real{0})
            << "i=" << i << " j=" << j;
      }
    }
  }
}

TEST(MhdCylindricalHighOrder,
     InactiveMpSmoothTensorMatchesExactStandaloneReduction) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D grid{/*nx=*/4, /*ny=*/3, /*lx=*/Real{8}, /*ly=*/Real{3},
                    /*origin_x=*/Real{15}, /*origin_y=*/Real{0},
                    /*nghost=*/4};
  const std::size_t size = grid.storage_size();
  HostRadialState state(size);
  std::fill(state.rho.begin(), state.rho.end(), Real{1});
  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
      const long double r_lo = static_cast<long double>(grid.r_at_edge(i));
      const long double r_hi =
          static_cast<long double>(grid.r_at_edge(i + 1));
      // Store the exact annular average of the smooth pressure p(r)=r.
      // The MP tensor rule recovers p at its points, integrates it under the
      // uniform curvature measure, and therefore gives <p>/r_center = 1.
      state.energy[grid.index(i, j)] = static_cast<Real>(
          radial_cell_average_power(r_lo, r_hi, /*power=*/1));
    }
  }
  const std::vector<Real> zero_flux(size, Real{0});

  for (const int order : {5, 7}) {
    SCOPED_TRACE(order);
    const std::vector<Real> fused = inactive_radial_residual(
        grid, order, /*gamma=*/Real{2}, state, zero_flux, zero_flux);
    const std::vector<Real> standalone = inactive_tensor_source(
        grid, order, /*gamma=*/Real{2}, state);
    for (int j = 0; j < grid.ny; ++j) {
      for (int i = 0; i < grid.nx; ++i) {
        const std::size_t index = grid.index(i, j);
        EXPECT_EQ(fused[index], standalone[index])
            << "i=" << i << " j=" << j;
        EXPECT_NEAR(fused[index], Real{1}, Real{2e-13})
            << "i=" << i << " j=" << j;
      }
    }
  }
}

TEST(MhdCylindricalHighOrder,
     InactiveMp7WideExpansionFallsBackToExactGolden) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D grid{/*nx=*/4, /*ny=*/3, /*lx=*/Real{8}, /*ly=*/Real{3},
                    /*origin_x=*/Real{15}, /*origin_y=*/Real{0},
                    /*nghost=*/4};
  const std::size_t size = grid.storage_size();
  HostRadialState state(size);
  std::fill(state.rho.begin(), state.rho.end(), Real{1});
  const Real pressure = std::scalbn(Real{1}, 404);
  std::fill(state.energy.begin(), state.energy.end(), pressure);
  std::vector<Real> radial_flux(size, Real{0});
  std::vector<Real> axial_flux(size, Real{0});
  constexpr int target_i = 0;
  constexpr int target_j = 1;
  const std::size_t k = grid.index(target_i, target_j);
  const std::size_t kr = grid.index(target_i + 1, target_j);
  const std::size_t kz = grid.index(target_i, target_j + 1);
  ASSERT_EQ(grid.dx(), Real{2});
  ASSERT_EQ(grid.dy(), Real{1});
  ASSERT_EQ(grid.r_at_cell_center(target_i), Real{16});

  radial_flux[k] = std::scalbn(Real{1}, 0);
  radial_flux[kr] = std::scalbn(Real{1}, 100);
  axial_flux[k] = std::scalbn(Real{1}, 200);
  axial_flux[kz] = std::scalbn(Real{1}, 300);

  // Before the tensor pressure is appended, the exact face sum contains four
  // non-overlapping bands:
  //
  //   15/32, -17*2^95, +2^200, -2^300.
  //
  // The pressure contribution is +2^400, a fifth band, so a four-component
  // certified expansion must route this cell through the exact fallback.  All
  // lower bands lie well below half an ulp of 2^400, making the final RNE
  // golden exactly 2^400.
  const std::vector<Real> residual = inactive_radial_residual(
      grid, /*order=*/7, /*gamma=*/Real{2}, state,
      radial_flux, axial_flux);
  const Real expected = std::scalbn(Real{1}, 400);
  ASSERT_TRUE(std::isfinite(residual[k]));
  EXPECT_EQ(residual[k], expected);
}

TEST(MhdCylindricalHighOrder,
     InactiveMp7SubnormalFinishRoutesToExactFallback) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D grid{/*nx=*/4, /*ny=*/3, /*lx=*/Real{8}, /*ly=*/Real{3},
                    /*origin_x=*/Real{15}, /*origin_y=*/Real{0},
                    /*nghost=*/4};
  const std::size_t size = grid.storage_size();
  HostRadialState state(size);
  std::fill(state.rho.begin(), state.rho.end(), Real{1});
  const Real pressure = std::scalbn(Real{1}, -1018);
  std::fill(state.energy.begin(), state.energy.end(), pressure);
  const std::vector<Real> radial_flux(size, Real{0});
  std::vector<Real> axial_flux(size, Real{0});
  constexpr int target_i = 0;
  constexpr int target_j = 1;
  const std::size_t k = grid.index(target_i, target_j);
  const std::size_t kz = grid.index(target_i, target_j + 1);
  ASSERT_EQ(grid.r_at_cell_center(target_i), Real{16});

  // p/r is exactly the smallest normal.  Cancel it with the high axial face
  // while the low axial face contributes one 2^-1074 quantum.  The exact
  // residual is the smallest subnormal, which the compact finish deliberately
  // delegates to the radix path to avoid double rounding.
  axial_flux[kz] = std::numeric_limits<Real>::min();
  axial_flux[k] = std::numeric_limits<Real>::denorm_min();
  const std::vector<Real> residual = inactive_radial_residual(
      grid, /*order=*/7, /*gamma=*/Real{2}, state,
      radial_flux, axial_flux);
  EXPECT_EQ(residual[k], std::numeric_limits<Real>::denorm_min());
}

TEST(MhdCylindricalHighOrder,
     DynamicRadialTensorConditionsDominantCurlFreeCrossStress) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D grid{/*nx=*/8, /*ny=*/4, Real{1}, Real{1},
                    Real{1}, Real{0}, /*nghost=*/4};
  const std::size_t size = grid.storage_size();
  const std::vector<Real> zero(size, Real{0});
  const std::vector<Real> one(size, Real{1});
  const std::vector<int> zero_flag(size, 0);
  std::vector<Real> energy(size);
  std::vector<Real> radial_material_flux(size);
  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
      const long double r_lo =
          static_cast<long double>(grid.origin_x) +
          static_cast<long double>(i) * static_cast<long double>(grid.dx());
      const long double r_hi = r_lo + static_cast<long double>(grid.dx());
      const long double pressure_average =
          2.0L + radial_cell_average_power(r_lo, r_hi, /*power=*/1);
      const std::size_t index = grid.index(i, j);
      energy[index] = static_cast<Real>(
          pressure_average / (static_cast<long double>(kGamma) - 1.0L) +
          0.5L);
      radial_material_flux[index] =
          static_cast<Real>(2.0L + r_lo + 0.5L);
    }
  }

  quasar::mhd::MhdField2D<Real> state{grid};
  quasar::mhd::MhdField2D<Real> flux_r{grid};
  quasar::mhd::MhdField2D<Real> flux_z{grid};
  quasar::mhd::MhdField2D<Real> residual{grid};
  zero_field(state, zero);
  zero_field(flux_r, zero);
  zero_field(flux_z, zero);
  zero_field(residual, zero);
  state.rho.copy_from_host(one.data(), size);
  state.energy.copy_from_host(energy.data(), size);
  state.by_face.copy_from_host(one.data(), size);
  flux_r.mx.copy_from_host(radial_material_flux.data(), size);

  const Real background_scale = std::scalbn(Real{1}, 600);
  const std::vector<Real> dominant(size, background_scale);
  quasar::mhd::MhdBackgroundField<Real> background{grid};
  background.active = true;
  background.globally_curl_free = true;
  background.b0x_face.copy_from_host(zero.data(), size);
  background.b0y_face.copy_from_host(dominant.data(), size);
  background.b0z_cell.copy_from_host(zero.data(), size);

  quasar::mhd::MhdMomentumFluxParts2D<Real> parts_r{grid};
  quasar::mhd::MhdMomentumFluxParts2D<Real> parts_z{grid};
  zero_momentum_flux_parts(parts_r, zero, zero_flag);
  zero_momentum_flux_parts(parts_z, zero, zero_flag);
  // For an identical-state radial Riemann problem, cross_b is the perturbation
  // field. Keep it in the factorized fallback slot so no O(B0*b) face stress is
  // ever rounded into the material pressure flux.
  parts_r.cross_b_point[0].y.copy_from_host(one.data(), size);

  quasar::numerics::RadialTables radial_tables{grid, /*scheme_order=*/7};
  quasar::mhd::BoundaryFlags4 outflow{};
  for (int side = 0; side < 4; ++side) outflow.side[side] = 1;
  quasar::mhd::launch_mhd_cylindrical_radial_momentum_residual(
      state, background, flux_r, flux_z, residual, outflow,
      /*stream=*/nullptr, kGamma, /*collocation_order=*/0,
      /*scheme_order=*/7, &parts_r, &parts_z, radial_tables.view());
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> radial_rate(size);
  residual.mx.copy_to_host(radial_rate.data(), size);
  for (int j = 0; j < grid.ny; ++j) {
    for (int i = 0; i < grid.nx; ++i) {
      const Real actual = radial_rate[grid.index(i, j)];
      ASSERT_TRUE(std::isfinite(actual));
      EXPECT_NEAR(actual, Real{-1}, Real{2e-12})
          << "i=" << i << " j=" << j;
    }
  }
}

TEST(MhdCylindricalHighOrder,
     DynamicRadialTensorUsesWaveAndPointCrossMetricRecords) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D grid{/*nx=*/8, /*ny=*/4, Real{1}, Real{1},
                    Real{1}, Real{0}, /*nghost=*/4};
  const std::size_t size = grid.storage_size();
  const std::vector<Real> zero(size, Real{0});
  const std::vector<Real> one(size, Real{1});
  const std::vector<int> zero_flag(size, 0);
  const std::vector<int> valid_flag(size, 1);
  constexpr Real stored_energy = Real{1.5};
  const Real gas_pressure = (kGamma - Real{1}) * stored_energy;
  const std::vector<Real> energy(size, stored_energy);
  const std::vector<Real> material_flux(size, gas_pressure);
  constexpr Real background_z = Real{2};
  const std::vector<Real> background_axial(size, background_z);
  constexpr Real wave_flux = Real{2};
  constexpr Real cross_sample = Real{3};
  quasar::numerics::RadialTables radial_tables{grid, /*scheme_order=*/7};
  quasar::mhd::BoundaryFlags4 outflow{};
  for (int side = 0; side < 4; ++side) outflow.side[side] = 1;

  const auto run = [&](Real radial_wave, bool point_cross) {
    quasar::mhd::MhdField2D<Real> state{grid};
    quasar::mhd::MhdField2D<Real> flux_r{grid};
    quasar::mhd::MhdField2D<Real> flux_z{grid};
    quasar::mhd::MhdField2D<Real> residual{grid};
    zero_field(state, zero);
    zero_field(flux_r, zero);
    zero_field(flux_z, zero);
    zero_field(residual, zero);
    state.rho.copy_from_host(one.data(), size);
    state.energy.copy_from_host(energy.data(), size);
    flux_r.mx.copy_from_host(material_flux.data(), size);

    quasar::mhd::MhdBackgroundField<Real> background{grid};
    background.active = true;
    background.globally_curl_free = true;
    background.b0x_face.copy_from_host(zero.data(), size);
    background.b0y_face.copy_from_host(background_axial.data(), size);
    background.b0z_cell.copy_from_host(zero.data(), size);

    quasar::mhd::MhdMomentumFluxParts2D<Real> parts_r{grid};
    quasar::mhd::MhdMomentumFluxParts2D<Real> parts_z{grid};
    zero_momentum_flux_parts(parts_r, zero, zero_flag);
    zero_momentum_flux_parts(parts_z, zero, zero_flag);
    const std::vector<Real> wave(size, radial_wave);
    parts_r.wave_x.copy_from_host(wave.data(), size);
    if (point_cross) {
      const std::vector<Real> cross(size, cross_sample);
      parts_r.cross_b_point[0].y.copy_from_host(cross.data(), size);
      parts_r.quadrature_valid.copy_from_host(valid_flag.data(), size);
    }

    quasar::mhd::launch_mhd_cylindrical_radial_momentum_residual(
        state, background, flux_r, flux_z, residual, outflow,
        /*stream=*/nullptr, kGamma, /*collocation_order=*/0,
        /*scheme_order=*/7, &parts_r, &parts_z, radial_tables.view());
    quasar::backend::device_synchronize(nullptr);

    std::vector<Real> radial_rate(size);
    residual.mx.copy_to_host(radial_rate.data(), size);
    return radial_rate;
  };

  const std::vector<Real> wave_rate = run(wave_flux, /*point_cross=*/false);
  const std::vector<Real> cross_rate = run(Real{0}, /*point_cross=*/true);
  const Real point_cross_flux =
      background_z * cross_sample *
      quasar::numerics::kMp7TransverseGaussWeights[0];
  for (int j = 0; j < grid.ny; ++j) {
    for (int i = 0; i < grid.nx; ++i) {
      const Real radius = grid.r_at_cell_center(i);
      const Real expected_wave = -wave_flux / radius;
      const Real expected_cross = -point_cross_flux / radius;
      const Real wave_tolerance =
          Real{512} * std::numeric_limits<Real>::epsilon() *
          std::max(Real{1}, std::abs(expected_wave));
      const Real cross_tolerance =
          Real{512} * std::numeric_limits<Real>::epsilon() *
          std::max(Real{1}, std::abs(expected_cross));
      EXPECT_NEAR(wave_rate[grid.index(i, j)], expected_wave, wave_tolerance)
          << "wave metric at i=" << i << " j=" << j;
      EXPECT_NEAR(cross_rate[grid.index(i, j)], expected_cross,
                  cross_tolerance)
          << "point cross metric at i=" << i << " j=" << j;
    }
  }
}

TEST(MhdCylindricalHighOrder,
     ConstantValidPointFamiliesCancelAcrossFacesAndTensors) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D grid{/*nx=*/8, /*ny=*/4, Real{1}, Real{1},
                    Real{1}, Real{0}, /*nghost=*/4};
  const std::size_t size = grid.storage_size();
  const std::vector<Real> zero(size, Real{0});
  const std::vector<Real> one(size, Real{1});
  const std::vector<int> zero_flag(size, 0);
  const std::vector<int> valid_flag(size, 1);
  quasar::numerics::RadialTables radial_tables{grid, /*scheme_order=*/7};
  quasar::mhd::BoundaryFlags4 outflow{};
  for (int side = 0; side < 4; ++side) outflow.side[side] = 1;

  const auto run = [&](bool static_background) {
    quasar::mhd::MhdField2D<Real> state{grid};
    quasar::mhd::MhdField2D<Real> flux_r{grid};
    quasar::mhd::MhdField2D<Real> flux_z{grid};
    quasar::mhd::MhdField2D<Real> residual{grid};
    zero_field(state, zero);
    zero_field(flux_r, zero);
    zero_field(flux_z, zero);
    zero_field(residual, zero);
    state.rho.copy_from_host(one.data(), size);

    quasar::mhd::MhdBackgroundField<Real> background{grid};
    background.active = true;
    background.globally_curl_free = !static_background;
    background.b0x_face.copy_from_host(zero.data(), size);
    background.b0y_face.copy_from_host(zero.data(), size);
    background.b0z_cell.copy_from_host(zero.data(), size);

    quasar::mhd::MhdMomentumFluxParts2D<Real> parts_r{grid};
    quasar::mhd::MhdMomentumFluxParts2D<Real> parts_z{grid};
    zero_momentum_flux_parts(parts_r, zero, zero_flag);
    zero_momentum_flux_parts(parts_z, zero, zero_flag);
    parts_r.quadrature_valid.copy_from_host(valid_flag.data(), size);
    parts_z.quadrature_valid.copy_from_host(valid_flag.data(), size);

    if (!static_background) {
      // The exact MP7 weights sum to 1+2^-54 even though an ordinary
      // binary64 reduction rounds them to one.  These valid face records must
      // therefore use the same constant-family semantics as the tensor rule:
      // B0_z*b_z cancels exactly despite its 2^600 scale.
      const std::vector<Real> dominant(
          size, std::scalbn(Real{1}, 600));
      const std::vector<Real> energy(size, Real{1.25});
      state.energy.copy_from_host(energy.data(), size);
      state.by_face.copy_from_host(one.data(), size);
      flux_r.mx.copy_from_host(one.data(), size);
      background.b0y_face.copy_from_host(dominant.data(), size);
      for (auto* parts : {&parts_r, &parts_z}) {
        for (auto& point : parts->cross_b_point) {
          point.y.copy_from_host(one.data(), size);
        }
      }
    } else {
      // A constant axial background is force-free.  A unit material flux
      // separately cancels the unit tensor pressure, so any static axial
      // face/tensor weight mismatch survives alone at roughly 2^945.
      const Real background_scale = std::scalbn(Real{1}, 500);
      const std::vector<Real> dominant(size, background_scale);
      const std::vector<Real> energy(
          size, Real{1} / (kGamma - Real{1}));
      const std::vector<Real> material_flux(size, Real{1});
      state.energy.copy_from_host(energy.data(), size);
      flux_r.mx.copy_from_host(material_flux.data(), size);
      background.b0y_face.copy_from_host(dominant.data(), size);
    }

    quasar::mhd::launch_mhd_cylindrical_radial_momentum_residual(
        state, background, flux_r, flux_z, residual, outflow,
        /*stream=*/nullptr, kGamma, /*collocation_order=*/0,
        /*scheme_order=*/7, &parts_r, &parts_z, radial_tables.view());
    quasar::backend::device_synchronize(nullptr);
    std::vector<Real> radial_rate(size);
    residual.mx.copy_to_host(radial_rate.data(), size);
    return radial_rate;
  };

  const std::vector<Real> dynamic_rate = run(/*static_background=*/false);
  const std::vector<Real> static_rate = run(/*static_background=*/true);
  for (int j = 0; j < grid.ny; ++j) {
    for (int i = 0; i < grid.nx; ++i) {
      const std::size_t index = grid.index(i, j);
      EXPECT_EQ(dynamic_rate[index], Real{0})
          << "dynamic i=" << i << " j=" << j;
      EXPECT_EQ(static_rate[index], Real{0})
          << "static i=" << i << " j=" << j;
    }
  }
}

TEST(MhdCylindricalHighOrder,
     MaximumRadialMomentumAccumulatorPathStaysFinite) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D grid{/*nx=*/8, /*ny=*/4, Real{1}, Real{1},
                    Real{1}, Real{0}, /*nghost=*/4};
  const std::size_t size = grid.storage_size();
  const std::vector<Real> zero(size, Real{0});
  const std::vector<int> zero_flag(size, 0);
  const std::vector<int> valid_flag(size, 1);

  // Force the proven maximum path rather than allowing any constant-family
  // shortcut.  Every state and background component varies in both r and z;
  // rho, m_phi, and pressure are all nonconstant; both directions carry
  // nonzero material/wave records; and every one of the four valid point-cross
  // records is distinct.  The resulting MP7 call count is therefore genuinely
  // 12 + 64 + 128 + 56 + 48 = 308.
  std::vector<Real> rho(size);
  std::vector<Real> mr(size);
  std::vector<Real> mz(size);
  std::vector<Real> mphi(size);
  std::vector<Real> energy(size);
  std::vector<Real> br(size);
  std::vector<Real> bz(size);
  std::vector<Real> bphi(size);
  std::vector<Real> b0r(size);
  std::vector<Real> b0z(size);
  std::vector<Real> b0phi(size);
  std::vector<Real> radial_flux(size);
  std::vector<Real> axial_flux(size);
  std::vector<Real> radial_wave(size);
  std::vector<Real> axial_wave(size);
  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
      const Real x = static_cast<Real>(i) + Real{0.5};
      const Real y = static_cast<Real>(j) + Real{0.5};
      const std::size_t index = grid.index(i, j);
      rho[index] = Real{2} + Real{0.002} * x + Real{0.003} * y;
      mr[index] = Real{0.11} + Real{0.001} * x - Real{0.0007} * y;
      mz[index] = Real{-0.09} + Real{0.0008} * x + Real{0.0011} * y;
      mphi[index] = Real{0.14} - Real{0.0013} * x + Real{0.0009} * y;
      energy[index] = Real{20} + Real{0.02} * x + Real{0.03} * y;
      br[index] = Real{0.21} + Real{0.0017} * x + Real{0.0012} * y;
      bz[index] = Real{-0.17} + Real{0.0014} * x - Real{0.0015} * y;
      bphi[index] = Real{0.13} - Real{0.0011} * x + Real{0.0018} * y;
      b0r[index] = Real{-0.31} + Real{0.0021} * x + Real{0.0016} * y;
      b0z[index] = Real{0.27} - Real{0.0019} * x + Real{0.0022} * y;
      b0phi[index] = Real{0.19} + Real{0.0023} * x - Real{0.0014} * y;
      radial_flux[index] = Real{1.1} + Real{0.004} * x + Real{0.003} * y;
      axial_flux[index] = Real{-0.8} + Real{0.002} * x - Real{0.005} * y;
      radial_wave[index] = Real{0.23} - Real{0.001} * x + Real{0.002} * y;
      axial_wave[index] = Real{-0.16} + Real{0.003} * x + Real{0.001} * y;
    }
  }

  quasar::mhd::MhdField2D<Real> state{grid};
  quasar::mhd::MhdField2D<Real> flux_r{grid};
  quasar::mhd::MhdField2D<Real> flux_z{grid};
  quasar::mhd::MhdField2D<Real> residual{grid};
  zero_field(state, zero);
  zero_field(flux_r, zero);
  zero_field(flux_z, zero);
  zero_field(residual, zero);
  state.rho.copy_from_host(rho.data(), size);
  state.mx.copy_from_host(mr.data(), size);
  state.my.copy_from_host(mz.data(), size);
  state.mz.copy_from_host(mphi.data(), size);
  state.energy.copy_from_host(energy.data(), size);
  state.bx_face.copy_from_host(br.data(), size);
  state.by_face.copy_from_host(bz.data(), size);
  state.bz_cell.copy_from_host(bphi.data(), size);
  flux_r.mx.copy_from_host(radial_flux.data(), size);
  flux_z.mx.copy_from_host(axial_flux.data(), size);

  quasar::mhd::MhdBackgroundField<Real> background{grid};
  background.active = true;
  background.globally_curl_free = false;
  background.b0x_face.copy_from_host(b0r.data(), size);
  background.b0y_face.copy_from_host(b0z.data(), size);
  background.b0z_cell.copy_from_host(b0phi.data(), size);

  quasar::mhd::MhdMomentumFluxParts2D<Real> parts_r{grid};
  quasar::mhd::MhdMomentumFluxParts2D<Real> parts_z{grid};
  zero_momentum_flux_parts(parts_r, zero, zero_flag);
  zero_momentum_flux_parts(parts_z, zero, zero_flag);
  parts_r.wave_x.copy_from_host(radial_wave.data(), size);
  parts_z.wave_x.copy_from_host(axial_wave.data(), size);
  for (int q = 0; q < 4; ++q) {
    std::vector<Real> cross_r(size);
    std::vector<Real> cross_z(size);
    std::vector<Real> cross_phi(size);
    for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
      for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
        const Real x = static_cast<Real>(i) + Real{0.5};
        const Real y = static_cast<Real>(j) + Real{0.5};
        const std::size_t index = grid.index(i, j);
        const Real point = static_cast<Real>(q + 1);
        cross_r[index] = Real{0.07} * point + Real{0.0004} * x +
                         Real{0.0003} * y;
        cross_z[index] = Real{-0.05} * point + Real{0.0002} * x -
                         Real{0.0005} * y;
        cross_phi[index] = Real{0.04} * point - Real{0.0006} * x +
                           Real{0.0002} * y;
      }
    }
    for (auto* parts : {&parts_r, &parts_z}) {
      parts->cross_b_point[q].x.copy_from_host(cross_r.data(), size);
      parts->cross_b_point[q].y.copy_from_host(cross_z.data(), size);
      parts->cross_b_point[q].z.copy_from_host(cross_phi.data(), size);
    }
  }
  for (auto* parts : {&parts_r, &parts_z}) {
    parts->quadrature_valid.copy_from_host(valid_flag.data(), size);
  }

  quasar::numerics::RadialTables radial_tables{
      grid, /*scheme_order=*/7};
  quasar::mhd::BoundaryFlags4 outflow{};
  for (int side = 0; side < 4; ++side) outflow.side[side] = 1;
  quasar::mhd::launch_mhd_cylindrical_radial_momentum_residual(
      state, background, flux_r, flux_z, residual, outflow,
      /*stream=*/nullptr, kGamma, /*collocation_order=*/0,
      /*scheme_order=*/7, &parts_r, &parts_z, radial_tables.view());
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> radial_rate(size);
  residual.mx.copy_to_host(radial_rate.data(), size);
  for (int j = 0; j < grid.ny; ++j) {
    for (int i = 0; i < grid.nx; ++i) {
      const Real actual = radial_rate[grid.index(i, j)];
      ASSERT_TRUE(std::isfinite(actual)) << "i=" << i << " j=" << j;
    }
  }
}

TEST(MhdCylindricalHighOrder, CylindricalMp7CflIsStable) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const auto config = cylindrical_config("mp7");
  quasar::mhd::MhdSolver2D solver{config};
  seed_cylindrical_entropy_wave(solver, config.grid);

  for (int step = 0; step < 200; ++step) {
    const Real dt = solver.cfl_limit();
    ASSERT_TRUE(std::isfinite(dt));
    ASSERT_GT(dt, Real{0});
    ASSERT_NO_THROW(solver.step_unchecked(dt)) << "step=" << step;
    ASSERT_EQ(solver.last_positivity_substeps(), 1) << "step=" << step;
  }

  const auto density = solver.state_component_to_host("rho");
  Real minimum_density = std::numeric_limits<Real>::infinity();
  Real maximum_density = -std::numeric_limits<Real>::infinity();
  for (int j = 0; j < config.grid.ny; ++j) {
    for (int i = 0; i < config.grid.nx; ++i) {
      const Real value = density[config.grid.index(i, j)];
      ASSERT_TRUE(std::isfinite(value));
      minimum_density = std::min(minimum_density, value);
      maximum_density = std::max(maximum_density, value);
    }
  }
  EXPECT_GT(minimum_density, Real{0.8});
  EXPECT_LT(maximum_density, Real{1.2});
}

TEST(MhdCylindricalHighOrder, CflIsNotLargerThanMuscl) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const auto mp7_config = cylindrical_config("mp7");
  const auto muscl_config = cylindrical_config("muscl_minmod");
  quasar::mhd::MhdSolver2D mp7{mp7_config};
  quasar::mhd::MhdSolver2D muscl{muscl_config};
  seed_cylindrical_entropy_wave(mp7, mp7_config.grid);
  seed_cylindrical_entropy_wave(muscl, muscl_config.grid);

  const Real mp7_limit = mp7.cfl_limit();
  const Real muscl_limit = muscl.cfl_limit();
  ASSERT_TRUE(std::isfinite(mp7_limit));
  ASSERT_TRUE(std::isfinite(muscl_limit));
  ASSERT_GT(mp7_limit, Real{0});
  ASSERT_GT(muscl_limit, Real{0});
  EXPECT_LE(mp7_limit, muscl_limit);
}
