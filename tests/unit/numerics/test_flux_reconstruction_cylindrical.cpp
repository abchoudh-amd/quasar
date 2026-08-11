// Focused device tests for equation-native cylindrical MP5/MP7 reconstruction.
//
// Mass-like variables are annular averages under r dr, azimuthal momentum uses
// r^2 dr, and toroidal induction uses dr. These tests seed exact averages under
// the selected component measure and measure its matching conservative face
// difference. A Cartesian-coefficient negative control is part of the
// contract: without it, an interface-only accuracy probe can pass while
// silently using the wrong finite-volume measure.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/numerics/interface_states.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/numerics/radial_tables.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::numerics::MhdInterfaceStates;
using quasar::numerics::RadialTables;
using quasar::numerics::RadialTablesView;

constexpr Real kGamma = Real{5} / Real{3};
constexpr Real kPressure = Real{3};
constexpr Real kVr = Real{0.35};
constexpr Real kBx = Real{0.7};
constexpr Real kBz = Real{0.1};
constexpr Real kAnnulusInnerRadius = Real{1};
constexpr Real kAnnulusWidth = Real{1};

enum class InterfaceState { left, right };

long double two_pi_long_double() {
  return 2.0L * std::acos(-1.0L);
}

long double weighted_sine_antiderivative(long double radius) {
  const long double wave_number = two_pi_long_double();
  return (std::sin(wave_number * radius) -
          wave_number * radius * std::cos(wave_number * radius)) /
         (wave_number * wave_number);
}

Real smooth_density(Real radius) {
  return Real{2} + Real{0.5} *
                       std::sin(Real{2} * quasar::pi_v<Real> * radius);
}

// Exact average of rho(r)=2+0.5*sin(2*pi*r) under the cylindrical measure
// r dr.  All integration arithmetic stays in long double so the staged data
// cannot become the limiting error in the MP7 probe.
Real smooth_ring_average(Real lower_edge, Real upper_edge) {
  const long double lo = static_cast<long double>(lower_edge);
  const long double hi = static_cast<long double>(upper_edge);
  const long double ring_measure = (hi * hi - lo * lo) / 2.0L;
  const long double sine_integral =
      weighted_sine_antiderivative(hi) -
      weighted_sine_antiderivative(lo);
  return static_cast<Real>(2.0L + 0.5L * sine_integral / ring_measure);
}

Real smooth_cartesian_average(Real lower_edge, Real upper_edge) {
  const long double lo = static_cast<long double>(lower_edge);
  const long double hi = static_cast<long double>(upper_edge);
  const long double wave_number = two_pi_long_double();
  const long double sine_integral =
      (std::cos(wave_number * lo) - std::cos(wave_number * hi)) /
      wave_number;
  return static_cast<Real>(2.0L + 0.5L * sine_integral / (hi - lo));
}

long double smooth_component_value(long double radius) {
  return 0.04L + 0.01L * std::sin(two_pi_long_double() * radius);
}

long double uniform_sine_antiderivative(long double radius) {
  const long double wave = two_pi_long_double();
  return -std::cos(wave * radius) / wave;
}

long double angular_sine_antiderivative(long double radius) {
  const long double wave = two_pi_long_double();
  return -radius * radius * std::cos(wave * radius) / wave +
         2.0L * radius * std::sin(wave * radius) / (wave * wave) +
         2.0L * std::cos(wave * radius) / (wave * wave * wave);
}

Real evolved_component_average(Real lower_edge, Real upper_edge,
                               bool angular_momentum) {
  const long double lo = static_cast<long double>(lower_edge);
  const long double hi = static_cast<long double>(upper_edge);
  if (!angular_momentum) {
    return static_cast<Real>(
        0.04L + 0.01L *
                    (uniform_sine_antiderivative(hi) -
                     uniform_sine_antiderivative(lo)) /
                    (hi - lo));
  }
  const long double measure = (hi * hi * hi - lo * lo * lo) / 3.0L;
  return static_cast<Real>(
      0.04L + 0.01L *
                  (angular_sine_antiderivative(hi) -
                   angular_sine_antiderivative(lo)) /
                  measure);
}

Real quadratic_average_under_measure(
    Real lower_edge, Real upper_edge, int measure_power,
    long double c0, long double c1, long double c2) {
  const long double lo = static_cast<long double>(lower_edge);
  const long double hi = static_cast<long double>(upper_edge);
  const auto integral_power = [lo, hi](int power) {
    return (std::pow(hi, power + 1) - std::pow(lo, power + 1)) /
           static_cast<long double>(power + 1);
  };
  const long double measure = integral_power(measure_power);
  const long double integral =
      c0 * integral_power(measure_power) +
      c1 * integral_power(measure_power + 1) +
      c2 * integral_power(measure_power + 2);
  return static_cast<Real>(integral / measure);
}

void fill_field(quasar::mhd::MhdField2D<Real>& field,
                const std::vector<Real>& values) {
  field.rho.copy_from_host(values.data(), values.size());
  field.mx.copy_from_host(values.data(), values.size());
  field.my.copy_from_host(values.data(), values.size());
  field.mz.copy_from_host(values.data(), values.size());
  field.energy.copy_from_host(values.data(), values.size());
  field.bx_face.copy_from_host(values.data(), values.size());
  field.by_face.copy_from_host(values.data(), values.size());
  field.bz_cell.copy_from_host(values.data(), values.size());
}

Real evolved_component_residual_error(int scheme_order, int nx,
                                      bool angular_momentum) {
  constexpr int nghost = 4;
  const Grid2D grid{nx, 2, kAnnulusWidth, Real{1}, kAnnulusInnerRadius,
                    Real{0}, nghost};
  const std::size_t size = grid.storage_size();
  const std::vector<Real> zero(size, Real{0});
  const std::vector<Real> one(size, Real{1});
  const std::vector<Real> energy(size, Real{5});
  const std::vector<Real> radial_field(size, Real{0.2});
  std::vector<Real> component(size);
  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
      component[grid.index(i, j)] = evolved_component_average(
          grid.r_at_edge(i), grid.r_at_edge(i + 1), angular_momentum);
    }
  }

  quasar::mhd::MhdField2D<Real> state{grid};
  fill_field(state, zero);
  state.rho.copy_from_host(one.data(), size);
  state.energy.copy_from_host(energy.data(), size);
  state.bx_face.copy_from_host(radial_field.data(), size);
  if (angular_momentum) {
    state.mz.copy_from_host(component.data(), size);
  } else {
    state.bz_cell.copy_from_host(component.data(), size);
  }

  const RadialTables table_owner{grid, scheme_order};
  MhdInterfaceStates<Real> interfaces{grid, /*direction=*/0};
  quasar::mhd::launch_mhd_reconstruct(
      state, quasar::mhd::MhdBackgroundField<Real>{}, /*dir=*/0, interfaces,
      scheme_order, quasar::mhd::BoundaryFlags4{}, kGamma,
      /*stream=*/nullptr, /*rate_only=*/false, table_owner.view());
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> face_value(size);
  if (angular_momentum) {
    interfaces.Lmz.copy_to_host(face_value.data(), size);
  } else {
    interfaces.Lbz.copy_to_host(face_value.data(), size);
  }
  quasar::mhd::MhdField2D<Real> flux{grid};
  quasar::mhd::MhdField2D<Real> residual{grid};
  fill_field(flux, zero);
  fill_field(residual, zero);
  if (angular_momentum) {
    flux.mz.copy_from_host(face_value.data(), size);
  } else {
    flux.bz_cell.copy_from_host(face_value.data(), size);
  }
  quasar::mhd::launch_mhd_flux_difference(
      flux, /*dir=*/0, residual, /*stream=*/nullptr,
      /*cylindrical=*/true);
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> rate(size);
  if (angular_momentum) {
    residual.mz.copy_to_host(rate.data(), size);
  } else {
    residual.bz_cell.copy_to_host(rate.data(), size);
  }
  Real error = Real{0};
  for (int j = 0; j < grid.ny; ++j) {
    for (int i = 0; i < grid.nx; ++i) {
      const long double lo = static_cast<long double>(grid.r_at_edge(i));
      const long double hi = static_cast<long double>(grid.r_at_edge(i + 1));
      const long double flo = smooth_component_value(lo);
      const long double fhi = smooth_component_value(hi);
      const long double exact = angular_momentum
          ? -(hi * hi * fhi - lo * lo * flo) /
                ((hi * hi * hi - lo * lo * lo) / 3.0L)
          : -(fhi - flo) / (hi - lo);
      error += std::abs(
          rate[grid.index(i, j)] - static_cast<Real>(exact));
    }
  }
  return error / static_cast<Real>(grid.nx * grid.ny);
}

template <class Average>
void seed_entropy_wave(quasar::mhd::MhdField2D<Real>& field,
                       const Grid2D& grid, Average&& average) {
  const std::size_t size = grid.storage_size();
  std::vector<Real> rho(size);
  std::vector<Real> mx(size);
  std::vector<Real> my(size, Real{0});
  std::vector<Real> mz(size, Real{0});
  std::vector<Real> energy(size);
  std::vector<Real> bx_face(size, kBx);
  std::vector<Real> by_face(size, Real{0});
  std::vector<Real> bz_cell(size, kBz);

  const Real magnetic_energy =
      Real{0.5} * (kBx * kBx + kBz * kBz);
  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
      const Real cell_rho = average(grid.r_at_edge(i),
                                    grid.r_at_edge(i + 1));
      const std::size_t index = grid.index(i, j);
      rho[index] = cell_rho;
      mx[index] = cell_rho * kVr;
      energy[index] = kPressure / (kGamma - Real{1}) +
                      Real{0.5} * cell_rho * kVr * kVr + magnetic_energy;
    }
  }

  field.rho.copy_from_host(rho.data(), size);
  field.mx.copy_from_host(mx.data(), size);
  field.my.copy_from_host(my.data(), size);
  field.mz.copy_from_host(mz.data(), size);
  field.energy.copy_from_host(energy.data(), size);
  field.bx_face.copy_from_host(bx_face.data(), size);
  field.by_face.copy_from_host(by_face.data(), size);
  field.bz_cell.copy_from_host(bz_cell.data(), size);
}

std::vector<Real> reconstruct_density(const Grid2D& grid, int scheme_order,
                                      InterfaceState state,
                                      RadialTablesView radial_tables,
                                      bool cartesian_seed = false) {
  quasar::mhd::MhdField2D<Real> field{grid};
  if (cartesian_seed) {
    seed_entropy_wave(field, grid, smooth_cartesian_average);
  } else {
    seed_entropy_wave(field, grid, smooth_ring_average);
  }

  const quasar::mhd::MhdBackgroundField<Real> background{};
  const quasar::mhd::BoundaryFlags4 flags{};
  MhdInterfaceStates<Real> interfaces{grid, /*direction=*/0};
  quasar::mhd::launch_mhd_reconstruct(
      field, background, /*dir=*/0, interfaces, scheme_order, flags, kGamma,
      /*stream=*/nullptr, /*rate_only=*/false, radial_tables);
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> density(grid.storage_size());
  const auto& source = state == InterfaceState::left
                           ? interfaces.Lrho
                           : interfaces.Rrho;
  source.copy_to_host(density.data(), density.size());
  return density;
}

// This is the same factored annular divergence used in mhd_update.hip:
//   (F_hi-F_lo)/dr + (F_hi+F_lo)/(2*r_center).
// With constant radial velocity, density flux is kVr*rho.
Real annular_residual_error(int scheme_order, int nx,
                            InterfaceState state, bool tables_active) {
  constexpr int nghost = 4;
  const Grid2D grid{nx, 2, kAnnulusWidth, Real{1}, kAnnulusInnerRadius,
                    Real{0}, nghost};
  const RadialTables table_owner{grid, scheme_order};
  RadialTablesView view = table_owner.view();
  if (!tables_active) view.active = 0;

  const std::vector<Real> density =
      reconstruct_density(grid, scheme_order, state, view);
  Real error_sum = Real{0};
  constexpr int j = 0;
  for (int i = 0; i < grid.nx; ++i) {
    const Real numerical_lo = density[grid.index(i, j)];
    const Real numerical_hi = density[grid.index(i + 1, j)];
    const Real exact_lo = smooth_density(grid.r_at_edge(i));
    const Real exact_hi = smooth_density(grid.r_at_edge(i + 1));
    const Real radius = grid.r_at_cell_center(i);
    const Real numerical = -kVr *
        ((numerical_hi - numerical_lo) / grid.dx() +
         (numerical_hi + numerical_lo) / (Real{2} * radius));
    const Real exact = -kVr *
        ((exact_hi - exact_lo) / grid.dx() +
         (exact_hi + exact_lo) / (Real{2} * radius));
    error_sum += std::abs(numerical - exact);
  }
  return error_sum / static_cast<Real>(grid.nx);
}

struct OrderProbe {
  Real coarse_error;
  Real fine_error;
  Real order;
};

OrderProbe annular_residual_order(int scheme_order, int coarse_nx,
                                  InterfaceState state, bool tables_active) {
  const Real coarse = annular_residual_error(
      scheme_order, coarse_nx, state, tables_active);
  const Real fine = annular_residual_error(
      scheme_order, 2 * coarse_nx, state, tables_active);
  const Real order =
      (std::isfinite(coarse) && std::isfinite(fine) &&
       coarse > Real{0} && fine > Real{0})
          ? std::log2(coarse / fine)
          : Real{-1};
  return {coarse, fine, order};
}

std::string probe_description(const OrderProbe& probe) {
  return "coarse_error=" + std::to_string(probe.coarse_error) +
         ", fine_error=" + std::to_string(probe.fine_error) +
         ", observed_order=" + std::to_string(probe.order);
}

void seed_cartesian_golden_profile(quasar::mhd::MhdField2D<Real>& field,
                                   const Grid2D& grid) {
  const std::size_t size = grid.storage_size();
  std::vector<Real> rho(size);
  std::vector<Real> mx(size);
  std::vector<Real> my(size);
  std::vector<Real> mz(size, Real{0});
  std::vector<Real> energy(size);
  std::vector<Real> bx_face(size, Real{0.5});
  std::vector<Real> by_face(size, Real{0});
  std::vector<Real> bz_cell(size, Real{0.125});
  constexpr Real vx = Real{0.25};
  constexpr Real vy = Real{-0.125};
  constexpr Real pressure = Real{2};
  constexpr Real kinetic_factor =
      Real{0.5} * (vx * vx + vy * vy);
  constexpr Real magnetic_energy =
      Real{0.5} * (Real{0.5} * Real{0.5} +
                   Real{0.125} * Real{0.125});

  for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
    // An exact binary64 degree-eight profile avoids a libm-dependent golden
    // seed while remaining smooth enough to exercise the full MP7 stencil.
    const std::uint64_t coordinate = static_cast<std::uint64_t>(i + 16);
    const std::uint64_t square = coordinate * coordinate;
    const std::uint64_t fourth = square * square;
    const std::uint64_t eighth = fourth * fourth;
    const Real density =
        Real{2} + std::ldexp(static_cast<Real>(eighth), -48);
    const Real total_energy = pressure / (kGamma - Real{1}) +
                              kinetic_factor * density + magnetic_energy;
    for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
      const std::size_t index = grid.index(i, j);
      rho[index] = density;
      mx[index] = density * vx;
      my[index] = density * vy;
      energy[index] = total_energy;
    }
  }

  field.rho.copy_from_host(rho.data(), size);
  field.mx.copy_from_host(mx.data(), size);
  field.my.copy_from_host(my.data(), size);
  field.mz.copy_from_host(mz.data(), size);
  field.energy.copy_from_host(energy.data(), size);
  field.bx_face.copy_from_host(bx_face.data(), size);
  field.by_face.copy_from_host(by_face.data(), size);
  field.bz_cell.copy_from_host(bz_cell.data(), size);
}

using CartesianReconstructionBits = std::array<std::uint64_t, 40>;

// MP7 Cartesian reconstruction for the exact-binary profile above.  The
// component-major layout is rho, mx, my, energy; each component contains L/R
// pairs at faces {2, 6, 11, 17, 22}.  Every pair and every selected face is
// nonconstant, so neither side can satisfy this golden through an early-out.
inline constexpr CartesianReconstructionBits kMp7CartesianGoldenBits{
    UINT64_C(0x4000001042945900), UINT64_C(0x400000104293bb80),
    UINT64_C(0x400000549d48b100), UINT64_C(0x400000549d47ef80),
    UINT64_C(0x400001c37edf1b80), UINT64_C(0x400001c37ede2d00),
    UINT64_C(0x40000909542b5040), UINT64_C(0x40000909542a2bc0),
    UINT64_C(0x40001c68182e9d00), UINT64_C(0x40001c68182d4b80),
    UINT64_C(0x3fe0001042945900), UINT64_C(0x3fe000104293bb80),
    UINT64_C(0x3fe000549d48b100), UINT64_C(0x3fe000549d47ef80),
    UINT64_C(0x3fe001c37edf1b80), UINT64_C(0x3fe001c37ede2d00),
    UINT64_C(0x3fe00909542b5040), UINT64_C(0x3fe00909542a2bc0),
    UINT64_C(0x3fe01c68182e9d00), UINT64_C(0x3fe01c68182d4b80),
    UINT64_C(0xbfd0001042945900), UINT64_C(0xbfd000104293bb80),
    UINT64_C(0xbfd000549d48b100), UINT64_C(0xbfd000549d47ef80),
    UINT64_C(0xbfd001c37edf1b80), UINT64_C(0xbfd001c37ede2d00),
    UINT64_C(0xbfd00909542b5040), UINT64_C(0xbfd00909542a2bc0),
    UINT64_C(0xbfd01c68182e9d00), UINT64_C(0xbfd01c68182d4b80),
    UINT64_C(0x4009b000a299cb79), UINT64_C(0x4009b000a299c552),
    UINT64_C(0x4009b0034e24d6e9), UINT64_C(0x4009b0034e24cf5a),
    UINT64_C(0x4009b011a2f4b712), UINT64_C(0x4009b011a2f4adc1),
    UINT64_C(0x4009b05a5d49b121), UINT64_C(0x4009b05a5d49a5b4),
    UINT64_C(0x4009b11c10f1d221), UINT64_C(0x4009b11c10f1c4f2)};

CartesianReconstructionBits written_cartesian_golden_bits(
    const MhdInterfaceStates<Real>& interfaces) {
  constexpr std::array<int, 5> faces{2, 6, 11, 17, 22};
  const Grid2D& grid = interfaces.grid;
  CartesianReconstructionBits result{};
  std::size_t output = 0;
  // Component-major layout: rho, mx, my, energy; within each component the
  // selected faces are stored as adjacent L/R pairs.
  const auto append_component = [&](const auto& left, const auto& right) {
    std::vector<Real> left_host(grid.storage_size());
    std::vector<Real> right_host(grid.storage_size());
    left.copy_to_host(left_host.data(), left_host.size());
    right.copy_to_host(right_host.data(), right_host.size());
    for (const int face : faces) {
      result[output++] =
          std::bit_cast<std::uint64_t>(left_host[grid.index(face, 0)]);
      result[output++] =
          std::bit_cast<std::uint64_t>(right_host[grid.index(face, 0)]);
    }
  };
  append_component(interfaces.Lrho, interfaces.Rrho);
  append_component(interfaces.Lmx, interfaces.Rmx);
  append_component(interfaces.Lmy, interfaces.Rmy);
  append_component(interfaces.Lenergy, interfaces.Renergy);
  return result;
}

CartesianReconstructionBits reconstruct_cartesian_golden_bits(
    const Grid2D& grid, RadialTablesView view) {
  quasar::mhd::MhdField2D<Real> field{grid};
  seed_cartesian_golden_profile(field, grid);
  MhdInterfaceStates<Real> interfaces{grid, /*direction=*/0};
  quasar::mhd::launch_mhd_reconstruct(
      field, quasar::mhd::MhdBackgroundField<Real>{}, /*dir=*/0, interfaces,
      /*scheme_order=*/7, quasar::mhd::BoundaryFlags4{}, kGamma,
      /*stream=*/nullptr, /*rate_only=*/false, view);
  quasar::backend::device_synchronize(nullptr);
  return written_cartesian_golden_bits(interfaces);
}

struct NativeInterfaceValues {
  Real left_mphi;
  Real right_mphi;
  Real left_bphi;
  Real right_bphi;
};

NativeInterfaceValues reconstruct_native_union(
    int scheme_order, int face, const std::vector<Real>& mphi_union,
    const std::vector<Real>& bphi_union) {
  const int half = scheme_order == 5 ? 2 : 3;
  const int union_width = 2 * half + 2;
  if (static_cast<int>(mphi_union.size()) != union_width ||
      static_cast<int>(bphi_union.size()) != union_width) {
    throw std::invalid_argument("native reconstruction union has wrong width");
  }

  constexpr int nghost = 4;
  const Grid2D grid{/*nx=*/12, /*ny=*/2, Real{12}, Real{1}, Real{0},
                    Real{0}, nghost};
  const std::size_t size = grid.storage_size();
  std::vector<Real> rho(size, Real{1});
  std::vector<Real> mx(size, Real{0});
  std::vector<Real> my(size, Real{0});
  std::vector<Real> mz(size);
  std::vector<Real> energy(size, Real{1e16});
  std::vector<Real> bx_face(size, Real{0.5});
  std::vector<Real> by_face(size, Real{0});
  std::vector<Real> bz_cell(size);
  const int first_cell = face - 1 - half;
  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
      const int union_index = std::clamp(i - first_cell, 0, union_width - 1);
      const std::size_t index = grid.index(i, j);
      mz[index] = mphi_union[static_cast<std::size_t>(union_index)];
      bz_cell[index] = bphi_union[static_cast<std::size_t>(union_index)];
    }
  }

  quasar::mhd::MhdField2D<Real> state{grid};
  state.rho.copy_from_host(rho.data(), size);
  state.mx.copy_from_host(mx.data(), size);
  state.my.copy_from_host(my.data(), size);
  state.mz.copy_from_host(mz.data(), size);
  state.energy.copy_from_host(energy.data(), size);
  state.bx_face.copy_from_host(bx_face.data(), size);
  state.by_face.copy_from_host(by_face.data(), size);
  state.bz_cell.copy_from_host(bz_cell.data(), size);

  const RadialTables table_owner{grid, scheme_order};
  MhdInterfaceStates<Real> interfaces{grid, /*direction=*/0};
  quasar::mhd::launch_mhd_reconstruct(
      state, quasar::mhd::MhdBackgroundField<Real>{}, /*dir=*/0, interfaces,
      scheme_order, quasar::mhd::BoundaryFlags4{}, kGamma,
      /*stream=*/nullptr, /*rate_only=*/false, table_owner.view());
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> left_mphi(size), right_mphi(size);
  std::vector<Real> left_bphi(size), right_bphi(size);
  interfaces.Lmz.copy_to_host(left_mphi.data(), size);
  interfaces.Rmz.copy_to_host(right_mphi.data(), size);
  interfaces.Lbz.copy_to_host(left_bphi.data(), size);
  interfaces.Rbz.copy_to_host(right_bphi.data(), size);
  const std::size_t target = grid.index(face, 0);
  return NativeInterfaceValues{
      left_mphi[target], right_mphi[target],
      left_bphi[target], right_bphi[target]};
}

}  // namespace

TEST(MhdFluxReconstructionCylindrical,
     Mp5AnnularConservativeResidualConvergesAtDesignOrder) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const OrderProbe probe = annular_residual_order(
      /*scheme_order=*/5, /*coarse_nx=*/24, InterfaceState::left,
      /*tables_active=*/true);
  ::testing::Test::RecordProperty("observed_order", probe.order);
  ASSERT_GT(probe.order, Real{0}) << probe_description(probe);
  EXPECT_GE(probe.order, Real{4.6}) << probe_description(probe);
}

TEST(MhdFluxReconstructionCylindrical,
     Mp7AnnularConservativeResidualConvergesAtDesignOrder) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const OrderProbe probe = annular_residual_order(
      /*scheme_order=*/7, /*coarse_nx=*/20, InterfaceState::left,
      /*tables_active=*/true);
  ::testing::Test::RecordProperty("observed_order", probe.order);
  ASSERT_GT(probe.order, Real{0}) << probe_description(probe);
  EXPECT_GE(probe.order, Real{6.4}) << probe_description(probe);
}

TEST(MhdFluxReconstructionCylindrical,
     Mp5AxisNativeCandidatesAreLimitedBeforeOverride) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const std::vector<Real> zero(6, Real{0});
  const NativeInterfaceValues mphi = reconstruct_native_union(
      /*scheme_order=*/5, /*face=*/1,
      std::vector<Real>{-1, -2, 2, 1, 3, 3}, zero);
  EXPECT_EQ(mphi.left_mphi, Real{2});

  const NativeInterfaceValues bphi = reconstruct_native_union(
      /*scheme_order=*/5, /*face=*/1, zero,
      std::vector<Real>{-3, -2, 2, 3, 3, 3});
  EXPECT_EQ(bphi.left_bphi, Real{3});
}

TEST(MhdFluxReconstructionCylindrical,
     Mp7NativeLimiterRejectsOuterTapCancellationOnBothSides) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const std::vector<Real> values{
      1000000.0, -2.0, -1.0, 0.0, 1.0, 2.0,
      787812.0342935974, 6550699.945732342};
  const NativeInterfaceValues reconstructed = reconstruct_native_union(
      /*scheme_order=*/7, /*face=*/6, values, values);
  EXPECT_EQ(reconstructed.left_mphi, Real{0});
  EXPECT_EQ(reconstructed.right_mphi, Real{0});
  EXPECT_EQ(reconstructed.left_bphi, Real{1});
  EXPECT_EQ(reconstructed.right_bphi, Real{1});
}

TEST(MhdFluxReconstructionCylindrical,
     AzimuthalMomentumAndFieldUseTheirConservedMeasures) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  for (const int scheme_order : {5, 7}) {
    const int coarse_nx = scheme_order == 5 ? 24 : 20;
    const Real threshold = scheme_order == 5 ? Real{4.5} : Real{6.3};
    for (const bool angular_momentum : {true, false}) {
      SCOPED_TRACE(
          std::string{scheme_order == 5 ? "MP5 " : "MP7 "} +
          (angular_momentum ? "m_phi" : "B_phi"));
      const Real coarse = evolved_component_residual_error(
          scheme_order, coarse_nx, angular_momentum);
      const Real fine = evolved_component_residual_error(
          scheme_order, 2 * coarse_nx, angular_momentum);
      ASSERT_GT(coarse, Real{0});
      ASSERT_GT(fine, Real{0});
      const Real observed = std::log2(coarse / fine);
      const std::string property =
          std::string{scheme_order == 5 ? "mp5_" : "mp7_"} +
          (angular_momentum ? "mphi_order" : "bphi_order");
      ::testing::Test::RecordProperty(property, observed);
      EXPECT_GE(observed, threshold)
          << "errors " << coarse << " -> " << fine;
    }
  }
}

TEST(MhdFluxReconstructionCylindrical,
     AxialFaceFluxUsesEachComponentNativeRadialMeasure) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  constexpr int nghost = 4;
  const Grid2D grid{/*nx=*/12, /*ny=*/6, Real{1}, Real{1}, Real{1},
                    Real{0}, nghost};
  const std::size_t size = grid.storage_size();
  MhdInterfaceStates<Real> interfaces{grid, /*direction=*/1};

  constexpr long double rho0 = 1.0L;
  constexpr long double rho_slope = 0.25L;
  constexpr long double axial_velocity = 0.3L;
  constexpr long double azimuthal_velocity = 0.2L;
  constexpr long double bphi_per_density = 0.15L;
  constexpr long double pressure = 2.0L;
  constexpr long double kinetic_coefficient =
      0.5L * (axial_velocity * axial_velocity +
              azimuthal_velocity * azimuthal_velocity);
  constexpr long double energy_c0 =
      pressure / (static_cast<long double>(kGamma) - 1.0L) +
      kinetic_coefficient * rho0 +
      0.5L * bphi_per_density * bphi_per_density * rho0 * rho0;
  constexpr long double energy_c1 =
      kinetic_coefficient * rho_slope +
      bphi_per_density * bphi_per_density * rho0 * rho_slope;
  constexpr long double energy_c2 = 0.5L * bphi_per_density *
      bphi_per_density * rho_slope * rho_slope;

  std::vector<Real> rho(size), mx(size, Real{0}), my(size), mz(size);
  std::vector<Real> energy(size), bx(size, Real{0}), by(size, Real{0});
  std::vector<Real> bz(size);
  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
      const Real lo = grid.r_at_edge(i);
      const Real hi = grid.r_at_edge(i + 1);
      const Real annular_rho = quadratic_average_under_measure(
          lo, hi, /*measure_power=*/1, rho0, rho_slope, 0.0L);
      const Real angular_rho = quadratic_average_under_measure(
          lo, hi, /*measure_power=*/2, rho0, rho_slope, 0.0L);
      const Real uniform_rho = quadratic_average_under_measure(
          lo, hi, /*measure_power=*/0, rho0, rho_slope, 0.0L);
      const std::size_t k = grid.index(i, j);
      rho[k] = annular_rho;
      my[k] = static_cast<Real>(axial_velocity) * annular_rho;
      mz[k] = static_cast<Real>(azimuthal_velocity) * angular_rho;
      bz[k] = static_cast<Real>(bphi_per_density) * uniform_rho;
      energy[k] = quadratic_average_under_measure(
          lo, hi, /*measure_power=*/1,
          energy_c0, energy_c1, energy_c2);
    }
  }

  const auto seed_component = [size](
      auto& left, auto& right, const std::vector<Real>& values) {
    left.copy_from_host(values.data(), size);
    right.copy_from_host(values.data(), size);
  };
  seed_component(interfaces.Lrho, interfaces.Rrho, rho);
  seed_component(interfaces.Lmx, interfaces.Rmx, mx);
  seed_component(interfaces.Lmy, interfaces.Rmy, my);
  seed_component(interfaces.Lmz, interfaces.Rmz, mz);
  seed_component(interfaces.Lenergy, interfaces.Renergy, energy);
  seed_component(interfaces.Lbx, interfaces.Rbx, bx);
  seed_component(interfaces.Lby, interfaces.Rby, by);
  seed_component(interfaces.Lbz, interfaces.Rbz, bz);

  constexpr int target_i = 5;
  constexpr int target_j = 3;
  const Real target_lo = grid.r_at_edge(target_i);
  const Real target_hi = grid.r_at_edge(target_i + 1);
  const Real expected_annular = quadratic_average_under_measure(
      target_lo, target_hi, /*measure_power=*/1, rho0, rho_slope, 0.0L);
  const Real expected_angular = quadratic_average_under_measure(
      target_lo, target_hi, /*measure_power=*/2, rho0, rho_slope, 0.0L);
  const Real expected_uniform = quadratic_average_under_measure(
      target_lo, target_hi, /*measure_power=*/0, rho0, rho_slope, 0.0L);
  ASSERT_NE(expected_annular, expected_angular);
  ASSERT_NE(expected_annular, expected_uniform);
  ASSERT_NE(expected_angular, expected_uniform);

  for (const int order : {5, 7}) {
    SCOPED_TRACE(order);
    const RadialTables radial_tables{grid, order};
    quasar::mhd::MhdField2D<Real> flux{grid};
    quasar::mhd::launch_mhd_hlld_flux(
        interfaces, quasar::mhd::MhdBackgroundField<Real>{}, /*dir=*/1,
        flux, quasar::mhd::BoundaryFlags4{}, kGamma, /*stream=*/nullptr,
        /*hll_only=*/false, /*momentum_parts=*/nullptr, order,
        quasar::mhd::FaceOwnershipFlags4{}, radial_tables.view());
    quasar::backend::device_synchronize(nullptr);

    std::vector<Real> rho_flux(size), mphi_flux(size), bphi_flux(size);
    flux.rho.copy_to_host(rho_flux.data(), size);
    flux.mz.copy_to_host(mphi_flux.data(), size);
    flux.bz_cell.copy_to_host(bphi_flux.data(), size);
    const std::size_t target = grid.index(target_i, target_j);
    constexpr Real tolerance = Real{2e-12};
    EXPECT_NEAR(
        rho_flux[target],
        static_cast<Real>(axial_velocity) * expected_annular, tolerance);
    EXPECT_NEAR(
        mphi_flux[target],
        static_cast<Real>(axial_velocity * azimuthal_velocity) *
            expected_angular,
        tolerance);
    EXPECT_NEAR(
        bphi_flux[target],
        static_cast<Real>(axial_velocity * bphi_per_density) *
            expected_uniform,
        tolerance);
  }
}

TEST(MhdFluxReconstructionCylindrical,
     InactiveTablesCollapseAnnularResidualToSecondOrder) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  for (const int scheme_order : {5, 7}) {
    SCOPED_TRACE("scheme_order=" + std::to_string(scheme_order));
    const int coarse_nx = scheme_order == 5 ? 24 : 20;
    const OrderProbe probe = annular_residual_order(
        scheme_order, coarse_nx, InterfaceState::left,
        /*tables_active=*/false);
    ::testing::Test::RecordProperty(
        scheme_order == 5 ? "mp5_observed_order" : "mp7_observed_order",
        probe.order);
    ASSERT_GT(probe.order, Real{0}) << probe_description(probe);
    EXPECT_LE(probe.order, Real{2.5}) << probe_description(probe);
  }
}

TEST(MhdFluxReconstructionCylindrical, BothInterfaceStatesAreHighOrder) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  for (const int scheme_order : {5, 7}) {
    const int coarse_nx = scheme_order == 5 ? 24 : 20;
    const Real threshold = scheme_order == 5 ? Real{4.6} : Real{6.4};
    for (const InterfaceState state : {InterfaceState::left,
                                       InterfaceState::right}) {
      SCOPED_TRACE("scheme_order=" + std::to_string(scheme_order) +
                   (state == InterfaceState::left ? ", left" : ", right"));
      const OrderProbe probe = annular_residual_order(
          scheme_order, coarse_nx, state, /*tables_active=*/true);
      const std::string property =
          (scheme_order == 5 ? "mp5_" : "mp7_") +
          std::string{state == InterfaceState::left ? "left_order"
                                                    : "right_order"};
      ::testing::Test::RecordProperty(property, probe.order);
      ASSERT_GT(probe.order, Real{0}) << probe_description(probe);
      EXPECT_GE(probe.order, threshold) << probe_description(probe);
    }
  }
}

TEST(MhdFluxReconstructionCylindrical,
     InactiveCartesianPathsMatchCheckedInGoldenBits) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  constexpr int nghost = 4;
  const Grid2D grid{/*nx=*/24, /*ny=*/2, kAnnulusWidth, Real{1},
                    kAnnulusInnerRadius, Real{0}, nghost};
  const RadialTables owner{grid, /*scheme_order=*/7};
  RadialTablesView explicitly_inactive = owner.view();
  explicitly_inactive.active = 0;

  const CartesianReconstructionBits default_inactive =
      reconstruct_cartesian_golden_bits(grid, RadialTablesView{});
  const CartesianReconstructionBits nonnull_inactive =
      reconstruct_cartesian_golden_bits(grid, explicitly_inactive);
  for (std::size_t index = 0; index < kMp7CartesianGoldenBits.size();
       ++index) {
    EXPECT_EQ(default_inactive[index], kMp7CartesianGoldenBits[index])
        << "default inactive view, golden scalar " << index;
    EXPECT_EQ(nonnull_inactive[index], kMp7CartesianGoldenBits[index])
        << "non-null inactive view, golden scalar " << index;
  }
}
