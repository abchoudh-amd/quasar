// Focused device tests for radius-weighted MP5/MP7 fluid reconstruction.
//
// Cylindrical conserved variables are annular averages under r dr.  These
// tests therefore seed exact ring averages and measure the conservative
// annular face difference used by mhd_update.hip.  A Cartesian-coefficient
// negative control is part of the contract: without it, an interface-only
// accuracy probe can pass while silently using the wrong finite-volume
// measure.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/numerics/interface_states.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/numerics/radial_tables.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
