// Focused r=0 contracts needed by cylindrical MP7.
//
// MP7 consumes four radial ghost cells.  The axis closure must therefore fill
// every layer with the physical m=0 parity, while the face-staggered radial
// magnetic field is mirrored about (and pinned on) the r=0 face.  Separately,
// the radial moment engine must see reflected cells through the signed
// monomial / |r| measure: the normalized moment in cell -1 is (-1)^m times
// the corresponding moment in cell 0.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/interface_states.hpp"
#include "quasar/numerics/radial_moments.hpp"
#include "quasar/numerics/radial_tables.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"
#include "quasar/physics/mhd/mhd_solver.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::mhd::MhdField2D;

constexpr int kAxisMode = 3;
constexpr Real kGamma = Real{5} / Real{3};
constexpr long double kWaveNumber = 4.0L;
constexpr long double kDensityAmplitude = 0.25L;
constexpr long double kAzimuthalMomentumAmplitude = 0.5L;

// Integer-valued binary64 samples make every copy/sign assertion exact.  The
// component-specific base also detects accidental cross-component reads.
Real pattern(int i, int j, Real base, int nghost) {
  return base + Real{100} * static_cast<Real>(i + nghost) +
         static_cast<Real>(j + nghost);
}

void seed_component(quasar::backend::DeviceBuffer<Real>& device,
                    const Grid2D& grid, Real base) {
  std::vector<Real> host(grid.storage_size());
  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
      host[grid.index(i, j)] = pattern(i, j, base, grid.nghost);
    }
  }
  device.copy_from_host(host.data(), host.size());
}

std::vector<Real> read_component(
    const quasar::backend::DeviceBuffer<Real>& device, const Grid2D& grid) {
  std::vector<Real> host(grid.storage_size());
  device.copy_to_host(host.data(), host.size());
  return host;
}

bool all_finite(const std::vector<Real>& values) {
  for (const Real value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

long double weighted_cosine_antiderivative(long double radius) {
  return radius * std::sin(kWaveNumber * radius) / kWaveNumber +
         std::cos(kWaveNumber * radius) /
             (kWaveNumber * kWaveNumber);
}

long double weighted_sine_antiderivative(long double radius) {
  return -radius * std::cos(kWaveNumber * radius) / kWaveNumber +
         std::sin(kWaveNumber * radius) /
             (kWaveNumber * kWaveNumber);
}

Real even_density(Real radius) {
  return static_cast<Real>(2.0L + kDensityAmplitude *
                                      std::cos(kWaveNumber * radius));
}

Real odd_azimuthal_momentum(Real radius) {
  return static_cast<Real>(kAzimuthalMomentumAmplitude *
                           std::sin(kWaveNumber * radius));
}

Real positive_radius_ring_average(const Grid2D& grid, int i,
                                  long double base,
                                  long double amplitude,
                                  bool sine_profile) {
  const long double lower = static_cast<long double>(grid.r_at_edge(i));
  const long double upper = static_cast<long double>(grid.r_at_edge(i + 1));
  const long double measure = (upper * upper - lower * lower) / 2.0L;
  const long double integral = sine_profile
      ? weighted_sine_antiderivative(upper) -
            weighted_sine_antiderivative(lower)
      : weighted_cosine_antiderivative(upper) -
            weighted_cosine_antiderivative(lower);
  return static_cast<Real>(base + amplitude * integral / measure);
}

void seed_axis_parity_profiles(quasar::mhd::MhdField2D<Real>& field,
                               const Grid2D& grid) {
  const std::size_t size = grid.storage_size();
  std::vector<Real> rho(size, Real{2});
  std::vector<Real> radial_momentum(size, Real{0});
  std::vector<Real> axial_momentum(size, Real{0});
  std::vector<Real> azimuthal_momentum(size, Real{0});
  std::vector<Real> energy(size, Real{20});
  std::vector<Real> radial_field(size, Real{0});
  std::vector<Real> axial_field(size, Real{0.3});
  std::vector<Real> azimuthal_field(size, Real{0});

  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    for (int i = 0; i < grid.nx + grid.nghost; ++i) {
      const std::size_t index = grid.index(i, j);
      rho[index] = positive_radius_ring_average(
          grid, i, /*base=*/2.0L, kDensityAmplitude,
          /*sine_profile=*/false);
      azimuthal_momentum[index] = positive_radius_ring_average(
          grid, i, /*base=*/0.0L, kAzimuthalMomentumAmplitude,
          /*sine_profile=*/true);
      radial_field[index] = Real{0.2} * grid.r_at_edge(i);
    }
  }

  field.rho.copy_from_host(rho.data(), size);
  field.mx.copy_from_host(radial_momentum.data(), size);
  field.my.copy_from_host(axial_momentum.data(), size);
  field.mz.copy_from_host(azimuthal_momentum.data(), size);
  field.energy.copy_from_host(energy.data(), size);
  field.bx_face.copy_from_host(radial_field.data(), size);
  field.by_face.copy_from_host(axial_field.data(), size);
  field.bz_cell.copy_from_host(azimuthal_field.data(), size);

  quasar::mhd::launch_mhd_fill_ghosts_fluid(
      field, /*side=x_lo=*/0, kAxisMode, nullptr);
  quasar::mhd::launch_mhd_fill_ghosts_field(
      field, /*side=x_lo=*/0, kAxisMode, nullptr);
}

struct AxisReconstructionErrors {
  Real even;
  Real odd;
};

AxisReconstructionErrors axis_reconstruction_errors(int nx) {
  constexpr int nghost = 4;
  const Grid2D grid{nx, /*ny=*/2, /*lx=*/Real{1}, /*ly=*/Real{1},
                    /*origin_x=*/Real{0}, /*origin_y=*/Real{0}, nghost};
  quasar::mhd::MhdField2D<Real> field{grid};
  seed_axis_parity_profiles(field, grid);

  const quasar::numerics::RadialTables table_owner{
      grid, /*scheme_order=*/7};
  const auto radial_tables = table_owner.view();
  quasar::numerics::MhdInterfaceStates<Real> interfaces{
      grid, /*direction=*/0};
  quasar::mhd::BoundaryFlags4 boundaries{};
  boundaries.side[0] = kAxisMode;
  boundaries.side[1] = 1;
  quasar::mhd::launch_mhd_reconstruct(
      field, quasar::mhd::MhdBackgroundField<Real>{}, /*dir=*/0,
      interfaces, /*scheme_order=*/7, boundaries, kGamma,
      /*stream=*/nullptr, /*rate_only=*/false, radial_tables);
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> left_density(grid.storage_size());
  std::vector<Real> right_density(grid.storage_size());
  std::vector<Real> left_azimuthal_momentum(grid.storage_size());
  std::vector<Real> right_azimuthal_momentum(grid.storage_size());
  interfaces.Lrho.copy_to_host(left_density.data(), left_density.size());
  interfaces.Rrho.copy_to_host(right_density.data(), right_density.size());
  interfaces.Lmz.copy_to_host(left_azimuthal_momentum.data(),
                              left_azimuthal_momentum.size());
  interfaces.Rmz.copy_to_host(right_azimuthal_momentum.data(),
                              right_azimuthal_momentum.size());

  Real even_error = Real{0};
  Real odd_error = Real{0};
  constexpr int measured_cells = 5;
  constexpr int j = 0;
  for (int cell = 0; cell < measured_cells; ++cell) {
    // Cell i owns outer face i+1.  Face zero is deliberately excluded because
    // the cylindrical divergence cancels it exactly at the axis.
    const int outer_face = cell + 1;
    const std::size_t index = grid.index(outer_face, j);
    const Real radius = grid.r_at_edge(outer_face);
    const Real exact_even = even_density(radius);
    const Real exact_odd = odd_azimuthal_momentum(radius);
    even_error += std::abs(left_density[index] - exact_even) +
                  std::abs(right_density[index] - exact_even);
    odd_error += std::abs(left_azimuthal_momentum[index] - exact_odd) +
                 std::abs(right_azimuthal_momentum[index] - exact_odd);
  }
  constexpr Real sample_count = Real{2 * measured_cells};
  return {even_error / sample_count, odd_error / sample_count};
}

Real convergence_order(Real coarse_error, Real fine_error) {
  if (!(std::isfinite(coarse_error) && std::isfinite(fine_error)) ||
      coarse_error <= Real{0} || fine_error <= Real{0}) {
    return Real{-1};
  }
  return std::log2(coarse_error / fine_error);
}

quasar::mhd::MhdConfig axis_mp7_config() {
  quasar::mhd::MhdConfig cfg;
  cfg.grid = Grid2D{/*nx=*/16, /*ny=*/12, /*lx=*/Real{1}, /*ly=*/Real{1},
                    /*origin_x=*/Real{0}, /*origin_y=*/Real{0},
                    /*nghost=*/4};
  cfg.gamma = Real{5} / Real{3};
  cfg.geometry = "cylindrical";
  cfg.reconstruction = "mp7";
  cfg.riemann = "hlld";
  cfg.integrator = "ssprk3";
  cfg.ct = "fd_ct_christlieb";
  cfg.positivity = "troubled_cell";
  cfg.cfl = Real{0.2};
  for (int side = 0; side < 4; ++side) {
    cfg.boundary.fluid[side] = "outflow";
    cfg.boundary.field[side] = "outflow";
  }
  cfg.boundary.fluid[0] = "axis";
  cfg.boundary.field[0] = "axis";
  return cfg;
}

void seed_uniform_axis_equilibrium(quasar::mhd::MhdSolver2D& solver,
                                   const Grid2D& grid, Real gamma) {
  const std::size_t size = grid.storage_size();
  constexpr Real rho0 = Real{1};
  constexpr Real pressure0 = Real{0.5};
  constexpr Real bz_axial = Real{0.2};
  const Real energy0 = pressure0 / (gamma - Real{1}) +
                       Real{0.5} * bz_axial * bz_axial;

  const std::vector<Real> rho(size, rho0);
  const std::vector<Real> zero(size, Real{0});
  const std::vector<Real> energy(size, energy0);
  const std::vector<Real> axial_field(size, bz_axial);
  solver.seed_state("rho", rho);
  solver.seed_state("mx", zero);
  solver.seed_state("my", zero);
  solver.seed_state("mz", zero);
  solver.seed_state("energy", energy);
  solver.seed_state("bx_face", zero);
  solver.seed_state("by_face", axial_field);
  solver.seed_state("bz_cell", zero);
}

}  // namespace

TEST(MhdAxisHighOrder, AxisGhostFillCoversFourLayers) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D grid{/*nx=*/8, /*ny=*/5, /*lx=*/Real{1}, /*ly=*/Real{1},
                    /*origin_x=*/Real{0}, /*origin_y=*/Real{0},
                    /*nghost=*/4};
  MhdField2D<Real> field{grid};

  constexpr Real rho_base = Real{1000};
  constexpr Real mx_base = Real{2000};
  constexpr Real my_base = Real{3000};
  constexpr Real mz_base = Real{4000};
  constexpr Real energy_base = Real{5000};
  constexpr Real bx_base = Real{6000};
  constexpr Real by_base = Real{7000};
  constexpr Real bz_base = Real{8000};
  seed_component(field.rho, grid, rho_base);
  seed_component(field.mx, grid, mx_base);
  seed_component(field.my, grid, my_base);
  seed_component(field.mz, grid, mz_base);
  seed_component(field.energy, grid, energy_base);
  seed_component(field.bx_face, grid, bx_base);
  seed_component(field.by_face, grid, by_base);
  seed_component(field.bz_cell, grid, bz_base);

  quasar::mhd::launch_mhd_fill_ghosts_fluid(
      field, /*side=x_lo=*/0, kAxisMode, nullptr);
  quasar::mhd::launch_mhd_fill_ghosts_field(
      field, /*side=x_lo=*/0, kAxisMode, nullptr);
  quasar::backend::device_synchronize(nullptr);

  const auto rho = read_component(field.rho, grid);
  const auto mx = read_component(field.mx, grid);
  const auto my = read_component(field.my, grid);
  const auto mz = read_component(field.mz, grid);
  const auto energy = read_component(field.energy, grid);
  const auto bx = read_component(field.bx_face, grid);
  const auto by = read_component(field.by_face, grid);
  const auto bz = read_component(field.bz_cell, grid);

  // The kernel intentionally spans the complete padded transverse range, so
  // verify each of the four MP7 layers on both ordinary rows and corner rows.
  for (int layer = 1; layer <= grid.nghost; ++layer) {
    const int ghost_i = -layer;
    const int cell_source_i = layer - 1;
    const int face_source_i = layer;
    for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
      const std::size_t ghost = grid.index(ghost_i, j);
      EXPECT_EQ(rho[ghost],
                pattern(cell_source_i, j, rho_base, grid.nghost));
      EXPECT_EQ(mx[ghost],
                -pattern(cell_source_i, j, mx_base, grid.nghost));
      EXPECT_EQ(my[ghost],
                pattern(cell_source_i, j, my_base, grid.nghost));
      EXPECT_EQ(mz[ghost],
                -pattern(cell_source_i, j, mz_base, grid.nghost));
      EXPECT_EQ(energy[ghost],
                pattern(cell_source_i, j, energy_base, grid.nghost));

      EXPECT_EQ(bx[ghost],
                -pattern(face_source_i, j, bx_base, grid.nghost));
      EXPECT_EQ(by[ghost],
                pattern(cell_source_i, j, by_base, grid.nghost));
      EXPECT_EQ(bz[ghost],
                -pattern(cell_source_i, j, bz_base, grid.nghost));
      EXPECT_EQ(bx[grid.index(0, j)], Real{0});
    }
  }
}

TEST(MhdAxisHighOrder, AxisRWeightedMirrorIsExact) {
  for (int m = 0; m <= 7; ++m) {
    const long double positive =
        quasar::numerics::normalized_cell_moment(0.5L, m);
    const long double negative =
        quasar::numerics::normalized_cell_moment(-0.5L, m);
    const long double parity = (m % 2 == 0) ? 1.0L : -1.0L;
    const long double analytic = 2.0L / static_cast<long double>(m + 2);

    EXPECT_LT(std::fabs(positive - analytic), 1.0e-15L) << "m=" << m;
    EXPECT_LT(std::fabs(negative - parity * analytic), 1.0e-15L)
        << "m=" << m;
    EXPECT_LT(std::fabs(negative - parity * positive), 1.0e-15L)
        << "m=" << m;
  }
}

TEST(MhdAxisHighOrder, AxisReconstructionIsHighOrder) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const AxisReconstructionErrors coarse = axis_reconstruction_errors(24);
  const AxisReconstructionErrors fine = axis_reconstruction_errors(48);
  const Real even_order = convergence_order(coarse.even, fine.even);
  const Real odd_order = convergence_order(coarse.odd, fine.odd);
  ::testing::Test::RecordProperty("even_coarse_error", coarse.even);
  ::testing::Test::RecordProperty("even_fine_error", fine.even);
  ::testing::Test::RecordProperty("even_observed_order", even_order);
  ::testing::Test::RecordProperty("odd_coarse_error", coarse.odd);
  ::testing::Test::RecordProperty("odd_fine_error", fine.odd);
  ::testing::Test::RecordProperty("odd_observed_order", odd_order);

  ASSERT_GT(even_order, Real{0})
      << "coarse=" << coarse.even << ", fine=" << fine.even;
  ASSERT_GT(odd_order, Real{0})
      << "coarse=" << coarse.odd << ", fine=" << fine.odd;
  EXPECT_GE(even_order, Real{6.4})
      << "coarse=" << coarse.even << ", fine=" << fine.even;
  EXPECT_GE(odd_order, Real{6.4})
      << "coarse=" << coarse.odd << ", fine=" << fine.odd;
}

TEST(MhdAxisHighOrder, AxisMp7RunsStably) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const auto cfg = axis_mp7_config();
  ASSERT_EQ(cfg.grid.nghost, 4);
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_uniform_axis_equilibrium(solver, cfg.grid, cfg.gamma);

  ASSERT_LT(solver.divergence_b_max(), Real{1e-12});
  const Real dt = Real{0.2} * solver.cfl_limit();
  ASSERT_GT(dt, Real{0});
  ASSERT_TRUE(std::isfinite(dt));
  for (int step = 0; step < 50; ++step) {
    ASSERT_NO_THROW(solver.step_unchecked(dt)) << "step=" << step;
  }

  EXPECT_TRUE(all_finite(solver.state_component_to_host("rho")));
  EXPECT_TRUE(all_finite(solver.state_component_to_host("energy")));
  EXPECT_TRUE(all_finite(solver.state_component_to_host("bx_face")));
  EXPECT_TRUE(all_finite(solver.state_component_to_host("by_face")));
  EXPECT_TRUE(all_finite(solver.state_component_to_host("bz_cell")));
  const Real final_divergence = solver.divergence_b_max();
  ::testing::Test::RecordProperty("divb_linf_after_50_steps",
                                  final_divergence);
  EXPECT_LT(final_divergence, Real{1e-10});
}
