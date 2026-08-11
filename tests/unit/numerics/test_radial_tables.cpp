#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/numerics/finite_volume_quadrature.hpp"
#include "quasar/numerics/radial_moments.hpp"
#include "quasar/numerics/radial_tables.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::numerics::RadialCellMeasure;
using quasar::numerics::RadialMomentTarget;
using quasar::numerics::RadialStencilRow;
using quasar::numerics::RadialTables;
using quasar::numerics::RadialTablesView;

std::vector<Real> read_device(const Real* source, std::size_t count) {
  std::vector<Real> result(count);
  if (count != 0) {
    quasar::backend::device_memcpy_d2h(
        result.data(), source, count * sizeof(Real));
  }
  return result;
}

long double reduced_radius(const Grid2D& grid, int cell) {
  return static_cast<long double>(grid.r_at_cell_center(cell)) /
         static_cast<long double>(grid.dx());
}

long double uniform_cell_average_power(long double centre, int power) {
  const long double lo = centre - 0.5L;
  const long double hi = centre + 0.5L;
  return (std::pow(hi, power + 1) - std::pow(lo, power + 1)) /
         static_cast<long double>(power + 1);
}

void expect_row(const std::vector<Real>& actual, std::size_t start,
                const RadialStencilRow& expected) {
  for (int k = 0; k < expected.width; ++k) {
    EXPECT_EQ(actual[start + static_cast<std::size_t>(k)], expected.c[k]);
  }
}

TEST(RadialTables, DefaultOwnerProducesInactiveNullView) {
  const RadialTables owner{};
  for (const RadialTablesView view : {owner.view(), owner.host_view()}) {
    EXPECT_EQ(view.active, 0);
    EXPECT_EQ(view.radial_count, 0);
    EXPECT_EQ(view.r1_left, nullptr);
    EXPECT_EQ(view.r1_mphi_left, nullptr);
    EXPECT_EQ(view.r1_bphi_left, nullptr);
    EXPECT_EQ(view.r2_points, nullptr);
    EXPECT_EQ(view.r2_mphi_points, nullptr);
    EXPECT_EQ(view.r2_bphi_points, nullptr);
    EXPECT_EQ(view.r3_weights, nullptr);
    EXPECT_EQ(view.r3_mphi_weights, nullptr);
    EXPECT_EQ(view.r3_bphi_weights, nullptr);
    EXPECT_EQ(view.r5_bphi_cell_to_face, nullptr);
    EXPECT_EQ(view.r6_limiter, nullptr);
    EXPECT_EQ(view.r6_mphi_limiter, nullptr);
    EXPECT_EQ(view.r6_bphi_limiter, nullptr);
  }
}

TEST(RadialTables, HostViewRowsMatchDeviceRowsBitForBit) {
  const Grid2D grid = Grid2D::from_cell_spacing(
      8, 2, Real{0.125}, Real{0.5}, Real{0}, Real{0}, /*halo=*/4);
  const RadialTables owner{grid, /*scheme_order=*/7};
  const RadialTablesView device = owner.view();
  const RadialTablesView host = owner.host_view();

  EXPECT_EQ(host.active, device.active);
  EXPECT_EQ(host.scheme_order, device.scheme_order);
  EXPECT_EQ(host.radial_lo, device.radial_lo);
  EXPECT_EQ(host.radial_count, device.radial_count);
  EXPECT_EQ(host.reconstruction_width, device.reconstruction_width);
  EXPECT_EQ(host.quadrature_nodes, device.quadrature_nodes);
  EXPECT_EQ(host.collocation_width, device.collocation_width);
  EXPECT_NE(host.r4_face_to_cell, device.r4_face_to_cell);

  const std::size_t rows = static_cast<std::size_t>(host.radial_count);
  const auto expect_family = [&](const Real* host_rows,
                                 const Real* device_rows,
                                 std::size_t count) {
    ASSERT_NE(host_rows, nullptr);
    ASSERT_NE(device_rows, nullptr);
    const std::vector<Real> staged = read_device(device_rows, count);
    EXPECT_TRUE(std::equal(staged.begin(), staged.end(), host_rows));
  };
  expect_family(host.r1_left, device.r1_left,
                rows * static_cast<std::size_t>(host.reconstruction_width));
  expect_family(host.r1_right, device.r1_right,
                rows * static_cast<std::size_t>(host.reconstruction_width));
  expect_family(host.r1_mphi_left, device.r1_mphi_left,
                rows * static_cast<std::size_t>(host.reconstruction_width));
  expect_family(host.r1_mphi_right, device.r1_mphi_right,
                rows * static_cast<std::size_t>(host.reconstruction_width));
  expect_family(host.r1_bphi_left, device.r1_bphi_left,
                rows * static_cast<std::size_t>(host.reconstruction_width));
  expect_family(host.r1_bphi_right, device.r1_bphi_right,
                rows * static_cast<std::size_t>(host.reconstruction_width));
  expect_family(host.r2_points, device.r2_points,
                rows * static_cast<std::size_t>(host.quadrature_nodes) *
                    static_cast<std::size_t>(host.reconstruction_width));
  expect_family(host.r2_mphi_points, device.r2_mphi_points,
                rows * static_cast<std::size_t>(host.quadrature_nodes) *
                    static_cast<std::size_t>(host.reconstruction_width));
  expect_family(host.r2_bphi_points, device.r2_bphi_points,
                rows * static_cast<std::size_t>(host.quadrature_nodes) *
                    static_cast<std::size_t>(host.reconstruction_width));
  expect_family(host.r3_weights, device.r3_weights,
                rows * static_cast<std::size_t>(host.quadrature_nodes));
  expect_family(host.r3_mphi_weights, device.r3_mphi_weights,
                rows * static_cast<std::size_t>(host.quadrature_nodes));
  expect_family(host.r3_bphi_weights, device.r3_bphi_weights,
                rows * static_cast<std::size_t>(host.quadrature_nodes));
  expect_family(host.r4_face_to_cell, device.r4_face_to_cell,
                rows * static_cast<std::size_t>(host.collocation_width));
  expect_family(host.r5_cell_to_face, device.r5_cell_to_face,
                rows * static_cast<std::size_t>(host.collocation_width));
  expect_family(host.r5_bphi_cell_to_face, device.r5_bphi_cell_to_face,
                rows * static_cast<std::size_t>(host.collocation_width));
  expect_family(host.r6_limiter, device.r6_limiter, rows * std::size_t{4});
  expect_family(
      host.r6_mphi_limiter, device.r6_mphi_limiter, rows * std::size_t{4});
  expect_family(
      host.r6_bphi_limiter, device.r6_bphi_limiter, rows * std::size_t{4});
}

TEST(RadialTables, AxisR1RowsMeetStrictAcceptanceThreshold) {
  const Grid2D axis = Grid2D::from_cell_spacing(
      8, 2, Real{0.125}, Real{0.5}, Real{0}, Real{0}, /*halo=*/4);
  for (const int width : {5, 7}) {
    const RadialStencilRow left = quasar::numerics::solve_radial_row(
        reduced_radius(axis, /*cell=*/-1), width, -width / 2,
        RadialMomentTarget::point_value, 0.5L);
    const RadialStencilRow right = quasar::numerics::solve_radial_row(
        reduced_radius(axis, /*cell=*/0), width, -width / 2,
        RadialMomentTarget::point_value, -0.5L);
    EXPECT_TRUE(std::isfinite(left.residual));
    EXPECT_TRUE(std::isfinite(right.residual));
    EXPECT_LE(left.residual, Real{1e-11});
    EXPECT_LE(right.residual, Real{1e-11});
  }
}

TEST(RadialTables, CollocationWidthFollowsSchemeNotOverpadding) {
  const Grid2D overpadded = Grid2D::from_cell_spacing(
      8, 2, Real{0.125}, Real{0.5}, Real{1}, Real{0}, /*halo=*/4);
  const RadialTables owner{overpadded, /*scheme_order=*/5};
  const RadialTablesView view = owner.view();

  EXPECT_EQ(view.reconstruction_width, 5);
  EXPECT_EQ(view.quadrature_nodes, 3);
  EXPECT_EQ(view.collocation_width, 6);
}

TEST(RadialTables, OverpaddedMp5DeviceRowsMatchNativeSixPointHostRows) {
  const Grid2D overpadded = Grid2D::from_cell_spacing(
      12, 2, Real{0.125}, Real{0.5}, Real{1}, Real{0}, /*halo=*/4);
  const RadialTables owner{overpadded, /*scheme_order=*/5};
  const RadialTablesView view = owner.view();
  ASSERT_EQ(view.collocation_width, 6);

  const std::size_t rows = static_cast<std::size_t>(view.radial_count);
  const auto r4 = read_device(view.r4_face_to_cell, rows * std::size_t{6});
  const auto r5 = read_device(view.r5_cell_to_face, rows * std::size_t{6});
  const auto r5_bphi =
      read_device(view.r5_bphi_cell_to_face, rows * std::size_t{6});
  for (const int logical : {-2, 0, 5, 11}) {
    SCOPED_TRACE(logical);
    ASSERT_TRUE(view.contains(logical));
    const std::size_t offset =
        static_cast<std::size_t>(view.row_index(logical)) * std::size_t{6};
    const long double rho = reduced_radius(overpadded, logical);
    expect_row(
        r4, offset,
        quasar::numerics::solve_radial_row(
            rho, 6, -3, RadialMomentTarget::cell_average, 0.5L));
    expect_row(
        r5, offset,
        quasar::numerics::solve_radial_row(
            rho, 6, -3, RadialMomentTarget::point_value, -0.5L));
    expect_row(
        r5_bphi, offset,
        quasar::numerics::solve_radial_row(
            rho, 6, -3, RadialMomentTarget::point_value,
            quasar::numerics::RadialCellMeasure::uniform, -0.5L));
  }
}

TEST(RadialTables, BphiRowsRecoverUniformCellMomentsAtPointsAndFaces) {
  const Grid2D grid = Grid2D::from_cell_spacing(
      12, 3, Real{0.125}, Real{0.25}, Real{1}, Real{0}, /*halo=*/4);

  for (const int order : {5, 7}) {
    SCOPED_TRACE(order);
    const RadialTables owner{grid, order};
    const RadialTablesView view = owner.host_view();
    const int width = view.reconstruction_width;
    const int half = width / 2;
    constexpr int cell = 5;
    constexpr int power = 4;
    const long double rho = reduced_radius(grid, cell);

    for (int node = 0; node < view.quadrature_nodes; ++node) {
      const long double xi = order == 5
          ? static_cast<long double>(
                quasar::numerics::kMp5TransverseNodes[node])
          : static_cast<long double>(
                quasar::numerics::kMp7TransverseNodes[node]);
      const Real recovered = quasar::numerics::weighted_quadrature_value(
          view.r2_bphi_row(cell, node), width, [&](int k) {
            return static_cast<Real>(uniform_cell_average_power(
                rho + static_cast<long double>(k - half), power));
          });
      EXPECT_NEAR(recovered, static_cast<Real>(std::pow(rho + xi, power)),
                  Real{2e-11});
    }

    // Exercise the production B0 interface loader, not just the raw R5 row:
    // its radial-face B0_phi path must select the flat-measure family.
    const std::size_t size = grid.storage_size();
    std::vector<Real> zero(size, Real{0});
    std::vector<Real> b0phi(size);
    for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
      for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
        b0phi[grid.index(i, j)] = static_cast<Real>(
            uniform_cell_average_power(reduced_radius(grid, i), power));
      }
    }
    constexpr int face = cell;
    const auto background = quasar::mhd::load_interface_background(
        grid, /*dir=*/0, zero.data(), zero.data(), b0phi.data(), face,
        /*j=*/1, view, /*collocation_order=*/0);
    const long double face_rho = static_cast<long double>(grid.r_at_edge(face)) /
                                 static_cast<long double>(grid.dx());
    EXPECT_NEAR(background.b0z,
                static_cast<Real>(std::pow(face_rho, power)), Real{2e-11});
  }
}

TEST(RadialTables, LimiterRowsCarryMeasureSpecificAxisGeometry) {
  const Grid2D axis = Grid2D::from_cell_spacing(
      8, 2, Real{0.125}, Real{0.5}, Real{0}, Real{0}, /*halo=*/4);
  const RadialTables owner{axis, /*scheme_order=*/7};
  const RadialTablesView view = owner.view();
  const std::size_t cell_zero =
      static_cast<std::size_t>(view.row_index(/*cell=*/0)) * 4;
  const std::size_t axis_face =
      static_cast<std::size_t>(view.row_index(/*cell=*/-1)) * 4;
  const std::size_t count =
      static_cast<std::size_t>(view.radial_count) * std::size_t{4};
  const auto annular = read_device(
      view.r6_limiter,
      count);
  const auto angular = read_device(view.r6_mphi_limiter, count);
  const auto uniform = read_device(view.r6_bphi_limiter, count);

  EXPECT_EQ(annular[cell_zero], Real{5} / Real{8});
  EXPECT_EQ(annular[cell_zero + 1], Real{3} / Real{8});
  EXPECT_EQ(annular[cell_zero + 2], Real{1} / Real{4});
  EXPECT_EQ(annular[cell_zero + 3], Real{25} / Real{44});

  EXPECT_EQ(angular[cell_zero], Real{17} / Real{24});
  // Rows enforce a bit-exact partition of unity; the binary64 encoding of
  // 7/24 is therefore formed as 1 - fl(17/24).
  EXPECT_EQ(angular[cell_zero + 1],
            Real{1} - Real{17} / Real{24});
  EXPECT_EQ(angular[cell_zero + 2], Real{1} / Real{6});
  EXPECT_EQ(angular[cell_zero + 3], Real{19} / Real{30});
  EXPECT_EQ(angular[axis_face], Real{1} / Real{2});
  EXPECT_EQ(angular[axis_face + 1], Real{1} / Real{2});
  EXPECT_EQ(angular[axis_face + 2], Real{7} / Real{8});
  EXPECT_EQ(angular[axis_face + 3], Real{7} / Real{8});

  for (const std::size_t offset : {axis_face, cell_zero}) {
    for (std::size_t k = 0; k < 4; ++k) {
      EXPECT_EQ(uniform[offset + k], Real{1} / Real{2});
    }
  }
}

TEST(RadialTables, DeviceReadbackEqualsHostRowsBitForBit) {
  const Grid2D grid = Grid2D::from_cell_spacing(
      32, 1, Real{0.125}, Real{1}, Real{0.5}, Real{0}, /*halo=*/4);
  const RadialTables owner{grid, 7};
  const RadialTablesView view = owner.view();

  ASSERT_EQ(view.active, 1);
  ASSERT_EQ(view.scheme_order, 7);
  ASSERT_EQ(view.radial_lo, -4);
  ASSERT_EQ(view.radial_count, 40);
  ASSERT_EQ(view.reconstruction_width, 7);
  ASSERT_EQ(view.quadrature_nodes, 4);
  ASSERT_EQ(view.collocation_width, 8);
  // Six R1 rows, three R2/R3 families, one R4 plus two R5 rows, and
  // three four-entry R6 rows are stored for every radial index.
  EXPECT_EQ(6 * view.reconstruction_width +
                3 * view.quadrature_nodes * view.reconstruction_width +
                3 * view.quadrature_nodes + 3 * view.collocation_width +
                3 * 4,
            174);

  const std::size_t rows = static_cast<std::size_t>(view.radial_count);
  const auto r1_left = read_device(
      view.r1_left, rows * static_cast<std::size_t>(view.reconstruction_width));
  const auto r1_right = read_device(
      view.r1_right, rows * static_cast<std::size_t>(view.reconstruction_width));
  const auto r2 = read_device(
      view.r2_points,
      rows * static_cast<std::size_t>(view.quadrature_nodes) *
          static_cast<std::size_t>(view.reconstruction_width));
  const auto r3 = read_device(
      view.r3_weights,
      rows * static_cast<std::size_t>(view.quadrature_nodes));
  const auto r4 = read_device(
      view.r4_face_to_cell,
      rows * static_cast<std::size_t>(view.collocation_width));
  const auto r5 = read_device(
      view.r5_cell_to_face,
      rows * static_cast<std::size_t>(view.collocation_width));
  const auto r5_bphi = read_device(
      view.r5_bphi_cell_to_face,
      rows * static_cast<std::size_t>(view.collocation_width));
  const auto r6 = read_device(view.r6_limiter, rows * std::size_t{4});
  const auto r6_mphi =
      read_device(view.r6_mphi_limiter, rows * std::size_t{4});
  const auto r6_bphi =
      read_device(view.r6_bphi_limiter, rows * std::size_t{4});

  for (int logical = view.radial_lo;
       logical < view.radial_lo + view.radial_count; ++logical) {
    const std::size_t row =
        static_cast<std::size_t>(logical - view.radial_lo);
    const long double rho = reduced_radius(grid, logical);

    expect_row(
        r1_left, row * 7,
        quasar::numerics::solve_radial_row(
            reduced_radius(grid, logical - 1), 7, -3,
            RadialMomentTarget::point_value, 0.5L));
    expect_row(
        r1_right, row * 7,
        quasar::numerics::solve_radial_row(
            rho, 7, -3, RadialMomentTarget::point_value, -0.5L));

    for (int node = 0; node < 4; ++node) {
      expect_row(
          r2, (row * 4 + static_cast<std::size_t>(node)) * 7,
          quasar::numerics::solve_radial_row(
              rho, 7, -3, RadialMomentTarget::point_value,
              static_cast<long double>(
                  quasar::numerics::kMp7TransverseNodes[node])));
    }
    expect_row(
        r3, row * 4,
        quasar::numerics::radial_gauss_weights(
            rho, 4, quasar::numerics::kMp7TransverseNodes,
            quasar::numerics::kMp7TransverseGaussWeights));
    expect_row(
        r4, row * 8,
        quasar::numerics::solve_radial_row(
            rho, 8, -4, RadialMomentTarget::cell_average, 0.5L));
    expect_row(
        r5, row * 8,
        quasar::numerics::solve_radial_row(
            rho, 8, -4, RadialMomentTarget::point_value, -0.5L));
    expect_row(
        r5_bphi, row * 8,
        quasar::numerics::solve_radial_row(
            rho, 8, -4, RadialMomentTarget::point_value,
            quasar::numerics::RadialCellMeasure::uniform, -0.5L));
    const long double right_rho = reduced_radius(grid, logical + 1);
    // Row i begins with the pair for the face between cells i and i+1. The
    // final entries are the independently derived left/right one-sided v_lc
    // factors under the same measure as that pair.
    const auto expect_limiter_family = [&](const std::vector<Real>& actual,
                                           RadialCellMeasure measure) {
      expect_row(
          actual, row * 4,
          quasar::numerics::solve_radial_row(
              rho, 2, 0, RadialMomentTarget::point_value, measure, 0.5L));
      const long double left_center =
          quasar::numerics::normalized_cell_moment(rho, 1, measure);
      const long double left_neighbor =
          quasar::numerics::normalized_cell_moment(rho - 1.0L, 1, measure);
      const Real left_beta = static_cast<Real>(
          (rho + 0.5L - left_center) / (left_center - left_neighbor));
      const long double right_center =
          quasar::numerics::normalized_cell_moment(right_rho, 1, measure);
      const long double right_neighbor =
          quasar::numerics::normalized_cell_moment(
              right_rho + 1.0L, 1, measure);
      const Real right_beta = static_cast<Real>(
          (right_rho - 0.5L - right_center) /
          (right_center - right_neighbor));
      EXPECT_EQ(actual[row * 4 + 2], left_beta);
      EXPECT_EQ(actual[row * 4 + 3], right_beta);
    };
    expect_limiter_family(r6, RadialCellMeasure::annular);
    expect_limiter_family(r6_mphi, RadialCellMeasure::angular_momentum);
    expect_limiter_family(r6_bphi, RadialCellMeasure::uniform);
  }
}

TEST(RadialTables, TileLocalIndexYieldsGlobalRadius) {
  const Real dr = std::ldexp(Real{1.1}, -20);
  constexpr int tile_start = 5;
  const Grid2D whole = Grid2D::from_cell_spacing(
      32, 1, dr, Real{1}, Real{0}, Real{0}, /*halo=*/4);
  Grid2D tile = Grid2D::from_cell_spacing(
      8, 1, dr, Real{1},
      std::fma(static_cast<Real>(tile_start), dr, Real{0}), Real{0},
      /*halo=*/4);
  // This is the concrete nested-FMA counterexample that a dyadic spacing
  // hides: the local and global expressions disagree before the partition
  // metadata is supplied.
  EXPECT_NE(tile.r_at_cell_center(-4),
            whole.r_at_cell_center(tile_start - 4));
  tile.global_origin_x = whole.origin_x;
  tile.global_cell_offset_x = tile_start;
  tile.has_global_x_mapping = 1;
  EXPECT_EQ(tile.r_at_cell_center(-4),
            whole.r_at_cell_center(tile_start - 4));
  const RadialTables whole_owner{whole, 7};
  const RadialTables tile_owner{tile, 7};
  const auto whole_view = whole_owner.view();
  const auto tile_view = tile_owner.view();

  const auto compare_family = [&](const Real* whole_device,
                                  const Real* tile_device, int row_width) {
    const auto whole_host = read_device(
        whole_device, static_cast<std::size_t>(whole_view.radial_count) *
                          static_cast<std::size_t>(row_width));
    const auto tile_host = read_device(
        tile_device, static_cast<std::size_t>(tile_view.radial_count) *
                         static_cast<std::size_t>(row_width));
    for (int local = tile_view.radial_lo;
         local < tile_view.radial_lo + tile_view.radial_count; ++local) {
      const int global = tile_start + local;
      ASSERT_TRUE(whole_view.contains(global));
      for (int k = 0; k < row_width; ++k) {
        const std::size_t whole_offset =
            static_cast<std::size_t>(whole_view.row_index(global)) *
                static_cast<std::size_t>(row_width) +
            static_cast<std::size_t>(k);
        const std::size_t tile_offset =
            static_cast<std::size_t>(tile_view.row_index(local)) *
                static_cast<std::size_t>(row_width) +
            static_cast<std::size_t>(k);
        EXPECT_EQ(whole_host[whole_offset], tile_host[tile_offset]);
      }
    }
  };

  compare_family(whole_view.r1_left, tile_view.r1_left, 7);
  compare_family(whole_view.r1_right, tile_view.r1_right, 7);
  compare_family(whole_view.r1_mphi_left, tile_view.r1_mphi_left, 7);
  compare_family(whole_view.r1_mphi_right, tile_view.r1_mphi_right, 7);
  compare_family(whole_view.r1_bphi_left, tile_view.r1_bphi_left, 7);
  compare_family(whole_view.r1_bphi_right, tile_view.r1_bphi_right, 7);
  compare_family(whole_view.r2_points, tile_view.r2_points, 4 * 7);
  compare_family(whole_view.r2_mphi_points, tile_view.r2_mphi_points, 4 * 7);
  compare_family(whole_view.r2_bphi_points, tile_view.r2_bphi_points, 4 * 7);
  compare_family(whole_view.r3_weights, tile_view.r3_weights, 4);
  compare_family(whole_view.r3_mphi_weights, tile_view.r3_mphi_weights, 4);
  compare_family(whole_view.r3_bphi_weights, tile_view.r3_bphi_weights, 4);
  compare_family(whole_view.r4_face_to_cell, tile_view.r4_face_to_cell, 8);
  compare_family(whole_view.r5_cell_to_face, tile_view.r5_cell_to_face, 8);
  compare_family(
      whole_view.r5_bphi_cell_to_face,
      tile_view.r5_bphi_cell_to_face, 8);
  compare_family(whole_view.r6_limiter, tile_view.r6_limiter, 4);
  compare_family(
      whole_view.r6_mphi_limiter, tile_view.r6_mphi_limiter, 4);
  compare_family(
      whole_view.r6_bphi_limiter, tile_view.r6_bphi_limiter, 4);
}

TEST(RadialTables, RejectsIllConditionedRow) {
  // This deliberately invalid annular halo contains a cell centred exactly at
  // r=0.  Its |r|-weighted GL rule cannot meet the R3 2n-2 exactness contract;
  // the owner must reject the diagnostic residual rather than upload it.
  const Grid2D grid = Grid2D::from_cell_spacing(
      8, 1, Real{1}, Real{1}, Real{0.5}, Real{0}, /*halo=*/4);
  EXPECT_THROW((RadialTables{grid, 7}), std::runtime_error);
}

}  // namespace
