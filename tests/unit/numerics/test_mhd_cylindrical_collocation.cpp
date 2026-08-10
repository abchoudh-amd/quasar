#include "quasar/core/grid.hpp"
#include "quasar/numerics/radial_moments.hpp"
#include "quasar/numerics/radial_tables.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::numerics::RadialMomentTarget;
using quasar::numerics::RadialTablesView;

class HostRadialCollocationRows {
 public:
  HostRadialCollocationRows(const Grid2D& grid, int scheme_order)
      : scheme_order_{scheme_order},
        radial_lo_{-grid.nghost},
        radial_count_{grid.pitch()},
        collocation_width_{scheme_order == 5 ? 6 : 8},
        face_to_cell_(static_cast<std::size_t>(radial_count_) *
                      static_cast<std::size_t>(collocation_width_)),
        cell_to_face_(static_cast<std::size_t>(radial_count_) *
                      static_cast<std::size_t>(collocation_width_)) {
    for (int cell = radial_lo_; cell < radial_lo_ + radial_count_; ++cell) {
      const long double reduced_radius =
          static_cast<long double>(grid.r_at_cell_center(cell)) /
          static_cast<long double>(grid.dx());
      const auto row = quasar::numerics::solve_radial_row(
          reduced_radius, collocation_width_, -collocation_width_ / 2,
          RadialMomentTarget::cell_average, 0.5L);
      std::copy_n(
          row.c, collocation_width_,
          face_to_cell_.begin() +
              static_cast<std::ptrdiff_t>(cell - radial_lo_) *
                  collocation_width_);
      const auto face_row = quasar::numerics::solve_radial_row(
          reduced_radius, collocation_width_, -collocation_width_ / 2,
          RadialMomentTarget::point_value, -0.5L);
      std::copy_n(
          face_row.c, collocation_width_,
          cell_to_face_.begin() +
              static_cast<std::ptrdiff_t>(cell - radial_lo_) *
                  collocation_width_);
    }
  }

  RadialTablesView view() const {
    RadialTablesView result{};
    result.active = 1;
    result.scheme_order = scheme_order_;
    result.radial_lo = radial_lo_;
    result.radial_count = radial_count_;
    result.collocation_width = collocation_width_;
    result.r4_face_to_cell = face_to_cell_.data();
    result.r5_cell_to_face = cell_to_face_.data();
    return result;
  }

 private:
  int scheme_order_;
  int radial_lo_;
  int radial_count_;
  int collocation_width_;
  std::vector<Real> face_to_cell_;
  std::vector<Real> cell_to_face_;
};

long double integer_power(long double value, int exponent) {
  long double result = 1.0L;
  for (int k = 0; k < exponent; ++k) result *= value;
  return result;
}

long double binomial_coefficient(int n, int k) {
  long double result = 1.0L;
  for (int factor = 1; factor <= k; ++factor) {
    result *= static_cast<long double>(n - k + factor);
    result /= static_cast<long double>(factor);
  }
  return result;
}

Real analytic_ring_average(const Grid2D& grid, int cell, int degree) {
  const long double center =
      static_cast<long double>(grid.r_at_cell_center(cell));
  const long double half_width =
      static_cast<long double>(grid.dx()) / 2.0L;
  long double weighted_integral = 0.0L;
  for (int power = 0; power <= degree + 1; power += 2) {
    weighted_integral +=
        binomial_coefficient(degree + 1, power) *
        integer_power(center, degree + 1 - power) *
        2.0L * integer_power(half_width, power + 1) /
        static_cast<long double>(power + 1);
  }
  return static_cast<Real>(weighted_integral /
                           (2.0L * center * half_width));
}

void expect_same_bits(Real actual, Real expected) {
  EXPECT_EQ(std::bit_cast<std::uint64_t>(actual),
            std::bit_cast<std::uint64_t>(expected));
}

void expect_cell_bx_polynomial_exactness(int scheme_order, int nghost,
                                         int maximum_degree) {
  const Grid2D grid = Grid2D::from_cell_spacing(
      12, 3, Real{0.125}, Real{0.25}, Real{4}, Real{0}, nghost);
  const HostRadialCollocationRows rows{grid, scheme_order};
  const RadialTablesView view = rows.view();
  std::vector<Real> radial_face_samples(grid.storage_size());

  for (int degree = 0; degree <= maximum_degree; ++degree) {
    for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
      for (int face = -grid.nghost; face < grid.nx + grid.nghost; ++face) {
        radial_face_samples[grid.index(face, j)] = static_cast<Real>(
            integer_power(
                static_cast<long double>(grid.r_at_edge(face)), degree));
      }
    }

    for (int cell = 0; cell < grid.nx; ++cell) {
      const Real expected = analytic_ring_average(grid, cell, degree);
      const Real actual = quasar::mhd::cell_bx(
          grid, radial_face_samples.data(), cell, 1, view);
      EXPECT_NEAR(actual, expected,
                  Real{1e-13} * std::max(Real{1}, std::abs(expected)))
          << "scheme_order=" << scheme_order << " degree=" << degree
          << " cell=" << cell;
    }
  }
}

void expect_one_sided_cell_bx_polynomial_exactness(
    int scheme_order, int nghost, int maximum_degree) {
  const Grid2D grid = Grid2D::from_cell_spacing(
      12, 3, Real{0.125}, Real{0.25}, Real{4}, Real{0}, nghost);
  const HostRadialCollocationRows rows{grid, scheme_order};
  const RadialTablesView view = rows.view();
  std::vector<Real> radial_face_samples(grid.storage_size());
  const int outer_cells[2] = {-grid.nghost,
                              grid.nx + grid.nghost - 1};

  for (int degree = 0; degree <= maximum_degree; ++degree) {
    for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
      for (int face = -grid.nghost; face < grid.nx + grid.nghost; ++face) {
        radial_face_samples[grid.index(face, j)] = static_cast<Real>(
            integer_power(
                static_cast<long double>(grid.r_at_edge(face)), degree));
      }
    }

    for (const int cell : outer_cells) {
      const Real expected = analytic_ring_average(grid, cell, degree);
      const Real actual = quasar::mhd::cell_bx(
          grid, radial_face_samples.data(), cell, 1, view);
      EXPECT_NEAR(actual, expected,
                  Real{5e-12} * std::max(Real{1}, std::abs(expected)))
          << "scheme_order=" << scheme_order << " degree=" << degree
          << " cell=" << cell;
    }
  }
}

TEST(MhdCylindricalCollocation,
     CellBxMp5RecoversRingAveragesThroughDegreeFive) {
  expect_cell_bx_polynomial_exactness(/*scheme_order=*/5, /*nghost=*/3,
                                      /*maximum_degree=*/5);
}

TEST(MhdCylindricalCollocation,
     CellBxMp5UsesSixPointRowsOnAnOverpaddedGrid) {
  expect_cell_bx_polynomial_exactness(/*scheme_order=*/5, /*nghost=*/4,
                                      /*maximum_degree=*/5);
}

TEST(MhdCylindricalCollocation,
     CellToFaceMp5UsesSixPointRowsOnAnOverpaddedGrid) {
  const Grid2D grid = Grid2D::from_cell_spacing(
      12, 3, Real{0.125}, Real{0.25}, Real{4}, Real{0}, /*halo=*/4);
  const HostRadialCollocationRows rows{grid, /*scheme_order=*/5};
  const RadialTablesView view = rows.view();
  std::vector<Real> cell_averages(grid.storage_size());

  for (int degree = 0; degree <= 5; ++degree) {
    for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
      for (int cell = -grid.nghost; cell < grid.nx + grid.nghost; ++cell) {
        cell_averages[grid.index(cell, j)] =
            analytic_ring_average(grid, cell, degree);
      }
    }
    for (int face = 0; face <= grid.nx; ++face) {
      const Real actual = quasar::mhd::cell_averages_to_face(
          grid, /*axis=*/0, face,
          [&](int cell) { return cell_averages[grid.index(cell, 1)]; },
          view);
      const Real expected = static_cast<Real>(integer_power(
          static_cast<long double>(grid.r_at_edge(face)), degree));
      EXPECT_NEAR(actual, expected,
                  Real{1e-13} * std::max(Real{1}, std::abs(expected)))
          << "degree=" << degree << " face=" << face;
    }
  }
}

TEST(MhdCylindricalCollocation,
     CellBxMp7RecoversRingAveragesThroughDegreeSeven) {
  expect_cell_bx_polynomial_exactness(/*scheme_order=*/7, /*nghost=*/4,
                                      /*maximum_degree=*/7);
}

TEST(MhdCylindricalCollocation,
     CellBxMp7OneSidedOuterGhostClosureIsRadiallyWeighted) {
  // The four-point GL rule integrates r times a degree-six polynomial exactly.
  // Both outermost padded cells force the one-sided branch; the high cell also
  // exercises its one-cell extrapolation beyond the final stored face.
  expect_one_sided_cell_bx_polynomial_exactness(
      /*scheme_order=*/7, /*nghost=*/4, /*maximum_degree=*/6);
}

TEST(MhdCylindricalCollocation, CellByIsBitIdenticalWithActiveRadialTables) {
  const Grid2D grid = Grid2D::from_cell_spacing(
      12, 10, Real{0.125}, Real{0.25}, Real{4}, Real{0}, /*halo=*/4);
  const HostRadialCollocationRows rows{grid, 7};
  const RadialTablesView active_view = rows.view();
  std::vector<Real> axial_face_samples(grid.storage_size());
  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
      axial_face_samples[grid.index(i, j)] =
          std::sin(Real{0.17} * static_cast<Real>(i)) +
          std::cos(Real{0.31} * static_cast<Real>(j));
    }
  }

  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
      const Real cartesian =
          quasar::mhd::cell_by(grid, axial_face_samples.data(), i, j);
      const Real with_radial_tables = quasar::mhd::cell_by(
          grid, axial_face_samples.data(), i, j, active_view);
      expect_same_bits(with_radial_tables, cartesian);
    }
  }
}

TEST(MhdCylindricalCollocation, InactiveViewPreservesLegacyOutputsBitForBit) {
  const Grid2D grid = Grid2D::from_cell_spacing(
      12, 10, Real{0.125}, Real{0.25}, Real{4}, Real{0}, /*halo=*/4);
  const HostRadialCollocationRows rows{grid, 7};
  RadialTablesView inactive_view = rows.view();
  inactive_view.active = 0;
  std::vector<Real> radial_face_samples(grid.storage_size());
  std::vector<Real> axial_face_samples(grid.storage_size());
  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
      radial_face_samples[grid.index(i, j)] =
          std::sin(Real{0.23} * static_cast<Real>(i + 2 * j));
      axial_face_samples[grid.index(i, j)] =
          std::cos(Real{0.19} * static_cast<Real>(2 * i - j));
    }
  }

  for (int j = -grid.nghost; j < grid.ny + grid.nghost; ++j) {
    for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
      expect_same_bits(
          quasar::mhd::cell_bx(
              grid, radial_face_samples.data(), i, j, inactive_view),
          quasar::mhd::cell_bx(grid, radial_face_samples.data(), i, j));
      expect_same_bits(
          quasar::mhd::cell_by(
              grid, axial_face_samples.data(), i, j, inactive_view),
          quasar::mhd::cell_by(grid, axial_face_samples.data(), i, j));
    }
  }
}

}  // namespace
