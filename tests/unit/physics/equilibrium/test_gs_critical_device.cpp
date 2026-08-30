// Host-vs-device equivalence for the critical-point search.
//
// The axis position and psi_boundary feed psi_N, which feeds the current, so a
// discrepancy here does not stay small -- it moves the equilibrium. The tests
// therefore check the located points exactly rather than within a tolerance:
// both paths run the same Newton iteration from the same seeds in the same
// order, so equality is the right assertion.
//
// The symmetric case is the one that matters most. An up-down symmetric
// configuration puts two X-points at exactly equal |psi - psi_axis|, and the
// host resolves that tie by scan order. A parallel merge would resolve it
// differently and silently pick the other separatrix, which is why the device
// merge is single-threaded.

#include "quasar/backend/memory.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/critical_points.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::equilibrium::CriticalKind;
using quasar::equilibrium::CriticalPointSet;
using quasar::equilibrium::GsCriticalResult;
using quasar::numerics::EllipticGrid;
using quasar::numerics::ScalarField;

GsCriticalResult run_device(const EllipticGrid& g, const ScalarField& psi) {
  DeviceBuffer<Real> d_psi{psi.size()};
  d_psi.copy_from_host(psi.data(), psi.size());

  quasar::equilibrium::GsOperatorScratch op{g};
  quasar::equilibrium::GsDerivativeFields deriv{g};
  quasar::equilibrium::launch_gs_compute_derivatives(g, d_psi.device_ptr(),
                                                     deriv, op, nullptr);

  quasar::equilibrium::GsCriticalScratch scratch{g};
  quasar::equilibrium::launch_gs_find_critical_points(g, d_psi.device_ptr(),
                                                      deriv, scratch, nullptr);
  return quasar::equilibrium::copy_critical_to_host(scratch, nullptr);
}

void expect_same_point(const quasar::equilibrium::CriticalPoint& host,
                       const quasar::equilibrium::CriticalPoint& dev,
                       const char* what) {
  EXPECT_EQ(host.valid, dev.valid) << what << ": valid";
  if (!host.valid) return;
  EXPECT_EQ(host.kind, dev.kind) << what << ": kind";
  EXPECT_EQ(host.r, dev.r) << what << ": r";
  EXPECT_EQ(host.z, dev.z) << what << ": z";
  EXPECT_EQ(host.psi, dev.psi) << what << ": psi";
}

// A single off-centre well: one O-point, no X-point, so the limited-plasma
// branch of psi_boundary is taken.
ScalarField single_well(const EllipticGrid& g) {
  ScalarField psi = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      const Real dr = (g.r(i) - Real{1.18}) / Real{0.50};
      const Real dz = (g.z(j) - Real{0.07}) / Real{0.44};
      psi[g.index(i, j)] = Real{0.25} * std::exp(-(dr * dr + dz * dz));
    }
  }
  return psi;
}

// Two wells separated by a saddle: produces both an O-point and a genuine
// X-point, exercising the separatrix branch.
ScalarField well_and_saddle(const EllipticGrid& g) {
  ScalarField psi = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      const Real r = g.r(i);
      const Real z = g.z(j);
      const Real a = std::exp(-(std::pow((r - Real{1.05}) / Real{0.33}, 2)
                              + std::pow((z - Real{0.22}) / Real{0.30}, 2)));
      const Real b = std::exp(-(std::pow((r - Real{1.05}) / Real{0.33}, 2)
                              + std::pow((z + Real{0.30}) / Real{0.30}, 2)));
      psi[g.index(i, j)] = Real{0.30} * a + Real{0.21} * b;
    }
  }
  return psi;
}

// Exactly up-down symmetric: the tie-breaking case.
ScalarField symmetric_double(const EllipticGrid& g) {
  ScalarField psi = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      const Real r = g.r(i);
      const Real z = g.z(j);
      const Real a = std::exp(-(std::pow((r - Real{1.10}) / Real{0.32}, 2)
                              + std::pow((z - Real{0.26}) / Real{0.28}, 2)));
      const Real b = std::exp(-(std::pow((r - Real{1.10}) / Real{0.32}, 2)
                              + std::pow((z + Real{0.26}) / Real{0.28}, 2)));
      psi[g.index(i, j)] = Real{0.28} * (a + b);
    }
  }
  return psi;
}

ScalarField many_critical_points(const EllipticGrid& g) {
  constexpr Real pi = Real{3.14159265358979323846};
  ScalarField psi = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      const Real x = (g.r(i) - g.r_min) / (g.r_max - g.r_min);
      const Real y = (g.z(j) - g.z_min) / (g.z_max - g.z_min);
      psi[g.index(i, j)] =
          std::cos(Real{12} * pi * x) * std::cos(Real{12} * pi * y);
    }
  }
  return psi;
}

void compare_case(const EllipticGrid& g, const ScalarField& psi,
                  const char* label, bool require_x_point) {
  SCOPED_TRACE(label);

  const CriticalPointSet host =
      quasar::equilibrium::find_critical_points(g, psi);
  const GsCriticalResult dev = run_device(g, psi);

  ASSERT_TRUE(host.axis.valid) << "host found no axis: case is not exercising "
                                  "the search";
  EXPECT_FALSE(dev.numerical_failure);
  EXPECT_FALSE(dev.x_point_overflow);

  expect_same_point(host.axis, dev.axis, "axis");
  EXPECT_EQ(host.psi_axis, dev.psi_axis) << "psi_axis";
  EXPECT_EQ(host.psi_boundary, dev.psi_boundary) << "psi_boundary";
  EXPECT_EQ(host.has_closed_surface, dev.has_closed_surface)
      << "has_closed_surface";

  ASSERT_EQ(host.x_points.size(), static_cast<std::size_t>(dev.n_x))
      << "X-point count";
  if (require_x_point) {
    ASSERT_GT(dev.n_x, 0) << "case was meant to produce an X-point";
  }
  for (int k = 0; k < dev.n_x; ++k) {
    expect_same_point(host.x_points[static_cast<std::size_t>(k)],
                      dev.x_points[k], "x_point");
  }
}

TEST(GsCriticalDevice, SingleWellMatchesHost) {
  const EllipticGrid g{65, 65, Real{0.3}, Real{2.0}, Real{-0.8}, Real{0.8}};
  compare_case(g, single_well(g), "single well (limited plasma)", false);
}

TEST(GsCriticalDevice, WellAndSaddleMatchesHost) {
  const EllipticGrid g{65, 65, Real{0.3}, Real{2.0}, Real{-0.8}, Real{0.8}};
  compare_case(g, well_and_saddle(g), "well and saddle", true);
}

// The tie-breaking case: two X-points at exactly equal separation in psi from
// the axis. Host and device must agree on WHICH one sets psi_boundary.
TEST(GsCriticalDevice, SymmetricConfigurationBreaksTiesIdentically) {
  const EllipticGrid g{65, 65, Real{0.3}, Real{2.0}, Real{-0.8}, Real{0.8}};
  compare_case(g, symmetric_double(g), "up-down symmetric", false);
}

TEST(GsCriticalDevice, NonSquareGridMatchesHost) {
  const EllipticGrid g{65, 41, Real{0.35}, Real{2.1}, Real{-0.75}, Real{0.85}};
  compare_case(g, well_and_saddle(g), "non-square grid", false);
}

TEST(GsCriticalDevice, FindsRotatedOffGridSaddle) {
  const EllipticGrid g{97, 81, Real{0.7}, Real{2.3}, Real{-0.6}, Real{0.6}};
  constexpr Real saddle_r = Real{1.503};
  constexpr Real saddle_z = Real{0.007};
  ScalarField psi = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      psi[g.index(i, j)] =
          (g.r(i) - saddle_r) * (g.z(j) - saddle_z);
    }
  }

  const CriticalPointSet host =
      quasar::equilibrium::find_critical_points(g, psi);
  const GsCriticalResult dev = run_device(g, psi);

  ASSERT_EQ(host.x_points.size(), 1u);
  ASSERT_EQ(dev.n_x, 1);
  EXPECT_FALSE(dev.numerical_failure);
  EXPECT_FALSE(dev.x_point_overflow);
  expect_same_point(host.x_points.front(), dev.x_points[0], "rotated saddle");
  EXPECT_NEAR(host.x_points.front().r, saddle_r, Real{1e-12});
  EXPECT_NEAR(host.x_points.front().z, saddle_z, Real{1e-12});
}

TEST(GsCriticalDevice, ReportsFixedCapacityOverflow) {
  const EllipticGrid g{257, 257, Real{0.4}, Real{2.4}, Real{-1}, Real{1}};
  const ScalarField psi = many_critical_points(g);

  const CriticalPointSet host =
      quasar::equilibrium::find_critical_points(g, psi);
  ASSERT_GT(host.x_points.size(),
            static_cast<std::size_t>(GsCriticalResult::kMaxXPoints));

  const GsCriticalResult dev = run_device(g, psi);
  EXPECT_FALSE(dev.numerical_failure);
  EXPECT_TRUE(dev.x_point_overflow);
  EXPECT_EQ(dev.n_x, GsCriticalResult::kMaxXPoints);
}

TEST(GsCriticalDevice, ExtremeFiniteFieldRejectsOverflowedDerivatives) {
  const EllipticGrid g{33, 33, Real{0.9}, Real{1.1}, Real{-0.1}, Real{0.1}};
  const Real amplitude =
      Real{0.3} * std::numeric_limits<Real>::max();
  ScalarField psi = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      const Real offset = g.r(i) - Real{1};
      psi[g.index(i, j)] = amplitude * (offset * offset);
      ASSERT_TRUE(std::isfinite(psi[g.index(i, j)]));
    }
  }

  const CriticalPointSet host =
      quasar::equilibrium::find_critical_points(g, psi);
  EXPECT_FALSE(host.axis.valid);
  EXPECT_TRUE(host.x_points.empty());

  const GsCriticalResult dev = run_device(g, psi);
  EXPECT_FALSE(dev.axis.valid);
  EXPECT_EQ(dev.n_x, 0);
  EXPECT_FALSE(dev.x_point_overflow);
  EXPECT_TRUE(dev.numerical_failure);
}

TEST(GsCriticalDevice, ReportsNonfiniteInputAsNumericalFailure) {
  const EllipticGrid g{33, 33, Real{0.9}, Real{1.1}, Real{-0.1}, Real{0.1}};
  ScalarField psi = single_well(g);
  psi[g.index(7, 11)] = std::numeric_limits<Real>::quiet_NaN();

  const GsCriticalResult dev = run_device(g, psi);
  EXPECT_TRUE(dev.numerical_failure);
  EXPECT_FALSE(dev.axis.valid);
  EXPECT_EQ(dev.n_x, 0);
}

// A monotone field has no interior extremum at all. Both paths must report no
// axis rather than inventing one -- this is the "vacuum field has no O-point"
// condition the solver relies on to detect an unconfined state.
TEST(GsCriticalDevice, MonotoneFieldReportsNoAxis) {
  const EllipticGrid g{49, 49, Real{0.4}, Real{1.9}, Real{-0.7}, Real{0.7}};
  ScalarField psi = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      psi[g.index(i, j)] = Real{0.6} * g.r(i) + Real{0.25} * g.z(j);
    }
  }

  const CriticalPointSet host =
      quasar::equilibrium::find_critical_points(g, psi);
  const GsCriticalResult dev = run_device(g, psi);

  ASSERT_FALSE(host.axis.valid) << "monotone field should have no axis";
  EXPECT_FALSE(dev.numerical_failure);
  EXPECT_FALSE(dev.axis.valid);
  EXPECT_EQ(host.has_closed_surface, dev.has_closed_surface);
}

// Repeated launches must give identical results. The finalize pass is
// single-threaded so there is no reduction nondeterminism to worry about, but
// the candidate kernel writes a grid-sized sparse array and this confirms the
// scan sees the same thing every time.
TEST(GsCriticalDevice, ResultIsReproducible) {
  const EllipticGrid g{65, 65, Real{0.3}, Real{2.0}, Real{-0.8}, Real{0.8}};
  const ScalarField psi = well_and_saddle(g);

  const GsCriticalResult first = run_device(g, psi);
  for (int trial = 0; trial < 4; ++trial) {
    const GsCriticalResult again = run_device(g, psi);
    EXPECT_EQ(first.axis.r, again.axis.r) << "trial " << trial;
    EXPECT_EQ(first.axis.z, again.axis.z) << "trial " << trial;
    EXPECT_EQ(first.psi_axis, again.psi_axis) << "trial " << trial;
    EXPECT_EQ(first.psi_boundary, again.psi_boundary) << "trial " << trial;
    EXPECT_EQ(first.n_x, again.n_x) << "trial " << trial;
  }
}

}  // namespace
