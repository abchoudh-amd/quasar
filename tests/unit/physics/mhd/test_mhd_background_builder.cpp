// Device construction of the prescribed background field B0.
//
// The deck-facing behaviour of the five sources is exercised from Python
// (tests/python/test_mhd_background_io.py), which drives real decks against
// closed-form references. This file covers the parts of the C++ surface that a
// deck cannot reach:
//
//   * the affine lowering of a registered analytic profile, including the
//     refusal that protects a nonlinear one from being silently linearized --
//     which needs a profile class, not a deck string;
//   * the annular vacuum projection as a device solve: that it leaves an
//     already-harmonic potential alone, that it drives a perturbed one back to
//     the discrete vacuum operator's null space, and that it is bitwise
//     reproducible run to run.
//
// The projection's reference is computed here in long double, independently of
// the kernel, and the comparison is against that oracle rather than against a
// stored golden.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/mhd_background_profile.hpp"
#include "quasar/physics/mhd/background_builder.hpp"
#include "quasar/physics/mhd/kernels.hpp"

#include <cmath>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::mhd::MhdBackgroundBuildSpec;
using quasar::mhd::MhdBackgroundField;

bool device_available() { return quasar::backend::device_count() > 0; }

// A deliberately quadratic profile. It satisfies the IMhdBackgroundProfile
// signature but not the affine condition its sample() contract relies on, so
// the lowering must refuse it by name instead of fitting a plane through three
// probes and calling that the answer.
class QuadraticProfile final : public quasar::numerics::IMhdBackgroundProfile {
 public:
  Real sample(int comp, Real x, Real y) const override {
    if (comp == 0) return x * x;
    if (comp == 1) return -Real{2} * x * y;
    return Real{0};
  }
};

Grid2D annular_grid() {
  // Both extents prime-ish and unequal so a transposed index would not survive,
  // and r_min - nghost*dr comfortably positive.
  return Grid2D{13, 11, Real{0.65}, Real{0.55}, Real{0.7}, Real{-0.25},
                /*halo=*/3};
}

std::size_t corner_count(const Grid2D& g) {
  return static_cast<std::size_t>(g.pitch() + 1) *
         static_cast<std::size_t>(g.height() + 1);
}

Real corner_radius(const Grid2D& g, int i) {
  return g.origin_x + static_cast<Real>(i - g.nghost) * g.dx();
}

// Residual of the discrete annular vacuum operator, in long double:
//   D_r[(1/r) D_r psi] + D_zz(A),  psi = r A.
// Reported as max|row residual| over interior nodes, which is exactly the
// quantity the projection's stopping test bounds.
long double vacuum_residual_linf(const Grid2D& g,
                                 const std::vector<Real>& a_corners) {
  const int nodes_x = g.pitch() + 1;
  const int nodes_y = g.height() + 1;
  const auto at = [&](int i, int j) {
    return static_cast<long double>(
        a_corners[static_cast<std::size_t>(j) *
                      static_cast<std::size_t>(nodes_x) +
                  static_cast<std::size_t>(i)]);
  };
  const long double dx = static_cast<long double>(g.dx());
  const long double dy = static_cast<long double>(g.dy());
  long double worst = 0.0L;
  for (int j = 1; j < nodes_y - 1; ++j) {
    for (int i = 1; i < nodes_x - 1; ++i) {
      const long double r = static_cast<long double>(corner_radius(g, i));
      const long double r_east = static_cast<long double>(corner_radius(g, i + 1));
      const long double r_west = static_cast<long double>(corner_radius(g, i - 1));
      const long double ring_e = (r_east - r) * (0.5L * r_east + 0.5L * r);
      const long double ring_w = (r - r_west) * (0.5L * r + 0.5L * r_west);
      const long double psi_c = r * at(i, j);
      const long double psi_e = r_east * at(i + 1, j);
      const long double psi_w = r_west * at(i - 1, j);
      const long double radial =
          ((psi_e - psi_c) / ring_e - (psi_c - psi_w) / ring_w) / dx;
      const long double axial =
          (at(i, j + 1) - 2.0L * at(i, j) + at(i, j - 1)) / (dy * dy);
      const long double residual = std::fabs(radial + axial);
      if (residual > worst) worst = residual;
    }
  }
  return worst;
}

// The projection's own convergence scale: max|B| implied by the raw potential,
// divided by the finer spacing. Reproduced here so the assertion below is
// stated in the same units the contract is.
long double field_scale(const Grid2D& g, const std::vector<Real>& a_corners) {
  const int nodes_x = g.pitch() + 1;
  const int nodes_y = g.height() + 1;
  const auto at = [&](int i, int j) {
    return static_cast<long double>(
        a_corners[static_cast<std::size_t>(j) *
                      static_cast<std::size_t>(nodes_x) +
                  static_cast<std::size_t>(i)]);
  };
  const long double dx = static_cast<long double>(g.dx());
  const long double dy = static_cast<long double>(g.dy());
  long double worst = 0.0L;
  for (int j = 0; j < nodes_y; ++j) {
    for (int i = 0; i < nodes_x; ++i) {
      if (j + 1 < nodes_y) {
        worst = std::fmax(worst, std::fabs(-(at(i, j + 1) - at(i, j)) / dy));
      }
      if (i + 1 < nodes_x) {
        const long double r_lo = static_cast<long double>(corner_radius(g, i));
        const long double r_hi = static_cast<long double>(corner_radius(g, i + 1));
        const long double value =
            (at(i + 1, j) - at(i, j)) / (r_hi - r_lo) +
            (0.5L * at(i + 1, j) + 0.5L * at(i, j)) / (0.5L * r_hi + 0.5L * r_lo);
        worst = std::fmax(worst, std::fabs(value));
      }
    }
  }
  return std::fmax(1.0L, worst / std::fmin(dx, dy));
}

MhdBackgroundBuildSpec annular_spec(const Grid2D& g, bool project) {
  MhdBackgroundBuildSpec spec;
  spec.grid = g;
  spec.cylindrical = 1;
  spec.magnetic_scale = Real{1};
  spec.vacuum_project = project ? 1 : 0;
  // Every side ignored: this file is about the construction, and the closure
  // rules have their own coverage in test_mhd_background_field.cpp.
  for (int side = 0; side < 4; ++side) spec.field_modes[side] = 0;
  return spec;
}

std::vector<Real> project(const Grid2D& g, std::vector<Real> a_host) {
  quasar::backend::DeviceBuffer<Real> a_device(a_host.size());
  a_device.copy_from_host(a_host.data(), a_host.size());
  MhdBackgroundField<Real> b0{g};
  quasar::mhd::build_background_from_corner_potential(annular_spec(g, true),
                                                      a_device, b0);
  a_device.copy_to_host(a_host.data(), a_host.size());
  quasar::backend::device_synchronize(nullptr);
  return a_host;
}

// -- Affine profile lowering --------------------------------------------------

TEST(MhdBackgroundBuilder, AffineLoweringReproducesTheRegisteredProfileExactly) {
  auto profile =
      quasar::Registry<quasar::numerics::IMhdBackgroundProfile>::instance()
          .create("linear_vacuum");
  ASSERT_TRUE(profile->set_parameter("gradient", Real{1.25}));
  ASSERT_TRUE(profile->set_parameter("shear", Real{-0.4}));
  const auto lowered =
      quasar::mhd::lower_affine_background_profile(*profile, "linear_vacuum");

  // Exact equality, not a tolerance: the lowering is only admissible because
  // the profile is affine, and an affine reconstruction of an affine function
  // through three exact probes reproduces it bit for bit.
  for (int comp = 0; comp < 3; ++comp) {
    for (const Real x : {Real{-3.5}, Real{0}, Real{0.125}, Real{7}}) {
      for (const Real y : {Real{-1}, Real{0}, Real{2.75}}) {
        EXPECT_EQ(lowered.constant[comp] +
                      (lowered.slope_x[comp] * x + lowered.slope_y[comp] * y),
                  profile->sample(comp, x, y))
            << "comp=" << comp << " x=" << x << " y=" << y;
      }
    }
  }
}

TEST(MhdBackgroundBuilder, UniformProfileLowersToConstantsWithZeroSlopes) {
  auto profile =
      quasar::Registry<quasar::numerics::IMhdBackgroundProfile>::instance()
          .create("uniform");
  ASSERT_TRUE(profile->set_parameter("bx0", Real{0.25}));
  ASSERT_TRUE(profile->set_parameter("by0", Real{-1.5}));
  ASSERT_TRUE(profile->set_parameter("bz0", Real{3}));
  const auto lowered =
      quasar::mhd::lower_affine_background_profile(*profile, "uniform");
  EXPECT_EQ(lowered.constant[0], Real{0.25});
  EXPECT_EQ(lowered.constant[1], Real{-1.5});
  EXPECT_EQ(lowered.constant[2], Real{3});
  for (int comp = 0; comp < 3; ++comp) {
    EXPECT_EQ(lowered.slope_x[comp], Real{0});
    EXPECT_EQ(lowered.slope_y[comp], Real{0});
  }
}

TEST(MhdBackgroundBuilder, NonAffineProfileIsRefusedRatherThanLinearized) {
  const QuadraticProfile profile;
  EXPECT_THROW(
      quasar::mhd::lower_affine_background_profile(profile, "quadratic"),
      std::invalid_argument);
}

// -- Vacuum projection --------------------------------------------------------

TEST(MhdBackgroundBuilder, ProjectionLeavesADiscreteVacuumPotentialAlone) {
  if (!device_available()) GTEST_SKIP() << "no HIP device";
  const Grid2D g = annular_grid();
  const int nodes_x = g.pitch() + 1;
  const int nodes_y = g.height() + 1;
  std::vector<Real> a(corner_count(g));
  // A_phi = C r has psi = C r^2, whose annular radial operator is a constant in
  // r and whose axial operator vanishes -- it is already in the discrete
  // operator's null space, so the projection has nothing to correct.
  for (int j = 0; j < nodes_y; ++j) {
    for (int i = 0; i < nodes_x; ++i) {
      a[static_cast<std::size_t>(j) * static_cast<std::size_t>(nodes_x) +
        static_cast<std::size_t>(i)] = Real{0.25} * corner_radius(g, i);
    }
  }
  const std::vector<Real> before = a;
  const std::vector<Real> after = project(g, a);

  const long double scale = field_scale(g, before);
  // Same 5e-11 relative contract the projection stops on.
  const long double allowed = 5.0e-11L * scale;
  EXPECT_LE(vacuum_residual_linf(g, before), allowed);
  EXPECT_LE(vacuum_residual_linf(g, after), allowed);
  for (std::size_t k = 0; k < after.size(); ++k) {
    // Interior values may move by the solver's own tolerance; the boundary is
    // fixed data and must be untouched exactly.
    const int i = static_cast<int>(k % static_cast<std::size_t>(nodes_x));
    const int j = static_cast<int>(k / static_cast<std::size_t>(nodes_x));
    if (i == 0 || j == 0 || i == nodes_x - 1 || j == nodes_y - 1) {
      EXPECT_EQ(after[k], before[k]) << "boundary node i=" << i << " j=" << j;
    }
  }
}

TEST(MhdBackgroundBuilder, ProjectionDrivesAPerturbedPotentialIntoTheNullSpace) {
  if (!device_available()) GTEST_SKIP() << "no HIP device";
  const Grid2D g = annular_grid();
  const int nodes_x = g.pitch() + 1;
  const int nodes_y = g.height() + 1;
  std::vector<Real> a(corner_count(g));
  for (int j = 0; j < nodes_y; ++j) {
    const Real z = g.origin_y + static_cast<Real>(j - g.nghost) * g.dy();
    for (int i = 0; i < nodes_x; ++i) {
      const Real r = corner_radius(g, i);
      // An axisymmetric dipole is continuum-vacuum on an r>0 domain, so its
      // sampled form carries only an O(h^2) discrete defect -- large compared
      // with the 5e-11 target, small enough that the harmonic continuation of
      // the same boundary is a nearby field.
      a[static_cast<std::size_t>(j) * static_cast<std::size_t>(nodes_x) +
        static_cast<std::size_t>(i)] =
          r / std::pow(r * r + z * z, Real{1.5});
    }
  }
  const std::vector<Real> before = a;
  const long double allowed = 5.0e-11L * field_scale(g, before);
  ASSERT_GT(vacuum_residual_linf(g, before), allowed)
      << "the unprojected sample must actually violate the operator, or this "
         "test proves nothing";

  const std::vector<Real> after = project(g, a);
  EXPECT_LE(vacuum_residual_linf(g, after), allowed);
}

TEST(MhdBackgroundBuilder, ProjectionIsBitwiseReproducible) {
  if (!device_available()) GTEST_SKIP() << "no HIP device";
  const Grid2D g = annular_grid();
  const int nodes_x = g.pitch() + 1;
  const int nodes_y = g.height() + 1;
  std::vector<Real> a(corner_count(g));
  for (int j = 0; j < nodes_y; ++j) {
    const Real z = g.origin_y + static_cast<Real>(j - g.nghost) * g.dy();
    for (int i = 0; i < nodes_x; ++i) {
      const Real r = corner_radius(g, i);
      a[static_cast<std::size_t>(j) * static_cast<std::size_t>(nodes_x) +
        static_cast<std::size_t>(i)] = r / std::pow(r * r + z * z, Real{1.5});
    }
  }
  // The conjugate gradient's inner products go through the deterministic
  // double-double tree, so two runs must agree to the last bit -- not merely to
  // the convergence tolerance.
  const std::vector<Real> first = project(g, a);
  const std::vector<Real> second = project(g, a);
  ASSERT_EQ(first.size(), second.size());
  for (std::size_t k = 0; k < first.size(); ++k) {
    EXPECT_EQ(first[k], second[k]) << "node " << k;
  }
}

TEST(MhdBackgroundBuilder, ProjectionRefusesAnAxisTouchingGrid) {
  const Grid2D g{8, 8, Real{1}, Real{1}, Real{0}, Real{0}, /*halo=*/2};
  quasar::backend::DeviceBuffer<Real> a(corner_count(g));
  MhdBackgroundField<Real> b0{g};
  // The r=0 parity closure needs its own axis row in this elliptic operator;
  // inferring one would be a different discretization wearing the same name.
  EXPECT_THROW(quasar::mhd::build_background_from_corner_potential(
                   annular_spec(g, true), a, b0),
               std::invalid_argument);
}

}  // namespace
