// Projection of a Grad-Shafranov equilibrium onto the staggered MHD mesh.
//
// The single property that matters here is DISCRETE SOLENOIDALITY. The MHD
// solver's seed_background() rejects a background that is not discretely
// divergence-free, and constrained transport preserves whatever divergence the
// initial state carries -- so a projection that is merely accurate but not
// exactly solenoidal poisons the entire downstream simulation.
//
// The tests therefore check div(B) to ROUND-OFF, not to truncation, and they
// check it on grids that do not align with the GS grid (the realistic case).

#include "quasar/physics/equilibrium/mhd_seeding.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::numerics::EllipticGrid;
using quasar::numerics::ScalarField;
using namespace quasar::equilibrium;

EllipticGrid gs_grid() {
  return EllipticGrid{33, 33, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
}

Grid2D mhd_grid(int n = 64) {
  return Grid2D::from_cell_spacing(n, n, Real{1.6} / static_cast<Real>(n),
                                   Real{1.6} / static_cast<Real>(n),
                                   Real{0.3}, Real{-0.8}, 4);
}

// An analytic flux function, so projection can be tested without a full solve.
ScalarField analytic_psi(const EllipticGrid& g) {
  ScalarField psi = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      const Real r = g.r(i);
      const Real z = g.z(j);
      psi[g.index(i, j)] =
          Real{0.4} * r * r - Real{0.15} * z * z + Real{0.05} * r * r * z;
    }
  }
  return psi;
}

CriticalPointSet dummy_criticals() {
  CriticalPointSet cps;
  cps.axis.valid = true;
  cps.axis.kind = CriticalKind::o_point;
  cps.axis.r = Real{1.1};
  cps.axis.z = Real{0};
  cps.axis.psi = Real{0.5};
  cps.psi_axis = Real{0.5};
  cps.psi_boundary = Real{0.1};
  cps.has_closed_surface = true;
  return cps;
}

}  // namespace

TEST(MhdSeeding, ProjectedFieldIsSolenoidalToRoundOff) {
  // The headline property. Both in-plane components come from differencing ONE
  // scalar potential, so the discrete divergence telescopes to zero identically
  // rather than merely converging to zero.
  const EllipticGrid g = gs_grid();
  const ScalarField psi = analytic_psi(g);
  const auto cps = dummy_criticals();
  const Grid2D mg = mhd_grid();

  const auto sb = project_to_mhd(g, psi, cps, mg, [](Real) { return Real{5}; });
  const Real scale = field_scale(sb);
  ASSERT_GT(scale, Real{0}) << "projection produced an identically zero field";
  EXPECT_LT(max_divergence(sb) / scale, 1e-11)
      << "projected background is not discretely solenoidal; MHD "
         "seed_background() would reject it";
}

TEST(MhdSeeding, SolenoidalityHoldsOnMisalignedGrids) {
  // The realistic case: the MHD mesh does not share nodes with the GS mesh.
  // Interpolation happens, and the property must survive it -- which it does
  // precisely because interpolation is applied to psi BEFORE differencing, not
  // to B afterwards.
  const EllipticGrid g = gs_grid();
  const ScalarField psi = analytic_psi(g);
  const auto cps = dummy_criticals();

  for (const int n : {37, 64, 91}) {
    const Grid2D mg = mhd_grid(n);
    const auto sb =
        project_to_mhd(g, psi, cps, mg, [](Real) { return Real{5}; });
    const Real scale = field_scale(sb);
    ASSERT_GT(scale, Real{0}) << "n=" << n;
    EXPECT_LT(max_divergence(sb) / scale, 1e-11) << "n=" << n;
  }
}

TEST(MhdSeeding, ToroidalComponentIsFOverMajorRadius) {
  const EllipticGrid g = gs_grid();
  const ScalarField psi = analytic_psi(g);
  const auto cps = dummy_criticals();
  const Grid2D mg = mhd_grid();
  const Real f_const = Real{7.5};

  const auto sb =
      project_to_mhd(g, psi, cps, mg, [&](Real) { return f_const; });
  for (int j = 0; j < mg.ny; j += 11) {
    for (int i = 0; i < mg.nx; i += 11) {
      const Real rc = mg.r_at_cell_center(i);
      EXPECT_NEAR(sb.b0z_cell[mg.index(i, j)], f_const / rc, 1e-12);
    }
  }
}

TEST(MhdSeeding, BufferSizesMatchTheMhdStorageContract) {
  // seed_background() requires buffers sized exactly grid.storage_size().
  const EllipticGrid g = gs_grid();
  const ScalarField psi = analytic_psi(g);
  const Grid2D mg = mhd_grid();
  const auto sb =
      project_to_mhd(g, psi, dummy_criticals(), mg, [](Real) { return Real{1}; });
  EXPECT_EQ(sb.b0x_face.size(), mg.storage_size());
  EXPECT_EQ(sb.b0y_face.size(), mg.storage_size());
  EXPECT_EQ(sb.b0z_cell.size(), mg.storage_size());
}

TEST(MhdSeeding, GhostCellsArePopulated) {
  // The MHD background is static and never ghost-refilled, so the projection
  // must fill the halo itself. An unfilled halo shows up much later as a
  // boundary artefact, so it is checked directly.
  const EllipticGrid g = gs_grid();
  const ScalarField psi = analytic_psi(g);
  const Grid2D mg = mhd_grid();
  const auto sb =
      project_to_mhd(g, psi, dummy_criticals(), mg, [](Real) { return Real{2}; });

  for (int j = -mg.nghost; j < 0; ++j) {
    for (int i = 0; i < mg.nx; ++i) {
      EXPECT_NE(sb.b0z_cell[mg.index(i, j)], Real{0})
          << "ghost cell (" << i << "," << j << ") left unset";
    }
  }
}

TEST(MhdSeeding, ClampsRatherThanExtrapolatingOutsideTheGsDomain) {
  // An MHD grid that overhangs the GS domain must not receive extrapolated
  // flux, which would manufacture a current sheet at the overhang.
  const EllipticGrid g = gs_grid();
  const ScalarField psi = analytic_psi(g);
  // Deliberately wider and taller than the GS domain.
  const Grid2D wide = Grid2D::from_cell_spacing(48, 48, Real{0.05}, Real{0.05},
                                                Real{0.2}, Real{-1.2}, 2);
  const auto sb =
      project_to_mhd(g, psi, dummy_criticals(), wide, [](Real) { return Real{1}; });

  for (const Real v : sb.b0x_face) EXPECT_TRUE(std::isfinite(v));
  for (const Real v : sb.b0y_face) EXPECT_TRUE(std::isfinite(v));
  const Real scale = field_scale(sb);
  ASSERT_GT(scale, Real{0});
  EXPECT_LT(max_divergence(sb) / scale, 1e-10)
      << "overhanging grid broke solenoidality";
}

TEST(MhdSeeding, RejectsTargetsThatReachTheAxis) {
  const EllipticGrid g = gs_grid();
  const ScalarField psi = analytic_psi(g);
  const auto cps = dummy_criticals();
  const auto f = [](Real) { return Real{1}; };
  const auto p = [](Real pn) { return Real{1} - pn; };

  const Grid2D on_axis = Grid2D::from_cell_spacing(
      32, 32, Real{0.05}, Real{0.05}, Real{0.2}, Real{-0.8}, 4);
  EXPECT_THROW(project_to_mhd(g, psi, cps, on_axis, f), std::invalid_argument);
  EXPECT_THROW(project_fluid(g, psi, cps, on_axis, p, Real{1}, Real{2}),
               std::invalid_argument);

  const Grid2D crossing_axis = Grid2D::from_cell_spacing(
      32, 32, Real{0.05}, Real{0.05}, Real{0.1}, Real{-0.8}, 4);
  EXPECT_THROW(project_to_mhd(g, psi, cps, crossing_axis, f),
               std::invalid_argument);
}

TEST(MhdSeeding, RejectsZeroHaloBeforeHighFaceStorageCanBeIndexed) {
  const EllipticGrid g = gs_grid();
  const ScalarField psi = analytic_psi(g);
  const Grid2D no_halo = Grid2D::from_cell_spacing(
      32, 32, Real{0.02}, Real{0.02}, Real{0.5}, Real{-0.4}, 0);

  EXPECT_THROW(project_to_mhd(
                   g, psi, dummy_criticals(), no_halo,
                   [](Real) { return Real{1}; }),
               std::invalid_argument);
  EXPECT_THROW(project_fluid(
                   g, psi, dummy_criticals(), no_halo,
                   [](Real pn) { return Real{1} - pn; }, Real{1}, Real{2}),
               std::invalid_argument);
}

TEST(MhdSeeding, ValidatesSourceFieldAndCriticalMetadata) {
  const EllipticGrid g = gs_grid();
  const Grid2D mg = mhd_grid();
  const auto f = [](Real) { return Real{1}; };
  const auto p = [](Real pn) { return Real{1} - pn; };

  EllipticGrid malformed_source = g;
  malformed_source.nr = 0;
  EXPECT_THROW(project_to_mhd(
                   malformed_source, analytic_psi(g), dummy_criticals(), mg, f),
               std::invalid_argument);

  Grid2D malformed_target = mg;
  malformed_target.nx = 0;
  EXPECT_THROW(project_to_mhd(
                   g, analytic_psi(g), dummy_criticals(), malformed_target, f),
               std::invalid_argument);

  ScalarField short_psi(g.size() - 1, Real{0});
  EXPECT_THROW(project_to_mhd(g, short_psi, dummy_criticals(), mg, f),
               std::invalid_argument);
  EXPECT_THROW(project_fluid(
                   g, short_psi, dummy_criticals(), mg, p, Real{1}, Real{2}),
               std::invalid_argument);

  ScalarField non_finite = analytic_psi(g);
  non_finite[g.index(4, 7)] = std::numeric_limits<Real>::quiet_NaN();
  EXPECT_THROW(project_to_mhd(g, non_finite, dummy_criticals(), mg, f),
               std::invalid_argument);

  CriticalPointSet invalid = dummy_criticals();
  invalid.axis.valid = false;
  EXPECT_THROW(project_to_mhd(g, analytic_psi(g), invalid, mg, f),
               std::invalid_argument);

  CriticalPointSet wrong_kind = dummy_criticals();
  wrong_kind.axis.kind = CriticalKind::x_point;
  EXPECT_THROW(project_to_mhd(g, analytic_psi(g), wrong_kind, mg, f),
               std::invalid_argument);

  CriticalPointSet incoherent_axis = dummy_criticals();
  incoherent_axis.axis.psi = std::nextafter(
      incoherent_axis.psi_axis, std::numeric_limits<Real>::infinity());
  EXPECT_THROW(project_fluid(
                   g, analytic_psi(g), incoherent_axis, mg, p, Real{1}, Real{2}),
               std::invalid_argument);

  CriticalPointSet open = dummy_criticals();
  open.has_closed_surface = false;
  EXPECT_THROW(project_fluid(
                   g, analytic_psi(g), open, mg, p, Real{1}, Real{2}),
               std::invalid_argument);

  CriticalPointSet overflowed = dummy_criticals();
  overflowed.critical_point_overflow = true;
  EXPECT_THROW(project_to_mhd(g, analytic_psi(g), overflowed, mg, f),
               std::invalid_argument);
}

TEST(MhdSeeding, RejectsInvalidProfileOutputsAndDensityEndpoints) {
  const EllipticGrid g = gs_grid();
  const ScalarField psi = analytic_psi(g);
  const Grid2D mg = mhd_grid();
  const auto cps = dummy_criticals();
  const Real nan = std::numeric_limits<Real>::quiet_NaN();

  EXPECT_THROW(project_to_mhd(g, psi, cps, mg, [nan](Real) { return nan; }),
               std::invalid_argument);
  EXPECT_THROW(project_fluid(
                   g, psi, cps, mg, [nan](Real) { return nan; }, Real{1}, Real{2}),
               std::invalid_argument);
  EXPECT_THROW(project_fluid(
                   g, psi, cps, mg, [](Real) { return Real{1}; }, Real{0}, Real{2}),
               std::invalid_argument);
}

TEST(MhdSeeding, ProfileSeedsExcludeDisconnectedPrivateFlux) {
  const EllipticGrid g{81, 41, Real{0.6}, Real{2.4}, Real{-0.5}, Real{0.5}};
  constexpr Real center = Real{1.51125};
  constexpr Real half_separation = Real{0.45};
  constexpr Real axis_r = center - half_separation;
  constexpr Real psi_axis = Real{0};
  constexpr Real psi_boundary =
      -half_separation * half_separation * half_separation * half_separation;

  ScalarField psi = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      const Real x = g.r(i) - center;
      const Real z = g.z(j);
      const Real well = x * x - half_separation * half_separation;
      psi[g.index(i, j)] = -(well * well + z * z);
    }
  }

  CriticalPointSet cps;
  cps.axis.valid = true;
  cps.axis.kind = CriticalKind::o_point;
  cps.axis.r = axis_r;
  cps.axis.z = Real{0};
  cps.axis.psi = psi_axis;
  cps.psi_axis = psi_axis;
  cps.psi_boundary = psi_boundary;
  cps.has_closed_surface = true;

  // Cell centers 0 and 2 coincide with source nodes 20 and 60. They have the
  // same scalar psi_N range, but only node 20 belongs to the axis component.
  const Grid2D mg = Grid2D::from_cell_spacing(
      3, 1, Real{0.45}, Real{0.1}, Real{0.825}, Real{-0.05}, 1);
  const auto sb = project_to_mhd(
      g, psi, cps, mg, [](Real pn) { return Real{2} - pn; });
  const auto fluid = project_fluid(
      g, psi, cps, mg, [](Real pn) { return Real{1} - pn; }, Real{1}, Real{10});

  const std::size_t core = mg.index(0, 0);
  const std::size_t private_lobe = mg.index(2, 0);
  EXPECT_GT(sb.b0z_cell[core] * mg.r_at_cell_center(0), Real{1});
  EXPECT_EQ(sb.b0z_cell[private_lobe] * mg.r_at_cell_center(2), Real{1});
  EXPECT_GT(fluid.pressure[core], Real{0});
  EXPECT_EQ(fluid.pressure[private_lobe], Real{0});
  EXPECT_GT(fluid.rho[core], Real{1});
  EXPECT_EQ(fluid.rho[private_lobe], Real{1});
}

TEST(MhdSeeding, DiagnosticsValidateLayoutAndSurfaceNonFiniteFields) {
  const EllipticGrid g = gs_grid();
  const ScalarField psi = analytic_psi(g);
  const Grid2D mg = mhd_grid();
  StaggeredBackground sb = project_to_mhd(
      g, psi, dummy_criticals(), mg, [](Real) { return Real{1}; });

  sb.b0x_face.pop_back();
  EXPECT_THROW(max_divergence(sb), std::invalid_argument);
  EXPECT_THROW(field_scale(sb), std::invalid_argument);

  sb = project_to_mhd(
      g, psi, dummy_criticals(), mg, [](Real) { return Real{1}; });
  sb.b0x_face[mg.index(0, 0)] =
      std::numeric_limits<Real>::quiet_NaN();
  EXPECT_TRUE(std::isinf(max_divergence(sb)));
  EXPECT_TRUE(std::isinf(field_scale(sb)));
}

TEST(MhdSeeding, FluidSeedIsPeakedOnAxis) {
  const EllipticGrid g = gs_grid();
  const ScalarField psi = analytic_psi(g);
  const auto cps = dummy_criticals();
  const Grid2D mg = mhd_grid();

  const auto seed = project_fluid(g, psi, cps, mg,
                                  [](Real pn) { return Real{100} * (Real{1} - pn); },
                                  Real{1}, Real{10});
  EXPECT_EQ(seed.rho.size(), mg.storage_size());
  EXPECT_EQ(seed.pressure.size(), mg.storage_size());
  for (const Real v : seed.rho) {
    EXPECT_GE(v, Real{1} - 1e-12);
    EXPECT_LE(v, Real{10} + 1e-12);
  }
  for (const Real v : seed.pressure) EXPECT_GE(v, Real{0});
}
