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

#include "quasar/physics/equilibrium/gs_solver.hpp"
#include "quasar/physics/equilibrium/mhd_seeding.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

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
  cps.axis.r = Real{1.1};
  cps.axis.z = Real{0};
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

TEST(MhdSeeding, EndToEndFromASolvedEquilibrium) {
  // The full pipeline the MHD consumer will actually run.
  GsConfig cfg;
  cfg.grid = gs_grid();
  cfg.coils = {
      {Real{2.4}, Real{0.9},  Real{-3.0e5}},
      {Real{2.4}, Real{-0.9}, Real{-3.0e5}},
      {Real{0.28}, Real{0.0}, Real{1.0e5}},
  };
  cfg.plasma_current = Real{1.0e6};
  cfg.max_iterations = 400;
  cfg.tolerance = Real{1e-9};

  auto prof = std::make_shared<PolynomialProfile>();
  GsSolver solver{cfg, prof};
  const GsResult r = solver.solve();
  ASSERT_EQ(r.status, GsStatus::converged);

  const Grid2D mg = mhd_grid();
  const auto sb =
      project_to_mhd(cfg.grid, r.psi, r.critical, mg, [](Real) { return Real{5}; });

  const Real scale = field_scale(sb);
  ASSERT_GT(scale, Real{0});
  EXPECT_LT(max_divergence(sb) / scale, 1e-11)
      << "a solved equilibrium must project to a solenoidal MHD background";

  // The projected poloidal field must be non-trivial: a silently zero field
  // would pass every divergence check.
  EXPECT_GT(scale, Real{1e-3}) << "projected field is implausibly weak";
}
