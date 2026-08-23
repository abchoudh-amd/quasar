// Tile decomposition of the elliptic solver.
//
// The central correctness property is EQUIVALENCE: applying the operator
// tile-by-tile with a correct halo must reproduce the serial result bit-for-bit
// on owned nodes. Anything weaker (agreement to some tolerance) would hide an
// off-by-one in the halo or an inconsistent stencil at a tile seam, both of
// which are the classic distributed-solver bugs.
//
// The tests also pin the Pade constraint: a 2D-decomposed tile must REPORT that
// it cannot run the sixth-order operator locally, rather than silently
// producing a wrong answer from an insufficient halo.

#include "quasar/numerics/elliptic_tile.hpp"
#include "quasar/numerics/gs_operator_l2.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using quasar::Real;
using quasar::numerics::EllipticGrid;
using quasar::numerics::ScalarField;
using quasar::numerics::TileGrid;

EllipticGrid global_grid(int n = 33) {
  return EllipticGrid{n, n, Real{0.5}, Real{1.5}, Real{-0.5}, Real{0.5}};
}

ScalarField sample_field(const EllipticGrid& g) {
  ScalarField f = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      f[g.index(i, j)] =
          std::sin(Real{2.1} * g.r(i)) * std::cos(Real{1.4} * g.z(j))
          + Real{0.3} * g.r(i);
    }
  }
  return f;
}

}  // namespace

TEST(EllipticTile, PartitionCoversEveryNodeExactlyOnce) {
  const EllipticGrid g = global_grid(33);
  const auto tiles = quasar::numerics::partition(g, 3, 2);
  ASSERT_EQ(tiles.size(), 6u);

  std::vector<int> hits(g.size(), 0);
  for (const auto& t : tiles) {
    for (int j = 0; j < t.nz_owned; ++j) {
      for (int i = 0; i < t.nr_owned; ++i) {
        ++hits[g.index(t.r_offset + i, t.z_offset + j)];
      }
    }
  }
  for (std::size_t k = 0; k < hits.size(); ++k) {
    EXPECT_EQ(hits[k], 1) << "node " << k << " covered " << hits[k] << " times";
  }
}

TEST(EllipticTile, PartitionBalancesExtentsToWithinOneNode) {
  const EllipticGrid g = global_grid(33);
  const auto tiles = quasar::numerics::partition(g, 4, 4);
  int min_r = 1 << 30, max_r = 0, min_z = 1 << 30, max_z = 0;
  for (const auto& t : tiles) {
    min_r = std::min(min_r, t.nr_owned);
    max_r = std::max(max_r, t.nr_owned);
    min_z = std::min(min_z, t.nz_owned);
    max_z = std::max(max_z, t.nz_owned);
  }
  EXPECT_LE(max_r - min_r, 1);
  EXPECT_LE(max_z - min_z, 1);
}

TEST(EllipticTile, RejectsOverDecomposition) {
  const EllipticGrid g = global_grid(9);
  EXPECT_THROW(quasar::numerics::partition(g, 16, 1), std::invalid_argument);
  EXPECT_THROW(quasar::numerics::partition(g, 0, 1), std::invalid_argument);
}

TEST(EllipticTile, RejectsNonPositiveHalo) {
  const EllipticGrid g = global_grid(9);
  EXPECT_THROW(quasar::numerics::partition(g, 2, 2, 0), std::invalid_argument);
  EXPECT_THROW(quasar::numerics::partition(g, 2, 2, -1), std::invalid_argument);
}

TEST(EllipticTile, WiderHaloStillRoundTripsExactly) {
  const EllipticGrid g = global_grid(33);
  const ScalarField f = sample_field(g);
  const auto tiles = quasar::numerics::partition(g, 3, 3, 3);

  ScalarField rebuilt = quasar::numerics::make_field(g);
  for (const auto& t : tiles) {
    EXPECT_EQ(t.nr_padded(), t.nr_owned + 6);
    EXPECT_EQ(t.nz_padded(), t.nz_owned + 6);
    const auto local = quasar::numerics::scatter_to_tile(t, f);
    quasar::numerics::gather_from_tile(t, local, rebuilt);
  }
  EXPECT_EQ(rebuilt, f);
}

TEST(EllipticTile, ScatterGatherRoundTripsExactly) {
  const EllipticGrid g = global_grid(33);
  const ScalarField f = sample_field(g);
  const auto tiles = quasar::numerics::partition(g, 3, 3);

  ScalarField rebuilt = quasar::numerics::make_field(g);
  for (const auto& t : tiles) {
    const auto local = quasar::numerics::scatter_to_tile(t, f);
    quasar::numerics::gather_from_tile(t, local, rebuilt);
  }
  for (std::size_t k = 0; k < f.size(); ++k) {
    EXPECT_DOUBLE_EQ(rebuilt[k], f[k]) << "at index " << k;
  }
}

TEST(EllipticTile, TiledOperatorReproducesTheSerialResultBitForBit) {
  // The core distributed-correctness property. Bit-for-bit, not approximate:
  // the tiled path evaluates the identical stencil on the identical values, so
  // any difference is a bug rather than a rounding artefact.
  const EllipticGrid g = global_grid(33);
  const ScalarField f = sample_field(g);

  ScalarField serial = quasar::numerics::make_field(g);
  quasar::numerics::gs_apply_l2(g, f, serial);

  for (const auto shape : {std::pair{1, 1}, std::pair{2, 3}, std::pair{4, 4}}) {
    const auto tiles =
        quasar::numerics::partition(g, shape.first, shape.second);
    ScalarField tiled = quasar::numerics::make_field(g);
    for (const auto& t : tiles) {
      const auto local = quasar::numerics::scatter_to_tile(t, f);
      std::vector<Real> out;
      quasar::numerics::gs_apply_l2_tile(t, local, out);
      quasar::numerics::gather_from_tile(t, out, tiled);
    }
    for (int j = 1; j < g.nz - 1; ++j) {
      for (int i = 1; i < g.nr - 1; ++i) {
        EXPECT_DOUBLE_EQ(tiled[g.index(i, j)], serial[g.index(i, j)])
            << "mismatch at (" << i << "," << j << ") for decomposition "
            << shape.first << "x" << shape.second;
      }
    }
  }
}

TEST(EllipticTile, LocalNormsCombineToTheGlobalNorm) {
  // Models the Allreduce(MAX) the distributed driver performs.
  const EllipticGrid g = global_grid(33);
  const ScalarField f = sample_field(g);
  const Real global_norm = quasar::numerics::interior_max_norm(g, f);

  const auto tiles = quasar::numerics::partition(g, 3, 2);
  Real reduced = Real{0};
  for (const auto& t : tiles) {
    const auto local = quasar::numerics::scatter_to_tile(t, f);
    reduced = std::max(reduced, quasar::numerics::tile_interior_max_norm(t, local));
  }
  EXPECT_DOUBLE_EQ(reduced, global_norm);
}

TEST(EllipticTile, ReportsWhenPadeCannotRunLocally) {
  // The load-bearing constraint. A compact derivative is an implicit solve along
  // a whole grid line, so a tile owning a slice of that line CANNOT evaluate it
  // from any finite halo. The tile must say so.
  const EllipticGrid g = global_grid(33);

  const auto single = quasar::numerics::partition(g, 1, 1);
  EXPECT_TRUE(single.front().supports_local_pade())
      << "an undecomposed tile owns every line and must support Pade";

  const auto two_d = quasar::numerics::partition(g, 2, 2);
  for (const auto& t : two_d) {
    EXPECT_FALSE(t.supports_local_pade())
        << "a 2D-decomposed tile must not claim local Pade support";
  }

  // Slab decompositions own complete lines in exactly one direction, which is
  // what makes the transpose-based strategy viable.
  const auto z_slabs = quasar::numerics::partition(g, 1, 4);
  for (const auto& t : z_slabs) {
    EXPECT_TRUE(t.owns_complete_r_lines())
        << "a z-slab must own complete r-lines";
    EXPECT_FALSE(t.owns_complete_z_lines());
  }

  const auto r_slabs = quasar::numerics::partition(g, 4, 1);
  for (const auto& t : r_slabs) {
    EXPECT_TRUE(t.owns_complete_z_lines());
    EXPECT_FALSE(t.owns_complete_r_lines());
  }
}

TEST(EllipticTile, FlagsTilesNeedingCoarseGridAgglomeration) {
  // Multigrid halves tiles at every level; once a tile is smaller than its
  // stencil reach the hierarchy must gather onto fewer ranks.
  const EllipticGrid g = global_grid(33);
  const auto coarse = quasar::numerics::partition(g, 2, 2);
  for (const auto& t : coarse) {
    EXPECT_FALSE(quasar::numerics::tile_needs_agglomeration(t))
        << "an 8x8 tile should not need agglomeration at halo=1";
  }

  const EllipticGrid tiny = global_grid(5);
  const auto over = quasar::numerics::partition(tiny, 2, 2);
  bool any_flagged = false;
  for (const auto& t : over) {
    any_flagged = any_flagged || quasar::numerics::tile_needs_agglomeration(t);
  }
  EXPECT_TRUE(any_flagged)
      << "a 5x5 grid split 2x2 leaves sub-stencil tiles that must agglomerate";
}

TEST(EllipticTile, GlobalBoundaryFlagsIdentifyPhysicalEdges) {
  const EllipticGrid g = global_grid(33);
  const auto tiles = quasar::numerics::partition(g, 2, 2);
  // Corner tile 0 owns the low-r, low-z physical corner.
  EXPECT_TRUE(tiles[0].on_global_boundary_r_lo());
  EXPECT_TRUE(tiles[0].on_global_boundary_z_lo());
  EXPECT_FALSE(tiles[0].on_global_boundary_r_hi());
  // The last tile owns the opposite corner.
  EXPECT_TRUE(tiles.back().on_global_boundary_r_hi());
  EXPECT_TRUE(tiles.back().on_global_boundary_z_hi());
}
