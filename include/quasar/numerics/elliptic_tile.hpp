#pragma once

// Tile decomposition support for the elliptic solver.
//
// This is the distribution-readiness layer: it defines how an EllipticGrid is
// partitioned, what each rank owns, and what the halo exchange must move. The
// operators in gs_operator_l2.hpp are already tile-local (a 3x3 stencil needs a
// one-cell halo), so the second-order path distributes straightforwardly.
//
// -- The Pade constraint, stated plainly ---------------------------------------
// The sixth-order path does NOT distribute the same way, and this is the single
// most important fact in this header.
//
// A compact (Pade) derivative is an implicit tridiagonal solve along an ENTIRE
// grid line. Every point on a line is coupled to every other point. A tile that
// owns a slice of that line cannot compute its own derivatives from a
// finite-width halo -- no halo width is sufficient, because the coupling is
// global along the line, not local.
//
// There are exactly three ways out, and the choice is a real design decision:
//
//   1. SLAB DECOMPOSITION. Decompose only along z, so every tile owns complete
//      r-lines; then transpose to own complete z-lines for the axial
//      derivatives. Each Pade solve is rank-local. Cost: a global transpose per
//      operator application, and the decomposition is 1D so it scales to at most
//      nz ranks.
//
//   2. DISTRIBUTED TRIDIAGONAL SOLVE. Keep the 2D decomposition and solve each
//      line cooperatively (recursive doubling / SPIKE). Cost: log(P)
//      communication rounds inside every derivative evaluation, and the pivoting
//      that these closures REQUIRE (see pade_line_solve.hpp) makes a
//      textbook pivot-free parallel algorithm invalid.
//
//   3. DEFECT CORRECTION AT THE TILE LEVEL. Apply L6 only where lines are
//      rank-local and fall back to L2 elsewhere. Cheapest, but the scheme is
//      then not uniformly sixth order, which defeats the purpose.
//
// This header implements the DATA STRUCTURES for (1) and (2) and the halo
// exchange for the L2 path. It deliberately does not pick between them: that
// choice depends on rank count and interconnect, and encoding it here would
// bake in an assumption that a 4096^2 run on eight GPUs and a 16384^2 run on
// 512 ranks do not share.
//
// What IS decided: `TileGrid::owns_complete_r_lines()` and
// `owns_complete_z_lines()` report whether a rank can legally run the Pade
// operator locally, so a caller can assert the precondition rather than
// silently producing a wrong answer.

#include "quasar/core/types.hpp"
#include "quasar/numerics/elliptic_grid.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace quasar::numerics {

// One rank's view of the global elliptic grid.
//
// Offsets are in GLOBAL node indices. The owned region is
// [r_offset, r_offset+nr_owned) x [z_offset, z_offset+nz_owned), surrounded by
// `halo` layers of ghost nodes that mirror neighbouring tiles (or, on a physical
// boundary, carry the Dirichlet data).
struct TileGrid {
  EllipticGrid global{};
  int r_offset{0};
  int z_offset{0};
  int nr_owned{0};
  int nz_owned{0};
  int halo{1};

  int nr_padded() const noexcept { return nr_owned + 2 * halo; }
  int nz_padded() const noexcept { return nz_owned + 2 * halo; }

  std::size_t size() const noexcept {
    return static_cast<std::size_t>(nr_padded())
         * static_cast<std::size_t>(nz_padded());
  }

  // Local (i,j) are owned-region indices; ghosts are negative or >= n_owned.
  std::size_t index(int i, int j) const noexcept {
    return static_cast<std::size_t>(i + halo)
         + static_cast<std::size_t>(nr_padded())
             * static_cast<std::size_t>(j + halo);
  }

  Real r(int i) const noexcept { return global.r(r_offset + i); }
  Real z(int j) const noexcept { return global.z(z_offset + j); }
  Real r_half(int i) const noexcept { return global.r_half(r_offset + i); }

  bool on_global_boundary_r_lo() const noexcept { return r_offset == 0; }
  bool on_global_boundary_z_lo() const noexcept { return z_offset == 0; }
  bool on_global_boundary_r_hi() const noexcept {
    return r_offset + nr_owned == global.nr;
  }
  bool on_global_boundary_z_hi() const noexcept {
    return z_offset + nz_owned == global.nz;
  }

  // Does this tile own every node of each r-line it touches? Required before
  // any radial Pade derivative may be evaluated locally.
  bool owns_complete_r_lines() const noexcept {
    return r_offset == 0 && nr_owned == global.nr;
  }
  bool owns_complete_z_lines() const noexcept {
    return z_offset == 0 && nz_owned == global.nz;
  }

  // True when the sixth-order operator can be applied entirely rank-locally.
  // A 2D-decomposed tile satisfies neither condition, which is exactly the
  // constraint documented at the top of this header.
  bool supports_local_pade() const noexcept {
    return owns_complete_r_lines() && owns_complete_z_lines();
  }
};

// Partition a global grid into a px-by-pz tile array, returning every tile.
// Remainder nodes are distributed to the leading tiles so extents differ by at
// most one.
inline std::vector<TileGrid> partition(const EllipticGrid& g, int px, int pz,
                                       int halo = 1) {
  if (px < 1 || pz < 1) {
    throw std::invalid_argument{"partition: process counts must be positive"};
  }
  if (halo < 1) {
    throw std::invalid_argument{"partition: halo must be at least one node"};
  }
  if (px > g.nr || pz > g.nz) {
    throw std::invalid_argument{"partition: more tiles than nodes"};
  }
  std::vector<TileGrid> tiles;
  tiles.reserve(static_cast<std::size_t>(px) * static_cast<std::size_t>(pz));

  const int base_r = g.nr / px;
  const int rem_r  = g.nr % px;
  const int base_z = g.nz / pz;
  const int rem_z  = g.nz % pz;

  int z_off = 0;
  for (int jz = 0; jz < pz; ++jz) {
    const int nz_owned = base_z + (jz < rem_z ? 1 : 0);
    int r_off = 0;
    for (int ix = 0; ix < px; ++ix) {
      const int nr_owned = base_r + (ix < rem_r ? 1 : 0);
      TileGrid t;
      t.global = g;
      t.r_offset = r_off;
      t.z_offset = z_off;
      t.nr_owned = nr_owned;
      t.nz_owned = nz_owned;
      t.halo = halo;
      tiles.push_back(t);
      r_off += nr_owned;
    }
    z_off += nz_owned;
  }
  return tiles;
}

// Scatter a global field into a tile's padded local buffer, filling ghosts from
// the global array. This models what a halo exchange delivers; the MPI path
// substitutes point-to-point transfers for the direct reads.
inline std::vector<Real> scatter_to_tile(const TileGrid& t,
                                         const ScalarField& global_field) {
  std::vector<Real> local(t.size(), Real{0});
  for (int j = -t.halo; j < t.nz_owned + t.halo; ++j) {
    for (int i = -t.halo; i < t.nr_owned + t.halo; ++i) {
      const int gi = t.r_offset + i;
      const int gj = t.z_offset + j;
      if (gi < 0 || gj < 0 || gi >= t.global.nr || gj >= t.global.nz) continue;
      local[t.index(i, j)] = global_field[t.global.index(gi, gj)];
    }
  }
  return local;
}

// Gather owned (non-ghost) values back into a global field.
inline void gather_from_tile(const TileGrid& t, const std::vector<Real>& local,
                             ScalarField& global_field) {
  for (int j = 0; j < t.nz_owned; ++j) {
    for (int i = 0; i < t.nr_owned; ++i) {
      const int gi = t.r_offset + i;
      const int gj = t.z_offset + j;
      global_field[t.global.index(gi, gj)] = local[t.index(i, j)];
    }
  }
}

// Tile-local second-order operator. Identical stencil to gs_apply_l2, evaluated
// on owned nodes using halo values for the neighbours. Global-boundary nodes are
// skipped (they carry Dirichlet data).
inline void gs_apply_l2_tile(const TileGrid& t, const std::vector<Real>& x,
                             std::vector<Real>& y) {
  y.assign(t.size(), Real{0});
  const Real dr = t.global.dr();
  const Real dz = t.global.dz();
  const Real inv_dr2 = Real{1} / (dr * dr);
  const Real inv_dz2 = Real{1} / (dz * dz);

  for (int j = 0; j < t.nz_owned; ++j) {
    for (int i = 0; i < t.nr_owned; ++i) {
      const int gi = t.r_offset + i;
      const int gj = t.z_offset + j;
      if (gi == 0 || gj == 0 || gi == t.global.nr - 1 || gj == t.global.nz - 1) {
        continue;
      }
      const Real r_i = t.r(i);
      const Real a_p = r_i / t.r_half(i) * inv_dr2;
      const Real a_m = r_i / t.global.r_half(gi - 1) * inv_dr2;
      const Real w_c = -(a_p + a_m + Real{2} * inv_dz2);

      y[t.index(i, j)] = w_c * x[t.index(i, j)]
                       + a_m * x[t.index(i - 1, j)]
                       + a_p * x[t.index(i + 1, j)]
                       + inv_dz2 * x[t.index(i, j - 1)]
                       + inv_dz2 * x[t.index(i, j + 1)];
    }
  }
}

// Local contribution to a global max-norm over owned interior nodes. The
// distributed driver combines these with one Allreduce(MAX).
inline Real tile_interior_max_norm(const TileGrid& t,
                                   const std::vector<Real>& f) {
  Real m = Real{0};
  for (int j = 0; j < t.nz_owned; ++j) {
    for (int i = 0; i < t.nr_owned; ++i) {
      const int gi = t.r_offset + i;
      const int gj = t.z_offset + j;
      if (gi == 0 || gj == 0 || gi == t.global.nr - 1 || gj == t.global.nz - 1) {
        continue;
      }
      const Real v = f[t.index(i, j)];
      if (std::isnan(v)) {
        return std::numeric_limits<Real>::quiet_NaN();
      }
      m = std::max(m, v < Real{0} ? -v : v);
    }
  }
  return m;
}

// Coarse-grid agglomeration threshold.
//
// Multigrid coarsening halves each tile; once a tile is smaller than its own
// stencil reach the level cannot be evaluated locally and tiles must be gathered
// onto fewer ranks. Below this size the hierarchy must agglomerate.
inline bool tile_needs_agglomeration(const TileGrid& t) noexcept {
  const int reach = 2 * t.halo + 1;
  return t.nr_owned < reach || t.nz_owned < reach;
}

}  // namespace quasar::numerics
