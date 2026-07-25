// MHD ghost-fill boundary contract.
//
// This pins the EXACT reference ghost-fill rules of the three registered MHD
// boundaries ("periodic" / "outflow" / "wall") as the post-port device
// contract. The current host implementation lives in src/physics/mhd/
// mhd_boundary.cpp; these tests encode its OBSERVABLE behavior (the values that
// land in the ghost layers), not its implementation, so a later device-kernel
// port must reproduce them bit-for-bit.
//
// The boundaries are obtained by registry NAME (no enum / switch) exactly as the
// solver does:
//   Registry<boundary::IMhdFluidBoundary>::instance().create("periodic")
//   Registry<boundary::IMhdFieldBoundary>::instance().create("wall")
// and applied to a seeded MhdField2D<Real> via fill_ghosts(field, Side).
//
// Index / staggering conventions (Grid2D::index(i,j), full padded storage):
//   * interior cells:  i in [0, nx),  j in [0, ny).
//   * x ghost layers (layer = 1..nghost): lo -> i = -layer ; hi -> i = nx-1+layer.
//   * y ghost layers (layer = 1..nghost): lo -> j = -layer ; hi -> j = ny-1+layer.
//   * Cell and face components share one allocation, but a normal face is offset
//     by half a cell: physical x faces are i=0..nx (and y faces j=0..ny).
//
// Reference rules pinned (per side):
//   periodic   : lo ghost (-layer)   <- interior (nx-layer)
//                hi ghost (nx-1+layer)<- interior (layer-1)
//   outflow    : lo ghost            <- interior 0       (zero-gradient)
//                hi ghost            <- interior nx-1
//   wall       : cell quantities mirror about the boundary cell face. Normal B
//                instead mirrors about its own staggered boundary face: Bn=0 at
//                face 0/n and Bn(-q)=-Bn(q), Bn(n+q)=-Bn(n-q).
//
// x-side fills touch only j in [0, ny); y-side fills span the FULL storage width
// i in [-ng, nx+ng). The solver fills x-sides THEN y-sides, so the four corner
// ghost blocks (both i and j ghost) are populated by the y-side fill copying the
// already-filled x-ghost columns down/up. We replicate that order here.

#include "quasar/backend/device.hpp"
#include "quasar/boundary/mhd_boundary.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using quasar::Real;
using quasar::Grid2D;
using quasar::Side;
using quasar::mhd::MhdField2D;
namespace boundary = quasar::boundary;

// A recognizable, per-(i,j) deterministic seed pattern so every ghost slot has a
// PREDICTABLE source value. The base offset distinguishes components so a fill
// that reads the wrong component is detected. Defined over the FULL padded
// storage [-ng, nx+ng) x [-ng, ny+ng) including the ghost halo (so corner-block
// predictions can reference already-filled x-ghost columns).
Real pattern(int i, int j, Real base) {
  return base + static_cast<Real>(i) * 1.0 + static_cast<Real>(j) * 0.01;
}

// Component bases: unique per component so a cross-component bug is observable.
constexpr Real kBaseRho = 100.0;
constexpr Real kBaseMx = 200.0;
constexpr Real kBaseMy = 300.0;
constexpr Real kBaseMz = 400.0;
constexpr Real kBaseEn = 500.0;
constexpr Real kBaseBx = 600.0;
constexpr Real kBaseBy = 700.0;
constexpr Real kBaseBz = 800.0;

// Stage a full-storage host buffer carrying pattern(i,j,base) into a device
// buffer. Sized to grid.storage_size().
void seed_component(quasar::backend::DeviceBuffer<Real>& dev, const Grid2D& g,
                    Real base) {
  const std::size_t n = g.storage_size();
  std::vector<Real> host(n, Real{0});
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      host[g.index(i, j)] = pattern(i, j, base);
    }
  }
  dev.copy_from_host(host.data(), n);
}

void seed_field(MhdField2D<Real>& f) {
  const Grid2D& g = f.grid;
  seed_component(f.rho, g, kBaseRho);
  seed_component(f.mx, g, kBaseMx);
  seed_component(f.my, g, kBaseMy);
  seed_component(f.mz, g, kBaseMz);
  seed_component(f.energy, g, kBaseEn);
  seed_component(f.bx_face, g, kBaseBx);
  seed_component(f.by_face, g, kBaseBy);
  seed_component(f.bz_cell, g, kBaseBz);
}

std::vector<Real> to_host(const quasar::backend::DeviceBuffer<Real>& dev,
                          const Grid2D& g) {
  std::vector<Real> host(g.storage_size(), Real{0});
  dev.copy_to_host(host.data(), host.size());
  return host;
}

Grid2D make_grid() {
  // Small grid with a multi-layer halo so the hi face ghost at layer==nghost is
  // exercised (nx-1+nghost) and corner blocks are non-trivial.
  return Grid2D{6, 5, 1.0, 1.0, 0.0, 0.0, /*nghost=*/3};
}

std::unique_ptr<boundary::IMhdFluidBoundary> make_fluid(const std::string& name) {
  return quasar::Registry<boundary::IMhdFluidBoundary>::instance().create(name);
}
std::unique_ptr<boundary::IMhdFieldBoundary> make_field(const std::string& name) {
  return quasar::Registry<boundary::IMhdFieldBoundary>::instance().create(name);
}

constexpr Real kTol = 1e-12;

}  // namespace

// ---------------------------------------------------------------------------
// PERIODIC fluid + field: each ghost layer equals the wrapped interior value.
//   lo ghost (-layer)    <- interior (nx-layer)
//   hi ghost (nx-1+layer)<- interior (layer-1)
// Verified on x and y sides for representative components.
// ---------------------------------------------------------------------------
TEST(MhdBoundary, PeriodicWrapsInteriorOnBothAxes) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D g = make_grid();
  MhdField2D<Real> f{g};
  seed_field(f);

  auto fluid = make_fluid("periodic");
  auto field = make_field("periodic");

  // Apply x-sides then y-sides (solver order).
  fluid->fill_ghosts(f, Side::x_lo);
  fluid->fill_ghosts(f, Side::x_hi);
  field->fill_ghosts(f, Side::x_lo);
  field->fill_ghosts(f, Side::x_hi);

  const std::vector<Real> rho = to_host(f.rho, g);
  const std::vector<Real> bx = to_host(f.bx_face, g);

  for (int layer = 1; layer <= g.nghost; ++layer) {
    const int gi_lo = -layer;
    const int gi_hi = g.nx - 1 + layer;
    const int si_lo = g.nx - layer;   // wrap source for lo
    const int si_hi = layer - 1;      // wrap source for hi
    for (int j = 0; j < g.ny; ++j) {
      EXPECT_NEAR(rho[g.index(gi_lo, j)], pattern(si_lo, j, kBaseRho), kTol)
          << "rho periodic x_lo layer=" << layer << " j=" << j;
      EXPECT_NEAR(rho[g.index(gi_hi, j)], pattern(si_hi, j, kBaseRho), kTol)
          << "rho periodic x_hi layer=" << layer << " j=" << j;
      EXPECT_NEAR(bx[g.index(gi_lo, j)], pattern(si_lo, j, kBaseBx), kTol)
          << "bx_face periodic x_lo layer=" << layer << " j=" << j;
      EXPECT_NEAR(bx[g.index(gi_hi, j)], pattern(si_hi, j, kBaseBx), kTol)
          << "bx_face periodic x_hi layer=" << layer << " j=" << j;
    }
  }

  // y-sides: fresh field so the x-fill does not contaminate the y prediction.
  MhdField2D<Real> fy{g};
  seed_field(fy);
  fluid->fill_ghosts(fy, Side::y_lo);
  fluid->fill_ghosts(fy, Side::y_hi);
  field->fill_ghosts(fy, Side::y_lo);
  field->fill_ghosts(fy, Side::y_hi);

  const std::vector<Real> en = to_host(fy.energy, g);
  const std::vector<Real> by = to_host(fy.by_face, g);
  for (int layer = 1; layer <= g.nghost; ++layer) {
    const int gj_lo = -layer;
    const int gj_hi = g.ny - 1 + layer;
    const int sj_lo = g.ny - layer;
    const int sj_hi = layer - 1;
    for (int i = 0; i < g.nx; ++i) {
      EXPECT_NEAR(en[g.index(i, gj_lo)], pattern(i, sj_lo, kBaseEn), kTol)
          << "energy periodic y_lo layer=" << layer << " i=" << i;
      EXPECT_NEAR(en[g.index(i, gj_hi)], pattern(i, sj_hi, kBaseEn), kTol)
          << "energy periodic y_hi layer=" << layer << " i=" << i;
      EXPECT_NEAR(by[g.index(i, gj_lo)], pattern(i, sj_lo, kBaseBy), kTol)
          << "by_face periodic y_lo layer=" << layer << " i=" << i;
      EXPECT_NEAR(by[g.index(i, gj_hi)], pattern(i, sj_hi, kBaseBy), kTol)
          << "by_face periodic y_hi layer=" << layer << " i=" << i;
    }
  }
}

// ---------------------------------------------------------------------------
// OUTFLOW: cell-centered and tangential quantities copy the outermost interior
// cell into every ghost layer. A normal face uses its staggered physical extent
// 0..n: low ghosts copy face 0, high physical face n remains authoritative, and
// only face ghosts beyond n copy face n.
// ---------------------------------------------------------------------------
TEST(MhdBoundary, OutflowCopiesOutermostInterior) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D g = make_grid();
  MhdField2D<Real> f{g};
  seed_field(f);

  auto fluid = make_fluid("outflow");
  auto field = make_field("outflow");

  fluid->fill_ghosts(f, Side::x_lo);
  fluid->fill_ghosts(f, Side::x_hi);
  field->fill_ghosts(f, Side::x_lo);
  field->fill_ghosts(f, Side::x_hi);

  const std::vector<Real> rho = to_host(f.rho, g);
  const std::vector<Real> bx = to_host(f.bx_face, g);
  for (int layer = 1; layer <= g.nghost; ++layer) {
    for (int j = 0; j < g.ny; ++j) {
      EXPECT_NEAR(rho[g.index(-layer, j)], pattern(0, j, kBaseRho), kTol)
          << "rho outflow x_lo layer=" << layer << " j=" << j;
      EXPECT_NEAR(rho[g.index(g.nx - 1 + layer, j)], pattern(g.nx - 1, j, kBaseRho),
                  kTol)
          << "rho outflow x_hi layer=" << layer << " j=" << j;
      EXPECT_NEAR(bx[g.index(-layer, j)], pattern(0, j, kBaseBx), kTol)
          << "bx_face outflow x_lo layer=" << layer << " j=" << j;
      EXPECT_NEAR(bx[g.index(g.nx - 1 + layer, j)], pattern(g.nx, j, kBaseBx),
                  kTol)
          << "bx_face outflow x_hi physical/ghost layer=" << layer
          << " j=" << j;
    }
  }

  MhdField2D<Real> fy{g};
  seed_field(fy);
  fluid->fill_ghosts(fy, Side::y_lo);
  fluid->fill_ghosts(fy, Side::y_hi);
  field->fill_ghosts(fy, Side::y_lo);
  field->fill_ghosts(fy, Side::y_hi);
  const std::vector<Real> my = to_host(fy.my, g);
  const std::vector<Real> by = to_host(fy.by_face, g);
  for (int layer = 1; layer <= g.nghost; ++layer) {
    for (int i = 0; i < g.nx; ++i) {
      EXPECT_NEAR(my[g.index(i, -layer)], pattern(i, 0, kBaseMy), kTol)
          << "my outflow y_lo layer=" << layer << " i=" << i;
      EXPECT_NEAR(my[g.index(i, g.ny - 1 + layer)], pattern(i, g.ny - 1, kBaseMy),
                  kTol)
          << "my outflow y_hi layer=" << layer << " i=" << i;
      EXPECT_NEAR(by[g.index(i, -layer)], pattern(i, 0, kBaseBy), kTol)
          << "by_face outflow y_lo layer=" << layer << " i=" << i;
      EXPECT_NEAR(by[g.index(i, g.ny - 1 + layer)], pattern(i, g.ny, kBaseBy),
                  kTol)
          << "by_face outflow y_hi physical/ghost layer=" << layer
          << " i=" << i;
    }
  }
}

// ---------------------------------------------------------------------------
// WALL (perfectly-conducting wall): mirror about the wall. Tangential /
// cell-centered components are EVEN (copy); the NORMAL momentum and NORMAL
  // face-B are ODD (sign-flip) so v.n=0 and n.B=0. Cell and face mirror indices
  // differ by half a cell because Bn lives on the wall itself:
//   x_lo: ghost(-layer) <- interior(layer-1)
//   x_hi: ghost(nx-1+layer) <- interior(nx-layer)
// On an x-side: mx & bx_face are ODD; rho, my, energy, by_face, bz_cell EVEN.
// ---------------------------------------------------------------------------
TEST(MhdBoundary, WallMirrorsWithCorrectParityX) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D g = make_grid();
  MhdField2D<Real> f{g};
  seed_field(f);

  auto fluid = make_fluid("wall");
  auto field = make_field("wall");

  fluid->fill_ghosts(f, Side::x_lo);
  fluid->fill_ghosts(f, Side::x_hi);
  field->fill_ghosts(f, Side::x_lo);
  field->fill_ghosts(f, Side::x_hi);

  const std::vector<Real> rho = to_host(f.rho, g);
  const std::vector<Real> mx = to_host(f.mx, g);   // ODD on x
  const std::vector<Real> my = to_host(f.my, g);   // EVEN (tangential)
  const std::vector<Real> bx = to_host(f.bx_face, g);  // ODD on x (normal face)
  const std::vector<Real> by = to_host(f.by_face, g);  // EVEN (tangential)
  const std::vector<Real> bz = to_host(f.bz_cell, g);  // EVEN (cell)

  for (int layer = 1; layer <= g.nghost; ++layer) {
    const int gi_lo = -layer;
    const int gi_hi = g.nx - 1 + layer;
    const int si_lo = layer - 1;     // mirror source for lo
    const int si_hi = g.nx - layer;  // mirror source for hi
    for (int j = 0; j < g.ny; ++j) {
      // EVEN.
      EXPECT_NEAR(rho[g.index(gi_lo, j)], pattern(si_lo, j, kBaseRho), kTol)
          << "rho wall even x_lo layer=" << layer << " j=" << j;
      EXPECT_NEAR(rho[g.index(gi_hi, j)], pattern(si_hi, j, kBaseRho), kTol)
          << "rho wall even x_hi layer=" << layer << " j=" << j;
      EXPECT_NEAR(my[g.index(gi_lo, j)], pattern(si_lo, j, kBaseMy), kTol)
          << "my wall even x_lo layer=" << layer << " j=" << j;
      EXPECT_NEAR(by[g.index(gi_lo, j)], pattern(si_lo, j, kBaseBy), kTol)
          << "by_face wall even x_lo layer=" << layer << " j=" << j;
      EXPECT_NEAR(bz[g.index(gi_hi, j)], pattern(si_hi, j, kBaseBz), kTol)
          << "bz_cell wall even x_hi layer=" << layer << " j=" << j;
      // ODD (sign flip): normal momentum mx and normal face bx_face.
      EXPECT_NEAR(mx[g.index(gi_lo, j)], -pattern(si_lo, j, kBaseMx), kTol)
          << "mx wall odd x_lo layer=" << layer << " j=" << j;
      EXPECT_NEAR(mx[g.index(gi_hi, j)], -pattern(si_hi, j, kBaseMx), kTol)
          << "mx wall odd x_hi layer=" << layer << " j=" << j;
      EXPECT_NEAR(bx[g.index(gi_lo, j)], -pattern(layer, j, kBaseBx), kTol)
          << "bx_face wall odd x_lo layer=" << layer << " j=" << j;
      if (layer == 1) {
        EXPECT_EQ(bx[g.index(g.nx, j)], Real{0})
            << "bx_face x_hi wall value j=" << j;
      } else {
        const int q = layer - 1;
        EXPECT_NEAR(bx[g.index(g.nx + q, j)],
                    -pattern(g.nx - q, j, kBaseBx), kTol)
            << "bx_face wall odd x_hi layer=" << layer << " j=" << j;
      }
    }
  }
  for (int j = 0; j < g.ny; ++j) {
    EXPECT_EQ(bx[g.index(0, j)], Real{0}) << "bx_face x_lo wall value j=" << j;
  }
}

// On a y-side the NORMAL momentum is my and the NORMAL face-B is by_face (ODD);
// mx and bx_face are tangential (EVEN).
TEST(MhdBoundary, WallMirrorsWithCorrectParityY) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D g = make_grid();
  MhdField2D<Real> f{g};
  seed_field(f);

  auto fluid = make_fluid("wall");
  auto field = make_field("wall");

  fluid->fill_ghosts(f, Side::y_lo);
  fluid->fill_ghosts(f, Side::y_hi);
  field->fill_ghosts(f, Side::y_lo);
  field->fill_ghosts(f, Side::y_hi);

  const std::vector<Real> mx = to_host(f.mx, g);   // EVEN (tangential)
  const std::vector<Real> my = to_host(f.my, g);   // ODD on y (normal)
  const std::vector<Real> bx = to_host(f.bx_face, g);  // EVEN (tangential)
  const std::vector<Real> by = to_host(f.by_face, g);  // ODD on y (normal face)

  for (int layer = 1; layer <= g.nghost; ++layer) {
    const int gj_lo = -layer;
    const int gj_hi = g.ny - 1 + layer;
    const int sj_lo = layer - 1;
    const int sj_hi = g.ny - layer;
    for (int i = 0; i < g.nx; ++i) {
      // EVEN tangential.
      EXPECT_NEAR(mx[g.index(i, gj_lo)], pattern(i, sj_lo, kBaseMx), kTol)
          << "mx wall even y_lo layer=" << layer << " i=" << i;
      EXPECT_NEAR(bx[g.index(i, gj_hi)], pattern(i, sj_hi, kBaseBx), kTol)
          << "bx_face wall even y_hi layer=" << layer << " i=" << i;
      // ODD normal.
      EXPECT_NEAR(my[g.index(i, gj_lo)], -pattern(i, sj_lo, kBaseMy), kTol)
          << "my wall odd y_lo layer=" << layer << " i=" << i;
      EXPECT_NEAR(my[g.index(i, gj_hi)], -pattern(i, sj_hi, kBaseMy), kTol)
          << "my wall odd y_hi layer=" << layer << " i=" << i;
      EXPECT_NEAR(by[g.index(i, gj_lo)], -pattern(i, layer, kBaseBy), kTol)
          << "by_face wall odd y_lo layer=" << layer << " i=" << i;
      if (layer == 1) {
        EXPECT_EQ(by[g.index(i, g.ny)], Real{0})
            << "by_face y_hi wall value i=" << i;
      } else {
        const int q = layer - 1;
        EXPECT_NEAR(by[g.index(i, g.ny + q)],
                    -pattern(i, g.ny - q, kBaseBy), kTol)
            << "by_face wall odd y_hi layer=" << layer << " i=" << i;
      }
    }
  }
  for (int i = 0; i < g.nx; ++i) {
    EXPECT_EQ(by[g.index(i, 0)], Real{0}) << "by_face y_lo wall value i=" << i;
  }
}

// ---------------------------------------------------------------------------
// CORNER ghost blocks: after filling x-sides THEN y-sides (solver order), the
// four corner ghost blocks (both i and j in the ghost range) must be populated,
// NOT left at their seeded sentinel. The y-side fill spans the full storage
// width, copying the already-filled x-ghost columns into the y-ghost rows, which
// produces a DOUBLE-wrapped corner under periodic BCs.
//
// Lower-left corner block (i = -li, j = -lj): the y_lo fill at row j = -lj reads
// source row sj = ny-lj across the full width, including the x-ghost column
// i = -li, which the x_lo fill already set to interior column nx-li. So the
// corner equals pattern(nx-li, ny-lj). We assert it differs from the stale seed
// sentinel pattern(-li, -lj) and equals the predicted double-wrap.
// ---------------------------------------------------------------------------
TEST(MhdBoundary, PeriodicFillsCornerGhostBlocks) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D g = make_grid();
  MhdField2D<Real> f{g};
  seed_field(f);

  auto fluid = make_fluid("periodic");
  auto field = make_field("periodic");

  // Solver order: all x-sides first, then all y-sides.
  fluid->fill_ghosts(f, Side::x_lo);
  fluid->fill_ghosts(f, Side::x_hi);
  fluid->fill_ghosts(f, Side::y_lo);
  fluid->fill_ghosts(f, Side::y_hi);
  field->fill_ghosts(f, Side::x_lo);
  field->fill_ghosts(f, Side::x_hi);
  field->fill_ghosts(f, Side::y_lo);
  field->fill_ghosts(f, Side::y_hi);

  const std::vector<Real> rho = to_host(f.rho, g);
  const std::vector<Real> bx = to_host(f.bx_face, g);

  for (int li = 1; li <= g.nghost; ++li) {
    for (int lj = 1; lj <= g.nghost; ++lj) {
      // Lower-left corner (-li, -lj): x_lo set column -li to nx-li; y_lo then
      // wraps row -lj to ny-lj across that column -> pattern(nx-li, ny-lj).
      const std::size_t k_ll = g.index(-li, -lj);
      const Real stale_ll = pattern(-li, -lj, kBaseRho);
      const Real expect_ll = pattern(g.nx - li, g.ny - lj, kBaseRho);
      EXPECT_NE(rho[k_ll], stale_ll)
          << "rho lower-left corner stale at (" << -li << "," << -lj << ")";
      EXPECT_NEAR(rho[k_ll], expect_ll, kTol)
          << "rho lower-left corner double-wrap at (" << -li << "," << -lj << ")";

      // Upper-right corner (nx-1+li, ny-1+lj): x_hi set column nx-1+li to li-1;
      // y_hi wraps row ny-1+lj to lj-1 -> pattern(li-1, lj-1).
      const std::size_t k_ur = g.index(g.nx - 1 + li, g.ny - 1 + lj);
      const Real expect_ur = pattern(li - 1, lj - 1, kBaseBx);
      EXPECT_NEAR(bx[k_ur], expect_ur, kTol)
          << "bx_face upper-right corner double-wrap at ("
          << (g.nx - 1 + li) << "," << (g.ny - 1 + lj) << ")";
    }
  }
}

// ---------------------------------------------------------------------------
// NO OUT-OF-BOUNDS / no corruption of the shared extent. The hi face ghost for
// bx_face/by_face at layer == nghost lives at index nx-1+nghost (j analogously),
// NOT nx+nghost. Seed a guard pattern across the FULL storage, fill ALL ghosts
// (periodic), and assert that every INTERIOR cell is unchanged (boundaries only
// write ghosts) and every touched index stays within [0, storage_size()).
// ---------------------------------------------------------------------------
TEST(MhdBoundary, BoundaryWritesStayWithinSharedStorage) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D g = make_grid();
  MhdField2D<Real> f{g};
  seed_field(f);

  // Snapshot the seeded face arrays for the interior-unchanged check.
  const std::vector<Real> bx0 = to_host(f.bx_face, g);
  const std::vector<Real> by0 = to_host(f.by_face, g);

  auto field = make_field("periodic");
  field->fill_ghosts(f, Side::x_lo);
  field->fill_ghosts(f, Side::x_hi);
  field->fill_ghosts(f, Side::y_lo);
  field->fill_ghosts(f, Side::y_hi);

  const std::vector<Real> bx1 = to_host(f.bx_face, g);
  const std::vector<Real> by1 = to_host(f.by_face, g);

  // The highest hi face ghost target index must exist within storage.
  const std::size_t hi_x_face = g.index(g.nx - 1 + g.nghost, g.ny - 1);
  const std::size_t hi_y_face = g.index(g.nx - 1, g.ny - 1 + g.nghost);
  EXPECT_LT(hi_x_face, g.storage_size());
  EXPECT_LT(hi_y_face, g.storage_size());

  // Interior cells (i in [0,nx), j in [0,ny)) must be untouched by ghost fills.
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const std::size_t k = g.index(i, j);
      EXPECT_EQ(bx1[k], bx0[k]) << "bx_face interior changed at (" << i << "," << j << ")";
      EXPECT_EQ(by1[k], by0[k]) << "by_face interior changed at (" << i << "," << j << ")";
    }
  }

  // The hi face ghost at layer == nghost must hold the wrapped interior value
  // (face wrap: nx-1+nghost <- nghost-1), confirming the write landed at the
  // shared-extent slot and not one column past it.
  for (int j = 0; j < g.ny; ++j) {
    EXPECT_NEAR(bx1[g.index(g.nx - 1 + g.nghost, j)],
                pattern(g.nghost - 1, j, kBaseBx), kTol)
        << "bx_face hi ghost at layer==nghost wrong at j=" << j;
  }
}
