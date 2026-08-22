// Constrained-transport device kernels: corner-EMF construction and the
// face-staggered B update. Staggering convention (mhd_field.hpp / ct_scheme.hpp):
//   bx_face(i,j) = Bx on the x_lo (left)   face of cell (i,j)
//   by_face(i,j) = By on the y_lo (bottom) face of cell (i,j)
//   ez_edge(i,j) = Ez at the lower-left corner of cell (i,j)
//   divB(i,j) = (bx_face(i+1,j)-bx_face(i,j))/dx + (by_face(i,j+1)-by_face(i,j))/dy
// The face-B update curl below is chosen so the matching finite-volume
// divergence telescopes to identically zero for any corner EMF.  Cartesian uses
// the ordinary face difference.  Cylindrical (r,z) uses r-weighted radial faces
// and annular cell volume, preserving (1/r)d(r Br)/dr+dBz/dz at round-off.

#include "quasar/core/grid.hpp"
#include "quasar/numerics/hlld_core.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include "backend/hip/hip_check.hpp"
#include "mhd_face_quadrature.hpp"

#include <hip/hip_runtime.h>

namespace {

using quasar::Real;
using quasar::numerics::MhdFlux;
using quasar::numerics::MhdState;
using quasar::mhd::detail::FaceBackgroundView;
using quasar::mhd::detail::FaceInterfaceView;

inline constexpr unsigned kBlock2D = 16;

// Range-safe rational interpolation.  High-order coefficients are large
// enough that coefficient*EMF overflows for a perfectly representable constant
// near DBL_MAX; adding the signed numerator before dividing has the same
// problem.  Each normalized coefficient*EMF product enters as an error-free
// FMA two-product expansion; the shared reducer retains every high/low
// component through one rounded numerator before applying the common rational
// denominator once. This guarantees exact-zero numerator cancellation and
// range safety, but not exact-rational correct rounding in every double-round
// edge case.
template <int Capacity>
__device__ inline Real finish_scaled_ct_terms(
    quasar::numerics::ScaledProductQuotientAccumulator<Capacity>& sum,
    Real denominator) {
  const quasar::numerics::ScaledValue numerator =
      quasar::numerics::finish_scaled_exact_product_sum_to_value(sum);
  return quasar::numerics::scaled_value_divide(numerator, denominator);
}

__device__ __attribute__((noinline)) Real scaled_face_average(Real e0, Real e1) {
  if (e0 == e1) return e0;
  if (!(isfinite(e0) && isfinite(e1))) {
    return Real{0.5} * e0 + Real{0.5} * e1;
  }
  quasar::numerics::ScaledProductQuotientAccumulator<4> sum;
  quasar::numerics::append_scaled_exact_product(sum, Real{1}, e0);
  quasar::numerics::append_scaled_exact_product(sum, Real{1}, e1);
  return finish_scaled_ct_terms(sum, Real{2});
}

__device__ __attribute__((noinline)) Real scaled_face_average_to_edge5(
    Real e0, Real e1, Real e2, Real e3, Real e4, Real e5) {
  if (e0 == e1 && e0 == e2 && e0 == e3 && e0 == e4 && e0 == e5) return e0;
  if (!(isfinite(e0) && isfinite(e1) && isfinite(e2) &&
        isfinite(e3) && isfinite(e4) && isfinite(e5))) {
    return (e0 - Real{8} * e1 + Real{37} * e2 + Real{37} * e3 -
            Real{8} * e4 + e5) / Real{60};
  }
  quasar::numerics::ScaledProductQuotientAccumulator<12> sum;
  quasar::numerics::append_scaled_exact_product(sum, Real{1}, e0);
  quasar::numerics::append_scaled_exact_product(sum, Real{-8}, e1);
  quasar::numerics::append_scaled_exact_product(sum, Real{37}, e2);
  quasar::numerics::append_scaled_exact_product(sum, Real{37}, e3);
  quasar::numerics::append_scaled_exact_product(sum, Real{-8}, e4);
  quasar::numerics::append_scaled_exact_product(sum, Real{1}, e5);
  return finish_scaled_ct_terms(sum, Real{60});
}

__device__ __attribute__((noinline)) Real scaled_face_average_to_edge7(
    Real e0, Real e1, Real e2, Real e3,
    Real e4, Real e5, Real e6, Real e7) {
  if (e0 == e1 && e0 == e2 && e0 == e3 && e0 == e4 && e0 == e5 &&
      e0 == e6 && e0 == e7) return e0;
  if (!(isfinite(e0) && isfinite(e1) && isfinite(e2) && isfinite(e3) &&
        isfinite(e4) && isfinite(e5) && isfinite(e6) && isfinite(e7))) {
    return (-Real{3} * e0 + Real{29} * e1 - Real{139} * e2 +
            Real{533} * e3 + Real{533} * e4 - Real{139} * e5 +
            Real{29} * e6 - Real{3} * e7) / Real{840};
  }
  quasar::numerics::ScaledProductQuotientAccumulator<16> sum;
  quasar::numerics::append_scaled_exact_product(sum, Real{-3}, e0);
  quasar::numerics::append_scaled_exact_product(sum, Real{29}, e1);
  quasar::numerics::append_scaled_exact_product(sum, Real{-139}, e2);
  quasar::numerics::append_scaled_exact_product(sum, Real{533}, e3);
  quasar::numerics::append_scaled_exact_product(sum, Real{533}, e4);
  quasar::numerics::append_scaled_exact_product(sum, Real{-139}, e5);
  quasar::numerics::append_scaled_exact_product(sum, Real{29}, e6);
  quasar::numerics::append_scaled_exact_product(sum, Real{-3}, e7);
  return finish_scaled_ct_terms(sum, Real{840});
}

template <class Sample>
__device__ __attribute__((noinline)) Real scaled_face_average_to_edge_weighted(
    const Real* coefficients, int width, const Sample& sample) {
  const Real reference = sample(width / 2);
  bool constant = true;
  bool finite = true;
  for (int k = 0; k < width; ++k) {
    const Real value = sample(k);
    constant = constant && value == reference;
    finite = finite && isfinite(value);
  }
  if (constant) return reference;
  if (!finite) {
    Real fallback = Real{0};
    for (int k = 0; k < width; ++k) {
      fallback += coefficients[k] * sample(k);
    }
    return fallback;
  }
  quasar::numerics::ScaledProductQuotientAccumulator<16> sum;
  for (int k = 0; k < width; ++k) {
    quasar::numerics::append_scaled_exact_product(
        sum, coefficients[k], sample(k));
  }
  return quasar::numerics::scaled_value_divide(
      quasar::numerics::finish_scaled_exact_product_sum_to_value(sum),
      Real{1});
}

// Directional Godunov electric field.  For an x-face F_x(By)=-Ez; for a
// y-face F_y(Bx)=+Ez.  Reusing HLLD here gives CT the same upwind wave fan as
// the fluid flux instead of a disconnected centred cell-EMF average.
__device__ inline Real face_ez(const quasar::Grid2D& g, int dir, int i, int j,
                               int order, int hll_only, Real gamma,
                               const FaceInterfaceView& v,
                               const FaceBackgroundView& bgv,
                               quasar::numerics::RadialTablesView radial_tables) {
  const MhdFlux f = quasar::mhd::detail::average_face_flux(
      g, dir, i, j, order, hll_only, gamma, v, bgv, radial_tables);
  return (dir == 0) ? -f.by : f.bx;
}

__device__ inline int face_index(int q, int n, int lo_mode, int hi_mode) {
  if (lo_mode == 0 && hi_mode == 0) {
    int w = q % n;
    if (w < 0) w += n;
    return w;
  }
  // Internal sides (mode 4) retain the requested guard coordinate. Physical
  // sides clamp to their one-sided closure. The widest actual MP7 interpolation
  // asks for [-nghost, n+nghost-1], all of which is present in padded storage.
  if (q < 0) return lo_mode == 4 ? q : 0;
  if (q > n) return hi_mode == 4 ? q : n;
  return q;
}

__device__ inline bool interface_is_equal_state(
    std::size_t k, const FaceInterfaceView& v) {
  return v.Lrho[k] == v.Rrho[k] && v.Lmx[k] == v.Rmx[k] &&
         v.Lmy[k] == v.Rmy[k] && v.Lmz[k] == v.Rmz[k] &&
         v.Lenergy[k] == v.Renergy[k] && v.Lbx[k] == v.Rbx[k] &&
         v.Lby[k] == v.Rby[k] && v.Lbz[k] == v.Rbz[k];
}

// Summarize the complete point-recovered Riemann problem at one directional
// face.  MP5/MP7 recover each Gauss-point state from five/seven neighbouring
// transverse face averages, respectively, so equality of only the central
// average is not enough to prove that the point states have no jump.  The
// distributed runtime exchanges this one-bit summary with the face EMF table;
// consumers can therefore prove the corner's exact one-dimensional limit
// without reaching through another tile's numerical halo.
__device__ inline bool face_problem_has_no_jump(
    const quasar::Grid2D& g, int dir, int i, int j, int order,
    const FaceInterfaceView& v, quasar::mhd::BoundaryFlags4 modes) {
  const int recovery_half = order >= 7 ? 3 : order >= 5 ? 2 : 0;
  for (int offset = -recovery_half; offset <= recovery_half; ++offset) {
    const int sample_i = dir == 0 ? i : i + offset;
    const int sample_j = dir == 0 ? j + offset : j;
    // Preserve the serial boundary proof: periodic coordinates wrap and
    // physical coordinates clamp to their one-sided closure. Internal mode 4
    // deliberately retains the exchanged guard coordinate.
    const int mapped_i =
        face_index(sample_i, g.nx, modes.side[0], modes.side[1]);
    const int mapped_j =
        face_index(sample_j, g.ny, modes.side[2], modes.side[3]);
    if (mapped_i < -g.nghost || mapped_i >= g.nx + g.nghost ||
        mapped_j < -g.nghost || mapped_j >= g.ny + g.nghost ||
        !interface_is_equal_state(g.index(mapped_i, mapped_j), v)) {
      return false;
    }
  }
  return true;
}

// Prove that every directional face problem feeding this corner has no jump.
// Each table entry already summarizes the complete point-recovered face solve,
// so the proof reads only the actual face-average-to-edge interpolation range.
// In a distributed run those entries are exchanged with the face EMF table;
// no interface-state read reaches beyond the numerical 2/3/4-cell halo.
__device__ inline bool directional_corner_stencil_has_no_jump(
    const quasar::Grid2D& g, int dir, int i, int j, int order,
    const int* no_jump, quasar::mhd::BoundaryFlags4 modes,
    quasar::numerics::RadialTablesView radial_tables) {
  // The y-face table varies in i, so an active cylindrical R5 row defines its
  // complete interpolation stencil.  In particular MUSCL owns a four-point
  // collocation row even though its Cartesian corner interpolation has only
  // two taps; checking the shorter range could falsely prove a 1-D limit while
  // an unseen R5 tap still contains a jump.
  const bool radial_stencil =
      dir == 1 && radial_tables.active != 0 &&
      radial_tables.scheme_order == order && radial_tables.contains(i);
  const int interpolation_lo = radial_stencil
      ? -radial_tables.collocation_width / 2
      : order >= 7 ? -4 : order >= 5 ? -3 : -1;
  const int interpolation_hi = radial_stencil
      ? interpolation_lo + radial_tables.collocation_width - 1
      : order >= 7 ? 3 : order >= 5 ? 2 : 0;
  for (int offset = interpolation_lo; offset <= interpolation_hi; ++offset) {
    const int sample_i = dir == 0 ? i : i + offset;
    const int sample_j = dir == 0 ? j + offset : j;
    const int ii = face_index(
        sample_i, g.nx, modes.side[0], modes.side[1]);
    const int jj = face_index(
        sample_j, g.ny, modes.side[2], modes.side[3]);
    if (ii < -g.nghost || ii >= g.nx + g.nghost ||
        jj < -g.nghost || jj >= g.ny + g.nghost ||
        no_jump[g.index(ii, jj)] == 0) {
      return false;
    }
  }
  return true;
}

// Directional face EMFs, read from the precomputed per-face tables. face_index
// maps any requested tap into [0,n] (one-sided closure) or [0,n) (periodic
// wrap), so a table covering [0,nx] x [0,ny] holds every value a corner stencil
// can ask for. The mapping is applied here, exactly as the original
// compute-on-demand form did, so the table index matches the value that form
// would have computed.
__device__ inline Real xface_ez(const quasar::Grid2D& g, int i, int j,
                                const double* __restrict__ table,
                                quasar::mhd::BoundaryFlags4 modes) {
  const int ii = face_index(i, g.nx, modes.side[0], modes.side[1]);
  const int jj = face_index(j, g.ny, modes.side[2], modes.side[3]);
  return table[g.index(ii, jj)];
}

__device__ inline Real yface_ez(const quasar::Grid2D& g, int i, int j,
                                const double* __restrict__ table,
                                quasar::mhd::BoundaryFlags4 modes) {
  const int ii = face_index(i, g.nx, modes.side[0], modes.side[1]);
  const int jj = face_index(j, g.ny, modes.side[2], modes.side[3]);
  return table[g.index(ii, jj)];
}

__device__ inline Real interp_x_to_corner(
    const quasar::Grid2D& g, int i, int j, int order,
    const double* __restrict__ x_table, quasar::mhd::BoundaryFlags4 modes) {
  if (order >= 7) {
    return scaled_face_average_to_edge7(
        xface_ez(g, i, j - 4, x_table, modes),
        xface_ez(g, i, j - 3, x_table, modes),
        xface_ez(g, i, j - 2, x_table, modes),
        xface_ez(g, i, j - 1, x_table, modes),
        xface_ez(g, i, j, x_table, modes),
        xface_ez(g, i, j + 1, x_table, modes),
        xface_ez(g, i, j + 2, x_table, modes),
        xface_ez(g, i, j + 3, x_table, modes));
  }
  if (order >= 5) {
    return scaled_face_average_to_edge5(
        xface_ez(g, i, j - 3, x_table, modes),
        xface_ez(g, i, j - 2, x_table, modes),
        xface_ez(g, i, j - 1, x_table, modes),
        xface_ez(g, i, j, x_table, modes),
        xface_ez(g, i, j + 1, x_table, modes),
        xface_ez(g, i, j + 2, x_table, modes));
  }
  return scaled_face_average(
      xface_ez(g, i, j - 1, x_table, modes),
      xface_ez(g, i, j, x_table, modes));
}

__device__ inline Real interp_y_to_corner(
    const quasar::Grid2D& g, int i, int j, int order,
    const double* __restrict__ y_table, quasar::mhd::BoundaryFlags4 modes,
    quasar::numerics::RadialTablesView radial_tables) {
  if (radial_tables.active != 0 && radial_tables.scheme_order == order &&
      radial_tables.contains(i)) {
    const int width = radial_tables.collocation_width;
    const int start = i - width / 2;
    return scaled_face_average_to_edge_weighted(
        radial_tables.r5_row(i), width, [&](int k) {
          return yface_ez(g, start + k, j, y_table, modes);
        });
  }
  if (order >= 7) {
    return scaled_face_average_to_edge7(
        yface_ez(g, i - 4, j, y_table, modes),
        yface_ez(g, i - 3, j, y_table, modes),
        yface_ez(g, i - 2, j, y_table, modes),
        yface_ez(g, i - 1, j, y_table, modes),
        yface_ez(g, i, j, y_table, modes),
        yface_ez(g, i + 1, j, y_table, modes),
        yface_ez(g, i + 2, j, y_table, modes),
        yface_ez(g, i + 3, j, y_table, modes));
  }
  if (order >= 5) {
    return scaled_face_average_to_edge5(
        yface_ez(g, i - 3, j, y_table, modes),
        yface_ez(g, i - 2, j, y_table, modes),
        yface_ez(g, i - 1, j, y_table, modes),
        yface_ez(g, i, j, y_table, modes),
        yface_ez(g, i + 1, j, y_table, modes),
        yface_ez(g, i + 2, j, y_table, modes));
  }
  return scaled_face_average(
      yface_ez(g, i - 1, j, y_table, modes),
      yface_ez(g, i, j, y_table, modes));
}

// Fill canonical directional face-EMF entries only.  The normal coordinate
// spans [0,n]; the transverse coordinate spans owned cells plus the true
// physical high closure.  Internal/periodic guards are populated from the
// derived-table exchange (or mapped locally by the serial corner kernel).
// Avoiding padded transverse face solves is essential: MP7 face quadrature at
// an outer guard would recursively request interface state seven cells deep.
__global__ void QUASAR_LAUNCH_BOUNDS_256 face_emf_kernel(
    quasar::Grid2D g, int scheme_order,
    quasar::numerics::RadialTablesView radial_tables,
    int hll_only, Real gamma,
    quasar::mhd::BoundaryFlags4 modes,
    FaceInterfaceView ifx, FaceInterfaceView ify, FaceBackgroundView bg,
    double* __restrict__ xface_table, double* __restrict__ yface_table,
    int* __restrict__ xface_no_jump,
    int* __restrict__ yface_no_jump) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > g.nx || j > g.ny) return;
  const std::size_t k = g.index(i, j);
  const bool physical_y_high =
      modes.side[3] != 0 && modes.side[3] != 4;
  const bool physical_x_high =
      modes.side[1] != 0 && modes.side[1] != 4;
  if (j < g.ny || (j == g.ny && physical_y_high)) {
    xface_table[k] = face_ez(
        g, 0, i, j, scheme_order, hll_only, gamma, ifx, bg, radial_tables);
    xface_no_jump[k] =
        face_problem_has_no_jump(g, 0, i, j, scheme_order, ifx, modes) ? 1 : 0;
  }
  if (i < g.nx || (i == g.nx && physical_x_high)) {
    yface_table[k] = face_ez(
        g, 1, i, j, scheme_order, hll_only, gamma, ify, bg, radial_tables);
    yface_no_jump[k] =
        face_problem_has_no_jump(g, 1, i, j, scheme_order, ify, modes) ? 1 : 0;
  }
}

struct CellStateView {
  const double *rho, *mx, *my, *mz, *energy, *bx, *by, *bz;
};

// Widest transverse stencil: MP7 reads 7 taps, MP5 reads 5 and leaves the tail
// of each row table unused.
inline constexpr int kMaxCellTaps = 7;

// The tensor point reconstruction at cell node (node_x, node_y) is
//   point_value_y( oy -> point_value_x( ox -> sample(ox, oy) ) ),
// whose INNER factor depends on (node_x, oy) only -- not on node_y. The two
// helpers below split that tensor product so the inner factor is built once per
// x-node and reused across every y-node, instead of being recomputed for each
// of the `count` y-nodes as the nested form did.
//
// This is pure common-subexpression elimination: `sample` is a side-effect-free
// function of memory, so each reused value is bit-identical to the one the
// nested form recomputed, and `point_value` still receives exactly the same
// operands in exactly the same order. The reduction order -- and therefore the
// floating-point result -- is unchanged.

// rows[k] is the x-direction point reconstruction on the cell row at transverse
// offset k-half, for k in [0, taps).
template <class Sample>
__device__ inline void cell_point_rows(int order, int node_x,
                                       int radial_index,
                                       quasar::numerics::RadialTablesView radial_tables,
                                       const Real* component_radial_row,
                                       const Sample& sample, Real* rows) {
  const int half = order >= 7 ? 3 : 2;
  const int taps = order >= 7 ? 7 : 5;
  for (int k = 0; k < taps; ++k) {
    const int oy = k - half;
    rows[k] = quasar::mhd::detail::point_value(
        order, node_x, /*dir=*/1, radial_index, radial_tables,
        component_radial_row,
        [&](int ox) { return sample(ox, oy); });
  }
}

// Reduce a row table to the tensor point value at y-node `node_y`. point_value
// samples transverse offsets oy in [-half, half] (plus the centre reference),
// which index the table as oy+half.
__device__ inline Real cell_point_from_rows(int order, int node_y,
                                            const Real* rows) {
  const int half = order >= 7 ? 3 : 2;
  return quasar::mhd::detail::point_value(
      order, node_y, [&](int oy) { return rows[oy + half]; });
}

__device__ inline Real physical_ez(
    const MhdState& u, const quasar::numerics::MhdBackground& bg) {
  const Real vx = u.mx / u.rho;
  const Real vy = u.my / u.rho;
  // E_z = -(v x B)_z = vy*(b_x+B0_x) - vx*(b_y+B0_y).
  return quasar::numerics::product_sum4(
      vy, u.bx, vy, bg.b0x, vx, -u.by, vx, -bg.b0y);
}

__device__ inline Real direct_cell_ez(
    const quasar::Grid2D& g, int i, int j, int collocation_order,
    const CellStateView& u, const FaceBackgroundView& bg,
    quasar::numerics::RadialTablesView radial_tables) {
  const MhdState state = quasar::mhd::load_cell_state(
      g, u.rho, u.mx, u.my, u.mz, u.energy, u.bx, u.by, u.bz,
      i, j, radial_tables, collocation_order);
  const auto background = bg.active != 0
      ? quasar::mhd::load_cell_background(
            g, bg.bx, bg.by, bg.bz, i, j, radial_tables, collocation_order)
      : quasar::numerics::MhdBackground{};
  return physical_ez(state, background);
}

// Block-shared cell-centred Bx/By tile.
//
// cell_bx/cell_by turn staggered face storage into a cell average with their
// own multi-tap reduction, and every cell in a block needs that value for the
// whole (2*half+1)^2 stencil around it. Neighbouring cells therefore ask for
// the same cell's value repeatedly -- across the block, each one is recomputed
// up to (2*half+1) times. Deriving the tile once per block and reading it from
// LDS removes that; the reduction itself is untouched, so each stored value is
// the same bits cell_bx/cell_by returned before.
//
// Sized for the 16x16 launch block plus the MP7 halo (3 each side).
inline constexpr int kEmfTileHalo = 3;
inline constexpr int kEmfTileDim = 16 + 2 * kEmfTileHalo;  // 22

struct CellBTile {
  // [x][y], indexed by (cell - tile_origin) + halo.
  Real bx[kEmfTileDim][kEmfTileDim];
  Real by[kEmfTileDim][kEmfTileDim];
  int origin_i;
  int origin_j;
  int valid;  // 0 => tile not usable, fall back to direct evaluation
};

__device__ inline Real cell_average_ez(
    const quasar::Grid2D& g, int i, int j, int order, Real gamma,
    const CellStateView& u, const FaceBackgroundView& bg,
    const CellBTile* tile,
    quasar::numerics::RadialTablesView radial_tables) {
  const int collocation_order = order <= 1 ? 1 : 0;
  const Real fallback = direct_cell_ez(
      g, i, j, collocation_order, u, bg, radial_tables);
  if (order != 5 && order != 7) return fallback;

  const int half = order == 5 ? 2 : 3;
  const int lo_i = -g.nghost;
  const int hi_i = g.nx + g.nghost - 1;
  const int lo_j = -g.nghost;
  const int hi_j = g.ny + g.nghost - 1;
  if (i - half < lo_i || i + half > hi_i ||
      j - half < lo_j || j + half > hi_j) {
    // Only the outermost padded cells take this closure; every physical cell
    // and the ghost layers used by interior high-order corners use the tensor
    // quadrature below.
    return fallback;
  }

  Real point[4][4]{};
  const int count = order == 5 ? 3 : 4;
  const int taps = 2 * half + 1;

  // Cell-centred Bx/By and the background are recovered from staggered face
  // storage by their own multi-tap reductions (cell_bx / cell_by /
  // load_cell_background). Their value at (i+ox, j+oy) depends only on that
  // cell, so the whole stencil block is materialised ONCE here. Sampling them
  // inside cell_point_rows instead would re-derive each of the taps x taps
  // cells for every x-node -- `count` times over -- and each re-derivation is
  // itself an 8-tap noinline reduction. Indexed [ox+half][oy+half].
  Real bx_block[kMaxCellTaps][kMaxCellTaps];
  Real by_block[kMaxCellTaps][kMaxCellTaps];
  Real bg_block[3][kMaxCellTaps][kMaxCellTaps];
  // Read Bx/By from the block-shared tile when this cell's whole stencil lies
  // inside it (the common case); otherwise derive them here. Both paths call
  // the same cell_bx/cell_by, so the values are identical either way.
  //
  // The stencil must also stay inside the padded domain, because tile entries
  // outside it were filled with zero rather than evaluated. The caller's
  // in-range guard above already established that for THIS cell's stencil, so
  // the tile-extent test is the only additional condition.
  const bool use_tile =
      tile != nullptr && tile->valid != 0 &&
      i - half >= tile->origin_i - kEmfTileHalo &&
      i + half <= tile->origin_i + kEmfTileDim - 1 - kEmfTileHalo &&
      j - half >= tile->origin_j - kEmfTileHalo &&
      j + half <= tile->origin_j + kEmfTileDim - 1 - kEmfTileHalo;
  for (int a = 0; a < taps; ++a) {
    for (int b = 0; b < taps; ++b) {
      const int ii = i + a - half;
      const int jj = j + b - half;
      if (use_tile) {
        const int ti = ii - tile->origin_i + kEmfTileHalo;
        const int tj = jj - tile->origin_j + kEmfTileHalo;
        bx_block[a][b] = tile->bx[ti][tj];
        by_block[a][b] = tile->by[ti][tj];
      } else {
        bx_block[a][b] = quasar::mhd::cell_bx(
            g, u.bx, ii, jj, radial_tables);
        by_block[a][b] = quasar::mhd::cell_by(
            g, u.by, ii, jj, radial_tables);
      }
      if (bg.active != 0) {
        const auto value = quasar::mhd::load_cell_background(
            g, bg.bx, bg.by, bg.bz, ii, jj, radial_tables);
        bg_block[0][a][b] = value.b0x;
        bg_block[1][a][b] = value.b0y;
        bg_block[2][a][b] = value.b0z;
      }
    }
  }

  // Per x-node row tables for the eight state components and (when active) the
  // three background components. Built once per qx and reused for every qy,
  // which is where the redundant work in the nested formulation was: the inner
  // x-reconstruction does not depend on qy.
  const double* const raw_components[6] = {u.rho, u.mx, u.my, u.mz,
                                           u.energy, u.bz};
  Real raw_rows[6][kMaxCellTaps];
  Real bx_rows[kMaxCellTaps];
  Real by_rows[kMaxCellTaps];
  Real bg_rows[3][kMaxCellTaps];

  for (int qx = 0; qx < count; ++qx) {
    for (int c = 0; c < 6; ++c) {
      const double* values = raw_components[c];
      const Real* component_row = radial_tables.active != 0 &&
              radial_tables.contains(i)
          ? c == 3 ? radial_tables.r2_mphi_row(i, qx)
                   : c == 5 ? radial_tables.r2_bphi_row(i, qx) : nullptr
          : nullptr;
      cell_point_rows(
          order, qx, i, radial_tables, component_row,
          [&](int ox, int oy) {
            return static_cast<Real>(values[g.index(i + ox, j + oy)]);
          }, raw_rows[c]);
    }
    cell_point_rows(
        order, qx, i, radial_tables, /*component_radial_row=*/nullptr,
        [&](int ox, int oy) { return bx_block[ox + half][oy + half]; },
        bx_rows);
    cell_point_rows(
        order, qx, i, radial_tables, /*component_radial_row=*/nullptr,
        [&](int ox, int oy) { return by_block[ox + half][oy + half]; },
        by_rows);
    if (bg.active != 0) {
      for (int which = 0; which < 3; ++which) {
        const Real* component_row = which == 2 &&
                radial_tables.active != 0 && radial_tables.contains(i)
            ? radial_tables.r2_bphi_row(i, qx) : nullptr;
        cell_point_rows(
            order, qx, i, radial_tables, component_row,
            [&](int ox, int oy) {
              return bg_block[which][ox + half][oy + half];
            }, bg_rows[which]);
      }
    }

    for (int qy = 0; qy < count; ++qy) {
      const MhdState state{
          cell_point_from_rows(order, qy, raw_rows[0]),
          cell_point_from_rows(order, qy, raw_rows[1]),
          cell_point_from_rows(order, qy, raw_rows[2]),
          cell_point_from_rows(order, qy, raw_rows[3]),
          cell_point_from_rows(order, qy, raw_rows[4]),
          cell_point_from_rows(order, qy, bx_rows),
          cell_point_from_rows(order, qy, by_rows),
          cell_point_from_rows(order, qy, raw_rows[5])};
      if (!quasar::mhd::detail::state_is_admissible(state, gamma)) {
        return fallback;
      }
      const quasar::numerics::MhdBackground background =
          bg.active == 0
              ? quasar::numerics::MhdBackground{}
              : quasar::numerics::MhdBackground{
                    cell_point_from_rows(order, qy, bg_rows[0]),
                    cell_point_from_rows(order, qy, bg_rows[1]),
                    cell_point_from_rows(order, qy, bg_rows[2])};
      point[qy][qx] = physical_ez(state, background);
    }
  }

  Real row[4]{};
  const Real* radial_weights = radial_tables.active != 0 &&
          radial_tables.contains(i)
      ? radial_tables.r3_row(i) : nullptr;
  for (int qy = 0; qy < count; ++qy) {
    if (radial_weights != nullptr) {
      row[qy] = quasar::numerics::weighted_quadrature_value(
          radial_weights, count, [&](int qx) { return point[qy][qx]; });
    } else if (order == 5) {
      row[qy] = quasar::numerics::weighted_quadrature_value(
          quasar::mhd::detail::kMp5TransverseGaussWeights,
          [&](int qx) { return point[qy][qx]; });
    } else {
      row[qy] = quasar::numerics::weighted_quadrature_value(
          quasar::mhd::detail::kMp7TransverseGaussWeights,
          [&](int qx) { return point[qy][qx]; });
    }
  }
  if (order == 5) {
    return quasar::numerics::weighted_quadrature_value(
        quasar::mhd::detail::kMp5TransverseGaussWeights,
        [&](int qy) { return row[qy]; });
  }
  return quasar::numerics::weighted_quadrature_value(
      quasar::mhd::detail::kMp7TransverseGaussWeights,
      [&](int qy) { return row[qy]; });
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 cell_emf_kernel(
    quasar::Grid2D g, int scheme_order,
    quasar::numerics::RadialTablesView radial_tables,
    Real gamma, CellStateView u,
    FaceBackgroundView bg, quasar::mhd::BoundaryFlags4 modes,
    double* __restrict__ cell_ez) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  // Cooperatively derive this block's cell-centred Bx/By tile before any thread
  // uses it. Only the MP paths read the tile; the others skip the fill so the
  // low-order launches keep their old cost. Every thread must reach the barrier,
  // so the out-of-range early return happens after it.
  __shared__ CellBTile tile;
  const bool tiled_order = (scheme_order == 5 || scheme_order == 7);
  if (tiled_order) {
    const int origin_i = static_cast<int>(blockIdx.x * blockDim.x);
    const int origin_j = static_cast<int>(blockIdx.y * blockDim.y);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      tile.origin_i = origin_i;
      tile.origin_j = origin_j;
      // cell_bx/cell_by index face storage over the tile's own stencil, so the
      // tile is only usable where that stays inside allocated storage. The
      // consumer re-checks its individual stencil against the tile extent.
      tile.valid = (blockDim.x == 16 && blockDim.y == 16) ? 1 : 0;
    }
    __syncthreads();
    if (tile.valid != 0) {
      // Flatten the (kEmfTileDim)^2 fill over the block's threads.
      const int threads = static_cast<int>(blockDim.x * blockDim.y);
      const int lane =
          static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
      const int total = kEmfTileDim * kEmfTileDim;
      const int lo_i = -g.nghost;
      const int hi_i = g.nx + g.nghost - 1;
      const int lo_j = -g.nghost;
      const int hi_j = g.ny + g.nghost - 1;
      for (int t = lane; t < total; t += threads) {
        const int a = t / kEmfTileDim;
        const int b = t % kEmfTileDim;
        const int ii = origin_i + a - kEmfTileHalo;
        const int jj = origin_j + b - kEmfTileHalo;
        // Outside the padded domain cell_bx/cell_by would read unallocated
        // storage. Those entries are never consumed (the guard in
        // cell_average_ez rejects any stencil reaching them), so leave them
        // zero rather than evaluating out of bounds.
        if (ii < lo_i || ii > hi_i || jj < lo_j || jj > hi_j) {
          tile.bx[a][b] = Real{0};
          tile.by[a][b] = Real{0};
          continue;
        }
        tile.bx[a][b] = quasar::mhd::cell_bx(
            g, u.bx, ii, jj, radial_tables);
        tile.by[a][b] = quasar::mhd::cell_by(
            g, u.by, ii, jj, radial_tables);
      }
    }
    __syncthreads();
  }

  if (i > g.nx || j > g.ny) return;
  const bool physical_x_high =
      modes.side[1] != 0 && modes.side[1] != 4;
  const bool physical_y_high =
      modes.side[3] != 0 && modes.side[3] != 4;
  if ((i == g.nx && !physical_x_high) ||
      (j == g.ny && !physical_y_high)) return;
  cell_ez[g.index(i, j)] = cell_average_ez(
      g, i, j, scheme_order, gamma, u, bg, tiled_order ? &tile : nullptr,
      radial_tables);
}

__device__ inline Real cell_emf_to_corner(
    const quasar::Grid2D& g, int i, int j, int order,
    const double* cell_ez, quasar::mhd::BoundaryFlags4 modes,
    quasar::numerics::RadialTablesView radial_tables) {
  const auto value = [&](int ii, int jj) {
    const int mapped_i = face_index(ii, g.nx, modes.side[0], modes.side[1]);
    const int mapped_j = face_index(jj, g.ny, modes.side[2], modes.side[3]);
    return cell_ez[g.index(mapped_i, mapped_j)];
  };
  if (radial_tables.active != 0 && radial_tables.scheme_order == order &&
      radial_tables.contains(i)) {
    const int radial_width = radial_tables.collocation_width;
    const int axial_width = order >= 7 ? 8 : order >= 5 ? 6 : 2;
    const int start_i = i - radial_width / 2;
    const int start_j = j - axial_width / 2;
    Real line[8]{};
    for (int q = 0; q < axial_width; ++q) {
      const int jj = start_j + q;
      line[q] = scaled_face_average_to_edge_weighted(
          radial_tables.r5_row(i), radial_width,
          [&](int k) { return value(start_i + k, jj); });
    }
    if (axial_width == 8) {
      return scaled_face_average_to_edge7(
          line[0], line[1], line[2], line[3],
          line[4], line[5], line[6], line[7]);
    }
    if (axial_width == 6) {
      return scaled_face_average_to_edge5(
          line[0], line[1], line[2], line[3], line[4], line[5]);
    }
    return scaled_face_average(line[0], line[1]);
  }
  if (order >= 7) {
    Real line[8];
    for (int q = 0; q < 8; ++q) {
      const int jj = j - 4 + q;
      line[q] = scaled_face_average_to_edge7(
          value(i - 4, jj), value(i - 3, jj), value(i - 2, jj),
          value(i - 1, jj), value(i, jj), value(i + 1, jj),
          value(i + 2, jj), value(i + 3, jj));
    }
    return scaled_face_average_to_edge7(
        line[0], line[1], line[2], line[3],
        line[4], line[5], line[6], line[7]);
  }
  if (order >= 5) {
    Real line[6];
    for (int q = 0; q < 6; ++q) {
      const int jj = j - 3 + q;
      line[q] = scaled_face_average_to_edge5(
          value(i - 3, jj), value(i - 2, jj), value(i - 1, jj),
          value(i, jj), value(i + 1, jj), value(i + 2, jj));
    }
    return scaled_face_average_to_edge5(
        line[0], line[1], line[2], line[3], line[4], line[5]);
  }
  const Real lo = scaled_face_average(
      value(i - 1, j - 1), value(i, j - 1));
  const Real hi = scaled_face_average(
      value(i - 1, j), value(i, j));
  return scaled_face_average(lo, hi);
}

// Exact local one-dimensionality check for the cell-centred reference.  Keep
// the coordinate that may vary (`1-axis`) separate and require bit-identical
// values along `axis` across the complete tensor edge-recovery stencil.
__device__ inline bool cell_emf_is_constant_along_axis(
    const quasar::Grid2D& g, int i, int j, int order,
    const double* cell_ez, quasar::mhd::BoundaryFlags4 modes, int axis,
    quasar::numerics::RadialTablesView radial_tables) {
  // The radial pass uses the active R5 width; the axial pass remains the
  // Cartesian order-selected stencil. Match that mixed tensor footprint here,
  // including cylindrical MUSCL's four radial taps but two axial taps.
  const bool weighted =
      radial_tables.active != 0 && radial_tables.scheme_order == order &&
      radial_tables.contains(i);
  const int radial_lo = weighted ? -radial_tables.collocation_width / 2
                                 : order >= 7 ? -4 : order >= 5 ? -3 : -1;
  const int radial_hi = weighted
      ? radial_lo + radial_tables.collocation_width - 1
      : order >= 7 ? 3 : order >= 5 ? 2 : 0;
  const int axial_lo = order >= 7 ? -4 : order >= 5 ? -3 : -1;
  const int axial_hi = order >= 7 ? 3 : order >= 5 ? 2 : 0;
  const auto value = [&](int ii, int jj) {
    const int mapped_i = face_index(ii, g.nx, modes.side[0], modes.side[1]);
    const int mapped_j = face_index(jj, g.ny, modes.side[2], modes.side[3]);
    return cell_ez[g.index(mapped_i, mapped_j)];
  };
  const int varying_lo = axis == 0 ? axial_lo : radial_lo;
  const int varying_hi = axis == 0 ? axial_hi : radial_hi;
  const int invariant_lo = axis == 0 ? radial_lo : axial_lo;
  const int invariant_hi = axis == 0 ? radial_hi : axial_hi;
  for (int varying = varying_lo; varying <= varying_hi; ++varying) {
    const Real reference = axis == 0
        ? value(i + invariant_lo, j + varying)
        : value(i + varying, j + invariant_lo);
    for (int invariant = invariant_lo + 1;
         invariant <= invariant_hi; ++invariant) {
      const Real candidate = axis == 0
          ? value(i + invariant, j + varying)
          : value(i + varying, j + invariant);
      if (candidate != reference) return false;
    }
  }
  return true;
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 ct_emf_kernel(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 modes, int scheme_order,
    int cylindrical, quasar::numerics::RadialTablesView radial_tables,
    const double* __restrict__ cell_ez,
    const double* __restrict__ xface_table,
    const double* __restrict__ yface_table,
    const int* __restrict__ xface_no_jump,
    const int* __restrict__ yface_no_jump,
    double* __restrict__ ez_edge, double* __restrict__ ex_edge,
    double* __restrict__ ey_edge) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  // Compute corner EMFs over the interior PLUS the high-edge corner layer
  // (i==nx / j==ny). The face-B curl-rate stencil at the last interior cell
  // (i==nx-1) reads the corner at index nx (emf_curl_rate: ez(i+1,j)); on a
  // periodic grid that corner is the wrap of the low edge, on a non-periodic
  // grid it is the one-sided x_hi closure. Computing only [0,nx) would leave
  // corner nx holding stale buffer data, which is what made div(B) grow after a
  // step even though the seed was divergence-free. Corner index nx/ny is an
  // in-bounds ghost slot (nghost>=2) and its 2x2 cell stencil stays in storage.
  if (i > g.nx || j > g.ny) return;
  const std::size_t k = g.index(i, j);

  const bool conducting =
      (i == 0 && (modes.side[0] == 2 || modes.side[0] == 3)) ||
      (i == g.nx && modes.side[1] == 2) ||
      (j == 0 && modes.side[2] == 2) ||
      (j == g.ny && modes.side[3] == 2);
  const bool coordinate_axis = cylindrical != 0 && g.origin_x == Real{0} && i == 0;
  if (conducting || coordinate_axis) {
    ez_edge[k] = Real{0};
  } else {
    const Real ex = interp_x_to_corner(
        g, i, j, scheme_order, xface_table, modes);
    const Real ey = interp_y_to_corner(
        g, i, j, scheme_order, yface_table, modes, radial_tables);
    const Real centered = cell_emf_to_corner(
        g, i, j, scheme_order, cell_ez, modes, radial_tables);
    // Upwind CT: the cell-centred transport is present once in each directional
    // Godunov EMF.  Remove one copy while retaining BOTH directional dissipation
    // terms.  In x-only data ey==centered, hence this reduces exactly to ex; the
    // y-only identity is its rotation.  The exponent-scaled three-term reduction
    // performs that cancellation before rounding, rather than first forming a
    // potentially dominant ex+ey partial sum.
    const bool x_only =
        directional_corner_stencil_has_no_jump(
            g, 1, i, j, scheme_order, yface_no_jump, modes, radial_tables) &&
        cell_emf_is_constant_along_axis(
            g, i, j, scheme_order, cell_ez, modes, /*axis=*/1, radial_tables);
    const bool y_only =
        directional_corner_stencil_has_no_jump(
            g, 0, i, j, scheme_order, xface_no_jump, modes, radial_tables) &&
        cell_emf_is_constant_along_axis(
            g, i, j, scheme_order, cell_ez, modes, /*axis=*/0, radial_tables);
    if (x_only && !y_only) {
      ez_edge[k] = ex;
    } else if (y_only && !x_only) {
      ez_edge[k] = ey;
    } else {
      ez_edge[k] = quasar::numerics::product_sum3(
          ex, Real{1}, ey, Real{1}, centered, Real{-1});
    }
  }
  // In-plane EMF components are zero for the 2D m=0 induction update (no z
  // gradient); populated for parity with the EmfField2D layout.
  ex_edge[k] = Real{0};
  ey_edge[k] = Real{0};
}

// CT EMF curl RATE (dB/dt; rk_stage applies dt later) written into the
// residual's face-B slots. Assigned (=) rather than accumulated, so it
// OVERWRITES the face-B contribution flux_difference left in dudt -- advancing
// face B by both the flux divergence and the CT curl is a double count. ONLY
// dudt.bx_face / dudt.by_face are touched; every other dudt component is left
// exactly as flux_difference wrote it.
//
// The face extents are deliberately asymmetric: Bx covers the physical high-x
// face (i == nx) over cell rows only, By covers j == ny over cell columns only.
// Those high-face rates are needed when cell-collocating dB/dt for split energy,
// and dropping them breaks the telescoping div(B) cancellation in the last cell
// row/column (pinned by MhdCtScheme.EmfCurlRateCoversHighFacesAndPreserves-
// Divergence and its cylindrical sibling).
__global__ void QUASAR_LAUNCH_BOUNDS_256 emf_curl_rate_kernel(
    quasar::Grid2D g, int cylindrical,
    double* __restrict__ dbx_face, double* __restrict__ dby_face,
    const double* __restrict__ ez_edge) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > g.nx || j > g.ny) return;
  const std::size_t k = g.index(i, j);
  const Real one2[2] = {Real{1}, Real{1}};
  // Bx lives on x faces: include the physical high-x face (i==nx), but only
  // y cell rows. By lives on y faces: include j==ny, but only x cell columns.
  // The high-face rates are needed when cell-collocating db/dt for split energy
  // and make the face extents explicit instead of leaving stale ghost values.
  if (i <= g.nx && j < g.ny) {
    const Real a[2] = {-ez_edge[g.index(i, j + 1)], ez_edge[k]};
    const Real d0[2] = {g.dy(), g.dy()};
    dbx_face[k] = quasar::numerics::scaled_product_quotient_sum(
        a, one2, d0, one2, 2);
  }
  if (i < g.nx && j <= g.ny) {
    if (cylindrical != 0) {
      const Real dr = g.dx();
      const Real r_center = g.r_at_cell_center(i);
      const Real ez_lo = ez_edge[k];
      const Real ez_hi = ez_edge[g.index(i + 1, j)];
      const Real a[4] = {ez_hi, -ez_lo, ez_hi, ez_lo};
      const Real b[4] = {Real{1}, Real{1}, Real{1}, Real{1}};
      const Real d0[4] = {dr, dr, r_center, r_center};
      const Real d1[4] = {Real{1}, Real{1}, Real{2}, Real{2}};
      dby_face[k] = quasar::numerics::scaled_product_quotient_sum(
          a, b, d0, d1, 4);
    } else {
      const Real a[2] = {ez_edge[g.index(i + 1, j)], -ez_edge[k]};
      const Real d0[2] = {g.dx(), g.dx()};
      dby_face[k] = quasar::numerics::scaled_product_quotient_sum(
          a, one2, d0, one2, 2);
    }
  }
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_ct_emf_prepare(
    const MhdField2D<Real>& u, const MhdBackgroundField<Real>& b0,
    const quasar::numerics::MhdInterfaceStates<Real>& ifx,
    const quasar::numerics::MhdInterfaceStates<Real>& ify,
    BoundaryFlags4 flags, EmfField2D<Real>& emf, Real gamma,
    stream_t stream, int scheme_order, bool hll_only,
    quasar::numerics::RadialTablesView radial_tables) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid((static_cast<unsigned>(g.nx) + 1 + block.x - 1) / block.x,
                  (static_cast<unsigned>(g.ny) + 1 + block.y - 1) / block.y);
  // When the background is inactive pass null device pointers; the kernel takes
  // the zero-background fast path and never dereferences them.
  const FaceInterfaceView xv{
      ifx.Lrho.device_ptr(), ifx.Lmx.device_ptr(), ifx.Lmy.device_ptr(),
      ifx.Lmz.device_ptr(), ifx.Lenergy.device_ptr(), ifx.Lbx.device_ptr(),
      ifx.Lby.device_ptr(), ifx.Lbz.device_ptr(), ifx.Rrho.device_ptr(),
      ifx.Rmx.device_ptr(), ifx.Rmy.device_ptr(), ifx.Rmz.device_ptr(),
      ifx.Renergy.device_ptr(), ifx.Rbx.device_ptr(), ifx.Rby.device_ptr(),
      ifx.Rbz.device_ptr()};
  const FaceInterfaceView yv{
      ify.Lrho.device_ptr(), ify.Lmx.device_ptr(), ify.Lmy.device_ptr(),
      ify.Lmz.device_ptr(), ify.Lenergy.device_ptr(), ify.Lbx.device_ptr(),
      ify.Lby.device_ptr(), ify.Lbz.device_ptr(), ify.Rrho.device_ptr(),
      ify.Rmx.device_ptr(), ify.Rmy.device_ptr(), ify.Rmz.device_ptr(),
      ify.Renergy.device_ptr(), ify.Rbx.device_ptr(), ify.Rby.device_ptr(),
      ify.Rbz.device_ptr()};
  const FaceBackgroundView bg{
      b0.active ? 1 : 0,
      b0.active ? b0.b0x_face.device_ptr() : nullptr,
      b0.active ? b0.b0y_face.device_ptr() : nullptr,
      b0.active ? b0.b0z_cell.device_ptr() : nullptr};
  const CellStateView state{
      u.rho.device_ptr(), u.mx.device_ptr(), u.my.device_ptr(), u.mz.device_ptr(),
      u.energy.device_ptr(), u.bx_face.device_ptr(), u.by_face.device_ptr(),
      u.bz_cell.device_ptr()};
  // Compute only canonical cell values plus true physical high closures.
  // Distributed internal/periodic guards are installed by the coordinator
  // before launch_mhd_ct_emf_finish; the serial finish maps them locally.
  hipLaunchKernelGGL(cell_emf_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream), g, scheme_order,
                     radial_tables, gamma,
                     state, bg, flags, emf.cell_ez_average.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  // Evaluate each canonical directional face EMF and its exact equality proof
  // once.  The corner kernel consumes the exchanged tables rather than
  // re-solving faces or recursively extending the state stencil.
  hipLaunchKernelGGL(face_emf_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream), g, scheme_order,
                     radial_tables, hll_only ? 1 : 0, gamma, flags, xv, yv, bg,
                     emf.xface_ez.device_ptr(), emf.yface_ez.device_ptr(),
                     emf.xface_no_jump.device_ptr(),
                     emf.yface_no_jump.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_ct_emf_finish(BoundaryFlags4 flags, EmfField2D<Real>& emf,
                              stream_t stream, int scheme_order,
                              bool cylindrical,
                              quasar::numerics::RadialTablesView radial_tables) {
  const Grid2D g = emf.grid;
  const dim3 block(kBlock2D, kBlock2D);
  // Launch over nx+1 x ny+1 so the high-edge corner EMF (index nx/ny) read by
  // the face-B curl-rate stencil at the last interior cell is produced too.
  const dim3 grid((static_cast<unsigned>(g.nx) + 1 + block.x - 1) / block.x,
                  (static_cast<unsigned>(g.ny) + 1 + block.y - 1) / block.y);
  hipLaunchKernelGGL(ct_emf_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, flags, scheme_order, cylindrical ? 1 : 0,
                     radial_tables,
                     emf.cell_ez_average.device_ptr(),
                     emf.xface_ez.device_ptr(), emf.yface_ez.device_ptr(),
                     emf.xface_no_jump.device_ptr(),
                     emf.yface_no_jump.device_ptr(),
                     emf.ez_edge.device_ptr(), emf.ex_edge.device_ptr(),
                     emf.ey_edge.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_ct_emf(const MhdField2D<Real>& u,
                       const MhdBackgroundField<Real>& b0,
                       const quasar::numerics::MhdInterfaceStates<Real>& ifx,
                       const quasar::numerics::MhdInterfaceStates<Real>& ify,
                       BoundaryFlags4 flags, EmfField2D<Real>& emf, Real gamma,
                       stream_t stream, int scheme_order, bool cylindrical,
                       bool hll_only,
                       quasar::numerics::RadialTablesView radial_tables) {
  launch_mhd_ct_emf_prepare(u, b0, ifx, ify, flags, emf, gamma, stream,
                            scheme_order, hll_only, radial_tables);
  launch_mhd_ct_emf_finish(flags, emf, stream, scheme_order, cylindrical,
                           radial_tables);
}

void launch_mhd_emf_curl_rate(const EmfField2D<Real>& emf, MhdField2D<Real>& dudt,
                              Grid2D grid, stream_t stream, bool cylindrical) {
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 launch_grid((static_cast<unsigned>(grid.nx) + 1 + block.x - 1) / block.x,
                         (static_cast<unsigned>(grid.ny) + 1 + block.y - 1) / block.y);
  hipLaunchKernelGGL(emf_curl_rate_kernel, launch_grid, block, 0,
                     static_cast<hipStream_t>(stream), grid, cylindrical ? 1 : 0,
                     dudt.bx_face.device_ptr(), dudt.by_face.device_ptr(),
                     emf.ez_edge.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
