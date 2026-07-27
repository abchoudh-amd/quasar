// Device face reconstruction for the finite-volume ideal-MHD solver.
//
// The kernel HONORS `scheme_order`:
//   * scheme_order == 1 -> piecewise-constant Godunov fallback used only by the
//     conservative positivity retry controller.
//   * scheme_order == 2 -> 2nd-order MUSCL-minmod reconstruction in PRIMITIVE
//     variables on every interface normal to `dir`. For dir=0, face (i,j) sits
//     between cells (i-1,j) and (i,j): the LEFT interface state is the
//     right-biased extrapolation from cell (i-1) and the RIGHT interface state
//     is the left-biased extrapolation from cell (i). The eight conserved
//     components are reconstructed after converting cell-centered conserved ->
//     primitive (so the minmod limiter acts on rho/v/p/B), then converted back
//     to conserved for the L/R buffers. THIS PATH IS UNCHANGED / BIT-IDENTICAL
//     to the original body.
//   * scheme_order == 5 -> MP5 CHARACTERISTIC reconstruction (Suresh-Huynh
//     1997, 5th-order monotonicity-preserving), 5-point stencil.
//   * scheme_order == 7 -> MP7 CHARACTERISTIC reconstruction (7th-order base
//     interpolation + the same MP limiting), 7-point stencil.
//
// The MP characteristic path MIRRORS the host driver (the registry classes in
// src/numerics/flux_reconstruction.cpp, now thin launchers): per interface the
// reference state is the overflow-safe arithmetic mean of the two flanking
// cell averages, an MhdEigensystem is built about it, each stencil cell's
// conserved delta from the
// reference is projected to the 7 characteristic waves, each wave is scalar-MP
// reconstructed over the stencil window (mp5_reconstruct / mp7_reconstruct), the
// reconstructed CharVec7 is mapped back to a conserved delta and added to the
// reference. The normal magnetic component (Bx for dir=0, By for dir=1) is NEVER
// passed through the projection -- it is overwritten on BOTH L and R with the
// constrained-transport interface face value: the one exact staggered sample
// shared by the two cells, under the SAME rule used by the MUSCL path.
//
// FIELD-SPLIT + ONE-SIDED BOUNDARY EXTENSIONS (B = B0 + b):
//   * The stored state / reconstructed output carry the PERTURBATION b
//     (u.bx/by/bz). The split EOS (mhd_state.hpp) keeps the energy as the
//     conserved variable
//       E'=rho*e+|m|^2/(2*rho)+0.5|b|^2,
//     so the gas pressure has NO B0 cross term or |B0|^2 baseline.
//     MP5/MP7 freezes the interface-collocated B0 across its stencil and uses
//     the corresponding total-field eigensystem. MUSCL reconstructs split
//     primitives; the Riemann kernel subsequently promotes every order to the
//     total field for wave speeds and fluxes. With B0 inactive the total/split
//     transformations reduce to the identity.
//   * `flags` (BoundaryFlags4, [x_lo,x_hi,y_lo,y_hi]) marks non-periodic sides.
//     At a non-periodic boundary cell along `dir` the order-2 limited slope drops
//     the EXTERIOR (ghost) neighbor so the reconstructed slope does not depend on
//     the ghost GRADIENT (the ghost VALUE is still read for the wall closure).
//     On periodic sides (flag 0) the two-sided MUSCL slope is used UNCHANGED, so
//     an all-zero flags set is bit-identical to the original body. The MP
//     characteristic path uses the symmetric (two-sided) stencil exactly as the
//     host MP driver does -- it does not special-case one-sided boundaries (the
//     deck sizes nghost to the scheme halo so all stencil reads stay in-bounds).

#include "quasar/core/grid.hpp"
#include "quasar/numerics/characteristic_projection.hpp"
#include "quasar/numerics/mhd_eigensystem.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/numerics/mp_limiter.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <limits>

namespace {

using quasar::Real;
using quasar::numerics::CharacteristicProjector;
using quasar::numerics::CharVec7;
using quasar::numerics::MhdEigensystem;
using quasar::numerics::MhdPrim;
using quasar::numerics::MhdState;

// File-local 2-D launch tiling (the MHD backend has no shared launch header; the
// per-cell kernels here all use the same square block). Kept tiny and internal
// so each .hip is self-contained.
inline constexpr unsigned kBlock2D = 16;

// Minmod limiter on two slopes (returns the smaller-magnitude same-sign slope,
// zero on a sign change). Device-callable, matches the host muscl_minmod logic.
__device__ inline Real minmod(Real a, Real b) {
  return quasar::numerics::minmod2(a, b);
}

__device__ inline MhdState load_state(const quasar::Grid2D& g,
                                      const double* rho, const double* mx,
                                      const double* my, const double* mz,
                                      const double* energy, const double* bx,
                                      const double* by, const double* bz,
                                      int i, int j,
                                      int collocation_order = 0) {
  return quasar::mhd::load_cell_state(g, rho, mx, my, mz, energy, bx, by, bz,
                                      i, j, collocation_order);
}

// One-sided slope selector. `drop` encodes which neighbor to discard at a
// non-periodic boundary so the limited slope never depends on a ghost GRADIENT:
//   drop =  0  -> two-sided minmod(back, fwd)   (interior / periodic, UNCHANGED)
//   drop = -1  -> drop the LOWER (exterior) neighbor -> forward slope only
//   drop = +1  -> drop the UPPER (exterior) neighbor -> backward slope only
// back = wc - wm (backward difference), fwd = wp - wc (forward difference).
__device__ inline Real limited_slope(Real back, Real fwd, int drop) {
  if (drop < 0) return fwd;   // lower neighbor exterior -> interior-biased fwd
  if (drop > 0) return back;  // upper neighbor exterior -> interior-biased back
  return minmod(back, fwd);
}

// Componentwise limited interface primitive: given the primitive at the cell
// (wc), its lower neighbor (wm) and upper neighbor (wp), return the value
// extrapolated to the +side (sign=+1, right face of the cell) or -side
// (sign=-1, left face of the cell): wc +/- 0.5 * slope, where `slope` is the
// (possibly one-sided) limited slope selected by `drop`.
__device__ inline MhdPrim muscl_face(const MhdPrim& wm, const MhdPrim& wc,
                                     const MhdPrim& wp, Real sign, int drop) {
  const Real h = Real{0.5} * sign;
  MhdPrim f;
  f.rho = wc.rho + h * limited_slope(wc.rho - wm.rho, wp.rho - wc.rho, drop);
  f.vx  = wc.vx  + h * limited_slope(wc.vx  - wm.vx,  wp.vx  - wc.vx,  drop);
  f.vy  = wc.vy  + h * limited_slope(wc.vy  - wm.vy,  wp.vy  - wc.vy,  drop);
  f.vz  = wc.vz  + h * limited_slope(wc.vz  - wm.vz,  wp.vz  - wc.vz,  drop);
  f.p   = wc.p   + h * limited_slope(wc.p   - wm.p,   wp.p   - wc.p,   drop);
  f.bx  = wc.bx  + h * limited_slope(wc.bx  - wm.bx,  wp.bx  - wc.bx,  drop);
  f.by  = wc.by  + h * limited_slope(wc.by  - wm.by,  wp.by  - wc.by,  drop);
  f.bz  = wc.bz  + h * limited_slope(wc.bz  - wm.bz,  wp.bz  - wc.bz,  drop);
  return f;
}

// Interior-biased "drop" code for the cell whose center index along `dir` is
// `c` (0..n-1 interior). `lo`/`hi` are the non-periodic flags for the low/high
// boundary on this axis. A cell on a non-periodic low boundary (c==0) drops its
// lower (ghost) neighbor; on a non-periodic high boundary (c==n-1) it drops its
// upper (ghost) neighbor. Everywhere else (and on periodic sides) the slope is
// two-sided. Interior cells (0 < c < n-1) are never one-sided.
__device__ inline int slope_drop(int c, int n, int lo, int hi) {
  if (lo != 0 && c <= 0) return -1;
  if (hi != 0 && c >= n - 1) return +1;
  return 0;
}

// Conserved normal-B accessor/mutator (the CT face quantity), matching the host
// normal_b / set_normal_b helpers.
__device__ inline void set_normal_b(MhdState& s, int dir, Real v) {
  Real& bn = (dir == 0) ? s.bx : s.by;
  // The stored split E' contains |b|^2/2. Replacing the reconstructed
  // normal component by the CT face must preserve gas pressure.  Accumulate
  // E + v^2/2 - bn^2/2 in one exponent-scaled reduction: either square may
  // overflow before an exact cancellation even when the final energy is
  // finite.  It also preserves E exactly when the old and new face values are
  // identical, which is common on constant states.
  s.energy = quasar::numerics::product_sum3(
      s.energy, Real{1}, v, Real{0.5} * v, bn, -Real{0.5} * bn);
  bn = v;
}

// Continuous CT interface normal-B is the exact staggered face shared by both
// Riemann states.
__device__ inline Real interface_normal_b(const quasar::Grid2D& g, int dir,
                                          int i, int j, const double* bx_face,
                                          const double* by_face) {
  return quasar::mhd::normal_face_b(g, dir, bx_face, by_face, i, j);
}

// Use the same canonical x-normal frame as the HLLD kernel.  An x/y swap is
// its own inverse, so reconstruction followed by the Riemann solve is exactly
// covariant under the solver's dir=1 rotation instead of choosing a different
// cyclic ordering for the two transverse characteristic fields.
__device__ inline MhdState rotate_xy(const MhdState& value, int dir) {
  if (dir == 0) return value;
  MhdState rotated = value;
  rotated.mx = value.my;
  rotated.my = value.mx;
  rotated.bx = value.by;
  rotated.by = value.bx;
  return rotated;
}

__device__ inline quasar::numerics::MhdBackground rotate_xy(
    const quasar::numerics::MhdBackground& value, int dir) {
  if (dir == 0) return value;
  return quasar::numerics::MhdBackground{
      value.b0y, value.b0x, value.b0z};
}

// Maximum supported MP stencil half-width (MP7 reads 3 each side).
inline constexpr int kMaxHalf = 3;
inline constexpr int kMaxWidth = 2 * kMaxHalf + 1;  // 7

// Reconstruct ONE interface conserved state in characteristic variables, the
// device mirror of the host CharacteristicMpRecon::recon_state. The stencil is
// centered on cell (i,j) shifted by `center_off` along `dir`; if `mirror` is
// true the stencil order is reversed so the right-face routine yields the
// left-face value. `half` is the stencil half-width (2 for MP5, 3 for MP7).
// from_char zeroes the normal-B DOF; the caller overwrites it with the CT face
// value. Reads inputs ONLY through the raw device pointers.
// Union of the two per-face stencils. The L state centres on cell (i-1) and the
// R state on cell (i), so together they span offsets [-1-half, half]: one more
// cell than either alone, and 2*half of the cells are common to both.
inline constexpr int kMaxUnionWidth = kMaxWidth + 1;  // 8

// Project every cell of the union stencil once. Both interface states share the
// same reference and eigensystem, so a cell's characteristic amplitudes are the
// same bits whichever state consumes them -- projecting per state re-derived
// 2*half of them (6 of 14 loads for MP7). Each entry also carries a load_state
// call, whose cell_bx/cell_by are themselves multi-tap reductions.
//
// chars[u] holds the projection of stencil offset (u + union_lo), where
// union_lo = -1 - half.
__device__ inline void project_union_stencil(
    const quasar::Grid2D& g, int dir, int i, int j, int half,
    const MhdEigensystem& eig, const MhdState& ref,
    const double* rho, const double* mx, const double* my, const double* mz,
    const double* energy, const double* bx, const double* by, const double* bz,
    CharVec7* chars) {
  const int di = (dir == 0) ? 1 : 0;
  const int dj = (dir == 0) ? 0 : 1;
  const int union_lo = -1 - half;
  const int union_width = 2 * half + 2;
  for (int u = 0; u < union_width; ++u) {
    const int off = union_lo + u;
    // The eigensystem is already transformed to the stored split variables.
    // Projecting U'-U'_ref directly avoids ever constructing the enormous total
    // energy E'+B0.b+|B0|^2/2. B0 is frozen at this interface, so prescribed
    // background gradients are not reconstructed as evolved waves.
    MhdState delta = rotate_xy(
        load_state(g, rho, mx, my, mz, energy, bx, by, bz,
                   i + off * di, j + off * dj), dir);
    delta.rho    -= ref.rho;
    delta.mx     -= ref.mx;
    delta.my     -= ref.my;
    delta.mz     -= ref.mz;
    delta.energy -= ref.energy;
    delta.bx     -= ref.bx;
    delta.by     -= ref.by;
    delta.bz     -= ref.bz;
    chars[u] = CharacteristicProjector::to_char(delta, eig);
  }
}

__device__ inline MhdState recon_char_state(
    int dir, int center_off, bool mirror, int half,
    const MhdEigensystem& eig, const MhdState& ref, const CharVec7* chars) {
  const int width = 2 * half + 1;
  // This state's stencil cell s sits at offset center_off + (s - half); the
  // shared table is indexed from union_lo = -1 - half.
  const int base = center_off + 1;

  // Reconstruct each of the 7 characteristic fields at the interface.
  CharVec7 w_face{};
  Real scalar[kMaxWidth];
  for (int k = 0; k < 7; ++k) {
    for (int s = 0; s < width; ++s) {
      const int src = mirror ? (width - 1 - s) : s;
      scalar[s] = chars[base + src].w[k];
    }
    if (half == 2) {
      w_face.w[k] = quasar::numerics::mp5_reconstruct(
          scalar[0], scalar[1], scalar[2], scalar[3], scalar[4]);
    } else {
      w_face.w[k] = quasar::numerics::mp7_reconstruct(
          scalar[0], scalar[1], scalar[2], scalar[3], scalar[4], scalar[5],
          scalar[6]);
    }
  }

  // Map the reconstructed characteristic delta back to a conserved delta and add
  // the reference. from_char zeroes the normal-B component (caller overwrites).
  const MhdState d = CharacteristicProjector::from_char(w_face, eig);
  MhdState s;
  s.rho    = ref.rho + d.rho;
  s.mx     = ref.mx + d.mx;
  s.my     = ref.my + d.my;
  s.mz     = ref.mz + d.mz;
  s.energy = ref.energy + d.energy;
  s.bx     = ref.bx + d.bx;
  s.by     = ref.by + d.by;
  s.bz     = ref.bz + d.bz;
  return rotate_xy(s, dir);
}

// 2nd-order MUSCL-minmod reconstruction of the LEFT and RIGHT interface states
// at face (i,j) (between the "minus" cell (i-di,j-dj) and the "plus" cell
// (i,j)). This is the EXACT order-2 path, factored into a helper so it serves
// both the order-2 branch AND the per-interface robustness fallback that a
// troubled MP5/MP7 face drops to. Normal-B is left as the per-cell stored value
// here; the caller overwrites it with the CT face value, identically for all
// orders. The math (load_state / to_primitive / slope_drop / muscl_face /
// to_conserved) is bit-identical to the original order-2 body.
__device__ inline void muscl_reconstruct_face(
    const quasar::Grid2D& g, int dir, int i, int j, Real gamma,
    int side_lo, int side_hi,
    const double* rho, const double* mx, const double* my, const double* mz,
    const double* energy, const double* bx, const double* by, const double* bz,
    MhdState& ul, MhdState& ur) {
  const int di = (dir == 0) ? 1 : 0;
  const int dj = (dir == 0) ? 0 : 1;
  const int im = i - di, jm = j - dj;       // minus cell (owns the L state)
  const int imm = i - 2 * di, jmm = j - 2 * dj;
  const int ip = i + di, jp = j + dj;       // plus-cell upper neighbor

  // Minus-cell stencil: (imm) (im) (i)  -> face value at +side of (im).
  const MhdState sm_lo = load_state(g, rho, mx, my, mz, energy, bx, by, bz,
                                    imm, jmm);
  const MhdState sm_c  = load_state(g, rho, mx, my, mz, energy, bx, by, bz,
                                    im, jm);
  const MhdState sm_hi = load_state(g, rho, mx, my, mz, energy, bx, by, bz,
                                    i, j);
  // Plus-cell stencil: (im) (i) (ip) -> face value at -side of (i).
  const MhdState sp_hi = load_state(g, rho, mx, my, mz, energy, bx, by, bz,
                                    ip, jp);

  const MhdPrim wm_lo = to_primitive(sm_lo, gamma);
  const MhdPrim wm_c  = to_primitive(sm_c, gamma);
  const MhdPrim wm_hi = to_primitive(sm_hi, gamma);
  const MhdPrim wp_hi = to_primitive(sp_hi, gamma);

  const int n_dir     = (dir == 0) ? g.nx : g.ny;
  const int minus_idx = (dir == 0) ? im : jm;
  const int plus_idx  = (dir == 0) ? i  : j;
  const int drop_m = slope_drop(minus_idx, n_dir, side_lo, side_hi);
  const int drop_p = slope_drop(plus_idx,  n_dir, side_lo, side_hi);

  const MhdPrim wl = muscl_face(wm_lo, wm_c, wm_hi, Real{+1}, drop_m);  // +side of minus
  const MhdPrim wr = muscl_face(wm_c, wm_hi, wp_hi, Real{-1}, drop_p);  // -side of plus

  ul = to_conserved(wl, gamma);
  ur = to_conserved(wr, gamma);
}

// Piecewise-constant states from the two cells adjacent to face (i,j).  A
// configured order-2/5/7 operator uses its ordinary high-order magnetic
// collocation (order 0): the solver preflight has already proved those live
// cells admissible.  The global low-order positivity retry passes order 1 instead,
// matching its bounded two-adjacent-face EOS everywhere else.
__device__ inline void piecewise_constant_reconstruct_face(
    const quasar::Grid2D& g, int dir, int i, int j, int collocation_order,
    const double* rho, const double* mx, const double* my, const double* mz,
    const double* energy, const double* bx, const double* by, const double* bz,
    MhdState& ul, MhdState& ur) {
  const int di = (dir == 0) ? 1 : 0;
  const int dj = (dir == 0) ? 0 : 1;
  ul = load_state(g, rho, mx, my, mz, energy, bx, by, bz,
                  i - di, j - dj, collocation_order);
  ur = load_state(g, rho, mx, my, mz, energy, bx, by, bz,
                  i, j, collocation_order);
}

// Per-interface admissibility test for a reconstructed conserved state: density
// and gas pressure must both be FINITE and strictly POSITIVE. Pressure is taken
// from to_primitive (the field-split EOS: p = (gamma-1)*(E - 0.5 rho|v|^2 -
// 0.5|b|^2)), so the test is consistent with the kernel's EOS and with the wave
// speeds the HLLD solver derives. A NaN fails every comparison (NaN > 0 is
// false), so this also rejects non-finite reconstructions before they can reach
// the Riemann solver or conservative stage-admissibility controller. isfinite() guards
// against +/-inf, whose sign would otherwise pass the > 0 test.
__device__ inline bool reconstructed_state_admissible(const MhdState& s, Real gamma) {
  if (!(isfinite(s.rho) && s.rho > Real{0})) return false;
  const MhdPrim w = to_primitive(s, gamma);
  if (!(isfinite(w.vx) && isfinite(w.vy) && isfinite(w.vz)
        && isfinite(w.p) && w.p > Real{0})) {
    return false;
  }
  // Reject any other non-finite component (momentum / B): a finite rho & p with
  // a NaN velocity or B still poisons the Riemann flux.
  return isfinite(s.mx) && isfinite(s.my) && isfinite(s.mz)
      && isfinite(s.energy) && isfinite(s.bx) && isfinite(s.by) && isfinite(s.bz);
}

// Rate-only reconstruction has the same physical-face domain as the Riemann
// solve. Mesh weighting is range-safe, but it is applied only after the HLLD/LF
// fan itself has been formed: an unrepresentable m/rho or other primitive is
// therefore a genuine face-fan failure, not a metric overflow that the scaled
// CFL reduction may rescue. Keep this wrapper to make that boundary explicit at
// the rate-only call site.
__device__ inline bool reconstructed_rate_state_admissible(
    const MhdState& s, Real gamma) {
  return reconstructed_state_admissible(s, gamma);
}

// A fixed quiet-NaN state is an explicit reconstruction failure token.  The
// Riemann/update path propagates it into the candidate, and the conservative
// admissibility controller deterministically rejects and rolls that candidate
// back.  Returning an arbitrary finite floor here would silently inject mass or
// energy and would defeat the solver's retry/error contract.
__device__ inline MhdState invalid_reconstruction_sentinel() {
  const Real invalid = std::numeric_limits<Real>::quiet_NaN();
  return MhdState{invalid, invalid, invalid, invalid,
                  invalid, invalid, invalid, invalid};
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 reconstruct_kernel(
    quasar::Grid2D g, int dir, int scheme_order, Real gamma, int rate_only,
    // Non-periodic boundary flags along this axis: lo/hi correspond to
    // (x_lo,x_hi) for dir==0 and (y_lo,y_hi) for dir==1.
    int side_lo, int side_hi,
    const double* __restrict__ rho, const double* __restrict__ mx,
    const double* __restrict__ my, const double* __restrict__ mz,
    const double* __restrict__ energy, const double* __restrict__ bx,
    const double* __restrict__ by, const double* __restrict__ bz,
    int b0_active, const double* __restrict__ b0x,
    const double* __restrict__ b0y, const double* __restrict__ b0z,
    double* __restrict__ Lrho, double* __restrict__ Lmx, double* __restrict__ Lmy,
    double* __restrict__ Lmz, double* __restrict__ Lenergy, double* __restrict__ Lbx,
    double* __restrict__ Lby, double* __restrict__ Lbz,
    double* __restrict__ Rrho, double* __restrict__ Rmx, double* __restrict__ Rmy,
    double* __restrict__ Rmz, double* __restrict__ Renergy, double* __restrict__ Rbx,
    double* __restrict__ Rby, double* __restrict__ Rbz) {
  const int launch_i = blockIdx.x * blockDim.x + threadIdx.x;
  const int launch_j = blockIdx.y * blockDim.y + threadIdx.y;
  // The normal axis spans every physical face, including its high layer.  The
  // transverse axis spans the complete padded cell range: MP5/MP7 face
  // quadrature reconstructs point states from neighbouring face averages, so a
  // boundary-adjacent physical face legitimately consumes reconstructed ghost
  // rows/columns.  Those ghosts have already been filled by the configured
  // boundary closure.  Normal reconstruction never extends beyond the existing
  // halo: at face 0 / n, MP7's outermost source cell is exactly -nghost /
  // n+nghost-1.
  const int i = dir == 1 ? launch_i - g.nghost : launch_i;
  const int j = dir == 0 ? launch_j - g.nghost : launch_j;
  if (dir == 0) {
    if (launch_i > g.nx || launch_j >= g.ny + 2 * g.nghost) return;
  } else {
    if (launch_i >= g.nx + 2 * g.nghost || launch_j > g.ny) return;
  }

  // Neighbor offsets along the reconstruction direction.
  const int di = (dir == 0) ? 1 : 0;
  const int dj = (dir == 0) ? 0 : 1;

  MhdState ul, ur;

  quasar::numerics::MhdBackground bg{};
  if (b0_active != 0) {
    bg = quasar::mhd::load_interface_background(
        g, dir, b0x, b0y, b0z, i, j,
        scheme_order <= 1 ? 1 : 0);
  }

  if (scheme_order <= 1) {
    // Positivity retry fallback: first-order piecewise-constant states. The
    // shared CT normal field is imposed below, with the energy adjustment that
    // preserves each cell's gas pressure.
    piecewise_constant_reconstruct_face(
        g, dir, i, j, /*collocation_order=*/1,
        rho, mx, my, mz, energy, bx, by, bz, ul, ur);
  } else if (scheme_order != 2) {
    // ---- MP5 / MP7 characteristic reconstruction --------------------------
    // Reference state = arithmetic mean of the two cells flanking the face
    // (cell i-1 and cell i for dir=0). Build the eigensystem about it.
    const MhdState cL =
        load_state(g, rho, mx, my, mz, energy, bx, by, bz, i - di, j - dj);
    const MhdState cR =
        load_state(g, rho, mx, my, mz, energy, bx, by, bz, i, j);
    MhdState ref;
    ref.rho    = Real{0.5} * cL.rho    + Real{0.5} * cR.rho;
    ref.mx     = Real{0.5} * cL.mx     + Real{0.5} * cR.mx;
    ref.my     = Real{0.5} * cL.my     + Real{0.5} * cR.my;
    ref.mz     = Real{0.5} * cL.mz     + Real{0.5} * cR.mz;
    ref.energy = Real{0.5} * cL.energy + Real{0.5} * cR.energy;
    ref.bx     = Real{0.5} * cL.bx     + Real{0.5} * cR.bx;
    ref.by     = Real{0.5} * cL.by     + Real{0.5} * cR.by;
    ref.bz     = Real{0.5} * cL.bz     + Real{0.5} * cR.bz;

    const MhdState canonical_ref = rotate_xy(ref, dir);
    const quasar::numerics::MhdBackground canonical_bg = rotate_xy(bg, dir);
    MhdEigensystem eig;
    eig.build(canonical_ref, canonical_bg, /*dir=*/0, gamma);

    const int half = (scheme_order == 5) ? 2 : 3;

    // Both states read the same reference and eigensystem, so project the union
    // of their two stencils once and let each pick its window out of it.
    CharVec7 chars[kMaxUnionWidth];
    project_union_stencil(g, dir, i, j, half, eig, canonical_ref,
                          rho, mx, my, mz, energy, bx, by, bz, chars);

    // L state: stencil centered on cell (i-1), reconstruct its RIGHT face.
    ul = recon_char_state(dir, /*center_off=*/-1, /*mirror=*/false,
                          half, eig, canonical_ref, chars);
    // R state: stencil centered on cell (i), reconstruct its LEFT face (mirror).
    ur = recon_char_state(dir, /*center_off=*/0, /*mirror=*/true,
                          half, eig, canonical_ref, chars);

    // ---- Per-interface robustness fallback (finiteness / positivity) ------
    // Characteristic MP reconstruction carries NO positivity guarantee: at a
    // strong shock / near-vacuum interface the MP-limited characteristic
    // amplitudes mapped back to conserved variables can yield a NEGATIVE or
    // non-finite reconstructed density / pressure, or the eigensystem built about
    // the arithmetic-mean reference (near-coincident fast/slow speeds, near-vacuum
    // reference) can emit non-finite eigenvectors -> NaN in ul/ur. HLLD then
    // sqrt()s a negative wave speed (-> NaN). If EITHER reconstructed state is
    // inadmissible, DISCARD the high-order result for THIS face only and fall back
    // to the robust 2nd-order MUSCL-minmod reconstruction (bit-identical to the
    // order-2 path). Local, per-thread drop: smooth interfaces keep full MP5/MP7
    // order (admissible -> no fallback); only troubled interfaces drop to 2nd
    // order. This keeps strong-shock runs FINITE and POSITIVE without degrading
    // smooth convergence, and is the last-resort guard for any non-finite/negative
    // state the Suresh-Huynh primitive pass above does not fully repair.
    if (!reconstructed_state_admissible(ul, gamma)
        || !reconstructed_state_admissible(ur, gamma)) {
      muscl_reconstruct_face(g, dir, i, j, gamma, side_lo, side_hi,
                             rho, mx, my, mz, energy, bx, by, bz, ul, ur);
    }
  } else {
    // ---- 2nd-order MUSCL-minmod (UNCHANGED, bit-identical) -----------------
    // Face (i,j) lies between the "minus" cell (i-di, j-dj) and the "plus" cell
    // (i,j). The LEFT state extrapolates from the minus cell to its +side; the
    // RIGHT state extrapolates from the plus cell to its -side.
    muscl_reconstruct_face(g, dir, i, j, gamma, side_lo, side_hi,
                           rho, mx, my, mz, energy, bx, by, bz, ul, ur);
  }

  // Normal-B: continuous CT face value on both sides (same rule for all orders).
  const Real bn = interface_normal_b(g, dir, i, j, bx, by);
  set_normal_b(ul, dir, bn);
  set_normal_b(ur, dir, bn);
  // MUSCL is not an invariant-domain reconstruction.  Componentwise primitive
  // slopes can combine individually admissible extreme cells into a face whose
  // kinetic energy overflows; a one-sided boundary extrapolation can also leave
  // the positive-density/pressure set.  Therefore EVERY configured order gets
  // one final local Godunov fallback after CT normalization.  Keep its magnetic
  // collocation consistent with the operator whose live state was preflighted:
  // configured order 2/5/7 -> 0, global low-order retry -> 1.
  if (!reconstructed_state_admissible(ul, gamma)
      || !reconstructed_state_admissible(ur, gamma)) {
    const int fallback_collocation = (scheme_order <= 1) ? 1 : 0;
    piecewise_constant_reconstruct_face(
        g, dir, i, j, fallback_collocation,
        rho, mx, my, mz, energy, bx, by, bz, ul, ur);
    set_normal_b(ul, dir, bn);
    set_normal_b(ur, dir, bn);
  }

  // The exact CT face can itself make a representable cell state impossible to
  // express at this interface (for example, |B_n|^2/2 may overflow).  Do not
  // leak a partially finite state into HLLD: emit one unambiguous token that the
  // conservative solver rejects rather than inventing a thermodynamic floor.
  const bool final_left_admissible = rate_only != 0
      ? reconstructed_rate_state_admissible(ul, gamma)
      : reconstructed_state_admissible(ul, gamma);
  const bool final_right_admissible = rate_only != 0
      ? reconstructed_rate_state_admissible(ur, gamma)
      : reconstructed_state_admissible(ur, gamma);
  if (!final_left_admissible || !final_right_admissible) {
    ul = invalid_reconstruction_sentinel();
    ur = invalid_reconstruction_sentinel();
  }

  const std::size_t k = g.index(i, j);
  Lrho[k] = ul.rho; Lmx[k] = ul.mx; Lmy[k] = ul.my; Lmz[k] = ul.mz;
  Lenergy[k] = ul.energy; Lbx[k] = ul.bx; Lby[k] = ul.by; Lbz[k] = ul.bz;
  Rrho[k] = ur.rho; Rmx[k] = ur.mx; Rmy[k] = ur.my; Rmz[k] = ur.mz;
  Renergy[k] = ur.energy; Rbx[k] = ur.bx; Rby[k] = ur.by; Rbz[k] = ur.bz;
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_reconstruct(const MhdField2D<Real>& u, const MhdBackgroundField<Real>& b0,
                            int dir, quasar::numerics::MhdInterfaceStates<Real>& out,
                            int scheme_order, BoundaryFlags4 flags, Real gamma,
                            stream_t stream, bool rate_only) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  // Include the complete transverse ghost range needed by the pointwise face
  // quadrature.  Along the normal axis only the physical face range [0,n] is
  // required.
  const unsigned launch_nx = static_cast<unsigned>(
      dir == 0 ? g.nx + 1 : g.nx + 2 * g.nghost);
  const unsigned launch_ny = static_cast<unsigned>(
      dir == 0 ? g.ny + 2 * g.nghost : g.ny + 1);
  const dim3 grid((launch_nx + block.x - 1) / block.x,
                  (launch_ny + block.y - 1) / block.y);
  // Select this axis's non-periodic flags: [x_lo,x_hi] for dir==0, [y_lo,y_hi].
  // MP characteristic projection uses the interface-collocated total field
  // B0+b; MUSCL remains a componentwise reconstruction of the split primitives.
  const int side_lo = (dir == 0) ? flags.side[0] : flags.side[2];
  const int side_hi = (dir == 0) ? flags.side[1] : flags.side[3];
  const int b0_active = b0.active ? 1 : 0;
  const double* b0x = b0.active ? b0.b0x_face.device_ptr() : nullptr;
  const double* b0y = b0.active ? b0.b0y_face.device_ptr() : nullptr;
  const double* b0z = b0.active ? b0.b0z_cell.device_ptr() : nullptr;
  hipLaunchKernelGGL(reconstruct_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream), g, dir, scheme_order, gamma,
                     rate_only ? 1 : 0, side_lo, side_hi,
                     u.rho.device_ptr(), u.mx.device_ptr(), u.my.device_ptr(),
                     u.mz.device_ptr(), u.energy.device_ptr(), u.bx_face.device_ptr(),
                     u.by_face.device_ptr(), u.bz_cell.device_ptr(),
                     b0_active, b0x, b0y, b0z,
                     out.Lrho.device_ptr(), out.Lmx.device_ptr(), out.Lmy.device_ptr(),
                     out.Lmz.device_ptr(), out.Lenergy.device_ptr(), out.Lbx.device_ptr(),
                     out.Lby.device_ptr(), out.Lbz.device_ptr(),
                     out.Rrho.device_ptr(), out.Rmx.device_ptr(), out.Rmy.device_ptr(),
                     out.Rmz.device_ptr(), out.Renergy.device_ptr(), out.Rbx.device_ptr(),
                     out.Rby.device_ptr(), out.Rbz.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
