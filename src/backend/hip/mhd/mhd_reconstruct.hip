// Device flux reconstruction for the conservative-FD ideal-MHD solver.
//
// The kernel HONORS `scheme_order`:
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
// reference state is the arithmetic mean of the two flanking cells, an
// MhdEigensystem is built about it, each stencil cell's conserved delta from the
// reference is projected to the 7 characteristic waves, each wave is scalar-MP
// reconstructed over the stencil window (mp5_reconstruct / mp7_reconstruct), the
// reconstructed CharVec7 is mapped back to a conserved delta and added to the
// reference. The normal magnetic component (Bx for dir=0, By for dir=1) is NEVER
// passed through the projection -- it is overwritten on BOTH L and R with the
// constrained-transport interface face value (average of the two adjacent
// staggered face samples), the SAME rule the MUSCL path uses.
//
// FIELD-SPLIT + ONE-SIDED BOUNDARY EXTENSIONS (B = B0 + b):
//   * The stored state / reconstructed output carry the PERTURBATION b
//     (u.bx/by/bz). The split EOS (mhd_state.hpp) keeps the energy as the
//     perturbation-only 0.5|b|^2, so the gas pressure has NO B0 cross term --
//     to_primitive()/to_conserved() on the perturbation field remain an exact
//     inverse pair. This kernel computes no sound speed / B0-dependent pressure,
//     so the B0 buffers are threaded through the ABI but are mathematically inert
//     here; the wave speeds that DO see the total field live in the Riemann
//     kernel. With b0 inactive this is bit-identical to the original body.
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

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

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
inline dim3 mhd_grid_2d(const quasar::Grid2D& g, dim3 block) {
  return dim3((static_cast<unsigned>(g.nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(g.ny) + block.y - 1) / block.y);
}

// Minmod limiter on two slopes (returns the smaller-magnitude same-sign slope,
// zero on a sign change). Device-callable, matches the host muscl_minmod logic.
__device__ inline Real minmod(Real a, Real b) {
  if (a * b <= Real{0}) return Real{0};
  return (fabs(a) < fabs(b)) ? a : b;
}

// Load the cell-centered conserved state at storage index k into a MhdState.
__device__ inline MhdState load_state(const double* rho, const double* mx,
                                      const double* my, const double* mz,
                                      const double* energy, const double* bx,
                                      const double* by, const double* bz,
                                      std::size_t k) {
  MhdState s;
  s.rho = rho[k]; s.mx = mx[k]; s.my = my[k]; s.mz = mz[k];
  s.energy = energy[k]; s.bx = bx[k]; s.by = by[k]; s.bz = bz[k];
  return s;
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
  if (dir == 0) s.bx = v; else s.by = v;
}

// Continuous CT interface normal-B = average of the two adjacent staggered face
// samples (bx_face for dir=0, by_face for dir=1). Mirrors the host
// interface_normal_b: 0.5*(face[k(i,j)] + face[k(i-1,j)]) for dir=0.
__device__ inline Real interface_normal_b(const quasar::Grid2D& g, int dir,
                                          int i, int j, const double* bx_face,
                                          const double* by_face) {
  const std::size_t k0 = g.index(i, j);
  const std::size_t km = (dir == 0) ? g.index(i - 1, j) : g.index(i, j - 1);
  const double* face = (dir == 0) ? bx_face : by_face;
  return Real{0.5} * (static_cast<Real>(face[k0]) + static_cast<Real>(face[km]));
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
__device__ inline MhdState recon_char_state(
    const quasar::Grid2D& g, int dir, int i, int j, int center_off, bool mirror,
    int half, const MhdEigensystem& eig, const MhdState& ref,
    const double* rho, const double* mx, const double* my, const double* mz,
    const double* energy, const double* bx, const double* by, const double* bz) {
  const int width = 2 * half + 1;
  const int di = (dir == 0) ? 1 : 0;
  const int dj = (dir == 0) ? 0 : 1;

  // Project each stencil cell's delta-from-ref into the 7 characteristic waves.
  // chars[s].w[k]: s over stencil (0..width-1), k over 7 waves.
  CharVec7 chars[kMaxWidth];
  for (int s = 0; s < width; ++s) {
    const int off = center_off + (s - half);
    const std::size_t k = g.index(i + off * di, j + off * dj);
    MhdState delta = load_state(rho, mx, my, mz, energy, bx, by, bz, k);
    delta.rho    -= ref.rho;
    delta.mx     -= ref.mx;
    delta.my     -= ref.my;
    delta.mz     -= ref.mz;
    delta.energy -= ref.energy;
    delta.bx     -= ref.bx;
    delta.by     -= ref.by;
    delta.bz     -= ref.bz;
    chars[s] = CharacteristicProjector::to_char(delta, eig);
  }

  // Reconstruct each of the 7 characteristic fields at the interface.
  CharVec7 w_face{};
  Real scalar[kMaxWidth];
  for (int k = 0; k < 7; ++k) {
    for (int s = 0; s < width; ++s) {
      const int src = mirror ? (width - 1 - s) : s;
      scalar[s] = chars[src].w[k];
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
  return s;
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
  const MhdState sm_lo = load_state(rho, mx, my, mz, energy, bx, by, bz,
                                    g.index(imm, jmm));
  const MhdState sm_c  = load_state(rho, mx, my, mz, energy, bx, by, bz,
                                    g.index(im, jm));
  const MhdState sm_hi = load_state(rho, mx, my, mz, energy, bx, by, bz,
                                    g.index(i, j));
  // Plus-cell stencil: (im) (i) (ip) -> face value at -side of (i).
  const MhdState sp_hi = load_state(rho, mx, my, mz, energy, bx, by, bz,
                                    g.index(ip, jp));

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

// Per-interface admissibility test for a reconstructed conserved state: density
// and gas pressure must both be FINITE and strictly POSITIVE. Pressure is taken
// from to_primitive (the field-split EOS: p = (gamma-1)*(E - 0.5 rho|v|^2 -
// 0.5|b|^2)), so the test is consistent with the kernel's EOS and with the wave
// speeds the HLLD solver derives. A NaN fails every comparison (NaN > 0 is
// false), so this also rejects non-finite reconstructions that the downstream
// positivity floors cannot recover (NaN < floor is false). isfinite() guards
// against +/-inf, whose sign would otherwise pass the > 0 test.
__device__ inline bool reconstructed_state_admissible(const MhdState& s, Real gamma) {
  if (!(isfinite(s.rho) && s.rho > Real{0})) return false;
  const MhdPrim w = to_primitive(s, gamma);
  if (!(isfinite(w.p) && w.p > Real{0})) return false;
  // Reject any other non-finite component (momentum / B): a finite rho & p with
  // a NaN velocity or B still poisons the Riemann flux.
  return isfinite(s.mx) && isfinite(s.my) && isfinite(s.mz)
      && isfinite(s.energy) && isfinite(s.bx) && isfinite(s.by) && isfinite(s.bz);
}

__global__ void reconstruct_kernel(
    quasar::Grid2D g, int dir, int scheme_order, Real gamma,
    // Non-periodic boundary flags along this axis: lo/hi correspond to
    // (x_lo,x_hi) for dir==0 and (y_lo,y_hi) for dir==1.
    int side_lo, int side_hi,
    const double* __restrict__ rho, const double* __restrict__ mx,
    const double* __restrict__ my, const double* __restrict__ mz,
    const double* __restrict__ energy, const double* __restrict__ bx,
    const double* __restrict__ by, const double* __restrict__ bz,
    double* __restrict__ Lrho, double* __restrict__ Lmx, double* __restrict__ Lmy,
    double* __restrict__ Lmz, double* __restrict__ Lenergy, double* __restrict__ Lbx,
    double* __restrict__ Lby, double* __restrict__ Lbz,
    double* __restrict__ Rrho, double* __restrict__ Rmx, double* __restrict__ Rmy,
    double* __restrict__ Rmz, double* __restrict__ Renergy, double* __restrict__ Rbx,
    double* __restrict__ Rby, double* __restrict__ Rbz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  // Produce interface states over the interior PLUS the high-edge face layer
  // (i==nx / j==ny). flux_difference at the last interior cell (i==nx-1) reads
  // the flux at face index nx, so the reconstruction that feeds that flux must
  // exist there too; computing only [0,nx) would leave face nx holding stale
  // buffer data and break the conservative telescoping. Face index nx/ny is an
  // in-bounds ghost slot (valid i in [-nghost, nx-1+nghost], nghost>=2), and the
  // stencil cells it reads are all within the padded storage because the deck
  // sizes nghost to the scheme's halo (2/3/4).
  if (i > g.nx || j > g.ny) return;

  // Neighbor offsets along the reconstruction direction.
  const int di = (dir == 0) ? 1 : 0;
  const int dj = (dir == 0) ? 0 : 1;

  MhdState ul, ur;

  if (scheme_order != 2) {
    // ---- MP5 / MP7 characteristic reconstruction --------------------------
    // Reference state = arithmetic mean of the two cells flanking the face
    // (cell i-1 and cell i for dir=0). Build the eigensystem about it.
    const std::size_t k_minus = g.index(i - di, j - dj);  // cell i-1
    const std::size_t k_plus  = g.index(i, j);            // cell i
    const MhdState cL = load_state(rho, mx, my, mz, energy, bx, by, bz, k_minus);
    const MhdState cR = load_state(rho, mx, my, mz, energy, bx, by, bz, k_plus);
    MhdState ref;
    ref.rho    = Real{0.5} * (cL.rho + cR.rho);
    ref.mx     = Real{0.5} * (cL.mx + cR.mx);
    ref.my     = Real{0.5} * (cL.my + cR.my);
    ref.mz     = Real{0.5} * (cL.mz + cR.mz);
    ref.energy = Real{0.5} * (cL.energy + cR.energy);
    ref.bx     = Real{0.5} * (cL.bx + cR.bx);
    ref.by     = Real{0.5} * (cL.by + cR.by);
    ref.bz     = Real{0.5} * (cL.bz + cR.bz);

    MhdEigensystem eig;
    eig.build(ref, dir, gamma);

    const int half = (scheme_order == 5) ? 2 : 3;

    // L state: stencil centered on cell (i-1), reconstruct its RIGHT face.
    ul = recon_char_state(g, dir, i, j, /*center_off=*/-1, /*mirror=*/false,
                          half, eig, ref, rho, mx, my, mz, energy, bx, by, bz);
    // R state: stencil centered on cell (i), reconstruct its LEFT face (mirror).
    ur = recon_char_state(g, dir, i, j, /*center_off=*/0, /*mirror=*/true,
                          half, eig, ref, rho, mx, my, mz, energy, bx, by, bz);


    // ---- Per-interface robustness fallback (finiteness / positivity) ------
    // Characteristic MP reconstruction carries NO positivity guarantee: at a
    // strong shock / near-vacuum interface the MP-limited characteristic
    // amplitudes mapped back to conserved variables can yield a NEGATIVE or
    // non-finite reconstructed density / pressure, or the eigensystem built about
    // the arithmetic-mean reference (near-coincident fast/slow speeds, near-vacuum
    // reference) can emit non-finite eigenvectors -> NaN in ul/ur. HLLD then
    // sqrt()s a negative wave speed (-> NaN) and the positivity floor cannot
    // recover it (NaN < floor is false). If EITHER reconstructed state is
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

  const std::size_t k = g.index(i, j);
  Lrho[k] = ul.rho; Lmx[k] = ul.mx; Lmy[k] = ul.my; Lmz[k] = ul.mz;
  Lenergy[k] = ul.energy; Lbx[k] = ul.bx; Lby[k] = ul.by; Lbz[k] = ul.bz;
  Rrho[k] = ur.rho; Rmx[k] = ur.mx; Rmy[k] = ur.my; Rmz[k] = ur.mz;
  Renergy[k] = ur.energy; Rbx[k] = ur.bx; Rby[k] = ur.by; Rbz[k] = ur.bz;
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_reconstruct(const MhdField2D<Real>& u, const MhdBackgroundField<Real>& /*b0*/,
                            int dir, quasar::numerics::MhdInterfaceStates<Real>& out,
                            int scheme_order, BoundaryFlags4 flags, Real gamma,
                            stream_t stream) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  // Launch over nx+1 x ny+1: the kernel also writes the high-edge face layer
  // (index nx/ny) that flux_difference consumes at the last interior cell.
  const dim3 grid((static_cast<unsigned>(g.nx) + 1 + block.x - 1) / block.x,
                  (static_cast<unsigned>(g.ny) + 1 + block.y - 1) / block.y);
  // Select this axis's non-periodic flags: [x_lo,x_hi] for dir==0, [y_lo,y_hi]
  // for dir==1. The split EOS makes B0 inert in this kernel (no sound speed /
  // B0-dependent pressure here), so the background buffers are accepted for ABI
  // parity but not read; the wave speeds that see the total field live in the
  // HLLD kernel.
  const int side_lo = (dir == 0) ? flags.side[0] : flags.side[2];
  const int side_hi = (dir == 0) ? flags.side[1] : flags.side[3];
  hipLaunchKernelGGL(reconstruct_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream), g, dir, scheme_order, gamma,
                     side_lo, side_hi,
                     u.rho.device_ptr(), u.mx.device_ptr(), u.my.device_ptr(),
                     u.mz.device_ptr(), u.energy.device_ptr(), u.bx_face.device_ptr(),
                     u.by_face.device_ptr(), u.bz_cell.device_ptr(),
                     out.Lrho.device_ptr(), out.Lmx.device_ptr(), out.Lmy.device_ptr(),
                     out.Lmz.device_ptr(), out.Lenergy.device_ptr(), out.Lbx.device_ptr(),
                     out.Lby.device_ptr(), out.Lbz.device_ptr(),
                     out.Rrho.device_ptr(), out.Rmx.device_ptr(), out.Rmy.device_ptr(),
                     out.Rmz.device_ptr(), out.Renergy.device_ptr(), out.Rbx.device_ptr(),
                     out.Rby.device_ptr(), out.Rbz.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
