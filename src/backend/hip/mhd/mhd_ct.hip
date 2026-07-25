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

#include <hip/hip_runtime.h>

namespace {

using quasar::Real;
using quasar::numerics::MhdFlux;
using quasar::numerics::MhdState;

inline constexpr unsigned kBlock2D = 16;

// Range-safe rational interpolation.  High-order coefficients are large
// enough that coefficient*EMF overflows for a perfectly representable constant
// near DBL_MAX; adding the signed numerator before dividing has the same
// problem.  These scalar terms implement the same largest-opposite-sign-first
// reduction as numerics::reduce_scaled_terms without runtime-indexed arrays;
// AMDGPU can keep them in registers instead of promoting per-thread arrays to
// tens of KiB of workgroup memory.
struct ScaledCtTerm {
  Real mantissa{};
  int exponent{};
};

__device__ inline ScaledCtTerm make_scaled_ct_term(
    Real coefficient, Real value, Real denominator) {
  if (value == Real{0}) return {};
  int ec = 0, ev = 0, ed = 0;
  const Real mc = frexp(coefficient, &ec);
  const Real mv = frexp(value, &ev);
  const Real md = frexp(denominator, &ed);
  int shift = 0;
  const Real mantissa = frexp((mc * mv) / md, &shift);
  return ScaledCtTerm{mantissa, ec + ev - ed + shift};
}

__device__ inline bool ct_term_larger(const ScaledCtTerm& lhs,
                                      const ScaledCtTerm& rhs) {
  return lhs.exponent > rhs.exponent ||
         (lhs.exponent == rhs.exponent &&
          fabs(lhs.mantissa) > fabs(rhs.mantissa));
}

__device__ inline ScaledCtTerm combine_scaled_ct_terms(
    const ScaledCtTerm& largest, const ScaledCtTerm& partner) {
  const Real sum = largest.mantissa +
      scalbn(partner.mantissa, partner.exponent - largest.exponent);
  if (sum == Real{0}) return {};
  int shift = 0;
  const Real mantissa = frexp(sum, &shift);
  return ScaledCtTerm{mantissa, largest.exponent + shift};
}

__device__ inline bool ct_term_active(const ScaledCtTerm& term) {
  return term.mantissa != Real{0};
}

__device__ inline void swap_scaled_ct_terms(ScaledCtTerm& lhs,
                                            ScaledCtTerm& rhs) {
  const ScaledCtTerm temporary = lhs;
  lhs = rhs;
  rhs = temporary;
}

// Keep the larger active term in `selected`.  Every invocation below names its
// operands explicitly; there is no runtime subscript or selected-term address
// for the GPU compiler to lower to scratch/LDS storage.
__device__ inline void select_larger_scaled_ct_term(
    ScaledCtTerm& selected, ScaledCtTerm& candidate) {
  if (ct_term_active(candidate) &&
      (!ct_term_active(selected) || ct_term_larger(candidate, selected))) {
    swap_scaled_ct_terms(selected, candidate);
  }
}

// Select the largest term opposite in sign to `largest`; if no opposite-sign
// term exists, select the largest same-sign term.  This is the range-preserving
// cancellation order used by reduce_scaled_terms, expressed as a fixed scan.
__device__ inline void select_scaled_ct_partner(
    const ScaledCtTerm& largest, ScaledCtTerm& selected,
    ScaledCtTerm& candidate) {
  if (!ct_term_active(candidate)) return;
  const bool candidate_opposite =
      (candidate.mantissa < Real{0}) != (largest.mantissa < Real{0});
  const bool selected_active = ct_term_active(selected);
  const bool selected_opposite = selected_active &&
      ((selected.mantissa < Real{0}) != (largest.mantissa < Real{0}));
  if (!selected_active ||
      (candidate_opposite && !selected_opposite) ||
      (candidate_opposite == selected_opposite &&
       ct_term_larger(candidate, selected))) {
    swap_scaled_ct_terms(selected, candidate);
  }
}

__device__ inline void reduce_scaled_ct_round(
    ScaledCtTerm& t0, ScaledCtTerm& t1, ScaledCtTerm& t2, ScaledCtTerm& t3,
    ScaledCtTerm& t4, ScaledCtTerm& t5, ScaledCtTerm& t6, ScaledCtTerm& t7) {
  select_larger_scaled_ct_term(t0, t1);
  select_larger_scaled_ct_term(t0, t2);
  select_larger_scaled_ct_term(t0, t3);
  select_larger_scaled_ct_term(t0, t4);
  select_larger_scaled_ct_term(t0, t5);
  select_larger_scaled_ct_term(t0, t6);
  select_larger_scaled_ct_term(t0, t7);
  if (!ct_term_active(t0)) return;

  select_scaled_ct_partner(t0, t1, t2);
  select_scaled_ct_partner(t0, t1, t3);
  select_scaled_ct_partner(t0, t1, t4);
  select_scaled_ct_partner(t0, t1, t5);
  select_scaled_ct_partner(t0, t1, t6);
  select_scaled_ct_partner(t0, t1, t7);
  if (!ct_term_active(t1)) return;
  t0 = combine_scaled_ct_terms(t0, t1);
  t1 = {};
}

__device__ inline Real finish_scaled_ct_terms(
    ScaledCtTerm t0, ScaledCtTerm t1, ScaledCtTerm t2, ScaledCtTerm t3,
    ScaledCtTerm t4, ScaledCtTerm t5, ScaledCtTerm t6, ScaledCtTerm t7) {
  reduce_scaled_ct_round(t0, t1, t2, t3, t4, t5, t6, t7);
  reduce_scaled_ct_round(t0, t1, t2, t3, t4, t5, t6, t7);
  reduce_scaled_ct_round(t0, t1, t2, t3, t4, t5, t6, t7);
  reduce_scaled_ct_round(t0, t1, t2, t3, t4, t5, t6, t7);
  reduce_scaled_ct_round(t0, t1, t2, t3, t4, t5, t6, t7);
  reduce_scaled_ct_round(t0, t1, t2, t3, t4, t5, t6, t7);
  reduce_scaled_ct_round(t0, t1, t2, t3, t4, t5, t6, t7);
  if (t0.mantissa != Real{0}) return scalbn(t0.mantissa, t0.exponent);
  if (t1.mantissa != Real{0}) return scalbn(t1.mantissa, t1.exponent);
  if (t2.mantissa != Real{0}) return scalbn(t2.mantissa, t2.exponent);
  if (t3.mantissa != Real{0}) return scalbn(t3.mantissa, t3.exponent);
  if (t4.mantissa != Real{0}) return scalbn(t4.mantissa, t4.exponent);
  if (t5.mantissa != Real{0}) return scalbn(t5.mantissa, t5.exponent);
  if (t6.mantissa != Real{0}) return scalbn(t6.mantissa, t6.exponent);
  if (t7.mantissa != Real{0}) return scalbn(t7.mantissa, t7.exponent);
  return Real{0};
}

__device__ __attribute__((noinline)) Real scaled_face_average(Real e0, Real e1) {
  if (!(isfinite(e0) && isfinite(e1))) {
    return Real{0.5} * e0 + Real{0.5} * e1;
  }
  return finish_scaled_ct_terms(
      make_scaled_ct_term(Real{1}, e0, Real{2}),
      make_scaled_ct_term(Real{1}, e1, Real{2}), {}, {}, {}, {}, {}, {});
}

__device__ __attribute__((noinline)) Real scaled_face_interpolation5(
    Real e0, Real e1, Real e2, Real e3, Real e4, Real e5) {
  if (!(isfinite(e0) && isfinite(e1) && isfinite(e2) &&
        isfinite(e3) && isfinite(e4) && isfinite(e5))) {
    return (Real{3} * e0 - Real{25} * e1 + Real{150} * e2 +
            Real{150} * e3 - Real{25} * e4 + Real{3} * e5) / Real{256};
  }
  return finish_scaled_ct_terms(
      make_scaled_ct_term(Real{3}, e0, Real{256}),
      make_scaled_ct_term(Real{-25}, e1, Real{256}),
      make_scaled_ct_term(Real{150}, e2, Real{256}),
      make_scaled_ct_term(Real{150}, e3, Real{256}),
      make_scaled_ct_term(Real{-25}, e4, Real{256}),
      make_scaled_ct_term(Real{3}, e5, Real{256}), {}, {});
}

__device__ __attribute__((noinline)) Real scaled_face_interpolation7(
    Real e0, Real e1, Real e2, Real e3,
    Real e4, Real e5, Real e6, Real e7) {
  if (!(isfinite(e0) && isfinite(e1) && isfinite(e2) && isfinite(e3) &&
        isfinite(e4) && isfinite(e5) && isfinite(e6) && isfinite(e7))) {
    return (-Real{5} * e0 + Real{49} * e1 - Real{245} * e2 +
            Real{1225} * e3 + Real{1225} * e4 - Real{245} * e5 +
            Real{49} * e6 - Real{5} * e7) / Real{2048};
  }
  return finish_scaled_ct_terms(
      make_scaled_ct_term(Real{-5}, e0, Real{2048}),
      make_scaled_ct_term(Real{49}, e1, Real{2048}),
      make_scaled_ct_term(Real{-245}, e2, Real{2048}),
      make_scaled_ct_term(Real{1225}, e3, Real{2048}),
      make_scaled_ct_term(Real{1225}, e4, Real{2048}),
      make_scaled_ct_term(Real{-245}, e5, Real{2048}),
      make_scaled_ct_term(Real{49}, e6, Real{2048}),
      make_scaled_ct_term(Real{-5}, e7, Real{2048}));
}

struct InterfaceView {
  const double *Lrho, *Lmx, *Lmy, *Lmz, *Lenergy, *Lbx, *Lby, *Lbz;
  const double *Rrho, *Rmx, *Rmy, *Rmz, *Renergy, *Rbx, *Rby, *Rbz;
};

struct BackgroundView {
  int active;
  const double *bx, *by, *bz;
};

__device__ inline MhdState load_interface_side(const InterfaceView& v,
                                                std::size_t k, bool left) {
  MhdState s;
  if (left) {
    s.rho=v.Lrho[k]; s.mx=v.Lmx[k]; s.my=v.Lmy[k]; s.mz=v.Lmz[k];
    s.energy=v.Lenergy[k]; s.bx=v.Lbx[k]; s.by=v.Lby[k]; s.bz=v.Lbz[k];
  } else {
    s.rho=v.Rrho[k]; s.mx=v.Rmx[k]; s.my=v.Rmy[k]; s.mz=v.Rmz[k];
    s.energy=v.Renergy[k]; s.bx=v.Rbx[k]; s.by=v.Rby[k]; s.bz=v.Rbz[k];
  }
  return s;
}

// Directional Godunov electric field.  For an x-face F_x(By)=-Ez; for a
// y-face F_y(Bx)=+Ez.  Reusing HLLD here gives CT the same upwind wave fan as
// the fluid flux instead of a disconnected centred cell-EMF average.
__device__ inline Real face_ez(const quasar::Grid2D& g, int dir, int i, int j,
                               int hll_only, Real gamma, const InterfaceView& v,
                               const BackgroundView& bgv) {
  const std::size_t k = g.index(i, j);
  MhdState L = load_interface_side(v, k, true);
  MhdState R = load_interface_side(v, k, false);
  quasar::numerics::MhdBackground b0{};
  if (bgv.active != 0) {
    b0 = quasar::mhd::load_interface_background(
        g, dir, bgv.bx, bgv.by, bgv.bz, i, j,
        hll_only != 0 ? 1 : 0);
  }
  const MhdState Lr = quasar::numerics::hlld::rotate_in(L, dir);
  const MhdState Rr = quasar::numerics::hlld::rotate_in(R, dir);
  const auto b0r = quasar::numerics::hlld::rotate_in(b0, dir);
  const MhdFlux fr = hll_only != 0
      ? quasar::numerics::hlld::lax_friedrichs_flux_split_x(
            Lr, Rr, b0r, gamma)
      : quasar::numerics::hlld::hlld_flux_split_x(Lr, Rr, b0r, gamma);
  const MhdFlux f = quasar::numerics::hlld::rotate_out(fr, dir);
  return (dir == 0) ? -f.by : f.bx;
}

__device__ inline int face_index(int q, int n, int lo_mode, int hi_mode) {
  if (lo_mode == 0 && hi_mode == 0) {
    int w = q % n;
    if (w < 0) w += n;
    return w;
  }
  return (q < 0) ? 0 : ((q > n) ? n : q);
}

__device__ inline Real xface_ez(const quasar::Grid2D& g, int i, int j,
                                int hll_only, Real gamma, const InterfaceView& x,
                                const BackgroundView& bg,
                                quasar::mhd::BoundaryFlags4 modes) {
  const int ii = face_index(i, g.nx, modes.side[0], modes.side[1]);
  const int jj = face_index(j, g.ny, modes.side[2], modes.side[3]);
  return face_ez(g, 0, ii, jj, hll_only, gamma, x, bg);
}

__device__ inline Real yface_ez(const quasar::Grid2D& g, int i, int j,
                                int hll_only, Real gamma, const InterfaceView& y,
                                const BackgroundView& bg,
                                quasar::mhd::BoundaryFlags4 modes) {
  const int ii = face_index(i, g.nx, modes.side[0], modes.side[1]);
  const int jj = face_index(j, g.ny, modes.side[2], modes.side[3]);
  return face_ez(g, 1, ii, jj, hll_only, gamma, y, bg);
}

__device__ inline Real interp_x_to_corner(
    const quasar::Grid2D& g, int i, int j, int order, int hll_only, Real gamma,
    const InterfaceView& x, const BackgroundView& bg,
    quasar::mhd::BoundaryFlags4 modes) {
  if (order >= 7) {
    return scaled_face_interpolation7(
        xface_ez(g, i, j - 4, hll_only, gamma, x, bg, modes),
        xface_ez(g, i, j - 3, hll_only, gamma, x, bg, modes),
        xface_ez(g, i, j - 2, hll_only, gamma, x, bg, modes),
        xface_ez(g, i, j - 1, hll_only, gamma, x, bg, modes),
        xface_ez(g, i, j, hll_only, gamma, x, bg, modes),
        xface_ez(g, i, j + 1, hll_only, gamma, x, bg, modes),
        xface_ez(g, i, j + 2, hll_only, gamma, x, bg, modes),
        xface_ez(g, i, j + 3, hll_only, gamma, x, bg, modes));
  }
  if (order >= 5) {
    return scaled_face_interpolation5(
        xface_ez(g, i, j - 3, hll_only, gamma, x, bg, modes),
        xface_ez(g, i, j - 2, hll_only, gamma, x, bg, modes),
        xface_ez(g, i, j - 1, hll_only, gamma, x, bg, modes),
        xface_ez(g, i, j, hll_only, gamma, x, bg, modes),
        xface_ez(g, i, j + 1, hll_only, gamma, x, bg, modes),
        xface_ez(g, i, j + 2, hll_only, gamma, x, bg, modes));
  }
  return scaled_face_average(
      xface_ez(g, i, j - 1, hll_only, gamma, x, bg, modes),
      xface_ez(g, i, j, hll_only, gamma, x, bg, modes));
}

__device__ inline Real interp_y_to_corner(
    const quasar::Grid2D& g, int i, int j, int order, int hll_only, Real gamma,
    const InterfaceView& y, const BackgroundView& bg,
    quasar::mhd::BoundaryFlags4 modes) {
  if (order >= 7) {
    return scaled_face_interpolation7(
        yface_ez(g, i - 4, j, hll_only, gamma, y, bg, modes),
        yface_ez(g, i - 3, j, hll_only, gamma, y, bg, modes),
        yface_ez(g, i - 2, j, hll_only, gamma, y, bg, modes),
        yface_ez(g, i - 1, j, hll_only, gamma, y, bg, modes),
        yface_ez(g, i, j, hll_only, gamma, y, bg, modes),
        yface_ez(g, i + 1, j, hll_only, gamma, y, bg, modes),
        yface_ez(g, i + 2, j, hll_only, gamma, y, bg, modes),
        yface_ez(g, i + 3, j, hll_only, gamma, y, bg, modes));
  }
  if (order >= 5) {
    return scaled_face_interpolation5(
        yface_ez(g, i - 3, j, hll_only, gamma, y, bg, modes),
        yface_ez(g, i - 2, j, hll_only, gamma, y, bg, modes),
        yface_ez(g, i - 1, j, hll_only, gamma, y, bg, modes),
        yface_ez(g, i, j, hll_only, gamma, y, bg, modes),
        yface_ez(g, i + 1, j, hll_only, gamma, y, bg, modes),
        yface_ez(g, i + 2, j, hll_only, gamma, y, bg, modes));
  }
  return scaled_face_average(
      yface_ez(g, i - 1, j, hll_only, gamma, y, bg, modes),
      yface_ez(g, i, j, hll_only, gamma, y, bg, modes));
}

__global__ void ct_emf_kernel(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 modes, int scheme_order,
    int cylindrical, int hll_only, Real gamma, InterfaceView ifx, InterfaceView ify,
    BackgroundView bg,
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
        g, i, j, scheme_order, hll_only, gamma, ifx, bg, modes);
    const Real ey = interp_y_to_corner(
        g, i, j, scheme_order, hll_only, gamma, ify, bg, modes);
    ez_edge[k] = scaled_face_average(ex, ey);
  }
  // In-plane EMF components are zero for the 2D m=0 induction update (no z
  // gradient); populated for parity with the EmfField2D layout.
  ex_edge[k] = Real{0};
  ey_edge[k] = Real{0};
}

// Face-B update: dB/dt = -curl E.
//   bx_face(i,j) -= dt * (ez(i,j+1) - ez(i,j)) / dy
//   by_face(i,j) += dt * (ez(i+1,j) - ez(i,j)) / dx
__global__ void face_b_update_kernel(
    quasar::Grid2D g, int cylindrical, Real dt,
    double* __restrict__ bx_face, double* __restrict__ by_face,
    const double* __restrict__ ez_edge) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > g.nx || j > g.ny) return;
  const std::size_t k = g.index(i, j);
  const Real one2[2] = {Real{1}, Real{1}};

  // Bx has nx+1 physical x faces and ny cell rows.  Include i==nx so the
  // last cell's high face advances by the same curl as every interior face.
  if (i <= g.nx && j < g.ny) {
    const Real bx_a[2] = {
        -ez_edge[g.index(i, j + 1)], ez_edge[k]};
    const Real bx_b[2] = {dt, dt};
    const Real bx_d0[2] = {g.dy(), g.dy()};
    bx_face[k] += quasar::numerics::scaled_product_quotient_sum(
        bx_a, bx_b, bx_d0, one2, 2);
  }

  // By has nx cell columns and ny+1 physical y faces.  Include j==ny for the
  // transposed high face.  These extents/stencils exactly match
  // emf_curl_rate_kernel, with dt restored and accumulation into the live B.
  if (i < g.nx && j <= g.ny) {
    if (cylindrical != 0) {
      const Real dr = g.dx();
      const Real r_center = g.r_at_cell_center(i);
      const Real ez_lo = ez_edge[k];
      const Real ez_hi = ez_edge[g.index(i + 1, j)];
      const Real a[4] = {ez_hi, -ez_lo, ez_hi, ez_lo};
      const Real b[4] = {dt, dt, dt, dt};
      const Real d0[4] = {dr, dr, r_center, r_center};
      const Real d1[4] = {Real{1}, Real{1}, Real{2}, Real{2}};
      by_face[k] += quasar::numerics::scaled_product_quotient_sum(
          a, b, d0, d1, 4);
    } else {
      const Real by_a[2] = {
          ez_edge[g.index(i + 1, j)], -ez_edge[k]};
      const Real by_b[2] = {dt, dt};
      const Real by_d0[2] = {g.dx(), g.dx()};
      by_face[k] += quasar::numerics::scaled_product_quotient_sum(
          by_a, by_b, by_d0, one2, 2);
    }
  }
}

// CT EMF curl RATE written into the residual's face-B slots. This is the SAME
// stencil as face_b_update_kernel above with the dt factor removed (it is the
// dB/dt rate; rk_stage applies dt later) and assigned (=) instead of accumulated
// so it OVERWRITES the face-B contribution that flux_difference left in dudt.
// ONLY dudt.bx_face / dudt.by_face are touched -- every other dudt component is
// left exactly as flux_difference wrote it. The indexing, dx/dy, and interior
// loop bounds match face_b_update_kernel exactly so the telescoping div(B)
// cancellation stays bit-for-bit identical.
__global__ void emf_curl_rate_kernel(
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

void launch_mhd_ct_emf(const MhdField2D<Real>& u,
                       const MhdBackgroundField<Real>& b0,
                       const quasar::numerics::MhdInterfaceStates<Real>& ifx,
                       const quasar::numerics::MhdInterfaceStates<Real>& ify,
                       BoundaryFlags4 flags, EmfField2D<Real>& emf, Real gamma,
                       stream_t stream, int scheme_order, bool cylindrical,
                       bool hll_only) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  // Launch over nx+1 x ny+1 so the high-edge corner EMF (index nx/ny) read by
  // the face-B curl-rate stencil at the last interior cell is produced too.
  const dim3 grid((static_cast<unsigned>(g.nx) + 1 + block.x - 1) / block.x,
                  (static_cast<unsigned>(g.ny) + 1 + block.y - 1) / block.y);
  // When the background is inactive pass null device pointers; the kernel takes
  // the zero-background fast path and never dereferences them.
  const InterfaceView xv{
      ifx.Lrho.device_ptr(), ifx.Lmx.device_ptr(), ifx.Lmy.device_ptr(),
      ifx.Lmz.device_ptr(), ifx.Lenergy.device_ptr(), ifx.Lbx.device_ptr(),
      ifx.Lby.device_ptr(), ifx.Lbz.device_ptr(), ifx.Rrho.device_ptr(),
      ifx.Rmx.device_ptr(), ifx.Rmy.device_ptr(), ifx.Rmz.device_ptr(),
      ifx.Renergy.device_ptr(), ifx.Rbx.device_ptr(), ifx.Rby.device_ptr(),
      ifx.Rbz.device_ptr()};
  const InterfaceView yv{
      ify.Lrho.device_ptr(), ify.Lmx.device_ptr(), ify.Lmy.device_ptr(),
      ify.Lmz.device_ptr(), ify.Lenergy.device_ptr(), ify.Lbx.device_ptr(),
      ify.Lby.device_ptr(), ify.Lbz.device_ptr(), ify.Rrho.device_ptr(),
      ify.Rmx.device_ptr(), ify.Rmy.device_ptr(), ify.Rmz.device_ptr(),
      ify.Renergy.device_ptr(), ify.Rbx.device_ptr(), ify.Rby.device_ptr(),
      ify.Rbz.device_ptr()};
  const BackgroundView bg{
      b0.active ? 1 : 0,
      b0.active ? b0.b0x_face.device_ptr() : nullptr,
      b0.active ? b0.b0y_face.device_ptr() : nullptr,
      b0.active ? b0.b0z_cell.device_ptr() : nullptr};
  hipLaunchKernelGGL(ct_emf_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, flags, scheme_order, cylindrical ? 1 : 0,
                     hll_only ? 1 : 0, gamma,
                     xv, yv, bg,
                     emf.ez_edge.device_ptr(), emf.ex_edge.device_ptr(),
                     emf.ey_edge.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_face_b_update(MhdField2D<Real>& u, const EmfField2D<Real>& emf,
                              Real dt, stream_t stream, bool cylindrical) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid((static_cast<unsigned>(g.nx) + 1 + block.x - 1) / block.x,
                  (static_cast<unsigned>(g.ny) + 1 + block.y - 1) / block.y);
  hipLaunchKernelGGL(face_b_update_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, cylindrical ? 1 : 0, dt,
                     u.bx_face.device_ptr(), u.by_face.device_ptr(),
                     emf.ez_edge.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
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
