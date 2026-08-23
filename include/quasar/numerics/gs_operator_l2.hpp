#pragma once

// Second-order compact discretization of the Grad-Shafranov operator
//
//   Delta* psi = r d_r( (1/r) d_r psi ) + d_zz psi
//
// This is the multigrid workhorse (the "L2" of the defect-correction pair). It
// is intentionally low order: its job is to be cheap, local, and smoothable, not
// accurate. Accuracy comes from the sixth-order Pade residual in
// defect_correction.hpp; L2 only ever supplies the correction.
//
// -- Why the conservative half-node form --------------------------------------
// Delta* is NOT the Poisson operator. Expanding it naively gives
//
//   d_rr psi - (1/r) d_r psi + d_zz psi
//
// which is a non-self-adjoint advection-diffusion-looking operator, and
// discretizing it that way produces a nonsymmetric matrix whose symmetry defect
// is O(dr/r). Multigrid still converges on it, but the natural energy argument
// is gone and red-black smoothing loses its predictable factor.
//
// Discretizing the conservative form directly instead,
//
//   r_i * [ (1/r_{i+1/2}) (psi_{i+1}-psi_i) - (1/r_{i-1/2}) (psi_i-psi_{i-1}) ]
//         / dr^2
//
// makes the radial part symmetric in the weighted inner product
// <u,v>_r = sum u_ij v_ij / r_i, which is the correct inner product for this
// operator. That is what keeps the V-cycle rate grid-independent, and it costs
// nothing extra to evaluate.
//
// -- Sign convention ----------------------------------------------------------
// apply() returns Delta* psi. Delta* is negative-definite on a Dirichlet domain
// (like the Laplacian), so the diagonal entry is negative and the Jacobi damping
// factor below carries the matching sign.

#include "quasar/core/types.hpp"
#include "quasar/numerics/elliptic_grid.hpp"

#include <algorithm>
#include <cstddef>

namespace quasar::numerics {

// Stencil weights for one interior node. Kept as a struct so the smoother, the
// residual, and the Newton Jacobian all read the same coefficients instead of
// each re-deriving them (a classic source of silent inconsistency between an
// operator and its preconditioner).
struct GsStencil {
  Real w_center{0};
  Real w_rm{0};  // i-1
  Real w_rp{0};  // i+1
  Real w_zm{0};  // j-1
  Real w_zp{0};  // j+1
};

inline GsStencil gs_stencil(const EllipticGrid& g, int i) {
  const Real dr = g.dr();
  const Real dz = g.dz();
  const Real inv_dr2 = Real{1} / (dr * dr);
  const Real inv_dz2 = Real{1} / (dz * dz);
  const Real r_i = g.r(i);

  // Conservative radial part: r_i * d_r( (1/r) d_r psi ).
  const Real a_p = r_i / g.r_half(i) * inv_dr2;      // toward i+1
  const Real a_m = r_i / g.r_half(i - 1) * inv_dr2;  // toward i-1

  GsStencil s;
  s.w_rp = a_p;
  s.w_rm = a_m;
  s.w_zp = inv_dz2;
  s.w_zm = inv_dz2;
  s.w_center = -(a_p + a_m + Real{2} * inv_dz2);
  return s;
}

// y := Delta* x on interior nodes. Boundary nodes of `y` are set to zero; the
// caller owns the Dirichlet data in `x`.
inline void gs_apply_l2(const EllipticGrid& g, const ScalarField& x,
                        ScalarField& y) {
  // resize-then-fill rather than assign(): assign() on a shorter vector
  // reallocates, and GCC's interprocedural constant propagation can convince
  // itself the old (smaller) capacity is still in play when this is inlined at
  // two different compile-time grid sizes, producing a spurious
  // -Wstringop-overflow. Behaviour is identical.
  y.resize(g.size());
  std::fill(y.begin(), y.end(), Real{0});
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      const GsStencil s = gs_stencil(g, i);
      y[g.index(i, j)] = s.w_center * x[g.index(i, j)]
                       + s.w_rm * x[g.index(i - 1, j)]
                       + s.w_rp * x[g.index(i + 1, j)]
                       + s.w_zm * x[g.index(i, j - 1)]
                       + s.w_zp * x[g.index(i, j + 1)];
    }
  }
}

// r := b - Delta* x on interior nodes (zero on the boundary).
inline void gs_residual_l2(const EllipticGrid& g, const ScalarField& x,
                           const ScalarField& b, ScalarField& r) {
  r.resize(g.size());
  std::fill(r.begin(), r.end(), Real{0});
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      const GsStencil s = gs_stencil(g, i);
      const std::size_t k = g.index(i, j);
      const Real ax = s.w_center * x[k]
                    + s.w_rm * x[g.index(i - 1, j)]
                    + s.w_rp * x[g.index(i + 1, j)]
                    + s.w_zm * x[g.index(i, j - 1)]
                    + s.w_zp * x[g.index(i, j + 1)];
      r[k] = b[k] - ax;
    }
  }
}

// One red-black Gauss-Seidel sweep for Delta* x = b.
//
// Red-black rather than lexicographic because the two colours are independent
// within a sweep, so this maps directly onto a GPU kernel with no ordering
// dependence -- the same reason the plan chose it. Lexicographic GS smooths
// slightly better per sweep but is inherently sequential.
inline void gs_smooth_rbgs(const EllipticGrid& g, ScalarField& x,
                           const ScalarField& b, int sweeps = 1) {
  for (int s = 0; s < sweeps; ++s) {
    for (int colour = 0; colour < 2; ++colour) {
      for (int j = 1; j < g.nz - 1; ++j) {
        for (int i = 1; i < g.nr - 1; ++i) {
          if (((i + j) & 1) != colour) continue;
          const GsStencil st = gs_stencil(g, i);
          const std::size_t k = g.index(i, j);
          const Real off = st.w_rm * x[g.index(i - 1, j)]
                         + st.w_rp * x[g.index(i + 1, j)]
                         + st.w_zm * x[g.index(i, j - 1)]
                         + st.w_zp * x[g.index(i, j + 1)];
          x[k] = (b[k] - off) / st.w_center;
        }
      }
    }
  }
}

}  // namespace quasar::numerics
