#pragma once

// Sixth-order Grad-Shafranov operator built from the compact (Pade) derivatives.
//
//   Delta* psi = psi_rr - psi_r / r + psi_zz
//
// This is the "L6" of the defect-correction pair: it defines what the converged
// answer means, and it is evaluated ONLY in residual assembly. It is never
// smoothed, never coarsened, and never inverted directly -- see the note in
// pade_derivative.hpp on why a globally-coupled operator cannot be a multigrid
// smoother.
//
// -- Expanded, not conservative -----------------------------------------------
// gs_operator_l2.hpp deliberately uses the conservative half-node form because
// that is what makes the second-order matrix symmetric and therefore
// multigrid-friendly. Here the opposite choice is right: the Pade operators
// deliver psi_rr and psi_r directly as sixth-order line derivatives, so the
// expanded form is the one that inherits their order. There is no conservative
// half-node analogue of a compact derivative to use instead.
//
// The two operators consequently do NOT agree pointwise -- they differ at
// O(h^2), which is exactly the defect that defect correction iterates away.
//
// -- Cost ---------------------------------------------------------------------
// Each application solves nz tridiagonal systems of length nr (for the radial
// derivatives) plus nr systems of length nz (for the axial one). All three
// required derivatives are computed with two line sweeps per direction, so one
// L6 application is roughly the cost of four line solves over the grid.

#include "quasar/core/types.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/numerics/pade_line_solve.hpp"

#include <cstddef>
#include <vector>

namespace quasar::numerics {

// y := Delta* x, sixth order, on interior nodes (zero on the boundary).
//
// The Pade line solves run across the FULL line including boundary nodes,
// because the one-sided closures need those values; only the returned operator
// is restricted to the interior.
inline void gs_apply_l6(const EllipticGrid& g, const ScalarField& x,
                        ScalarField& y) {
  y.assign(g.size(), Real{0});

  const Real dr = g.dr();
  const Real dz = g.dz();

  // Radial sweeps: for each z row, take d/dr and d2/dr2 along r.
  std::vector<Real> line(static_cast<std::size_t>(g.nr));
  std::vector<Real> d_r(g.size(), Real{0});
  std::vector<Real> d_rr(g.size(), Real{0});
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      line[static_cast<std::size_t>(i)] = x[g.index(i, j)];
    }
    const auto first  = pade::first_derivative(line, dr);
    const auto second = pade::second_derivative(line, dr);
    for (int i = 0; i < g.nr; ++i) {
      d_r[g.index(i, j)]  = first[static_cast<std::size_t>(i)];
      d_rr[g.index(i, j)] = second[static_cast<std::size_t>(i)];
    }
  }

  // Axial sweeps: for each r column, take d2/dz2 along z.
  std::vector<Real> colline(static_cast<std::size_t>(g.nz));
  std::vector<Real> d_zz(g.size(), Real{0});
  for (int i = 0; i < g.nr; ++i) {
    for (int j = 0; j < g.nz; ++j) {
      colline[static_cast<std::size_t>(j)] = x[g.index(i, j)];
    }
    const auto second = pade::second_derivative(colline, dz);
    for (int j = 0; j < g.nz; ++j) {
      d_zz[g.index(i, j)] = second[static_cast<std::size_t>(j)];
    }
  }

  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      const std::size_t k = g.index(i, j);
      y[k] = d_rr[k] - d_r[k] / g.r(i) + d_zz[k];
    }
  }
}

inline void gs_residual_l6(const EllipticGrid& g, const ScalarField& x,
                           const ScalarField& b, ScalarField& r) {
  gs_apply_l6(g, x, r);
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      const std::size_t k = g.index(i, j);
      r[k] = b[k] - r[k];
    }
  }
  // Boundary entries of the residual are meaningless (Dirichlet data is exact
  // there); zero them so norms are taken over genuine unknowns only.
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      if (g.on_boundary(i, j)) r[g.index(i, j)] = Real{0};
    }
  }
}

// Minimum grid size for which the sixth-order operator is usable at all: both
// axes must admit the two-sided Pade closures.
inline bool l6_is_applicable(const EllipticGrid& g) noexcept {
  return g.nr >= pade::min_line_extent && g.nz >= pade::min_line_extent;
}

}  // namespace quasar::numerics
