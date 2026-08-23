#pragma once

// Defect correction: sixth-order accuracy from a second-order solver.
//
// The iteration is
//
//   r_k = b - L6(x_k)            <- sixth-order residual, defines the answer
//   L2 e_k = r_k                 <- second-order multigrid solve (approximate)
//   x_{k+1} = x_k + omega e_k
//
// At a fixed point r = 0, so x satisfies L6 x = b EXACTLY -- the converged
// solution carries sixth-order accuracy even though every linear solve performed
// was second order. L2 acts purely as a preconditioner; its accuracy never
// enters the answer, only the convergence rate.
//
// This is the mechanism that resolves the tension between the two design
// choices: Pade operators are globally coupled and cannot be smoothed, while
// multigrid needs a local operator. Neither constraint is violated because the
// two operators appear in different roles.
//
// -- Convergence: measured, not assumed ---------------------------------------
// The iteration matrix is (I - omega * L2^{-1} L6). It is tempting to argue
// that since L6 = L2 + O(h^2) the contraction must be O(h^2) and convergence
// must take a handful of iterations that get cheaper with refinement. That
// argument is wrong, and the measured behaviour says so plainly:
//
//   omega   inner   its(n=33)   its(n=65)
//   0.6       1        50          50
//   0.8       2        24          24     <- default
//   1.0       2        53          52
//   1.2       any     diverges    diverges
//
// The contraction is a FIXED factor (~0.64 at omega=1), not O(h^2), and the
// iteration count is grid-independent rather than decreasing. The reason is that
// L2^{-1} L6 has eigenvalues spread on both sides of 1: the high-frequency modes
// where L2 and L6 disagree most are precisely the ones L2^{-1} handles worst, so
// the spectral radius is set by that mismatch and not by the O(h^2) size of the
// difference operator.
//
// Grid-independence is still the property that matters -- cost per solve scales
// linearly with the grid, which is what makes the scheme usable at 4096^2.
//
// -- Relaxation ---------------------------------------------------------------
// omega = 0.8 is the default because it is near-optimal (24 iterations vs 53 at
// omega = 1) and comfortably inside the stability limit: omega = 1.2 diverges.
// The unrelaxed iteration converges but wastes roughly a factor of two. Do not
// raise omega above 1 without re-measuring; the margin is thin.
//
// The inner multigrid solve does not need to be tight. Going from 1 to 2 inner
// V-cycles cuts the outer count meaningfully (33 -> 24 at omega = 0.8), but 4
// cycles buys almost nothing further (22), so 2 is the default.

#include "quasar/core/types.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/numerics/geometric_multigrid.hpp"
#include "quasar/numerics/gs_operator_l2.hpp"
#include "quasar/numerics/gs_operator_l6.hpp"

#include <cmath>
#include <vector>

namespace quasar::numerics {

struct DefectCorrectionConfig {
  int  max_iterations{60};
  Real tolerance{Real{1e-10}};   // relative, on the L6 residual
  Real relaxation{Real{0.8}};    // near-optimal; >1 diverges (see note above)
  int  inner_cycles{2};          // V-cycles per outer step
};

struct DefectCorrectionReport {
  bool converged{false};
  int  iterations{0};
  Real initial_residual{0};
  Real final_residual{0};
  std::vector<Real> residual_history{};
};

// Solve L6 x = b to `cfg.tolerance` using L2 multigrid as the preconditioner.
// `x` must carry the Dirichlet boundary values on entry; they are preserved.
inline DefectCorrectionReport solve_defect_corrected(
    const EllipticGrid& g, ScalarField& x, const ScalarField& b,
    GsMultigrid& mg, DefectCorrectionConfig cfg = {}) {
  DefectCorrectionReport rep;

  ScalarField r = make_field(g);
  ScalarField e = make_field(g);

  gs_residual_l6(g, x, b, r);
  rep.initial_residual = interior_max_norm(g, r);
  rep.final_residual = rep.initial_residual;
  rep.residual_history.push_back(rep.initial_residual);

  if (rep.initial_residual == Real{0}) {
    rep.converged = true;
    return rep;
  }

  for (int it = 1; it <= cfg.max_iterations; ++it) {
    // Correction solve. `e` starts at zero with zero boundary data, so the
    // correction never perturbs the Dirichlet values held in `x`.
    e.assign(g.size(), Real{0});
    for (int c = 0; c < cfg.inner_cycles; ++c) mg.v_cycle(e, r);

    for (int j = 1; j < g.nz - 1; ++j) {
      for (int i = 1; i < g.nr - 1; ++i) {
        x[g.index(i, j)] += cfg.relaxation * e[g.index(i, j)];
      }
    }

    gs_residual_l6(g, x, b, r);
    rep.final_residual = interior_max_norm(g, r);
    rep.residual_history.push_back(rep.final_residual);
    rep.iterations = it;

    if (rep.final_residual <= cfg.tolerance * rep.initial_residual) {
      rep.converged = true;
      break;
    }
  }
  return rep;
}

}  // namespace quasar::numerics
