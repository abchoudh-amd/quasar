#pragma once

// Sixth-order compact (Pade) derivative operators on a uniform 1D line.
//
// These are the building block of the high-order elliptic path: the
// Grad-Shafranov operator Delta* is assembled from compact first and second
// derivatives rather than from a wide explicit stencil. A compact scheme
// attains sixth order with a three-point derivative coupling, at the cost of
// making each derivative evaluation an implicit tridiagonal solve along the
// line:
//
//   first derivative   alpha f'_{i-1} + f'_i + alpha f'_{i+1}
//                        = a (f_{i+1}-f_{i-1})/(2h) + b (f_{i+2}-f_{i-2})/(4h)
//   second derivative  alpha f''_{i-1} + f''_i + alpha f''_{i+1}
//                        = a (f_{i+1}-2f_i+f_{i-1})/h^2
//                        + b (f_{i+2}-2f_i+f_{i-2})/(4h^2)
//
// That implicitness is the defining property of this operator and it drives
// the surrounding design: because the resulting system couples every node on a
// line, no local relaxation can smooth it, so a Pade operator is NEVER used as
// the multigrid smoother. It is applied only in residual evaluation, with a
// compact second-order operator carrying the multigrid correction (defect
// correction). See numerics/defect_correction.hpp.
//
// -- Coefficient provenance ---------------------------------------------------
// Every constant below was derived by exact rational Taylor matching, not
// transcribed. The interior pairs are the classical Lele values, recovered here
// by solving the order conditions over Fraction arithmetic; the closures were
// derived the same way. tests/unit/numerics/test_pade_derivative.cpp re-derives
// the order numerically through a manufactured solution, which is what actually
// gates the claim.
//
// A caution that cost real effort to find: the interior order conditions are
// degenerate. For the first derivative every even-order condition vanishes
// identically by symmetry, and for the second derivative every odd-order one
// does. Solving the first three conditions in index order therefore silently
// yields the FOURTH-order scheme (alpha=1/4 and alpha=1/10 respectively), which
// is a correct scheme and passes every smoothness check -- it simply is not
// sixth order. The non-degenerate conditions must be selected explicitly.
//
// -- Boundary closures --------------------------------------------------------
// The Grad-Shafranov domain is annular (r_min > 0), so both radial ends are
// physical boundaries and need one-sided rows. Closures were surveyed over
// (number of implicit neighbours) x (explicit stencil width). Two families
// reach sixth order:
//
//   one implicit neighbour, six points : alpha = 5
//   two implicit neighbours, five pts  : alpha = (8, 6)
//
// The one-neighbour family is used. It is not the higher-order option
// available, and that is deliberate: it keeps the boundary row inside the same
// tridiagonal structure as the interior, so the line solve stays strictly
// tridiagonal with no bandwidth increase, and it keeps the off-diagonal
// magnitude smaller. Widening to two implicit neighbours reaches seventh order
// but makes alpha = (10, 10), degrading the diagonal dominance the parallel
// cyclic reduction solve depends on. Order beyond six buys nothing here because
// the scheme as a whole is sixth order.
//
// The tridiagonal systems these rows produce are diagonally dominant in the
// interior (alpha < 1/2) but NOT at the boundary rows (alpha = 5 and 126/11).
// The solver must therefore use pivoting-free cyclic reduction only after the
// boundary rows are eliminated, or accept partial pivoting on those two rows.

#include "quasar/core/types.hpp"

namespace quasar::numerics::pade {

// -- Interior coefficients ----------------------------------------------------
// Sixth-order tridiagonal first derivative (Lele). Leading truncation term is
// -h^6 f^(7)/1260.
struct FirstDerivativeInterior {
  static constexpr Real alpha = Real{1} / Real{3};
  static constexpr Real a     = Real{14} / Real{9};   // (i+1, i-1) pair, over 2h
  static constexpr Real b     = Real{1} / Real{9};    // (i+2, i-2) pair, over 4h
};

// Sixth-order tridiagonal second derivative (Lele). Leading truncation term is
// -23 h^6 f^(8)/55440.
struct SecondDerivativeInterior {
  static constexpr Real alpha = Real{2} / Real{11};
  static constexpr Real a     = Real{12} / Real{11};  // (i+1, i, i-1), over h^2
  static constexpr Real b     = Real{3} / Real{11};   // (i+2, i, i-2), over 4h^2
};

// -- Boundary closures --------------------------------------------------------
// Row at the first interior node (node 0), written for the low end of the line:
//
//   f'_0 + alpha f'_1 = (1/h) sum_{s=0}^{5} w[s] f_s
//
// The high end mirrors this: reverse the stencil order and negate every weight
// (the first derivative is odd under reflection). Constant-exactness requires
// sum(w) == 0, which these satisfy exactly.
struct FirstDerivativeBoundary {
  static constexpr int  stencil_width = 6;
  static constexpr Real alpha         = Real{5};
  static constexpr Real w[stencil_width] = {
      Real{-197} / Real{60},
      Real{-5}   / Real{12},
      Real{5},
      Real{-5}   / Real{3},
      Real{5}    / Real{12},
      Real{-1}   / Real{20},
  };
};

// Row at the first interior node for the second derivative:
//
//   f''_0 + alpha f''_1 = (1/h^2) sum_{s=0}^{6} w[s] f_s
//
// The high end mirrors this by reversing the stencil order; weights are NOT
// negated (the second derivative is even under reflection). Both
// constant-exactness (sum(w) == 0) and linear-exactness (sum(s*w[s]) == 0) hold
// exactly. Seven points are needed here where the first derivative needed six.
struct SecondDerivativeBoundary {
  static constexpr int  stencil_width = 7;
  static constexpr Real alpha         = Real{126} / Real{11};
  static constexpr Real w[stencil_width] = {
      Real{13097} / Real{990},
      Real{-2943} / Real{110},
      Real{573}   / Real{44},
      Real{167}   / Real{99},
      Real{-18}   / Real{11},
      Real{57}    / Real{110},
      Real{-131}  / Real{1980},
  };
};

// -- Near-boundary rows (node 1 and its mirror) -------------------------------
// Node 1 cannot use the five-point interior stencil without reaching off the
// line, so it needs its own row. The obvious shortcut -- dropping node 1 to the
// standard fourth-order compact pair -- does NOT work, and the manufactured
// solution caught it: the resulting scheme converged at ~3.5 order at node 0
// and ~4.9 at node 1 while the deep interior held exactly 6.0.
//
// The reason is that the node-0 closure derivation assumes its implicit
// neighbour carries a sixth-order derivative. Coupling it to a fourth-order
// node-1 row invalidates that assumption and the error feeds back through the
// implicit solve, degrading node 0 below even the fourth order of the row that
// caused it. Near-boundary rows in a compact scheme cannot be chosen
// independently; the whole set of rows has to close at the target order
// together.
//
// These rows are one-sided in their explicit stencil but keep the tridiagonal
// derivative coupling:
//
//   a_lo f'_0 + f'_1 + a_hi f'_2 = (1/h) sum_s w[s] f_{1+s}
//
// with stencil offsets {-1, 0, 1, 2, 3} relative to node 1.
struct FirstDerivativeNearBoundary {
  static constexpr int  stencil_width = 5;
  static constexpr int  stencil_lo    = -1;  // first offset, relative to node 1
  static constexpr Real alpha_lo      = Real{1} / Real{8};
  static constexpr Real alpha_hi      = Real{3} / Real{4};
  static constexpr Real w[stencil_width] = {
      Real{-43} / Real{96},
      Real{-5}  / Real{6},
      Real{9}   / Real{8},
      Real{1}   / Real{6},
      Real{-1}  / Real{96},
  };
};

// Second-derivative near-boundary row, offsets {-1, 0, 1, 2, 3, 4}.
//
// Note alpha_hi = -131/22, which is large and negative. That is not a
// transcription slip: it is what sixth order costs at this node, and it is the
// reason the line solve cannot assume diagonal dominance. Lower-order variants
// of this row have benign coefficients but reintroduce exactly the order
// reduction described above.
struct SecondDerivativeNearBoundary {
  static constexpr int  stencil_width = 6;
  static constexpr int  stencil_lo    = -1;
  static constexpr Real alpha_lo      = Real{2} / Real{11};
  static constexpr Real alpha_hi      = Real{-131} / Real{22};
  static constexpr Real w[stencil_width] = {
      Real{177}  / Real{88},
      Real{-507} / Real{44},
      Real{783}  / Real{44},
      Real{-201} / Real{22},
      Real{81}   / Real{88},
      Real{-3}   / Real{44},
  };
};

// Minimum number of interior points a line must have for these closures to be
// applied at both ends without overlap. Below this the line must fall back to a
// lower-order scheme; the multigrid hierarchy hits this on coarse levels, which
// is harmless because coarse levels only ever run the compact second-order
// operator.
inline constexpr int min_line_extent =
    2 * (SecondDerivativeBoundary::stencil_width +
         SecondDerivativeNearBoundary::stencil_width);

}  // namespace quasar::numerics::pade
