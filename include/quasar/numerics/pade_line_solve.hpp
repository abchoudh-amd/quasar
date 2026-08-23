#pragma once

// Host-side reference application of the compact (Pade) derivative operators
// along a single uniform line.
//
// This is the correctness oracle for the operator, not the production path.
// Production evaluation happens on device with a parallel cyclic reduction line
// solve; this serial pivoting version exists so the coefficients and
// closures can be verified independently of any HIP kernel, and so the
// manufactured-solution order study has something to compare against. Keeping
// the two paths separate is what makes a kernel bug distinguishable from a
// coefficient bug.
//
// -- Why this solve pivots ----------------------------------------------------
// The systems are diagonally dominant in the interior (alpha = 1/3 and 2/11,
// both < 1/2) but emphatically not in the boundary and near-boundary rows
// (alpha = 5, 126/11, and -131/22). Plain Thomas without pivoting FAILS on the
// second-derivative system: eliminating row 1 into row 2 drives the row-2
// diagonal to exactly zero, and the next elimination factor blows up to ~8e14.
// The resulting derivative error at node 0 was 4.6 and grew under refinement.
//
// This is not an ill-conditioning problem -- the matrix condition number is
// 3.3e4 and grid-independent, and the assembled rows each satisfy
// A*exact - rhs = O(h^6). It is purely the unpivoted algorithm meeting a zero
// pivot. Partial pivoting fixes it completely.
//
// Pivoting makes the factorization produce a second superdiagonal (fill-in), so
// the band carries an extra `upper2` array. Any device line solve must handle
// the same structure: a textbook pivot-free cyclic reduction is NOT valid for
// these closures.

#include "quasar/core/types.hpp"
#include "quasar/numerics/pade_derivative.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quasar::numerics::pade {

// Solve a tridiagonal system in place with partial (row-interchange) pivoting.
// `lower[0]` and `upper[n-1]` are ignored on entry. Pivoting introduces fill-in
// one column beyond the original band, held internally in `upper2`.
inline void tridiagonal_solve_pivoting(std::vector<Real>& lower,
                                       std::vector<Real>& diag,
                                       std::vector<Real>& upper,
                                       std::vector<Real>& rhs) {
  const std::size_t n = diag.size();
  if (n == 0) return;
  if (lower.size() != n || upper.size() != n || rhs.size() != n) {
    throw std::invalid_argument{
        "tridiagonal_solve_pivoting: inconsistent band sizes"};
  }
  std::vector<Real> upper2(n, Real{0});  // fill-in from row interchanges

  for (std::size_t i = 0; i + 1 < n; ++i) {
    // Row i has entries (diag[i], upper[i], upper2[i]); row i+1 has
    // (lower[i+1], diag[i+1], upper[i+1]) aligned one column to the right.
    if (std::abs(lower[i + 1]) > std::abs(diag[i])) {
      std::swap(diag[i], lower[i + 1]);
      std::swap(upper[i], diag[i + 1]);
      std::swap(upper2[i], upper[i + 1]);
      std::swap(rhs[i], rhs[i + 1]);
    }
    if (diag[i] == Real{0}) {
      throw std::runtime_error{"tridiagonal_solve_pivoting: singular system"};
    }
    const Real factor = lower[i + 1] / diag[i];
    diag[i + 1]  -= factor * upper[i];
    upper[i + 1] -= factor * upper2[i];
    rhs[i + 1]   -= factor * rhs[i];
  }

  if (diag[n - 1] == Real{0}) {
    throw std::runtime_error{"tridiagonal_solve_pivoting: singular system"};
  }
  rhs[n - 1] /= diag[n - 1];
  if (n == 1) return;

  rhs[n - 2] = (rhs[n - 2] - upper[n - 2] * rhs[n - 1]) / diag[n - 2];
  for (std::size_t i = n - 2; i-- > 0;) {
    rhs[i] = (rhs[i] - upper[i] * rhs[i + 1] - upper2[i] * rhs[i + 2])
           / diag[i];
  }
}

// Sixth-order compact first derivative of `f` along a line of `n` uniformly
// spaced points, spacing `h`. Both ends use the one-sided closure, so the line
// must satisfy n >= min_line_extent.
inline std::vector<Real> first_derivative(const std::vector<Real>& f, Real h) {
  const std::size_t n = f.size();
  if (static_cast<int>(n) < min_line_extent) {
    throw std::invalid_argument{
        "pade::first_derivative: line shorter than min_line_extent"};
  }
  using I = FirstDerivativeInterior;
  using B = FirstDerivativeBoundary;

  std::vector<Real> lower(n, Real{0}), diag(n, Real{1}), upper(n, Real{0}),
      rhs(n, Real{0});

  // Low-end closure: f'_0 + alpha f'_1 = (1/h) sum w[s] f_s
  upper[0] = B::alpha;
  Real acc = Real{0};
  for (int s = 0; s < B::stencil_width; ++s) acc += B::w[s] * f[s];
  rhs[0] = acc / h;

  // High-end closure: mirrored stencil with negated weights (the first
  // derivative is odd under reflection). Verified to retain sixth order.
  lower[n - 1] = B::alpha;
  acc = Real{0};
  for (int s = 0; s < B::stencil_width; ++s) {
    acc += -B::w[s] * f[n - 1 - static_cast<std::size_t>(s)];
  }
  rhs[n - 1] = acc / h;

  // Node 1 and node n-2: sixth-order one-sided rows. A fourth-order compact
  // pair here silently degrades node 0 to ~3.5 order (see the note on
  // FirstDerivativeNearBoundary).
  {
    using N = FirstDerivativeNearBoundary;
    lower[1] = N::alpha_lo;
    upper[1] = N::alpha_hi;
    Real s = Real{0};
    for (int k = 0; k < N::stencil_width; ++k) {
      s += N::w[k] * f[static_cast<std::size_t>(1 + N::stencil_lo + k)];
    }
    rhs[1] = s / h;

    // Mirror at node n-2: reflect the stencil and negate (odd parity).
    const std::size_t m = n - 2;
    lower[m] = N::alpha_hi;
    upper[m] = N::alpha_lo;
    s = Real{0};
    for (int k = 0; k < N::stencil_width; ++k) {
      s += -N::w[k] * f[m - static_cast<std::size_t>(N::stencil_lo + k)];
    }
    rhs[m] = s / h;
  }

  for (std::size_t i = 2; i + 2 < n; ++i) {
    lower[i] = I::alpha;
    upper[i] = I::alpha;
    rhs[i]   = I::a * (f[i + 1] - f[i - 1]) / (Real{2} * h)
             + I::b * (f[i + 2] - f[i - 2]) / (Real{4} * h);
  }

  tridiagonal_solve_pivoting(lower, diag, upper, rhs);
  return rhs;
}

// Sixth-order compact second derivative, same conventions.
inline std::vector<Real> second_derivative(const std::vector<Real>& f, Real h) {
  const std::size_t n = f.size();
  if (static_cast<int>(n) < min_line_extent) {
    throw std::invalid_argument{
        "pade::second_derivative: line shorter than min_line_extent"};
  }
  using I = SecondDerivativeInterior;
  using B = SecondDerivativeBoundary;
  const Real h2 = h * h;

  std::vector<Real> lower(n, Real{0}), diag(n, Real{1}), upper(n, Real{0}),
      rhs(n, Real{0});

  upper[0] = B::alpha;
  Real acc = Real{0};
  for (int s = 0; s < B::stencil_width; ++s) acc += B::w[s] * f[s];
  rhs[0] = acc / h2;

  // High end: mirrored stencil, weights NOT negated (the second derivative is
  // even under reflection). Verified to retain sixth order.
  lower[n - 1] = B::alpha;
  acc = Real{0};
  for (int s = 0; s < B::stencil_width; ++s) {
    acc += B::w[s] * f[n - 1 - static_cast<std::size_t>(s)];
  }
  rhs[n - 1] = acc / h2;

  {
    using N = SecondDerivativeNearBoundary;
    lower[1] = N::alpha_lo;
    upper[1] = N::alpha_hi;
    Real s = Real{0};
    for (int k = 0; k < N::stencil_width; ++k) {
      s += N::w[k] * f[static_cast<std::size_t>(1 + N::stencil_lo + k)];
    }
    rhs[1] = s / h2;

    // Mirror at node n-2: reflect the stencil, weights unchanged (even parity).
    const std::size_t m = n - 2;
    lower[m] = N::alpha_hi;
    upper[m] = N::alpha_lo;
    s = Real{0};
    for (int k = 0; k < N::stencil_width; ++k) {
      s += N::w[k] * f[m - static_cast<std::size_t>(N::stencil_lo + k)];
    }
    rhs[m] = s / h2;
  }

  for (std::size_t i = 2; i + 2 < n; ++i) {
    lower[i] = I::alpha;
    upper[i] = I::alpha;
    rhs[i]   = I::a * (f[i + 1] - Real{2} * f[i] + f[i - 1]) / h2
             + I::b * (f[i + 2] - Real{2} * f[i] + f[i - 2]) / (Real{4} * h2);
  }

  tridiagonal_solve_pivoting(lower, diag, upper, rhs);
  return rhs;
}

}  // namespace quasar::numerics::pade
