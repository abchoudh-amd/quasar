// Order verification for the compact (Pade) derivative operators.
//
// The scheme's order is the whole point of choosing a compact operator, so it
// is verified numerically here rather than asserted from the coefficient
// derivation. The test function is deliberately non-polynomial: a polynomial of
// degree <= 6 is differentiated exactly by a sixth-order scheme, so a
// polynomial test would pass even if the coefficients were wrong in a way that
// only shows up on a general smooth function.
//
// The observed rate is measured on the interior and on the boundary rows
// separately. That separation matters: the interior coefficients are classical
// and unlikely to be wrong, whereas the one-sided closures were derived for
// this code and are exactly where an order-reduction bug hides. A test that
// only reports a global norm can be dominated by interior points and pass at
// close to sixth order while the boundary rows are second order.

#include "quasar/numerics/pade_derivative.hpp"
#include "quasar/numerics/pade_line_solve.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

using quasar::Real;

// Non-polynomial manufactured function with O(1) derivatives of every order on
// [0, 1], so no derivative order is accidentally small.
Real mms_f(Real x) { return std::exp(std::sin(Real{2.3} * x + Real{0.4})); }

Real mms_df(Real x) {
  const Real u = Real{2.3} * x + Real{0.4};
  return std::exp(std::sin(u)) * std::cos(u) * Real{2.3};
}

Real mms_d2f(Real x) {
  const Real u = Real{2.3} * x + Real{0.4};
  const Real c = std::cos(u);
  const Real s = std::sin(u);
  return std::exp(s) * (c * c - s) * Real{2.3} * Real{2.3};
}

struct ErrorPair {
  Real interior{0};
  Real boundary{0};
};

// `boundary_rows` counts the rows at each end treated as boundary-influenced:
// the closure row itself plus the reduced-order near-boundary row.
constexpr int kBoundaryRows = 2;

template <class Fn, class Exact>
ErrorPair max_error(int n, Fn&& apply, Exact&& exact) {
  const Real h = Real{1} / static_cast<Real>(n - 1);
  std::vector<Real> f(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    f[static_cast<std::size_t>(i)] = mms_f(static_cast<Real>(i) * h);
  }
  const std::vector<Real> got = apply(f, h);

  ErrorPair err;
  for (int i = 0; i < n; ++i) {
    const Real want = exact(static_cast<Real>(i) * h);
    const Real e = std::abs(got[static_cast<std::size_t>(i)] - want);
    const bool is_boundary = (i < kBoundaryRows) || (i >= n - kBoundaryRows);
    if (is_boundary) {
      err.boundary = std::max(err.boundary, e);
    } else {
      err.interior = std::max(err.interior, e);
    }
  }
  return err;
}

Real observed_order(Real coarse_err, Real fine_err) {
  return std::log2(coarse_err / fine_err);
}

}  // namespace

TEST(PadeDerivative, FirstDerivativeInteriorIsSixthOrder) {
  const auto apply = [](const std::vector<Real>& f, Real h) {
    return quasar::numerics::pade::first_derivative(f, h);
  };
  const auto e1 = max_error(65, apply, mms_df);
  const auto e2 = max_error(129, apply, mms_df);
  const Real rate = observed_order(e1.interior, e2.interior);
  EXPECT_GT(rate, 5.7) << "interior first derivative lost order; coarse="
                       << e1.interior << " fine=" << e2.interior;
}

TEST(PadeDerivative, FirstDerivativeBoundaryIsSixthOrder) {
  const auto apply = [](const std::vector<Real>& f, Real h) {
    return quasar::numerics::pade::first_derivative(f, h);
  };
  const auto e1 = max_error(65, apply, mms_df);
  const auto e2 = max_error(129, apply, mms_df);
  const Real rate = observed_order(e1.boundary, e2.boundary);
  // The one-sided closure is the derived-here piece. A drop to ~4 or ~2 means
  // the closure weights or the high-end mirror parity are wrong.
  EXPECT_GT(rate, 5.5) << "boundary closure lost order; coarse=" << e1.boundary
                       << " fine=" << e2.boundary;
}

TEST(PadeDerivative, SecondDerivativeInteriorIsSixthOrder) {
  const auto apply = [](const std::vector<Real>& f, Real h) {
    return quasar::numerics::pade::second_derivative(f, h);
  };
  const auto e1 = max_error(65, apply, mms_d2f);
  const auto e2 = max_error(129, apply, mms_d2f);
  const Real rate = observed_order(e1.interior, e2.interior);
  EXPECT_GT(rate, 5.7) << "interior second derivative lost order; coarse="
                       << e1.interior << " fine=" << e2.interior;
}

TEST(PadeDerivative, SecondDerivativeBoundaryIsSixthOrder) {
  const auto apply = [](const std::vector<Real>& f, Real h) {
    return quasar::numerics::pade::second_derivative(f, h);
  };
  const auto e1 = max_error(65, apply, mms_d2f);
  const auto e2 = max_error(129, apply, mms_d2f);
  const Real rate = observed_order(e1.boundary, e2.boundary);
  EXPECT_GT(rate, 5.5) << "boundary closure lost order; coarse=" << e1.boundary
                       << " fine=" << e2.boundary;
}

// Constant-exactness. Both operators annihilate a constant exactly in exact
// arithmetic (the closure weights sum to zero, checked symbolically during
// derivation), so the only error here is floating-point.
//
// The tolerance must scale with the operator's own roundoff amplification, not
// be a fixed absolute number. Forming the right-hand side divides by h (first
// derivative) or h^2 (second), and the boundary weights are O(10), so the
// achievable floor grows as h shrinks. Measured behaviour: the second
// derivative sits at a constant 7.2e3 multiple of eps/h^2 across four decades
// of h -- a fixed prefactor with no h-dependent term, which is exactly the
// signature of pure roundoff rather than a truncation error. A fixed 1e-12
// bound would pass at h=1 and fail at h=0.25 while testing nothing about the
// operator.
TEST(PadeDerivative, ConstantFieldHasZeroDerivative) {
  constexpr Real kEps = std::numeric_limits<Real>::epsilon();
  // Empirical amplification factors with generous headroom.
  constexpr Real kFirstFactor  = Real{1e3};
  constexpr Real kSecondFactor = Real{5e4};

  for (const Real h : {Real{1}, Real{0.5}, Real{0.25}, Real{0.0625}}) {
    const int n = 40;
    const std::vector<Real> f(static_cast<std::size_t>(n), Real{3.5});

    const auto d1 = quasar::numerics::pade::first_derivative(f, h);
    const auto d2 = quasar::numerics::pade::second_derivative(f, h);

    const Real tol1 = kFirstFactor * kEps / h;
    const Real tol2 = kSecondFactor * kEps / (h * h);
    for (int i = 0; i < n; ++i) {
      EXPECT_NEAR(d1[static_cast<std::size_t>(i)], Real{0}, tol1)
          << "h=" << h << " i=" << i;
      EXPECT_NEAR(d2[static_cast<std::size_t>(i)], Real{0}, tol2)
          << "h=" << h << " i=" << i;
    }
  }
}

// Linear-exactness. A linear field has constant first derivative and zero
// second derivative; this is the independent check on sum(s*w[s]) == 0 for the
// second-derivative closure.
TEST(PadeDerivative, LinearFieldIsDifferentiatedExactly) {
  const int n = 40;
  const Real h = Real{0.25};
  const Real slope = Real{-1.75};
  std::vector<Real> f(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    f[static_cast<std::size_t>(i)] = slope * static_cast<Real>(i) * h + Real{2};
  }

  const auto d1 = quasar::numerics::pade::first_derivative(f, h);
  const auto d2 = quasar::numerics::pade::second_derivative(f, h);
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(d1[static_cast<std::size_t>(i)], slope, 1e-10) << "i=" << i;
    EXPECT_NEAR(d2[static_cast<std::size_t>(i)], Real{0}, 1e-9) << "i=" << i;
  }
}

// Guard the degeneracy that produced a silently fourth-order scheme during
// derivation: pin the interior coefficients to their sixth-order values so a
// future edit cannot substitute the (also valid, also smooth) fourth-order pair
// without a test failing.
TEST(PadeDerivative, InteriorCoefficientsAreTheSixthOrderFamily) {
  using F = quasar::numerics::pade::FirstDerivativeInterior;
  using S = quasar::numerics::pade::SecondDerivativeInterior;
  EXPECT_DOUBLE_EQ(F::alpha, 1.0 / 3.0);   // not 1/4 (fourth order)
  EXPECT_DOUBLE_EQ(F::a, 14.0 / 9.0);
  EXPECT_DOUBLE_EQ(F::b, 1.0 / 9.0);
  EXPECT_DOUBLE_EQ(S::alpha, 2.0 / 11.0);  // not 1/10 (fourth order)
  EXPECT_DOUBLE_EQ(S::a, 12.0 / 11.0);
  EXPECT_DOUBLE_EQ(S::b, 3.0 / 11.0);
}

TEST(PadeLineSolve, HandlesEmptyAndSingleRowSystems) {
  std::vector<Real> lower;
  std::vector<Real> diag;
  std::vector<Real> upper;
  std::vector<Real> rhs;
  EXPECT_NO_THROW(quasar::numerics::pade::tridiagonal_solve_pivoting(
      lower, diag, upper, rhs));

  lower = {Real{0}};
  diag = {Real{4}};
  upper = {Real{0}};
  rhs = {Real{10}};
  quasar::numerics::pade::tridiagonal_solve_pivoting(lower, diag, upper, rhs);
  ASSERT_EQ(rhs.size(), 1u);
  EXPECT_DOUBLE_EQ(rhs[0], Real{2.5});
}

TEST(PadeLineSolve, RejectsASingularSingleRowSystem) {
  std::vector<Real> lower{Real{0}};
  std::vector<Real> diag{Real{0}};
  std::vector<Real> upper{Real{0}};
  std::vector<Real> rhs{Real{1}};
  EXPECT_THROW(quasar::numerics::pade::tridiagonal_solve_pivoting(
                   lower, diag, upper, rhs),
               std::runtime_error);
}

TEST(PadeLineSolve, SolvesTwoRowsWithPivoting) {
  std::vector<Real> lower{Real{0}, Real{2}};
  std::vector<Real> diag{Real{1}, Real{3}};
  std::vector<Real> upper{Real{1}, Real{0}};
  std::vector<Real> rhs{Real{3}, Real{8}};
  quasar::numerics::pade::tridiagonal_solve_pivoting(lower, diag, upper, rhs);
  EXPECT_DOUBLE_EQ(rhs[0], Real{1});
  EXPECT_DOUBLE_EQ(rhs[1], Real{2});
}

TEST(PadeLineSolve, RejectsInconsistentBandSizes) {
  std::vector<Real> lower{Real{0}};
  std::vector<Real> diag{Real{1}};
  std::vector<Real> upper;
  std::vector<Real> rhs{Real{1}};
  EXPECT_THROW(quasar::numerics::pade::tridiagonal_solve_pivoting(
                   lower, diag, upper, rhs),
               std::invalid_argument);
}

TEST(PadeDerivative, ShortLineIsRejected) {
  const std::vector<Real> f(4, Real{1});
  EXPECT_THROW(quasar::numerics::pade::first_derivative(f, Real{0.1}),
               std::invalid_argument);
  EXPECT_THROW(quasar::numerics::pade::second_derivative(f, Real{0.1}),
               std::invalid_argument);
}
