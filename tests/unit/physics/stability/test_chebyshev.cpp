// Chebyshev spectral basis: nodes, quadrature, and differentiation.
//
// Everything here is checked against exact analytic answers, which is possible
// because the basis has properties that hold to machine precision rather than
// to a discretization error:
//
//   * A constant must differentiate to EXACTLY zero. This is what the
//     negative-sum trick buys, and it is the property the closed-form diagonal
//     loses at high order.
//   * Clenshaw-Curtis quadrature is exact for polynomials up to the node count,
//     so integrating one must give the analytic value to rounding.
//   * Differentiating a polynomial of degree <= order is exact.
//
// Beyond those, spectral convergence on a non-polynomial function is checked by
// refinement, since that is the actual reason for using this basis.

#include "quasar/backend/memory.hpp"
#include "quasar/physics/stability/kernels.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::stability::ChebyshevBasis;
using quasar::stability::RadialDomains;

struct Basis {
  RadialDomains domains{};
  ChebyshevBasis basis{};
  std::vector<Real> nodes;
  std::vector<Real> weights;
  std::vector<Real> diff;
  int n_nodes{0};
  int n_domains{0};

  Basis(int order, const std::vector<Real>& breakpoints) {
    domains.n_domains = static_cast<int>(breakpoints.size()) - 1;
    for (std::size_t k = 0; k < breakpoints.size(); ++k) {
      domains.breakpoints[k] = breakpoints[k];
    }
    DeviceBuffer<RadialDomains> d_dom{1};
    d_dom.copy_from_host(&domains, 1);

    basis.resize(order, domains.n_domains);
    quasar::stability::launch_build_chebyshev_basis(d_dom.device_ptr(), basis,
                                                    nullptr);
    quasar::backend::device_synchronize(nullptr);

    n_nodes = basis.n_nodes;
    n_domains = domains.n_domains;
    nodes.resize(basis.nodes.size());
    weights.resize(basis.weights.size());
    diff.resize(basis.diff.size());
    basis.nodes.copy_to_host(nodes.data(), nodes.size());
    basis.weights.copy_to_host(weights.data(), weights.size());
    basis.diff.copy_to_host(diff.data(), diff.size());
  }

  Real node(int d, int i) const {
    return nodes[static_cast<std::size_t>(d) * n_nodes + i];
  }
  Real weight(int d, int i) const {
    return weights[static_cast<std::size_t>(d) * n_nodes + i];
  }
  Real D(int d, int i, int j) const {
    return diff[static_cast<std::size_t>(d) * n_nodes * n_nodes
                + static_cast<std::size_t>(i) * n_nodes + j];
  }

  // Apply the differentiation matrix of domain d to sampled values.
  std::vector<Real> differentiate(int d, const std::vector<Real>& f) const {
    std::vector<Real> out(static_cast<std::size_t>(n_nodes), Real{0});
    for (int i = 0; i < n_nodes; ++i) {
      Real acc = Real{0};
      for (int j = 0; j < n_nodes; ++j) acc += D(d, i, j) * f[static_cast<std::size_t>(j)];
      out[static_cast<std::size_t>(i)] = acc;
    }
    return out;
  }
};

std::vector<Real> sample(const Basis& b, int d, Real (*f)(Real)) {
  std::vector<Real> v(static_cast<std::size_t>(b.n_nodes));
  for (int i = 0; i < b.n_nodes; ++i) v[static_cast<std::size_t>(i)] = f(b.node(d, i));
  return v;
}

// Nodes must include both endpoints exactly, since interface continuity is
// imposed on shared node values rather than by interpolating.
TEST(Chebyshev, LobattoNodesHitTheSubintervalEndpoints) {
  const Basis b{16, {Real{0.1}, Real{0.4}, Real{0.9}}};
  ASSERT_EQ(b.n_domains, 2);

  // Node 0 is at +1 on the reference interval, which maps to the UPPER end.
  EXPECT_DOUBLE_EQ(b.node(0, 0), Real{0.4});
  EXPECT_DOUBLE_EQ(b.node(0, b.n_nodes - 1), Real{0.1});
  EXPECT_DOUBLE_EQ(b.node(1, 0), Real{0.9});
  EXPECT_DOUBLE_EQ(b.node(1, b.n_nodes - 1), Real{0.4});

  // Every node lies inside its subinterval.
  for (int d = 0; d < b.n_domains; ++d) {
    for (int i = 0; i < b.n_nodes; ++i) {
      EXPECT_GE(b.node(d, i), b.domains.breakpoints[d] - Real{1e-15});
      EXPECT_LE(b.node(d, i), b.domains.breakpoints[d + 1] + Real{1e-15});
    }
  }
}

// The property the negative-sum trick exists to guarantee.
//
// The bound is RELATIVE to the row magnitude, not absolute, because
// differentiation matrix entries grow like order^2 divided by the interval
// width -- at order 96 on a 0.65-wide interval the largest entry is ~1e4, so an
// absolute bound would be measuring the wrong thing.
//
// What matters is that the relative residual does not GROW with order. That is
// exactly what the closed-form diagonal fails at: it is a difference of large
// nearly-equal quantities, so its cancellation error worsens as the order
// rises, and the operator stops annihilating constants precisely where spectral
// accuracy was the point.
TEST(Chebyshev, DifferentiatesAConstantToZero) {
  for (const int order : {8, 32, 64, 96}) {
    const Basis b{order, {Real{0.2}, Real{0.85}}};
    const std::vector<Real> ones(static_cast<std::size_t>(b.n_nodes), Real{1});
    const auto d = b.differentiate(0, ones);

    for (int i = 0; i < b.n_nodes; ++i) {
      Real row_scale = Real{0};
      for (int j = 0; j < b.n_nodes; ++j) row_scale += std::abs(b.D(0, i, j));
      EXPECT_LT(std::abs(d[static_cast<std::size_t>(i)]),
                Real{1e-13} * row_scale)
          << "order " << order << ", node " << i << ", row scale " << row_scale;
    }
  }
}

// Rows of a differentiation matrix must sum to zero -- the same statement, made
// directly against the matrix rather than through an application.
TEST(Chebyshev, DifferentiationMatrixRowsSumToZero) {
  const Basis b{48, {Real{0.05}, Real{0.5}, Real{0.95}}};
  for (int d = 0; d < b.n_domains; ++d) {
    for (int i = 0; i < b.n_nodes; ++i) {
      Real sum = Real{0};
      Real scale = Real{0};
      for (int j = 0; j < b.n_nodes; ++j) {
        sum += b.D(d, i, j);
        scale += std::abs(b.D(d, i, j));
      }
      EXPECT_LT(std::abs(sum), Real{1e-13} * scale)
          << "domain " << d << " row " << i << " (scale " << scale << ")";
    }
  }
}

// A polynomial of degree <= order is differentiated exactly, up to rounding.
TEST(Chebyshev, DifferentiatesPolynomialsExactly) {
  const Basis b{20, {Real{0.3}, Real{1.2}}};

  // f = 3x^4 - 2x^3 + x - 5,  f' = 12x^3 - 6x^2 + 1
  const auto f = sample(b, 0, [](Real x) {
    return Real{3} * x * x * x * x - Real{2} * x * x * x + x - Real{5};
  });
  const auto got = b.differentiate(0, f);

  for (int i = 0; i < b.n_nodes; ++i) {
    const Real x = b.node(0, i);
    const Real want = Real{12} * x * x * x - Real{6} * x * x + Real{1};
    EXPECT_NEAR(got[static_cast<std::size_t>(i)], want,
                Real{1e-10} * (Real{1} + std::abs(want)))
        << "node " << i << " at x = " << x;
  }
}

// Clenshaw-Curtis is exact for polynomials up to the node count.
TEST(Chebyshev, QuadratureIsExactForPolynomials) {
  const Basis b{24, {Real{0.2}, Real{0.7}, Real{1.4}}};

  // Integrate f = x^3 - 2x + 1 over the union of the subintervals, which is
  // [0.2, 1.4]. Analytic: [x^4/4 - x^2 + x].
  const auto antiderivative = [](Real x) {
    return x * x * x * x / Real{4} - x * x + x;
  };
  const Real want = antiderivative(Real{1.4}) - antiderivative(Real{0.2});

  Real got = Real{0};
  for (int d = 0; d < b.n_domains; ++d) {
    for (int i = 0; i < b.n_nodes; ++i) {
      const Real x = b.node(d, i);
      got += b.weight(d, i) * (x * x * x - Real{2} * x + Real{1});
    }
  }
  EXPECT_NEAR(got, want, Real{1e-13} * std::abs(want));
}

// Weights must be positive and sum to the subinterval length. A sign error
// would still integrate polynomials correctly on a symmetric interval, so this
// is checked separately.
TEST(Chebyshev, QuadratureWeightsArePositiveAndSumToLength) {
  const Basis b{32, {Real{0.1}, Real{0.6}, Real{1.5}}};
  for (int d = 0; d < b.n_domains; ++d) {
    Real sum = Real{0};
    for (int i = 0; i < b.n_nodes; ++i) {
      EXPECT_GT(b.weight(d, i), Real{0}) << "domain " << d << " node " << i;
      sum += b.weight(d, i);
    }
    const Real length = b.domains.breakpoints[d + 1] - b.domains.breakpoints[d];
    EXPECT_NEAR(sum, length, Real{1e-13} * length) << "domain " << d;
  }
}

// The reason for using this basis at all: error on a smooth non-polynomial
// function must fall faster than any power of the order.
TEST(Chebyshev, DifferentiationConvergesSpectrally) {
  const auto error_at = [](int order) {
    const Basis b{order, {Real{0.3}, Real{1.1}}};
    const auto f = sample(b, 0, [](Real x) { return std::exp(std::sin(Real{3} * x)); });
    const auto got = b.differentiate(0, f);

    Real worst = Real{0};
    for (int i = 0; i < b.n_nodes; ++i) {
      const Real x = b.node(0, i);
      const Real want = Real{3} * std::cos(Real{3} * x)
                      * std::exp(std::sin(Real{3} * x));
      worst = std::max(worst, std::abs(got[static_cast<std::size_t>(i)] - want));
    }
    return worst;
  };

  const Real e8 = error_at(8);
  const Real e16 = error_at(16);
  const Real e24 = error_at(24);

  // Spectral convergence means each step gains orders of magnitude, not a
  // constant factor. Requiring 100x per 8 orders is far short of what actually
  // happens but far beyond any algebraic rate.
  EXPECT_GT(e8 / e16, Real{100}) << e8 << " -> " << e16;
  EXPECT_GT(e16 / e24, Real{100}) << e16 << " -> " << e24;
  EXPECT_LT(e24, Real{1e-10}) << "order 24 should already be near machine "
                                 "precision on this function";
}

// Adjacent subintervals must agree at the breakpoint they share, which is what
// makes continuity a condition on node values rather than an interpolation.
TEST(Chebyshev, AdjacentDomainsShareTheirInterfaceNodeExactly) {
  const Basis b{20, {Real{0.05}, Real{0.37}, Real{0.61}, Real{0.98}}};
  ASSERT_EQ(b.n_domains, 3);

  for (int d = 0; d + 1 < b.n_domains; ++d) {
    // Node 0 of a domain is its upper end; the last node is its lower end.
    const Real upper_of_lower = b.node(d, 0);
    const Real lower_of_upper = b.node(d + 1, b.n_nodes - 1);
    EXPECT_EQ(upper_of_lower, lower_of_upper)
        << "interface " << d << " is not represented identically on both sides";
  }
}

}  // namespace
