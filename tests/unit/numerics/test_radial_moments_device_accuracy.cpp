// Accuracy, batch-independence and reproducibility gate for the device radial
// stencil solve.
//
// The host implementation was deleted, and it carried the whole calculation in
// `long double`. This file therefore builds its own `long double` oracle -- the
// same assembly, the same partial-pivoted elimination, the same single
// iterative-refinement step -- and asserts the binary64 device path is no
// meaningfully worse.
//
// That comparison is the substance of the port. Dropping from a 64-bit mantissa
// to a 53-bit one on a Vandermonde system of width up to eight is exactly the
// kind of change that could quietly stop satisfying `RadialTables`'
// 1e-11 acceptance threshold, and a bare tolerance on the coefficients would
// not reveal it. What makes the port safe is the refinement step: it renders
// the result backward stable, so the residual tracks working precision times
// the matrix norm rather than the condition number.

#include "quasar/numerics/radial_cell_moments.hpp"
#include "quasar/numerics/radial_moments.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using quasar::Real;
using quasar::numerics::kMaxRadialStencilWidth;
using quasar::numerics::RadialCellMeasure;
using quasar::numerics::RadialMomentTarget;
using quasar::numerics::RadialRowSpec;
using quasar::numerics::RadialStencilRow;
using quasar::numerics::solve_radial_row;
using quasar::numerics::solve_radial_rows;

using LD = long double;

// The threshold RadialTables enforces on every row it accepts.
constexpr Real kAcceptedResidual = Real{1e-11};

constexpr int W = kMaxRadialStencilWidth;
using Vector = std::array<LD, W>;
using Matrix = std::array<std::array<LD, W>, W>;

LD integer_power(LD x, int exponent) {
  LD result = 1.0L;
  while (exponent > 0) {
    if ((exponent & 1) != 0) result *= x;
    x *= x;
    exponent >>= 1;
  }
  return result;
}

LD binomial(int n, int k) {
  if (k < 0 || k > n) return 0.0L;
  if (k > n - k) k = n - k;
  LD value = 1.0L;
  for (int j = 1; j <= k; ++j) {
    value *= static_cast<LD>(n - k + j);
    value /= static_cast<LD>(j);
  }
  return value;
}

LD power_integral(int exponent, LD lo, LD hi) {
  const int antiderivative = exponent + 1;
  return (integer_power(hi, antiderivative) - integer_power(lo, antiderivative))
         / static_cast<LD>(antiderivative);
}

int measure_power(RadialCellMeasure measure) {
  switch (measure) {
    case RadialCellMeasure::uniform: return 0;
    case RadialCellMeasure::annular: return 1;
    case RadialCellMeasure::angular_momentum: return 2;
  }
  return 0;
}

LD weighted_cell_integral(LD cell_rho, LD origin, int m,
                          RadialCellMeasure measure) {
  const int radial_power = measure_power(measure);
  const LD displacement = cell_rho - origin;
  const auto segment = [&](LD lo, LD hi, LD sign) {
    LD value = 0.0L;
    for (int p = 0; p <= m; ++p) {
      const LD coefficient =
          binomial(m, p) * integer_power(displacement, m - p);
      LD weighted = 0.0L;
      for (int q = 0; q <= radial_power; ++q) {
        weighted += binomial(radial_power, q)
                  * integer_power(cell_rho, radial_power - q)
                  * power_integral(p + q, lo, hi);
      }
      value += sign * coefficient * weighted;
    }
    return value;
  };
  constexpr LD lo = -0.5L;
  constexpr LD hi = 0.5L;
  if (measure != RadialCellMeasure::annular) return segment(lo, hi, 1.0L);
  if (cell_rho >= 0.5L) return segment(lo, hi, 1.0L);
  if (cell_rho <= -0.5L) return segment(lo, hi, -1.0L);
  const LD axis = -cell_rho;
  return segment(lo, axis, -1.0L) + segment(axis, hi, 1.0L);
}

LD normalized_moment(LD cell_rho, LD origin, int m,
                     RadialCellMeasure measure) {
  return weighted_cell_integral(cell_rho, origin, m, measure)
         / weighted_cell_integral(cell_rho, origin, 0, measure);
}

Vector gaussian_solve(Matrix m, Vector rhs, int width) {
  for (int col = 0; col < width; ++col) {
    int pivot = col;
    LD magnitude = std::fabs(m[col][col]);
    for (int row = col + 1; row < width; ++row) {
      const LD candidate = std::fabs(m[row][col]);
      if (candidate > magnitude) {
        pivot = row;
        magnitude = candidate;
      }
    }
    if (pivot != col) {
      std::swap(m[pivot], m[col]);
      std::swap(rhs[pivot], rhs[col]);
    }
    for (int row = col + 1; row < width; ++row) {
      const LD multiplier = m[row][col] / m[col][col];
      m[row][col] = 0.0L;
      for (int k = col + 1; k < width; ++k) {
        m[row][k] = std::fma(-multiplier, m[col][k], m[row][k]);
      }
      rhs[row] = std::fma(-multiplier, rhs[col], rhs[row]);
    }
  }
  Vector solution{};
  for (int row = width - 1; row >= 0; --row) {
    LD value = rhs[row];
    for (int k = row + 1; k < width; ++k) {
      value = std::fma(-m[row][k], solution[k], value);
    }
    solution[row] = value / m[row][row];
  }
  return solution;
}

struct Oracle {
  Real c[W]{};
  Real residual{};
};

Oracle long_double_oracle(const RadialRowSpec& spec) {
  Matrix A{};
  Vector b{};
  for (int m = 0; m < spec.width; ++m) {
    if (spec.target == RadialMomentTarget::point_value) {
      b[m] = integer_power(static_cast<LD>(spec.node_xi), m);
      for (int k = 0; k < spec.width; ++k) {
        A[m][k] = normalized_moment(
            static_cast<LD>(spec.rho_anchor) + (spec.offset + k),
            static_cast<LD>(spec.rho_anchor), m, spec.measure);
      }
    } else {
      b[m] = normalized_moment(static_cast<LD>(spec.rho_anchor),
                               static_cast<LD>(spec.rho_anchor), m,
                               spec.measure);
      for (int k = 0; k < spec.width; ++k) {
        A[m][k] = integer_power(
            static_cast<LD>(spec.offset + k) + static_cast<LD>(spec.node_xi),
            m);
      }
    }
  }

  Vector x = gaussian_solve(A, b, spec.width);
  Vector defect{};
  for (int m = 0; m < spec.width; ++m) {
    defect[m] = b[m];
    for (int k = 0; k < spec.width; ++k) {
      defect[m] = std::fma(-A[m][k], x[k], defect[m]);
    }
  }
  const Vector correction = gaussian_solve(A, defect, spec.width);
  for (int k = 0; k < spec.width; ++k) x[k] += correction[k];

  LD sum = 0.0L;
  for (int k = 0; k < spec.width; ++k) sum += x[k];

  Oracle out;
  for (int k = 0; k < spec.width; ++k) {
    out.c[k] = static_cast<Real>(x[k] / sum);
  }
  Real prefix = Real{0};
  for (int k = 0; k + 1 < spec.width; ++k) prefix += out.c[k];
  out.c[spec.width - 1] = Real{1} - prefix;

  LD residual = 0.0L;
  for (int m = 0; m < spec.width; ++m) {
    LD reconstructed = 0.0L;
    for (int k = 0; k < spec.width; ++k) {
      reconstructed =
          std::fma(static_cast<LD>(out.c[k]), A[m][k], reconstructed);
    }
    residual = std::max(residual, std::fabs(reconstructed - b[m]));
  }
  out.residual = static_cast<Real>(residual);
  return out;
}

// The widths, measures and targets RadialTables actually builds, swept across
// the padded radial grid including its axis ghosts.
std::vector<RadialRowSpec> table_like_specs() {
  const RadialCellMeasure measures[3] = {
      RadialCellMeasure::uniform, RadialCellMeasure::annular,
      RadialCellMeasure::angular_momentum};
  std::vector<RadialRowSpec> specs;
  for (int logical = -4; logical < 64; ++logical) {
    const Real rho = static_cast<Real>(logical) + Real{0.5};
    for (const RadialCellMeasure measure : measures) {
      for (const Real xi : {Real{0.5}, Real{-0.5}, Real{0.2}}) {
        specs.push_back(RadialRowSpec{rho, xi, 5, -2,
                                      RadialMomentTarget::point_value,
                                      measure});
        specs.push_back(RadialRowSpec{rho, xi, 7, -3,
                                      RadialMomentTarget::point_value,
                                      measure});
      }
      specs.push_back(RadialRowSpec{rho, Real{0.5}, 6, -3,
                                    RadialMomentTarget::cell_average, measure});
      specs.push_back(RadialRowSpec{rho, Real{0.5}, 8, -4,
                                    RadialMomentTarget::cell_average, measure});
      specs.push_back(RadialRowSpec{rho, Real{0.5}, 2, 0,
                                    RadialMomentTarget::point_value, measure});
    }
  }
  return specs;
}

}  // namespace

TEST(RadialMomentsDeviceAccuracy, ResidualsStayInsideTheAcceptanceThreshold) {
  const std::vector<RadialRowSpec> specs = table_like_specs();
  const std::vector<RadialStencilRow> rows = solve_radial_rows(specs);
  ASSERT_EQ(rows.size(), specs.size());

  Real worst = Real{0};
  for (std::size_t i = 0; i < rows.size(); ++i) {
    ASSERT_TRUE(std::isfinite(rows[i].residual)) << "row " << i;
    worst = std::max(worst, rows[i].residual);
  }
  // Not merely "under the threshold": the margin is the finding that justified
  // moving this calculation off long double at all.
  EXPECT_LT(worst, kAcceptedResidual)
      << "worst binary64 residual " << worst << " against a "
      << kAcceptedResidual << " acceptance threshold";
}

TEST(RadialMomentsDeviceAccuracy, IsNoWorseThanTheLongDoubleOracle) {
  const std::vector<RadialRowSpec> specs = table_like_specs();
  const std::vector<RadialStencilRow> rows = solve_radial_rows(specs);
  ASSERT_EQ(rows.size(), specs.size());

  Real worst_device = Real{0};
  Real worst_oracle = Real{0};
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const Oracle oracle = long_double_oracle(specs[i]);
    worst_device = std::max(worst_device, rows[i].residual);
    worst_oracle = std::max(worst_oracle, oracle.residual);

    // The coefficients themselves must agree to the working precision of the
    // system, scaled by the size of the coefficients: a high-order stencil has
    // alternating signs and entries far larger than one.
    Real scale = Real{0};
    for (int k = 0; k < specs[i].width; ++k) scale += std::abs(oracle.c[k]);
    for (int k = 0; k < specs[i].width; ++k) {
      EXPECT_NEAR(rows[i].c[k], oracle.c[k], Real{1e-9} * scale)
          << "row " << i << " coefficient " << k;
    }
  }

  // Binary64 is allowed to be worse than long double -- it has eleven fewer
  // mantissa bits -- but not by the orders of magnitude that would put it near
  // the acceptance threshold. A factor of thirty is generous against the
  // measured ~1.7 and still fails loudly if the refinement step is ever lost.
  EXPECT_LT(worst_device, Real{30} * worst_oracle + kAcceptedResidual / Real{100})
      << "device " << worst_device << " vs long double " << worst_oracle;
}

TEST(RadialMomentsDeviceAccuracy, BatchingDoesNotChangeAnyRow) {
  // Every system in the batch is factored independently, so a row's answer must
  // not depend on what it was batched with. This is what lets RadialTables
  // solve the whole grid in one call and still match a single-row solve.
  const std::vector<RadialRowSpec> specs = table_like_specs();
  const std::vector<RadialStencilRow> batched = solve_radial_rows(specs);
  ASSERT_EQ(batched.size(), specs.size());

  for (std::size_t i = 0; i < specs.size(); i += 37) {
    const RadialStencilRow single =
        solve_radial_row(specs[i].rho_anchor, specs[i].width, specs[i].offset,
                         specs[i].target, specs[i].measure, specs[i].node_xi);
    ASSERT_EQ(single.width, batched[i].width) << "row " << i;
    for (int k = 0; k < single.width; ++k) {
      EXPECT_EQ(single.c[k], batched[i].c[k]) << "row " << i << " entry " << k;
    }
    EXPECT_EQ(single.residual, batched[i].residual) << "row " << i;
  }
}

TEST(RadialMomentsDeviceAccuracy, RepeatedSolvesAreBitwiseIdentical) {
  const std::vector<RadialRowSpec> specs = table_like_specs();
  const std::vector<RadialStencilRow> first = solve_radial_rows(specs);
  const std::vector<RadialStencilRow> second = solve_radial_rows(specs);
  ASSERT_EQ(first.size(), second.size());
  for (std::size_t i = 0; i < first.size(); ++i) {
    for (int k = 0; k < first[i].width; ++k) {
      EXPECT_EQ(first[i].c[k], second[i].c[k]) << "row " << i;
    }
    EXPECT_EQ(first[i].residual, second[i].residual) << "row " << i;
  }
}

TEST(RadialMomentsDeviceAccuracy, RowsArePartitionsOfUnityExactly) {
  // The in-order binary64 sum being exactly one is an invariant the
  // reconstruction relies on, not a tolerance. It survives the move to the
  // device because the final coefficient is still formed as 1 - prefix.
  const std::vector<RadialRowSpec> specs = table_like_specs();
  const std::vector<RadialStencilRow> rows = solve_radial_rows(specs);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    Real sum = Real{0};
    for (int k = 0; k < rows[i].width; ++k) sum += rows[i].c[k];
    EXPECT_EQ(sum, Real{1}) << "row " << i;
  }
}
