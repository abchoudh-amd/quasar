#include "quasar/numerics/radial_moments.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace quasar::numerics {
namespace {

using Vector = std::array<long double, kMaxRadialStencilWidth>;
using Matrix =
    std::array<std::array<long double, kMaxRadialStencilWidth>,
               kMaxRadialStencilWidth>;

long double integer_power(long double x, int exponent) {
  long double result = 1.0L;
  while (exponent > 0) {
    if ((exponent & 1) != 0) result *= x;
    x *= x;
    exponent >>= 1;
  }
  return result;
}

long double binomial(int n, int k) {
  if (k < 0 || k > n) return 0.0L;
  if (k > n - k) k = n - k;
  long double value = 1.0L;
  for (int j = 1; j <= k; ++j) {
    value *= static_cast<long double>(n - k + j);
    value /= static_cast<long double>(j);
  }
  return value;
}

long double power_integral(int exponent, long double lo, long double hi) {
  const int antiderivative_exponent = exponent + 1;
  return (integer_power(hi, antiderivative_exponent)
          - integer_power(lo, antiderivative_exponent))
         / static_cast<long double>(antiderivative_exponent);
}

// Integrate (x-origin)^m |x| over a unit-width cell centered at cell_rho.
// Expanding around the cell center keeps the calculation well scaled when the
// global radius is large, unlike subtracting two nearly equal powers of x.
long double shifted_weighted_cell_integral(
    long double cell_rho, long double origin, int m) {
  const long double displacement = cell_rho - origin;
  const auto integrate_segment = [&](long double lo, long double hi,
                                     long double radial_sign) {
    long double value = 0.0L;
    for (int p = 0; p <= m; ++p) {
      const long double coefficient =
          binomial(m, p) * integer_power(displacement, m - p);
      const long double weighted_power =
          cell_rho * power_integral(p, lo, hi)
          + power_integral(p + 1, lo, hi);
      value += radial_sign * coefficient * weighted_power;
    }
    return value;
  };

  constexpr long double lo = -0.5L;
  constexpr long double hi = 0.5L;
  if (cell_rho >= 0.5L) return integrate_segment(lo, hi, 1.0L);
  if (cell_rho <= -0.5L) return integrate_segment(lo, hi, -1.0L);

  const long double axis = -cell_rho;
  return integrate_segment(lo, axis, -1.0L)
         + integrate_segment(axis, hi, 1.0L);
}

long double shifted_normalized_cell_moment(
    long double cell_rho, long double origin, int m) {
  const long double volume =
      shifted_weighted_cell_integral(cell_rho, origin, 0);
  if (!(volume > 0.0L) || !std::isfinite(volume)) {
    throw std::runtime_error("radial cell has invalid |r|-weighted volume");
  }
  return shifted_weighted_cell_integral(cell_rho, origin, m) / volume;
}

Vector gaussian_solve(Matrix matrix, Vector rhs, int width) {
  for (int column = 0; column < width; ++column) {
    int pivot = column;
    long double pivot_magnitude = std::fabs(matrix[column][column]);
    for (int row = column + 1; row < width; ++row) {
      const long double candidate = std::fabs(matrix[row][column]);
      if (candidate > pivot_magnitude) {
        pivot = row;
        pivot_magnitude = candidate;
      }
    }
    if (!(pivot_magnitude > 0.0L) || !std::isfinite(pivot_magnitude)) {
      throw std::runtime_error("singular radial moment system");
    }
    if (pivot != column) {
      std::swap(matrix[pivot], matrix[column]);
      std::swap(rhs[pivot], rhs[column]);
    }

    for (int row = column + 1; row < width; ++row) {
      const long double multiplier =
          matrix[row][column] / matrix[column][column];
      matrix[row][column] = 0.0L;
      for (int k = column + 1; k < width; ++k) {
        matrix[row][k] = std::fma(
            -multiplier, matrix[column][k], matrix[row][k]);
      }
      rhs[row] = std::fma(-multiplier, rhs[column], rhs[row]);
    }
  }

  Vector solution{};
  for (int row = width - 1; row >= 0; --row) {
    long double value = rhs[row];
    for (int k = row + 1; k < width; ++k) {
      value = std::fma(-matrix[row][k], solution[k], value);
    }
    solution[row] = value / matrix[row][row];
    if (!std::isfinite(solution[row])) {
      throw std::runtime_error("non-finite radial moment solution");
    }
  }
  return solution;
}

Real in_order_sum(const Real* values, int width) {
  Real sum = 0.0;
  for (int k = 0; k < width; ++k) sum += values[k];
  return sum;
}

void normalize_binary64_row(RadialStencilRow& row, const Vector& solution) {
  long double sum = 0.0L;
  for (int k = 0; k < row.width; ++k) sum += solution[k];
  if (sum == 0.0L || !std::isfinite(sum)) {
    throw std::runtime_error("radial moment row has invalid constant mode");
  }
  for (int k = 0; k < row.width; ++k) {
    row.c[k] = static_cast<Real>(solution[k] / sum);
    if (!std::isfinite(row.c[k])) {
      throw std::runtime_error("radial moment row is not representable");
    }
  }

  // Make the binary64 partition of unity an invariant, rather than merely a
  // tolerance.  With the stencils supported here the prefix is close to one,
  // so Sterbenz-exact subtraction makes the final in-order addition exactly
  // one.  The small correction is no larger than ordinary conversion error.
  Real prefix = 0.0;
  for (int k = 0; k + 1 < row.width; ++k) prefix += row.c[k];
  row.c[row.width - 1] = Real{1} - prefix;
  if (in_order_sum(row.c, row.width) != Real{1}) {
    throw std::runtime_error(
        "radial moment row cannot represent an exact partition of unity");
  }
}

}  // namespace

long double normalized_cell_moment(long double rho, int m) {
  if (!std::isfinite(rho)) {
    throw std::invalid_argument("radial cell coordinate must be finite");
  }
  if (m < 0) {
    throw std::invalid_argument("radial moment degree must be non-negative");
  }
  return shifted_normalized_cell_moment(rho, 0.0L, m);
}

RadialStencilRow solve_radial_row(
    long double rho_anchor, int width, int offset,
    RadialMomentTarget target, long double node_xi) {
  if (!std::isfinite(rho_anchor) || !std::isfinite(node_xi)) {
    throw std::invalid_argument("radial stencil coordinates must be finite");
  }
  if (width < 1 || width > kMaxRadialStencilWidth) {
    throw std::invalid_argument("radial stencil width must be in [1, 8]");
  }
  if (target != RadialMomentTarget::point_value
      && target != RadialMomentTarget::cell_average) {
    throw std::invalid_argument("unknown radial moment target");
  }

  Matrix matrix{};
  Vector rhs{};
  for (int m = 0; m < width; ++m) {
    if (target == RadialMomentTarget::point_value) {
      rhs[m] = integer_power(node_xi, m);
      for (int k = 0; k < width; ++k) {
        const long double cell_rho =
            rho_anchor + static_cast<long double>(offset + k);
        matrix[m][k] = shifted_normalized_cell_moment(
            cell_rho, rho_anchor, m);
      }
    } else {
      rhs[m] = shifted_normalized_cell_moment(
          rho_anchor, rho_anchor, m);
      for (int k = 0; k < width; ++k) {
        const long double point =
            static_cast<long double>(offset + k) + node_xi;
        matrix[m][k] = integer_power(point, m);
      }
    }
  }

  Vector solution = gaussian_solve(matrix, rhs, width);

  // One long-double iterative-refinement step, using the unmodified moment
  // system to recover the defect left by elimination.
  Vector defect{};
  for (int m = 0; m < width; ++m) {
    defect[m] = rhs[m];
    for (int k = 0; k < width; ++k) {
      defect[m] = std::fma(-matrix[m][k], solution[k], defect[m]);
    }
  }
  const Vector correction = gaussian_solve(matrix, defect, width);
  for (int k = 0; k < width; ++k) solution[k] += correction[k];

  RadialStencilRow row{};
  row.width = width;
  row.offset = offset;
  normalize_binary64_row(row, solution);

  long double residual = 0.0L;
  for (int m = 0; m < width; ++m) {
    long double reconstructed = 0.0L;
    for (int k = 0; k < width; ++k) {
      reconstructed = std::fma(
          static_cast<long double>(row.c[k]), matrix[m][k], reconstructed);
    }
    residual = std::max(residual, std::fabs(reconstructed - rhs[m]));
  }
  row.residual = static_cast<Real>(residual);
  return row;
}

RadialStencilRow radial_gauss_weights(
    long double rho_anchor, int count, const Real* node_xi,
    const Real* cartesian_weights) {
  if (!std::isfinite(rho_anchor)) {
    throw std::invalid_argument("radial Gauss coordinate must be finite");
  }
  if (count < 1 || count > kMaxRadialStencilWidth) {
    throw std::invalid_argument("radial Gauss count must be in [1, 8]");
  }
  if (node_xi == nullptr || cartesian_weights == nullptr) {
    throw std::invalid_argument("radial Gauss arrays must not be null");
  }

  Vector weighted{};
  long double normalization = 0.0L;
  for (int q = 0; q < count; ++q) {
    const long double node = static_cast<long double>(node_xi[q]);
    const long double weight =
        static_cast<long double>(cartesian_weights[q]);
    if (!std::isfinite(node) || !std::isfinite(weight)) {
      throw std::invalid_argument("radial Gauss rule must be finite");
    }
    weighted[q] = weight * std::fabs(rho_anchor + node);
    normalization += weighted[q];
  }
  if (!(normalization > 0.0L) || !std::isfinite(normalization)) {
    throw std::runtime_error("radial Gauss rule has invalid normalization");
  }
  for (int q = 0; q < count; ++q) weighted[q] /= normalization;

  RadialStencilRow row{};
  row.width = count;
  normalize_binary64_row(row, weighted);

  long double residual = 0.0L;
  const int exact_degree = 2 * count - 2;
  for (int m = 0; m <= exact_degree; ++m) {
    long double quadrature = 0.0L;
    for (int q = 0; q < count; ++q) {
      quadrature = std::fma(
          static_cast<long double>(row.c[q]),
          integer_power(static_cast<long double>(node_xi[q]), m),
          quadrature);
    }
    const long double exact = shifted_normalized_cell_moment(
        rho_anchor, rho_anchor, m);
    residual = std::max(residual, std::fabs(quadrature - exact));
  }
  row.residual = static_cast<Real>(residual);
  return row;
}

}  // namespace quasar::numerics
