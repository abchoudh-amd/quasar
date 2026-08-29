#pragma once

// Deterministic spectral condition estimate for a real symmetric matrix.
//
// The matrix is interpreted in LAPACK column-major storage and only the
// selected triangle is read.  For a nonsingular symmetric matrix the 2-norm
// condition number is exactly
//
//   max_i |lambda_i| / min_i |lambda_i|.
//
// The estimate is intended for assembled energy, inertia, and shift-invert
// matrices.  `digits_lost` is log10(condition_estimate); callers can therefore
// reject a marginal result whose expected loss exceeds their precision budget.

#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/generalized_eigensolver.hpp"

namespace quasar::numerics {

enum class SymmetricConditionStatus {
  success,
  numerically_singular,
  failed_to_converge,
  invalid_solver_argument,
};

struct SymmetricConditionEstimate {
  SymmetricConditionStatus status{SymmetricConditionStatus::success};
  int order{0};
  int solver_info{0};
  int unconverged_off_diagonal_count{0};
  int invalid_argument_position{0};

  Real smallest_absolute_eigenvalue{0};
  Real largest_absolute_eigenvalue{0};
  Real condition_estimate{0};
  Real digits_lost{0};

  [[nodiscard]] bool ok() const noexcept {
    return status == SymmetricConditionStatus::success;
  }
};

// Computes a dense spectral condition estimate on the active HIP device and
// synchronizes `stream` before returning the scalar result.  `relative_floor`
// sets the numerical-singularity threshold as
//
//   min |lambda| <= relative_floor * max |lambda|.
//
// Passing zero selects `epsilon * order`.  The input is not modified.
[[nodiscard]] SymmetricConditionEstimate estimate_symmetric_condition(
    const backend::DeviceBuffer<Real>& matrix,
    int order,
    MatrixTriangle triangle = MatrixTriangle::lower,
    Real relative_floor = Real{0},
    backend::stream_t stream = nullptr);

}  // namespace quasar::numerics
