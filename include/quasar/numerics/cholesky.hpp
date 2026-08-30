#pragma once

// Symmetric positive-definite test by Cholesky factorization on the active HIP
// device.
//
// The matrix uses LAPACK column-major storage and only the selected triangle is
// read.  The input is never modified: the factorization runs on a device-to-
// device copy, because the caller's matrix is generally an oracle that later
// stages still consume.
//
// This exists as a public `numerics` entry point rather than as a private
// helper because consumers outside this module need the check.  The toroidal
// stability assembly, for example, must classify a non-positive-definite
// inertia matrix before it is handed to the generalized eigensolver, and
// reaching into `src/backend/hip/numerics/` to do so would cross a module
// boundary the project treats as implementation-private.
//
// Cost is a single O(order^3) factorization.  When the caller is going to run
// `solve_generalized_symmetric_eigenproblem` on the same matrix as the mass
// term anyway, that solve already reports the same condition through its own
// `mass_not_positive_definite` status; prefer reading it from there instead of
// paying for a second factorization.

#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/generalized_eigensolver.hpp"

namespace quasar::numerics {

enum class CholeskyStatus {
  success,
  not_positive_definite,
  invalid_solver_argument,
};

struct CholeskyResult {
  CholeskyStatus status{CholeskyStatus::success};
  int order{0};

  // Exact LAPACK-compatible value returned through hipSOLVER's devInfo.
  int solver_info{0};

  // One-based order of the first leading principal minor that is not positive
  // definite; zero unless `status` is `not_positive_definite`.
  int failed_leading_minor_order{0};
  int invalid_argument_position{0};

  [[nodiscard]] bool ok() const noexcept {
    return status == CholeskyStatus::success;
  }
};

// Factors a copy of `matrix` and reports whether it is positive definite.
// Synchronizes `stream` before returning so the status is ready to consume.
// `matrix` must contain exactly order*order values; an invalid shape is a
// programming error and throws std::invalid_argument.  Library failures throw;
// the mathematically meaningful outcomes are returned in `status`.
[[nodiscard]] CholeskyResult check_symmetric_positive_definite(
    const backend::DeviceBuffer<Real>& matrix,
    int order,
    MatrixTriangle triangle = MatrixTriangle::lower,
    backend::stream_t stream = nullptr);

}  // namespace quasar::numerics
