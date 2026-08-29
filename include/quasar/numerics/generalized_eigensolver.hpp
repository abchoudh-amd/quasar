#pragma once

// Dense symmetric-definite generalized eigensolver on the active HIP device.
//
// Matrices use LAPACK/BLAS column-major storage.  The solve is the type-1
// problem
//
//     A x = lambda B x,
//
// with A symmetric and B symmetric positive definite.  Only the selected
// triangle is read.  hipSOLVER overwrites both matrix arguments, so the public
// const inputs are copied device-to-device before the library call.  On
// success, eigenvalues are ascending and eigenvector j occupies column j of
// `eigenvectors`; the vectors are normalized so V^T B V = I.

#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"

namespace quasar::numerics {

enum class MatrixTriangle {
  lower,
  upper,
};

// Numerical status reported by LAPACK-compatible `info` from
// hipsolverDnDsygvd.  Runtime/library failures still throw: these values are
// reserved for mathematically meaningful outcomes callers may need to score.
enum class GeneralizedEigenStatus {
  success,
  failed_to_converge,
  mass_not_positive_definite,
  invalid_solver_argument,
};

struct GeneralizedEigenResult {
  GeneralizedEigenStatus status{GeneralizedEigenStatus::success};
  int order{0};

  // Exact LAPACK-compatible value returned through hipSOLVER's devInfo.
  int solver_info{0};

  // Populated for the corresponding non-success status.  With eigenvectors
  // requested, LAPACK's positive convergence code identifies a failed
  // submatrix rather than a count of unconverged off-diagonal entries.  The
  // exact backend value is therefore retained in `solver_info` without a
  // misleading count interpretation.
  int failed_leading_minor_order{0};
  int invalid_argument_position{0};

  backend::DeviceBuffer<Real> eigenvalues{};   // order
  backend::DeviceBuffer<Real> eigenvectors{};  // order x order, column-major

  [[nodiscard]] bool ok() const noexcept {
    return status == GeneralizedEigenStatus::success;
  }
};

// Solves A x = lambda B x on `stream` and synchronizes that stream before
// returning so the numerical status and result buffers are ready to consume.
// A and B must contain exactly order*order values, reside on the same device,
// and be stored column-major.  Invalid shapes/ownership are programming errors
// and throw std::invalid_argument; numerical failures are returned in `status`.
[[nodiscard]] GeneralizedEigenResult solve_generalized_symmetric_eigenproblem(
    const backend::DeviceBuffer<Real>& a,
    const backend::DeviceBuffer<Real>& b,
    int order,
    MatrixTriangle triangle = MatrixTriangle::lower,
    backend::stream_t stream = nullptr);

}  // namespace quasar::numerics
