#pragma once

// Batched dense LU solve for many small independent systems on the active HIP
// device.
//
// This exists because the tree's other dense solvers are the wrong shape for
// the problem. `cholesky.hpp` is a positive-definiteness *test* and performs no
// solve at all, and `block_tridiagonal.hpp` drives hipSOLVER's single-matrix
// getrf/getrs, which is right for one large block and wrong for thousands of
// 8x8 systems: the per-call launch overhead would dwarf the arithmetic. The
// radial moment tables need exactly that second shape -- roughly twenty rows
// per radial index across the padded grid, each a general non-symmetric
// Vandermonde-like system of width at most eight -- so this wraps
// rocSOLVER's strided-batched getrf/getrs, which hipSOLVER does not expose.
//
// Storage is LAPACK column-major with a fixed stride between systems: system
// `s` occupies `matrices[s * order * order ...]` and `rhs[s * order ...]`. Both
// are overwritten: `matrices` becomes the LU factors and `rhs` becomes the
// solution, matching LAPACK.
//
// Determinism. Each system is factored independently with partial pivoting, so
// the result depends only on the values, not on the batch size or the launch
// geometry -- there is no cross-system reduction to reorder. That is a stronger
// guarantee than the deterministic-mode flag hipSOLVER needs for its reducing
// factorizations, and it is why this needs no such flag.

#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"

namespace quasar::numerics {

enum class BatchedLuStatus {
  success,
  singular,
  invalid_solver_argument,
};

struct BatchedLuResult {
  BatchedLuStatus status{BatchedLuStatus::success};
  int order{0};
  int count{0};

  // Index of the first system whose factorization reported a zero pivot, and
  // the one-based column that failed within it. Both zero unless `status` is
  // `singular`.
  int first_singular_system{0};
  int failed_pivot_column{0};

  [[nodiscard]] bool ok() const noexcept {
    return status == BatchedLuStatus::success;
  }
};

// Factors and solves `count` independent `order x order` systems in place.
// Synchronizes `stream` before returning so the status is ready to consume.
//
// `matrices` must hold exactly order*order*count values and `rhs` exactly
// order*count; a mismatched shape is a programming error and throws
// std::invalid_argument. Library failures throw. A singular system is a
// mathematically meaningful outcome and is reported in `status`, with the
// corresponding entries of `rhs` left as rocSOLVER produced them.
[[nodiscard]] BatchedLuResult solve_batched_dense_lu(
    backend::DeviceBuffer<Real>& matrices,
    backend::DeviceBuffer<Real>& rhs,
    int order, int count,
    backend::stream_t stream = nullptr);

}  // namespace quasar::numerics
