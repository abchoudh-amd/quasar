#pragma once

// Pivoted block-Thomas factorization and solve on the active HIP device.
//
// Every dense block is column-major.  Blocks themselves are block-major:
// block k starts at k*block_order*block_order.  A multi-RHS block starts at
// k*block_order*rhs_count and is a column-major block_order x rhs_count matrix,
// so element (k,row,rhs) is stored at
//
//   k*block_order*rhs_count + row + rhs*block_order.
//
// hipSOLVER performs partial pivoting *inside* each diagonal block.  There is
// deliberately no pivoting between adjacent block rows: a globally nonsingular
// matrix can therefore be unsupported when a block Schur complement is
// singular (for example [[0,1],[1,0]] with scalar blocks).  Such a case is
// returned explicitly as `singular_diagonal_block`, with its zero-based block
// and pivot indices, rather than silently producing infinities.

#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"

namespace quasar::numerics {

enum class BlockTridiagonalStatus {
  success,
  singular_diagonal_block,
  invalid_factorization_argument,
  invalid_solve_argument,
  solve_failed,
};

// Reusable block-LU factors.  `upper_reduced` stores D_k^{-1} U_k after the
// intra-block LU solve; `diagonal_lu` and `pivots` retain the hipSOLVER factors.
// Keeping the factors makes repeated shift-invert applications avoid a fresh
// matrix factorization for every Lanczos vector.
struct BlockTridiagonalFactorization {
  BlockTridiagonalStatus status{BlockTridiagonalStatus::success};
  int block_count{0};
  int block_order{0};

  // Raw LAPACK-compatible info values are separate.  In particular, a getrs
  // result never overwrites the getrf result that identified a singular pivot.
  int factorization_info{0};
  int solve_info{0};
  int failure_block{-1};  // zero-based
  int failure_pivot{-1};  // zero-based within failure_block

  backend::DeviceBuffer<Real> lower{};
  backend::DeviceBuffer<Real> diagonal_lu{};
  backend::DeviceBuffer<Real> upper_reduced{};
  backend::DeviceBuffer<int> pivots{};

  [[nodiscard]] bool ok() const noexcept {
    return status == BlockTridiagonalStatus::success;
  }
};

struct BlockTridiagonalSolveResult {
  BlockTridiagonalStatus status{BlockTridiagonalStatus::success};
  int block_count{0};
  int block_order{0};
  int rhs_count{0};

  int factorization_info{0};
  int solve_info{0};
  int failure_block{-1};  // zero-based
  int failure_pivot{-1};  // zero-based within failure_block

  // Block-major multi-RHS layout described above.
  backend::DeviceBuffer<Real> solution{};

  [[nodiscard]] bool ok() const noexcept {
    return status == BlockTridiagonalStatus::success;
  }
};

// Factor the block-tridiagonal matrix with diagonal blocks D_k, lower blocks
// L_k = A_{k+1,k}, and upper blocks U_k = A_{k,k+1}.  Inputs are const and are
// copied device-to-device.  All buffers must have their exact documented sizes
// and reside on one device.
[[nodiscard]] BlockTridiagonalFactorization factor_block_tridiagonal(
    const backend::DeviceBuffer<Real>& lower,
    const backend::DeviceBuffer<Real>& diagonal,
    const backend::DeviceBuffer<Real>& upper,
    int block_count,
    int block_order,
    backend::stream_t stream = nullptr);

// Solve from reusable factors.  `rhs` is not modified.  The returned solution
// is ready when this function returns.
[[nodiscard]] BlockTridiagonalSolveResult solve_block_tridiagonal(
    const BlockTridiagonalFactorization& factorization,
    const backend::DeviceBuffer<Real>& rhs,
    int rhs_count,
    backend::stream_t stream = nullptr);

// One-shot convenience overload preserving both factorization_info and
// solve_info in the returned status object.
[[nodiscard]] BlockTridiagonalSolveResult solve_block_tridiagonal(
    const backend::DeviceBuffer<Real>& lower,
    const backend::DeviceBuffer<Real>& diagonal,
    const backend::DeviceBuffer<Real>& upper,
    const backend::DeviceBuffer<Real>& rhs,
    int block_count,
    int block_order,
    int rhs_count,
    backend::stream_t stream = nullptr);

}  // namespace quasar::numerics
