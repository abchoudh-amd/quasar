#pragma once

// Pivoted block-Thomas factorization and solve on the active HIP device.
//
// Every dense block is column-major.  Blocks themselves are block-major.  Block
// orders may differ from block to block, so all storage is addressed through
// prefix offsets carried by `BlockPartition`:
//
//   diagonal block k     n_k   x n_k     at partition.diagonal_offset(k)
//   lower block L_k      n_k+1 x n_k     at partition.off_diagonal_offset(k)
//   upper block U_k      n_k   x n_k+1   at partition.off_diagonal_offset(k)
//   right-hand side k    n_k   x rhs     at partition.rhs_offset(k, rhs)
//
// with leading dimensions n_k+1 for L_k and n_k for the diagonal, U_k, and the
// right-hand side.  For the uniform case this reduces to the obvious
// k*block_order*block_order and k*block_order*rhs_count strides, so element
// (k,row,rhs) of a uniform multi-RHS block lives at
//
//   k*block_order*rhs_count + row + rhs*block_order.
//
// -- Why non-uniform blocks exist ----------------------------------------------
// The Chebyshev spectral-element operator this module was written for has
// n_domains*order + 1 radial nodes, because adjacent domains share their common
// Lobatto endpoint.  That count is never a multiple of `order` for order > 1, so
// grouping radial nodes into equal-size blocks is impossible: the natural
// blocking is n_domains blocks of `order` nodes plus one final block holding the
// last shared endpoint.  A uniform-only API would force either a padded pencil
// with spurious modes or a dense factorization.  The block-Thomas recurrence
// itself does not care -- the off-diagonal blocks simply become rectangular.
//
// hipSOLVER performs partial pivoting *inside* each diagonal block.  There is
// deliberately no pivoting between adjacent block rows: a globally nonsingular
// matrix can therefore be unsupported when a block Schur complement is
// singular (for example [[0,1],[1,0]] with scalar blocks).  Such a case is
// returned explicitly as `singular_diagonal_block`, with its zero-based block
// and pivot indices, rather than silently producing infinities.

#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"

#include <cstddef>
#include <vector>

namespace quasar::numerics {

enum class BlockTridiagonalStatus {
  success,
  singular_diagonal_block,
  invalid_factorization_argument,
  invalid_solve_argument,
  solve_failed,
};

// Block structure of a block-tridiagonal matrix: the order of every block row,
// plus the prefix offsets addressing each storage array.  Construction validates
// the orders and rejects any arithmetic that would overflow addressable storage,
// so every accessor below is total.
class BlockPartition {
 public:
  BlockPartition() = default;

  // Each entry is the order of one block row and must be positive.
  explicit BlockPartition(std::vector<int> block_orders);

  [[nodiscard]] static BlockPartition uniform(int block_count,
                                              int block_order);

  [[nodiscard]] bool empty() const noexcept { return orders_.empty(); }
  [[nodiscard]] int block_count() const noexcept {
    return static_cast<int>(orders_.size());
  }
  [[nodiscard]] int block_order(int block) const;
  [[nodiscard]] int maximum_block_order() const noexcept { return maximum_; }

  // Zero when the partition is non-uniform.
  [[nodiscard]] int uniform_block_order() const noexcept {
    return uniform_order_;
  }
  [[nodiscard]] bool is_uniform() const noexcept {
    return uniform_order_ != 0;
  }

  [[nodiscard]] std::size_t total_rows() const noexcept { return total_rows_; }

  [[nodiscard]] std::size_t diagonal_offset(int block) const;
  [[nodiscard]] std::size_t diagonal_size() const noexcept {
    return diagonal_size_;
  }

  // Valid for block in [0, block_count() - 1): the coupling between block and
  // block + 1.  Both L_block and U_block use this offset in their own arrays.
  [[nodiscard]] std::size_t off_diagonal_offset(int block) const;
  [[nodiscard]] std::size_t off_diagonal_size() const noexcept {
    return off_diagonal_size_;
  }

  [[nodiscard]] std::size_t pivot_offset(int block) const;
  [[nodiscard]] std::size_t pivot_size() const noexcept { return total_rows_; }

  [[nodiscard]] std::size_t rhs_offset(int block, int rhs_count) const;
  [[nodiscard]] std::size_t rhs_size(int rhs_count) const;

  [[nodiscard]] bool operator==(const BlockPartition& other) const noexcept {
    return orders_ == other.orders_;
  }

 private:
  void build();

  std::vector<int> orders_{};
  std::vector<std::size_t> diagonal_offsets_{};
  std::vector<std::size_t> off_diagonal_offsets_{};
  std::vector<std::size_t> row_offsets_{};
  std::size_t diagonal_size_{0};
  std::size_t off_diagonal_size_{0};
  std::size_t total_rows_{0};
  int maximum_{0};
  int uniform_order_{0};
};

// Reusable block-LU factors.  `upper_reduced` stores D_k^{-1} U_k after the
// intra-block LU solve; `diagonal_lu` and `pivots` retain the hipSOLVER factors.
// Keeping the factors makes repeated shift-invert applications avoid a fresh
// matrix factorization for every Lanczos vector.
struct BlockTridiagonalFactorization {
  BlockTridiagonalStatus status{BlockTridiagonalStatus::success};
  BlockPartition partition{};

  // Convenience mirrors of `partition`.  `block_order` is zero for a
  // non-uniform partition; consult `partition` in that case.
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
    const BlockPartition& partition,
    backend::stream_t stream = nullptr);

// Uniform-block overload.
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

// One-shot convenience overloads preserving both factorization_info and
// solve_info in the returned status object.
[[nodiscard]] BlockTridiagonalSolveResult solve_block_tridiagonal(
    const backend::DeviceBuffer<Real>& lower,
    const backend::DeviceBuffer<Real>& diagonal,
    const backend::DeviceBuffer<Real>& upper,
    const backend::DeviceBuffer<Real>& rhs,
    const BlockPartition& partition,
    int rhs_count,
    backend::stream_t stream = nullptr);

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
