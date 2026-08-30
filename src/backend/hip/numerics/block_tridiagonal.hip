#include "quasar/numerics/block_tridiagonal.hpp"

#include "backend/hip/numerics/linear_algebra_detail.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace quasar::numerics {
namespace {

std::size_t checked_multiply(std::size_t a, std::size_t b,
                             const char* what) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::length_error{std::string{"block tridiagonal "} + what
                            + " overflows addressable storage"};
  }
  return a * b;
}

std::size_t checked_add(std::size_t a, std::size_t b, const char* what) {
  if (b > std::numeric_limits<std::size_t>::max() - a) {
    throw std::length_error{std::string{"block tridiagonal "} + what
                            + " overflows addressable storage"};
  }
  return a + b;
}

void require_size(const backend::DeviceBuffer<Real>& buffer,
                  std::size_t expected,
                  const char* name) {
  if (buffer.size() != expected) {
    throw std::invalid_argument{
        std::string{"block tridiagonal "} + name
        + " buffer has the wrong size"};
  }
}

void require_owner(const backend::DeviceBuffer<Real>& buffer,
                   std::size_t expected_size,
                   int device,
                   const char* name) {
  if (expected_size != 0 && buffer.owner_device() != device) {
    throw std::invalid_argument{
        std::string{"block tridiagonal "} + name
        + " buffer resides on a different device"};
  }
}

void validate_factorization_storage(
    const BlockTridiagonalFactorization& factorization) {
  const BlockPartition& partition = factorization.partition;
  require_size(factorization.lower, partition.off_diagonal_size(),
               "factor lower");
  require_size(factorization.diagonal_lu, partition.diagonal_size(),
               "factor diagonal");
  require_size(factorization.upper_reduced, partition.off_diagonal_size(),
               "factor upper");
  if (factorization.pivots.size() != partition.pivot_size()) {
    throw std::invalid_argument{
        "block tridiagonal factor pivot buffer has the wrong size"};
  }

  const int device = factorization.diagonal_lu.owner_device();
  require_owner(factorization.lower, partition.off_diagonal_size(), device,
                "factor lower");
  require_owner(factorization.upper_reduced, partition.off_diagonal_size(),
                device, "factor upper");
  if (factorization.pivots.owner_device() != device) {
    throw std::invalid_argument{
        "block tridiagonal factor pivots reside on a different device"};
  }
}

void assign_failure(BlockTridiagonalSolveResult& result,
                    const BlockTridiagonalFactorization& factorization) {
  result.status = factorization.status;
  result.factorization_info = factorization.factorization_info;
  result.solve_info = factorization.solve_info;
  result.failure_block = factorization.failure_block;
  result.failure_pivot = factorization.failure_pivot;
}

BlockTridiagonalStatus classify_solve_info(int info) noexcept {
  return info < 0 ? BlockTridiagonalStatus::invalid_solve_argument
                  : BlockTridiagonalStatus::solve_failed;
}

void require_partition(const BlockPartition& partition) {
  if (partition.empty()) {
    throw std::invalid_argument{
        "block tridiagonal partition must contain at least one block"};
  }
}

int checked_rhs_count(int rhs_count) {
  if (rhs_count <= 0) {
    throw std::invalid_argument{
        "block tridiagonal rhs_count must be positive"};
  }
  return rhs_count;
}

}  // namespace

BlockPartition::BlockPartition(std::vector<int> block_orders)
    : orders_{std::move(block_orders)} {
  build();
}

BlockPartition BlockPartition::uniform(int block_count, int block_order) {
  if (block_count <= 0 || block_order <= 0) {
    throw std::invalid_argument{
        "block tridiagonal dimensions must both be positive"};
  }
  return BlockPartition{
      std::vector<int>(static_cast<std::size_t>(block_count), block_order)};
}

void BlockPartition::build() {
  if (orders_.empty()) {
    throw std::invalid_argument{
        "block tridiagonal partition must contain at least one block"};
  }
  for (const int order : orders_) {
    if (order <= 0) {
      throw std::invalid_argument{
          "block tridiagonal block orders must all be positive"};
    }
  }

  const std::size_t count = orders_.size();
  diagonal_offsets_.assign(count + 1, 0);
  off_diagonal_offsets_.assign(count, 0);
  row_offsets_.assign(count + 1, 0);

  maximum_ = *std::max_element(orders_.begin(), orders_.end());
  uniform_order_ =
      std::all_of(orders_.begin(), orders_.end(),
                  [this](int order) { return order == orders_.front(); })
          ? orders_.front()
          : 0;

  for (std::size_t block = 0; block < count; ++block) {
    const auto order = static_cast<std::size_t>(orders_[block]);
    row_offsets_[block + 1] =
        checked_add(row_offsets_[block], order, "row count");
    diagonal_offsets_[block + 1] = checked_add(
        diagonal_offsets_[block],
        checked_multiply(order, order, "diagonal block"), "diagonal storage");
  }
  total_rows_ = row_offsets_[count];
  diagonal_size_ = diagonal_offsets_[count];

  // L_k and U_k both couple block k to block k+1 and hold n_k*n_{k+1} values,
  // so a single offset array serves both arrays.
  std::size_t off_diagonal = 0;
  for (std::size_t block = 0; block + 1 < count; ++block) {
    off_diagonal_offsets_[block] = off_diagonal;
    off_diagonal = checked_add(
        off_diagonal,
        checked_multiply(static_cast<std::size_t>(orders_[block]),
                         static_cast<std::size_t>(orders_[block + 1]),
                         "off-diagonal block"),
        "off-diagonal storage");
  }
  off_diagonal_offsets_[count - 1] = off_diagonal;
  off_diagonal_size_ = off_diagonal;
}

int BlockPartition::block_order(int block) const {
  if (block < 0 || block >= block_count()) {
    throw std::out_of_range{"block tridiagonal block index out of range"};
  }
  return orders_[static_cast<std::size_t>(block)];
}

std::size_t BlockPartition::diagonal_offset(int block) const {
  if (block < 0 || block >= block_count()) {
    throw std::out_of_range{"block tridiagonal block index out of range"};
  }
  return diagonal_offsets_[static_cast<std::size_t>(block)];
}

std::size_t BlockPartition::off_diagonal_offset(int block) const {
  if (block < 0 || block + 1 >= block_count()) {
    throw std::out_of_range{
        "block tridiagonal off-diagonal index out of range"};
  }
  return off_diagonal_offsets_[static_cast<std::size_t>(block)];
}

std::size_t BlockPartition::pivot_offset(int block) const {
  if (block < 0 || block >= block_count()) {
    throw std::out_of_range{"block tridiagonal block index out of range"};
  }
  return row_offsets_[static_cast<std::size_t>(block)];
}

std::size_t BlockPartition::rhs_offset(int block, int rhs_count) const {
  if (block < 0 || block >= block_count()) {
    throw std::out_of_range{"block tridiagonal block index out of range"};
  }
  if (rhs_count <= 0) {
    throw std::invalid_argument{
        "block tridiagonal rhs_count must be positive"};
  }
  return checked_multiply(row_offsets_[static_cast<std::size_t>(block)],
                          static_cast<std::size_t>(rhs_count),
                          "right-hand-side offset");
}

std::size_t BlockPartition::rhs_size(int rhs_count) const {
  if (rhs_count <= 0) {
    throw std::invalid_argument{
        "block tridiagonal rhs_count must be positive"};
  }
  return checked_multiply(total_rows_, static_cast<std::size_t>(rhs_count),
                          "right-hand-side storage");
}

BlockTridiagonalFactorization factor_block_tridiagonal(
    const backend::DeviceBuffer<Real>& lower,
    const backend::DeviceBuffer<Real>& diagonal,
    const backend::DeviceBuffer<Real>& upper,
    const BlockPartition& partition,
    backend::stream_t stream) {
  require_partition(partition);
  require_size(lower, partition.off_diagonal_size(), "lower");
  require_size(diagonal, partition.diagonal_size(), "diagonal");
  require_size(upper, partition.off_diagonal_size(), "upper");

  const int device = diagonal.owner_device();
  require_owner(lower, partition.off_diagonal_size(), device, "lower");
  require_owner(upper, partition.off_diagonal_size(), device, "upper");
  backend::DeviceGuard device_guard{device};

  const int block_count = partition.block_count();
  BlockTridiagonalFactorization result;
  result.partition = partition;
  result.block_count = block_count;
  result.block_order = partition.uniform_block_order();
  result.lower = backend::DeviceBuffer<Real>{
      partition.off_diagonal_size(), backend::uninitialized,
      backend::on_device(device)};
  result.diagonal_lu = backend::DeviceBuffer<Real>{
      partition.diagonal_size(), backend::uninitialized,
      backend::on_device(device)};
  result.upper_reduced = backend::DeviceBuffer<Real>{
      partition.off_diagonal_size(), backend::uninitialized,
      backend::on_device(device)};
  result.pivots = backend::DeviceBuffer<int>{
      partition.pivot_size(), backend::uninitialized,
      backend::on_device(device)};

  detail::copy_device_buffer(result.lower, lower,
                             partition.off_diagonal_size(), stream);
  detail::copy_device_buffer(result.diagonal_lu, diagonal,
                             partition.diagonal_size(), stream);
  detail::copy_device_buffer(result.upper_reduced, upper,
                             partition.off_diagonal_size(), stream);

  detail::SolverHandle solver{stream};
  detail::BlasHandle blas{stream};
  backend::DeviceBuffer<int> device_factor_info{1, backend::on_device(device)};
  backend::DeviceBuffer<int> device_solve_info{1, backend::on_device(device)};

  // One workspace sized for the largest diagonal block serves every block.
  const int maximum_order = partition.maximum_block_order();
  int workspace_size = 0;
  detail::check_hipsolver(
      hipsolverDnDgetrf_bufferSize(
          solver.get(), maximum_order, maximum_order,
          result.diagonal_lu.device_ptr(), maximum_order, &workspace_size),
      "getrf workspace query");
  if (workspace_size < 0) {
    throw std::runtime_error{
        "hipSOLVER getrf workspace query returned a negative size"};
  }
  backend::DeviceBuffer<Real> workspace{
      static_cast<std::size_t>(workspace_size), backend::uninitialized,
      backend::on_device(device)};

  const Real minus_one = Real{-1};
  const Real plus_one = Real{1};
  for (int block_index = 0; block_index < block_count; ++block_index) {
    const int order = partition.block_order(block_index);
    Real* diagonal_block =
        result.diagonal_lu.device_ptr() + partition.diagonal_offset(
                                              block_index);
    int* pivots =
        result.pivots.device_ptr() + partition.pivot_offset(block_index);

    // Installed hipSOLVER releases take the workspace pointer here, but no
    // lwork argument; lwork is used only to size this allocation.
    detail::check_hipsolver(
        hipsolverDnDgetrf(solver.get(), order, order, diagonal_block, order,
                          workspace.device_ptr(), pivots,
                          device_factor_info.device_ptr()),
        "getrf");
    result.factorization_info = detail::read_info(device_factor_info, stream);
    if (result.factorization_info != 0) {
      result.failure_block = block_index;
      if (result.factorization_info > 0) {
        result.status = BlockTridiagonalStatus::singular_diagonal_block;
        result.failure_pivot = result.factorization_info - 1;
      } else {
        result.status =
            BlockTridiagonalStatus::invalid_factorization_argument;
      }
      return result;
    }

    if (block_index + 1 == block_count) continue;

    const int next_order = partition.block_order(block_index + 1);
    const std::size_t off_diagonal_offset =
        partition.off_diagonal_offset(block_index);
    // U_k is order x next_order with leading dimension order.
    Real* reduced_upper =
        result.upper_reduced.device_ptr() + off_diagonal_offset;
    detail::check_hipsolver(
        hipsolverDnDgetrs(solver.get(), HIPBLAS_OP_N, order, next_order,
                          diagonal_block, order, pivots, reduced_upper, order,
                          device_solve_info.device_ptr()),
        "getrs while reducing upper block");
    result.solve_info = detail::read_info(device_solve_info, stream);
    if (result.solve_info != 0) {
      result.status = classify_solve_info(result.solve_info);
      result.failure_block = block_index;
      return result;
    }

    // D_{k+1} <- D_{k+1} - L_k (D_k^{-1} U_k), with L_k next_order x order and
    // the reduced upper block order x next_order.
    const Real* lower_block = result.lower.device_ptr() + off_diagonal_offset;
    Real* next_diagonal =
        result.diagonal_lu.device_ptr()
        + partition.diagonal_offset(block_index + 1);
    detail::check_hipblas(
        hipblasDgemm(blas.get(), HIPBLAS_OP_N, HIPBLAS_OP_N, next_order,
                     next_order, order, &minus_one, lower_block, next_order,
                     reduced_upper, order, &plus_one, next_diagonal,
                     next_order),
        "Schur-complement update");
  }

  // The final devInfo read synchronized the stream, so all factors are ready
  // for use even if the caller chooses a different stream for the solve.
  result.status = BlockTridiagonalStatus::success;
  return result;
}

BlockTridiagonalFactorization factor_block_tridiagonal(
    const backend::DeviceBuffer<Real>& lower,
    const backend::DeviceBuffer<Real>& diagonal,
    const backend::DeviceBuffer<Real>& upper,
    int block_count,
    int block_order,
    backend::stream_t stream) {
  return factor_block_tridiagonal(
      lower, diagonal, upper, BlockPartition::uniform(block_count, block_order),
      stream);
}

BlockTridiagonalSolveResult solve_block_tridiagonal(
    const BlockTridiagonalFactorization& factorization,
    const backend::DeviceBuffer<Real>& rhs,
    int rhs_count,
    backend::stream_t stream) {
  BlockTridiagonalSolveResult result;
  result.block_count = factorization.block_count;
  result.block_order = factorization.block_order;
  result.rhs_count = rhs_count;
  result.factorization_info = factorization.factorization_info;
  result.solve_info = factorization.solve_info;

  if (!factorization.ok()) {
    assign_failure(result, factorization);
    return result;
  }

  const BlockPartition& partition = factorization.partition;
  require_partition(partition);
  validate_factorization_storage(factorization);
  checked_rhs_count(rhs_count);
  const std::size_t rhs_size = partition.rhs_size(rhs_count);
  require_size(rhs, rhs_size, "right-hand-side");

  const int device = factorization.diagonal_lu.owner_device();
  require_owner(rhs, rhs_size, device, "right-hand-side");
  backend::DeviceGuard device_guard{device};

  result.solution = backend::DeviceBuffer<Real>{
      rhs_size, backend::uninitialized, backend::on_device(device)};
  detail::copy_device_buffer(result.solution, rhs, rhs_size, stream);

  detail::SolverHandle solver{stream};
  detail::BlasHandle blas{stream};
  backend::DeviceBuffer<int> device_solve_info{1, backend::on_device(device)};

  const Real minus_one = Real{-1};
  const Real plus_one = Real{1};
  const int block_count = partition.block_count();

  for (int block_index = 0; block_index < block_count; ++block_index) {
    const int order = partition.block_order(block_index);
    Real* rhs_block = result.solution.device_ptr()
                    + partition.rhs_offset(block_index, rhs_count);

    if (block_index > 0) {
      const int previous_order = partition.block_order(block_index - 1);
      const Real* lower_block =
          factorization.lower.device_ptr()
          + partition.off_diagonal_offset(block_index - 1);
      const Real* previous_rhs =
          result.solution.device_ptr()
          + partition.rhs_offset(block_index - 1, rhs_count);
      // b_k <- b_k - L_{k-1} y_{k-1}, for every RHS at once.  L_{k-1} is
      // order x previous_order with leading dimension order.
      detail::check_hipblas(
          hipblasDgemm(blas.get(), HIPBLAS_OP_N, HIPBLAS_OP_N, order,
                       rhs_count, previous_order, &minus_one, lower_block,
                       order, previous_rhs, previous_order, &plus_one,
                       rhs_block, order),
          "block forward-substitution update");
    }

    detail::check_hipsolver(
        hipsolverDnDgetrs(
            solver.get(), HIPBLAS_OP_N, order, rhs_count,
            factorization.diagonal_lu.device_ptr()
                + partition.diagonal_offset(block_index),
            order,
            factorization.pivots.device_ptr()
                + partition.pivot_offset(block_index),
            rhs_block, order, device_solve_info.device_ptr()),
        "getrs while solving right-hand side");
    result.solve_info = detail::read_info(device_solve_info, stream);
    if (result.solve_info != 0) {
      result.status = classify_solve_info(result.solve_info);
      result.failure_block = block_index;
      return result;
    }
  }

  for (int block_index = block_count - 1; block_index-- > 0;) {
    const int order = partition.block_order(block_index);
    const int next_order = partition.block_order(block_index + 1);
    Real* rhs_block = result.solution.device_ptr()
                    + partition.rhs_offset(block_index, rhs_count);
    const Real* next_solution = result.solution.device_ptr()
                              + partition.rhs_offset(block_index + 1,
                                                     rhs_count);
    // x_k <- y_k - (D_k^{-1} U_k) x_{k+1}, with the reduced upper block
    // order x next_order.
    detail::check_hipblas(
        hipblasDgemm(blas.get(), HIPBLAS_OP_N, HIPBLAS_OP_N, order, rhs_count,
                     next_order, &minus_one,
                     factorization.upper_reduced.device_ptr()
                         + partition.off_diagonal_offset(block_index),
                     order, next_solution, next_order, &plus_one, rhs_block,
                     order),
        "block back-substitution update");
  }

  backend::device_synchronize(stream);
  result.status = BlockTridiagonalStatus::success;
  return result;
}

BlockTridiagonalSolveResult solve_block_tridiagonal(
    const backend::DeviceBuffer<Real>& lower,
    const backend::DeviceBuffer<Real>& diagonal,
    const backend::DeviceBuffer<Real>& upper,
    const backend::DeviceBuffer<Real>& rhs,
    const BlockPartition& partition,
    int rhs_count,
    backend::stream_t stream) {
  auto factorization =
      factor_block_tridiagonal(lower, diagonal, upper, partition, stream);
  return solve_block_tridiagonal(factorization, rhs, rhs_count, stream);
}

BlockTridiagonalSolveResult solve_block_tridiagonal(
    const backend::DeviceBuffer<Real>& lower,
    const backend::DeviceBuffer<Real>& diagonal,
    const backend::DeviceBuffer<Real>& upper,
    const backend::DeviceBuffer<Real>& rhs,
    int block_count,
    int block_order,
    int rhs_count,
    backend::stream_t stream) {
  return solve_block_tridiagonal(
      lower, diagonal, upper, rhs,
      BlockPartition::uniform(block_count, block_order), rhs_count, stream);
}

}  // namespace quasar::numerics
