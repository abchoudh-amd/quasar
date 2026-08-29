#include "quasar/numerics/block_tridiagonal.hpp"

#include "backend/hip/numerics/linear_algebra_detail.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace quasar::numerics {
namespace {

struct BlockSizes {
  std::size_t block{};
  std::size_t off_diagonal{};
  std::size_t diagonal{};
};

BlockSizes checked_block_sizes(int block_count, int block_order) {
  if (block_count <= 0 || block_order <= 0) {
    throw std::invalid_argument{
        "block tridiagonal dimensions must both be positive"};
  }
  const auto count = static_cast<std::size_t>(block_count);
  const auto order = static_cast<std::size_t>(block_order);
  if (order > std::numeric_limits<std::size_t>::max() / order) {
    throw std::length_error{"block order overflows addressable storage"};
  }
  const std::size_t block = order * order;
  if (block > std::numeric_limits<std::size_t>::max() / count) {
    throw std::length_error{"block diagonal storage size overflows"};
  }
  return BlockSizes{block, (count - 1) * block, count * block};
}

std::size_t checked_rhs_size(int block_count, int block_order, int rhs_count) {
  if (rhs_count <= 0) {
    throw std::invalid_argument{
        "block tridiagonal rhs_count must be positive"};
  }
  const auto rows = static_cast<std::size_t>(block_count)
                  * static_cast<std::size_t>(block_order);
  const auto columns = static_cast<std::size_t>(rhs_count);
  if (rows > std::numeric_limits<std::size_t>::max() / columns) {
    throw std::length_error{"block right-hand-side storage size overflows"};
  }
  return rows * columns;
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
    const BlockTridiagonalFactorization& factorization,
    const BlockSizes& sizes) {
  require_size(factorization.lower, sizes.off_diagonal, "factor lower");
  require_size(factorization.diagonal_lu, sizes.diagonal,
               "factor diagonal");
  require_size(factorization.upper_reduced, sizes.off_diagonal,
               "factor upper");
  const auto pivot_size = static_cast<std::size_t>(factorization.block_count)
                        * static_cast<std::size_t>(factorization.block_order);
  if (factorization.pivots.size() != pivot_size) {
    throw std::invalid_argument{
        "block tridiagonal factor pivot buffer has the wrong size"};
  }

  const int device = factorization.diagonal_lu.owner_device();
  require_owner(factorization.lower, sizes.off_diagonal, device,
                "factor lower");
  require_owner(factorization.upper_reduced, sizes.off_diagonal, device,
                "factor upper");
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

}  // namespace

BlockTridiagonalFactorization factor_block_tridiagonal(
    const backend::DeviceBuffer<Real>& lower,
    const backend::DeviceBuffer<Real>& diagonal,
    const backend::DeviceBuffer<Real>& upper,
    int block_count,
    int block_order,
    backend::stream_t stream) {
  const BlockSizes sizes = checked_block_sizes(block_count, block_order);
  require_size(lower, sizes.off_diagonal, "lower");
  require_size(diagonal, sizes.diagonal, "diagonal");
  require_size(upper, sizes.off_diagonal, "upper");

  const int device = diagonal.owner_device();
  require_owner(lower, sizes.off_diagonal, device, "lower");
  require_owner(upper, sizes.off_diagonal, device, "upper");
  backend::DeviceGuard device_guard{device};

  BlockTridiagonalFactorization result;
  result.block_count = block_count;
  result.block_order = block_order;
  result.lower = backend::DeviceBuffer<Real>{
      sizes.off_diagonal, backend::uninitialized, backend::on_device(device)};
  result.diagonal_lu = backend::DeviceBuffer<Real>{
      sizes.diagonal, backend::uninitialized, backend::on_device(device)};
  result.upper_reduced = backend::DeviceBuffer<Real>{
      sizes.off_diagonal, backend::uninitialized, backend::on_device(device)};
  result.pivots = backend::DeviceBuffer<int>{
      static_cast<std::size_t>(block_count)
          * static_cast<std::size_t>(block_order),
      backend::uninitialized, backend::on_device(device)};

  detail::copy_device_buffer(result.lower, lower, sizes.off_diagonal, stream);
  detail::copy_device_buffer(result.diagonal_lu, diagonal, sizes.diagonal,
                             stream);
  detail::copy_device_buffer(result.upper_reduced, upper,
                             sizes.off_diagonal, stream);

  detail::SolverHandle solver{stream};
  detail::BlasHandle blas{stream};
  backend::DeviceBuffer<int> device_factor_info{1, backend::on_device(device)};
  backend::DeviceBuffer<int> device_solve_info{1, backend::on_device(device)};

  int workspace_size = 0;
  detail::check_hipsolver(
      hipsolverDnDgetrf_bufferSize(
          solver.get(), block_order, block_order,
          result.diagonal_lu.device_ptr(), block_order, &workspace_size),
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
    const auto diagonal_offset =
        static_cast<std::size_t>(block_index) * sizes.block;
    const auto pivot_offset = static_cast<std::size_t>(block_index)
                            * static_cast<std::size_t>(block_order);
    Real* diagonal_block = result.diagonal_lu.device_ptr() + diagonal_offset;
    int* pivots = result.pivots.device_ptr() + pivot_offset;

    // Installed hipSOLVER releases take the workspace pointer here, but no
    // lwork argument; lwork is used only to size this allocation.
    detail::check_hipsolver(
        hipsolverDnDgetrf(solver.get(), block_order, block_order,
                          diagonal_block, block_order, workspace.device_ptr(),
                          pivots, device_factor_info.device_ptr()),
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

    const auto off_diagonal_offset =
        static_cast<std::size_t>(block_index) * sizes.block;
    Real* reduced_upper =
        result.upper_reduced.device_ptr() + off_diagonal_offset;
    detail::check_hipsolver(
        hipsolverDnDgetrs(solver.get(), HIPBLAS_OP_N, block_order,
                          block_order, diagonal_block, block_order, pivots,
                          reduced_upper, block_order,
                          device_solve_info.device_ptr()),
        "getrs while reducing upper block");
    result.solve_info = detail::read_info(device_solve_info, stream);
    if (result.solve_info != 0) {
      result.status = classify_solve_info(result.solve_info);
      result.failure_block = block_index;
      return result;
    }

    // D_{k+1} <- D_{k+1} - L_k (D_k^{-1} U_k).
    const Real* lower_block = result.lower.device_ptr() + off_diagonal_offset;
    Real* next_diagonal = diagonal_block + sizes.block;
    detail::check_hipblas(
        hipblasDgemm(blas.get(), HIPBLAS_OP_N, HIPBLAS_OP_N,
                     block_order, block_order, block_order, &minus_one,
                     lower_block, block_order, reduced_upper, block_order,
                     &plus_one, next_diagonal, block_order),
        "Schur-complement update");
  }

  // The final devInfo read synchronized the stream, so all factors are ready
  // for use even if the caller chooses a different stream for the solve.
  result.status = BlockTridiagonalStatus::success;
  return result;
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

  const BlockSizes sizes = checked_block_sizes(factorization.block_count,
                                               factorization.block_order);
  validate_factorization_storage(factorization, sizes);
  const std::size_t rhs_size = checked_rhs_size(
      factorization.block_count, factorization.block_order, rhs_count);
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
  const auto rhs_block_size = static_cast<std::size_t>(factorization.block_order)
                            * static_cast<std::size_t>(rhs_count);

  for (int block_index = 0; block_index < factorization.block_count;
       ++block_index) {
    const auto rhs_offset =
        static_cast<std::size_t>(block_index) * rhs_block_size;
    Real* rhs_block = result.solution.device_ptr() + rhs_offset;

    if (block_index > 0) {
      const auto off_diagonal_offset =
          static_cast<std::size_t>(block_index - 1) * sizes.block;
      const Real* lower_block =
          factorization.lower.device_ptr() + off_diagonal_offset;
      const Real* previous_rhs = rhs_block - rhs_block_size;
      // b_k <- b_k - L_{k-1} y_{k-1}, for every RHS at once.
      detail::check_hipblas(
          hipblasDgemm(blas.get(), HIPBLAS_OP_N, HIPBLAS_OP_N,
                       factorization.block_order, rhs_count,
                       factorization.block_order, &minus_one, lower_block,
                       factorization.block_order, previous_rhs,
                       factorization.block_order, &plus_one, rhs_block,
                       factorization.block_order),
          "block forward-substitution update");
    }

    const auto diagonal_offset =
        static_cast<std::size_t>(block_index) * sizes.block;
    const auto pivot_offset = static_cast<std::size_t>(block_index)
                            * static_cast<std::size_t>(factorization.block_order);
    detail::check_hipsolver(
        hipsolverDnDgetrs(
            solver.get(), HIPBLAS_OP_N, factorization.block_order, rhs_count,
            factorization.diagonal_lu.device_ptr() + diagonal_offset,
            factorization.block_order,
            factorization.pivots.device_ptr() + pivot_offset, rhs_block,
            factorization.block_order, device_solve_info.device_ptr()),
        "getrs while solving right-hand side");
    result.solve_info = detail::read_info(device_solve_info, stream);
    if (result.solve_info != 0) {
      result.status = classify_solve_info(result.solve_info);
      result.failure_block = block_index;
      return result;
    }
  }

  for (int block_index = factorization.block_count - 1;
       block_index-- > 0;) {
    const auto off_diagonal_offset =
        static_cast<std::size_t>(block_index) * sizes.block;
    const auto rhs_offset =
        static_cast<std::size_t>(block_index) * rhs_block_size;
    Real* rhs_block = result.solution.device_ptr() + rhs_offset;
    const Real* next_solution = rhs_block + rhs_block_size;
    // x_k <- y_k - (D_k^{-1} U_k) x_{k+1}.
    detail::check_hipblas(
        hipblasDgemm(
            blas.get(), HIPBLAS_OP_N, HIPBLAS_OP_N,
            factorization.block_order, rhs_count, factorization.block_order,
            &minus_one,
            factorization.upper_reduced.device_ptr() + off_diagonal_offset,
            factorization.block_order, next_solution,
            factorization.block_order, &plus_one, rhs_block,
            factorization.block_order),
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
    int block_count,
    int block_order,
    int rhs_count,
    backend::stream_t stream) {
  auto factorization = factor_block_tridiagonal(
      lower, diagonal, upper, block_count, block_order, stream);
  return solve_block_tridiagonal(factorization, rhs, rhs_count, stream);
}

}  // namespace quasar::numerics
