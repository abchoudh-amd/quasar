#include "quasar/physics/stability/spectral_blocks.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace quasar::stability {
namespace {

constexpr unsigned kBlock2D = 16;

struct BlockIndexTables {
  backend::DeviceBuffer<int> permutation{};
  backend::DeviceBuffer<int> block_of{};
  backend::DeviceBuffer<int> index_in_block{};
  backend::DeviceBuffer<int> block_orders{};
  backend::DeviceBuffer<std::size_t> diagonal_offsets{};
  backend::DeviceBuffer<std::size_t> off_diagonal_offsets{};
};

BlockIndexTables upload_tables(const SpectralBlockStructure& structure,
                               int owner, stream_t stream) {
  const numerics::BlockPartition& partition = structure.partition;
  const int block_count = partition.block_count();
  const auto order = static_cast<std::size_t>(structure.order);

  std::vector<int> block_of(order);
  std::vector<int> index_in_block(order);
  std::vector<int> block_orders(static_cast<std::size_t>(block_count));
  std::vector<std::size_t> diagonal_offsets(
      static_cast<std::size_t>(block_count));
  std::vector<std::size_t> off_diagonal_offsets(
      static_cast<std::size_t>(block_count));

  std::size_t cursor = 0;
  for (int block = 0; block < block_count; ++block) {
    const int block_order = partition.block_order(block);
    block_orders[static_cast<std::size_t>(block)] = block_order;
    diagonal_offsets[static_cast<std::size_t>(block)] =
        partition.diagonal_offset(block);
    off_diagonal_offsets[static_cast<std::size_t>(block)] =
        block + 1 < block_count ? partition.off_diagonal_offset(block) : 0;
    for (int local = 0; local < block_order; ++local) {
      block_of[cursor] = block;
      index_in_block[cursor] = local;
      ++cursor;
    }
  }
  if (cursor != order) {
    throw std::logic_error{
        "spectral block partition does not cover the matrix order"};
  }

  BlockIndexTables tables;
  tables.permutation = backend::DeviceBuffer<int>{
      order, backend::uninitialized, backend::on_device(owner)};
  tables.block_of = backend::DeviceBuffer<int>{
      order, backend::uninitialized, backend::on_device(owner)};
  tables.index_in_block = backend::DeviceBuffer<int>{
      order, backend::uninitialized, backend::on_device(owner)};
  tables.block_orders = backend::DeviceBuffer<int>{
      block_orders.size(), backend::uninitialized, backend::on_device(owner)};
  tables.diagonal_offsets = backend::DeviceBuffer<std::size_t>{
      diagonal_offsets.size(), backend::uninitialized,
      backend::on_device(owner)};
  tables.off_diagonal_offsets = backend::DeviceBuffer<std::size_t>{
      off_diagonal_offsets.size(), backend::uninitialized,
      backend::on_device(owner)};

  tables.permutation.copy_from_host_async(structure.permutation.data(), order,
                                          stream);
  tables.block_of.copy_from_host_async(block_of.data(), order, stream);
  tables.index_in_block.copy_from_host_async(index_in_block.data(), order,
                                             stream);
  tables.block_orders.copy_from_host_async(block_orders.data(),
                                           block_orders.size(), stream);
  tables.diagonal_offsets.copy_from_host_async(diagonal_offsets.data(),
                                               diagonal_offsets.size(),
                                               stream);
  tables.off_diagonal_offsets.copy_from_host_async(
      off_diagonal_offsets.data(), off_diagonal_offsets.size(), stream);
  return tables;
}

// Non-negative IEEE doubles are monotonically ordered by their bit patterns, so
// an unsigned integer atomicMax reproduces a floating-point max exactly.  The
// result is independent of thread order, which keeps the check deterministic.
__device__ void accumulate_absolute_maximum(unsigned long long* target,
                                            Real value) {
  const Real magnitude = fabs(value);
  atomicMax(target, static_cast<unsigned long long>(
                        __double_as_longlong(magnitude)));
}

__global__ void extract_shifted_kernel(
    const Real* __restrict__ stiffness, const Real* __restrict__ inertia,
    Real shift, const int* __restrict__ permutation,
    const int* __restrict__ block_of, const int* __restrict__ index_in_block,
    const int* __restrict__ block_orders,
    const std::size_t* __restrict__ diagonal_offsets,
    const std::size_t* __restrict__ off_diagonal_offsets, int order,
    Real* __restrict__ lower, Real* __restrict__ diagonal,
    Real* __restrict__ upper,
    unsigned long long* __restrict__ outside_maximum) {
  const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (row >= order || column >= order) return;

  const std::size_t source =
      static_cast<std::size_t>(permutation[row])
      + static_cast<std::size_t>(permutation[column])
            * static_cast<std::size_t>(order);
  const Real value = stiffness[source] - shift * inertia[source];

  const int row_block = block_of[row];
  const int column_block = block_of[column];
  const int row_local = index_in_block[row];
  const int column_local = index_in_block[column];

  if (row_block == column_block) {
    const std::size_t leading =
        static_cast<std::size_t>(block_orders[row_block]);
    diagonal[diagonal_offsets[row_block]
             + static_cast<std::size_t>(row_local)
             + static_cast<std::size_t>(column_local) * leading] = value;
    return;
  }
  if (row_block == column_block + 1) {
    // L_{column_block}: block_orders[row_block] x block_orders[column_block].
    const std::size_t leading =
        static_cast<std::size_t>(block_orders[row_block]);
    lower[off_diagonal_offsets[column_block]
          + static_cast<std::size_t>(row_local)
          + static_cast<std::size_t>(column_local) * leading] = value;
    return;
  }
  if (row_block + 1 == column_block) {
    // U_{row_block}: block_orders[row_block] x block_orders[column_block].
    const std::size_t leading =
        static_cast<std::size_t>(block_orders[row_block]);
    upper[off_diagonal_offsets[row_block]
          + static_cast<std::size_t>(row_local)
          + static_cast<std::size_t>(column_local) * leading] = value;
    return;
  }
  accumulate_absolute_maximum(outside_maximum, value);
}

__global__ void permute_dense_kernel(const Real* __restrict__ source,
                                     const int* __restrict__ permutation,
                                     int order,
                                     Real* __restrict__ destination) {
  const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (row >= order || column >= order) return;
  const std::size_t from =
      static_cast<std::size_t>(permutation[row])
      + static_cast<std::size_t>(permutation[column])
            * static_cast<std::size_t>(order);
  destination[static_cast<std::size_t>(row)
              + static_cast<std::size_t>(column)
                    * static_cast<std::size_t>(order)] = source[from];
}

void require_square(const backend::DeviceBuffer<Real>& buffer, int order,
                    const char* name) {
  const auto expected = static_cast<std::size_t>(order)
                      * static_cast<std::size_t>(order);
  if (buffer.size() != expected) {
    throw std::invalid_argument{
        std::string{"spectral block extraction: "} + name
        + " must be order*order column-major values"};
  }
}

dim3 square_grid(int order) {
  const auto span = static_cast<unsigned>(order);
  return dim3{(span + kBlock2D - 1) / kBlock2D,
              (span + kBlock2D - 1) / kBlock2D};
}

}  // namespace

SpectralBlockStructure build_spectral_block_structure(
    const SpectralDofLayout& layout,
    const std::vector<std::size_t>& free_to_full_complex,
    int chebyshev_order) {
  SpectralBlockStructure structure;
  if (chebyshev_order <= 0) {
    throw std::invalid_argument{
        "build_spectral_block_structure: chebyshev_order must be positive"};
  }

  // Every harmonic must carry the same radial count for "radial node k" to name
  // one block row across the whole matrix.
  const std::size_t radial_count = layout.n_radial(-layout.m_max());
  for (int m = -layout.m_max(); m <= layout.m_max(); ++m) {
    if (layout.n_radial(m) != radial_count) {
      structure.status =
          SpectralBlockStatus::nonuniform_harmonic_radial_counts;
      return structure;
    }
  }

  const int n_domains = layout.n_domains();
  const auto expected_radial =
      static_cast<std::size_t>(n_domains) * chebyshev_order + 1;
  if (radial_count != expected_radial) {
    structure.status =
        SpectralBlockStatus::nonuniform_harmonic_radial_counts;
    return structure;
  }
  if (n_domains < 1 || expected_radial < 2) {
    structure.status = SpectralBlockStatus::degenerate_partition;
    return structure;
  }

  // Invert the fixed-boundary map: full complex DOF -> free complex DOF.
  std::vector<std::size_t> full_to_free(layout.complex_dof_count(),
                                        free_to_full_complex.size());
  for (std::size_t free = 0; free < free_to_full_complex.size(); ++free) {
    const std::size_t full = free_to_full_complex[free];
    if (full >= full_to_free.size()) {
      throw std::invalid_argument{
          "build_spectral_block_structure: the fixed-boundary map addresses a "
          "complex DOF outside the layout"};
    }
    full_to_free[full] = free;
  }

  // Radial-major walk.  Block d covers radial nodes [d*order, (d+1)*order); the
  // final single node forms block n_domains.
  std::vector<int> permutation;
  permutation.reserve(2 * free_to_full_complex.size());
  std::vector<int> block_orders;
  block_orders.reserve(static_cast<std::size_t>(n_domains) + 1);

  const auto emit_range = [&](std::size_t first, std::size_t last) {
    int emitted = 0;
    for (std::size_t radial = first; radial < last; ++radial) {
      for (int m = -layout.m_max(); m <= layout.m_max(); ++m) {
        for (int component = 0;
             component < SpectralDofLayout::kComponentCount; ++component) {
          const std::size_t full = layout.complex_dof(
              radial, m, static_cast<DisplacementComponent>(component));
          const std::size_t free = full_to_free[full];
          if (free == free_to_full_complex.size()) continue;  // eliminated
          permutation.push_back(static_cast<int>(2 * free));
          permutation.push_back(static_cast<int>(2 * free + 1));
          emitted += 2;
        }
      }
    }
    return emitted;
  };

  for (int domain = 0; domain < n_domains; ++domain) {
    const auto first = static_cast<std::size_t>(domain) * chebyshev_order;
    const auto last = first + static_cast<std::size_t>(chebyshev_order);
    const int emitted = emit_range(first, last);
    if (emitted <= 0) {
      structure.status = SpectralBlockStatus::degenerate_partition;
      return structure;
    }
    block_orders.push_back(emitted);
  }
  const int final_emitted = emit_range(expected_radial - 1, expected_radial);
  if (final_emitted <= 0) {
    structure.status = SpectralBlockStatus::degenerate_partition;
    return structure;
  }
  block_orders.push_back(final_emitted);

  if (block_orders.size() < 2) {
    structure.status = SpectralBlockStatus::degenerate_partition;
    return structure;
  }

  structure.partition = numerics::BlockPartition{std::move(block_orders)};
  structure.order = static_cast<int>(permutation.size());
  structure.permutation = std::move(permutation);
  if (structure.partition.total_rows()
      != static_cast<std::size_t>(structure.order)) {
    throw std::logic_error{
        "build_spectral_block_structure: partition and permutation disagree"};
  }
  structure.status = SpectralBlockStatus::supported;
  return structure;
}

void launch_extract_shifted_block_tridiagonal(
    const backend::DeviceBuffer<Real>& stiffness,
    const backend::DeviceBuffer<Real>& inertia, Real shift,
    const SpectralBlockStructure& structure, SpectralBlockMatrix& out,
    stream_t stream) {
  if (!structure.ok()) {
    throw std::invalid_argument{
        "launch_extract_shifted_block_tridiagonal: the block structure is not "
        "supported for this topology"};
  }
  require_square(stiffness, structure.order, "stiffness");
  require_square(inertia, structure.order, "inertia");
  if (stiffness.owner_device() != inertia.owner_device()) {
    throw std::invalid_argument{
        "launch_extract_shifted_block_tridiagonal: stiffness and inertia "
        "reside on different devices"};
  }

  const int owner = stiffness.owner_device();
  backend::DeviceGuard guard{owner};
  const numerics::BlockPartition& partition = structure.partition;

  // Zero-initialized: the off-diagonal blocks are rectangular and every entry
  // is written by exactly one thread, but zeroing keeps a partition/permutation
  // mismatch from surfacing as uninitialized memory rather than a clean failure.
  out.lower = backend::DeviceBuffer<Real>{partition.off_diagonal_size(),
                                          backend::on_device(owner)};
  out.diagonal = backend::DeviceBuffer<Real>{partition.diagonal_size(),
                                             backend::on_device(owner)};
  out.upper = backend::DeviceBuffer<Real>{partition.off_diagonal_size(),
                                          backend::on_device(owner)};

  const BlockIndexTables tables = upload_tables(structure, owner, stream);
  backend::DeviceBuffer<unsigned long long> outside{
      1, backend::on_device(owner)};

  auto* hip_stream = static_cast<hipStream_t>(stream);
  const dim3 grid = square_grid(structure.order);
  const dim3 block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(
      extract_shifted_kernel, grid,
      block, 0, hip_stream, stiffness.device_ptr(),
      inertia.device_ptr(), shift, tables.permutation.device_ptr(),
      tables.block_of.device_ptr(), tables.index_in_block.device_ptr(),
      tables.block_orders.device_ptr(),
      tables.diagonal_offsets.device_ptr(),
      tables.off_diagonal_offsets.device_ptr(), structure.order,
      out.lower.device_ptr(), out.diagonal.device_ptr(),
      out.upper.device_ptr(), outside.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  unsigned long long host_outside = 0;
  outside.copy_to_host_async(&host_outside, 1, stream);
  backend::device_synchronize(stream);
  const long long signed_bits = static_cast<long long>(host_outside);
  double magnitude = 0;
  static_assert(sizeof(magnitude) == sizeof(signed_bits),
                "double and long long must share a width for the bit cast");
  __builtin_memcpy(&magnitude, &signed_bits, sizeof(magnitude));
  out.maximum_outside_pattern = magnitude;
}

void launch_permute_dense_symmetric(
    const backend::DeviceBuffer<Real>& source,
    const SpectralBlockStructure& structure,
    backend::DeviceBuffer<Real>& out, stream_t stream) {
  if (!structure.ok()) {
    throw std::invalid_argument{
        "launch_permute_dense_symmetric: the block structure is not supported "
        "for this topology"};
  }
  require_square(source, structure.order, "source");

  const int owner = source.owner_device();
  backend::DeviceGuard guard{owner};
  const auto entries = static_cast<std::size_t>(structure.order)
                     * static_cast<std::size_t>(structure.order);
  out = backend::DeviceBuffer<Real>{entries, backend::uninitialized,
                                    backend::on_device(owner)};

  backend::DeviceBuffer<int> permutation{
      static_cast<std::size_t>(structure.order), backend::uninitialized,
      backend::on_device(owner)};
  permutation.copy_from_host_async(structure.permutation.data(),
                                   structure.permutation.size(), stream);

  auto* hip_stream = static_cast<hipStream_t>(stream);
  const dim3 grid = square_grid(structure.order);
  const dim3 block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(permute_dense_kernel, grid, block, 0, hip_stream,
                     source.device_ptr(), permutation.device_ptr(),
                     structure.order, out.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());
  // `permutation` is freed at scope exit, so the kernel must have consumed it.
  backend::device_synchronize(stream);
}

}  // namespace quasar::stability
