#pragma once

// Block-tridiagonal view of the assembled Chebyshev x Fourier pencil.
//
// The assembly emits dense column-major matrices indexed harmonic-major: the
// harmonic block is the slow index, then the radial node, then the displacement
// component, then the real/imaginary quadrature.  In that ordering the matrices
// have no useful band structure.
//
// They do have structure, though, and it is exact rather than approximate.  Two
// degrees of freedom couple only when their radial nodes share a Chebyshev
// domain, because the quadrature loop in the assembly skips any (row, column)
// pair with no common domain outright.  Reordering radial-major -- radial node
// slow, then harmonic, component, and quadrature -- therefore produces a matrix
// whose nonzeros lie in the domain-local diagonal blocks and nowhere else.
//
// -- Choosing the blocks -------------------------------------------------------
// Domain d owns radial nodes [d*order, (d+1)*order], sharing its endpoints with
// its neighbours, so there are n_domains*order + 1 radial nodes in total.  That
// count is not a multiple of `order` for order > 1, which is why the blocking
// below is deliberately non-uniform:
//
//   block d, for d in [0, n_domains):  radial nodes [d*order, (d+1)*order)
//   block n_domains:                   the single final radial node
//
// Under this partition the degrees of freedom of domain d fall entirely in
// blocks d and d+1, so the reordered matrix is block-tridiagonal exactly.  Any
// other split of the shared endpoints produces the same count mismatch
// somewhere else; padding it to uniform blocks would inject spurious modes into
// the pencil, so `numerics::BlockPartition` carries per-block orders instead.
//
// The extraction below does not assume the structure: it also reduces the
// largest magnitude found OUTSIDE the block-tridiagonal pattern, so a caller can
// assert that what it discarded really was zero.

#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/block_tridiagonal.hpp"
#include "quasar/physics/stability/kernels.hpp"
#include "quasar/physics/stability/spectral_layout.hpp"

#include <cstddef>
#include <vector>

namespace quasar::stability {

enum class SpectralBlockStatus {
  supported,
  // Harmonic-specific radial splits at a rational interface give different
  // harmonics different radial counts, so "radial node k" no longer names one
  // block row across the whole matrix.  The solver refuses rational topology
  // upstream; this is the structural restatement of that restriction.
  nonuniform_harmonic_radial_counts,
  // A partition needs at least two block rows to be tridiagonal in any useful
  // sense; a single domain of order 1 degenerates.
  degenerate_partition,
};

struct SpectralBlockStructure {
  SpectralBlockStatus status{
      SpectralBlockStatus::nonuniform_harmonic_radial_counts};
  numerics::BlockPartition partition{};

  // permutation[p] is the free real DOF index, in assembly ordering, that
  // occupies position p of the radial-major ordering.
  std::vector<int> permutation{};
  int order{0};

  [[nodiscard]] bool ok() const noexcept {
    return status == SpectralBlockStatus::supported;
  }
};

// Builds the radial-major permutation and the block partition.  `layout`
// describes the unconstrained space and `free_to_full_complex` is the
// fixed-boundary map produced with the matrices, so eliminated degrees of
// freedom are excluded exactly as they are from the assembled matrices.
[[nodiscard]] SpectralBlockStructure build_spectral_block_structure(
    const SpectralDofLayout& layout,
    const std::vector<std::size_t>& free_to_full_complex,
    int chebyshev_order);

// Block-tridiagonal storage sized by a SpectralBlockStructure.
struct SpectralBlockMatrix {
  backend::DeviceBuffer<Real> lower{};
  backend::DeviceBuffer<Real> diagonal{};
  backend::DeviceBuffer<Real> upper{};

  // Largest absolute value found outside the block-tridiagonal pattern.  Zero
  // confirms the structural claim above; anything else means the extraction
  // silently dropped real couplings.
  Real maximum_outside_pattern{0};
};

// Extracts (stiffness - shift * inertia) into block-tridiagonal storage in the
// radial-major ordering.  Both inputs are dense column-major of side
// structure.order, in assembly ordering, on the same device.
void launch_extract_shifted_block_tridiagonal(
    const backend::DeviceBuffer<Real>& stiffness,
    const backend::DeviceBuffer<Real>& inertia, Real shift,
    const SpectralBlockStructure& structure, SpectralBlockMatrix& out,
    stream_t stream);

// Applies the same permutation to a dense symmetric matrix, so the mass term
// used for the Lanczos inner product is expressed in the reordered basis.
void launch_permute_dense_symmetric(
    const backend::DeviceBuffer<Real>& source,
    const SpectralBlockStructure& structure,
    backend::DeviceBuffer<Real>& out, stream_t stream);

}  // namespace quasar::stability
