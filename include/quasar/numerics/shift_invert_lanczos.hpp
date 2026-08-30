#pragma once

// Shift-invert Lanczos for the symmetric-definite pencil K x = lambda M x.
//
// The iteration is Lanczos in the M inner product applied to
//
//   OP = (K - sigma M)^{-1} M,
//
// which is self-adjoint with respect to M.  Its eigenvalues theta relate to the
// pencil eigenvalues by
//
//   lambda = sigma + 1 / theta,
//
// so the eigenvalues of largest |theta| -- the ones Lanczos converges to first
// -- are exactly those of K x = lambda M x nearest the shift.  Choosing sigma
// just below the part of the spectrum of interest is therefore the whole art of
// using this routine; a shift in the middle of a dense cluster converges slowly
// and a shift equal to an eigenvalue makes the shifted matrix singular, which is
// reported rather than divided by.
//
// -- Why this exists alongside the dense eigensolver ---------------------------
// `solve_generalized_symmetric_eigenproblem` computes the entire spectrum in
// O(order^3).  Stability analysis needs a handful of eigenvalues at the bottom
// of the spectrum, and the shifted matrix inherits the block-tridiagonal
// structure of a spectral-element discretization.  Factoring it once with
// `factor_block_tridiagonal` and reusing those factors for every Lanczos vector
// replaces the cubic solve with one block factorization plus O(iterations)
// block back-substitutions and dense mass products.  For a few eigenvalues out
// of a few thousand that is a large constant-factor saving; for a small matrix,
// or when the whole spectrum is wanted, the dense path remains the right answer.
//
// -- Numerical choices ---------------------------------------------------------
// Full reorthogonalization against every previous basis vector is used, not
// selective or partial reorthogonalization.  The basis is O(order * iterations)
// which is small next to the dense matrices already in flight, and losing M
// orthogonality is the classic way this iteration silently returns duplicated
// "eigenvalues" that are really one eigenvalue found several times.  The extra
// cost is O(order * iterations^2), well below the factorization.
//
// The starting vector is generated from `random_seed` by an explicit
// counter-based hash rather than a library RNG, so a given seed reproduces the
// same Krylov space on every run and every device.

#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/block_tridiagonal.hpp"

#include <cstddef>
#include <vector>

namespace quasar::numerics {

enum class ShiftInvertStatus {
  success,
  // (K - sigma M) could not be factored: the shift is at or extremely near an
  // eigenvalue, or a block Schur complement is singular.
  shifted_matrix_singular,
  // The M inner product of the residual collapsed before `wanted` Ritz values
  // converged.  With an M that is genuinely positive definite this means the
  // Krylov space was exhausted, which happens for tiny orders.
  invariant_subspace_exhausted,
  // The mass matrix is not positive definite, so the M inner product is not an
  // inner product and the iteration is not defined.
  mass_not_positive_definite,
  not_converged,
};

struct ShiftInvertLanczosConfig {
  // Number of eigenvalues nearest `shift` to return.
  int wanted{1};
  Real shift{0};

  // Zero selects min(order, 2 * wanted + 40).  The iteration always stops at
  // `order`, where the Krylov space is complete.
  int maximum_iterations{0};

  // Convergence is declared when the Lanczos residual bound for every wanted
  // Ritz pair, relative to the largest |theta| seen, falls below this value.
  Real residual_tolerance{Real{1e-10}};

  int random_seed{1};
};

struct ShiftInvertLanczosResult {
  ShiftInvertStatus status{ShiftInvertStatus::not_converged};
  int order{0};
  int iterations{0};
  int converged{0};

  // Pencil eigenvalues nearest the shift, ascending.  Size is the number that
  // converged, which is `wanted` on success.
  std::vector<Real> eigenvalues{};

  // Lanczos residual bound for each returned eigenvalue, in the same order.
  std::vector<Real> residuals{};

  // Propagated from the block factorization of (K - sigma M).
  BlockTridiagonalStatus factorization_status{
      BlockTridiagonalStatus::success};
  int factorization_failure_block{-1};

  [[nodiscard]] bool ok() const noexcept {
    return status == ShiftInvertStatus::success;
  }
};

// `shifted` must be a factorization of (K - shift*M) whose partition describes
// the full pencil, and `mass` must be the dense column-major M of the same
// order on the same device.  `mass` is indexed in the same permuted ordering the
// partition assumes; no permutation is applied here.
//
// Synchronizes `stream` before returning.  Shape and ownership violations are
// programming errors and throw std::invalid_argument; numerical outcomes are
// returned in `status`.
[[nodiscard]] ShiftInvertLanczosResult solve_shift_invert_lanczos(
    const BlockTridiagonalFactorization& shifted,
    const backend::DeviceBuffer<Real>& mass,
    const ShiftInvertLanczosConfig& config,
    backend::stream_t stream = nullptr);

}  // namespace quasar::numerics
