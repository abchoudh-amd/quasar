#include "quasar/numerics/shift_invert_lanczos.hpp"

#include "backend/hip/numerics/linear_algebra_detail.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace quasar::numerics {
namespace {

// Counter-based hash producing a reproducible starting vector.  A library RNG
// would reproduce across runs but not necessarily across standard-library
// versions, and the Krylov space -- hence the Ritz values at a fixed iteration
// count -- depends on it.
Real start_component(std::uint64_t seed, std::uint64_t index) {
  std::uint64_t state = seed * 0x9e3779b97f4a7c15ull + index + 1ull;
  state ^= state >> 30;
  state *= 0xbf58476d1ce4e5b9ull;
  state ^= state >> 27;
  state *= 0x94d049bb133111ebull;
  state ^= state >> 31;
  // Map to (-1, 1) using 53 significand bits.
  const std::uint64_t mantissa = state >> 11;
  const Real unit = static_cast<Real>(mantissa)
                  / static_cast<Real>(std::uint64_t{1} << 53);
  return Real{2} * unit - Real{1};
}

struct SmallSymmetricEigen {
  std::vector<Real> values{};   // ascending, length n
  std::vector<Real> vectors{};  // n x n column-major
};

// Dense eigen-decomposition of the small (iterations x iterations) Lanczos
// tridiagonal matrix.  Running it through hipSOLVER rather than a hand-rolled
// QL keeps one implementation of the symmetric eigenproblem in the module.
SmallSymmetricEigen solve_small_symmetric(const std::vector<Real>& dense,
                                          int n, int device,
                                          backend::stream_t stream) {
  const auto entries = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  backend::DeviceBuffer<Real> matrix{
      entries, backend::uninitialized, backend::on_device(device)};
  backend::DeviceBuffer<Real> values{
      static_cast<std::size_t>(n), backend::uninitialized,
      backend::on_device(device)};
  backend::DeviceBuffer<int> device_info{1, backend::on_device(device)};
  matrix.copy_from_host_async(dense.data(), entries, stream);

  detail::SolverHandle solver{stream};
  int workspace_size = 0;
  detail::check_hipsolver(
      hipsolverDnDsyevd_bufferSize(
          solver.get(), HIPSOLVER_EIG_MODE_VECTOR, HIPBLAS_FILL_MODE_LOWER, n,
          matrix.device_ptr(), n, values.device_ptr(), &workspace_size),
      "syevd workspace query for the Lanczos tridiagonal");
  if (workspace_size < 0) {
    throw std::runtime_error{
        "hipSOLVER syevd workspace query returned a negative size"};
  }
  backend::DeviceBuffer<Real> workspace{
      static_cast<std::size_t>(workspace_size), backend::uninitialized,
      backend::on_device(device)};
  detail::check_hipsolver(
      hipsolverDnDsyevd(solver.get(), HIPSOLVER_EIG_MODE_VECTOR,
                        HIPBLAS_FILL_MODE_LOWER, n, matrix.device_ptr(), n,
                        values.device_ptr(), workspace.device_ptr(),
                        workspace_size, device_info.device_ptr()),
      "syevd for the Lanczos tridiagonal");
  const int info = detail::read_info(device_info, stream);
  if (info != 0) {
    throw std::runtime_error{
        "the Lanczos tridiagonal eigensolve failed with info "
        + std::to_string(info)};
  }

  SmallSymmetricEigen result;
  result.values.resize(static_cast<std::size_t>(n));
  result.vectors.resize(entries);
  values.copy_to_host_async(result.values.data(), result.values.size(),
                            stream);
  matrix.copy_to_host_async(result.vectors.data(), result.vectors.size(),
                            stream);
  backend::device_synchronize(stream);
  return result;
}

Real device_dot(detail::BlasHandle& blas, int order, const Real* a,
                const Real* b, backend::stream_t stream) {
  Real value = 0;
  detail::check_hipblas(hipblasDdot(blas.get(), order, a, 1, b, 1, &value),
                        "dot");
  backend::device_synchronize(stream);
  return value;
}

}  // namespace

ShiftInvertLanczosResult solve_shift_invert_lanczos(
    const BlockTridiagonalFactorization& shifted,
    const backend::DeviceBuffer<Real>& mass,
    const ShiftInvertLanczosConfig& config,
    backend::stream_t stream) {
  if (!shifted.ok()) {
    ShiftInvertLanczosResult failure;
    failure.status = ShiftInvertStatus::shifted_matrix_singular;
    failure.factorization_status = shifted.status;
    failure.factorization_failure_block = shifted.failure_block;
    failure.order = static_cast<int>(shifted.partition.total_rows());
    return failure;
  }
  if (shifted.partition.empty()) {
    throw std::invalid_argument{
        "solve_shift_invert_lanczos: the factorization has no partition"};
  }

  const std::size_t rows = shifted.partition.total_rows();
  if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument{
        "solve_shift_invert_lanczos: order exceeds the addressable range"};
  }
  const int order = static_cast<int>(rows);
  if (mass.size() != rows * rows) {
    throw std::invalid_argument{
        "solve_shift_invert_lanczos: mass must contain exactly order*order "
        "column-major values"};
  }
  if (mass.owner_device() != shifted.diagonal_lu.owner_device()) {
    throw std::invalid_argument{
        "solve_shift_invert_lanczos: mass and factors reside on different "
        "devices"};
  }
  if (config.wanted <= 0 || config.wanted > order) {
    throw std::invalid_argument{
        "solve_shift_invert_lanczos: wanted must be in [1, order]"};
  }
  if (!std::isfinite(config.shift)
      || !std::isfinite(config.residual_tolerance)
      || config.residual_tolerance < Real{0}) {
    throw std::invalid_argument{
        "solve_shift_invert_lanczos: shift and tolerance must be finite, with "
        "a non-negative tolerance"};
  }
  if (config.maximum_iterations < 0) {
    throw std::invalid_argument{
        "solve_shift_invert_lanczos: maximum_iterations must not be negative"};
  }

  const int requested_iterations =
      config.maximum_iterations > 0 ? config.maximum_iterations
                                    : 2 * config.wanted + 40;
  const int maximum_iterations = std::min(order, requested_iterations);

  ShiftInvertLanczosResult result;
  result.order = order;

  const int device = mass.owner_device();
  backend::DeviceGuard guard{device};
  detail::BlasHandle blas{stream};

  const auto order_size = static_cast<std::size_t>(order);
  // Basis vectors are stored as consecutive columns so the reorthogonalization
  // projections are a single gemv against the leading block of the basis.
  backend::DeviceBuffer<Real> basis{
      order_size * static_cast<std::size_t>(maximum_iterations),
      backend::uninitialized, backend::on_device(device)};
  backend::DeviceBuffer<Real> residual{
      order_size, backend::uninitialized, backend::on_device(device)};
  backend::DeviceBuffer<Real> mass_residual{
      order_size, backend::uninitialized, backend::on_device(device)};
  backend::DeviceBuffer<Real> work{
      order_size, backend::uninitialized, backend::on_device(device)};
  backend::DeviceBuffer<Real> projections{
      static_cast<std::size_t>(maximum_iterations), backend::uninitialized,
      backend::on_device(device)};

  std::vector<Real> host_start(order_size);
  for (std::size_t index = 0; index < order_size; ++index) {
    host_start[index] = start_component(
        static_cast<std::uint64_t>(config.random_seed), index);
  }
  residual.copy_from_host_async(host_start.data(), host_start.size(), stream);

  const Real one = Real{1};
  const Real zero = Real{0};
  const Real minus_one = Real{-1};

  // beta_0 = sqrt(r^T M r).  A non-positive value here is the mass matrix
  // failing to be positive definite on the starting vector.
  const auto apply_mass = [&](const Real* source, Real* destination) {
    detail::check_hipblas(
        hipblasDsymv(blas.get(), HIPBLAS_FILL_MODE_LOWER, order, &one,
                     mass.device_ptr(), order, source, 1, &zero, destination,
                     1),
        "mass product");
  };

  apply_mass(residual.device_ptr(), mass_residual.device_ptr());
  Real beta = device_dot(blas, order, residual.device_ptr(),
                         mass_residual.device_ptr(), stream);
  if (!(beta > Real{0}) || !std::isfinite(beta)) {
    result.status = ShiftInvertStatus::mass_not_positive_definite;
    return result;
  }
  beta = std::sqrt(beta);

  std::vector<Real> alpha_values;
  std::vector<Real> beta_values;
  alpha_values.reserve(static_cast<std::size_t>(maximum_iterations));
  beta_values.reserve(static_cast<std::size_t>(maximum_iterations));

  int iterations = 0;
  bool exhausted = false;
  std::vector<Real> host_projections(
      static_cast<std::size_t>(maximum_iterations));

  for (int step = 0; step < maximum_iterations; ++step) {
    Real* current = basis.device_ptr() + static_cast<std::size_t>(step)
                                             * order_size;
    // q_step = r / beta
    detail::copy_device_buffer_raw(current, residual.device_ptr(), order_size,
                                   stream);
    const Real inverse_beta = Real{1} / beta;
    detail::check_hipblas(
        hipblasDscal(blas.get(), order, &inverse_beta, current, 1), "scale");

    // w = (K - sigma M)^{-1} M q_step, reusing the block factors.
    apply_mass(current, work.device_ptr());
    const BlockTridiagonalSolveResult solve =
        solve_block_tridiagonal(shifted, work, 1, stream);
    if (!solve.ok()) {
      result.status = ShiftInvertStatus::shifted_matrix_singular;
      result.factorization_status = solve.status;
      result.factorization_failure_block = solve.failure_block;
      result.iterations = step;
      return result;
    }
    detail::copy_device_buffer_raw(residual.device_ptr(),
                                   solve.solution.device_ptr(), order_size,
                                   stream);

    // w <- w - beta * q_{step-1}
    if (step > 0) {
      const Real* previous =
          basis.device_ptr() + static_cast<std::size_t>(step - 1) * order_size;
      const Real negative_beta = -beta;
      detail::check_hipblas(
          hipblasDaxpy(blas.get(), order, &negative_beta, previous, 1,
                       residual.device_ptr(), 1),
          "three-term recurrence");
    }

    // alpha = q_step^T M w
    apply_mass(residual.device_ptr(), mass_residual.device_ptr());
    const Real alpha = device_dot(blas, order, current,
                                  mass_residual.device_ptr(), stream);
    if (!std::isfinite(alpha)) {
      result.status = ShiftInvertStatus::invariant_subspace_exhausted;
      result.iterations = step;
      return result;
    }
    alpha_values.push_back(alpha);

    const Real negative_alpha = -alpha;
    detail::check_hipblas(
        hipblasDaxpy(blas.get(), order, &negative_alpha, current, 1,
                     residual.device_ptr(), 1),
        "three-term recurrence");

    // Full reorthogonalization in the M inner product, applied twice.  One
    // pass is not enough once the basis starts to lose orthogonality: the
    // classical Kahan "twice is enough" argument is what keeps a converged
    // Ritz value from reappearing as a spurious duplicate.
    const int basis_size = step + 1;
    for (int pass = 0; pass < 2; ++pass) {
      apply_mass(residual.device_ptr(), mass_residual.device_ptr());
      detail::check_hipblas(
          hipblasDgemv(blas.get(), HIPBLAS_OP_T, order, basis_size, &one,
                       basis.device_ptr(), order,
                       mass_residual.device_ptr(), 1, &zero,
                       projections.device_ptr(), 1),
          "reorthogonalization projection");
      detail::check_hipblas(
          hipblasDgemv(blas.get(), HIPBLAS_OP_N, order, basis_size,
                       &minus_one, basis.device_ptr(), order,
                       projections.device_ptr(), 1, &one,
                       residual.device_ptr(), 1),
          "reorthogonalization update");
    }

    apply_mass(residual.device_ptr(), mass_residual.device_ptr());
    Real next_beta = device_dot(blas, order, residual.device_ptr(),
                                mass_residual.device_ptr(), stream);
    ++iterations;
    if (!std::isfinite(next_beta) || next_beta <= Real{0}) {
      exhausted = true;
      break;
    }
    next_beta = std::sqrt(next_beta);
    if (step + 1 < maximum_iterations) beta_values.push_back(next_beta);
    beta = next_beta;
  }

  result.iterations = iterations;
  if (iterations == 0) {
    result.status = ShiftInvertStatus::invariant_subspace_exhausted;
    return result;
  }

  // Assemble and diagonalize the Lanczos tridiagonal.
  const auto small = static_cast<std::size_t>(iterations);
  std::vector<Real> tridiagonal(small * small, Real{0});
  for (std::size_t index = 0; index < small; ++index) {
    tridiagonal[index + index * small] = alpha_values[index];
    if (index + 1 < small) {
      const Real off = beta_values[index];
      tridiagonal[(index + 1) + index * small] = off;
      tridiagonal[index + (index + 1) * small] = off;
    }
  }
  const SmallSymmetricEigen spectrum =
      solve_small_symmetric(tridiagonal, iterations, device, stream);

  // theta of largest magnitude are the pencil eigenvalues nearest the shift.
  std::vector<int> ordering(small);
  std::iota(ordering.begin(), ordering.end(), 0);
  std::sort(ordering.begin(), ordering.end(),
            [&spectrum](int a, int b) {
              const Real left = std::abs(spectrum.values[
                  static_cast<std::size_t>(a)]);
              const Real right = std::abs(spectrum.values[
                  static_cast<std::size_t>(b)]);
              if (left != right) return left > right;
              return a < b;
            });

  Real largest_theta = Real{0};
  for (const Real value : spectrum.values) {
    largest_theta = std::max(largest_theta, std::abs(value));
  }
  if (!(largest_theta > Real{0})) {
    result.status = ShiftInvertStatus::invariant_subspace_exhausted;
    return result;
  }

  // Residual bound for Ritz pair i is |beta_last * s(last, i)|.  When the
  // recurrence terminated on an exact invariant subspace there is no trailing
  // beta and every pair is exact.
  const Real trailing_beta = exhausted ? Real{0} : beta;

  struct Candidate {
    Real eigenvalue;
    Real residual;
    bool converged;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(small);
  for (const int index : ordering) {
    const auto slot = static_cast<std::size_t>(index);
    const Real theta = spectrum.values[slot];
    if (theta == Real{0}) continue;  // lambda would be infinite
    const Real last_component =
        spectrum.vectors[(small - 1) + slot * small];
    const Real bound = std::abs(trailing_beta * last_component);
    candidates.push_back(
        Candidate{config.shift + Real{1} / theta, bound / largest_theta,
                  bound <= config.residual_tolerance * largest_theta});
    if (static_cast<int>(candidates.size()) == config.wanted) break;
  }

  const bool too_few =
      static_cast<int>(candidates.size()) < config.wanted;
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              return a.eigenvalue < b.eigenvalue;
            });

  result.converged = 0;
  for (const Candidate& candidate : candidates) {
    result.eigenvalues.push_back(candidate.eigenvalue);
    result.residuals.push_back(candidate.residual);
    if (candidate.converged) ++result.converged;
  }

  result.status = (!too_few && result.converged >= config.wanted)
                      ? ShiftInvertStatus::success
                      : ShiftInvertStatus::not_converged;
  return result;
}

}  // namespace quasar::numerics
