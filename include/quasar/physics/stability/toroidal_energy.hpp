#pragma once

// Dense fixed-boundary ideal-MHD energy and inertia matrices in toroidal
// PEST coordinates.
//
// The continuum convention is fixed by docs/theory/toroidal_ideal_mhd_energy.rst:
//
//   lambda = psi_N,  theta = -theta*_stored,
//   perturbations proportional to exp[i(m theta - n phi)].
//
// SpectralDofLayout describes the unconstrained complex displacement space.
// FixedBoundaryDofMap removes only the normal displacement xi^lambda at the
// outer conducting surface.  Tangential edge displacements remain in the
// space.  Each retained complex coefficient is represented by two real DOFs
// with the physical convention
//
//   xi = c cos(m theta - n phi) + s sin(m theta - n phi),
//   z  = c - i s.
//
// Consequently a Hermitian entry h = a + i b is realified as
//
//       [ a   b ]
//       [-b   a ]
//
// for row coefficient (c,s) and column coefficient (c,s).  This sign is part
// of the public ABI; it must not be changed independently of eigenfunction
// reconstruction.

#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/stability/kernels.hpp"
#include "quasar/physics/stability/spectral_layout.hpp"
#include "quasar/physics/stability/toroidal_equilibrium.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace quasar::stability {

class FixedBoundaryDofMap {
 public:
  static constexpr std::size_t eliminated =
      std::numeric_limits<std::size_t>::max();

  explicit FixedBoundaryDofMap(const SpectralDofLayout& layout);

  [[nodiscard]] std::size_t full_complex_dof_count() const noexcept {
    return full_to_free_complex_.size();
  }
  [[nodiscard]] std::size_t complex_dof_count() const noexcept {
    return free_to_full_complex_.size();
  }
  [[nodiscard]] std::size_t dof_count() const noexcept {
    return 2 * complex_dof_count();
  }

  [[nodiscard]] std::size_t free_complex_dof(
      std::size_t full_complex_dof) const;
  [[nodiscard]] std::size_t full_complex_dof(
      std::size_t free_complex_dof) const;

  [[nodiscard]] std::size_t complex_dof(
      const SpectralDofLayout& layout, std::size_t radial, int m,
      DisplacementComponent component) const;
  [[nodiscard]] std::size_t dof(
      const SpectralDofLayout& layout, std::size_t radial, int m,
      DisplacementComponent component, FourierQuadrature quadrature) const;

  [[nodiscard]] bool is_eliminated(std::size_t full_complex_dof) const;
  [[nodiscard]] const std::vector<std::size_t>& full_to_free_complex()
      const noexcept {
    return full_to_free_complex_;
  }
  [[nodiscard]] const std::vector<std::size_t>& free_to_full_complex()
      const noexcept {
    return free_to_full_complex_;
  }

 private:
  std::vector<std::size_t> full_to_free_complex_{};
  std::vector<std::size_t> free_to_full_complex_{};
};

enum class ToroidalAssemblyStatus : std::uint64_t {
  success = 0,
  invalid_toroidal_mode = std::uint64_t{1} << 0,
  invalid_gamma = std::uint64_t{1} << 1,
  zero_flux_scale = std::uint64_t{1} << 2,
  unsupported_magnetic_axis = std::uint64_t{1} << 3,
  invalid_shape = std::uint64_t{1} << 4,
  invalid_topology = std::uint64_t{1} << 5,
  storage_overflow = std::uint64_t{1} << 6,
  storage_limit_exceeded = std::uint64_t{1} << 7,
  nonfinite_input = std::uint64_t{1} << 8,
  nonpositive_jacobian = std::uint64_t{1} << 9,
  nonpositive_density = std::uint64_t{1} << 10,
  inconsistent_field_pitch = std::uint64_t{1} << 11,
  inconsistent_flux_scale = std::uint64_t{1} << 12,
  non_pest_geometry = std::uint64_t{1} << 13,
  nonfinite_matrix = std::uint64_t{1} << 14,
  nonpositive_mass_diagonal = std::uint64_t{1} << 15,
  mass_not_positive_definite = std::uint64_t{1} << 16,
  invalid_metric = std::uint64_t{1} << 17,
  nonhermitian_matrix = std::uint64_t{1} << 18,
};

constexpr ToroidalAssemblyStatus operator|(ToroidalAssemblyStatus lhs,
                                            ToroidalAssemblyStatus rhs) {
  return static_cast<ToroidalAssemblyStatus>(
      static_cast<std::uint64_t>(lhs) | static_cast<std::uint64_t>(rhs));
}

constexpr ToroidalAssemblyStatus operator&(ToroidalAssemblyStatus lhs,
                                            ToroidalAssemblyStatus rhs) {
  return static_cast<ToroidalAssemblyStatus>(
      static_cast<std::uint64_t>(lhs) & static_cast<std::uint64_t>(rhs));
}

constexpr ToroidalAssemblyStatus& operator|=(ToroidalAssemblyStatus& lhs,
                                              ToroidalAssemblyStatus rhs) {
  lhs = lhs | rhs;
  return lhs;
}

[[nodiscard]] constexpr bool has_status(ToroidalAssemblyStatus value,
                                        ToroidalAssemblyStatus flag) {
  return (value & flag) != ToroidalAssemblyStatus::success;
}

struct ToroidalAssemblyConfig {
  int n_toroidal{1};
  Real gamma{Real{5} / Real{3}};

  // Relative validation tolerances for B^phi/B^theta=q, J B^theta=S, and
  // the PEST requirement that J/R^2 be a flux function.
  // The real spectral builder's analytic circular convergence gate reaches
  // approximately 1.3e-4 at 4096 contour rays / 256 Fourier points.  These
  // defaults accept that resolved geometry while still rejecting an
  // orientation error, whose signed residual is O(2).
  Real field_pitch_tolerance{Real{2e-4}};
  Real flux_scale_tolerance{Real{2e-4}};
  Real pest_tolerance{Real{2e-4}};

  // Relative mismatch allowed between independently assembled H_ab and
  // conjugate(H_ba), and between the final real matrix and its transpose.
  Real hermitian_tolerance{
      Real{128} * std::numeric_limits<Real>::epsilon()};

  // Limit for the two persistent dense real matrices.  Temporary validation
  // storage (the Hermitian-residual scratch and the Cholesky copy/workspace) is
  // intentionally excluded.  Zero disables the policy limit; arithmetic
  // overflow is always rejected independently.
  std::size_t max_dense_storage_bytes{std::size_t{8} << 30};
};

struct ToroidalAssemblyDiagnostics {
  ToroidalAssemblyStatus status{ToroidalAssemblyStatus::success};
  Real minimum_jacobian{0};
  Real minimum_density{0};
  Real maximum_field_pitch_deviation{0};
  Real maximum_flux_scale_deviation{0};
  Real maximum_pest_deviation{0};
  Real minimum_mass_diagonal{0};
  Real hermitian_residual{0};
  std::size_t first_invalid_metric_point{
      std::numeric_limits<std::size_t>::max()};
  int mass_failed_leading_minor_order{0};
  std::size_t dense_storage_bytes{0};

  [[nodiscard]] bool ok() const noexcept {
    return status == ToroidalAssemblyStatus::success;
  }
};

// Column-major real matrices suitable for the dense generalized symmetric
// eigensolver.  The scratch buffers are retained deliberately: launch
// semantics remain stream-ordered and no temporary device allocation is freed
// before the assembly kernel has consumed it.
struct ToroidalMatrixPair {
  backend::DeviceBuffer<Real> stiffness{};
  backend::DeviceBuffer<Real> inertia{};

  backend::DeviceBuffer<Real> q_lambda{};
  backend::DeviceBuffer<Real> jacobian_lambda{};
  backend::DeviceBuffer<Real> jacobian_theta{};

  backend::DeviceBuffer<int> occurrence_count{};
  backend::DeviceBuffer<int> occurrence_domain{};
  backend::DeviceBuffer<int> occurrence_local_node{};
  backend::DeviceBuffer<int> harmonic{};
  backend::DeviceBuffer<int> component{};

  std::vector<std::size_t> full_to_free_complex{};
  std::vector<std::size_t> free_to_full_complex{};
  int complex_order{0};
  int real_order{0};
  ToroidalAssemblyDiagnostics diagnostics{};
};

// Exact storage estimate for two `real_order x real_order` dense matrices,
// excluding the O(N) mapping and derivative scratch.  Arithmetic overflow is
// a normal preflight outcome and is returned as `storage_overflow`; this
// function never throws.
[[nodiscard]] ToroidalAssemblyStatus toroidal_dense_storage_bytes(
    std::size_t real_order, std::size_t& bytes) noexcept;

// Revalidate an existing matrix pair.  This independently checks real-matrix
// symmetry and performs a deterministic hipSOLVER Cholesky factorization on a
// copy of inertia.  The original matrices remain unchanged.  A positive
// `mass_failed_leading_minor_order` identifies the first leading principal
// minor that is not positive definite.
void launch_validate_toroidal_matrix_pair(
    ToroidalMatrixPair& matrices, Real hermitian_tolerance, stream_t stream);

// Assemble the compressible Glasser/Bernstein ideal-MHD plasma energy and
// inertia for one nonzero toroidal mode number.  Arithmetic and validation of
// sampled fields remain on device; only a compact diagnostic summary is
// copied back.  Shape/device-contract violations throw std::invalid_argument,
// while physical/topology validation failures are returned in
// out.diagnostics.status with empty matrices.  Rational interfaces retain the
// established all-component one-sided split for their tagged harmonic.  The
// fixed-boundary map eliminates only outer xi^lambda; the annular inner edge
// and tangential outer components retain their natural weak-form conditions.
void launch_assemble_fixed_boundary_toroidal_matrices(
    const ChebyshevBasis& basis, const RadialDomains& domains,
    const FluxCoordinateGrid& coords,
    const ToroidalEquilibriumFields& equilibrium,
    const SpectralDofLayout& layout, const ToroidalAssemblyConfig& config,
    ToroidalMatrixPair& out, stream_t stream);

}  // namespace quasar::stability
