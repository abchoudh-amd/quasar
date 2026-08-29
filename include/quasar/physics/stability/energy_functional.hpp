#pragma once

// Cylindrical Newcomb ideal-MHD energy functional.
//
// This is the scalar, fixed-conducting-boundary gate for the later toroidal
// delta-W operator.  The displacement variable is the PHYSICAL radial
// displacement xi_r, not r*xi_r, and the weak-form measure is plain dr:
//
//   K(xi_r, eta_r) = integral [f xi_r' eta_r' + g xi_r eta_r] dr.
//
// The SI coefficient convention is documented in
// docs/theory/newcomb_cylindrical_energy.rst.  In particular, this matrix does
// not include the positive pi/2 normalization in delta W / L; callers that
// need the physical energy per axial length apply that factor after forming
// xi_r^T K xi_r.  The current implementation is deliberately restricted to a
// cylindrical annulus r > 0; axis regularity is mode-dependent and is not
// represented by the fixed-endpoint scalar gate.

#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/stability/kernels.hpp"

#include <cstddef>

namespace quasar::stability {

// Fourier convention exp[i(m theta + k z)].  For a toroidal perturbation
// exp[i(m theta - n phi)] in the large-aspect-ratio limit, k = -n/R0.
struct CylindricalNewcombMode {
  int  m{0};
  Real k{0};
};

// Coefficients sampled at every LOCAL Chebyshev node.  Storage is
// domain-major and retains both copies of a subdomain interface, exactly like
// ChebyshevBasis::{nodes,weights}.  Each side always keeps its own quadrature
// contribution.  Its xi_r endpoint is merged at a regular interface and kept
// independent at an interface tagged rational for the assembled harmonic.
struct CylindricalNewcombCoefficients {
  CylindricalNewcombCoefficients() = default;
  explicit CylindricalNewcombCoefficients(std::size_t n_local_nodes) {
    resize(n_local_nodes);
  }

  void resize(std::size_t n_local_nodes);

  backend::DeviceBuffer<Real> f{};
  backend::DeviceBuffer<Real> g{};
};

// Dense fixed-boundary weak-form matrix.  `values` is column-major for direct
// consumption by hipSOLVER: entry (row, column) is
// values[row + column * n_xi_r_dofs].  The two conducting-wall endpoint DOFs
// are eliminated.  Duplicated subdomain-interface nodes share one DOF when
// regular and retain two one-sided DOFs for a tagged resonant harmonic.
struct CylindricalNewcombMatrix {
  CylindricalNewcombMatrix() = default;
  explicit CylindricalNewcombMatrix(int n_xi_r_dofs_in) {
    resize(n_xi_r_dofs_in);
  }

  void resize(int n_xi_r_dofs_in);

  backend::DeviceBuffer<Real> values{};
  // Persistent assembly map, one entry per local Chebyshev node.  Retaining
  // it here keeps launch semantics asynchronous; a temporary device buffer
  // would otherwise be destroyed before the kernel finished using it.
  backend::DeviceBuffer<int> local_to_dof{};
  int n_xi_r_dofs{0};
};

// Number of free xi_r DOFs after deterministic interface merging and
// elimination of xi_r(a)=xi_r(b)=0.  The one-argument overload is the
// backward-compatible all-regular topology.  The harmonic-aware overload
// splits exactly those internal breakpoints whose provenance contains `m`.
// Both validate the Chebyshev metadata and device-buffer sizes before
// returning.
int cylindrical_newcomb_fixed_boundary_dof_count(const ChebyshevBasis& basis);
int cylindrical_newcomb_fixed_boundary_dof_count(
    const ChebyshevBasis& basis, const RadialDomains& domains, int m);

// Evaluate Newcomb Eqs. (16) and (18), restored to SI units, at the local
// Chebyshev nodes.  r, B_theta, B_z, and dp_dr are device-only arrays with the
// same domain-major/local-node layout and size as basis.nodes.  `r` is the
// physical cylindrical radius and dp_dr is the derivative with respect to
// that radius, regardless of which coordinate labels basis.nodes.  The device
// validation reduces r > 0 to one status scalar; no profile array is staged
// through the host.
void launch_evaluate_cylindrical_newcomb_coefficients(
    const ChebyshevBasis& basis, const backend::DeviceBuffer<Real>& r,
    const backend::DeviceBuffer<Real>& b_theta,
    const backend::DeviceBuffer<Real>& b_z,
    const backend::DeviceBuffer<Real>& dp_dr, CylindricalNewcombMode mode,
    CylindricalNewcombCoefficients& out, stream_t stream);

// Assemble, for a general monotonically increasing basis coordinate s,
//
//   K_ab = integral [ (f/J) dL_a/ds dL_b/ds + (g J) L_a L_b ] ds,
//   J = dr/ds > 0,
//
// by Clenshaw-Curtis quadrature on each Chebyshev subdomain.  Requiring the
// explicit coordinate Jacobian prevents a psi_N basis from being silently
// interpreted as physical dr; a physical-radius basis passes J=1.  `r` is
// also supplied so the annulus restriction can be validated on device.
//
// Local interface nodes map through `domains`: regular interfaces share one
// xi_r DOF, while a breakpoint tagged with `m` has independent left and right
// DOFs.  Each matrix pair is owned by one thread, domains are accumulated in
// ascending order, and no atomics are used.
void launch_assemble_cylindrical_newcomb_matrix(
    const ChebyshevBasis& basis, const RadialDomains& domains, int m,
    const backend::DeviceBuffer<Real>& r,
    const backend::DeviceBuffer<Real>& dr_dcoordinate,
    const CylindricalNewcombCoefficients& coefficients,
    CylindricalNewcombMatrix& out, stream_t stream);

}  // namespace quasar::stability
