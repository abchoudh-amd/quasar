#pragma once

// Backend-neutral declaration of the ideal-MHD stability kernel-launch ABI.
//
// Mirrors physics/equilibrium/kernels.hpp: a per-physics seam speaking
// stability types, with the definitions under src/backend/hip/stability/.
// Pure declarations -- no __global__, no <hip/hip_runtime.h>.
//
// -- Straight-field-line coordinates -------------------------------------------
// The delta-W operator lives in PEST coordinates (psi, theta*, phi), where
// theta* is chosen so field lines are straight:
//
//   dphi/dtheta* = q(psi)   (independent of theta*)
//
// The construction follows from the same integrand the safety factor already
// uses. Along a flux surface a field line advances toroidally at
//
//   dphi/dl = B_phi / (r B_pol)
//
// and one full poloidal circuit advances it by 2*pi*q, which IS the definition
// of q. So the running partial of that integral, normalized by q, is exactly
// the straight-field-line angle:
//
//   theta*(l) = (1/q) * integral_0^l  B_phi / (r B_pol)  dl'
//
// which runs from 0 to 2*pi over the circuit by construction. Nothing new has
// to be derived or fitted; theta* is the antiderivative of the quantity
// compute_q already accumulates, and that is worth knowing because it means the
// coordinate inherits the accuracy of the existing q calculation rather than
// introducing an independent error.
//
// -- Why this matters for the eigenproblem -------------------------------------
// In a geometric poloidal angle the field-line-bending operator couples every
// poloidal harmonic to every other, densely. In a straight-field-line angle the
// parallel gradient acting on exp(i(m*theta* - n*phi)) is proportional to
// (m - n*q), which is diagonal in m. That is what makes the discretized
// operator banded and well conditioned, and it is why published tokamak
// stability results are quoted in this convention.

#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

namespace quasar::stability {

using stream_t = ::quasar::backend::stream_t;
using ::quasar::numerics::EllipticGrid;

// Structured (psi, theta*) grid with the metric the energy functional needs.
//
// Storage is surface-major: entry (i, j) for radial index i and poloidal index
// j lives at i * n_theta + j.
struct FluxCoordinateGrid {
  FluxCoordinateGrid() = default;
  FluxCoordinateGrid(int n_psi, int n_theta) { resize(n_psi, n_theta); }

  void resize(int n_psi, int n_theta);

  // Radial labels and the safety factor on each surface.
  backend::DeviceBuffer<Real> psi_n{};   // n_psi
  backend::DeviceBuffer<Real> q{};       // n_psi
  backend::DeviceBuffer<int>  valid{};   // n_psi: 1 if the surface is usable

  // Geometry at uniform theta*.
  backend::DeviceBuffer<Real> r{};       // n_psi * n_theta
  backend::DeviceBuffer<Real> z{};

  // Mapping derivatives, needed for the metric and retained because the
  // assembly uses them directly rather than only through the metric.
  backend::DeviceBuffer<Real> dr_dpsi{};
  backend::DeviceBuffer<Real> dz_dpsi{};
  backend::DeviceBuffer<Real> dr_dtheta{};
  backend::DeviceBuffer<Real> dz_dtheta{};

  // Jacobian of (psi_n, theta*, phi) -> (r, phi, z), including the r factor
  // from the toroidal direction.
  backend::DeviceBuffer<Real> jacobian{};

  // Contravariant metric components. g^{phi phi} = 1/r^2 exactly and is not
  // stored.
  backend::DeviceBuffer<Real> g_psipsi{};
  backend::DeviceBuffer<Real> g_psitheta{};
  backend::DeviceBuffer<Real> g_thetatheta{};

  int n_psi{0};
  int n_theta{0};
};

// Build the coordinate system from traced flux surfaces and the field.
//
// `surfaces` must have been traced and reduced already; only closed surfaces
// are used, and an open one is marked invalid rather than silently producing
// garbage geometry.
//
// The theta* integration and the resampling onto uniform theta* run one thread
// per SURFACE: both are sequential walks around the contour, and the cumulative
// integral is a running sum whose order sets the result.
void launch_build_flux_coordinates(const EllipticGrid& g,
                                   const equilibrium::GsFluxSurfaces& surfaces,
                                   const equilibrium::GsMagneticField& field,
                                   Real axis_r, Real axis_z,
                                   FluxCoordinateGrid& out, stream_t stream);

// Verify the defining property: the field-line pitch dphi/dtheta* must equal q
// and be independent of theta*. Writes the maximum relative deviation for EACH
// surface into `d_deviation` (length coords.n_psi), with zero on surfaces
// marked invalid.
//
// Per surface rather than a single global maximum, and that is not a cosmetic
// choice. The surfaces nearest the magnetic axis are the worst-conditioned in
// the whole set -- B_poloidal appears in a denominator and vanishes at the axis
// -- so a global max is almost always reporting the innermost surface and says
// nothing about the bulk. Collapsing it too early hid exactly that during
// development, and made the error look as though it refused to converge under
// refinement.
//
// This exists as a kernel rather than only as a test because the deviation is
// resolution-dependent and a caller assembling delta-W deserves to know which
// surfaces its coordinate system is good enough on.
void launch_check_straightness(const FluxCoordinateGrid& coords,
                               const EllipticGrid& g,
                               const equilibrium::GsMagneticField& field,
                               Real* d_deviation, stream_t stream);

// -- Rational surfaces and the radial domain layout ----------------------------
//
// The eigenfunction is singular where q = m/n, so a single global Chebyshev
// expansion loses its spectral convergence exactly where the unstable modes
// localize. The fix is to break the radial domain at those surfaces, so each
// subinterval spans smooth data.
//
// Which surfaces are resonant depends on the toroidal mode number, so the
// layout is rebuilt per n rather than computed once.

struct RationalSurfaces {
  // Far more than any physical case needs: a resonance requires q to pass
  // through m/n, and q spans a range of a few over the whole plasma.
  static constexpr int kMaxRational = 64;

  Real psi_n[kMaxRational]{};  // ascending
  int  m[kMaxRational]{};      // poloidal number at each resonance
  int  count{0};
  bool overflow{false};
};

struct RadialDomains {
  static constexpr int kMaxDomains = 80;

  // Ascending, with breakpoints[0] == psi_lo and breakpoints[n_domains] ==
  // psi_hi. A Chebyshev expansion is built on each [k, k+1] interval.
  Real breakpoints[kMaxDomains + 1]{};
  int  n_domains{0};
  bool overflow{false};
};

// Locate every psi_n where q = m/n, by scanning adjacent surfaces for a
// crossing and interpolating within the bracketing pair.
//
// Single-threaded: the output is an ordered list built by a sequential scan,
// and the count is a running index. Parallelizing would need a prefix sum over
// a handful of entries to produce the same ordering.
void launch_locate_rational_surfaces(const FluxCoordinateGrid& coords,
                                     int n_toroidal, RationalSurfaces* d_out,
                                     stream_t stream);

// Lay out Chebyshev subintervals, breaking at each rational surface.
//
// `min_domains` forces a minimum subdivision so a mode number with no
// resonance in range still gets a usable layout rather than one interval
// spanning the whole plasma. Resonances closer together than `min_width` are
// merged: two breakpoints a fraction of a percent apart would create a
// subinterval too thin to carry a meaningful expansion, and would wreck the
// conditioning of the block it produces.
void launch_build_radial_domains(const RationalSurfaces* d_rational,
                                 Real psi_lo, Real psi_hi, int min_domains,
                                 Real min_width, RadialDomains* d_out,
                                 stream_t stream);

// -- Chebyshev spectral basis ---------------------------------------------------
//
// Each radial subinterval from RadialDomains carries a Chebyshev expansion.
// This provides the pieces the energy assembly needs on a subinterval: the
// Chebyshev-Gauss-Lobatto nodes, the spectral differentiation matrix, and
// Clenshaw-Curtis quadrature weights.
//
// -- Why Gauss-Lobatto and not Gauss --------------------------------------------
// Lobatto nodes include both endpoints. That is what allows continuity between
// adjacent subintervals to be imposed as a condition on shared node values
// rather than through an interpolation, which would introduce an error at every
// interface -- and the interfaces are placed at resonant surfaces, where the
// solution is least forgiving.
//
// -- Conditioning ---------------------------------------------------------------
// The differentiation matrix has a condition number growing like the SQUARE of
// the polynomial order, and applying it twice squares that again. This is the
// reason the energy functional is assembled in weak form: integrating by parts
// leaves only first derivatives, so the operator conditioning grows like
// order^2 rather than order^4. At order 64 that is the difference between
// losing about four digits and losing about eight.
//
// The differentiation matrix is built with the negative-sum trick -- the
// diagonal is set so each row sums to exactly zero -- because the closed-form
// diagonal entries suffer catastrophic cancellation at high order. Without it
// the matrix fails to differentiate a constant to zero, which is the one thing
// it must do exactly.

struct ChebyshevBasis {
  ChebyshevBasis() = default;
  ChebyshevBasis(int order, int n_domains) { resize(order, n_domains); }

  void resize(int order, int n_domains);

  // Per-domain physical node positions: domain d, node i at d * n_nodes + i.
  backend::DeviceBuffer<Real> nodes{};
  // Quadrature weights on the physical interval, same layout.
  backend::DeviceBuffer<Real> weights{};
  // Differentiation matrices in physical units, one dense (n_nodes x n_nodes)
  // block per domain: entry (i, j) of domain d at
  // d * n_nodes * n_nodes + i * n_nodes + j.
  backend::DeviceBuffer<Real> diff{};

  int order{0};      // polynomial order; n_nodes = order + 1
  int n_nodes{0};
  int n_domains{0};
};

// Build nodes, weights, and differentiation matrices for every subinterval of
// `domains`.
void launch_build_chebyshev_basis(const RadialDomains* d_domains,
                                  ChebyshevBasis& out, stream_t stream);

}  // namespace quasar::stability
