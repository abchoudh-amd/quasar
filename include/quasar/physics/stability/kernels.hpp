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

}  // namespace quasar::stability
