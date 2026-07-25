// FD-CT (Christlieb-style) constrained-transport scheme for ideal MHD.
//
// References:
//   - C. R. Evans & J. F. Hawley, "Simulation of magnetohydrodynamic flows: a
//     constrained transport method", ApJ 332 (1988) 659. (The curl-of-EMF face
//     update that keeps div(B) at round-off.)
//   - D. S. Balsara & D. S. Spicer, "A staggered mesh algorithm using high order
//     Godunov fluxes to ensure solenoidal magnetic fields in MHD simulations",
//     JCP 149 (1999) 270. (Construction of CT electric fields from Godunov
//     magnetic fluxes.)
//   - A. J. Christlieb et al., finite-difference CT family -- the corner EMF is
//     reconstructed from the same interface states the conservative flux uses, so
//     the induction update is consistent with the fluid update.
//
// This translation unit is the registry-facing CT scheme. Its three methods are
// thin launchers over the MHD HIP kernels. launch_mhd_ct_emf recomputes HLLD's
// directional magnetic fluxes at x/y faces, maps them to upwind face Ez, and
// interpolates them to corners (second order on this registry seam; the solver
// passes MP5/MP7 order explicitly). update_face_b exposes the direct curl update
// for standalone callers, while MhdSolver2D uses launch_mhd_emf_curl_rate and
// advances that rate inside the same SSP-RK stage as the fluid state. Both paths
// use the identical staggered curl, so div(curl E) telescopes to round-off.
//
// ---------------------------------------------------------------------------
// Staggering (matches mhd_field.hpp; see ct_scheme.hpp):
//   bx_face(i,j) = Bx on the x_lo (left)   face of cell (i,j)
//   by_face(i,j) = By on the y_lo (bottom) face of cell (i,j)
//   ez_edge(i,j) = Ez at the lower-left corner of cell (i,j)
//   divB(i,j) = (bx_face(i+1,j) - bx_face(i,j))/dx
//             + (by_face(i,j+1) - by_face(i,j))/dy
// ---------------------------------------------------------------------------

#include "quasar/numerics/ct_scheme.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"

namespace quasar::numerics {

// Concrete FD-CT scheme, registered "fd_ct_christlieb". Every method delegates to
// the device launchers declared in physics/mhd/kernels.hpp and synchronizes the
// default stream so the result is visible to the (host) caller before return.
class ChristliebFdCt : public ICtScheme {
 public:
  void compute_emf(const quasar::mhd::MhdField2D<Real>& u,
                   const MhdInterfaceStates<Real>& ifx,   // dir=0 faces
                   const MhdInterfaceStates<Real>& ify,   // dir=1 faces
                   quasar::mhd::EmfField2D<Real>& emf,
                   Real gamma) const override {
    // The registry CT path is the zero-background, all-periodic seam: the solver
    // residual path calls launch_mhd_ct_emf directly with its own b0/flags. Here
    // an inactive background and all-zero boundary flags reproduce the periodic
    // no-background EMF.
    const quasar::mhd::MhdBackgroundField<Real> b0{};   // inactive => zero-B0 fast path
    const quasar::mhd::BoundaryFlags4 flags{};          // all-zero => periodic
    quasar::mhd::launch_mhd_ct_emf(u, b0, ifx, ify, flags, emf, gamma, nullptr);
    backend::device_synchronize(nullptr);
  }

  void update_face_b(quasar::mhd::MhdField2D<Real>& u,
                     const quasar::mhd::EmfField2D<Real>& emf,
                     Real dt) const override {
    quasar::mhd::launch_mhd_face_b_update(u, emf, dt, nullptr);
    backend::device_synchronize(nullptr);
  }

  Real divergence_b_linf(const quasar::mhd::MhdField2D<Real>& u) const override {
    Real linf = Real{0};
    quasar::mhd::launch_mhd_ct_divb_linf(u, divb_scratch_, &linf, nullptr);
    return linf;
  }

 private:
  // Scheme-owned block-partials scratch for the div(B) device reduction, reused
  // across calls so the diagnostic does not hipMalloc/hipFree each time. mutable
  // because divergence_b_linf is const (it only reads the field).
  mutable backend::DeviceBuffer<Real> divb_scratch_{};
};

}  // namespace quasar::numerics

QUASAR_REGISTER_CT_SCHEME("fd_ct_christlieb", ::quasar::numerics::ChristliebFdCt)
