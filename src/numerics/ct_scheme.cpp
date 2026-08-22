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
// This translation unit is the registry-facing CT scheme, and it carries exactly
// one method: the div(B) diagnostic, a thin launcher over launch_mhd_ct_divb_linf
// that MhdSolver2D::divergence_b_max() reaches through the same kernel.
//
// The EMF construction and the face-B advance are NOT here. MhdSolver2D builds
// the corner EMF with launch_mhd_ct_emf_prepare/_finish -- passing its own
// background, boundary flags, and MP5/MP7 order -- and then advances face B as
// an ordinary residual component via launch_mhd_emf_curl_rate + rk_stage, so it
// rides the same SSP-RK convex combination as the other seven components. A
// virtual compute_emf/update_face_b pair on this class could only ever describe
// a reduced periodic/no-background second-order path that the solver does not
// take, and applying the curl directly to the field on top of the flux
// divergence is precisely the double count the residual routing avoids.
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

#include "quasar/backend/device.hpp"   // DeviceBuffer for the reduction scratch
#include "quasar/core/registry.hpp"
#include "quasar/physics/mhd/kernels.hpp"

namespace quasar::numerics {

// Concrete FD-CT scheme, registered "fd_ct_christlieb". Delegates to the device
// launcher declared in physics/mhd/kernels.hpp; the reduction writes its result
// to the host before returning, so no extra synchronize is needed here.
class ChristliebFdCt : public ICtScheme {
 public:
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
