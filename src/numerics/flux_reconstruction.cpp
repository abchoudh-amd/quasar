#include "quasar/numerics/flux_reconstruction.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/physics/mhd/kernels.hpp"

// -----------------------------------------------------------------------------
// Flux reconstruction schemes (MUSCL-minmod, MP5, MP7) -- THIN REGISTRY LAUNCHERS.
//
// Conservative finite-difference reconstruction in the Shu-Osher sense: the
// stored cell values are treated as point values of the reconstructed variable,
// and each scheme produces the LEFT-biased and RIGHT-biased interface point
// values that the (sibling) HLLD Riemann solver consumes.
//
// The actual reconstruction math lives ENTIRELY on the device, in
// launch_mhd_reconstruct (src/backend/hip/mhd/mhd_reconstruct.hip), which honors
// the per-scheme `scheme_order` (2 = MUSCL-minmod, 5 = MP5, 7 = MP7). These
// registry classes are stateless adapters: each maps its scheme to the right
// order and dispatches the device kernel, then synchronizes. No host compute is
// performed here -- the scalar MP helpers were hoisted to
// include/quasar/numerics/mp_limiter.hpp and are consumed device-side.
//
// References:
//   * MUSCL slope limiting (minmod): van Leer (1979), J. Comput. Phys. 32, 101.
//   * MP5: Suresh & Huynh (1997), "Accurate Monotonicity-Preserving Schemes with
//     Runge-Kutta Time Stepping", J. Comput. Phys. 136, 83-99.
//   * MP7: the 7th-order base interpolation wrapped in the same MP machinery.
//
// Reference state for the characteristic eigensystem (MP5/MP7): the arithmetic
// mean of the two cells adjacent to the interface. Normal-B (Bx for dir=0, By for
// dir=1) is the constrained-transport face quantity, taken from the staggered
// face storage and written verbatim into BOTH L and R states; it never passes
// through the 7-wave characteristic projection. Reconstruction is performed
// dimension-by-direction (each call reconstructs only along the requested normal
// `dir`); corner coupling is delegated to the CT/EMF stage.
// -----------------------------------------------------------------------------

namespace quasar::numerics {

namespace {

// Default-constructed background is inactive (B0 identically zero) -> the kernel
// takes the zero-background fast path. An all-zero BoundaryFlags4 selects the
// periodic / interior (two-sided) stencil. These registry launchers always use
// those defaults; the solver's hot path threads the real b0/flags directly into
// launch_mhd_reconstruct.
inline void dispatch_reconstruct(const quasar::mhd::MhdField2D<Real>& u, int dir,
                                 MhdInterfaceStates<Real>& out, int scheme_order,
                                 Real gamma) {
  const quasar::mhd::MhdBackgroundField<Real> b0{};  // inactive
  const quasar::mhd::BoundaryFlags4 flags{};         // all-zero (periodic)
  quasar::mhd::launch_mhd_reconstruct(u, b0, dir, out, scheme_order, flags, gamma,
                                      /*stream=*/nullptr);
  backend::device_synchronize(nullptr);
}

}  // namespace

// =============================================================================
// MUSCL-minmod (2nd-order, primitive-variable, slope-limited) -> scheme_order 2
// =============================================================================
class MusclMinmodRecon : public IFluxReconstruction {
 public:
  int  required_nghost() const override { return 2; }
  bool is_characteristic() const override { return false; }

  void reconstruct_faces(const quasar::mhd::MhdField2D<Real>& u, int dir,
                         MhdInterfaceStates<Real>& out, Real gamma) const override {
    dispatch_reconstruct(u, dir, out, /*scheme_order=*/2, gamma);
  }
};

// =============================================================================
// MP5 (Suresh-Huynh 5th-order monotonicity-preserving) -> scheme_order 5
// =============================================================================
class Mp5Recon : public IFluxReconstruction {
 public:
  int  required_nghost() const override { return 3; }
  bool is_characteristic() const override { return true; }

  void reconstruct_faces(const quasar::mhd::MhdField2D<Real>& u, int dir,
                         MhdInterfaceStates<Real>& out, Real gamma) const override {
    dispatch_reconstruct(u, dir, out, /*scheme_order=*/5, gamma);
  }
};

// =============================================================================
// MP7 (7th-order base interpolation + MP limiting) -> scheme_order 7
// =============================================================================
class Mp7Recon : public IFluxReconstruction {
 public:
  int  required_nghost() const override { return 4; }
  bool is_characteristic() const override { return true; }

  void reconstruct_faces(const quasar::mhd::MhdField2D<Real>& u, int dir,
                         MhdInterfaceStates<Real>& out, Real gamma) const override {
    dispatch_reconstruct(u, dir, out, /*scheme_order=*/7, gamma);
  }
};

}  // namespace quasar::numerics

QUASAR_REGISTER_FLUX_RECONSTRUCTION("muscl_minmod",
                                    ::quasar::numerics::MusclMinmodRecon)
QUASAR_REGISTER_FLUX_RECONSTRUCTION("mp5", ::quasar::numerics::Mp5Recon)
QUASAR_REGISTER_FLUX_RECONSTRUCTION("mp7", ::quasar::numerics::Mp7Recon)
