#include "quasar/boundary/mhd_boundary.hpp"

#include "quasar/core/registry.hpp"
#include "quasar/physics/mhd/kernels.hpp"

// MHD fluid + magnetic-field boundary conditions for the 2D CT MHD slice.
//
// The conserved/CT buffers live in DeviceBuffer<Real>; the ghost-fill math runs
// on device in src/backend/hip/mhd/mhd_boundary_hip.hip (launch_mhd_fill_ghosts_
// fluid / _field), declared in quasar/physics/mhd/kernels.hpp. The six registry
// BC classes below are thin launchers: each fill_ghosts selects a mode
// (0=periodic, 1=outflow, 2=wall) and launches the matching device kernel
// for the requested side on the default stream.
//
// The launchers do NOT synchronize. Every kernel here and on the solver path
// shares the default stream, so the fills are correctly ordered against the
// reconstruction/flux kernels that read the ghosts and against each other (the
// x-sides complete before a y-side reads the x-ghost columns into the corner
// blocks). The solver issues a single device_synchronize at the end of
// compute_residual / combine_stage, so the per-side syncs that previously sat
// here were 8 redundant device-wide stalls per RK stage; dropping them keeps the
// fills batched behind that one sync like the other launchers. Any standalone
// reader (e.g. divergence_b_max, or a host stage-back) goes through a blocking
// copy_to_host on the default stream, which orders after the queued fills, so
// correctness does not depend on an explicit sync here.
//
// Side enum order (quasar::Side) is x_lo=0, x_hi=1, y_lo=2, y_hi=3, matching the
// `side` integer the kernels expect via static_cast<int>(side). The device
// kernels reproduce the index maps and the three critical correctness rules
// (shared face/cell storage extent, full-width y-side corner coverage, and the
// perfectly-conducting wall's odd sign on the normal component only — the wall
// enforces v·n=0 and n·B=0); see the .hip for details.

namespace quasar::boundary {
namespace {

using mhd::MhdField2D;

// Mode codes threaded into the device kernels.
constexpr int kPeriodic = 0;
constexpr int kOutflow = 1;
constexpr int kWall = 2;  // perfectly-conducting wall: v·n=0, n·B=0
constexpr int kAxis = 3;  // r=0: (vr,vphi,Br,Bphi) odd; axial/scalars even

// ---------------------------------------------------------------------------
// Fluid boundaries
// ---------------------------------------------------------------------------

class PeriodicFluidBC final : public IMhdFluidBoundary {
 public:
  void fill_ghosts(MhdField2D<Real>& f, Side side) const override {
    mhd::launch_mhd_fill_ghosts_fluid(f, static_cast<int>(side), kPeriodic, nullptr);
  }
};

class OutflowFluidBC final : public IMhdFluidBoundary {
 public:
  void fill_ghosts(MhdField2D<Real>& f, Side side) const override {
    mhd::launch_mhd_fill_ghosts_fluid(f, static_cast<int>(side), kOutflow, nullptr);
  }
};

class WallFluidBC final : public IMhdFluidBoundary {
 public:
  void fill_ghosts(MhdField2D<Real>& f, Side side) const override {
    mhd::launch_mhd_fill_ghosts_fluid(f, static_cast<int>(side), kWall, nullptr);
  }
};

class AxisFluidBC final : public IMhdFluidBoundary {
 public:
  void fill_ghosts(MhdField2D<Real>& f, Side side) const override {
    if (side == Side::x_lo) {
      mhd::launch_mhd_fill_ghosts_fluid(f, static_cast<int>(side), kAxis, nullptr);
    }
  }
};

// ---------------------------------------------------------------------------
// Magnetic-field boundaries
// ---------------------------------------------------------------------------

class PeriodicFieldBC final : public IMhdFieldBoundary {
 public:
  void fill_ghosts(MhdField2D<Real>& f, Side side) const override {
    mhd::launch_mhd_fill_ghosts_field(f, static_cast<int>(side), kPeriodic, nullptr);
  }
};

class OutflowFieldBC final : public IMhdFieldBoundary {
 public:
  void fill_ghosts(MhdField2D<Real>& f, Side side) const override {
    mhd::launch_mhd_fill_ghosts_field(f, static_cast<int>(side), kOutflow, nullptr);
  }
};

class WallFieldBC final : public IMhdFieldBoundary {
 public:
  void fill_ghosts(MhdField2D<Real>& f, Side side) const override {
    mhd::launch_mhd_fill_ghosts_field(f, static_cast<int>(side), kWall, nullptr);
  }
};

class AxisMhdFieldBC final : public IMhdFieldBoundary {
 public:
  void fill_ghosts(MhdField2D<Real>& f, Side side) const override {
    if (side == Side::x_lo) {
      mhd::launch_mhd_fill_ghosts_field(f, static_cast<int>(side), kAxis, nullptr);
    }
  }
};

}  // namespace

QUASAR_REGISTER_MHD_FLUID_BOUNDARY("periodic", PeriodicFluidBC)
QUASAR_REGISTER_MHD_FLUID_BOUNDARY("outflow", OutflowFluidBC)
QUASAR_REGISTER_MHD_FLUID_BOUNDARY("wall", WallFluidBC)
QUASAR_REGISTER_MHD_FLUID_BOUNDARY("axis", AxisFluidBC)

QUASAR_REGISTER_MHD_FIELD_BOUNDARY("periodic", PeriodicFieldBC)
QUASAR_REGISTER_MHD_FIELD_BOUNDARY("outflow", OutflowFieldBC)
QUASAR_REGISTER_MHD_FIELD_BOUNDARY("wall", WallFieldBC)
QUASAR_REGISTER_MHD_FIELD_BOUNDARY("axis", AxisMhdFieldBC)

bool mhd_boundary_is_periodic(const std::string& name) { return name == "periodic"; }

}  // namespace quasar::boundary
