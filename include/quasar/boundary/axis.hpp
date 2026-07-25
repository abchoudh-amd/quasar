#pragma once

#include "quasar/boundary/boundary_condition.hpp"

namespace quasar::boundary {

// On-axis (r=0) field boundary for the axisymmetric cylindrical m=0 scheme. It
// is auto-wired to the x_lo (i=0) side when geometry is cylindrical. The r=0
// parity closure keeps the 1/r FDTD curls regular on the axis: it runs as a
// ghost fill before each curl and as a boundary-node correction after each curl
// (the i=0 edge sits at r=0). The kernel applies the axis closure to the x_lo
// side unconditionally, so configure / set_corner_skip are no-ops and the
// per-call Side is only used to gate the closure to the x_lo face.
class AxisFieldBC final : public IFieldBoundary {
 public:
  void fill_ghosts(YeeField2D<Real>& field, Side side) const override;
  void correct_after_b(YeeField2D<Real>& field, Side side, Real dt) const override;
  void correct_after_e(YeeField2D<Real>& field, Side side, Real dt) const override;
};

// On-axis (r=0) particle boundary: crossing the axis is an azimuth shift by pi,
// so r -> -r and both (vr,vphi) change sign while vz is unchanged. fold_current
// is a no-op (the axis field closure already
// regularizes the on-axis current via the field parity).
class AxisParticleBC final : public IParticleBoundary {
 public:
  void apply(pic::ParticleSpecies& species, Side side) const override;
};

}  // namespace quasar::boundary
