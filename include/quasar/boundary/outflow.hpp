#pragma once

#include "quasar/backend/memory.hpp"
#include "quasar/boundary/boundary_condition.hpp"

namespace quasar::boundary {

// First-order Mur (characteristic) outflow field wall: an open boundary that
// lets an outgoing wave leave with little reflection. fill_ghosts is a no-op;
// the boundary nodes are corrected after each curl (one-sided B closure +
// Mur-advanced tangential E). The Mur update needs the previous-step wall and
// adjacent values, kept in a per-instance device strip. Each side gets its own
// OutflowFieldBC (the registry builds one per side), so the strip is per-face.
class OutflowFieldBC final : public IFieldBoundary {
 public:
  void configure(int fdtd_order) override { order_ = fdtd_order; }
  void set_corner_skip(bool skip_lo, bool skip_hi) override {
    skip_lo_ = skip_lo;
    skip_hi_ = skip_hi;
  }
  void fill_ghosts(YeeField2D<Real>& field, Side side) const override {}
  void correct_after_b(YeeField2D<Real>& field, Side side, Real dt) const override;
  void correct_after_e(YeeField2D<Real>& field, Side side, Real dt) const override;

 private:
  int order_{2};
  // Set on a y-face: skip the doubly-tangential ez at the lo/hi corner column an
  // x-face already owns. Ignored on x-faces (they own their corners).
  bool skip_lo_{false};
  bool skip_hi_{false};
  // History strip [a_u0, a_u1, b_u0, b_u1] for the two tangential-E components,
  // length 4*line. Lazily allocated on first use; `primed_` gates the one-time
  // seed pass that fills it from the live field without advancing.
  mutable backend::DeviceBuffer<Real> mur_{};
  mutable bool primed_{false};
};

}  // namespace quasar::boundary
