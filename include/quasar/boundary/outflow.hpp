#pragma once

#include "quasar/backend/memory.hpp"
#include "quasar/boundary/boundary_condition.hpp"

namespace quasar::boundary {

// First-order Mur (characteristic) outflow field wall: an open boundary that
// lets an outgoing wave leave with little reflection. fill_ghosts linearly
// continues the live boundary state for the curl; tangential E is then advanced
// by Mur after Ampere. The Mur update needs the previous-step wall and
// adjacent values, kept in a per-instance device strip. Each side gets its own
// OutflowFieldBC (the registry builds one per side), so the strip is per-face.
class OutflowFieldBC final : public IFieldBoundary {
 public:
  int ghost_continuation_mode() const noexcept override { return 2; }
  void configure_geometry(bool cylindrical) override { cylindrical_ = cylindrical; }
  void set_corner_skip(bool skip_lo, bool skip_hi) override {
    skip_lo_ = skip_lo;
    skip_hi_ = skip_hi;
  }
  void fill_ghosts(YeeField2D<Real>& field, Side side) const override;
  void correct_after_b(YeeField2D<Real>& field, Side side, Real dt) const override;
  void correct_after_e(YeeField2D<Real>& field, Side side, Real dt) const override;
  std::vector<Real> checkpoint_history() const override;
  bool checkpoint_history_primed() const noexcept override { return primed_; }
  void restore_checkpoint_history(std::span<const Real> history,
                                  bool primed) override;

 private:
  int history_stride(const Grid2D& grid, Side side) const;
  void ensure_history(const Grid2D& grid, Side side) const;
  bool cylindrical_{false};
  // Skip the doubly-tangential field at a low/high transverse endpoint shared
  // with another outflow; the solver's diagonal characteristic owns it.
  bool skip_lo_{false};
  bool skip_hi_{false};
  // History strip [a_u0, a_u1, b_u0, b_u1] for the two tangential-E components,
  // length 4*stride (including a transverse high face). Lazily allocated on
  // first use; `primed_` gates the one-time
  // seed pass that fills it from the live field without advancing.
  mutable backend::DeviceBuffer<Real> mur_{};
  mutable bool primed_{false};
};

}  // namespace quasar::boundary
