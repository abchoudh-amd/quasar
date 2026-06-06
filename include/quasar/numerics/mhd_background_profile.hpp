#pragma once

// Background-field profile interface for the field-split ideal-MHD formulation
// B = B0 + b. A profile samples the static background field B0 on the staggered
// MHD mesh: face-centered normal components (b0x on x-faces, b0y on y-faces) and
// the cell-centered out-of-plane component (b0z). The MHD solver queries a
// concrete profile once to populate its background buffers; the perturbation b
// is then evolved against that fixed B0.
//
// Concrete profiles self-register by name via
// QUASAR_REGISTER_MHD_BACKGROUND_PROFILE so the input deck selects them by
// string (e.g. "uniform"). The interface is host-only: it is consumed at setup
// time on the host to fill device buffers, not called from device kernels.

#include "quasar/core/types.hpp"

namespace quasar::numerics {

class IMhdBackgroundProfile {
 public:
  virtual ~IMhdBackgroundProfile() = default;

  // Sample a background-field component at position (x, y).
  //   comp == 0 -> b0x, evaluated at the x-face location
  //   comp == 1 -> b0y, evaluated at the y-face location
  //   comp == 2 -> b0z, evaluated at the cell center
  // The caller supplies the appropriate staggered (x, y) for the requested
  // component; the profile only maps position+component to a field value.
  virtual Real sample(int comp, Real x, Real y) const = 0;
};

}  // namespace quasar::numerics
