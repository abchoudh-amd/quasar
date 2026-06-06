// Uniform (spatially constant) background-field profile for the field-split
// ideal-MHD formulation B = B0 + b. UniformBackgroundProfile returns a fixed
// value per component independent of position, so sample(comp, x, y) ignores
// (x, y). The registry default-constructs it under the name "uniform"; the
// default-constructed object is the zero field (B0 == 0), which makes the
// field-split formulation degenerate to the plain (non-split) MHD equations.
// The per-component constants are implementation-internal and settable so a
// driver may pin a non-zero uniform background after create("uniform").

#include "quasar/numerics/mhd_background_profile.hpp"

#include "quasar/core/registry.hpp"
#include "quasar/core/types.hpp"

namespace quasar::numerics {

// File-local concrete profile, registered by name "uniform". Spatially constant:
// sample() returns the stored per-component constant regardless of (x, y). The
// registry-created default is the zero field; set_uniform() lets a driver pin a
// non-zero uniform background.
class UniformBackgroundProfile : public IMhdBackgroundProfile {
 public:
  void set_uniform(Real b0x, Real b0y, Real b0z) {
    b0x_ = b0x;
    b0y_ = b0y;
    b0z_ = b0z;
  }

  Real sample(int comp, Real /*x*/, Real /*y*/) const override {
    switch (comp) {
      case 0:  return b0x_;  // b0x on x-faces
      case 1:  return b0y_;  // b0y on y-faces
      case 2:  return b0z_;  // b0z at cell centers
      default: return Real{0};
    }
  }

 private:
  Real b0x_{Real{0}};
  Real b0y_{Real{0}};
  Real b0z_{Real{0}};
};

}  // namespace quasar::numerics

QUASAR_REGISTER_MHD_BACKGROUND_PROFILE("uniform", ::quasar::numerics::UniformBackgroundProfile)
