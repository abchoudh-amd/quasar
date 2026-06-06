// Translation-unit anchor for the MHD state module. All equation-of-state math
// (to_primitive/to_conserved/pressure/fast_magnetosonic_speed), the eigensystem,
// the characteristic projector, and the interface-state container are
// header-inline / header-template so they are callable from HIP device code and
// host alike. This .cpp gives the HIP-tagged numerics object a non-trivial body
// (the source list in src/numerics/CMakeLists.txt enumerates it) and provides
// the one out-of-line free function declared in mhd_state.hpp.

#include "quasar/numerics/mhd_state.hpp"

#include "quasar/numerics/characteristic_projection.hpp"
#include "quasar/numerics/interface_states.hpp"
#include "quasar/numerics/mhd_eigensystem.hpp"

namespace quasar::numerics {

const char* mhd_state_version() noexcept {
  return "quasar-mhd-state-1.0";
}

// Explicit instantiation of the interface-state container at the default
// precision, so its host accessors get a concrete out-of-line object here.
template struct MhdInterfaceStates<Real>;

}  // namespace quasar::numerics
