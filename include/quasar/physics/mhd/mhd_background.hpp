#pragma once

#include "quasar/backend/memory.hpp"  // backend::DeviceBuffer
#include "quasar/core/grid.hpp"       // Grid2D
#include "quasar/core/types.hpp"      // Real

#include <cstddef>
#include <string>
#include <unordered_map>

namespace quasar::mhd {

// Device-side lowering of an analytic background profile.
//
// numerics::IMhdBackgroundProfile is a host interface with a virtual sample(),
// and a vtable cannot cross to the device. Both registered profiles are affine
// over an element -- which is exactly the condition sample()'s own contract
// requires for the returned value to BE the element's finite-volume moment --
// so an affine parameter block reproduces them exactly. See
// physics/mhd/background_builder.hpp for how one is derived from a registered
// profile, and why a nonlinear profile is refused rather than linearized.
//
//   B_comp(x, y) = constant[comp] + slope_x[comp]*x + slope_y[comp]*y
//
// evaluated at the centre of the staggered element that owns the component.
struct MhdBackgroundAffineProfile {
  Real constant[3]{};
  Real slope_x[3]{};
  Real slope_y[3]{};
};

// Solver-owned static background magnetic field B0. The components use the same
// staggering as the evolved state's B in MhdField2D: the in-plane B0x/B0y live on
// cell faces (CT primary storage) and the out-of-plane toroidal B0z is
// cell-centered. Each buffer is sized to the grid's full padded storage
// (grid.storage_size()), allocated identically to MhdField2D's bx_face/by_face/
// bz_cell. When `active` is false the background is identically zero and solvers
// take a fast path that skips reading these buffers.
template <class T>
struct MhdBackgroundField {
  Grid2D grid{};
  backend::DeviceBuffer<T> b0x_face{};  // same staggering as MhdField2D::bx_face
  backend::DeviceBuffer<T> b0y_face{};  // same staggering as MhdField2D::by_face
  backend::DeviceBuffer<T> b0z_cell{};  // same staggering as MhdField2D::bz_cell
  bool active{false};                   // false => B0 identically zero (fast path)
  // Domain-wide proof supplied by an analytic profile or a validated explicit
  // vacuum field. When true, the static Maxwell stress is omitted in every cell
  // together; a cell-local curl-free decision would break shared-face
  // conservation at the boundary of a curl-free patch.
  bool globally_curl_free{false};

  MhdBackgroundField() = default;
  explicit MhdBackgroundField(Grid2D g)
    : grid{g},
      b0x_face{g.storage_size()}, b0y_face{g.storage_size()},
      b0z_cell{g.storage_size()} {}

  std::size_t component_size() const noexcept { return grid.storage_size(); }
};

// Deck-facing background specification. `profile` is a registry name selecting
// the background construction scheme (default: a constant uniform vector). The
// `params` carries scalar parameters for any registry-created analytic profile.
// The legacy bx0/by0/bz0 fields are the uniform-vector parameters consumed when
// profile == "uniform"; they override same-named entries in `params` so the
// native and Python construction paths have identical precedence.
struct MhdBackgroundSpec {
  bool enabled{false};
  std::string profile{"uniform"};  // registry name (default uniform vector)
  Real bx0{}, by0{}, bz0{};        // uniform-vector params (profile="uniform")
  std::unordered_map<std::string, Real> params{};
  // Constant applied to every native analytic-profile sample. This lets a
  // frontend convert the profile's output units without replacing the sampled
  // buffers and thereby discarding the profile's trusted curl-free capability.
  Real profile_scale{Real{1}};
  // Trusted domain-wide assertion that the supplied static field is curl-free;
  // it is construction metadata, not a property inferred from a relative
  // sample tolerance. The solver rejects obvious discrete contradictions as a
  // defense-in-depth check, then omits the analytically zero pure-B0 Lorentz
  // force. The vector-potential vacuum projection supplies this proof because
  // differencing its enormous Maxwell stress would otherwise create a
  // truncation-level self-force that can dwarf a low-beta plasma.
  bool curl_free{false};
};

}  // namespace quasar::mhd
