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

#include <string_view>

namespace quasar::numerics {

class IMhdBackgroundProfile {
 public:
  virtual ~IMhdBackgroundProfile() = default;

  // Return one background-field component for the staggered element whose
  // CENTER is (x, y):
  //   comp == 0 -> b0x, the x-face element centered at (x, y)
  //   comp == 1 -> b0y, the y-face element centered at (x, y)
  //   comp == 2 -> b0z, the cell element centered at (x, y)
  // The caller supplies the appropriate staggered center for the requested
  // component; the profile maps element+component to a field value.
  //
  // CONTRACT: the returned value is stored and consumed as that element's
  // DISCRETE FINITE-VOLUME MOMENT -- a face average for comp 0/1 and a cell
  // average for comp 2 -- not as a point value. The two are identical for any
  // profile that is affine over an element (both built-ins, "uniform" and
  // "linear_vacuum", are affine, so evaluating them at the supplied center is
  // exact). A profile with nonzero curvature over an element must therefore
  // return the element's average, e.g. by integrating its analytic form or by
  // applying the standard midpoint correction (h^2/24) * laplacian; returning a
  // bare midpoint value instead injects an O(h^2) projection error that caps the
  // scheme's order regardless of the selected reconstruction.
  virtual Real sample(int comp, Real x, Real y) const = 0;

  // Optional scalar configuration hook used by the Python deck's ``params``
  // mapping. Returning false reports an unknown parameter. Profiles with no
  // parameters inherit the default rejection; this keeps registry creation
  // default-constructible while making named analytic profiles fully usable.
  virtual bool set_parameter(std::string_view /*name*/, Real /*value*/) {
    return false;
  }

  // True only when every field produced by this analytic profile is curl-free
  // over the whole Cartesian domain. The solver uses this as a domain-wide
  // well-balanced proof. Explicit buffer overrides invalidate the analytic
  // proof unless the config separately requests and passes native curl
  // validation (used by the cylindrical vector-potential vacuum projection).
  virtual bool globally_curl_free() const noexcept { return false; }
};

}  // namespace quasar::numerics
