#pragma once

// Device-resident construction of the static background field B0.
//
// This is the B0 half of what python/quasar/mhd/io.py used to do in NumPy:
// sample an analytic profile on the staggered padded mesh, or take the discrete
// curl of a corner vector potential (optionally after projecting it onto the
// discrete annular vacuum operator), then prove the result is discretely
// solenoidal and compatible with the configured boundary closure.
//
// The proof half already ran on device (WP2, launch_mhd_validate_background_*);
// what moves here is the construction. The two are deliberately kept adjacent:
// a background is only ever handed to the solver once it has passed, and the
// field never touches host memory in between.
//
// -- What stays on the host ---------------------------------------------------
// Reading an npz is file I/O, not calculation, so `a_file` and `file` are still
// parsed in Python. What changes is that the loaded array is uploaded once and
// every subsequent operation on it -- scaling, projection, curl, unit
// conversion, validation -- is a kernel.
//
// -- Analytic profiles --------------------------------------------------------
// numerics::IMhdBackgroundProfile is a host interface with a virtual sample(),
// and a vtable cannot cross to the device (the same constraint that lowers
// equilibrium's profiles to a ProfileCoefficients POD). Both registered
// profiles are affine over an element, which is exactly the condition their own
// sample() contract already requires for the returned value to BE the element's
// finite-volume moment. So the lowering is an affine one, and it is derived by
// probing the registry object rather than by re-listing the profiles here:
// there is one definition of "linear_vacuum", and it is the registered class.
// A profile that is not affine has no device path and is refused by name.

#include "quasar/backend/memory.hpp"
#include "quasar/core/field_source.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/field_evaluator.hpp"
#include "quasar/numerics/mhd_background_profile.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"

#include <string>

namespace quasar::mhd {

// Probe a configured profile at three points and check the affine form
// reproduces it at two more. Throws std::invalid_argument naming the profile if
// it does not; see the header note on why this is a refusal and not a fallback.
MhdBackgroundAffineProfile lower_affine_background_profile(
    const numerics::IMhdBackgroundProfile& profile, const std::string& name);

struct MhdBackgroundBuildSpec {
  Grid2D grid{};
  int cylindrical{0};
  // Deck tesla -> the solver's B/sqrt(mu0) variable.
  Real magnetic_scale{Real{1}};
  // Per-side homogeneous field closure used by the compatibility check, in
  // side order (x_lo, x_hi, y_lo, y_hi): 0 ignored, 1 periodic, 2 wall, 3 axis.
  int field_modes[4]{};

  // -- profile source --------------------------------------------------------
  MhdBackgroundAffineProfile profile{};

  // -- corner-potential source ----------------------------------------------
  // Uniform multiplier applied to the sampled A before anything else, and the
  // uniform out-of-plane component the curl cannot produce.
  Real b_scale{Real{1}};
  Real bz0{Real{0}};
  int vacuum_project{0};
};

// Sample the affine profile over the padded staggered meshes: b0x at the left
// face, b0y at the bottom face, b0z at the cell centre.
void build_background_from_profile(const MhdBackgroundBuildSpec& spec,
                                   MhdBackgroundField<Real>& out,
                                   backend::stream_t stream = nullptr);

// Scale, optionally project, and curl a lab-Y corner vector potential.
// `a_corners` holds (height+1)*(pitch+1) values in row-major [j][i] order over
// the padded corner grid; it is consumed, not retained.
void build_background_from_corner_potential(
    const MhdBackgroundBuildSpec& spec, backend::DeviceBuffer<Real>& a_corners,
    MhdBackgroundField<Real>& out, backend::stream_t stream = nullptr);

// Same, but with the corner potential evaluated from a conductor geometry
// instead of loaded. The evaluator is taken through the numerics interface, not
// as a concrete Biot-Savart object, so this module does not depend on
// magnetostatics: the deck layer picks the evaluator by registry name and hands
// it in. Nothing round-trips through host memory -- the corner grid expands
// into a device point cloud, the evaluator writes device planes, and the lab-Y
// plane feeds the same scale/project/curl chain as the file source.
void build_background_from_conductors(const MhdBackgroundBuildSpec& spec,
                                      const core::IFieldSource& conductors,
                                      const numerics::IFieldEvaluator& evaluator,
                                      MhdBackgroundField<Real>& out,
                                      backend::stream_t stream = nullptr);

// Apply the unit conversion to an already-assembled explicit field (the `file`
// deck source). Separated from the two constructions above because there is
// nothing to construct: the deck supplied every sample.
void scale_explicit_background(const MhdBackgroundBuildSpec& spec,
                               MhdBackgroundField<Real>& out,
                               backend::stream_t stream = nullptr);

// Discrete solenoidality and boundary-closure compatibility, with deck-facing
// messages. Every source above must pass this before the field is seeded; the
// solver runs the same sweeps again at construction, deliberately, because the
// deck path and the native path are separate entry points.
void validate_deck_background(const MhdBackgroundBuildSpec& spec,
                              const MhdBackgroundField<Real>& b0,
                              backend::stream_t stream = nullptr);

}  // namespace quasar::mhd
