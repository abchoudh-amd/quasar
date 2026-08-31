#pragma once

// Device-resident construction of the seeded ideal-MHD conserved state.
//
// The six benchmark initial conditions used to be NumPy expressions in
// python/quasar/mhd/io.py, one per deck `initial.type`. They are analytic
// profiles evaluated once per padded cell, which makes them exactly the shape
// of work the rest of this tree runs on device: the deck layer supplies O(1)
// scalars, a kernel evaluates every cell, and the host keeps only the parameter
// validation and the throw.
//
// The name is still what the deck selects. `initial_condition_kind` maps the
// registered string to the enumerator the launch ABI carries, so adding a
// generator means adding an enumerator, a kernel branch, and a name -- the deck
// schema is untouched. A `Registry<IMhdInitialCondition>` would be the other
// option and is deliberately not used: a registry entry is a host object with a
// vtable, and a vtable cannot cross to the device (the same reason
// equilibrium's profiles are lowered to ProfileCoefficients). Selecting on an
// enumerator inside the kernel keeps one dispatch instead of one kernel launch
// per generator.
//
// Two properties of the original Python are preserved deliberately:
//
//   * The magnetic energy is re-formed from the FACE-to-CELL collocated field
//     rather than the raw face slots, using the solver's own
//     mhd_staggering.hpp quadrature (including its cylindrical radial moments).
//     The seeded energy and the solver EOS therefore see the same B.
//   * The positivity preflight runs on the exact conserved arrays that are
//     handed to the solver, not on the primitives they were built from, so a
//     loss of internal energy in the float64 assembly is reported here rather
//     than at the first CFL reduction.
//
// One property is deliberately NOT preserved: cell coordinates come from
// Grid2D::x_at_cell_center / y_at_cell_center, which are FMAs, where the Python
// used `origin + (i + 0.5) * d`. The results differ in the last bit. This is
// the better of the two -- it is the coordinate mapping the solver itself uses
// for the radius and for every geometric factor, so the seed is now consistent
// with the mesh it is seeded onto instead of merely close to it.

#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace quasar::mhd {

enum class MhdInitialConditionKind : int {
  brio_wu = 0,
  alfven_wave = 1,
  orszag_tang = 2,
  blast = 3,
  rotor = 4,
  confined_blob = 5,
};

// The deck-facing names, in enumerator order. Exposed so the Python validator
// and the bindings select against this list instead of mirroring it.
std::vector<std::string> registered_mhd_initial_conditions();

// Throws std::invalid_argument naming the offending string.
MhdInitialConditionKind initial_condition_kind(std::string_view name);

// Complete description of one seeded state.
//
// Every field is a scalar the deck already carries or an O(1) reduction of
// deck scalars; nothing here is per-cell. Fields are grouped by the generator
// that reads them, and a generator ignores the groups it does not name -- this
// is a tagged parameter block, not a union, because a union would cost the
// bindings a per-kind setter for no benefit at this size.
struct MhdInitialConditionSpec {
  MhdInitialConditionKind kind{MhdInitialConditionKind::brio_wu};

  // Padded storage layout and coordinates. `grid.nghost` is the halo the deck
  // resolved from its reconstruction scheme.
  Grid2D grid{};
  Real gamma{Real{5} / Real{3}};
  int cylindrical{0};
  // 2, 5 or 7. Selects the native radial collocation width independently of an
  // overpadded halo, exactly as the solver's own face-to-cell quadrature does.
  int scheme_order{2};
  // SI decks express B in tesla; the solver evolves B/sqrt(mu0). Applied to
  // every magnetic component after the analytic profile is formed and before
  // the energy is recollocated, matching the original ordering.
  Real magnetic_scale{Real{1}};

  // -- brio_wu --------------------------------------------------------------
  // Left state for x < `interface`, right state otherwise. Component order is
  // rho, p, vx, vy, vz, bx, by, bz.
  Real interface{Real{0.5}};
  Real left[8]{};
  Real right[8]{};

  // -- alfven_wave ----------------------------------------------------------
  Real rho{Real{1}};
  Real pressure{Real{0.1}};
  Real b0{Real{1}};
  // The reference axial field including any background contribution; only its
  // SIGN is used, to orient the +x-propagating eigenmode.
  Real total_b0{Real{1}};
  Real amplitude{Real{1e-6}};
  // Full wavelengths across lx.
  std::int64_t wavenumber{1};
  // 1 for a normalized deck, 1/sqrt(mu0) for SI. Distinct from
  // `magnetic_scale`: this one converts dB into a velocity, and it also enters
  // the sub-cell energy correction, so it is not recoverable from the other.
  Real magnetic_velocity_scale{Real{1}};

  // -- orszag_tang ----------------------------------------------------------
  // Reads `b_uniform[0]` as its b0 amplitude; rho and p are gamma-derived.

  // -- blast / rotor / confined_blob ---------------------------------------
  Real b_uniform[3]{};       // uniform bx, by, bz
  Real center[2]{};          // cx, cy
  Real r_in{Real{0.1}};      // blast disk radius
  Real r0{Real{0.1}};        // rotor rigid-body radius
  Real r1{Real{0.115}};      // rotor taper outer radius
  Real rho_in{Real{10}};
  Real rho_out{Real{1}};
  Real p_in{Real{1}};
  Real p_out{Real{0.1}};
  Real p_core{Real{10}};     // blast core pressure
  // Blast's ambient pressure, and the rotor's -- which is uniform, so the
  // rotor's deck key is plain `p`. One field rather than two because they are
  // the same quantity: the pressure outside the feature.
  Real p_ambient{Real{0.1}};
  Real rho_ambient{Real{1}};
  Real u0{Real{2}};          // rotor rim speed at r0
  Real blob_half{};          // confined_blob square half-width
};

// Build the conserved state on device.
//
// `out` must already be sized to `spec.grid`. On return every component holds
// the ghost-padded, solver-internal-unit conserved state in the same layout
// MhdSolver2D::seed_state consumes. Throws std::invalid_argument when the
// configuration is rejected or when the assembled state is inadmissible; in
// that case `out` is left with partial data and must not be seeded.
void build_initial_state(const MhdInitialConditionSpec& spec,
                         MhdField2D<Real>& out,
                         backend::stream_t stream = nullptr);

}  // namespace quasar::mhd
