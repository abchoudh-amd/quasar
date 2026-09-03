#pragma once

// Device-resident construction of the seeded initial electromagnetic field.
//
// The three deck `fields.initial.type` generators used to be NumPy expressions
// in python/quasar/pic/cli.py, evaluated on the host and pushed through
// seed_field(), which is a bare copy_from_host. They are analytic profiles
// evaluated once per Yee lattice point, which makes them exactly the shape of
// work the rest of this tree runs on device -- and the direct analogue of
// physics/mhd/initial_conditions.hpp, which was moved here first.
//
// This is not only a placement question. The PIC particle sampler carries an
// explicit guarantee that "there is deliberately no second host sampler", so one
// deck seed gives one sample whether the run is serial or distributed. The
// FIELD seed had no such guarantee: the distributed runner reused the same host
// code against a sink, which meant every rank materialized the full GLOBAL
// lattices before slicing out its tile. Parameterizing the kernel by the tile
// grid removes both the second definition and that materialization.
//
// The name is still what the deck selects. `pic_initial_field_kind` maps the
// registered string to the enumerator the launch ABI carries. A
// `Registry<IPicInitialField>` is deliberately not used, for the same reason
// initial_conditions.hpp gives: a registry entry is a host object with a vtable,
// and a vtable cannot cross to the device. Selecting on an enumerator inside the
// kernel keeps one dispatch instead of one launch per generator.
//
// -- What stays on the host ---------------------------------------------------
// Everything that is O(1) in the deck: parsing, the unit conversion and its
// representability check, the per-generator validity refusals (component,
// geometry, boundary, mode range, discrete-dispersion branch), and the handful
// of scalars those refusals compute along the way -- omega*dt, the Yee
// derivative symbol, the direction ratios, the Bessel root. Those are
// transcendentals of the MODE NUMBERS, not of position, so they are not per-cell
// work and they belong where the error messages are.
//
// -- The cylindrical B_phi half-step ------------------------------------------
// The solver stores B at t=-dt/2 while E is at t=0, so a standing cavity mode
// needs Bphi = -(dt/2) D_r Ez rather than zero. The host code formed that radial
// derivative through a helper that HAND-REIMPLEMENTED the axis-even /
// outer-PEC-odd parity continuation -- a while-loop index reflection with a sign
// flip -- and asserted in a comment that it matched the live boundary kernel,
// with nothing enforcing it.
//
// It is done differently here, and that is the main correctness point of this
// header: seed Ez, run the CONFIGURED field-boundary ghost fill, then take the
// derivative off the ghost-filled array. The parity is then whatever the
// registered closure actually says it is, by construction, and there is no
// second definition left to drift.

#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace quasar::pic {

enum class PicInitialFieldKind : int {
  seed_perturbation = 0,
  seed_tm_cavity = 1,
  seed_em_wave = 2,
};

// Storage components in YeeField2D declaration order. These are STORAGE names:
// on a cylindrical (r,z) deck the physical Ez is storage `ey` and the physical
// Bphi is storage `bz`, exactly as the deck layer has always spelled them.
enum class PicFieldComponent : int {
  ex = 0,
  ey = 1,
  ez = 2,
  bx = 3,
  by = 4,
  bz = 5,
};

// The deck-facing names, in enumerator order. Exposed so the Python validator
// and the bindings select against this list instead of mirroring it.
std::vector<std::string> registered_pic_initial_fields();

// Throws std::invalid_argument naming the offending string.
PicInitialFieldKind pic_initial_field_kind(std::string_view name);

// Complete description of one seeded field state.
//
// Every field is a scalar the deck already carries or an O(1) reduction of deck
// scalars; nothing here is per-cell. Fields are grouped by the generator that
// reads them, and a generator ignores the groups it does not name -- a tagged
// parameter block, not a union, matching MhdInitialConditionSpec.
struct PicInitialFieldSpec {
  PicInitialFieldKind kind{PicInitialFieldKind::seed_perturbation};

  // Padded storage layout. On a distributed run this is the TILE grid, so each
  // rank fills only what it owns.
  Grid2D grid{};
  int cylindrical{0};

  // Electric component the deck named, as a storage index.
  int component{static_cast<int>(PicFieldComponent::ez)};
  // Magnetic partner this generator writes, or -1 for none. Cartesian
  // seed_perturbation seeds E alone; the cylindrical branch defers its magnetic
  // partner to the half-step pass below, because that one needs filled ghosts.
  int magnetic_component{-1};

  // Full wavelengths across the x extent, and the transverse mode for the
  // cavity. mode_x doubles as the radial Bessel mode index on a cylindrical
  // seed_perturbation.
  int mode_x{1};
  int mode_y{0};

  // Already converted to the solver's normalized units by the deck layer.
  Real amplitude{Real{0}};

  // -- seed_perturbation (cylindrical) --------------------------------------
  // j_{0,mode_x}, the mode_x-th zero of J0, which puts the node at the outer
  // PEC wall.
  Real bessel_root{Real{0}};

  // -- seed_tm_cavity -------------------------------------------------------
  // sin(omega*dt/2): the magnetic half-step amplitude.
  Real half_time{Real{0}};
  // Already normalized by their own hypot, so the kernel does no division.
  Real direction_x{Real{0}};
  Real direction_y{Real{0}};

  // -- seed_em_wave ---------------------------------------------------------
  // omega*dt/2, the phase the magnetic partner leads by.
  Real magnetic_phase{Real{0}};
  // +1 or -1: which sign the magnetic partner carries for the chosen
  // polarization (Ez pairs with -By, Ey with +Bz).
  Real magnetic_sign{Real{1}};
};

// Second pass for the cylindrical standing mode: Bphi = -(dt/2) * D_r Ez,
// evaluated on the GHOST-FILLED electric component so the radial stencil reads
// the configured boundary closure rather than a private parity rule.
//
// Run only after the caller has filled the field ghosts. The stencil reaches
// two cells past the axis at fourth order, which is why the solver's halo is the
// thing that has to be wide enough, not this spec.
struct PicRadialHalfStepSpec {
  Grid2D grid{};
  // 2 or 4; selects the staggered radial derivative width.
  int fdtd_order{2};
  // -dt/2. Kept as the raw half-step rather than folded into a single constant
  // so the kernel forms the derivative first and scales second, in the order
  // the host expression did.
  Real half_dt{Real{0}};
  // Radial cell spacing in the solver's normalized length units.
  Real dr{Real{1}};
  int source_component{static_cast<int>(PicFieldComponent::ey)};
  int target_component{static_cast<int>(PicFieldComponent::bz)};
};

}  // namespace quasar::pic
