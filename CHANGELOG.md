# Changelog

All notable changes to Quasar are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/), and the project is pre-1.0 so
interfaces may still change between entries.

## [Unreleased]

### Added
- PIC: `outflow` field boundary kind (`boundary.field: outflow`) — a first-order
  characteristic (Mur) open wall that lets outgoing radiation leave with little
  reflection, at both 2nd- and 4th-order FDTD. Stable as an outflow channel
  (outflow on one axis) or mixed with `pec` walls; an open box with outflow on
  all four sides is not yet stable (first-order Mur corners need a dedicated
  corner closure).
- PIC: support for `units: normalized` decks and physically consistent `units: SI`
  decks. SI decks are non-dimensionalized through `Normalization` before stepping
  (grid, dt, charge/mass/velocity/density, external field) and diagnostics are
  converted back to SI on output.
- PIC field boundary conditions are live: per-side `boundary.field` (`periodic` /
  `pec` / `outflow`) selectable from the deck, with stable energy-conserving PEC
  reflection.
- PIC current-smoothing filter pipeline is wired from the deck
  (`numerics.current_filter`: `binomial` / `compensated_binomial`, with passes).
- Field-evaluator selection by registry name: `external_field.evaluator.type` may
  be `biot_savart`, `uniform`, `dipole`, or `gradient`; the coil CLI selects its
  evaluator through the registry too.
- PIC `fields.initial` seeding (`seed_perturbation`, `seed_em_wave`) and field-only
  decks (a deck may define species, an external field, or an initial field seed).
- All nine PIC example decks (`two_stream`, `filtered_two_stream`, `landau_damping`,
  `weibel`, `em_wave_propagation`, `beam_in_channel`, `pec_cavity`,
  `magnetized_plasma`, `coil_confinement`) are now runnable, documented, and covered
  by integration tests.
- PIC test coverage: a real FDTD plane-wave dispersion check, a behavioral
  mixed-boundary test, Gauss-residual diagnostics tests, an end-to-end CLI run
  smoke test, and a `--log-every` / `--write-every` diagnostics test.
- CPU-only registry-linkage tests for the current-filter and
  field-solver/pusher/deposit registries (a dropped registration now fails in the
  standard suite, not only at device-build time), and coil deck-parse tests for
  the `plane` observation and every geometry type.
- PIC: per-component field-to-host accessors on the solver
  (`field_component_to_host`, `external_field_component_to_host`) so diagnostics
  copy only the requested Yee components from the device instead of the whole
  six-field dict every snapshot.

### Fixed
- PIC: quiet-start macro-particle weighting no longer biases the initial number
  density when `n_particles` is not a perfect square. The block layout uses
  `side = ceil(sqrt(N))` points per axis and truncates to `N`; the weight is now
  `density * cell_area` (one layout cell per particle) instead of
  `density * block_area / N`, so the seeded density matches the deck value
  regardless of `N`.
- PIC: the particle field gather now honors per-axis boundaries. It previously
  wrapped the interpolation stencil periodically on every axis; on a non-periodic
  (wall) axis it now clamps into the boundary ghost layer (matching the deposit),
  so near-wall particles see the boundary field instead of the wrapped far edge.
  The sampled external field's ghost layer is edge-replicated to match.
- Coil: `observation.plane` now normalizes the `u_axis_xyz` / `v_axis_xyz`
  direction vectors before scaling by the extent, so a non-unit axis no longer
  silently changes the sampled plane size; a zero axis raises a clear error.
- PIC: charge conservation across particle boundary crossings. The periodic-wrap
  kernel co-shifts the previous particle position, and specular reflection now
  deposits into ghost cells with an image-charge fold-back instead of mirroring the
  previous position outside the domain (which had teleported current to the far
  edge via the periodic wrap).
- PIC: the FDTD curls are adjoint (forward E-update / backward B-update). With both
  curls forward, a hard field wall was exponentially unstable; the adjoint form is
  the standard Yee scheme and leaves periodic results unchanged.
- Magnetostatics: `helix`/`solenoid` vertex count is computed in `size_t` with an
  upper bound, fixing signed-int overflow on large `n_turns * n_segments_per_turn`.
- PIC: the sampled external field is now node-collocated like the rest of the
  solver. The external-field sampler previously placed `Ex`/`Ey` on the true
  Yee-staggered edges while the particle gather reads every component from the
  cell node, biasing the external electric force on every particle by a half-cell
  interpolation error; all six components are now sampled at the cell node.
- `quasar.pic.cli` persists the Yee ghost width (`nghost`) into `out.npz`, and
  `quasar.pic.postprocess.reshape_with_ghost` strips the halo using that explicit
  value instead of reverse-engineering it from the flat buffer size.
- Registry: type-keyed factory registration. Stateless `make_unique` lambdas were
  collapsed by identical-code folding, so `Registry::create(name)` could return
  the wrong concrete type — the root cause of the long-standing PIC field/particle
  boundary "heisenbug". Boundary dispatch now goes through the interface.
- PIC: the periodic particle wrap is side/axis-aware, so periodic and
  non-periodic (e.g. absorbing) walls can coexist per side.
- PIC: current filters now honor boundary periodicity instead of always wrapping
  across the far edge; non-periodic axes clamp the filter stencil at the wall.
- PIC: oversized particle displacements in the Esirkepov deposit now fail loudly
  instead of silently truncating the current-deposition window.
- PIC: `dipole` and `gradient` external-field deck evaluators now require and use
  their evaluator parameters rather than default-constructing zero/trivial fields.
- PIC: `PicDeck.validate()` now rejects non-finite (`NaN`/`inf`) and out-of-range
  deck values up front — domain lengths/origin, normalization reference density,
  `dt_s`, species charge/mass/density/temperature/drift, external-field vectors,
  initial-field amplitude, diagnostics cadence, and unknown `diagnostics.fields`
  names — instead of letting them propagate into the solver. `diagnostics.fields`
  names are normalized to lowercase on every construction path, and
  `--steps-override` only accepts a positive integer.
- Coil: `observation.points` decks are now bounded by `MAX_OBSERVATION_POINTS`
  like the grid/plane/line observation kinds.

### Changed
- PIC field boundaries are now imposed by one-sided / characteristic node
  corrections after each curl rather than by filling a ghost halo. `pec` keeps
  its reflecting, energy-conserving behavior (tangential E / normal B pinned;
  remaining components closed with one-sided stencils) at both FDTD orders; the
  4th-order boundary closure is locally reduced to 2nd order on the outer two
  layers (interior 4th-order convergence is preserved). Periodic still uses the
  ghost wrap.
- The kernel-launch ABIs moved to installed public headers; the
  physics/boundary/numerics modules no longer reach into the private
  `src/backend/hip/` tree (the blanket `src/` include path was removed from the
  module CMake helper and re-granted only to the backend HIP modules). The PIC
  ABI now lives at `include/quasar/physics/pic/kernels.hpp` (a per-physics seam)
  so the backend axis stays physics-neutral; `magnetostatics_kernels.hpp` keeps
  its raw-pointer form under `include/quasar/backend/`.
- The numerics `IFieldEvaluator` takes an axis-neutral `core::IFieldSource`
  (implemented by `magnetostatics::ConductorSystem`) instead of naming the
  magnetostatics type directly, so the analytic-field evaluators no longer depend
  on the magnetostatics module. The Python `IFieldEvaluator.evaluate_B` /
  `evaluate_grad_B` binding accepts any `IFieldSource` polymorphically rather than
  the concrete `ConductorSystem`.
- `IFieldEvaluator` gains a `configure(params)` seam (a name→flat-float-list
  parameter map) applied after registry construction, so the PIC driver builds
  every external-field evaluator purely by registry name and configures it from
  the deck — the previous per-type `if/elif` ladder in `quasar.pic.cli` that
  hand-picked a constructor is gone, matching the registry-only coil CLI.
- Backend headers under `include/quasar/backend/` are now HIP-free: an opaque
  `stream_t` and backend-neutral device-memory free functions, with all HIP calls
  confined to `src/backend/hip/`. The kernel-launch ABIs no longer leak
  `<hip/hip_runtime.h>` into the physics/boundary/numerics layers.
- Observation-point types (`PointCloud`, `ObservationGrid`, `PlaneSlice`,
  `LineProbe`) moved to `quasar::core`; the numerics `IFieldEvaluator` interface
  no longer depends on a magnetostatics header.
- `BiotSavartConfig` drops the never-consumed `tile_segments` / `block_size`
  fields; kernel tiling is compile-time and tuned per-gfx in
  `cmake/QuasarLaunchParams.cmake`.
- The `quasar-coil` and `quasar-pic` CLIs are standardized: both use the
  `command` subparser dest, a `func` handler, and a default-quiet output model
  with a `--verbose` flag. The coil CLI's previous default-chatty `--quiet`
  behavior is replaced by `--verbose` (off by default).
- PIC `EmPicConfig` carries the particle shape as a deck-vocabulary string
  (`shape`: `cic` / `tsc`) instead of an integer `shape_order`; the solver derives
  the field-solver / pusher / deposit registry names directly from the deck order
  and shape, dropping the internal switch ladders. The specular current fold-back
  is now a `fold_current` hook on `IParticleBoundary` rather than a special case
  in the solver step.

### Optimized
- PIC: per-step full-grid scratch allocations hoisted out of the current filter
  and particle compaction; the Biot–Savart double-precision path uploads the host
  SoA directly instead of an element-by-element copy.
- PIC: the particle gather computes its shape weights once per particle (was four
  times) and gathers self + external E/B in a single stencil sweep; the
  charge-conserving deposit computes each window node index once across the
  Jx/Jy/Jz loops.
- PIC: the per-logged-step alive-particle count reuses the species compaction
  counter instead of allocating a per-block reduction buffer each call.
- PIC: the current deposit no longer synchronizes the device every step. It
  accumulates a persistent overflow flag that the solver drains on a cadence and
  at `finalize()`, removing the per-step host-device round-trip that drained the
  GPU pipeline mid-step (the fatal "reduce dt" overflow is now reported up to one
  cadence interval late).
- Magnetostatics: transient Biot–Savart output buffers skip the allocation
  zero-fill (the kernel overwrites them in full), via a new
  `device_alloc_uninit` / `DeviceBuffer(n, uninitialized)` path.

### Upcoming changes
- A `file_grid` field evaluator name is reserved in the registry but not yet
  implemented (it throws and is intentionally not deck-selectable) pending the
  file-backed grid loader.
