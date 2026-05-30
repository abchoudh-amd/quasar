# Changelog

All notable changes to Quasar are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/), and the project is pre-1.0 so
interfaces may still change between entries.

## [Unreleased]

### Added
- PIC: support for `units: normalized` decks and physically consistent `units: SI`
  decks. SI decks are non-dimensionalized through `Normalization` before stepping
  (grid, dt, charge/mass/velocity/density, external field) and diagnostics are
  converted back to SI on output.
- PIC field boundary conditions are live: per-side `boundary.field` (`periodic` /
  `pec`) selectable from the deck, with stable energy-conserving PEC reflection.
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

### Fixed
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
- `quasar.pic.postprocess.reshape_with_ghost` infers the ghost width instead of
  assuming a single ghost layer (4th-order grids use two).
- Registry: type-keyed factory registration. Stateless `make_unique` lambdas were
  collapsed by identical-code folding, so `Registry::create(name)` could return
  the wrong concrete type — the root cause of the long-standing PIC field/particle
  boundary "heisenbug". Boundary dispatch now goes through the interface.
- PIC: the periodic particle wrap is side/axis-aware, so periodic and
  non-periodic (e.g. absorbing) walls can coexist per side.

### Changed
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

### Optimized
- PIC: per-step full-grid scratch allocations hoisted out of the current filter
  and particle compaction; the Biot–Savart double-precision path uploads the host
  SoA directly instead of an element-by-element copy.

### Added
- PIC test coverage: a real FDTD plane-wave dispersion check, a behavioral
  mixed-boundary test, Gauss-residual diagnostics tests, an end-to-end CLI run
  smoke test, and a `--log-every` / `--write-every` diagnostics test.
