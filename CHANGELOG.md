# Changelog

All notable changes to Quasar are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/), and the project is pre-1.0 so
interfaces may still change between entries.

## [Unreleased]

### Fixed
- PIC: charge conservation across particle boundary crossings. The periodic-wrap
  and specular-reflect kernels now co-shift the previous particle position so the
  Esirkepov deposit sees a sub-cell displacement (previously a seam-crossing
  particle injected uncancelled current).
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

### Optimized
- PIC: per-step full-grid scratch allocations hoisted out of the current filter
  and particle compaction; the Biot–Savart double-precision path uploads the host
  SoA directly instead of an element-by-element copy.

### Added
- PIC test coverage: a real FDTD plane-wave dispersion check, a behavioral
  mixed-boundary test, Gauss-residual diagnostics tests, an end-to-end CLI run
  smoke test, and a `--log-every` / `--write-every` diagnostics test.
