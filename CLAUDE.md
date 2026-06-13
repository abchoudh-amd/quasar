# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Quasar is a HIP-accelerated (AMD ROCm, `gfx942`) numerical simulation framework with a C++20 core, pybind11 bindings, and a pure-Python user-facing layer. Current vertical slices are **magnetostatics** (Biot–Savart for coil design), a minimal **electromagnetic PIC** module, and a high-order **ideal-MHD** module (MP5/MP7 reconstruction, HLLD, FD-CT; Cartesian and axisymmetric cylindrical).

The build is HIP-only at present — configuring with `-DQUASAR_ENABLE_HIP=OFF` is a hard error (see top `CMakeLists.txt`). A host backend is planned but not yet present.

## Build and test

Configure/build/test via CMake presets:

```bash
cmake --preset hip-gfx942-release
cmake --build --preset hip-gfx942-release -j
ctest --preset hip-gfx942-release            # all tests
ctest --preset hip-gfx942-release -R <regex> # single test by name regex
```

The build tree lives at `build/hip-gfx942-{release,debug}/`. A `hip-gfx942-debug` preset also exists.

The compiled Python extension `quasar._core` lands at `build/hip-gfx942-release/python/quasar/_core*.so`. To run Python entry points or pytest against the build tree:

```bash
PYTHONPATH=build/hip-gfx942-release/python python -m quasar.coil.cli run examples/single_loop/input.yaml
PYTHONPATH=build/hip-gfx942-release/python pytest tests/python -k <pattern>
```

A single GoogleTest binary can be run directly, e.g. `build/hip-gfx942-release/tests/unit/physics/test_magnetostatics`, optionally with `--gtest_filter=...`.

## Architecture

The codebase is organized around **four orthogonal axes**: `physics × numerics × boundary × backend`. Each axis has matching trees under `include/quasar/<axis>/` (public interfaces, installed) and `src/<axis>/` (private implementations). Adding a scheme in one axis should not touch the others.

Key conventions:

- **Backend isolation** — all HIP `.hip` / device code lives under `src/backend/hip/`. Code outside that directory must go through abstractions in `include/quasar/backend/` (`device.hpp`, `memory.hpp`).
- **Plugin registry** — concrete schemes/BCs/physics self-register via `core/registry.hpp` so the YAML/Python input deck selects implementations by string name. Drivers in `apps/` should not contain `if/else` chains over physics types.
- **Public vs private** — only headers under `include/quasar/` are installed. Anything in `src/` (including `src/**/detail/`) is implementation-private; do not include from `src/` outside its own translation unit.
- **Apps are thin** — any future `apps/*/main.cpp` should only parse input and call the library. Today both vertical slices are driven from Python (`quasar.coil.cli`, `quasar.pic.cli`) and `apps/` holds only a placeholder.

> **Axis orthogonality is aspirational where the numerics and boundary axes are concerned.** The `numerics` solver/pusher/deposit interfaces (`IFieldSolver`, `IParticlePusher`, `IDepositScheme`) are currently phrased in the EM-PIC concrete types (`YeeField2D`/`JField2D`, `pic::ParticleSpecies`) and their out-of-line definitions + registrations live in `src/physics/pic/pic_solver.cpp` (see `docs/dev-guide/adding_a_field_solver.rst`, `adding_a_pusher.rst`, `adding_a_deposit_scheme.rst`). This is deliberate: with EM-PIC as the only consumer, templating these over field/particle types would add abstraction for a second physics module that does not yet exist. When a non-PIC consumer of these schemes appears, that is the point to template them and split the implementations into `src/numerics/`.
>
> Two further instances of this same single-consumer tradeoff are accepted under the same reasoning:
> - **Build-graph coupling to the PIC backend.** `quasar_numerics` (`filter.cpp`) and `quasar_boundary` (`periodic`/`wall`/`outflow`/`axis`) call `launch_pic_*` device kernels and operate on `pic::ParticleSpecies`/`YeeField2D`, so `src/numerics/CMakeLists.txt` and `src/boundary/CMakeLists.txt` link `quasar_pic_hip`. This inverts the nominal physics→numerics/boundary dependency direction, but the kernels these axes need live in the PIC backend and EM-PIC is their only consumer. The split (a numerics/boundary-owned backend target, or relocating the PIC-typed registrations into the PIC module) waits for the same trigger as the interface templating above.
> - **MHD numerics/boundary interfaces.** `IFluxReconstruction`, `IRiemannSolver`, `ICtScheme`, `ISsprkIntegrator`, `IPositivityLimiter`, and the MHD fluid/field boundary interfaces are phrased in `quasar::mhd::MhdField2D` and hard-include `physics/mhd/`. MHD is their only consumer; templating them over the field type waits for a second fluid consumer.

Concrete physics modules currently present:

- `physics/magnetostatics` — Biot–Savart field evaluator over conductor geometries with observation point sets. Exposed to Python as `quasar.coil` with a CLI at `python -m quasar.coil.cli run <input.yaml>` that writes `out.npz`.
- `physics/pic` — minimal EM-PIC vertical slice (`EmPicConfig`, FDTD, particle shapes). Driven from Python at `python -m quasar.pic.cli run <input.yaml>` (no C++ app exists; `apps/` holds only a placeholder `CMakeLists.txt`). Deck I/O is under `quasar.pic`.
- `physics/mhd` — high-order ideal-MHD vertical slice (MP5/MP7 characteristic reconstruction, HLLD Riemann solver, FD constrained transport, SSP-RK3, troubled-cell positivity floor; Cartesian and axisymmetric cylindrical `(r,z)`). Driven from Python at `python -m quasar.mhd.cli run <input.yaml>`. The numerics live under `numerics/` (the second consumer of that axis after PIC); deck I/O is under `quasar.mhd`.
- `physics/analytic_fields` — closed-form reference fields used by tests/examples.

CMake module helpers live in `cmake/`: `QuasarAddModule.cmake` (per-axis target creation), `QuasarHipRuntime.cmake` (HIP runtime detection), `QuasarLaunchParams.cmake` (per-arch kernel launch tuning).

## Examples and tests

`examples/<case>/` contains a YAML input deck plus a `README.md` documenting how to run it and its analytical reference. Each example has a corresponding integration test in `tests/python/test_examples.py`. When adding a new physics example, add both the README and the matching test entry — the test runs the CLI against the deck and compares the output `.npz` to the closed-form reference.

C++ unit tests under `tests/unit/<axis>/` mirror the public header tree. Python tests under `tests/python/` cover bindings, deck I/O, CLI, and post-processing.

Dev-guide RST under `docs/dev-guide/` (`adding_a_background_field`, `adding_a_boundary`, `adding_a_deposit_scheme`, `adding_a_field_evaluator`, `adding_a_field_solver`, `adding_a_filter`, `adding_a_geometry`, `adding_an_mhd_scheme`, `adding_a_pusher`) is the source of truth for the steps to add a new pluggable component — consult these before adding a new scheme. The MHD numerics axes (Riemann solver, CT scheme, SSP-RK integrator, positivity limiter) are covered by `adding_an_mhd_scheme`.
