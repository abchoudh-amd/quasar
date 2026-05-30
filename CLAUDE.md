# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Quasar is a HIP-accelerated (AMD ROCm, `gfx942`) numerical simulation framework with a C++20 core, pybind11 bindings, and a pure-Python user-facing layer. Current vertical slices are **magnetostatics** (Biot–Savart for coil design) and a minimal **electromagnetic PIC** module.

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

The codebase is organized around **four orthogonal axes**: `physics × numerics × boundary × backend`. Each axis has matching trees under `include/quasar/<axis>/` (public interfaces, installed) and `src/<axis>/` (private implementations). Adding a scheme in one axis should not touch the others. See `plans/directory_plan.md` for the canonical layout intent.

Key conventions:

- **Backend isolation** — all HIP `.hip` / device code lives under `src/backend/hip/`. Code outside that directory must go through abstractions in `include/quasar/backend/` (`device.hpp`, `memory.hpp`).
- **Plugin registry** — concrete schemes/BCs/physics self-register via `core/registry.hpp` so the YAML/Python input deck selects implementations by string name. Drivers in `apps/` should not contain `if/else` chains over physics types.
- **Public vs private** — only headers under `include/quasar/` are installed. Anything in `src/` (including `src/**/detail/`) is implementation-private; do not include from `src/` outside its own translation unit.
- **Apps are thin** — any future `apps/*/main.cpp` should only parse input and call the library. Today both vertical slices are driven from Python (`quasar.coil.cli`, `quasar.pic.cli`) and `apps/` holds only a placeholder.

Concrete physics modules currently present:

- `physics/magnetostatics` — Biot–Savart field evaluator over conductor geometries with observation point sets. Exposed to Python as `quasar.coil` with a CLI at `python -m quasar.coil.cli run <input.yaml>` that writes `out.npz`.
- `physics/pic` — minimal EM-PIC vertical slice (`EmPicConfig`, FDTD, particle shapes). Driven from Python at `python -m quasar.pic.cli run <input.yaml>` (no C++ app exists; `apps/` holds only a placeholder `CMakeLists.txt`). Deck I/O is under `quasar.pic`.
- `physics/analytic_fields` — closed-form reference fields used by tests/examples.

CMake module helpers live in `cmake/`: `QuasarAddModule.cmake` (per-axis target creation), `QuasarHipRuntime.cmake` (HIP runtime detection), `QuasarLaunchParams.cmake` (per-arch kernel launch tuning).

## Examples and tests

`examples/<case>/` contains a YAML input deck plus a `README.md` documenting how to run it and its analytical reference. Each example has a corresponding integration test in `tests/python/test_examples.py`. When adding a new physics example, add both the README and the matching test entry — the test runs the CLI against the deck and compares the output `.npz` to the closed-form reference.

C++ unit tests under `tests/unit/<axis>/` mirror the public header tree. Python tests under `tests/python/` cover bindings, deck I/O, CLI, and post-processing.

Dev-guide RST under `docs/dev-guide/` (`adding_a_boundary`, `adding_a_field_evaluator`, `adding_a_filter`, `adding_a_geometry`, `adding_a_pusher`) is the source of truth for the steps to add a new pluggable component — consult these before adding a new scheme.
