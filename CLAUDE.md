# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Quasar is a HIP-accelerated AMD ROCm numerical simulation framework with `gfx942` and `gfx950` presets, a C++20 core, pybind11 bindings, and a pure-Python user-facing layer. Current vertical slices are **magnetostatics** (Biot–Savart for coil design), **electromagnetic PIC**, and **ideal MHD** (high-order MP5/MP7 reconstruction on Cartesian and axisymmetric cylindrical grids, HLLD, and FD-CT).

The build supports only the HIP backend; configuring with
`-DQUASAR_ENABLE_HIP=OFF` is a hard error (see the top-level `CMakeLists.txt`).

## Build and test

Configure/build/test via CMake presets:

```bash
python -m pip install -r tests/python/requirements.txt
cmake --preset hip-gfx942-release
cmake --build --preset hip-gfx942-release -j
ctest --preset hip-gfx942-release            # default suite (excludes 'slow')
ctest --preset hip-gfx942-release-all        # everything, including 'slow'
ctest --preset hip-gfx942-release -R <regex> # single test by name regex
cmake --preset hip-gfx942-distributed-debug  # require MPI + parallel HDF5
cmake --build --preset hip-gfx942-distributed-debug -j
ctest --preset hip-gfx942-distributed-debug  # CTest GPU resource scheduling
ctest --preset hip-gfx942-distributed-debug-all
```

Preset build trees live under `build/<preset>/`; release and debug presets are
provided for both `gfx942` and `gfx950`. Distributed release/debug presets set
`QUASAR_ENABLE_DISTRIBUTED=ON`, so a missing MPI or parallel-HDF5 dependency is
a configuration error rather than silent test deregistration. Distributed
debug presets also enable the dedicated, default-off
`QUASAR_DISTRIBUTED_TEST_HOOKS` fault-injection option.

Distributed test presets load `tests/ctest_gpus.json`; CTest assigns declared
`RESOURCE_GROUPS` and `tests/run_with_ctest_gpus.sh` maps them to
`HIP_VISIBLE_DEVICES`. Pass a site-local `--resource-spec-file` when the visible
GPU inventory differs from the checked-in eight-GPU node description.

Tests labelled `slow` are excluded from the default test presets because they
dominate a full run — currently just `python_test_examples`, which drives every
example CLI end-to-end and takes over an hour. Use the matching `-all` test
preset (or `ctest -L slow` in a build tree) to run them; do that before
releasing, and whenever a change touches `examples/`, a physics CLI, or the deck
schema. Add the label in `tests/python/CMakeLists.txt`
(`_quasar_python_slow_tests`) when a new test crosses that cost threshold.

The compiled Python extension `quasar._core` lands at `build/hip-gfx942-release/python/quasar/_core*.so`. To run Python entry points or pytest against the build tree:

```bash
PYTHONPATH=build/hip-gfx942-release/python python -m quasar.coil.cli run examples/single_loop/input.yaml
PYTHONPATH=build/hip-gfx942-release/python pytest tests/python -k <pattern>
```

A single GoogleTest binary can be run directly, for example
`build/hip-gfx942-release/tests/unit/core/quasar_test_grid_2d`, optionally with
`--gtest_filter=...`.

## Architecture

The codebase is organized around **four orthogonal solver axes**:
`physics × numerics × boundary × backend`, plus a cross-cutting **distributed
execution axis**. The solver axes have matching trees under
`include/quasar/<axis>/` and `src/<axis>/`; distributed has the same public/private
pair but deliberately depends on the complete solver slices while remaining off
the `quasar_core` aggregate target. Adding a serial scheme in one solver axis
should not touch the others. Extending distributed execution follows
`docs/dev-guide/adding_distributed_physics.rst`. Distributed PIC particle
migration is device-resident: the periodic wrap and the position-to-endpoint map
run in a kernel (`launch_pic_route_departing_particles`), which also groups the
resulting wire records by destination rank and orders them by stable id, so the
host slices bytes rather than routing particles. The seed and
checkpoint-restore routing paths in `src/distributed/pic_runtime.cpp` are the
knowing exception and still resolve owners on the host.

Key conventions:

- **Backend isolation** — all HIP `.hip` / device code lives under `src/backend/hip/`. Code outside that directory must go through abstractions in `include/quasar/backend/` (`device.hpp`, `memory.hpp`).
- **Plugin registry** — concrete schemes/BCs/physics self-register via `core/registry.hpp` so the YAML/Python input deck selects implementations by string name. Drivers in `apps/` should not contain `if/else` chains over physics types.
- **Device-resident evaluator axis** — `numerics::IFieldEvaluator` takes a
  `core::DevicePointCloud` and returns `core::DeviceVectorField` /
  `core::DeviceTensorField` (SoA planes), never host `Field<Vec3>`. Chained
  consumers — the PIC external-field sampler, the MHD background builder — keep
  the planes on the device; `.to_host()` belongs only at an output boundary (a
  CLI, a Python binding, a test). Host-side convenience wrappers live under
  `tests/unit/support/` precisely so production code cannot reach them. See
  `docs/dev-guide/adding_a_field_evaluator.rst`. The structured observation sets
  (`ObservationGrid`, `PlaneSlice`, `LineProbe`) expand straight into a
  `DevicePointCloud` via `to_device_point_cloud()`, backed by kernels in
  `src/backend/hip/core/`; that expansion agrees with the host `point_at()`
  accessors bit for bit, which is why the module is compiled
  `-ffp-contract=off`.
- **Public vs private** — headers under `include/quasar/` are the supported
  public include surface. Anything in `src/` (including `src/**/detail/`) is
  implementation-private; do not include from `src/` outside its own module.
  The current CMake workflow stages Python into the build tree and does not
  define system-install or package-export targets.
- **Apps are thin** — any future `apps/*/main.cpp` should only parse input and call the library. Today all three vertical slices are driven from Python (`quasar.coil.cli`, `quasar.pic.cli`, and `quasar.mhd.cli`); `apps/` contains only the explanatory CMake file.

> **Axis orthogonality is aspirational where the numerics and boundary axes are concerned.** The `numerics` solver/pusher/deposit interfaces (`IFieldSolver`, `IParticlePusher`, `IDepositScheme`) are currently phrased in the EM-PIC concrete types (`YeeField2D`/`JField2D`, `pic::ParticleSpecies`) and their out-of-line definitions + registrations live in `src/physics/pic/pic_solver.cpp` (see `docs/dev-guide/adding_a_field_solver.rst`, `adding_a_pusher.rst`, `adding_a_deposit_scheme.rst`). This is deliberate: with EM-PIC as the only consumer, templating these over field/particle types would add abstraction for a second physics module that does not yet exist. When a non-PIC consumer of these schemes appears, that is the point to template them and split the implementations into `src/numerics/`.
>
> Two further instances of this same single-consumer tradeoff are accepted under the same reasoning:
> - **Build-graph coupling to the PIC backend.** `quasar_numerics` (`filter.cpp`) and `quasar_boundary` (`periodic`/`wall`/`outflow`/`axis`) call `launch_pic_*` device kernels and operate on `pic::ParticleSpecies`/`YeeField2D`, so `src/numerics/CMakeLists.txt` and `src/boundary/CMakeLists.txt` link `quasar_pic_hip`. This inverts the nominal physics→numerics/boundary dependency direction, but the kernels these axes need live in the PIC backend and EM-PIC is their only consumer. The split (a numerics/boundary-owned backend target, or relocating the PIC-typed registrations into the PIC module) waits for the same trigger as the interface templating above.
> - **MHD numerics/boundary interfaces.** `IFluxReconstruction`, `IRiemannSolver`, `ICtScheme`, `ISsprkIntegrator`, `IPositivityLimiter`, and the MHD fluid/field boundary interfaces are phrased in `quasar::mhd::MhdField2D` and hard-include `physics/mhd/`. MHD is their only consumer; templating them over the field type waits for a second fluid consumer.

Concrete physics modules currently present:

- `physics/magnetostatics` — Biot–Savart field evaluator over conductor geometries with observation point sets. Exposed to Python as `quasar.coil` with a CLI at `python -m quasar.coil.cli run <input.yaml>` that writes `out.npz`. **Fully GPU-resident**: the geometry generators write filament vertices with kernels in `src/backend/hip/magnetostatics/geometry_hip.hip`, `ConductorSystem` flattens them into cached per-segment device planes, and the fp64 evaluator implements the device-resident `IFieldEvaluator`, so nothing is uploaded at evaluation time. Two things to know before extending it: `Filament` and `ConductorSystem` are move-only because the vertices are `DeviceBuffer`s, and copying a `ConductorSystem` duplicates device allocations; and `BiotSavartEvaluatorF`, the fp32 sibling, deliberately stays off the evaluator interface and keeps host `Field<Vec3f>` results for the precision-comparison test, though its narrowing is itself a kernel. Host arithmetic that remains is scalar and does not scale with the vertex count: the orthonormal basis per generator call, and a racetrack's frame points and corners.
- `physics/pic` — EM-PIC vertical slice (`EmPicConfig`, Cartesian and axisymmetric-cylindrical FDTD, particle shapes, charge-conserving deposition, particle and field boundaries, and diagnostics). Driven from Python at `python -m quasar.pic.cli run <input.yaml>`; no C++ app exists. Deck I/O is under `quasar.pic`. Two things to know before touching the particle lifecycle: **initial sampling is device-resident** (`physics/pic/particle_sampling.hpp` — quiet-start lattice, Maxwellian draw, perturbation and the `|v| < c` and in-domain checks are all kernels, and the deck layer supplies only O(1) scalars), and the generator is **counter-based Philox keyed on `(seed, species)` and counted by particle index**, not a stream, which is what makes a sample independent of block size, thread order and how the population is partitioned across devices — and is why a seeded deck's velocities changed when it landed. There is deliberately no second host sampler: the serial CLI and the distributed runner both go through these kernels so one deck seed gives one sample either way.
- `physics/mhd` — high-order ideal-MHD vertical slice (MP5/MP7 characteristic reconstruction with Cartesian or radius-weighted cylindrical finite-volume moments, HLLD Riemann solver, FD constrained transport, SSP-RK3, conservative troubled-cell positivity control). Driven from Python at `python -m quasar.mhd.cli run <input.yaml>`. The numerics live under `numerics/` (the second consumer of that axis after PIC); deck I/O is under `quasar.mhd`. **Setup is device-resident too**: the six benchmark initial conditions (`physics/mhd/initial_conditions.hpp`) and all five background-field sources (`physics/mhd/background_builder.hpp`) are kernels, and `quasar.mhd.io` supplies only deck parsing, validation and an O(1) scalar parameter block. Three things to know before extending either. Generators dispatch on an **enumerator inside one kernel**, not a registry: a registry entry is a host object with a vtable and a vtable cannot cross to the device (same constraint as equilibrium's `ProfileCoefficients`); the deck still selects by string. Analytic background profiles are lowered to an **affine POD derived by probing the registered profile**, so the registry stays the single definition of a name, and a profile that is not affine is refused rather than silently linearized — which is admissible only because `IMhdBackgroundProfile::sample` already requires an affine profile's centre value to BE its element moment. And the opt-in cylindrical vacuum projection deliberately does **not** reuse the equilibrium multigrid even though its operator is `Delta*` up to a row scaling by `r`: `GsDeviceMultigrid` lives behind the equilibrium physics module, and calling it from here would create an mhd → equilibrium edge; moving that class down to the numerics axis (where its host twin already is) is the real fix and has not been done.
- `physics/equilibrium` — free-boundary Grad–Shafranov solver (sixth-order compact operator, defect-corrected geometric multigrid, Green's-function boundary coupling, critical-point location, flux surfaces and `q(ψ)`), plus projection onto the staggered MHD mesh. **Fully GPU-resident**: all arithmetic runs through the launch ABI in `physics/equilibrium/kernels.hpp` with kernels under `src/backend/hip/equilibrium/`; `GsSolver` retains only control flow. Two consequences worth knowing before extending it: profiles are lowered to a `ProfileCoefficients` POD at construction because a vtable cannot cross to the device, so only `PolynomialProfile` works today; and the module is compiled with `-ffp-contract=off`, which is load-bearing for both the Padé line solve and the compensated current integral (see `src/backend/hip/equilibrium/CMakeLists.txt`). No Python bindings or CLI yet.
- `physics/stability` — fixed-boundary ideal-MHD stability of a converged
  Grad–Shafranov equilibrium: PEST straight-field-line coordinates, per-`n`
  rational-surface location, Chebyshev spectral elements crossed with a Fourier
  poloidal basis, and a dense Hermitian energy/inertia pencil solved with
  hipSOLVER. The first consumer of `GsDeviceResult`, and the first eigenvalue
  problem in the tree. Like equilibrium it is fully GPU-resident behind
  `physics/stability/kernels.hpp` with kernels under `src/backend/hip/stability/`,
  and it is compiled `-ffp-contract=off` for the same reason. Three things to
  know before extending it: the model is **annular** (the axis is excluded and
  the inner surface gets the natural condition, so results are not full-axis
  tokamak results); `n = 0` and rational-surface topologies are refused rather
  than approximated; and the shift-invert path in `spectral_blocks.hpp` is
  exercised and cross-checked but does not yet replace the dense solve. No
  Python bindings or CLI yet.
- `physics/analytic_fields` — analytic and rectilinear file-backed fields
  (`uniform`, `gradient`, `dipole`, `file_grid`) used by simulations, tests, and
  examples. **Fully GPU-resident**: the classes in `src/physics/analytic_fields/`
  hold only parameter validation, a launch, and a status check, with the
  arithmetic in `src/backend/hip/analytic_fields/` behind the launch ABI in
  `physics/analytic_fields/kernels.hpp`, compiled `-ffp-contract=off`. Two things
  to know before extending it: `FileGridEvaluator` keeps its node values on the
  device from `configure()` onward and runs its finiteness and solenoidality
  admissibility sweeps there, so the host holds the map only while parsing it;
  and the extended exponent range these evaluators used to get from host
  `long double` is now carried explicitly through `numerics::ScaledValue`
  (`include/quasar/numerics/scaled_arithmetic.hpp`), which is what the
  `-ffp-contract=off` requirement protects.

The **cylindrical radial moment tables** (`numerics/radial_moments.hpp`,
`radial_tables.hpp`) are solved on the device in one batched factorization per
`RadialTables` construction, over `numerics/batched_lu.hpp` and rocSOLVER's
strided-batched LU. rocSOLVER rather than hipSOLVER because hipSOLVER has no
batched `getrf`/`getrs` and these systems are at most 8x8. The
iterative-refinement step in `solve_radial_rows` is load-bearing rather than
optional: it is what makes the binary64 result backward stable, and hence what
keeps every row under the 1e-11 residual threshold `RadialTables` enforces —
measured worst case 3.4e-12 against a 2.0e-12 `long double` reference.

CMake module helpers live in `cmake/`: `QuasarAddModule.cmake` (per-axis target
creation, including `DETACHED` modules such as distributed),
`QuasarDistributed.cmake` (tri-state MPI/parallel-HDF5 discovery),
`QuasarHipRuntime.cmake` (HIP runtime detection), and
`QuasarLaunchParams.cmake` (per-arch kernel launch tuning).

## Examples and tests

`examples/<case>/` contains a YAML input deck plus a `README.md` documenting how to run it and its analytical reference. Each example has a corresponding integration test in `tests/python/test_examples.py`. When adding a new physics example, add both the README and the matching test entry — the test runs the CLI against the deck and compares the output `.npz` to the closed-form reference.

`python_test_examples` is labelled `slow` and is therefore **not** run by the default `ctest --preset` invocation. After changing an example, a physics CLI, or the deck schema, run it explicitly with the `-all` test preset (or `ctest -L slow`).

C++ unit tests under `tests/unit/<axis>/` mirror the public header tree. Python tests under `tests/python/` cover bindings, deck I/O, CLI, and post-processing.

Dev-guide RST under `docs/dev-guide/` (`adding_a_background_field`, `adding_a_boundary`, `adding_a_deposit_scheme`, `adding_a_field_evaluator`, `adding_a_field_solver`, `adding_a_filter`, `adding_a_geometry`, `adding_an_initial_condition`, `adding_an_mhd_scheme`, `adding_a_pusher`) is the source of truth for the steps to add a new pluggable component — consult these before adding a new scheme. The MHD numerics axes (Riemann solver, CT scheme, SSP-RK integrator, positivity limiter) are covered by `adding_an_mhd_scheme`; the six seeded initial conditions, which are selected by string but dispatched on an enumerator inside one kernel rather than through a registry, are covered by `adding_an_initial_condition`.
`adding_distributed_physics` is the corresponding source of truth for adding a
new physics slice to MPI/multi-GPU execution. `extending_stability` covers the
ideal-MHD stability slice, which has no plugin registry but does have a chain of
stage contracts and several deliberate refusals that must not be quietly
relaxed.
