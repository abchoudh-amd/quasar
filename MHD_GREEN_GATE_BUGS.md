# MHD GREEN-gate bug report

Status snapshot from the `quasar-build` blind-TDD GREEN gate for the high-order
ideal-MHD module (`physics/mhd`). Records every defect surfaced while driving the
C++ (`ctest -R mhd`) and Python (`pytest tests/python/test_mhd_*`) suites to pass,
which are fixed, and which numerics defects remain open.

Build/test environment: HIP `gfx942`, cmake 4.3.2 (venv path), ctest/pytest via
`srun` on a `ctr-*` node. cmake is not on PATH; `python` is `/usr/bin/python3`;
inline env vars need `/usr/bin/env` (a stray `~/.local/bin/env` shadows it);
`gdb` absent, use `/opt/rocm/bin/rocgdb`.

---

## Test status at last gate run

| Test (ctest `-R mhd`) | Status | Notes |
| --- | --- | --- |
| `numerics_test_mhd_state` | PASS | conserved↔primitive, gamma-law pressure, fast speed |
| `numerics_test_mhd_eigensystem` | PASS | L·R=I, diagonalization, degeneracy regularization |
| `numerics_test_flux_reconstruction` | PASS (builds + runs) | after harness rename |
| `numerics_test_ct_divergence` | PASS | isolated CT telescoping holds |
| `numerics_test_ssprk3_integrator` | PASS | |
| `numerics_test_positivity_limiter` | PASS | |
| `mhd_test_mhd_scheme_registry_linkage` | PASS | all 7 registries populated |
| `mhd_test_mhd_solver_conservation` | PASS | mass/mom/energy conserved (was bug #5) |
| `mhd_test_mhd_divergence_free` | PASS | div(B) ~1.5e-14 after 6 steps (was bug #4) |
| `mhd_test_mhd_cylindrical_source` | PASS | equilibrium stationary + cyl div(B) ~0 (was bug #6) |

**All 10 `-R mhd` ctests pass.** Bugs #4, #5, #6 are RESOLVED — they were a single
structural defect (see "ROOT CAUSE" below), not three independent numerics bugs.
The earlier "open bug" diagnoses (stale-ghost / missing geometric-source terms)
were WRONG: the proposed ghost-refill fix had already been applied and made div(B)
*worse* (1.73 → 2.45), and the cylindrical equilibrium seed has v=0, B_phi=0 so its
geometric source is identically zero — the drift was the same structural defect.

Python: `test_mhd_io` + `test_mhd_bindings` + `test_mhd_cli` pass (47). MHD example
integration tests (`test_examples.py`) now RUN to completion (the CLI no longer
aborts mid-step); `state_*` npz arrays are written interior-only per contract.
Three example assertions remain RED, all traceable to the documented v1
"device reconstruction is MUSCL-only" simplification, NOT to the gate bugs:
  * `MhdLinearWaveConvergenceTest` — ill-posed: the Alfven seed holds rho exactly
    uniform, so the rho-L1 error is round-off (6.7e-14 → 5.0e-14) and the
    "convergence rate" is meaningless noise.
  * `OrszagTangExampleTest` / `BrioWuExampleTest` — strong-gradient flows where the
    2nd-order device MUSCL path underperforms the mp7 design order (energy/shock
    structure outside tolerance). These need MP5/MP7 ported to the device kernel
    (Known simplification #2), out of scope for the GREEN gate.

---

## FIXED bugs

### Bug #1 — Test harness: wrong interface-state accessor names (compile error)
- **Where:** `tests/unit/numerics/test_flux_reconstruction.cpp`
- **Symptom:** build failure — `MhdInterfaceStates<double>` has no member `left_at`/`right_at`.
- **Cause:** the RED test writer (Task 2.2) guessed accessor names `left_at`/`right_at`;
  the approved contract and the implementation (Task 3.1) use `state_left`/`state_right`.
- **Routing:** test layer (Phase 2), not an implementer — the source matched the contract.
- **Fix:** renamed `left_at`→`state_left`, `right_at`→`state_right` in the test.

### Bug #2 — Python test: wrong boundary dataclass name (import error)
- **Where:** `tests/python/test_mhd_io.py`
- **Symptom:** `ImportError: cannot import name 'Boundary' from 'quasar.mhd.io'`.
- **Cause:** test imported `Boundary`; the implementation mirrors the PIC convention
  `BoundaryConfig` (Task 3.11 followed `quasar.pic.io`). Plan did not pin the name;
  implementer followed the established codebase pattern correctly.
- **Routing:** test layer.
- **Fix:** renamed `Boundary`→`BoundaryConfig` in the test (field-compatible).

### Bug #3 — Python tests: contract divergences (nghost + CFL layer)
- **Where:** `tests/python/test_mhd_bindings.py`, `tests/python/test_mhd_io.py`,
  `tests/python/test_mhd_cli.py`
- **Symptoms / causes:**
  - bindings built `Grid2D` with the default `nghost=1` and expected
    `MhdSolver2D` to auto-widen to the mp7 requirement (4). The **locked contract**
    is: the ctor auto-sizes only when `nghost==0`; a *positive* under-size is a hard
    `ValueError`. Implementer honored the contract; test guessed auto-widen.
  - `test_explicit_dt_above_cfl_limit_rejected` asserted deck `.validate()` rejects
    an over-CFL explicit dt. CFL needs the constructed solver + seeded state
    (`MhdSolver2D::cfl_limit()`), so `validate()` is the wrong layer — `quasar.pic.io`
    does no CFL check in `validate()` either. The CLI/solver enforces it before stepping.
- **Routing:** test layer (both reflect the approved contract).
- **Fix:** bindings construct grid with `nghost=0` (auto-size); the dt-CFL assertion
  moved to the CLI/solver path (CLI `run` on an over-CFL deck exits non-zero before
  stepping; `validate()` now only checks dt_s is positive-float-or-"auto").

### Bug #4a — Boundary host-heap out-of-bounds write (the crash) — FIXED
- **Where:** `src/physics/mhd/mhd_boundary.cpp`, `fill_normal_face`
- **Symptom:** every solver `step()` aborted nondeterministically with
  `std::bad_variant_access` (Python) / glibc `corrupted size vs. prev_size` (C++) /
  SIGSEGV inside `libamdhip64`. Three C++ tests segfaulted; CLI stepping aborted.
- **Cause:** the hi-side normal-face ghost write computed
  `gi = nx + layer` for `layer ∈ [1, nghost]`. At `layer == nghost`,
  `gi = nx + nghost`, so `Grid2D::index(nx+nghost, j)` yields
  `i + nghost = nx + 2*nghost = pitch` — **one column past the row**. The
  face-staggered arrays share the *same* storage extent as cell arrays
  (valid `i ∈ [-nghost, nx-1+nghost]`); there is no extra slot at `nx+nghost`.
  This OOB write corrupted the `StagedComponent` `std::vector` host heap, which then
  surfaced far downstream as the variant/heap aborts.
- **Why it looked like a kernel bug:** the corruption manifested during a later
  device staging copy / first `reconstruct` dispatch. The kernels owner correctly
  proved the kernels clean in isolation (full link, nghost 2 & 4, per-launch error
  checks) and pointed the diagnosis back to the boundary host path.
- **Fix:** hi normal-face ghost target rebased to `gi = nx - 1 + layer` (matching the
  cell-centered convention), with periodic/outflow/reflecting **source** indices
  adjusted to keep the staggered-face semantics correct:
  - periodic: `si = nx - layer` (lo) / `layer - 1` (hi)
  - outflow: `si = 0` (lo) / `nx - 1` (hi)
  - reflecting: `si = layer - 1` (lo) / `nx - layer` (hi), normal component odd sign.
  Verified in-bounds for nghost up to 4 (`0 ≤ i+nghost < pitch`, idx `< storage_size()`).
  `fill_cell` was already correct (hi `nx-1+layer`).

### Bug #4b — Constrained-transport double-update of face B — PARTIALLY FIXED
- **Where:** `src/physics/mhd/mhd_solver.cpp` (`compute_residual`/`combine_stage`)
  + kernels `src/backend/hip/mhd/`.
- **Symptom:** div(B) exploded 0 → ~5.25 after one step on an analytically
  divergence-free seed.
- **Cause:** the staggered poloidal field `bx_face`/`by_face` was advanced **twice**:
  (1) `launch_mhd_flux_difference` accumulates the Godunov magnetic-flux divergence
  into *all 8* `dudt` slots (the HLLD flux has nonzero `f.by`/`f.bz`), and
  (2) `combine_stage` then ran `rk_stage` over those slots **and** added the CT EMF
  curl via a separate `launch_mhd_face_b_update`. In a CT scheme the poloidal face B
  must evolve *only* by `∂B/∂t = -∇×E`.
- **Fix applied:** introduced `launch_mhd_emf_curl_rate(emf, dudt, grid, stream)`
  which *overwrites* (`=`) only `dudt.bx_face`/`dudt.by_face` with the EMF curl rate
  (same stencil as `face_b_update`, dt factored out), called in `compute_residual`
  after `launch_mhd_ct_emf`. `combine_stage` now runs a single uniform `rk_stage`
  over all 8 components and **dropped** the separate `face_b_update`. `bz_cell`
  (toroidal, cell-centered) and the 5 fluid vars still flow through
  `flux_difference` unchanged. Discrete `div(curl) = 0`, so a convex combination of
  div-free fields plus `c·(curl rate)` stays div-free.
- **Result:** div(B) dropped 5.25 → **1.73** (improved but **NOT** resolved — see
  open bug #4 for the residual boundary/ghost cause).

---

## ROOT CAUSE (bugs #4, #5, #6 — one defect, now FIXED)

All three "open" failures were a single structural defect: **the high-edge
face/corner layer at index `nx` / `ny` was consumed but never produced.**

The device producer kernels all guarded with `if (i >= g.nx || j >= g.ny) return;`,
writing outputs only over the interior index range `[0,nx) × [0,ny)`:
  * `reconstruct_kernel` (mhd_reconstruct.hip) — L/R interface states.
  * `hlld_kernel`        (mhd_riemann.hip)     — the numerical flux.
  * `ct_emf_kernel`      (mhd_ct.hip)          — the corner EMF `ez_edge`.

But the consumers read the slot at index `nx` / `ny` (the high-edge face of the
last interior cell):
  * `flux_difference_kernel` at cell `i = nx-1` reads the flux at face
    `index(nx, j)` (`F_{i+1/2}`). That slot was never written by `hlld_kernel`, so
    it held stale device-buffer data → the conservative sum did NOT telescope →
    **mass/momentum/energy drift (was bug #5)** and the **cylindrical equilibrium
    drift (was bug #6)** — the cyl seed's geometric source is identically zero
    (v=0, B_phi=0), so the drift was this stale flux, not a missing source term.
  * `emf_curl_rate_kernel` at cell `i = nx-1` reads `ez_edge(i+1=nx, j)`. That
    corner was never written by `ct_emf_kernel`, so the face-B residual at the
    seam was wrong → **div(B) growth (was bug #4, and the cyl div(B) failure)**.

### Fix (applied)
1. **Extend the three producers to the high-edge layer.** Change each guard to
   `if (i > g.nx || j > g.ny) return;` and launch over `(nx+1) × (ny+1)` so the
   `index(nx, ·)` / `index(·, ny)` slots are produced. Index `nx`/`ny` is an
   in-bounds ghost slot (valid `i ∈ [-nghost, nx-1+nghost]`, `nghost ≥ 2`) and the
   stencils these kernels read stay within the padded storage.
   Files: `src/backend/hip/mhd/{mhd_reconstruct,mhd_riemann,mhd_ct}.hip`.
2. **High-edge one-sided CT closure.** `ct_emf_kernel` now also drops the exterior
   column/row at a NON-periodic `x_hi` (`i == nx`) / `y_hi` (`j == ny`) edge
   (`flags.side[1]` / `flags.side[3]`), mirroring the existing low-edge closure.
   Periodic sides keep the two-sided wrap average (bit-identical).
3. **Fill the four CORNER ghost blocks in the boundary BC.** The extended
   `ct_emf_kernel` reads the high-edge corner whose 2×2 cell stencil includes a
   *corner* ghost (both `i` and `j` in the ghost range). `fill_cell` / the y-branch
   of `fill_normal_face` (mhd_boundary.cpp) filled y-side ghosts only over the
   interior width `[0,nx)`, leaving the corners stale. They now span the FULL
   storage width `[-nghost, nx+nghost)`; since the solver fills x-sides before
   y-sides, the already-populated x-ghost columns get copied into the y-ghost rows,
   filling the corners. Without this, div(B) plateaued at ~0.026 localized at the
   `(nx-1, ny-1)` corner; with it, div(B) drops to ~1.5e-14.
   File: `src/physics/mhd/mhd_boundary.cpp`.
4. **Interior-only `state_*` npz output.** Once stepping ran to completion, the CLI
   was found to write `state_*` as the full ghost-padded storage; the .npz contract
   (and every `state_*` reader) is interior-only `nx*ny`. `_flatten_for_npz` /
   `_snapshot_fields` now strip the ghost halo via `_interior_slice`.
   File: `python/quasar/mhd/cli.py`.

The ghost-refill code the earlier report *proposed* for bug #4 (refill ghosts on the
stage output in `combine_stage`, and before measuring in `divergence_b_max`) was
ALREADY present and is correct/load-bearing — it just was not the root cause and on
its own left div(B) at 2.45. It is retained.

---

## Known simplifications (documented by implementers, not yet upgraded)

These are honestly-flagged v1 deviations from the locked design, surfaced during
blind implementation. They do not crash but should be addressed before the module is
considered feature-complete to the approved plan.

1. **Reconstruction is dimension-by-direction, not genuinely multi-dimensional.**
   `flux_reconstruction.cpp` reconstructs along each normal independently; the locked
   decision called for genuinely multi-D corner-coupled reconstruction. Documented as
   a v1 limitation in the file header. The smooth-wave convergence test will witness
   whether the effective order meets the ~7 target once the stepping bugs are fixed.
2. **Device reconstruction path is MUSCL-only.** The device kernel
   (`mhd_reconstruct.hip`) uses 2nd-order MUSCL for all `scheme_order`; MP5/MP7 are
   implemented in the host registry scheme but not ported to the device path.
   NOTE: `flux_reconstruction.cpp` *does* expose its MP5/MP7 limiters as
   `QUASAR_HOST_DEVICE` helpers, so the device kernel can be upgraded to call them.
3. **Device Riemann is full HLLD** (Miyoshi–Kusano) with an HLL fallback — matches
   the host scheme.

---

## Routing summary (who owns each fix)

| Bug | Owner file(s) | Layer |
| --- | --- | --- |
| #1 | `tests/unit/numerics/test_flux_reconstruction.cpp` | test (fixed) |
| #2, #3 | `tests/python/test_mhd_{io,bindings,cli}.py` | test (fixed) |
| #4a | `src/physics/mhd/mhd_boundary.cpp` | source (fixed) |
| #4b | `src/physics/mhd/mhd_solver.cpp` + `src/backend/hip/mhd/mhd_ct.hip` | source (fixed) |
| #4, #5, #6 | `src/backend/hip/mhd/{mhd_reconstruct,mhd_riemann,mhd_ct}.hip` (high-edge producer range + hi one-sided CT closure) | source (FIXED) |
| #4, #5, #6 | `src/physics/mhd/mhd_boundary.cpp` (corner-ghost fill) | source (FIXED) |
| npz shape | `python/quasar/mhd/cli.py` (interior-only `state_*`) | source (FIXED) |

## Reproduce

```bash
JOBID=$(squeue -u "$USER" -h -o "%A" -t R --name=cursor-agent | head -n1)
CT=/home/AMD/abchoudh/.venvs/rocprofiler-compute-asan_ci_test_integration/lib/python3.12/site-packages/cmake/data/bin/ctest
srun --jobid="$JOBID" "$CT" --test-dir build/hip-gfx942-release -R mhd --output-on-failure
```
