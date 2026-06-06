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
| `mhd_test_mhd_solver_conservation` | **FAIL** | mass/mom/energy drift (open bug #5) |
| `mhd_test_mhd_divergence_free` | **FAIL** | div(B) ~1.73 after one step (open bug #4) |
| `mhd_test_mhd_cylindrical_source` | **FAIL** | equilibrium drift + cyl div(B) (open bug #6) |

Python: `test_mhd_io` + `test_mhd_bindings` pass (49); non-stepping CLI/over-CFL
pass (5). Stepping CLI cases depend on the open numerics bugs below.

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

## OPEN bugs (numerics — not yet fixed)

These are the highest-risk coupled-numerics items called out in the build plan
(FD-CT boundary consistency, conservative-FD telescoping, cylindrical geometric
source). Diagnosed but **fix not applied** (work paused at user request).

### Bug #4 — Residual div(B) growth from boundary/ghost inconsistency
- **Test:** `mhd_test_mhd_divergence_free.SeedIsDivergenceFreeAndStaysSo`
  (div1 = **1.73**, want < 1e-9).
- **Diagnosed cause (two coupled defects):**
  1. **Stale ghosts at measurement.** `MhdSolver2D::divergence_b_max()` calls
     `ct_->divergence_b_linf(state())` with **no ghost refill**. The divergence
     stencil at the last interior cell reads `bx_face(i+1,j)`/`by_face(i,j+1)` —
     ghost faces last filled at the *previous* `compute_residual` top, i.e. before
     the stage updates. Boundary ring sees post-step interior vs pre-step ghost.
  2. **Contaminated ghost-face residual.** `launch_mhd_emf_curl_rate` overwrites
     only the **interior** `dudt` face slots `[0,nx)×[0,ny)`. The **ghost** face
     `dudt` slots still hold the `flux_difference` contamination, and `rk_stage`
     runs over the whole storage (incl. ghosts), so it advances ghost faces by the
     wrong (non-CT) rate. Those wrong ghost faces feed both the next stage's
     reconstruction and the divergence measure.
- **Proposed fix (host orchestration in `mhd_solver.cpp`):** re-fill ghosts on the
  stage **output** inside `combine_stage` (after `rk_stage`) so ghost faces are
  re-derived from the BC each stage; and make `divergence_b_max()` measure a
  ghost-consistent field (fill ghosts before measuring). Currently `fill_ghosts` is
  called only at the *top* of `compute_residual` on the *input*.

### Bug #5 — Fluid conservation drift on a periodic grid
- **Test:** `mhd_test_mhd_solver_conservation.PeriodicConservesMassMomentumEnergy`
  (mass 1067.36 → 1083.77, **+16.4 ≈ 1.5%**; momentum/energy similarly), and
  `AdvanceConservesMass` (+6.05).
- **Diagnosed cause:** likely the same boundary/ghost inconsistency as Bug #4. On a
  periodic domain, `Σ_interior −(F_{i+1/2}−F_{i−1/2})/dx` telescopes to
  `−(F_{nx−1/2} − F_{−1/2})/dx = 0` **only if** the boundary-face fluxes match across
  the period — which requires the reconstruction at face 0 and face nx to read
  periodic-consistent ghost **cells** at every stage. With ghosts only filled on the
  first stage's input, later stages reconstruct from stale/contaminated ghosts and
  the flux no longer telescopes.
- **Proposed verification:** after the ghost fix, instrument a one-`compute_residual`
  call on a periodic uniform state and assert `Σ_interior dudt.rho ≈ 0`. If it is not
  ~0, the residual is in the `flux_difference` periodic wrap or the reconstruction
  ghost read (route to kernels/reconstruction owner), not the host orchestration.

### Bug #6 — Cylindrical (r,z) equilibrium drift + cyl div(B)
- **Test:** `mhd_test_mhd_cylindrical_source` —
  `BalancedEquilibriumStaysStationary` (rho drift 0.209, energy drift 0.244, want
  < 1e-8) and `DivergenceStaysAtMachineEpsilon` (0.55, want < 1e-9).
- **Diagnosed cause(s):**
  1. **Incomplete geometric source.** The geometric-source *kernel*
     (`src/backend/hip/mhd/mhd_update.hip :: geometric_source_kernel`) implements
     only `S_{m_r} = (ρ v_φ² − B_φ²)/r` and `S_{B_φ} = −(v_r B_φ − v_φ B_r)/r`. The
     kernel's own comment also describes a pressure-curvature term and an `m_φ`
     angular-momentum term, **but neither is implemented**. Whether a balanced state
     stays stationary depends on the exact split between the Cartesian-style radial
     `flux_difference` and the geometric source; with terms missing, the seeded
     equilibrium is not discretely balanced and drifts. (Flagged by the source
     implementer at delivery — kernel comment vs code mismatch.)
  2. The cyl div(B) failure shares the boundary/ghost root cause in Bug #4 (plus the
     cylindrical metric in the divergence/curl operators must be verified consistent).
- **Proposed fix:** complete the geometric-source terms to be discretely consistent
  with the radial `flux_difference` convention (so a documented equilibrium is a
  fixed point), in `src/physics/mhd/mhd_geometric_source.*` and the
  `geometric_source_kernel`; re-check after Bug #4's ghost fix lands.

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
| #4b | `src/physics/mhd/mhd_solver.cpp` + `src/backend/hip/mhd/mhd_ct.hip` | source (partial) |
| #4 (open) | `src/physics/mhd/mhd_solver.cpp` (ghost refills) | source (open) |
| #5 (open) | `mhd_solver.cpp`; possibly `mhd_*.hip` flux/reconstruction | source (open) |
| #6 (open) | `src/physics/mhd/mhd_geometric_source.*` + `mhd_update.hip` | source (open) |

## Reproduce

```bash
JOBID=$(squeue -u "$USER" -h -o "%A" -t R --name=cursor-agent | head -n1)
CT=/home/AMD/abchoudh/.venvs/rocprofiler-compute-asan_ci_test_integration/lib/python3.12/site-packages/cmake/data/bin/ctest
srun --jobid="$JOBID" "$CT" --test-dir build/hip-gfx942-release -R mhd --output-on-failure
```
