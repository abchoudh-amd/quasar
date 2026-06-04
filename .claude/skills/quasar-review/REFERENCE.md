# Quasar Review Reference

Ten-dimension focus table (grounded in real Quasar paths), reviewer/skeptic prompt
templates, JSON schema, gate command table, and build-env caveats for
[`SKILL.md`](SKILL.md).

## Agent dispatch

All subagents spawn via the `Agent` tool with `subagent_type="general-purpose"`.
The skeptics in Phase 2 MUST be separate invocations from the Phase 1 reviewer that
produced the finding (independent perspectives). Each skeptic is independent of the
other two.

## Parallel-fanout dispatch rule

In Phase 1, dispatch one reviewer subagent per relevant dimension **all in a single
tool-call message** so they execute concurrently. Wait for every reviewer to return
before pooling findings. Skip a dimension whose `classification.json` signal is
false and irrelevant (e.g. skip `conventions` changelog check when no code changed),
**but** correctness / security / performance / design / maintainability and the
three domain dimensions (numerics / mathematics / physics) always run when any
`include/`, `src/`, or numerics/physics code is in scope.

## Commit-vs-full-repo dispatch note

- **Commit mode** (`--commits`, default `HEAD~1..HEAD`): reviewers read
  `diff.patch` and report issues the diff introduces or leaves unaddressed; anchors
  come from the new/RIGHT side of the diff.
- **Full-repo mode** (`--full-repo`): there is no diff. Reviewers read whole files
  from `files.txt` and report issues anywhere in the tracked source; anchors are the
  current `path:line`. The changelog / example-test gates are not applicable in this
  mode (the helper marks them `pass` with a note).

## Per-dimension focus list

| Dimension | What the reviewer hunts for |
| --- | --- |
| correctness | Logic errors, off-by-one in grid/ghost indexing, wrong conditionals, unhandled empty/zero cases, broken control flow, regressions vs. prior behavior. |
| security | Unsafe `subprocess`/`shell=True`, untrusted YAML deck / path / file handling in `python/quasar/**/cli.py`, unsafe deserialization (`.npz`/YAML), secrets in code. |
| performance | Hot-path host/device transfers, per-step reallocations, redundant Biot–Savart/field-eval work, O(n²) particle loops, needless copies (SoA vs AoS), avoidable kernel launches. |
| design | Four-axis orthogonality (`physics × numerics × boundary × backend`): a change in one axis should not touch another. Backend isolation — HIP/device code only under `src/backend/hip/`; everything else goes through `include/quasar/backend/{device,memory}.hpp`. Public-vs-private — only `include/quasar/` is installed; no including from `src/` across TUs. Plugin registry — schemes/BCs self-register via `include/quasar/core/registry.hpp` so YAML/Python selects by string; drivers stay free of `if/else` over physics types. Phrase each finding as a concrete remedy (move to correct axis/layer, route through the backend abstraction, register the scheme). |
| maintainability | Structural simplification (eliminate a whole branch/layer over local polish), tangled control flow, dead code, leaky abstractions, duplication, type cleanliness (needless optionality, stringly-typed values that want an enum), thin wrappers re-implementing an existing helper. Favor the simplest maintainable remedy. |
| tests | New/changed behavior lacking tests. C++ unit tests under `tests/unit/<axis>/` mirror the public header tree; Python tests under `tests/python/`. A new `examples/<case>/` needs a README **and** a matching entry in `tests/python/test_examples.py` (CLI run vs. closed-form `.npz` reference). |
| conventions | Missing `CHANGELOG.md` (repo root) entry for user-facing changes; dev-guide steps under `docs/dev-guide/` followed when adding a pluggable component (`adding_a_field_solver`, `adding_a_pusher`, `adding_a_deposit_scheme`, `adding_a_boundary`, …); registry registration present; HIP-only build assumption (configuring with `-DQUASAR_ENABLE_HIP=OFF` is a hard error). |
| **numerics** | **CFL / time-step stability** — does new/changed time integration respect the Courant limit? `cfl_dt(grid, fdtd_order, c)` in `include/quasar/core/grid.hpp` (~line 96); the PIC `advance()` dt rejection in `src/physics/pic/pic_solver.cpp` (~line 272). Flag any `dt` path that bypasses the guard. **Discretization & indexing** — grid spacing `dx()/dy()`, ghost halo `nghost`, periodic wrapping (`core/grid.hpp`); Yee staggering (`core/yee_field.hpp`). **Shape / interpolation order** — CIC/TSC weights (`include/quasar/numerics/shape.hpp`). **Stencil order & convergence** — FD curl operators and 2nd/4th-order accuracy (`include/quasar/numerics/stencil.hpp`; convergence test `tests/unit/physics/pic/test_fdtd_order_convergence.cpp`, `tests/unit/numerics/test_stencil_order.cpp`). **FP precision & singularity guards** — `kEps_v`/`kRelEps_v` tolerances (`include/quasar/core/types.hpp`); geometry-scaled Biot–Savart near-line guard (`src/backend/hip/magnetostatics/biot_savart_segment.hpp`). |
| **mathematics** | **Vector calculus correctness** — curl/div/grad operators and the **adjoint relationship** that makes the scheme charge-conserving: forward-difference `curl_b` (E-update) is the adjoint of the backward-difference divergence in the deposit; breaking it breaks Gauss's law (`include/quasar/numerics/stencil.hpp`). **Coordinate systems** — lab↔PIC frame mapping and slice `plane` ("xy"/"xz") in `include/quasar/physics/pic/pic_solver.hpp`; planned cylindrical r-z (`plans/pic-cylindrical-rz-build-plan.md`, `specs/`) — watch for Cartesian assumptions (e.g. missing 1/r metric factors) leaking into would-be-general code. **Linear algebra** — `Vec3T<T>` operators in `include/quasar/core/types.hpp`. **Numerical integration** — quadrature/weights in shape functions and segment integration. |
| **physics** | **Units & dimensional consistency** — SI reference constants and unit tags (`include/quasar/core/normalization.hpp`); precision-templated `mu0_v`, `mu0_over_4pi_v`, `pi_v` (`include/quasar/core/types.hpp`); natural-unit assumptions (c=ε₀=μ₀=1) applied consistently. **Conservation laws** — discrete charge continuity ∂ρ/∂t+∇·J=0 (Esirkepov deposit `include/quasar/numerics/deposit.hpp`; `tests/unit/physics/pic/test_charge_conservation.cpp`); energy u_em=½(E²+B²) + kinetic (`include/quasar/physics/pic/diagnostics.hpp`; `tests/unit/physics/pic/test_energy_conservation.cpp`). **Maxwell / Gauss's law** preservation (`numerics/stencil.hpp` comments). **Sign conventions & frame mapping** (`tests/unit/physics/pic/test_external_plane_mapping.cpp`). **Boundary physics** — PEC tangential-E pinning, specular reflection, absorbing outflow (`include/quasar/boundary/wall.hpp`). |

## Reviewer subagent prompt template

```
You are a REVIEWER for the quasar-review skill, dimension: <DIMENSION>.

You are reviewing the Quasar simulation framework (HIP/C++20 + pybind11 + Python;
EM-PIC and magnetostatics). This is a self-review meant to catch issues early.
Be thorough but fair.

Review target: <commit range "<range>"> OR <full repo (all tracked source)>.

<COMMIT MODE>
Diff to review:
<contents of /tmp/quasar-review-<slug>/diff.patch>
Report only issues the diff introduces or leaves unaddressed; anchor to the
new/RIGHT side of the diff.
</COMMIT MODE>

<FULL-REPO MODE>
Files in scope (read the ones relevant to your dimension):
<contents of /tmp/quasar-review-<slug>/files.txt>
Report issues anywhere in the tracked source; anchor to current path:line.
</FULL-REPO MODE>

Changed-file classification:
<contents of classification.json>

Your focus for this dimension:
<focus row from the per-dimension table in REFERENCE.md>

You MAY read surrounding source for context (note: only include/quasar/ is the
public interface; src/ is private). For numerics/mathematics/physics, ground every
finding in the actual scheme — cite the operator, conservation law, CFL guard, unit
tag, or coordinate mapping it violates, and reference the real file.

For each issue, return a finding with: severity, a 0-100 confidence, the exact file
and line, a description phrased as "expected X but observed Y", the smallest
suggested change, and evidence (a source reference). If you find nothing, return an
empty findings array.

Output exactly one JSON object matching the finding schema in REFERENCE.md. Do not
include prose outside the JSON.
```

## Finding JSON schema

```json
{
  "dimension": "correctness|security|performance|design|maintainability|tests|conventions|numerics|mathematics|physics",
  "findings": [
    {
      "severity": "blocker|major|minor",
      "confidence": 0,
      "file": "src/physics/pic/pic_solver.cpp",
      "line": 0,
      "description": "expected X but observed Y",
      "suggestion": "smallest change that resolves it",
      "evidence": "source reference, or automated_check:<name> if backed by a gate"
    }
  ]
}
```

The orchestrator assigns each finding a stable `finding_id` (e.g. `<dimension>-<n>`)
before Phase 2 so skeptic votes can be tallied.

## Confidence floor

Before the skeptic pass, drop findings with `confidence < 60` (and built-in
`/code-review` findings with `confidence < 80`). Survivors go to the 3-skeptic vote.

## Skeptic subagent prompt template

```
You are a SKEPTIC for the quasar-review skill. Your default stance is that the
finding below is WRONG. Try hard to refute it against the actual code.

Finding:
<finding JSON, including file, line, description, suggestion>

Relevant code (the diff hunk or whole file) and surrounding source:
<the hunk/file plus any context you read>

Refute the finding if ANY of these hold: the described behavior does not occur; the
code already handles the case (e.g. the CFL guard is applied upstream, the deposit
already preserves continuity, units already cancel); the "issue" is intended/correct
per the scheme; the file/line anchor is wrong; the suggestion would not improve
anything. Only let it stand if you genuinely cannot refute it.

Output exactly: {"finding_id": "<id>", "refuted": true|false, "reason": "<one sentence>"}
```

## 3-skeptic majority rule

For each surviving finding, spawn exactly three skeptics (independent invocations,
ideally dispatched together). Tally:

- Keep the finding when **≥ 2 of 3** return `refuted: false`.
- Drop it otherwise; note the dropped false-positive count in the report summary.
- A finding whose `evidence` is `automated_check:<name>` (a real failed gate) is
  grounded and **skips** the skeptic pass — keep it.

## Gate command table

| Gate | When | How |
| --- | --- | --- |
| ruff (lint) | always | `quasar-review.sh --lint` runs `ruff check` on changed `python/quasar/**/*.py` and `tests/python/**/*.py`. Warns (not fails) if ruff is absent. |
| changelog presence | always (commit mode) | Static gate inside `--lint`; warns when `include/`/`src/`/`python/` changed but `CHANGELOG.md` did not. |
| example/test sync | always (commit mode) | Static gate inside `--lint`; warns when `examples/` changed without a matching `tests/python/test_examples.py` change. |
| build | only with `--gates` | Gate subagent runs `cmake --preset hip-gfx942-release && cmake --build --preset hip-gfx942-release -j`. |
| C++ tests | only with `--gates` | Gate subagent runs `ctest --preset hip-gfx942-release` (filtered with `-R` to the touched axis where possible). |
| Python tests | only with `--gates` | Gate subagent runs `PYTHONPATH=build/hip-gfx942-release/python pytest tests/python -k <pattern>`. |

Build/test runs go through the `slurm` srun wrapper automatically (the PreToolUse
hook). A failing build/test attaches `evidence: "automated_check:build"` /
`"automated_check:ctest"` / `"automated_check:pytest"` to the relevant finding(s),
grounding them (they skip the skeptic pass).

## Build-env caveats (read before `--gates`)

- **cmake is not on PATH.** Use the cmake 4.x absolute path and run the build/test
  via `srun` (Slurm). See the `slurm` skill and the user memory
  `reference_quasar_build_env`.
- **Build-tree Python sync.** Edits to `python/quasar/**` do not reach pytest until
  they are re-copied into `build/hip-gfx942-release/python/`. If a gate runs pytest
  after a source change, ensure the build tree is refreshed first (memory
  `reference_build_tree_python_sync`). For a pure review (no edits) this is moot.
- **HIP-only build.** Only the `hip-gfx942-*` / `hip-gfx950-*` presets exist;
  configuring with `-DQUASAR_ENABLE_HIP=OFF` is a hard error.

## Report shape (Phase 3)

```
# Quasar review — <target> (<range or "full repo">)

<1-2 sentence overall summary: N findings kept, M false-positives pruned, gates run.>

## Blockers
- path:line — description. Suggested: <smallest change>. [dimension]

## Major
- path:line — ... [dimension]

## Minor
- path:line — ... [dimension]

## Gate results
| gate | status | note |
| ruff | pass/warn/fail | ... |
| changelog | pass/warn | ... |
| example_test_sync | pass/warn | ... |
| build | pass/fail/skipped | ... |
| ctest | pass/fail/skipped | ... |
| pytest | pass/fail/skipped | ... |
```

After printing, offer: "Save this to `reviews/<target-slug>.md`?" Write only on yes.
