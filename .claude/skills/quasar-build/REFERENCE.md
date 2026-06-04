# Quasar Build Reference

Subagent prompt templates, verdict JSON schema, blindness stripping procedure,
disjoint-path invariant check, gate commands, plan output template, and file-output
rules for [`SKILL.md`](SKILL.md).

## Agent dispatch

All subagents spawn via the `Agent` tool with `subagent_type="general-purpose"`. The
critic in Phase 1 MUST be a separate invocation from the planner. A Phase 3 blind
implementer MUST never be the same invocation as any Phase 2 test writer (test context
must not bleed into the implementer).

## Parallel-fanout dispatch + disjoint-path invariant

Within Phases 2, 3, and 4 dispatch one subagent per task, **all in a single tool-call
message** so they run concurrently. Before every dispatch, verify the union of
`Files written:` across same-phase tasks has no collisions:

```python
def assert_disjoint_paths(tasks):
    seen = {}
    for t in tasks:
        for p in t.files_written:
            if p in seen:
                halt(reason="parallel-conflict", route_to="plan",
                     detail=f"Tasks {seen[p]} and {t.id} both claim {p}")
            seen[p] = t.id
```

On collision, halt and re-enter Phase 1 with the conflict as critic feedback. Wait for
ALL subagents in a phase to return before running that phase's gate.

## Phase 1: planner subagent prompt template

```
You are the PLANNER for the quasar-build orchestrator (Quasar: HIP/C++20 + pybind11 +
Python; EM-PIC and magnetostatics). There is NO spec — work from the requirement.

Requirement: <one-line feature requirement>
Classification:
- Axis: <physics|numerics|boundary|backend> ; module: <name>
- Touches python/quasar or bindings/python: <yes|no>
- Adds a new examples/<case>/: <yes|no>
- internal-only (skip docs): <yes|no>
- Critic feedback from previous round (if any): <issues[]>
- User-revision feedback (if any): <notes>

Produce a plan in the output template in REFERENCE.md. Requirements:
- Behavioral acceptance criteria: observable terms ONLY, NO test names/paths.
- `### Architecture`: at least one ASCII diagram fenced as ```text (write-plan
  conventions); name each component's responsibility in prose.
- Per writer-phase task (Phases 2, 3, 4): a `Files written:` list, DISJOINT within
  the phase.
- Public interface signatures the tests and the implementer will both target
  (function/class/CLI/registry name). These are the contract that keeps the
  implementer blind to tests.
- If adding a pluggable scheme, cite the matching docs/dev-guide/adding_a_*.rst and
  require self-registration via include/quasar/core/registry.hpp.
- If adding an example, require both examples/<case>/ (YAML + README) and a
  tests/python/test_examples.py entry.
- Quasar build is HIP-only (hip-gfx942-* / gfx950-* presets); do not propose a host
  backend.

Output the full template content as Markdown. Do not invent test file contents.
```

## Phase 1: critic subagent prompt template (MANDATORY)

```
You are the CRITIC for the quasar-build orchestrator. Adversarially review the plan
below and emit a verdict in the JSON schema in REFERENCE.md.

Plan: <full plan content>
Requirement + classification: <inputs>

Check for:
- Behavioral acceptance criteria are observable and contain NO test names/paths.
- Interface signatures are concrete enough that an implementer who CANNOT read the
  tests could satisfy them (this is the crux of blind TDD).
- `### Architecture` has an ASCII diagram fenced as ```text; components named.
- Per-task `Files written:` populated and DISJOINT within each phase.
- Axis placement respects the four-axis orthogonality and backend isolation
  (HIP/device code only under src/backend/hip/); pluggable schemes self-register.
- New example has both the deck/README and a test_examples.py entry.
- Missing failure modes, conservation/stability concerns (CFL, charge/energy),
  unit/dimensional consistency, scope creep.

Output exactly one JSON verdict. loopback_target must be "plan" for any non-approve.
```

## Phase 2: test-writer subagent prompt template

```
You are a TEST WRITER for the quasar-build orchestrator, RED phase. Write FAILING
tests that pin the behavior below. The implementation does not exist yet (or is
incomplete) — your tests are expected to fail now.

Target test file: <tests/unit/<axis>/test_*.cpp | tests/python/test_*.py |
                   tests/python/test_examples.py entry>
Behavioral acceptance criteria for this slice: <observable bullets>
Public interface signatures to target: <signatures / registry names>
Quasar conventions: C++ tests use GoogleTest under tests/unit/<axis>/ mirroring the
public header tree; Python tests use pytest under tests/python/. A new example is
exercised by a CLI-run-vs-.npz-reference case in tests/python/test_examples.py.

Write ONLY the test file at the target path. Assert the behavior, not the
implementation details. Do not stub or write any source under include/ or src/.
```

## Phase 3: blind-implementer subagent prompt template

```
You are a SOURCE IMPLEMENTER for the quasar-build orchestrator, GREEN phase. You are
BLIND to tests by contract.

HARD CONSTRAINTS:
- You MUST NOT read, open, grep, glob, or list anything under tests/unit/ or
  tests/python/. You MUST NOT run ctest or pytest. Verify ONLY via `cmake --build`.
- If you think you need to see a test to proceed, STOP and report what behavioral
  detail is ambiguous — do not seek out test files.

Your task:
- Behavioral acceptance criteria slice (observable): <bullets>
- Public interface signatures to implement EXACTLY: <signatures / registry names>
- Files written (only these): <paths under include/, src/, python/quasar/, bindings/>
- In-scope paths: <...>   Out-of-scope paths: <...>
- Quasar conventions: backend/device code only under src/backend/hip/; everything
  else goes through include/quasar/backend/{device,memory}.hpp. Only include/quasar/
  is public. Pluggable schemes self-register via include/quasar/core/registry.hpp so
  the YAML/Python deck selects by string. Build is HIP-only.

Implement until the criteria are met to the best of your judgment; confirm it compiles
with the build command. Return your diff. Do not touch any path outside Files written.
```

## Verdict JSON schema (Phase 1 critic; reused for any reviewer routing)

```json
{
  "phase": "plan",
  "verdict": "approve | request-changes | block",
  "loopback_target": "plan | none",
  "issues": [
    { "severity": "blocker | major | minor",
      "description": "expected X but observed Y (observable, no test names)",
      "evidence": "plan section / source path / 'automated_check:<name>'" }
  ]
}
```

Rules: `request-changes` ⇒ non-`none` `loopback_target`; `block` ⇒ ≥1 `blocker` issue.

## GREEN-gate failure translation + blindness stripping procedure

Run on every Phase 3g failure before re-dispatching a blind implementer:

1. For each failing test, write ONE behavioral sentence describing the unmet behavior
   ("expected the 4th-order curl to be zero on a constant field but observed a nonzero
   residual"). Do not quote the assertion.
2. Strip anything matching: `test_*`, `*_test`, `*.spec.*`, `conftest.py`; a pytest
   node ID (`path::Class::method`) or GoogleTest name (`Suite.Case`); any path under
   `tests/`; runner-output markers (`==== test session starts ====`, `--- FAIL`,
   `[  FAILED  ]`, `[  RUN     ]`).
3. Keep evidence only when it references a source file, plan section, or
   `automated_check:<name>`.
4. Map each sentence to the implementer whose `Files written:` owns the behavior;
   re-dispatch only those. Untouched implementers do not re-run.
5. If stripping leaves no behavioral content (failure was a pure test-harness issue),
   do NOT dispatch an implementer — fix the test (route to Phase 2) or escalate.

The orchestrator is the only context holding both test and source; this procedure is
what preserves the implementer's blindness across retries.

## Gate command table

| Gate | Phase | Command |
| --- | --- | --- |
| RED (tests fail) | 2 | Build the new tests; confirm failure is missing-symbol/unmet-behavior, not a harness compile error. |
| build | 3, 3g | `cmake --preset hip-gfx942-release && cmake --build --preset hip-gfx942-release -j` |
| C++ tests | 3g | `ctest --preset hip-gfx942-release -R <axis-regex>` |
| Python tests | 3g | refresh build tree, then `PYTHONPATH=build/hip-gfx942-release/python pytest tests/python -k <pattern>` |
| docs | 4 | docs build if configured, else presence check of the new RST + TOC |
| review | 5 | `Skill(skill="quasar-review", args="--gates")` scoped to the change |

**Caveats** (memory `reference_quasar_build_env`, `reference_build_tree_python_sync`):
cmake is not on PATH (use the cmake 4.x absolute path; runs via `srun`); Python edits
under `python/quasar/**` must be re-copied into `build/hip-gfx942-release/python/`
(or rebuilt) before pytest sees them. Build is HIP-only.

## Plan output template

```md
# Feature Plan: [Feature Name]

## Classification
- Axis / module:
- Touches python/quasar or bindings/python: yes / no
- Adds new examples/<case>/: yes / no
- internal-only (skip docs): yes / no
- refactor-only: yes / no  (if yes — route out of this skill)
- Open questions:

## High-level design
### Architecture
```text
  +-----------+      +-----------+
  | Component | ---> | Component |
  +-----------+      +-----------+
```
- Components / modules touched (new vs modified):
- Axis placement & ownership boundaries:
- Dev-guide page (if pluggable scheme): docs/dev-guide/adding_a_*.rst
- Registry registration point: include/quasar/core/registry.hpp

### Key interfaces  (the blind contract)
- Public signatures (function / class / CLI / registry name):
- Integration points (header : symbol):

## Behavioral acceptance criteria  (observable, NO test names)
- [criterion 1]
- [criterion 2]

## Phase 2: RED tests  (parallel writers)
### Task 2.x: [test group]
- Files written: [`tests/unit/<axis>/test_x.cpp`]   # disjoint within phase
- Behavior pinned:
- Interface targeted:

## Phase 3: GREEN source  (parallel BLIND implementers)
### Task 3.x: [implementation slice]
- Files written: [`include/...`, `src/...`]   # disjoint within phase; NO tests/
- In-scope paths:  / Out-of-scope paths:
- Behavioral acceptance criteria for this slice (no test names):
- Interface signatures to implement exactly:

## Phase 4: Docs + changelog
- Status: applicable | skipped (internal-only)
### Task 4.x: [page / entry]
- Files written: [`docs/...rst`] / [`CHANGELOG.md`]
- Dev-guide category / changelog section:

## Phase 5: quasar-review fix loop
- Scope: <touched paths>

## Phase 6: Commit + push to main
- Files to stage (explicit, no `git add -A`):
- Commit message: `feat(<scope>): ...`
- Push gate: confirm before `git push origin main`

## Delivery checklist
- [ ] Acceptance criteria observable, no test names
- [ ] `### Architecture` has a ```text ASCII diagram
- [ ] Interface signatures concrete enough for a test-blind implementer
- [ ] Every writer task has disjoint `Files written:`
- [ ] Phase 3 `Files written:` contains NO tests/ paths
- [ ] New example has deck+README AND a test_examples.py entry
- [ ] Pluggable scheme cites a dev-guide page and self-registers
```

## File-output rules

1. Save the plan under `<repo-root>/plans/` (resolve via `git rev-parse
   --show-toplevel`; create the dir if needed).
2. Kebab-case name: `<repo-root>/plans/<feature>-build-plan.md`.
3. If the filename exists, append `-v2` rather than overwriting.
4. Include the exact saved path in the final response.
