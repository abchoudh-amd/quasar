---
name: quasar-build
description: Orchestrates a feature end-to-end in the Quasar simulation framework with no spec required — a planner+critic plan (one user approval), then blind-TDD execution: failing C++/Python tests first, a source implementer that never sees the test code, a GREEN gate that drives fixes as behavioral feedback, docs+changelog, a quasar-review pass, and finally a commit and a confirmation-gated push to main. Use when the user asks to build or implement a feature in Quasar, add a pluggable scheme (field solver / pusher / deposit / boundary / geometry / filter) or a new example, do TDD without writing a spec, or run the Quasar feature pipeline.
---

# Quasar Build

Drive a feature end-to-end in the Quasar tree (HIP/C++20 + pybind11 + Python; EM-PIC
and magnetostatics) **without a spec**. A planner+critic loop produces a short plan
you approve once; then execution runs **blind TDD** — failing tests are written first,
a source implementer that **never reads the test code** makes them pass, and a GREEN
gate converts any failure into behavioral feedback. It finishes by running the
sibling `quasar-review` skill, committing, and pushing to `main` behind one explicit
confirmation.

This is a lighter cousin of the personal `build-feature` skill: no `write-spec` phase,
no `create-pr` ceremony. The orchestrator never writes feature code itself — every
phase hands off to a subagent.

## Quick start

1. Phase 1 — classify the requirement, run planner + **mandatory** critic, get one
   user approval. Save the plan.
2. Phase 2 — RED: parallel test writers write **failing** tests.
3. Phase 3 — GREEN: parallel **blind** implementers write source (no test access).
4. Phase 3g — GREEN gate: build + ctest + pytest; drive fixes as behavioral feedback.
5. Phase 4 — docs + changelog (parallel writers).
6. Phase 5 — `quasar-review` fix loop.
7. Phase 6 — commit the feature, then **confirm** and push to `main`.

## Phase 1: Plan (planner + mandatory critic, one user approval)

No separate entry gate phase — classification is step (a) of Phase 1.

(a) **Classify** the requirement (ask only for missing items, then pause):
- Requirement present? If absent, ask for a one-line feature requirement and stop.
- Which axis: `physics` / `numerics` / `boundary` / `backend` (and which module).
- Touches `python/quasar/` or `bindings/python/`?
- Adds a new `examples/<case>/`?
- `internal-only`? (skip Phase 4 docs)
- `refactor-only`? (no new behavior → route out: do the edit directly, then
  `quasar-review`; this skill is for behavior changes).
- Resolve repo root via `git rev-parse --show-toplevel`.

(b) **Planner** subagent drafts the plan in the [REFERENCE.md](REFERENCE.md) template:
Mermaid architecture diagrams (understand-codebase style) — a call-graph `flowchart TD`
over the feature's new public functions/interfaces and a class diagram of its new
classes —, **behavioral acceptance criteria** (observable,
**no test names**), per-task `Files written:` (disjoint within a phase), public
interface signatures, and — when adding a pluggable scheme — the matching
`docs/dev-guide/adding_a_*.rst` steps plus registry self-registration via
`include/quasar/core/registry.hpp`.

(c) **Critic** subagent (mandatory, separate invocation) adversarially reviews and
emits the verdict JSON. Loop planner↔critic until `verdict=approve`; if they fail to
converge after ~3 rounds, halt and escalate.

(d) **User-approval gate.** Present the plan path + a short summary, then
`AskUserQuestion` (approve / revise). On `revise`, re-enter the planner↔critic loop
with the feedback. Only an explicit approve advances. Save to
`<repo-root>/plans/<feature>-build-plan.md` (see file-output rules).

## Phase 2: RED — failing tests (parallel writers)

Dispatch one `Agent` (`subagent_type="general-purpose"`) per test file, **all in a
single message**. Each writer sees the acceptance criteria + interface signatures and
writes **failing** tests:
- C++ GoogleTest under `tests/unit/<axis>/` (axes: `core`, `numerics`, `physics`,
  `boundary`, `backend`).
- pytest under `tests/python/`.
- a `tests/python/test_examples.py` entry when a new `examples/<case>/` is added
  (CLI run vs. closed-form `.npz` reference — CLAUDE.md convention).

**RED gate:** build the tests and confirm they **fail for the right reason** (missing
symbol / unmet behavior), not a harness compile error. A harness error routes back to
the test writer, not forward.

## Phase 3: GREEN — blind implementation (parallel writers)

Dispatch one implementer subagent per task with disjoint `Files written:`, **all in a
single message**. Each implementer receives **only**: its behavioral criteria slice,
interface signatures, in-scope/out-of-scope paths, and `Files written:`. It is
instructed it **may not read `tests/unit/**`, `tests/python/**`, or run the suite** —
it self-verifies via `cmake --build` only. Implements under `include/` + `src/` (and
`python/quasar/`, `bindings/python/` as needed). See the blindness contract in
[REFERENCE.md](REFERENCE.md).

## Phase 3g: GREEN gate (orchestrator)

The orchestrator is the **only** agent that holds test source and feature source
together. Run the build + tests (commands below). On failure, **translate each failing
test into a behavioral sentence** ("expected charge continuity on the wall axis but
observed ∇·J ≠ 0"), run the blindness stripping procedure (remove test names, node
IDs, `tests/**` paths, runner output), and re-dispatch **only** the implementers whose
`Files written:` intersect the failure. If stripping empties the payload (a pure
test-harness problem), escalate instead of leaking test artifacts. Loop, cap 3, then
escalate residuals.

### Quasar gate commands

```bash
# Build (cmake NOT on PATH — use the cmake 4.x absolute path; runs via srun)
cmake --preset hip-gfx942-release && cmake --build --preset hip-gfx942-release -j
# C++ tests, filtered to the touched axis
ctest --preset hip-gfx942-release -R <regex>
# Python tests — refresh the build tree first (see caveat), then:
PYTHONPATH=build/hip-gfx942-release/python pytest tests/python -k <pattern>
```

**Build-tree Python sync caveat:** edits under `python/quasar/**` do not reach pytest
until re-copied into `build/hip-gfx942-release/python/` — rebuild (or re-copy) before
running pytest. Shell runs route through the `slurm` skill's srun wrapper automatically.

## Phase 4: Docs + changelog (parallel writers)

One writer per docs page + one for the changelog. Add `docs/` RST per the relevant
dev-guide when a new pluggable component or example is introduced (skip if
`internal-only`, with rationale). Add a `CHANGELOG.md` entry at the repo root. Gate:
docs build if configured, else a presence check.

## Phase 5: Pre-commit review (quasar-review fix loop)

Invoke the sibling review skill: `Skill(skill="quasar-review", args="--gates")` scoped
to the change. Fix **every** finding by routing back: source/logic/numerics/math/
physics bug → Phase 3 (blind), missing/weak test → Phase 2, design/scope/convention →
Phase 1. Loop, cap 3, then escalate residuals to the user. This reuses quasar-review's
ten dimensions including numerics / mathematics / physics.

## Phase 6: Commit, then confirm and push to main

1. **Stage only the feature's files** by name — **never** `git add -A` / `git add .`
   (the working tree carries unrelated in-progress work; see CLAUDE.md and the current
   `git status`).
2. Commit with a Conventional Commits message (`feat`/`fix`/`perf`/…) ending with the
   `Co-Authored-By` trailer.
3. **Push confirmation gate.** Pushing to `main` is high blast-radius. Show the commit
   (`git show --stat`) and ask an explicit yes/no via `AskUserQuestion` before
   `git push origin main`. On **no**, stop with the commit in place (nothing lost).

## Behavior rules

- The orchestrator never writes feature/test code itself; it dispatches subagents.
- **Blindness is absolute:** no `tests/**` path, test name, pytest node ID, or runner
  output may ever reach a Phase 3 subagent.
- Parallelism is **intra-phase only**; phases stay sequential to preserve RED→GREEN.
- Enforce the **disjoint `Files written:`** invariant before every parallel dispatch;
  on collision, halt and route back to Phase 1.
- Never `git add -A`; never push to `main` without the confirmation gate.

## Routing

- Refactor-only (no new behavior) → edit directly, then `quasar-review`.
- Reviewing existing commits / the whole repo → `quasar-review`.
- A rocprofiler-compute feature → the personal `build-feature` skill.

## Additional resources

- [REFERENCE.md](REFERENCE.md) — planner/critic/test-writer/blind-implementer prompts, verdict JSON schema, blindness stripping procedure, disjoint-path check, gate commands, plan template, file-output rules.
- [EXAMPLES.md](EXAMPLES.md) — worked scenarios (new deposit scheme via blind TDD, a blind-retry round, internal-only skip, the push-to-main confirmation gate).
