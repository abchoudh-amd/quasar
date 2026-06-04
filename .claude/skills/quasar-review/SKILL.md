---
name: quasar-review
description: Performs an ambitious, multi-agent deep review of the Quasar simulation framework, scoped to a git commit range or the full repo (not a PR diff). Fans out parallel reviewer subagents across ten dimensions — the seven standard ones (correctness, security, performance, design, maintainability, tests, conventions) plus three domain dimensions tuned for an EM-PIC / magnetostatics code: numerics, mathematics, and physics. Runs local ruff lint plus optional build/ctest/pytest gates, prunes findings with a 3-skeptic majority vote, and reports one prioritized review inline. Use when the user asks to deeply/ambitiously review my Quasar changes, review my commits or the whole repo, audit the numerics/math/physics, self-review before committing or opening a PR, or do a thorough review of the simulation code.
---

# Quasar Review

Ambitiously review the Quasar tree — a HIP/C++20 + pybind11 + Python EM-PIC and
magnetostatics framework. Scope is a **git commit range** or the **full repo**,
not a PR or a fixed-base diff. It fans out parallel reviewer subagents, verifies
findings adversarially, and reports inline — it never edits, commits, or pushes.

This is the Quasar-local analog of the personal `deep-review` skill
(which targets rocprofiler-compute PRs). The key addition here is the three
**domain dimensions** — numerics, mathematics, physics — that catch the bugs a
generic reviewer misses: CFL/stability violations, broken charge/energy
conservation, wrong curl/div adjoints, unit and dimensional errors, and
coordinate-system sign mistakes.

Helper script: `.claude/skills/quasar-review/scripts/quasar-review.sh`

## Quick start

1. Phase 0 — `--scope` to cache the scope and classify changed files.
2. Phase 1 — dispatch one reviewer subagent per relevant dimension **in a single
   message**, and invoke the built-in `/code-review` once.
3. Phase 1g — run lint gates (always); run build + ctest + pytest only if the
   user passed `--gates`.
4. Phase 2 — verify each finding with **3 independent skeptics**; keep only
   majority-survivors.
5. Phase 3 — synthesize one prioritized report inline; offer to save it.

## Phase 0: Scope

```bash
.claude/skills/quasar-review/scripts/quasar-review.sh --scope [--commits <range> | --full-repo]
```

Default target is `--commits HEAD~1..HEAD` (the last commit). `--commits` accepts
any git range (`HEAD~3..HEAD`, `<sha>..<sha>`, or a single `<sha>`). `--full-repo`
scopes to all tracked source under `include/ src/ python/ apps/ tests/`. In commit
mode the script writes `diff.patch`; in full-repo mode there is no diff and
reviewers read whole files. The cached `classification.json` (booleans:
`touches_include/src/backend_hip/physics/numerics/boundary/python/tests_cpp/
tests_py/changelog/docs/examples`) tells you which dimensions and gates are
relevant.

## Phase 1: Parallel reviewer fan-out

Dispatch one `Agent` (`subagent_type="general-purpose"`) per relevant dimension,
**all in a single tool-call message** so they run concurrently. **Ten dimensions:**

1. Correctness / logic bugs
2. Security (unsafe subprocess/file handling, deserialization, injection)
3. Performance (hot-path allocations, redundant device/host transfers, O(n²))
4. Design / architecture fit — the four-axis orthogonality (`physics × numerics ×
   boundary × backend`), backend isolation, public-vs-private headers, plugin
   registry (see CLAUDE.md).
5. Maintainability — structural simplification, dead code, type cleanliness.
6. Test coverage adequacy (C++ unit tests + Python tests; new example ⇒ test).
7. Project conventions (CHANGELOG entry, dev-guide steps, registry registration).
8. **Numerics** — CFL/stability, discretization, shape/stencil order, FP guards.
9. **Mathematics** — vector calculus (curl/div/grad adjoints), coordinates, Vec3T.
10. **Physics** — units/dimensions, charge/energy conservation, Maxwell/Gauss, signs, BC physics.

The three domain dimensions (8–10) **always run** when any `include/`, `src/`, or
numerics/physics code is in scope — they are the point of this skill. Each reviewer
reads `diff.patch` (commit mode) or the whole files in `files.txt` (full-repo mode),
may read surrounding source for context, and returns the findings JSON in
[REFERENCE.md](REFERENCE.md). Also invoke the built-in `/code-review` once and fold
in its findings with confidence ≥ 80.

## Phase 1g: Local gates

- **Lint always**: `quasar-review.sh --lint [target]` runs `ruff` on changed
  Python under `python/quasar/` and `tests/python/`, plus static gates (changelog
  presence, example/test sync). Cheap; run every time.
- **Build + tests only with `--gates`**: spawn a gate subagent that runs the
  Quasar build and tests (commands below) via the `slurm` skill's srun wrapper (the
  PreToolUse hook routes shell runs automatically). A failed gate becomes grounded
  `evidence:automated_check:<name>` on the related finding.

### Quasar gate commands

```bash
# Build (cmake is NOT on PATH — use the cmake 4.x absolute path; runs via srun)
cmake --preset hip-gfx942-release && cmake --build --preset hip-gfx942-release -j
# C++ unit tests (filter to the touched axis where possible)
ctest --preset hip-gfx942-release [-R <regex>]
# Python tests (re-copy python/ edits into the build tree first — see REFERENCE)
PYTHONPATH=build/hip-gfx942-release/python pytest tests/python -k <pattern>
```

See [REFERENCE.md](REFERENCE.md) for the build-env caveats (cmake path, build-tree
Python sync) before invoking gates.

## Phase 2: Adversarial verification (3-skeptic majority)

For each finding above the confidence floor (60; 80 for `/code-review`), spawn
**three independent skeptic subagents** that each try to refute it against the real
code (refute-by-default). Keep a finding only when **≥ 2 of 3** return
`refuted: false`. Findings backed by a failed gate skip the skeptic pass — they are
already grounded. See [REFERENCE.md](REFERENCE.md) for the skeptic prompt and tally.

## Phase 3: Synthesize and report

- Dedupe by `(file, line, theme)`; order blocker → major → minor.
- Print **one report inline**: a short opening summary, findings grouped by
  severity (each `path:line` + ≤ 2-sentence rationale + smallest suggested change),
  then a gate-results table.
- After printing, **offer** to save to `<repo-root>/reviews/<target-slug>.md`. Write
  the file only on explicit user confirmation.

## Behavior rules

- **Report-only.** Never edit the working tree, commit, or push. The only file this
  skill may write is the optional saved report, and only after the user says yes.
- Phrase findings as observations and suggestions, not directives.
- A finding without a concrete `file:line` anchor goes in the summary, not as a
  line-anchored item.
- Re-tone any aggressive built-in `/code-review` phrasing into suggestions.

## Routing

- Reviewing a **rocprofiler-compute** PR or branch → the `deep-review` / `review-pr` skills.
- A known crash/regression needing root-cause first → debug the failure, then review.

## Additional resources

- [REFERENCE.md](REFERENCE.md) — ten-dimension focus table (grounded in real Quasar paths), reviewer/skeptic prompts, JSON schema, gate table, build-env caveats.
- [EXAMPLES.md](EXAMPLES.md) — worked scenarios incl. a numerics CFL finding and a physics charge-conservation finding.
