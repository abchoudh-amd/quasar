# Quasar Review Examples

Four worked scenarios. All review a commit range or the full repo of the Quasar
tree; none edit code, commit, or push.

## 1. Last-commit review, lint-only fast path

User: "Deeply review my last commit before I push."

1. `quasar-review.sh --scope` → target defaults to `--commits HEAD~1..HEAD`;
   `classification.json` shows `touches_src: true`, `touches_physics: true`,
   `touches_tests_cpp: true`, `touches_changelog: true`.
2. Fan out 10 reviewer subagents in one message + invoke `/code-review`.
3. `quasar-review.sh --lint` → ruff pass, changelog pass, example_test_sync pass.
   `--gates` not passed, so no build/ctest/pytest.
4. Reviewers surface 2 minor findings (a redundant device copy, an unclear name).
   Each goes to 3 skeptics; both survive 3/3.
5. Report inline:

   ```
   # Quasar review — commits (HEAD~1..HEAD)

   2 minor findings kept, 0 pruned. Lint gates green; build/tests not run (no --gates).

   ## Minor
   - src/physics/pic/pic_solver.cpp:188 — expected the B-field SoA buffer to be
     copied once per step but observed a copy inside the particle loop. Suggested:
     hoist the copy above the loop. [performance]
   - src/physics/pic/pic_solver.cpp:241 — variable `tmp2` is unclear. Suggested:
     rename to `e_half`. [maintainability]

   ## Gate results
   | gate | status | note |
   | ruff | pass | no changed Python |
   | changelog | pass | entry present |
   | example_test_sync | pass | no example added |
   | build | skipped | pass --gates to run |
   | ctest | skipped | pass --gates to run |
   | pytest | skipped | pass --gates to run |
   ```

   Then: "Save this to `reviews/commits-HEAD-1-HEAD.md`?" — user declines; nothing
   written.

## 2. Numerics finding — a dt path that skips the CFL guard

User: "Ambitiously review my changes to the PIC time loop."

1. `--scope --commits HEAD~2..HEAD`; `classification.json` shows
   `touches_physics: true`, `touches_numerics: true`.
2. The **numerics** reviewer flags (confidence 84, blocker): "expected every
   integration path to clamp `dt` to the Courant limit via `cfl_dt(grid,
   fdtd_order, c)` (`include/quasar/core/grid.hpp`), but observed a new
   `advance_substep()` that takes a caller-supplied `dt` and never checks it
   against the Yee CFL limit — unlike `advance()` in `src/physics/pic/pic_solver.cpp`
   (~line 272), which rejects oversized `dt`. An over-CFL step makes the FDTD update
   unstable (energy blows up)."
3. Phase 2 spawns 3 skeptics. Reading `grid.hpp` and `pic_solver.cpp`, all three
   confirm the new path bypasses the guard → 3/3 `refuted: false`, kept.
4. Report ranks it **blocker**:

   ```
   ## Blockers
   - src/physics/pic/pic_solver.cpp:301 — expected advance_substep() to clamp/reject
     dt > cfl_dt(...) as advance() does, but observed no CFL check. Suggested: apply
     the same guard (or assert dt <= cfl_dt(grid, fdtd_order, c)) before stepping.
     [numerics]
   ```

This is the domain dimension catching a stability bug a generic correctness reviewer
would likely pass over.

## 3. Physics finding — broken charge conservation, grounded by a gate

User: "Review my deposit change and run the tests."

1. `--scope --commits HEAD~1..HEAD` shows `touches_numerics: true`
   (`numerics/deposit.hpp`), `touches_tests_cpp: false`.
2. The **physics** reviewer flags (confidence 79, major): "expected the deposit to
   satisfy discrete continuity ∂ρ/∂t + ∇·J = 0 (the Esirkepov current must be the
   adjoint of the E-update curl, see `numerics/stencil.hpp`), but observed the
   periodic-axis branch using a forward difference where the wall-axis branch uses a
   backward one — the mismatch breaks Gauss's law at the boundary."
3. `--gates` gate subagent runs the targeted C++ test:
   `ctest --preset hip-gfx942-release -R charge_conservation`. The
   `test_charge_conservation` case **fails**. That finding's evidence becomes
   `automated_check:ctest` → **grounded, skips the skeptic pass**.
4. Report ranks it **blocker** (a failing conservation test is grounded):

   ```
   ## Blockers
   - include/quasar/numerics/deposit.hpp:34 — expected matched finite differences so
     ∂ρ/∂t+∇·J=0 holds on both periodic and wall axes, but the periodic branch's
     current is not the adjoint of the curl. Suggested: use the same difference
     direction as the E-update curl on both axes. [physics] [evidence: automated_check:ctest]

   ## Gate results
   | ctest | fail | test_charge_conservation failed |
   | build | pass | |
   ```

## 4. Full-repo audit (no diff; reviewers read whole files)

User: "Do a thorough numerics/math/physics audit of the whole repo."

1. `quasar-review.sh --scope --full-repo` → `files.txt` lists all tracked source
   under `include/ src/ python/ apps/ tests/`; `diff.patch` is empty. The changelog
   and example_test_sync gates report `pass` with "not applicable in full-repo mode".
2. Fan out the 10 dimensions; the numerics/mathematics/physics reviewers read whole
   files (`numerics/stencil.hpp`, `core/grid.hpp`, `core/normalization.hpp`,
   `boundary/wall.hpp`, the PIC solver) rather than a diff.
3. The **mathematics** reviewer flags (confidence 71, major): "expected curl/grad
   operators to carry coordinate-metric factors so the planned cylindrical r-z path
   (`plans/pic-cylindrical-rz-build-plan.md`) can reuse them, but observed
   `numerics/stencil.hpp` hard-codes Cartesian differences with no metric seam —
   adding r-z later will require touching the operators rather than extending them."
   Note this is a *full-repo, forward-looking* design-of-the-math finding, not a diff
   regression.
4. Skeptics split 2/3 `refuted: false` (one argues r-z is out of current scope) →
   kept as **minor** (downgraded from major given the skeptic split), reported with
   the coordinate-system rationale.

This shows full-repo mode surfacing structural math/physics issues that no
commit-scoped diff would reveal.
