# Quasar Build Examples

Worked scenarios for the spec-free, blind-TDD orchestrator. All run on the Quasar tree
and finish at a commit plus a confirmation-gated push to `main`.

## 1. Happy path — add a new deposit scheme via blind TDD

User: "Add a new charge-conserving deposit scheme `vb_deposit` for the PIC module."

**Phase 1 (plan).**
- Classify: axis `numerics`; module deposit; touches python `no`; new example `no`;
  internal-only `no`; refactor-only `no`. Repo root resolved.
- Planner drafts the plan: cites `docs/dev-guide/adding_a_deposit_scheme.rst`, requires
  self-registration via `include/quasar/core/registry.hpp`, and defines the interface
  contract — `class VbDeposit2D : public IDepositScheme` with the registry name
  `"vb"`. Acceptance criteria (observable): "discrete continuity ∂ρ/∂t + ∇·J = 0 holds
  cell-by-cell on both periodic and wall axes"; "selecting deposit `vb` in a deck runs
  without error". `Files written:` — Phase 2 `tests/unit/numerics/test_vb_deposit.cpp`;
  Phase 3 `include/quasar/numerics/vb_deposit.hpp`, `src/physics/pic/pic_solver.cpp`
  (registration).
- The plan's `### Architecture` carries the two required Mermaid diagrams — a
  call graph and a class diagram of the new scheme:
  ```mermaid
  flowchart TD
    cli[pic.cli run] --> step[EmPicSolver.step]
    step --> deposit[VbDeposit2D.deposit]
    deposit --> J[JField2D]
  ```
  ```mermaid
  classDiagram
    IDepositScheme <|-- VbDeposit2D
    VbDeposit2D : +deposit(species, J)
  ```
- **Critic** flags that the criteria don't state the shape order coverage; planner adds
  "holds for CIC and TSC shape orders". Critic returns `approve`.
- User approves. Plan saved to `plans/vb-deposit-build-plan.md`.

**Phase 2 (RED).** One test writer writes `tests/unit/numerics/test_vb_deposit.cpp`
asserting the continuity residual is ~0 for CIC and TSC. RED gate: the suite builds but
`test_vb_deposit` fails (the `"vb"` scheme isn't registered) — failing for the right
reason. ✓

**Phase 3 (GREEN, blind).** One implementer gets ONLY the criteria + the
`IDepositScheme` / `"vb"` signatures + its `Files written:`. Its prompt forbids reading
`tests/**` or running the suite; it verifies with `cmake --build`. It writes
`vb_deposit.hpp` and registers `"vb"`.

**Phase 3g (GREEN gate).** Orchestrator runs `ctest -R deposit`. Pass → proceed.

**Phase 4.** Docs writer adds a note to the deposit dev-guide / reference; changelog
writer adds an "Added" entry to `CHANGELOG.md`.

**Phase 5.** `Skill(quasar-review, --gates)` over the change → the **physics** reviewer
confirms continuity; clean.

**Phase 6.** Stage exactly the four feature files (no `git add -A`), commit
`feat(pic): add charge-conserving vb deposit scheme`. Show `git show --stat`, ask to
push. User says yes → `git push origin main`.

## 2. A blind-retry round (the core mechanism)

Continuing scenario 1, suppose the GREEN gate's `ctest -R deposit` fails: the wall-axis
continuity assertion fails.

1. The orchestrator does NOT forward the assertion. It writes one behavioral sentence:
   "expected ∂ρ/∂t + ∇·J = 0 on the wall axis but observed a nonzero residual at the
   boundary cells" and strips all test artifacts (no `test_vb_deposit`, no node ID, no
   `tests/` path, no runner output).
2. It re-dispatches ONLY the implementer whose `Files written:` owns `vb_deposit.hpp`,
   with that sentence as an added acceptance criterion. The implementer — still blind —
   fixes the boundary difference direction.
3. GREEN gate re-runs `ctest -R deposit` → pass. The implementer never saw a single
   line of test code.

This is the blindness-preservation procedure (REFERENCE.md) doing its job inside a TDD
loop: the failure informs the fix as *behavior*, never as a test artifact.

## 3. Internal-only change (docs skipped)

User: "Speed up the SoA B-field copy in the PIC step — internal only."

- Phase 1: classify `internal-only: yes`. Criteria are observable but internal
  ("per-step B copy happens once, not per particle; results bit-identical"). Critic
  approves; user approves.
- Phase 2: a unit test pins identical field output before/after.
- Phase 3: blind implementer hoists the copy.
- Phase 3g: ctest green.
- Phase 4: **skipped** with rationale "internal-only"; changelog still gets an
  "Optimized" entry.
- Phase 5: quasar-review (performance + correctness clean).
- Phase 6: commit `perf(pic): hoist SoA B-copy out of particle loop`; push gate.

## 4. Push-to-main confirmation gate (declined)

End of any run, Phase 6:

```
Committed: feat(pic): add charge-conserving vb deposit scheme
 include/quasar/numerics/vb_deposit.hpp        | 84 +++++++
 src/physics/pic/pic_solver.cpp                |  3 +
 tests/unit/numerics/test_vb_deposit.cpp       | 61 ++++++
 docs/dev-guide/adding_a_deposit_scheme.rst    |  6 +
 CHANGELOG.md                                  |  1 +

Push this commit to origin/main? [yes/no]
```

User answers **no**. The skill stops here: the commit is already in place on the local
`main`, nothing is lost, and the user can push later or amend. The skill never pushes
to `main` without an explicit yes — pushing is the one high-blast-radius step and it is
always gated.

## Anti-patterns (do not do)

- ❌ Forwarding a failing assertion, pytest node ID, or `tests/...` path into a Phase 3
  implementer prompt — breaks blindness. Translate to behavior first.
- ❌ `git add -A` / `git add .` in Phase 6 — the tree has unrelated in-progress work;
  stage feature files by name only.
- ❌ Making the Phase 1 critic optional — it is mandatory; the plan must reach
  `verdict=approve` before the user gate.
- ❌ Letting an implementer "peek" at a test to disambiguate — it must instead report
  the ambiguous behavioral detail back to the orchestrator.
- ❌ Running pytest against stale `python/quasar/**` edits — refresh the build tree
  first.
