# Deferred: field-boundary-condition heisenbug

## Status
Field boundary conditions (PEC walls + explicit periodic ghost fill) are fully
**wired but disabled** behind the `QUASAR_PIC_FIELD_GHOSTS` compile flag. Default
builds use the original implicit periodic wrap (`Grid2D::periodic_index` inside
the FDTD stencil), so periodic runs are correct and PEC field walls are inert.

What IS live and working (Phase 2 delivered):
- The plugin **registry** is now genuinely used: `EmPic2D3V` constructs its
  per-side particle + field BC objects via `Registry<...>::create(name)`.
- WHOLE_ARCHIVE linkage (`REGISTERS` flag in `QuasarAddModule.cmake`) so the
  static-init registrations in `src/boundary` / `src/numerics` survive linking
  (guarded by `tests/unit/boundary/test_registry_linkage.cpp`).
- **Particle** BCs dispatch through `IParticleBoundary::apply` (periodic wrap is
  no longer a silent `break` — fixes the old C4), eliminating the duplicate
  switch in `step()` (D2). The `wall.cpp`/`periodic.cpp` classes are the single
  implementation.

## The bug — ROOT-CAUSED AND FIXED (was never a HIP bug)
The crash was **not** a kernarg/async HIP issue. It was a **registry factory
collision**: `QUASAR_REGISTRY_REGISTER` registered each concrete BC with a
stateless `[]{ return std::make_unique<Class>(); }` lambda. Those lambdas are
structurally identical across types (allocate 8 bytes, store a vptr, return), so
identical-code folding collapsed them into one — and every
`Registry<...>::create(name)` returned whichever concrete type the folded body
kept (a `PecFieldBC`). Calling `IParticleBoundary::apply(species, side)` on a
`PecFieldBC` dispatched through the field-BC vtable, reinterpreting the species
as a `YeeField2D` — which is exactly the "garbage `Grid2D` by value" symptom.

Why the earlier evidence pointed elsewhere: the host `fields_.grid` guard never
fired because the host struct really was fine; the corruption was the *wrong
object/vtable* on the call, observed only on the device side as a bad grid.

### Fix
`include/quasar/core/registry.hpp` now offers type-keyed registration
(`register_type<Derived>()` -> `&make<Derived>`), and the macro uses it. Each
`make<Derived>` is a distinct address-taken function specialization the compiler
cannot fold, so the factory builds the correct concrete type. Verified: particle
BC interface dispatch passes 210/210 stress runs; the field-ghost path no longer
faults either.

## Remaining (field ghosts — a feature, not the bug)
Particle BCs now dispatch through the interface. Field ghosts stay gated behind
`QUASAR_PIC_FIELD_GHOSTS` because the physics is incomplete, not because of a
crash:
1. ~~Root-cause + fix the crash~~ — DONE (registry fix above).
2. Switch `ddx/ddy_staggered` in `include/quasar/numerics/stencil.hpp` from
   `periodic_index` to `index` (ghost-aware) so PEC walls are physical.
3. Restore the `fdtd_order==4 => nghost>=2` constructor check in `pic_solver.cpp`.
4. Add the real PEC reflection test (`test_pec_plane_wave_reflection.cpp`,
   currently a stub) and a periodic-equivalence test (ghost fill must reproduce
   the wrap bit-for-bit), then enable the flag by default.
