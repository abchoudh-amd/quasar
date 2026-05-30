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

## The bug
Enabling `QUASAR_PIC_FIELD_GHOSTS` (which calls `fill_field_ghosts()` before each
half-update and switches the stencil to ghost reads) causes an **intermittent
(~30% of runs)** `HIP error 700 (illegal memory access)` that surfaces in
`periodic_fields_kernel`. The kernel intermittently receives a **garbage
`Grid2D` by value** (`ny=0`, `nghost=300495749`, random `nx`).

## Ruled out (with evidence)
- **Stencil math**: fault persists with the old wrapping `periodic_index`
  stencil, so it is not the ghost-read change itself.
- **ODR / stale build**: persists after a full `rm -rf build` from-scratch
  rebuild.
- **Corrupt host `fields_.grid`**: a guard comparing `fields_.grid` to `grid_`
  at step entry AND after every sub-stage (fill/advance_b/push/bc/deposit/filter)
  **never fired** — the host struct is valid right up to the launch, yet the
  launcher reads garbage `g.nx/ny/nghost`.
- **Zero-work launch**: guards for `total<=0` / `size()==0` added; not the cause.

## Leading hypothesis
A kernarg-marshaling or async-ordering issue specific to
`periodic_fields_kernel`'s argument list (`Grid2D` by value + 2 ints + 6
`double*`). The constant `nghost=300495749` smells like a fixed wrong offset /
uninitialized kernarg slot. Next steps to try:
- Pass `Grid2D` fields as scalars (nx, ny, nghost, origin, lx, ly) instead of
  the whole struct by value, or pass a `const Grid2D*` device copy.
- Run under `rocgdb` with a hardware watchpoint on the kernarg region; or build a
  minimal repro launching only `periodic_fields_kernel` in a tight loop.
- Check alignment/padding of `Grid2D` (mixed int/double) against the HIP kernarg
  ABI; compare with the working `fdtd_*_kernel` which also takes `Grid2D` by
  value but with fewer trailing pointer args.

## Re-enable checklist
1. Root-cause + fix the kernarg issue above.
2. `-DQUASAR_PIC_FIELD_GHOSTS` (or remove the guard) in the build.
3. Switch `ddx/ddy_staggered` in `include/quasar/numerics/stencil.hpp` from
   `periodic_index` back to `index` (ghost-aware).
4. Restore the `fdtd_order==4 => nghost>=2` constructor check in `pic_solver.cpp`.
5. Add the real PEC reflection test (`test_pec_plane_wave_reflection.cpp`,
   currently a stub) and a periodic-equivalence test (ghost fill must reproduce
   the wrap bit-for-bit).
