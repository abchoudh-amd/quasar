# cyl_cavity_tm010

The TM010 ("pillbox") mode of a cylindrical resonant cavity, solved in the
axisymmetric meridional `(r, z)` plane (`geometry: cylindrical`). The PIC grid
axes are `i = r` and `j = z`; the inner-radius wall (`x_lo`) is the symmetry
axis and is replaced by the solver's on-axis closure, while the outer radial
wall (`x_hi`) and the two axial walls (`y_lo`, `y_hi`) are perfect electric
conductors (PEC). There are no particles — this is a field-only validation deck.

## Geometry and parameters

- Cavity radius `R = lx_m = 0.10 m`.
- Axial height `H = ly_m = 0.10 m`.
- Grid `128 x 128`, 2nd-order FDTD, CIC shape.
- An axial-`Ez` `seed_perturbation` excites the cavity at `t = 0` (the deck spells
  the component with the physical name `Ez`; see "Component convention" below).
  It also initializes `Bphi` at the stored `t = -dt/2` leapfrog time from the
  matching Yee curl, so the standing wave begins at its electric-field maximum.

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/cyl_cavity_tm010/input.yaml
```

## Analytical reference

The TM010 mode of an ideal cylindrical cavity has the radial profile
`Ez(r) = E0 * J0(j01 * r / R)` and is independent of `z`, where `J0` is the
zeroth-order Bessel function and `j01 = 2.40483` is its first zero. Its resonant
frequency depends only on the radius `R`:

```
f_010 = j01 * c / (2 * pi * R)
```

With `R = 0.10 m` and `c = 2.99792458e8 m/s`:

```
f_010 = 2.40483 * 2.99792458e8 / (2 * pi * 0.10) ≈ 1.1473 GHz
```

## Component convention

The deck spells the seed component by its **physical cylindrical name**,
`component: Ez` (axial electric field). In cylindrical mode the deck loader
reinterprets the names `Er` / `Ez` / `Ephi` (and `Br` / `Bz` / `Bphi`) as
physical axes and maps them to the Cartesian-named storage slots in exactly one
place (`quasar.pic.io._resolve_seed_component`):

```
Er  -> ex      Ez (axial) -> ey      Ephi (azimuthal) -> ez
Br  -> bx      Bz (axial) -> by      Bphi (azimuthal) -> bz
```

So `component: Ez` resolves to the **`ey`** slot — the same axial slot the
verified `cyl_gyro_orbit` example uses for its axial `B` (`by`) and axial
velocity (`vy`). Spelling the physical name means you do not have to know that
the axial field lives in the `ey` slot. (A raw slot name such as `Ey` is still
accepted and is equivalent; in cylindrical mode `Ez` is the *physical* axial
field, not the literal `ez` slot — use `Ephi` for the azimuthal `ez` slot.)

The TM010 mode couples the axial `Ez` (`ey` slot) with the azimuthal `Bphi`
(`bz` slot), so the diagnostics record both storage slots.

## Reference signature

For a cylindrical deck the `seed_perturbation` is the axisymmetric radial
eigenmode `J0(j01 r/R)` (uniform in `z`), the physically appropriate TM010
excitation — a plain `sin(2 pi r/R)` would be dominated by higher radial Bessel
modes (TM020) and ring at the wrong line. The resonant frequency is **measured
from the cosine recurrence of the weighted TM010 amplitude**, avoiding FFT-bin
error. The post-processing:

1. Reads axial-E and azimuthal-B snapshots from `out.npz`
   (`snapshot_field_ey` / `snapshot_field_bz`, with final `field_ey` /
   `field_bz`), sampled every `cadence = 4` steps.
2. Projects each full axial-E field onto `J0(j01 r/R)` with the cylindrical
   `r dr dz` control-volume weights and measures off-mode leakage.
3. Fits the projected amplitude to its cosine recurrence, checking both
   frequency and the absence of a half-timestep startup phase error.
4. Reconstructs `Bphi^(n+1/2)` with the same radial Yee curl and verifies the
   full staggered electric/magnetic leapfrog invariant, including half dual-cell
   weights at PEC faces.

The measured frequency is about `1.14741635 GHz`, versus
`f_010 ≈ 1.14742528 GHz`; the remaining `7.8e-6` relative difference is the
expected second-order radial-grid dispersion. The integration bound remains
looser than one observed run for portability while still rejecting a wrong curl,
axis closure, radial measure, or leapfrog initialization.
