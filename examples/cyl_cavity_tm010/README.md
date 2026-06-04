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
(`bz` slot), so the deck seeds the axial `Ez` and the diagnostic reads the `ey`
slot.

## Reference signature

For a cylindrical deck the `seed_perturbation` is the axisymmetric radial
eigenmode `J0(j01 r/R)` (uniform in `z`), the physically appropriate TM010
excitation — a plain `sin(2 pi r/R)` would be dominated by higher radial Bessel
modes (TM020) and ring at the wrong line. The resonant frequency is **measured
as the dominant FFT peak of the near-axis axial-E (`ey`) time series**, and the
radial profile is checked against `J0(j01 r/R)`. The post-processing:

1. Reads the axial-E snapshots from `out.npz` (`snapshot_field_ey`, with the
   final field in `field_ey`), sampled every `cadence = 4` steps.
2. Takes a representative near-axis `Ez(t)` trace (and/or projects the radial
   profile onto `J0(j01 r/R)`).
3. FFTs that trace and locates the dominant peak frequency `f_peak`.

The measured `f_peak` matches the analytic `f_010 ≈ 1.1473 GHz` to within a few
percent (≈5% at `128 x 128`: the residual is the 2nd-order Yee grid dispersion of
the cylindrical Bessel operator plus the finite FFT bin width), and the radial
profile matches `J0(j01 r/R)` to sub-percent RMS.
