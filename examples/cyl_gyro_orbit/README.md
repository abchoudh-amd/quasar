# cyl_gyro_orbit

A small set of zero-macro-weight electron test particles gyrating in a uniform
axial magnetic field, solved in
axisymmetric cylindrical `(r, z)` coordinates (`geometry: cylindrical`). The grid
axes are `i = r` and `j = z`. This deck validates the cylindrical particle
pusher against the textbook gyrofrequency and checks energy conservation in a
uniform external magnetic field. The `r = 0` axis boundary is configured because
the grid is cylindrical, but this full-domain quiet start is not a near-axis
boundary validation.

## Physical field spelling (recommended) and the slot convention

In cylindrical mode the recommended way to specify a uniform external field is by
its **physical cylindrical axes** with `B_rzphi`, ordered `[B_r, B_z, B_phi]`
(the same `r, z, phi` order as the velocity triad). A pure **axial** field is
`B_z`, so:

```yaml
external_field:
  evaluator: {type: uniform, B_rzphi: [0.0, 1.0, 0.0]}   # B_z = 1.0 T, axial
```

The deck loader translates `B_rzphi` to the storage-slot vector the solver needs
in exactly one place (`quasar.pic.io._rzphi_to_uniform_lab`), so you do **not**
need to know the implicit slot convention below.

For the curious, that convention is: the grid triad is `(i=r, j=z,
out-of-plane=phi)`, so physically `B_r -> bx slot`, `B_z (axial) -> by slot`,
`B_phi -> bz slot`. With `plane: xy` the sampler maps lab axes straight through
to slots (`bx<-lab x`, `by<-lab y`, `bz<-lab z`), and for plane `xy` the
physical `[B_r, B_z, B_phi]` ordering happens to coincide with the slot ordering.
The older raw slot spelling

```yaml
external_field:
  evaluator: {type: uniform, B_T: [0.0, 1.0, 0.0]}   # equivalent for plane xy
```

is still accepted (backward compatible) but requires knowing that the axial field
lives in the **second** (`by`) slot — exactly the footgun `B_rzphi` removes.
Supplying both `B_rzphi` and `B_T` is an error.

The velocity triad in cylindrical mode maps as `vx -> vr`, `vy -> vz`,
`vz -> vphi`. The axial field (`by`) is perpendicular to the `vr`–`vphi`
(`vx`–`vz` slot) plane, so the in-plane velocity components gyrate while the
parallel (`vy -> vz`) drift is unaffected. The seeded `drift_v`
is `(vx, vy, vz) = (5.93e5, 0, 0)`, i.e. a pure `vr` velocity perpendicular to
the axial `B`, which sets a clean perpendicular speed `v_perp`.

## Parameters

- Magnetic field: `B = 1.0 T`, axial.
- Species: electrons, `q = -1.602176634e-19 C`, `m = 9.1093837015e-31 kg`,
  `n_particles = 64`, and `density_per_m3 = 0` (no deposited self-field).
- Perpendicular drift speed: `v_perp ≈ 5.93e5 m/s`, with zero thermal spread
  so every particle has the same reference Larmor radius.
- Grid `128 x 128`, `r ∈ [0.0, 0.10] m`, `z ∈ [-0.05, 0.05] m`, 2nd-order FDTD.
  The axisymmetric `m = 0` scheme requires the radial domain to start at `r = 0`
  (the on-axis closure), so `origin_x_m` is `0`.

## Boundaries

A cylindrical deck must not use a periodic outer-radius (`x_hi`) boundary — that
is rejected at deck load. Since `BoundaryConfig` defaults every side to
`periodic`, the deck sets all sides explicitly:

```yaml
boundary:
  field:    {x_lo: axis, x_hi: pec, y_lo: pec, y_hi: pec}
  particle: {x_lo: axis, x_hi: absorbing, y_lo: absorbing, y_hi: absorbing}
```

`x_lo` is the symmetry axis and is explicitly marked with the on-axis closure.
The outer
radial wall and both axial walls use PEC field boundaries; particles that wander
out of the domain are absorbed (marked dead) rather than wrapped. The orbit is
placed well inside `r > 0` with a tiny Larmor radius, so particles do not
actually reach the walls during the run.

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/cyl_gyro_orbit/input.yaml
```

## Analytical reference

Gyrofrequency (cyclotron frequency):

```
omega_c = |q| B / m
        = 1.602176634e-19 * 1.0 / 9.1093837015e-31
        ≈ 1.7588e11 rad/s      (T_c = 2*pi/omega_c ≈ 3.573e-11 s)
```

Larmor radius for the seeded perpendicular speed:

```
r_L = m v_perp / (|q| B)
    = 9.1093837015e-31 * 5.93e5 / (1.602176634e-19 * 1.0)
    ≈ 3.37e-6 m
```

(For reference, the 100 eV thermal speed `sqrt(2 e T / m)` would give
`v_perp ≈ 5.93e6 m/s` and `r_L ≈ 3.37e-5 m`; this deck uses a smaller, fixed
drift so the orbit radius is deterministic.)

## Reference signature

The integration / post-processing checks:

1. **Perpendicular speed / Larmor scale** — the uniform magnetic field preserves
   the seeded `v_perp`, so the inferred scale
   `r_L = m v_perp / (|q| B)` remains the analytic `3.37e-6 m`.
2. **Gyrofrequency** — the frequency recovered from the velocity trajectory matches
   `T_c = 2*pi*m / (|q| B)` to within the same tolerance.
3. **No magnetic work** — the total kinetic energy stays constant to the
   integration tolerance because a static magnetic field does no work.

## A note on energy and macro-weight

The particles deliberately use `density_per_m3: 0`, which gives them zero
macro-weight. Their physical charge-to-mass ratio still drives the external-field
Boris push, but they deposit no charge or current and therefore create no
self-field. This isolates the cylindrical pusher: an unneutralized finite-density
electron bunch would introduce a real collective electric field and shift the
velocity spectrum away from the textbook single-particle cyclotron frequency.
In the isolated uniform magnetic field the pusher conserves each particle's
speed `|v|` to roundoff because the magnetic force does no work.

The per-species particle state (positions and velocities) is written to
`out.npz` under the `species_electron_*` keys; the axial field is available as
`external_by` / `field_by`.
