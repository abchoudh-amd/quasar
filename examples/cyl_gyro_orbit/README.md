# cyl_gyro_orbit

A small bunch of electrons gyrating in a uniform axial magnetic field, solved in
axisymmetric cylindrical `(r, z)` coordinates (`geometry: cylindrical`). The grid
axes are `i = r` and `j = z`. This deck validates the cylindrical particle
pusher against the textbook Larmor radius and gyrofrequency, and exercises the
on-axis behaviour (no spurious energy gain as a guiding centre approaches `r = 0`).

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
parallel (`vz -> vphi`... see note) drift is unaffected. The seeded `drift_v`
is `(vx, vy, vz) = (5.93e5, 0, 0)`, i.e. a pure `vr` velocity perpendicular to
the axial `B`, which sets a clean perpendicular speed `v_perp`.

## Parameters

- Magnetic field: `B = 1.0 T`, axial.
- Species: electrons, `q = -1.602176634e-19 C`, `m = 9.1093837015e-31 kg`,
  `n_particles = 64`.
- Perpendicular drift speed: `v_perp ≈ 5.93e5 m/s` (plus a 1 eV thermal spread).
- Grid `128 x 128`, `r ∈ [0.0, 0.10] m`, `z ∈ [-0.05, 0.05] m`, 2nd-order FDTD.
  The axisymmetric `m = 0` scheme requires the radial domain to start at `r = 0`
  (the on-axis closure), so `origin_x_m` is `0`.

## Boundaries

A cylindrical deck must not use a periodic outer-radius (`x_hi`) boundary — that
is rejected at deck load. Since `BoundaryConfig` defaults every side to
`periodic`, the deck sets all sides explicitly:

```yaml
boundary:
  field:    {x_lo: pec, x_hi: pec, y_lo: pec, y_hi: pec}
  particle: {x_lo: absorbing, x_hi: absorbing, y_lo: absorbing, y_hi: absorbing}
```

`x_lo` is the symmetry axis and is auto-replaced by the solver's on-axis closure,
so its value is a don't-care (set to `pec`/`absorbing` for clarity). The outer
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

1. **Larmor radius** — the radial excursion of a particle's orbit about its
   guiding centre matches `r_L = m v_perp / (|q| B)` to within the documented
   tolerance (a few percent, covering the finite time step and CIC field
   interpolation).
2. **Gyrofrequency** — the orbital period recovered from the trajectory matches
   `T_c = 2*pi*m / (|q| B)` to within the same tolerance.
3. **No spurious energy gain near the axis** — because some guiding centres sit
   close to `r = 0`, the cylindrical pusher / on-axis closure must not inject
   energy as a particle approaches the axis. The total kinetic energy of the
   bunch stays bounded (no growth beyond the few-% tolerance) over the run.

## A note on energy: this is a self-consistent PIC run, not a single-particle integrator

The cylindrical Boris pusher conserves a single particle's speed `|v|` exactly
(to machine precision) in a uniform axial `B` — the magnetic force does no work.
This deck, however, is a full self-consistent EM-PIC run: the 64 macro-particles
deposit current, which sources self-fields that are gathered back and *do* act on
the particles. Because the Larmor radius (`r_L ≈ 3.4e-6 m`) is far smaller than a
grid cell (`≈ 7.8e-4 m`), the macro-particles are concentrated within a few cells
and their under-resolved self-fields exchange real energy with the bunch (the
familiar PIC grid-heating / finite-grid self-force effect). So per-snapshot bunch
kinetic energy is *not* expected to be constant here; what is validated is the
gyrofrequency and Larmor radius (set by the external field) and the absence of an
*unbounded* energy blow-up. A pure single-particle energy-conservation check would
disable the deposit/self-fields or resolve `r_L` on the grid; that is intentionally
out of scope for this minimal example.

The per-species particle state (positions and velocities) is written to
`out.npz` under the `species_electron_*` keys; the axial field is available as
`external_by` / `field_by`.
