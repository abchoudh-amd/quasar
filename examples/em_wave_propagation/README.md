# em_wave_propagation

A vacuum plane wave seeded in `Ez`/`By`, propagating in `+x` on a fully periodic
grid with no particles. The initializer uses the fourth-order Yee modified
wavenumber and stores `By` at `t=-dt/2`, so this is an exact discrete traveling
eigenmode rather than a continuum wave sampled at inconsistent leapfrog times.

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/em_wave_propagation/input.yaml
```

`units: normalized`. Output `out.npz` keys:

- `snapshot_field_ez`, `snapshot_field_by` — field snapshots (cadence 8).
- `field_ez`, `field_by` — final fields.

## Reference signature

The integration test checks `Ez(x,t)` and `By(x,t-dt/2)` directly against the
fourth-order discrete dispersion relation. It removes ghosts and the duplicated
periodic high face, then verifies the exactly conserved leapfrog quadratic form

```text
H^n = 1/2 <E^n,E^n>
    + 1/2 <B^(n-1/2), B^(n+1/2)>.
```

The simultaneous-looking `sum(E^2+B^2)` is not the invariant because E and B
are stored at different times.
