# em_wave_propagation

A vacuum plane wave seeded in `Ez`/`By`, propagating in `+x` on a fully periodic
grid with no particles. Pure 4th-order FDTD field solve; the total EM energy stays
bounded as the wave translates.

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/em_wave_propagation/input.yaml
```

`units: normalized`. Output `out.npz` keys:

- `snapshot_field_ez`, `snapshot_field_by` — field snapshots (cadence 8).
- `field_ez`, `field_by` — final fields.

## Reference signature

The total EM energy `sum(Ez^2 + By^2)` is bounded (no numerical blow-up); it
oscillates mildly with the leapfrog half-step staggering. The integration test
asserts `energy.max() < 2 * energy.min()`.
