# pec_cavity

A closed PEC cavity (reflecting on all four field walls) seeded with an `Ez`/`By`
pulse and no particles. With lossless PEC walls the total EM energy is conserved
as the pulse bounces around the cavity. Uses 4th-order FDTD (`nghost = 2`).

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/pec_cavity/input.yaml
```

`units: normalized`. Output `out.npz` keys:

- `snapshot_field_ez`, `snapshot_field_by` — field snapshots (cadence 8).
- `field_ez`, `field_by` — final fields.

## Reference signature

The total EM energy `sum(Ez^2 + By^2)` stays bounded (no blow-up) — the PEC walls
are lossless and stable. The integration test asserts
`energy.max() < 2 * energy.min()`.
