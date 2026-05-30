# coil_confinement

Warm electrons in the magnetic field of a circular current loop (1 kA), sampled
via the Biot-Savart external-field evaluator. Demonstrates seeding `B_ext` for a
PIC run from a coil geometry, with the SI → internal-unit conversion applied to
both the grid and the sampled field.

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/coil_confinement/input.yaml
```

`units: SI`. Output `out.npz` keys:

- `external_bz` — the loop's `Bz` sampled on the grid (SI tesla).
- `field_bz`, `snapshot_field_bz` — self-consistent field output.
- `species_electron_*` — final particle state.

## Reference signature

The external `Bz` buffer carries the loop's non-trivial field. The integration
test asserts `max(|external_bz|) > 1e-4` T and finite output.
