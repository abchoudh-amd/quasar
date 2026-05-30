# magnetized_plasma

Warm electrons in a uniform out-of-plane magnetic field (1 T) supplied by the
bound `uniform` field evaluator. Particles gyrate at the cyclotron frequency. The
deck exercises the SI → internal-unit conversion together with a non-Biot-Savart
external evaluator selected by registry name.

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/magnetized_plasma/input.yaml
```

`units: SI` (converted internally via the plasma normalization). Output `out.npz`
keys:

- `external_bz` — the uniform 1 T external field on the grid (SI tesla).
- `field_bz`, `snapshot_field_bz` — self-consistent field output.
- `species_electron_*` — final particle state.

## Reference signature

The external `Bz` buffer carries the configured 1 T field. The integration test
asserts `max(|external_bz|) > 0.5` T and finite output.
