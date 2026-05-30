# weibel

Two electron populations counter-streaming along `±y`. The counter-streaming /
temperature anisotropy drives the electromagnetic Weibel instability, which grows
a transverse magnetic field (`Bz`) from noise. Uses 4th-order FDTD (`nghost = 2`).

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/weibel/input.yaml
```

`units: normalized`. Output `out.npz` keys:

- `snapshot_field_bz`  — Bz snapshots (cadence 16).
- `field_bz`           — final Bz.
- `species_electron_up_*`, `species_electron_down_*` — final particle state.

## Reference signature

The transverse magnetic energy `sum(Bz^2)` grows over the run. The integration
test asserts `energy[-1] > energy[0]`.
