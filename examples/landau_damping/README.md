# landau_damping

Warm Maxwellian electrons with a small seeded `Ex` perturbation. Collisionless
Landau damping removes energy from the longitudinal mode (the field decays rather
than grows as in `two_stream`).

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/landau_damping/input.yaml
```

`units: normalized`. Output `out.npz` keys:

- `snapshot_field_ex`  — Ex snapshots (cadence 8).
- `field_ex`           — final Ex.
- `species_electron_*` — final particle state.

## Reference signature

The seeded single-mode `Ex` energy decays over the run (Landau damping). The
integration test runs a short trajectory and checks the deck loads, seeds, and
steps to finite output.
