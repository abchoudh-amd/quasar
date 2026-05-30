# beam_in_channel

A drifting electron beam confined to a channel: periodic along `x`, reflecting
(PEC field + specular particle) walls in `y`. Exercises mixed dict-form
boundaries with separate `field` and `particle` side-maps. The beam streams along
`x` and bounces off the `y` walls, staying confined (no particle loss).

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/beam_in_channel/input.yaml
```

`units: normalized`. Output `out.npz` keys:

- `field_bz`, `snapshot_field_bz` — field output (cadence 8).
- `species_electron_beam_*` — final particle state (incl. `_alive`).

## Reference signature

The specular `y` walls keep every particle alive (the beam is confined). The
integration test asserts the alive count equals the particle count.
