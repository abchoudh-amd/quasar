# beam_in_channel

A drifting electron beam confined to a channel: periodic along `x`, reflecting
(PEC field + specular particle) walls in `y`. Exercises mixed dict-form
boundaries with separate `field` and `particle` side-maps. The beam streams along
`x` and bounces off the `y` walls, staying confined (no particle loss). A cold,
heavy positive species is loaded on the same deterministic quiet-start positions
as the electrons. Equal densities, counts, and opposite charges cancel the
deposited charge cell-by-cell at startup, so the zero self-electric-field seed
satisfies the discrete Gauss constraint to roundoff. The ions remain nearly
stationary on this short beam-transit timescale.

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/beam_in_channel/input.yaml
```

`units: normalized`. Output `out.npz` keys:

- `field_bz`, `snapshot_field_bz` — field output (cadence 8).
- `species_electron_beam_*`, `species_ion_background_*` — final particle states
  (including `_alive`).

## Reference signature

The paired quiet start is initially charge-neutral on the mesh, and the specular
`y` walls keep every particle alive. The integration test asserts the equal and
opposite load and checks both alive counts.
