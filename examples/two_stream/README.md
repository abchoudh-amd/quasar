# two_stream

Two counter-streaming cold electron beams (drift `±0.2 c`) on a periodic grid.
The canonical electrostatic two-stream instability: the longitudinal
electric-field energy grows exponentially before saturating.

## Run

From the repository root, with the build-tree Python package on `PYTHONPATH`:

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/two_stream/input.yaml
```

The deck is in `units: normalized` (`c = eps0 = mu0 = 1`); velocities are in units
of `c`. Output is written next to `input.yaml` as `out.npz`. Keys:

- `snapshot_field_ex`  — `(nsnap, storage)` Ex snapshots (cadence 8).
- `snapshot_steps`     — step index of each snapshot.
- `field_ex`           — final Ex on the ghost-padded grid.
- `species_*_x/y/vx/…` — final per-species particle state.

## Reference signature

The longitudinal field energy `sum(Ex^2)` grows by orders of magnitude over the
run. The integration test
`tests/python/test_examples.py::PicAspirationalExampleTests` asserts
`energy[-1] > 50 * energy[0]`.
