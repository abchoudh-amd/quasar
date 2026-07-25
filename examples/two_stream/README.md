# two_stream

Two counter-streaming cold electron beams (drift `±0.2 c`) on a periodic grid.
The equilibrium charge is cancelled by a fixed uniform background. Both beams
receive the same small velocity perturbation,
`delta vx = 1e-6 sin(2 pi x/L)`, which creates a deterministic resolved current
mode while leaving the initial charge density uniform. The longitudinal field
then follows the canonical electrostatic two-stream instability.

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

## Linear reference

For equal cold beams with individual normalized plasma frequency `omega_pb=1`,
the unstable root satisfies

```
gamma^2 = omega_pb sqrt(omega_pb^2 + 4 (k v0)^2)
          - omega_pb^2 - (k v0)^2 .
```

Here `k=2 pi/L` and `v0=0.2`, giving `gamma ~= 0.355`. The integration test fits
the Fourier amplitude of that exact mode during its linear window and requires
agreement with the cold dispersion relation, in addition to substantial net
growth. This avoids relying on random macro-particle noise or a final/initial
energy ratio that mixes the initial transient with nonlinear saturation.
