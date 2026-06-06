# mhd_guide_field

A **guide-field** (static field-split) demonstration. A small-amplitude,
circularly-polarized Alfvén wave is evolved on top of a **static uniform guide
field** `B0`, using the field-split form **B = B0 + b**: the solver carries `B0`
as a fixed background and evolves only the perturbation `b`. The total field
`B0 + b` is what enters the fluxes, the fast-magnetosonic (CFL) speed, and the
Alfvén dynamics — `B0` itself never changes in time.

This is the same CP Alfvén eigenmode as `mhd_linear_wave`, but with the
background field supplied through the `background_field:` block rather than baked
into the initial condition, so it exercises the static field-split path
end-to-end (config → solver background seed → CFL probe → step).

## Run

From the repository root, with the build-tree Python package on `PYTHONPATH`:

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.mhd.cli run examples/mhd_guide_field/input.yaml
```

The deck is in `units: normalized` (`gamma = 5/3`). Output is written next to
`input.yaml` as `out.npz`.

## How the field split is set up

The case keeps the two contributions cleanly separated:

- The Alfvén-wave initial condition seeds **only the transverse perturbation**
  (`dvy`, `dvz`, `dBy`, `dBz`) with its own in-plane background turned off
  (`initial.params.b0: 0.0`). The evolved (perturbation) state therefore has
  **zero mean in-plane field**.
- The `background_field:` block supplies the **uniform guide field**
  `B0 = (bx0, 0, 0)` along the propagation axis `x`. With `bx0 = 1` and
  `rho = 1` the Alfvén speed is `vA = bx0 / sqrt(rho) = 1`.

The total field the solver feels at every cell is `B0 + b`.

## Reference behavior

- **Field split.** `B0` is uniform (and stays constant); only `b` is evolved.
  The `state_*` keys in `out.npz` store the **perturbation** state `b`, exactly
  as the no-background runs do.
- **Divergence.** A uniform `B0` is exactly divergence-free, and the seeded
  perturbation is solenoidal, so `div(B0 + b) = div(b)` stays at round-off
  throughout. The `divb_linf` series remains at machine-epsilon scale.
- **CFL.** The fast-magnetosonic speed uses the **total** field, so the guide
  field `B0` raises the fast speed and the CFL-limited `dt` is **tighter** than
  the `B0 = 0` case at the same `cfl`. The CFL probe is taken *after* `B0` is
  seeded, so the timestep already accounts for the guide field.

## Expected qualitative outcome

The run is **stable and finite**: density stays positive, every conserved
component remains finite, and `|div B|inf` stays at round-off. The integration
test in `tests/python/test_examples.py` checks finiteness, positive density, and
a small `divb_linf` for the evolved state.
