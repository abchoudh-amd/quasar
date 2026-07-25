# weibel

Two electron populations counter-streaming along `±y`. The counter-streaming /
temperature anisotropy drives the electromagnetic Weibel instability. A common
`10^-6` sinusoidal `vy` perturbation seeds the `kx = 2π/Lx`, `ky = 0`
filamentation mode deterministically without perturbing charge density. A fixed
uniform positive background neutralizes the two periodic electron populations.
Uses 4th-order FDTD (`nghost = 2`).

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

For cold symmetric beams with drift speed `v0 = 0.3`, the seeded mode has growth
rate

```text
gamma^2 = 2 k^2 omega_p^2 v0^2 /
          (sqrt((k^2 + 2 omega_p^2)^2 + 8 k^2 omega_p^2 v0^2)
           + k^2 + 2 omega_p^2).
```

The integration test strips ghosts and periodic duplicate Yee nodes, projects
`Bz` onto that mode, checks its fitted linear growth rate against this cold
dispersion root, and requires the mode to carry more than 99% of the final
magnetic energy.
