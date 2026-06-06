# brio_wu

The Brio-Wu MHD shock tube (Brio & Wu, *J. Comput. Phys.* **75**, 400 (1988)) —
the canonical 1D Riemann problem for ideal magnetohydrodynamics. A discontinuity
at `x = 0.5` separates a left and a right constant state; with `Bx ≠ 0` held
constant across the tube and `By` changing sign, the solution develops the
distinctive **compound wave** that makes this the standard hard test for MHD
Riemann solvers. The case is run here on a thin, y-periodic 2D grid so the 2D
mp7 / HLLD / constrained-transport solver exercises a 1D problem.

## Run

From the repository root, with the build-tree Python package on `PYTHONPATH`:

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.mhd.cli run examples/brio_wu/input.yaml
```

The deck is in `units: normalized` (`gamma = 2`). Output is written next to
`input.yaml` as `out.npz` with the conserved fields `rho, mx, my, mz, energy`
and the magnetic field `bx, by, bz`, plus the `divb` monitor.

## Canonical reference

Standard initial states (Brio & Wu 1988):

| state | rho   | p   | vx | vy | vz | Bx   | By  | Bz |
|-------|-------|-----|----|----|----|------|-----|----|
| left  | 1.000 | 1.0 | 0  | 0  | 0  | 0.75 |  1  | 0  |
| right | 0.125 | 0.1 | 0  | 0  | 0  | 0.75 | -1  | 0  |

At the standard output time `t = 0.1`, the solution (left to right) is a fast
rarefaction, a compound wave (a slow shock fused to a slow rarefaction), a
contact discontinuity, a slow shock, and a right-moving fast rarefaction. The
density plateaus near `rho ≈ 1` on the far left and `rho ≈ 0.125` on the far
right, with intermediate plateaus set by the wave structure. `Bx` stays at
`0.75` everywhere (`div B = 0` along the tube), and the solution is symmetric
about the contact in the usual Brio-Wu sense.

The integration test in `tests/python/test_examples.py` runs this deck and
checks the far-field densities against the left/right input states and verifies
the `div B` monitor stays at the floor.
