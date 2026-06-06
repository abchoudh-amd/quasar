# orszag_tang

The **Orszag-Tang vortex** (Orszag & Tang, *J. Fluid Mech.* **90**, 129 (1979)) —
the canonical 2D ideal-MHD problem for the transition to MHD turbulence. A
smooth initial vortex in a doubly-periodic unit box steepens into interacting
shocks and thin current sheets. It is the standard 2D integration test for an
MHD code: it stresses the Riemann solver, the constrained-transport `div B`
control, and the positivity limiter all at once, while conserving total mass and
energy.

## Run

From the repository root, with the build-tree Python package on `PYTHONPATH`:

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.mhd.cli run examples/orszag_tang/input.yaml
```

The deck is in `units: normalized` (`gamma = 5/3`). Output is written next to
`input.yaml` as `out.npz`.

## Canonical initial condition

On `[0,1] × [0,1]` with `gamma = 5/3`:

```
rho = gamma^2                      p = gamma
v   = (-sin(2*pi*y),  sin(2*pi*x),  0)
B   = (-sin(2*pi*y),  sin(4*pi*x),  0) / sqrt(4*pi)
```

i.e. uniform density `rho = gamma^2 = 25/9` and pressure `p = gamma = 5/3`, with
the standard magnetic normalization `B0 = 1/sqrt(4*pi)`.

## Reference / validation

Run to the canonical time `t = 0.5`. The expected outcome is the well-known
two-vortex / central current-sheet structure with a smooth density field free of
spurious oscillations (a working positivity limiter keeps `rho, p > 0`
everywhere). The principal quantitative checks are **conservation**: total mass
`sum(rho)` and total energy `sum(energy)` should be conserved to scheme
tolerance over the run, and the `div B` monitor should stay near the floor.

The integration test in `tests/python/test_examples.py` runs this deck and
asserts conservation of total mass and total energy (relative drift within a
small tolerance) and that `div B` remains controlled. If the diagnostics writer
emits `mass_initial` / `energy_initial` keys, the test compares against those;
otherwise it derives the initial totals from the analytic IC.
