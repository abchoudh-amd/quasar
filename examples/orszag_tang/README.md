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

The canonical late-time comparison is usually shown at `t = 0.5`. The shipped
deck deliberately stops earlier, at `t ≈ 0.19`, before its `128²` grid
under-resolves the strongest current sheet. At that output time the smooth
vortices have begun forming shocks and current sheets, while density and pressure
remain positive. The principal quantitative checks are **conservation**: total
mass `sum(rho)` and total energy `sum(energy)` should be conserved to scheme
tolerance over this resolved window, and the `div B` monitor should stay near
the floor.

The integration test in `tests/python/test_examples.py` runs this deck and
asserts conservation of total mass and total energy (relative drift within a
small tolerance) and that `div B` remains controlled. If the diagnostics writer
emits `mass_initial` / `energy_initial` keys, the test compares against those;
otherwise it derives the initial totals from the analytic IC.

## A note on `cfl`, `steps`, and the integration window

This deck uses `cfl = 0.1` for accuracy and stability headroom as the current
sheets sharpen, and runs for `steps = 1000` (reaching `t ~= 0.19`) rather than
the canonical `t = 0.5`. MP5/MP7 characteristic reconstruction alone has no
positivity guarantee, but the complete solver preserves the mathematical
admissible set conservatively. An inadmissible SSP-RK candidate is discarded,
the state is rolled back exactly, and the interval is retried with smaller
substeps and a piecewise-constant HLL anchor as needed. Evolution does not clamp
individual cells or re-derive energy from a pressure floor.

The early stop therefore defines the validation window; it is not a workaround
for non-conservative floor repair. In this resolved window, total mass and energy
are conserved to round-off and `div B` stays at machine epsilon. (`div B` here
is checked as an absolute L-inf; it scales with `|B|` and accumulates telescoping
round-off with step count, but stays well under the test tolerance.) A canonical
`t = 0.5` morphology comparison should use a longer run and demonstrate
resolution convergence; that late-time comparison is outside this shipped
regression's scope.
