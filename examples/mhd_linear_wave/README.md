# mhd_linear_wave

A smooth, small-amplitude **circularly-polarized Alfvén wave** in a fully
periodic box. The CP Alfvén wave is an exact nonlinear solution of the ideal MHD
equations: it propagates along the background field without steepening, so after
an integer number of wave periods the state returns *exactly* to its initial
condition. This makes it a standard problem for measuring the spatial and
temporal convergence of an MHD scheme. The reconstruction used here is MP7,
while the complete method uses third-order SSPRK time integration.

## Run

From the repository root, with the build-tree Python package on `PYTHONPATH`:

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.mhd.cli run examples/mhd_linear_wave/input.yaml
```

The deck is in `units: normalized` (`gamma = 5/3`). Output is written next to
`input.yaml` as `out.npz`.

## What this case measures

The shipped grid is a plain `32 × 32`. The convergence test uses three x
resolutions (`16 → 32 → 64`) and eight periodic y cells; the solution is exactly
constant in y, so a square refinement would add cost without information. It
measures the L1 error against the translated analytic finite-volume average.
Under fixed-CFL refinement, `dt` decreases in proportion to the x cell width, so
the third-order SSPRK temporal error eventually dominates the seventh-order
reconstruction error. The coupled space-time rate is therefore expected to
approach three, not seven; both adjacent measured rates must exceed 2.5 so a
second-order regression cannot pass. Isolating MP7's spatial order would require
holding temporal error negligible while refining the grid.

## Reference setup

Background: `rho = 1`, `p = 0.1`, `B0 = 1` along the propagation axis `x`, so the
Alfvén speed `vA = B0 / sqrt(rho) = 1`. The transverse perturbation (amplitude
`A = 1e-3`, wavenumber `k = 2π / lx`) is circularly polarized. This amplitude
is still small, but unlike `1e-6` it keeps the three-grid convergence error above
accumulated float64 round-off:

```
dvy = -A sin(k x)   dvz = -A cos(k x)
dBy =  A sin(k x)   dBz =  A cos(k x)
```

These are the pointwise fields. Quasar stores conserved cell averages,
`By` as a y-normal face average, and `Bz` as a cell average. For this x-only
mode, every transverse stored amplitude is therefore multiplied by
`sinc(k Δx / 2) = sinc(π / nx)`. The total energy is instead spatially constant
for circular polarization, so its exact cell average retains the full
transverse contribution `A²`; it is not obtained by squaring the averaged
components.

One wave period is `T = lx / vA = 1.0`, and the example's `t_end = 1.0` is
exactly one period, so its analytic reference equals the initial state. The
integration test in `tests/python/test_examples.py` uses the same wave at
`t = 1/16` and compares against the analytically shifted face average. That
non-return phase avoids a round-off-dominated cancellation while its three-grid
rate check distinguishes SSPRK3 behavior from second order. It does not claim
to measure the isolated seventh-order spatial operator.
