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

The grid is a plain `32 × 32`; a convergence test doubles `nx`/`ny`
programmatically (32 → 64 → 128 → …) and measures the L1 error of the evolved
state against the initial condition. Because the analytic reference *is* the
unchanged initial profile (the wave is periodic in time), the error is purely
the scheme's truncation error. Under fixed-CFL refinement, `dt` decreases in
proportion to the cell width, so the third-order SSPRK temporal error eventually
dominates the seventh-order reconstruction error. The coupled space-time rate is
therefore expected to approach three, not seven; isolating MP7's spatial order
would require holding temporal error negligible while refining the grid.

## Reference setup

Background: `rho = 1`, `p = 0.1`, `B0 = 1` along the propagation axis `x`, so the
Alfvén speed `vA = B0 / sqrt(rho) = 1`. The transverse perturbation (amplitude
`A = 1e-6`, wavenumber `k = 2π / lx`) is circularly polarized:

```
dvy = -A sin(k x)   dvz = -A cos(k x)
dBy =  A sin(k x)   dBz =  A cos(k x)
```

One wave period is `T = lx / vA = 1.0`, and `t_end = 1.0` is exactly one period,
so the analytic reference equals the initial state. The integration test in
`tests/python/test_examples.py` runs the deck at two or more resolutions and
checks monotone super-first-order convergence against that exact solution. It
does not claim to measure the isolated seventh-order spatial operator.
