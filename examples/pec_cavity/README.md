# pec_cavity

A square PEC cavity (conducting on all four field walls) containing the
two-dimensional `TM_11` eigenmode and no particles. The seed uses the exact
fourth-order Yee spatial eigenvector: `Ez` is cell-centred at integer time and
`Bx`/`By` are placed on their own face lattices at `t=-dt/2`. Their amplitudes
come from the leapfrog discrete frequency, so the run begins as a single cavity
mode rather than an incompatible travelling wave. The deck uses fourth-order
FDTD (`nghost = 2`).

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/pec_cavity/input.yaml
```

`units: normalized`. Output `out.npz` keys:

- `snapshot_field_ez`, `snapshot_field_bx`, `snapshot_field_by` — padded field
  snapshots (cadence 8); component-aware readers must strip ghosts while
  retaining the independent non-periodic high faces of `Bx` and `By`.
- `field_ez`, `field_bx`, `field_by` — final fields.

## Reference signature

For side lengths `Lx=Ly=1`, the continuum limit is

```text
Ez = sin(pi*x/Lx) sin(pi*y/Ly) cos(omega*t),
omega -> pi*sqrt(1/Lx^2 + 1/Ly^2).
```

The finite-grid reference uses the fourth-order staggered symbols

```text
kx~ = [9/4 sin(pi/(2 nx)) - 1/12 sin(3 pi/(2 nx))] / dx,
ky~ = [9/4 sin(pi/(2 ny)) - 1/12 sin(3 pi/(2 ny))] / dy,
omega*dt = 2 asin(dt*sqrt(kx~^2 + ky~^2)/2).
```

The integration test projects `Ez` onto `TM_11` and checks that frequency,
checks the matching fourth-order nodal `div(B)`, verifies odd PEC continuation
of tangential `Ez`, and evaluates the conserved leapfrog Yee energy

```text
H^n = 1/2 ||E^n||_V^2
    + 1/2 <B^(n-1/2), B^(n+1/2)>_V,
```

with each component's own dual control-volume weights. Ghost cells and
non-physical duplicate endpoints are never included.
