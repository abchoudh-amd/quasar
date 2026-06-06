# mhd_blast

A **magnetized MHD blast wave** — the standard strong-shock / positivity stress
test for an ideal-MHD code. A small, highly over-pressured circular core sits in
a low-pressure, strongly magnetized ambient medium. The core drives a strong
fast-magnetosonic shell that expands **anisotropically**: the shock runs faster
along the threaded magnetic field than across it, producing the characteristic
elongated shell. The rapidly-evacuated core makes this a hard test for the
positivity limiter and the density/pressure floors.

## Run

From the repository root, with the build-tree Python package on `PYTHONPATH`:

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.mhd.cli run examples/mhd_blast/input.yaml
```

The deck is in `units: normalized` (`gamma = 5/3`). Output is written next to
`input.yaml` as `out.npz`.

## Canonical setup

On `[-0.5, 0.5] × [-0.5, 0.5]` with `gamma = 5/3`:

```
ambient:           rho = 1,   p = 0.1
core (r < 0.1):    p = 10                (over-pressured by 100x)
threaded field:    B = (1/sqrt(2), 1/sqrt(2), 0),  |B| = 1
```

## Reference / validation

Run to `t = 0.2`. The expected outcome is a fast-magnetosonic shell that is
**near-circular but elongated along the diagonal field direction**, with the
interior swept clean. The decisive check is **robustness**: density and pressure
must stay strictly positive everywhere (`rho ≥ rho_floor`, `p ≥ p_floor`)
despite the strong rarefaction, and the `div B` monitor must stay controlled.

The integration test in `tests/python/test_examples.py` runs this deck and
asserts that the run completes with `min(rho) > 0` and `min(p) > 0` (no floor
violations / NaNs) and that the shell has expanded outward from the core.
