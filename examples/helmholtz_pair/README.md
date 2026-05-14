# helmholtz_pair

Two coaxial circular loops separated by exactly one loop radius — the
Helmholtz configuration. The on-axis B-field near the midpoint is
quasi-uniform: both the first and second derivatives of `B_z` with respect
to `z` vanish at `z = 0`.

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.coil.cli run examples/helmholtz_pair/input.yaml
```

Writes `examples/helmholtz_pair/out.npz` with the same keys as the single-loop
example, evaluated at nine points along the axis spanning `z ∈ [-0.02, +0.02] m`
(i.e. `|z| <= 0.2 R`).

## Analytical reference

At the midpoint (`z = 0`) for `R = 0.1 m`, `I = 1 A`:

```
B_z(0) = (4/5)^(3/2) * mu0 * I / R
       = (8 / (5 * sqrt(5))) * mu0 * I / R
       ≈ 8.9918e-06 T
```

The leading deviation from the midpoint value scales as the 4th power of `z/R`:

```
B_z(z) / B_z(0) ≈ 1 - (144/125) * (z/R)^4 + O((z/R)^6)
```

Over the example's range `|z| <= 0.02 m` (i.e. `z/R <= 0.2`), that bound is
`(144/125) * 0.2^4 ≈ 1.8e-3`, so every point matches `B_z(0)` to better than
0.2%. Push the line wider (e.g. `[-0.04, +0.04]`) and you can watch the
deviation grow to ~3% at the endpoints.

The integration test `tests/python/test_examples.py::HelmholtzPairExampleTest`
checks both the midpoint value and the near-uniform variation across the line.
