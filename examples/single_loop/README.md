# single_loop

A single circular current loop, observed at five points along its symmetry axis.

## Run

From the repository root, with the build-tree Python package on `PYTHONPATH`:

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.coil.cli run examples/single_loop/input.yaml
```

The output is written next to `input.yaml` as `out.npz`. Keys:

- `B_xyz`        — `(5, 3)` flat array of magnetic flux density in tesla.
- `B_magnitude`  — `(5,)` array, `|B|` per observation point.
- `dims`         — `(1,)` array `[5]` (number of points along the line).
- `observation_kind` — string `"line"`.

## Analytical reference

For a single circular loop of radius `R` carrying current `I` in the z = 0
plane, the on-axis B-field at height `z` is

```
B_z(z) = mu0 * I * R^2 / (2 * (R^2 + z^2)^(3/2))
```

With `R = 0.1 m`, `I = 1 A`, evaluated at `z ∈ {0, 0.05, 0.10, 0.15, 0.20}` m:

| z (m) | B_z (T)            |
|------:|--------------------|
| 0.00  | 6.2832e-06          |
| 0.05  | 4.4951e-06          |
| 0.10  | 2.2214e-06          |
| 0.15  | 9.5493e-07          |
| 0.20  | 4.5677e-07          |

The integration test `tests/python/test_examples.py::SingleLoopExampleTest`
verifies that the polygon approximation (`n_segments=256`) matches these
values to within 1e-4 relative error.
