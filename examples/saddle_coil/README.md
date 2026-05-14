# saddle_coil

A non-planar closed current loop in the shape of a saddle ring:

```
x(phi) = R cos(phi)
y(phi) = R sin(phi)
z(phi) = a cos(2 phi)
```

with `R = 0.10 m` and `a = 0.02 m`. The polyline rises above the xy-plane
near `phi = 0, π` ("peaks") and dips below near `phi = π/2, 3π/2"
("valleys"). It carries `I = 1 A`.

This example demonstrates two things:

1. The plain `polyline` geometry type can describe arbitrary 3D conductor
   paths — no closed-form generator required.
2. C2 symmetry constraints in the geometry translate directly into
   constraints on the magnetic field. Rotating the saddle by π about the
   z-axis leaves the loop invariant, so at the origin the field must be
   directed purely along z (the in-plane components average to zero).

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.coil.cli run examples/saddle_coil/input.yaml
```

Writes `examples/saddle_coil/out.npz` with `B_xyz` of shape `(5, 3)` —
one row per observation point.

## Regenerate the polyline at higher resolution

The 32-segment polyline above is good enough for the symmetry test, but for
visualization you may want a denser discretization. Generate it with:

```python
import math
N = 256
R, a = 0.10, 0.02
for k in range(N + 1):
    phi = 2 * math.pi * k / N
    print(f"- [{R*math.cos(phi):+.8f}, {R*math.sin(phi):+.8f}, "
          f"{a*math.cos(2*phi):+.8f}]")
```

(replace the `points_xyz_m` block in `input.yaml` with the output).

## Symmetry check

By the loop's C2 symmetry about the z-axis (φ → φ + π leaves x, y, z all
invariant), the magnetic field at the origin must satisfy `B_x = B_y = 0`.
The integration test
`tests/python/test_examples.py::SaddleCoilExampleTest` asserts this to
~1e-10 relative.
