# solenoid

A 200-turn helical solenoid, probed along its symmetry axis from outside the
solenoid into the interior and back out the other end. Demonstrates the
characteristic "tabletop" profile: roughly uniform `B_z ≈ μ₀ n I` inside the
coil, dropping to ~½ that value at each end and tailing off to zero far
outside.

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.coil.cli run examples/solenoid/input.yaml
```

Writes `examples/solenoid/out.npz` with:

- `B_xyz`        — `(29, 3)` array of B in tesla
- `B_magnitude`  — `(29,)` array of `|B|`
- `dims`         — `(1,)` array `[29]`
- `observation_kind` — `"line"`

## Geometry

- Radius `R = 0.02 m`
- Length `L = 0.50 m` (aspect ratio 25 : 1, so the surface-current
  approximation is a good reference)
- 200 turns, 24 segments per turn → 4 800 segments

`n = N / L = 400 turns/m`, current `I = 1 A`, so the ideal infinite-solenoid
asymptote is

```
mu0 n I = 4*pi*1e-7 * 400 * 1.0 ≈ 5.0265e-4 T
```

## Analytical reference

For an idealized surface-current solenoid:

```
B_z(z) = (mu0 n I / 2) * [(L/2 + z)/sqrt(R^2 + (L/2 + z)^2)
                        + (L/2 - z)/sqrt(R^2 + (L/2 - z)^2)]
```

Spot values for `R = 0.02`, `L = 0.50`, `n = 400`, `I = 1`:

| z (m)  | B_z (T)            | Notes                  |
|-------:|--------------------|------------------------|
|  0.000 | 5.01054e-04         | Midpoint (≈ μ₀nI)      |
|  0.249 | 2.63676e-04         | Just inside one end    |
|  0.250 | 2.51127e-04         | Right at the end       |
|  0.251 | 2.38577e-04         | Just outside           |
|  0.350 | 4.74110e-06         | Well outside           |

The integration test (`tests/python/test_examples.py::SolenoidExampleTest`)
verifies the midpoint within 2% and the "half-field at the end" property
within 5%; tolerances cover both the finite-aspect correction and the
discrete-helix discretization.
