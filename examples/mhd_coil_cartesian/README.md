# mhd_coil_cartesian — inline Helmholtz-pair background

This example exercises the Cartesian coil-seeded MHD path. Two coaxial loops
form the same Helmholtz geometry as `examples/helmholtz_pair`, scaled to
`1000 A`, and a small confined plasma blob evolves in the central bore.

The run uses the lab `Y = 0` meridional cut:

| MHD quantity | Lab quantity | Meaning |
|---|---|---|
| `x`, `bx` | `X`, `B_X` | transverse bore coordinate |
| `y`, `by` | `Z`, `B_Z` | coil axis |
| `bz` | `B_Y` | out-of-plane component |

The loops are centered at `Z = ±R/2` with radius `R = 0.1 m`. Their axial
fields reinforce in the `0.04 m × 0.04 m` domain around the midpoint, where the
field is nearly uniform. At the origin the ideal-filament reference is

```text
B_Z(0) = (4/5)^(3/2) mu0 I / R ≈ 8.9918e-3 T.
```

## How the background is built

The loop geometry lives directly under `background_field.conductors`. After
the MP7 solver selects its four-cell halo, the loader derives the corresponding
padded corner grid, evaluates the Biot–Savart vector potential there, and takes
the Cartesian discrete curl of lab-`Y` `A`. The resulting face field is
discretely divergence-free by construction and remains fixed as `B0`; the MHD
state stores and evolves only the perturbation `b`.

The domain is an open computational crop inside the coils, so both the fluid
and perturbation-field boundaries use `outflow`. The conductors lie well outside
the padded sampling region.

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.mhd.cli run examples/mhd_coil_cartesian/input.yaml
```

The five-step run writes `examples/mhd_coil_cartesian/out.npz`. The stored
`state_bx`, `state_by`, and `state_bz` arrays are the evolved perturbation, not
the static Helmholtz field; add the background field when reconstructing the
physical total field.
