# square_toroid_mhd — coil-seeded ideal-MHD in a square-toroid bore

An ideal-MHD run on the **poloidal cross-section** of a 1 m square-toroid magnet
(major radius `R0 = 1.0 m`, `0.30 m` square bore). The confining **poloidal**
magnetic field is computed from a **Biot–Savart coil calculation** and carried as
a static, divergence-free **field-split background** `B0`; a confined plasma blob
evolves on it, threaded by a uniform toroidal guide field `bz = 0.1`.

## Coordinate mapping

This is the lab `Y = 0` meridional cut (same slice as
`examples/square_toroid_pic_1m`):

| MHD axis          | Lab axis | Physical meaning                  |
|-------------------|----------|-----------------------------------|
| `x` (`bx`=B_R)    | `X`      | major radius `R`                  |
| `y` (`by`=B_z)    | `Z`      | cross-section vertical            |
| out-of-plane `bz` | `Y`      | toroidal direction (guide field)  |

The MHD domain is the bore interior: 90% of the `0.30 m` cross-section
(`0.27 m` square) centered on the right-side bore at `(R, Z) = (1.0, 0.0)`.

## Why field-split (and not a coil-field initial condition)

The constrained-transport (CT) scheme freezes the *discrete* `div B` at its
seeded value, and **no open boundary condition can reproduce the exterior coil
field**. If the coil field is seeded into the evolving *state*, the boundary
ghost-fill (outflow/wall/periodic) corrupts the non-uniform field at the boundary
ring, the discrete `div B` there becomes large, and the error floods inward — the
run is physically meaningless. (A *uniform* field, like the rotor example's, is
the only case an open BC reproduces exactly.)

The fix is the solver's **field-split** form `B = B0 + b`:

- `B0` is the **static** coil field. It is never ghost-refilled, so no BC can
  corrupt it.
- `b` is the evolving plasma perturbation. It **starts at zero**, so the CT
  `div(b) = 0` holds at the seed and is preserved exactly forever.

This is the same mechanism as `examples/mhd_guide_field`, generalized from a
uniform guide field to a **non-uniform, curl-free** coil field.

### Conservation note

The field-split conservative bookkeeping is exact only for a `B0` with no net
Maxwell self-force. A **curl-free vacuum coil field** satisfies exactly that:
`div(B0 B0 − ½|B0|² I) = (∇×B0) × B0 = 0`, so the missing static momentum source
is identically zero and the split stays conservative to truncation order. This is
why `background_field.a_file` (the coil mode) is the one **non-uniform** `B0` the
deck loader accepts; an arbitrary non-uniform `B0` is still rejected.

## Divergence-free coil background

1. `coil.yaml` evaluates the magnetic vector potential `A` (with `B = curl A`,
   provided by `BiotSavartEvaluator::evaluate_A`) on the lab `Y = 0` plane at the
   cell-corner grid of the **full padded** MHD domain (interior + `nghost`
   ghosts), writing `A_xyz_grid` (shape `(Ny+1, 1, Nx+1, 3)`,
   `Nx = nx + 2·nghost`).
2. `background_field.a_file` differences the corner-node lab-`Y` component `A`
   into the face-staggered background:

   ```
   b0x_face(i,j) = -(A[j+1,i] - A[j,i]) / dy      # B_R on the left face
   b0y_face(i,j) =  (A[j,i+1] - A[j,i]) / dx      # B_z on the bottom face
   ```

   Because `A` lives at the corners over the padded grid, the cell-centered
   discrete divergence telescopes to **identically zero** everywhere, including
   the ghost layers the reconstruction stencil reads.

The vector-potential path is validated against `curl A = B` in
`tests/unit/physics/magnetostatics/test_vector_potential.cpp`.

## Plasma

A confined plasma blob (`initial.type: confined_blob`): denser, higher-pressure
plasma (`rho_in`, `p_in`) inside a centered square of half-width `blob_half`,
ambient (`rho_out`, `p_out`) outside, initially at rest with zero in-plane
perturbation. It evolves on the field-split coil background plus the uniform
toroidal guide field.

## Magnet

Axisymmetric square-cross-section current sheets (top/bottom at `Z = ±a/2`, inner/
outer cylinders at `R = R0 ± a/2`, `150 kA` per sheet group over 64 filaments)
plus 8 discrete bore-hugging rectangular TF coils (`25 kA` each). The TF coils
give a toroidal field `B_phi ~ mu0 N I / (2 pi R) ~ 0.04 T` at `R = R0`.

## Running

The MHD background reads the coil output, so run the coil deck **first**:

```bash
# (re)generate both decks
python examples/square_toroid_mhd/build_yaml.py

# 1. coil vector potential -> examples/square_toroid_mhd/coil.npz
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.coil.cli run examples/square_toroid_mhd/coil.yaml

# 2. MHD run -> examples/square_toroid_mhd/out.npz
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.mhd.cli run examples/square_toroid_mhd/input.yaml --log-every 50
```

The coil grid spans the **padded** domain, so it depends on the reconstruction
halo (`nghost`): `mp7` needs `nghost = 4`. `build_yaml.py`'s `NGHOST` constant
must stay in sync with `numerics.reconstruction`.

## Output

`out.npz` holds the final interior **perturbation** state (`state_rho`,
`state_bx`, `state_by`, `state_bz`, …; the in-plane `state_bx/by` are the
evolving `b`, not the total field), the seeded `state_*_initial` snapshot, and the
`divb_linf` series.

- `divb_linf[0]` (the seed) is at round-off: `b = 0` is trivially solenoidal.
- The reported `divb_linf` **grows** over the run, but this is a boundary-ring
  measurement artifact: `divergence_b_max` ghost-fills with the open BC before
  measuring, so the perturbation against the static `B0` edge shows a nonzero
  `div B` in the boundary ring only. The **strict interior** `div(b)` stays at
  round-off (the CT update preserves it); the artifact does not propagate inward.
