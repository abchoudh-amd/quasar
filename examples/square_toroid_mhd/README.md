# square_toroid_mhd — coil-seeded ideal-MHD in a square-toroid bore

An axisymmetric ideal-MHD run on the **poloidal cross-section** of a 1 m
square-toroid magnet
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
It is an annular cylindrical domain (`geometry: cylindrical`, `R_min=0.865 m`),
so neither radial boundary is a coordinate-axis boundary. This 90% box is an
artificial computational crop through the vacuum field, not a material wall.
The coil field has a nonzero normal component at parts of its boundary. The
deck therefore uses `outflow` for both fluid and perturbation-field boundaries;
a conducting-wall field parity would incorrectly require zero normal field,
while mixing fluid walls with field outflow would give the coupled MHD fluxes
incompatible physical boundary models.

## Why field-split (and not a coil-field initial condition)

The non-uniform coil field extends through the padded domain and cannot be
represented faithfully by applying a generic state boundary operator to the
last interior value. If it were seeded into the evolving state, ghost filling
would replace that known exterior structure and generally inject discrete
divergence at the boundary ring.

The fix is the solver's **field-split** form `B = B0 + b`:

- `B0` is the **static** coil field, constructed directly throughout the padded
  grid. It is never ghost-refilled, so its exterior stencil values are retained.
- `b` is the evolving plasma perturbation. It **starts at zero**, so the CT
  `div(b) = 0` holds at the seed and the mimetic CT update preserves it. The
  open field boundary acts only on this perturbation, consistently with the
  open fluid boundary.

This is the same mechanism as `examples/mhd_guide_field`, generalized from a
uniform guide field to a **non-uniform** coil field.

### Conservation note

The Riemann flux uses the total field, so the full Maxwell stress and magnetic
energy flux are present. After constrained transport supplies the authoritative
`db/dt`, the solver transforms the total-field energy rate to the stored split
energy with `dE'/dt = dE_total/dt - B0·db/dt`. This discrete change of variables
supports any static, discretely divergence-free background; the optional vacuum
projection used here prepares a particularly clean coil field but is not a
restriction of the split equations.

## Divergence-free coil background

1. `coil.yaml` evaluates the magnetic vector potential `A` (with `B = curl A`,
   provided by `BiotSavartEvaluator::evaluate_A`) on the lab `Y = 0` plane at the
   cell-corner grid of the **full padded** MHD domain (interior + `nghost`
   ghosts), writing `A_xyz_grid` (shape `(Ny+1, 1, Nx+1, 3)`,
   `Nx = nx + 2·nghost`).
2. With `params.vacuum_project: true`, the loader holds `A_phi` fixed on the
   outer boundary of the padded grid and solves the discrete annular vacuum
   equation for `psi = R A_phi` at every interior corner. This turns the sampled
   continuum vacuum solution into a field that is curl-free under the solver's
   own staggered differences; failure to converge is a hard input error. The
   full padded radial interval must lie at `R > 0`.
3. The loader applies the axisymmetric annular curl to the projected corner
   values, producing the face-staggered background:

   ```
   b0x_face(i,j) = -(A[j+1,i] - A[j,i]) / dZ
   b0y_face(i,j) = (R_hi A[j,i+1] - R_lo A[j,i])
                    / (0.5 (R_hi^2 - R_lo^2))
   ```

   The discrete-curl construction makes the ring-weighted divergence telescope,
   including in the ghost layers read by reconstruction, while the optional
   projection prepares a discretely vacuum field to its solve target. Without
   the projection flag, the loader differences the supplied potential directly.
   The resulting current-carrying background is valid when it passes the
   solenoidal acceptance threshold; the split equations do not require a vacuum
   field.

The vector-potential path is validated against `curl A = B` in
`tests/unit/physics/magnetostatics/test_vector_potential.cpp`.

## Plasma

A confined plasma blob (`initial.type: confined_blob`): denser, higher-pressure
plasma (`rho_in`, `p_in`) inside a centered square of half-width `blob_half`,
ambient (`rho_out`, `p_out`) outside, initially at rest with zero in-plane
perturbation. It evolves on the field-split coil background plus the uniform
toroidal guide field. The deck uses SI throughout: density is kg/m³, pressure
and energy density are Pa, velocity is m/s, and magnetic input/output is tesla.

The shipped pressures are `p_in = 1000 Pa` and `p_out = 100 Pa`.  The projected
coil reaches about 0.6 T (magnetic pressure of order 10⁵ Pa), so this remains a
strongly magnetized, low-beta case.  The earlier 1/0.1 Pa values put a
discontinuous blob at beta of order 10⁻⁶—below the truncation scale
of the supported cylindrical MUSCL grid—and its discrete evolution exhausted
positive internal energy at a blob corner.  The current values are the lowest
tested decade with a comfortable positive-pressure margin through all 400
steps; they do not weaken the magnet or relax the conservative positivity
check.

## Magnet

`coil.yaml` contains only the axisymmetric square-cross-section current sheets
(top/bottom at `Z = ±a/2`, inner/outer cylinders at `R = R0 ± a/2`, `150 kA`
per sheet group over 64 filaments), which generate the poloidal background.
Eight conceptual TF windings at `25 kA` give
`B_phi ~ mu0 N I / (2 pi R) ~ 0.04 T`; their toroidal guide field is represented
directly by the evolved `initial.bz` instead of incorrectly treating a discrete
3-D TF-coil `A_y` slice as axisymmetric `A_phi`.

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

The cylindrical run uses `muscl_minmod` and is second-order in space. MP5/MP7
currently use Cartesian finite-volume moments and are rejected for radial
ring-volume averages until radius-weighted high-order moments are implemented.
The coil grid spans the **padded** domain, so it depends on the reconstruction
halo: MUSCL needs `nghost = 2`. `build_yaml.py`'s `NGHOST` constant must stay in
sync with `numerics.reconstruction`.

## Output

`out.npz` holds the final interior **perturbation** state (`state_rho`,
`state_bx`, `state_by`, `state_bz`, …; the in-plane `state_bx/by` are the
evolving `b`, not the total field), the seeded `state_*_initial` snapshot, and the
`divb_linf` series. Magnetic state fields and the divergence diagnostic are
converted back from the solver's internal `B/sqrt(mu0)` normalization to tesla
and tesla/meter, respectively.

- `divb_linf[0]` (the seed) is at round-off: `b = 0` is trivially solenoidal.
- Subsequent `divb_linf` samples should remain at round-off because the annular
  CT curl and ring divergence form a mimetic `div(curl)=0` pair.
