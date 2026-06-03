# H⁺ / μ⁻ plasma in the poloidal cross-section of a 1 m square-toroid

End-to-end PIC demo on the **poloidal cross-section** of a square-cross-section
toroidal magnet with major radius **R₀ = 1 m** and a **0.30 m square** bore.
Unlike the sibling [`examples/square_toroid_pic`](../square_toroid_pic/) (which
simulates the equatorial *x–y* slice), this deck sets **`plane: xz`** so the 2D
grid is the lab *y = 0* meridional cut — the plane that actually shows the square
cross-section.

## Geometry mapping

The toroid's symmetry axis is **z_lab**. With `plane: xz`, the PIC external-field
sampler evaluates the magnet on the lab *y = 0* plane and maps the field into the
grid frame through a right-handed 90° rotation about lab-x:

```
PIC x  = lab x = major radius R     (radial)
PIC y  = lab z                      (cross-section vertical)
out-of-plane = lab y  ->  PIC Bz
```

The *y = 0* plane cuts the torus in a square cross-section centered at
(x, z) = (R₀, 0). The **toroidal field B_φ** (from the TF coils) points along
±lab-y, i.e. **out of the cross-section plane** — so the PIC sees it as the
out-of-plane `external_bz` that magnetizes the particles and confines them within
the cross-section. The current sheets' poloidal field lies **in-plane**
(`external_bx`, `external_by`).

### Magnet

- Four axisymmetric square-cross-section **current sheets** (top, bottom, inner
  cylinder, outer cylinder), **150 kA per sheet group** spread over 64 filaments.
- **8 discrete rectangular TF coils**, **25 kA each**, bore-hugging, at equal
  toroidal angles. These give the confining out-of-plane B_φ.

### PIC domain = 90% of the cross-section

The PIC FDTD grid covers **90% of the 0.30 m cross-section width in both
dimensions**: a **0.27 m × 0.27 m** square, 128×128 cells
(`dx = dy ≈ 2.11 × 10⁻³` m), centered on the bore at (x, z) = (1.0, 0.0) m. The
absorbing particle BC at the domain edge stands in for the magnet wall, so the
alive-count time series measures confinement loss out of the cross-section.

### Particles fill half the cross-section

Both species are seeded into a **centered 0.15 m square block** (half the
cross-section width per dimension) at the bore center, with a **1 eV** Maxwellian.
Equal density and matched positions make the patch initially quasineutral.

| species | charge (C)           | mass (kg)               |
|---------|----------------------|-------------------------|
| H⁺      | +1.602176634×10⁻¹⁹   | 1.67262192369×10⁻²⁷     |
| μ⁻      | −1.602176634×10⁻¹⁹   | 1.883531627×10⁻²⁸       |

## Computed field (field-first check)

Before running the PIC, compute the magnet field on the same plane with the coil
CLI and sanity-check confinement:

```bash
# Emit input.yaml AND field_check.yaml (same conductors, y=0 observation grid)
python examples/square_toroid_pic_1m/build_yaml.py --field-check

PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.coil.cli run examples/square_toroid_pic_1m/field_check.yaml
```

On the bore axis the out-of-plane **B_φ ≈ 0.096 T** (peak ≈ 0.58 T across the
window; total |B| peak ≈ 0.85 T) — stronger than the bare-TF estimate
`μ₀ N I / (2π R₀) ≈ 0.040 T` because the 150 kA sheets contribute too. The 1 eV
μ⁻ thermal gyroradius is then **≈ 0.4 mm**, comfortably inside the 0.15 m block,
so the species are well-confined.

> **Numerical caveat.** That 0.4 mm gyroradius is *below* the 2.1 mm cell size, so
> individual gyro-orbits are spatially under-resolved at 128². This is fine for a
> bulk confinement/loss demo but not for resolving single-particle drift orbits;
> raise `nx`/`ny` (or lower the field) if you need the orbit resolved.

## Run

```bash
# Regenerate the deck (optional — input.yaml is checked in)
python examples/square_toroid_pic_1m/build_yaml.py

# Smoke test (50 steps; finishes in seconds on an MI300X)
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/square_toroid_pic_1m/input.yaml \
    --print-config --steps-override 50

# Full ~5 µs run (~2.0M CFL steps at dt ≈ 2.49 × 10⁻¹² s)
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/square_toroid_pic_1m/input.yaml

# Render PNGs from out.npz
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.postprocess examples/square_toroid_pic_1m/out.npz
```

The CLI writes `out.npz` next to the input, with field arrays (`external_bx`,
`external_by`, `external_bz`, `field_bz`, `field_ex`, `field_ey`), per-species
particle dumps (`species_H+_{x,y,vx,vy,vz,weight,alive}`, `species_mu-_{...}`),
and metadata (`final_step`, `final_time_s`, `nx`, `ny`, `plane`). With
`plane: xz`, `external_bz` is the out-of-plane B_φ and the in-plane axes
(`external_bx`, `external_by`) are lab x and lab z respectively.
