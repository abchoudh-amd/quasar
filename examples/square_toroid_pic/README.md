# H⁺ / μ⁻ plasma in a square-toroid magnet

End-to-end PIC demo: seed a quasineutral H⁺ + μ⁻ plasma in the equatorial
slice of the square-cross-section toroidal magnet from
[`examples/square_toroid`](../square_toroid/), then step the
electromagnetic PIC solver under that external B-field.

## Geometry mapping

The toroid's symmetry axis is **z_lab**. The PIC sampler always evaluates the
external field at z=0 (`src/physics/pic/external_field_sampler.cpp`), so the
PIC plane (x_pic, y_pic) maps to the lab equatorial slice (x_L, y_L, z_L=0).
The donut bore is an annular ring centered on the lab z-axis (inner radius
`R0 − a/2`, outer radius `R0 + a/2`). The external Bz seen by PIC is the
axial (out-of-plane) component of the toroid's field — which is what drives
in-plane cyclotron rotation of the species.

For a tokamak-style toroidal-field profile (1/R falloff), the equatorial
slice cuts across the bore: Bz is strongest in a band along the bore
midline and reverses sign across the bore boundaries.

### TF coils (toroidal field)

In addition to the four axisymmetric current sheets (which produce a purely
poloidal field with B_φ = 0), the deck emits **16 discrete rectangular TF
coils** at equal toroidal angles `θ_k = 2π k / 16`, k = 0..15. Each coil
is a closed polyline in its own poloidal (R, z) plane, hugging the bore
with corners at `(R, z) ∈ {R0±a/2} × {±a/2}`, carrying `I_TF = 1000 A`
with right-hand circulation about `+φ̂`.

The resulting toroidal field on the bore axis is

```
B_φ(R) ≈ μ₀ N I_TF / (2π R) ≈ 0.032 T at R = R₀ = 0.10 m.
```

In the equatorial PIC slice, B_φ projects into the in-plane components
**B_x and B_y** (perpendicular to the local R̂ at every grid point), so
the PIC now sees a fully 3D external field — out-of-plane B_z from the
sheets plus an in-plane azimuthal contribution from the TF coils. Cross-
field drifts in the (x, y) plane become non-trivial. 16 coils gives a
small toroidal ripple in |B_φ|; for a smoother ring raise `N_TF_COILS`
in `build_yaml.py`.

### PIC domain = bore cross-section

The PIC domain is sized to **the right-side bore cross-section only**:
`a × a` = `0.04 × 0.04` m centered on `(x, y) = (R0, 0) = (0.10, 0.00)` m.
With `nx = ny = 128`, `dx ≈ 3.1 × 10⁻⁴` m resolves the bore-scale structure;
the CFL timestep is `dt ≈ 3.68 × 10⁻¹³` s. The absorbing particle BC
therefore coincides with the toroid's conductor sheet — a particle that
leaves the PIC domain is, physically, one that hit the magnet wall, so the
alive-count time series is a direct measurement of confinement loss inside
the bore.

## Species

| species | charge (C)                | mass (kg)                | gyrofreq @ B=0.05 T |
|---------|---------------------------|--------------------------|---------------------|
| H⁺      | +1.602176634 × 10⁻¹⁹      | 1.67262192369 × 10⁻²⁷    | ≈ 4.8 × 10⁶ rad/s   |
| μ⁻      | −1.602176634 × 10⁻¹⁹      | 1.883531627 × 10⁻²⁸      | ≈ 4.3 × 10⁷ rad/s   |

At equal temperature the μ⁻ gyroradius is ≈ √(m_μ / m_H) ≈ 0.34 × that of
H⁺. Both species are seeded into a **0.02 m × 0.02 m block** (the toroid's
half-width × half-height) centered on the right-side bore at
(x, y) = (R₀, 0) = (0.10, 0.00) m, with a 10 eV Maxwellian velocity
distribution. Equal density and matched positions make the patch
initially quasineutral; the absorbing particle BC then quantifies loss as
the populations diffuse out of the magnetic bore.

## Numerical caveats

- The model is a stack of concentric current loops bounding a square toroidal
  volume — a discretized current-sheet approximation, not real TF coils. The
  bore field is non-monotonic across the bore midline; for a clean 1/R
  toroidal profile, run more loop filaments and/or use a true helical-TF
  geometry.
- The deck targets **1 µs of physical time** (~2.71M CFL steps at the
  default 128² grid, `dt ≈ 3.685 × 10⁻¹³` s in the bore-sized domain).
  For quick iteration use `--steps-override` (see below).

## Run

```bash
# Regenerate the deck (optional — input.yaml is checked in)
python examples/square_toroid_pic/build_yaml.py

# Smoke test (50 steps; finishes in seconds on an MI300X)
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/square_toroid_pic/input.yaml \
    --print-config --steps-override 50

# Full 1 µs run (~360k steps)
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/square_toroid_pic/input.yaml

# Render PNGs from out.npz
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.postprocess examples/square_toroid_pic/out.npz
```

The CLI writes `out.npz` next to the input, with field arrays
(`external_bz`, `field_bz`, `field_ex`, `field_ey`), per-species particle
dumps (`species_H+_{x,y,vx,vy,vz,weight,alive}`,
`species_mu-_{...}`), and metadata (`final_step`, `final_time_s`,
`nx`, `ny`).
