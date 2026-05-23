# H⁺ / μ⁻ plasma in a square-toroid magnet

End-to-end PIC demo: seed a quasineutral H⁺ + μ⁻ plasma in the equatorial
slice of the square-cross-section toroidal magnet from
[`examples/square_toroid`](../square_toroid/), then step the
electromagnetic PIC solver under that external B-field.

## Geometry mapping

The toroid's symmetry axis is **z_lab**. The PIC sampler always evaluates the
external field at z=0 (`src/physics/pic/external_field_sampler.cpp`), so the
PIC plane (x_pic, y_pic) maps to the lab equatorial slice (x_L, y_L, z_L=0).
The donut bore appears in PIC as an annular ring centered on the origin
(inner radius `R0 − a/2`, outer radius `R0 + a/2`). The external Bz seen by
PIC is the axial (out-of-plane) component of the toroid's field — which is
what drives in-plane cyclotron rotation of the species.

For a tokamak-style toroidal-field profile (1/R falloff), the equatorial
slice cuts across the bore: Bz is strongest in a band along the bore
midline and reverses sign across the bore boundaries.

## Species

| species | charge (C)                | mass (kg)                | gyrofreq @ B=0.05 T |
|---------|---------------------------|--------------------------|---------------------|
| H⁺      | +1.602176634 × 10⁻¹⁹      | 1.67262192369 × 10⁻²⁷    | ≈ 4.8 × 10⁶ rad/s   |
| μ⁻      | −1.602176634 × 10⁻¹⁹      | 1.883531627 × 10⁻²⁸      | ≈ 4.3 × 10⁷ rad/s   |

At equal temperature the μ⁻ gyroradius is ≈ √(m_μ / m_H) ≈ 0.34 × that of
H⁺. The two species are seeded with equal density and uniform spatial
distribution over the PIC domain so the configuration is initially
quasineutral.

## Numerical caveats

- The model is a stack of concentric current loops bounding a square toroidal
  volume — a discretized current-sheet approximation, not real TF coils. The
  bore field is non-monotonic across the bore midline; for a clean 1/R
  toroidal profile, run more loop filaments and/or use a true helical-TF
  geometry.
- Steps + dt are tuned for a fast smoke test, not a full gyration. The
  default `dt = auto` resolves CFL but does not resolve gyration to
  visualization quality. For a real run, override `time.steps` and bias
  toward a smaller PIC grid (so CFL gives a larger `dt`) so multiple
  gyroperiods fit in the run.

## Run

```bash
# Regenerate the deck (optional — input.yaml is checked in)
python examples/square_toroid_pic/build_yaml.py

# Smoke test (50 steps; finishes in seconds on an MI300X)
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/square_toroid_pic/input.yaml \
    --print-config --steps-override 50

# Full run (2000 steps)
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
