# square_quad_field

Magnetic-field calculator for a **30 cm square-frame quadrupole**: 256 straight
filaments running **along the lab z-axis** (perpendicular to the 30 cm plane),
arranged as a square frame and evaluated with the Biot-Savart evaluator.

```
              +z  +z  +z ... +z          y = +0.15 m  (top row, +z)
            o   o   o   o   o   o
   -z  o                         o  -z
   -z  o                         o  -z    x = -0.15 m (left, -z)
   -z  o            x            o  -z    x = +0.15 m (right, -z)
   -z  o          (null)         o  -z
   -z  o                         o  -z
            o   o   o   o   o   o
              +z  +z  +z ... +z          y = -0.15 m  (bottom row, +z)

   o = a z-directed wire piercing the page;  current sign as labeled.
   Each side: 64 wires, 234.375 A each (15 kA per side).
```

Top & bottom rows carry **+z**, left & right rows carry **-z**. This `+/+/-/-`
pattern is a **magnetic quadrupole**: in the z=0 midplane the field is purely
**transverse** (`B_x`, `B_y`), **zero on the central axis**, and grows roughly
linearly with distance from the center. The observation grid is the 0.25 cm
patch (`x, y in [-1.25, +1.25] mm`, `z = 0`) used by the sibling PIC example
`square_quad_pic`, so this deck is literally the field the PIC particles feel.

> The four conductors are modeled as **independent** z-directed wires (the
> currents do not connect into a closed circuit) -- a magnetostatic
> idealization, the same style as `square_toroid`'s discretized sheets.

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.coil.cli run examples/square_quad_field/input.yaml
```

`units: SI`. Output `out.npz` keys:

- `B_xyz` -- B at each observation point, shape `(128*128, 3)`, tesla.
- `B_magnitude` -- `|B|` per point.
- `dims`, `observation_kind` -- grid metadata (`[128, 128, 1]`, `"grid"`).

## Reference signature

At the z=0 midplane of long z-wires the per-wire field is the 2D form
`B = mu_0 I / (2 pi d)`, azimuthal about each wire; the total is their
superposition. Expectations, checked by the integration test:

- `B = 0` at the exact geometric center. Because the even 128-point observation
  lattice does not contain that point, its nearest four samples are small but
  nonzero compared with the patch-edge field.
- The field is transverse: `|B_z| << |B_x|, |B_y|` everywhere on the patch.
- Sampled points match the infinite-wire superposition within a few percent
  (the wires are long but finite, `z in [-1, +1] m`).

## Regenerate

Both this deck and `square_quad_pic/input.yaml` come from one geometry source:

```bash
python examples/square_quad_field/build_yaml.py
```
