# square_quad_pic

EM-PIC run of an **H+ / mu- plasma** sitting on the central null of the 30 cm
square-frame **magnetic quadrupole** built in `square_quad_field`. The 256
z-directed filaments (top/bottom `+z`, left/right `-z`) are imposed as the
external Biot-Savart field on a tiny **0.25 cm x 0.25 cm** patch centered on the
magnet's null.

```
   external B on the z=0 patch (quadrupole): transverse, zero at center

         ^ y                 B field (schematic)
         |     \   |   /
         |      \  |  /
   ------+------- null -------> x      |B| grows ~linearly outward
         |      /  |  \
         |     /   |   \
```

## Physics notes

- **Quadrupole / central null.** In the z=0 midplane the magnet field is
  transverse (`B_x`, `B_y`) and vanishes at the center, so the patch is a
  magnetic null with a near-linear gradient -- a transverse-focusing region.
- **In-plane B couples to v_z.** The slice is 2D-xy but the field is in-plane,
  so the magnetic force `q v x B` rotates in-plane velocity into the
  **out-of-plane** `v_z`. This is the 2D3V model working as intended; it is
  *not* in-plane cyclotron gyration (which needs an out-of-plane `B_z`).
- **Two species, 10 keV.** H+ (`q=+e`, proton mass) and mu- (`q=-e`, muon mass)
  are each loaded `maxwellian_uniform` at `temperature_eV = 10000`. The lighter,
  oppositely-charged muon responds faster and bends the opposite way.
- **Field boundary is PEC, not outflow.** The first-order Mur `outflow` path does
  not preserve the deposited-current Gauss constraint, so charged species are
  rejected with Mur boundaries. This example uses the charge-compatible `pec`
  boundary. Because the self-consistent fields are negligible next to the static
  external quadrupole, PEC reflection is not expected to control the illustrated
  particle dynamics.

## Run

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.pic.cli run examples/square_quad_pic/input.yaml
```

128x128 cells, 100 ppc, two species => ~3.28M particles, 10000 steps.

`units: SI`. Output `out.npz` keys:

- `external_bx`, `external_by` -- the quadrupole field sampled on the grid
  (transverse; `external_bz ~ 0`).
- `field_{ex,ey,ez,bx,by,bz}` and `snapshot_field_*` -- self-consistent fields
  (cadence 500 => 20 snapshots).
- `species_{H+,mu-}_{x,y,vx,vy,vz,weight,alive}` -- final per-particle state.

## Timescale / why 10000 steps

The EM solver is **explicit**, and the automatic policy bounds `dt` by the
light-crossing CFL limit:

```
dt_auto = 0.5 / (c * sqrt(1/dx^2 + 1/dy^2)) ~ 2.3e-14 s   (dx = dy ~ 19.5 um)
```

So the shipped 10000 steps span only **~0.23 ns**. A full **1 us** study would
need ~43 million steps at this resolution -- infeasible as a routine example.
The automatic policy does not enforce plasma-frequency or particle-orbit
resolution; production studies must impose those additional timestep criteria.
To extend the physical time, either raise `time.steps` deliberately (and expect
a long GPU run) or coarsen the grid (larger `dx` => larger `dt`).

## Reference signature

The integration test runs a short, reduced proxy of this deck and asserts: the
run completes with finite fields; `external_bx`/`external_by` carry the
transverse quadrupole field (non-trivial, `~0` at the null) while `external_bz`
stays negligible; particle count is non-increasing under the absorbing walls;
and the species move (non-zero velocities).

## Regenerate

```bash
python examples/square_quad_field/build_yaml.py
```
