# mhd_rotor

The classic **MHD rotor** (Balsara & Spicer, *J. Comput. Phys.* **149**, 270
(1999); Tóth, *J. Comput. Phys.* **161**, 605 (2000)). A dense, rapidly-rotating
disk sits in a static, uniformly-magnetized ambient medium. As the disk spins it
winds up the threaded field, launching strong **torsional Alfvén waves** that
carry angular momentum outward and brake the rotor; the wound-up magnetic
pressure compresses the central disk into an oblate shape. It is the standard
test for rotational discontinuities, strong torsional Alfvén waves, and
positivity in the rarefied wake.

## Run

From the repository root, with the build-tree Python package on `PYTHONPATH`:

```bash
PYTHONPATH=build/hip-gfx942-release/python \
  python -m quasar.mhd.cli run examples/mhd_rotor/input.yaml
```

The deck is in `units: normalized` (`gamma = 5/3`). Output is written next to
`input.yaml` as `out.npz`.

## Canonical setup

On `[0,1] × [0,1]`, rotor centered at `(0.5, 0.5)`, with `gamma = 5/3`:

```
r0 = 0.1 (disk radius),  r1 = 0.115 (linear taper to ambient)
inside  (r < r0):  rho = 10,  v = omega * (-(y-0.5), (x-0.5), 0),  omega = u0/r0
outside (r > r1):  rho = 1,   v = 0
taper   (r0<r<r1): linear blend f = (r1-r)/(r1-r0) of rho and the rim velocity
everywhere:        p = 1,  B = (5/sqrt(4*pi), 0, 0)
```

with rim speed `u0 = 2`, i.e. angular velocity `omega = u0/r0 = 20`.

## Reference / validation

Run to the canonical time `t = 0.15`. The expected outcome is the well-known
torsional Alfvén-wave pattern: the initially circular disk is wound up and
flattened, with the magnetic field swept into a rotating central structure and
Alfvén waves propagating into the ambient medium. As with the blast wave, the
decisive robustness check is **positivity** — `rho` and `p` must stay strictly
positive in the rarefied core/wake — together with a controlled `div B` monitor.

The integration test in `tests/python/test_examples.py` runs this deck and
asserts the run completes with `min(rho) > 0` and `min(p) > 0` and that the
central disk has spun up the surrounding field (nontrivial `by` away from the
axis where it began at zero).

The `div B` check is **field-relative**: the test asserts `div B * dx / |B|`
(dimensionless, ~1e-14) is at machine epsilon, not the raw *absolute*
`divb_linf_final`. The absolute L-inf scales with `|B|` and accumulates
telescoping round-off that grows with the 4000-step count (~1.4e-10 here), so a
strict absolute `1e-10` bound would flag a solenoidal field. This deck runs at
`cfl = 0.2` for headroom with the non-positivity-preserving MP7 reconstruction
(see the `orszag_tang` example for the full rationale).
