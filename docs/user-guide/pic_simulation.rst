PIC simulation workflow
=======================

The ``quasar.pic`` Python package wraps the 2D-3V electromagnetic PIC core
behind a YAML deck + CLI: ``python -m quasar.pic.cli run <input.yaml>``
builds the solver, seeds species, applies an external field, steps the
solver, and writes results to ``out.npz`` next to the deck.

Deck schema (``quasar.pic.io``)
-------------------------------

.. code-block:: yaml

   units: SI

   domain:
     nx: 128
     ny: 128
     lx_m: 0.30
     ly_m: 0.30
     origin_x_m: -0.15      # optional, default 0
     origin_y_m: -0.15      # optional, default 0

   numerics:
     fdtd_order: 2          # 2 or 4
     shape: cic             # 'cic' (linear) or 'tsc' (quadratic)

   external_field:          # optional
     evaluator:
       type: biot_savart
       conductors:          # same schema as quasar.coil
         - name: loop_0
           current_A: 1000.0
           geometry:
             type: circular_loop
             center_xyz: [0.0, 0.0, 0.0]
             axis_xyz:   [0.0, 0.0, 1.0]
             radius_m: 0.04
             n_segments: 64

   species:
     - name: H+
       charge_C: 1.602176634e-19
       mass_kg: 1.67262192369e-27
       n_particles: 20000
       initial:
         distribution: maxwellian_uniform   # only option today
         density_per_m3: 1.0e15
         temperature_eV: 10.0
         drift_v: [0.0, 0.0, 0.0]           # optional

   time:
     dt_s: auto              # float or 'auto' (CFL-limited)
     steps: 2000

   diagnostics:
     output_path: out.npz
     cadence: 200            # snapshot every N steps; 0 = final only
     fields: [bz, ex, ey]    # any subset of {ex,ey,ez,bx,by,bz}
     per_species: true

Conductor specs in ``external_field.evaluator.conductors`` are passed
through ``quasar.coil.io._build_geometry`` and therefore accept any
geometry the coil pipeline supports (``circular_loop``, ``helix``,
``solenoid``, ``polygon``, ``generic_polyline``, ``racetrack``).

CLI
---

.. code-block:: bash

   PYTHONPATH=build/hip-gfx942-release/python \\
     python -m quasar.pic.cli run examples/square_toroid_pic/input.yaml \\
       --print-config --steps-override 50

Flags:

* ``--seed N``            — RNG seed for initial-condition sampling (0).
* ``--print-config``      — echo the resolved deck and ``dt`` before running.
* ``--steps-override N``  — override ``time.steps`` (smoke tests / CI).

Output (``out.npz``)
--------------------

Top-level keys:

* ``final_step``, ``final_time_s``, ``nx``, ``ny``  — scalars in 1-D arrays.
* ``external_bx``, ``external_by``, ``external_bz`` — sampled external
  field, flat ``(nx+2)*(ny+2)`` arrays including one ghost cell per side.
* ``field_<name>`` — per-component Yee field at the final step (same layout).
* ``species_<name>_{x,y,vx,vy,vz,weight,alive}`` — per-species particle
  snapshots when ``per_species: true``.
* ``snapshot_steps``, ``snapshot_times_s``, ``snapshot_field_<name>`` —
  populated when ``cadence > 0``.

Worked example
--------------

``examples/square_toroid_pic/`` runs a quasineutral H⁺ + μ⁻ plasma in the
equatorial slice of the square-cross-section toroidal magnet from
``examples/square_toroid``. See ``examples/square_toroid_pic/README.md``
for the physics rationale and reference numbers.

Postprocessing
--------------

``quasar.pic.postprocess`` renders field heatmaps and per-species
particle scatter from an ``out.npz``:

.. code-block:: bash

   PYTHONPATH=build/hip-gfx942-release/python \\
     python -m quasar.pic.postprocess examples/square_toroid_pic/out.npz

matplotlib is imported lazily; install it locally if you want plots.
