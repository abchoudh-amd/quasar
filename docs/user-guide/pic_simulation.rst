PIC simulation workflow
=======================

The ``quasar.pic`` Python package wraps the 2D-3V electromagnetic PIC core
behind a YAML deck + CLI: ``python -m quasar.pic.cli run <input.yaml>``
builds the solver, seeds species, applies an external field, steps the
solver, and writes results to ``out.npz`` next to the deck.

Deck schema (``quasar.pic.io``)
-------------------------------

.. code-block:: yaml

   units: SI                # 'SI' or 'normalized'

   normalization:           # optional; required when units == normalized
     reference_density_per_m3: 1.0e15
     reference_species: electron

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
     current_filter:        # optional; ordered current-smoothing pipeline
       - {type: binomial, n_passes: 2}
       # types: 'binomial', 'compensated_binomial'

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
         distribution: maxwellian_uniform   # 'maxwellian_uniform' or 'maxwellian_block'
         density_per_m3: 1.0e15
         temperature_eV: 10.0
         drift_v: [0.0, 0.0, 0.0]           # optional
         # maxwellian_block also needs a region (metres):
         # region: {x_min_m: ..., x_max_m: ..., y_min_m: ..., y_max_m: ...}

   fields:                   # optional initial field seed (normalized decks)
     initial:
       type: seed_em_wave    # 'seed_perturbation' or 'seed_em_wave'
       component: ez         # field component to seed
       mode: [1, 0]          # spatial mode numbers
       amplitude: 1.0e-3

   boundary:                 # optional, default all-periodic
     particle: [periodic, periodic, specular, specular]
                             # one of {periodic, specular, absorbing}; either a
                             # single string (applied to all four sides) or a
                             # 4-list ordered [x_lo, x_hi, y_lo, y_hi].
     field: [periodic, periodic, pec, outflow]
                             # one of {periodic, pec, outflow}; same single-string
                             # or 4-list [x_lo, x_hi, y_lo, y_hi] form.

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
* ``--verbose``           — print informational output (the driver is quiet by
  default).
* ``--print-config``      — echo the resolved deck and ``dt`` before running.
* ``--steps-override N``  — override ``time.steps`` (smoke tests / CI).
* ``--log-every N``       — every ``N`` steps, print a progress line
  (step, time, step-rate, ETA, per-species alive count) and append a row to the
  scalar ``series_*`` diagnostics. ``0`` (default) disables periodic logging;
  a final row is still recorded at the end of the run.
* ``--write-every N``     — flush a rolling checkpoint of ``out.npz`` every
  ``N`` steps. ``0`` (default) writes only once at the end of the run.

Output (``out.npz``)
--------------------

Top-level keys:

* ``final_step``, ``final_time_s``, ``nx``, ``ny``  — scalars in 1-D arrays.
* ``external_bx``, ``external_by``, ``external_bz`` — sampled external
  field, flat ``(nx+2*g)*(ny+2*g)`` arrays where ``g = required_nghost(fdtd_order)``
  (1 for 2nd-order, 2 for 4th-order). Use ``quasar.pic.postprocess.reshape_with_ghost``
  to recover the interior view.
* ``field_<name>`` — per-component Yee field at the final step (same layout).
* ``species_<name>_{x,y,vx,vy,vz,weight,alive}`` — per-species particle
  snapshots when ``per_species: true``.
* ``snapshot_steps``, ``snapshot_times_s``, ``snapshot_field_<name>`` —
  populated when ``cadence > 0``.
* ``series_step``, ``series_time_s``, ``series_alive_<name>`` — scalar time
  series recorded at each ``--log-every`` tick (plus a final row). The
  per-species ``series_alive_<name>`` is the live-particle count, computed by a
  device-side reduction so logging does not copy the full particle arrays.

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
