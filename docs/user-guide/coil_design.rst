Coil-design workflow
====================

This walk-through covers the three entry points to the magnetostatics
module:

* the YAML input deck consumed by ``python -m quasar.coil.cli run``;
* the Python API in :mod:`quasar.coil`; and
* the post-processing helpers in :mod:`quasar.coil.postprocess`.

End-to-end example
------------------

Take ``examples/single_loop/input.yaml`` as a starting point:

.. code-block:: yaml

   units: SI

   conductors:
     - name: single_loop
       current_A: 1.0
       geometry:
         type: circular_loop
         center_xyz: [0.0, 0.0, 0.0]
         axis_xyz:   [0.0, 0.0, 1.0]
         radius_m: 0.1
         n_segments: 256

   observation:
     type: line
     start_xyz: [0.0, 0.0, 0.00]
     end_xyz:   [0.0, 0.0, 0.20]
     n_points: 5

   output:
     format: npz
     path: out.npz
     fields: [B_xyz, B_magnitude]

Run with

.. code-block:: bash

   PYTHONPATH=build/hip-gfx942-release/python \
     python -m quasar.coil.cli run examples/single_loop/input.yaml

The CLI parses the YAML through :mod:`quasar.coil.io`, materializes the
conductor system + observation cloud, launches the Biot-Savart HIP
kernel via :py:class:`quasar.coil.BiotSavartEvaluator`, and writes a
``.npz`` archive next to the input.

Schema reference
----------------

Top-level keys
~~~~~~~~~~~~~~

============== =================================================================
Key            Meaning
============== =================================================================
``units``      Must be ``SI``. Other unit systems are not supported.
``conductors`` Non-empty list of conductor specifications (see below).
``observation`` Single observation-set specification (see below).
``output``     Output archive descriptor (``format``, ``path``, ``fields``).
============== =================================================================

Conductor spec
~~~~~~~~~~~~~~

Each conductor is a mapping with ``name`` (string), ``current_A``
(float), and ``geometry`` (mapping with a ``type`` discriminator):

================= =========================================================
``type``          Required additional fields
================= =========================================================
``circular_loop`` ``center_xyz``, ``axis_xyz``, ``radius_m``, ``n_segments``
``helix``         ``center_xyz``, ``axis_xyz``, ``radius_m``, ``pitch_m``,
                  ``n_turns``, ``n_segments_per_turn``
``solenoid``      ``center_xyz``, ``axis_xyz``, ``radius_m``, ``length_m``,
                  ``n_turns``, ``n_segments_per_turn``
``racetrack``     ``center_xyz``, ``axis_xyz``, ``straight_length_m``,
                  ``arc_radius_m``, ``n_arc_segments``
``polygon``       ``center_xyz``, ``axis_xyz``, ``circumradius_m``,
                  ``n_sides``
``polyline``      ``points_xyz_m`` (list of ``[x, y, z]``)
================= =========================================================

Observation spec
~~~~~~~~~~~~~~~~

================ ==========================================================
``type``         Required additional fields
================ ==========================================================
``points``       ``points_xyz_m``
``line``         ``start_xyz``, ``end_xyz``, ``n_points``
``plane``        ``origin_xyz``, ``u_axis_xyz``, ``v_axis_xyz``,
                 ``u_extent_m``, ``v_extent_m``, ``nu``, ``nv``
``grid``         ``bounds_m`` (list of three ``[min, max]`` pairs),
                 ``resolution`` (``[nx, ny, nz]``)
================ ==========================================================

Output spec
~~~~~~~~~~~

``format`` must currently be ``npz`` (VTK output is a future
deliverable). ``fields`` is a list of zero or more of ``B_xyz``,
``B_magnitude`` and (for ``type: grid``) ``B_xyz_grid``. The archive
also always includes ``dims`` and ``observation_kind`` for downstream
postprocessing.

Python API
----------

The :mod:`quasar.coil` package mirrors the C++ surface one-to-one:

.. code-block:: python

   from quasar.coil import (
       BiotSavartEvaluator, ConductorSystem, PointCloud, Vec3,
       circular_loop,
   )

   cs = ConductorSystem()
   cs.add(circular_loop(
       center=Vec3(0, 0, 0), axis=Vec3(0, 0, 1),
       radius_m=0.1, n_segments=64, current_A=1.0,
   ))

   obs = PointCloud()
   obs.add(Vec3(0, 0, 0.05))

   eval_ = BiotSavartEvaluator()
   B    = eval_.evaluate_B(cs, obs)            # numpy (M, 3)
   gB   = eval_.evaluate_grad_B(cs, obs)       # numpy (M, 3, 3)

``BiotSavartConfig`` carries a ``hipStream_t`` slot for users who need
to chain the evaluator onto a pre-existing HIP stream.

Plotting and post-processing
----------------------------

:mod:`quasar.coil.postprocess` provides data-shaping helpers
(``magnitude``, ``reshape_to_grid``, ``slice_xy``/``slice_xz``/
``slice_yz``) and matplotlib-backed plotters
(``plot_magnitude_slice``, ``plot_line_profile``). The plotters lazy-
import :mod:`matplotlib`, so the data-shaping helpers stay usable on
machines without it.

.. code-block:: python

   from quasar.coil import postprocess
   import numpy as np

   archive   = np.load("out.npz")
   B_flat    = archive["B_xyz"]
   dims      = archive["dims"]
   B_grid    = postprocess.reshape_to_grid(B_flat, dims)
   fig, ax   = postprocess.plot_magnitude_slice(B_grid, axis="z")
   fig.savefig("Bmag_z.png", dpi=150)

Worked examples
---------------

The ``examples/`` directory ships four runnable decks:

* ``single_loop`` - on-axis profile of one circular loop.
* ``helmholtz_pair`` - quasi-uniform region between two coaxial loops.
* ``solenoid`` - 200-turn helical solenoid with the canonical tabletop
  axial profile.
* ``saddle_coil`` - non-planar polyline ring and a C2-symmetry sanity
  check at the origin.

Each example has a ``README.md`` with the analytical reference values
the integration tests check against.
