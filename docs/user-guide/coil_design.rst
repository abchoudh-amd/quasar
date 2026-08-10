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

Flags:

* ``--print-config``  — echo the parsed deck before running.
* ``--verbose``       — print informational output (quiet by default).

(The PIC driver, ``quasar.pic.cli``, shares the same ``run <input.yaml>``
subcommand shape and the ``--print-config`` / ``--verbose`` flags; see
:doc:`pic_simulation` for its additional ``--seed`` / ``--log-every`` /
``--write-every`` / ``--steps-override`` options.)

Reusing a magnet in plasma simulations
---------------------------------------

The conductor record under ``conductors:`` is shared by the coil, MHD, and PIC
deck loaders. For a serial MHD deck with ``units: SI``, place the same non-empty
list directly under ``background_field``:

.. code-block:: yaml

   background_field:
     enabled: true
     conductors:
       - name: single_loop
         current_A: 1.0
         geometry:
           type: circular_loop
           center_xyz: [0.0, 0.0, 0.0]
           axis_xyz: [0.0, 0.0, 1.0]
           radius_m: 0.1
           n_segments: 256

The MHD loader derives its padded corner grid from the domain and the solver's
actual reconstruction halo, evaluates ``A`` in-process, and takes the matching
Cartesian or cylindrical discrete curl. This is the preferred coil-to-MHD
workflow because there is no observation grid or intermediate file to keep in
sync. See :doc:`mhd_background_field` and the one-deck
``examples/mhd_coil_cartesian/`` and ``examples/square_toroid_mhd/`` cases.
Use ``background_field.a_file`` only when a precomputed ``A_xyz_grid`` archive
is specifically required.

For PIC, put the list under the Biot--Savart external evaluator:

.. code-block:: yaml

   external_field:
     evaluator:
       type: biot_savart
       conductors:
         - name: single_loop
           current_A: 1.0
           geometry:
             type: circular_loop
             center_xyz: [0.0, 0.0, 0.0]
             axis_xyz: [0.0, 0.0, 1.0]
             radius_m: 0.1
             n_segments: 256

PIC samples the prescribed field on its own component lattices; see
:doc:`pic_simulation`. The standalone coil CLI remains useful for line, plane,
or grid surveys and for exporting reusable field maps.

Schema reference
----------------

Top-level keys
~~~~~~~~~~~~~~

================= ===============================================================
Key               Meaning
================= ===============================================================
``units``         Must be ``SI``. Other unit systems are not supported.
``conductors``    Optional list of conductor specifications (see below). It is
                  required and non-empty for ``biot_savart``; analytic and
                  plugin evaluators may omit it.
``observation``   Single observation-set specification (see below).
``output``        Output archive descriptor (``format``, ``path``, ``fields``).
``evaluator``     Optional evaluator mapping (default ``{type: biot_savart}``).
                  Analytic evaluator parameters are listed below.
================= ===============================================================

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

Evaluator spec
~~~~~~~~~~~~~~

The following registry evaluators are supported. Parameters are validated and
passed to the C++ evaluator's ``configure`` method; unknown keys are rejected.

================ =============================================================
``type``         Parameters
================ =============================================================
``biot_savart``  No parameters; evaluates the ``conductors`` list.
``uniform``      ``B_T`` (optional, default ``[0,0,0]``) and optional
                 ``E_V_per_m``.
``dipole``       Required ``moment_Am2`` and optional ``origin_xyz_m``.
``gradient``     Required trace-free 3x3 ``grad_T_per_m`` plus optional
                 ``B0_T`` and ``origin_xyz_m``. Trace-free enforces
                 :math:`\nabla\cdot\mathbf B=0`.
``file_grid``    Required ``path`` to a rectilinear NumPy field map (relative
                 to the deck): ``B_xyz_grid[nz,ny,nx,3]``, ``grid_origin[3]``,
                 and ``grid_spacing[3]`` (positive on every non-singleton axis;
                 zero is accepted and canonicalized on singleton axes). Samples
                 use trilinear interpolation and reject points outside the map.
                 A singleton axis is one geometric plane, so such a map supports
                 field sampling but not a complete magnetic-field Jacobian.
                 The shorter ``origin`` / ``spacing`` names are accepted as
                 aliases, but an archive must provide exactly one spelling of
                 each and exactly one field representation (``B_xyz_grid`` or
                 flat ``B_xyz`` with ``dims``). Unknown archive keys are rejected.
``<plugin>``     Any other live registered evaluator name. Supply its generic
                 scalar/flat-list values under ``params``.
================ =============================================================

For example, a plugin registered as ``my_plugin`` can be selected without a
coil-CLI code change:

.. code-block:: yaml

   evaluator:
     type: my_plugin
     params:
       gain: 2.0
       axis: [1.0, 0.0, -1.0]

The selectable names are the live, sorted result of
``_core.magnetostatics.field_evaluator_names()``. For a plugin, parameter keys
must be non-empty strings and values must be finite real scalars or flat finite
lists/tuples. Scalars are normalized to one-element lists. Booleans,
strings/bytes, mappings, nested sequences, and non-finite values are rejected.
Plugin parameters are already-resolved values in deck units; built-ins retain
the named, unit-aware schemas in the table above. The selected evaluator's
``configure`` method performs its own key and arity validation.

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

``format`` must be ``npz``. ``fields`` is a non-empty list drawn from ``B_xyz``,
``B_magnitude``, ``A_xyz``, and (for ``type: grid``) ``B_xyz_grid`` and
``A_xyz_grid``. Vector-potential fields require an evaluator whose capability
metadata reports A support and whose implementation overrides ``evaluate_A``;
``biot_savart`` is the current built-in provider, and a plugin may advertise the
same capability. Unknown field names are rejected. The archive
also always includes ``dims`` and ``observation_kind`` for downstream
postprocessing. Grid observations additionally include ``grid_origin`` and
``grid_spacing``, so an archive containing ``B_xyz_grid`` can be consumed
directly by the ``file_grid`` evaluator.

For a full three-dimensional map (at least two nodes on every axis), the
evaluator's Jacobian is the exact derivative of the piecewise-trilinear
interpolant. At an internal grid knot, where that derivative need not be unique,
the implementation uses the cell on the positive-coordinate (right) side; the
outermost upper boundary necessarily uses the final cell on its left. If any
axis is singleton, ``provides_grad_B`` is false and a direct Jacobian request is
rejected: samples on one plane cannot determine the normal derivative.

The C++ ``FileGridEvaluator(path)`` constructor also accepts a dependency-free
text form, with x varying fastest in the data block:

.. code-block:: text

   QUASAR_FILE_GRID 1
   dims nx ny nz
   origin x0 y0 z0
   spacing dx dy dz
   data
   Bx(0,0,0) By(0,0,0) Bz(0,0,0)
   ... exactly nx*ny*nz rows ...

Python API
----------

The :mod:`quasar.coil` package exposes the commonly used C++ surface:

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

The C++ ``BiotSavartConfig`` carries an opaque
``quasar::backend::stream_t`` slot for chaining evaluation onto a pre-existing
device stream. Python deliberately exposes only the default configuration; it
does not expose or accept raw device-stream handles.

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

The ``examples/`` directory ships six runnable magnetostatics decks:

* ``single_loop`` - on-axis profile of one circular loop.
* ``helmholtz_pair`` - quasi-uniform region between two coaxial loops.
* ``solenoid`` - 200-turn helical solenoid with the canonical tabletop
  axial profile.
* ``saddle_coil`` - non-planar polyline ring and a C2-symmetry sanity
  check at the origin.
* ``square_toroid`` - square-cross-section toroidal magnet built from
  four current sheets.
* ``square_quad_field`` - square-frame quadrupole built from straight
  axial filaments.

Each example has a ``README.md`` with the analytical references or physical
invariants checked by its integration test.
