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

   normalization:           # optional SI -> internal-unit reference scales
     reference_density_per_m3: 1.0e15
     reference_species: electron

   neutralizing_background: false
                             # set true for a non-neutral population on a
                             # doubly periodic domain

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
       - {type: binomial, n_passes: 2}   # n_passes must be >= 1
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
       # Other evaluator forms:
       #   uniform:  {type: uniform,  B_T: [0, 0, 1.0], E_V_per_m: [0, 0, 0]}
       #   dipole:   {type: dipole,   moment_Am2: [0, 0, 1], origin_xyz_m: [0, 0, 0]}
       #   gradient: {type: gradient, B0_T: [0, 0, 0],
       #              grad_T_per_m: [[1, 0, 0], [0, 1, 0], [0, 0, -2]]}
       #   file_grid: {type: file_grid, path: field_map.npz}

   species:
     - name: H+
       charge_C: 1.602176634e-19
       mass_kg: 1.67262192369e-27
       n_particles: 20000
       initial:
         distribution: maxwellian_uniform   # 'maxwellian_uniform' or 'maxwellian_block'
         density_per_m3: 1.0e15
         temperature_eV: 10.0
         drift_v: [0.0, 0.0, 0.0]           # optional physical velocity at t=0
         # Optional deterministic velocity mode, in the deck's velocity units:
         velocity_perturbation:
           amplitude_v: [1.0e-6, 0.0, 0.0]
           mode: [1, 0]                      # full wavelengths across x and y
           phase_rad: 0.0
         # maxwellian_block also needs a region (metres):
         # region: {x_min_m: ..., x_max_m: ..., y_min_m: ..., y_max_m: ...}

   fields:                   # optional initial field seed
     initial:
       type: seed_em_wave    # also 'seed_perturbation' or 'seed_tm_cavity'
       component: ez         # field component to seed
       mode: [1, 0]          # +x propagation; my must be zero
       amplitude: 1.0e-3     # V/m for E in SI decks; internal units if normalized

   boundary:                 # optional, default all-periodic
     particle: [periodic, periodic, specular, specular]
                             # one of {periodic, specular, absorbing}; either a
                             # single string (applied to all four sides) or a
                             # 4-list ordered [x_lo, x_hi, y_lo, y_hi].
     field: [periodic, periodic, pec, outflow]
                             # one of {periodic, pec, outflow}; same single-string
                             # or 4-list [x_lo, x_hi, y_lo, y_hi] form. Outflow
                             # is vacuum-only; charged species require periodic
                             # or PEC field boundaries.

   time:
     dt_s: auto              # float or 'auto' (Maxwell CFL-limited only)
     steps: 2000             # safety cap
     t_end_s: 1.0e-8         # optional; final position step is clipped exactly

   diagnostics:
     output_path: out.npz
     cadence: 200            # snapshot every N steps; 0 = final only
     fields: [bz, ex, ey]    # any subset of {ex,ey,ez,bx,by,bz}
     per_species: true

Fourth-order FDTD requires ``domain.nx >= 2`` and ``domain.ny >= 2``. Its
two-layer non-periodic ghost continuation needs distinct physical source cells
on both sides; a one-cell dimension is therefore rejected before allocation.

Conductor specs in ``external_field.evaluator.conductors`` are passed
through ``quasar.coil.io._build_geometry`` and therefore accept any
geometry the coil pipeline supports (``circular_loop``, ``helix``,
``solenoid``, ``polygon``, ``polyline``, ``racetrack``).

Shared conductor geometry
~~~~~~~~~~~~~~~~~~~~~~~~~

The list is intentionally the same as the top-level ``conductors`` list in a
:doc:`coil-design deck <coil_design>`. Copy a validated magnet geometry under
``external_field.evaluator.conductors`` to use it as PIC's prescribed field;
PIC samples the evaluator directly at its component locations, so no standalone
coil-CLI run or field-map file is required. The ``examples/coil_confinement/``
and ``examples/square_toroid_pic/`` decks demonstrate this inline workflow.

MHD reuses the same records under ``background_field.conductors``. Its loader
evaluates vector potential on a solver-derived padded corner grid and takes a
discrete curl, because a static MHD background must be represented on the
face-staggered constrained-transport lattice. See
:doc:`mhd_background_field` and ``examples/mhd_coil_cartesian/``.

Registered evaluator plugins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Any name in the live, sorted
``_core.magnetostatics.field_evaluator_names()`` result can be selected as an
external evaluator. Evaluators other than the built-ins use a generic
``params`` mapping:

.. code-block:: yaml

   external_field:
     evaluator:
       type: my_plugin
       params:
         gain: 2.0
         axis: [1.0, 0.0, -1.0]

A plugin parameter key must be a non-empty string. Its value must be either a
finite real scalar or a flat list/tuple of finite reals; a scalar is normalized
to a one-element list before ``configure`` is called. Booleans, strings/bytes,
mappings, nested sequences, and non-finite values are rejected. Generic plugin
parameters are already-resolved values in the deck's declared units and receive
no aliases or implicit conversion. The built-in evaluators retain the named,
unit-aware schemas shown in the main deck example.

``file_grid`` loads a path confined to the deck directory. The NumPy archive
contains ``B_xyz_grid`` with shape ``(nz, ny, nx, 3)``, plus three-element
``grid_origin`` and ``grid_spacing`` arrays. Spacing must be positive on each
non-singleton axis; zero is accepted for a singleton axis in a coil-generated
map and canonicalized internally. The evaluator performs
trilinear interpolation at the component-specific Yee locations and rejects an
attempt to sample beyond the map. In ``units: SI`` the coordinates and field are
metres and tesla; in ``units: normalized`` they are already internal units. A
singleton axis represents one geometric plane rather than a constant extrusion,
so the map can still supply ``B`` on that plane but reports no trustworthy full
Jacobian and rejects ``evaluate_grad_B``. In particular, a nonzero cylindrical
external field—which requires a continuous ``div(B)`` check—must use a full
three-dimensional file grid or another gradient-capable evaluator.

``initial.drift_v`` and ``initial.velocity_perturbation`` describe the physical
particle velocity distribution at ``t=0``. They are not the preceding leapfrog
half-step velocity and must not be pre-staggered by a deck. On the first solver
step, Boris applies a half-width force update before the full position drift.
That half interval must be representable when a populated species is charged;
for a neutral species the force map is the identity, so only the full position
interval needs to be representable.

``initial.velocity_perturbation`` adds
``amplitude_v * sin(2*pi*(mx*x/Lx + my*y/Ly) + phase_rad)`` to every sampled
particle velocity after the thermal quiet start. It is useful for reproducible
linear-instability studies: applying the same perturbation to equal
counter-streaming beams seeds a current mode while leaving their initial charge
density uniform. At least one mode number and one amplitude component must be
nonzero, and every mode must lie strictly below the corresponding mesh Nyquist
limit.  On a cylindrical domain containing ``r = 0``, the physical ``vr`` and
``vphi`` components must be odd at the axis while axial ``vz`` is even.  The
loader enforces the compatible radial-mode/phase combinations rather than
accepting a formally sinusoidal but multivalued axis velocity.

A doubly periodic Maxwell domain is a torus, so the volume integral of
``div(E)`` is identically zero.  Its initial particle charge must therefore sum
to zero.  The solver accepts explicitly balanced multispecies populations and
rejects a resolved nonzero total charge.  Set ``neutralizing_background: true``
only when the intended model includes a fixed, uniform counter-charge (for
example an electron-only plasma with immobile ions); the background is never
inserted implicitly.

The uniform background cancels the net charge, not the cellwise noise of a
finite particle load. Quasar does not currently perform an initial discrete
Poisson projection, so an arbitrary zero-field particle seed need not satisfy
the local Gauss constraint. The charge-compatible current update preserves any
initial residual instead of removing it. Also, ``dt_s: auto`` enforces the
Maxwell CFL limit only; users must separately resolve plasma-frequency,
gyrofrequency, and particle-crossing timescales for their model.

Initial electromagnetic seeds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``seed_em_wave`` creates a Cartesian ``+x`` travelling wave and requires
``mode: [mx, 0]`` with transverse ``Ey`` or ``Ez``.  ``seed_perturbation``
creates one transverse component varying in ``x`` (or the documented radial
Bessel profile in cylindrical geometry).  Both interpret ``mode[0]`` as a
periodic full-wavelength count.  A travelling-wave mode must lie strictly below
the Cartesian Nyquist limit; at Nyquist, opposite propagation directions alias
to the same samples.  Cylindrical Bessel indices are likewise restricted to the
resolved radial spectrum.

``seed_tm_cavity`` instead creates a rectangular Cartesian PEC eigenmode:

.. code-block:: yaml

   boundary:
     field: {x_lo: pec, x_hi: pec, y_lo: pec, y_hi: pec}
     particle: {x_lo: specular, x_hi: specular,
                y_lo: specular, y_hi: specular}
   fields:
     initial:
       type: seed_tm_cavity
       component: Ez
       mode: [1, 1]          # positive TM_mn wall-mode indices
       amplitude: 1.0

Here ``Ez`` is the out-of-plane electric component on cell centres and both
mode indices must be positive.  The seeder uses
``sin(m*pi*x/Lx) sin(n*pi*y/Ly)`` on that lattice, the matching ``Bx`` and
``By`` sine/cosine profiles on their face lattices, and initializes magnetic
fields at ``t=-dt/2`` from the selected second- or fourth-order Yee modified
wavenumbers.  Thus its frequency is the leapfrog discrete Maxwell frequency,
not merely the continuum value.  All four field sides must be ``pec``; other
boundary topologies and cylindrical geometry are rejected.

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
* ``--write-every N``     — every ``N`` completed steps, write a self-contained,
  step-indexed snapshot ``out_<step>.npz`` (10-digit zero-padded step, e.g.
  ``out_0000000010.npz``) into the deck's output directory. ``0`` (default)
  writes only the end-of-run ``out.npz``. The end-of-run ``out.npz`` (accumulated
  snapshot and scalar series) is always written.

Output (``out.npz``)
--------------------

Top-level keys:

* ``final_step``, ``final_time_s``, ``nx``, ``ny``, ``nghost`` — scalars in
  1-D arrays.
* ``geometry``, ``origin_x``, ``origin_y``, ``lx``, ``ly``, ``unit_system`` —
  coordinate metadata, and ``boundary_field`` — the four field-side boundary
  kinds in ``[x_lo, x_hi, y_lo, y_hi]`` order.
* ``external_bx``, ``external_by``, ``external_bz`` — sampled external
  field, flat ``(nx+2*g)*(ny+2*g)`` arrays.  The actual ``g`` is stored in
  ``nghost`` and is the larger of the FDTD stencil requirement (1 for order 2,
  2 for order 4) and the particle-shape support (order-2 TSC also requires
  ``g = 2``).
* ``field_<name>`` — per-component Yee field at the final step (same layout).
* ``species_<name>_{x,y,vx,vy,vz,weight,alive}`` — per-species particle
  snapshots when ``per_species: true``.
* ``snapshot_steps``, ``snapshot_times_s``, ``snapshot_field_<name>`` —
  populated when ``cadence > 0``.
* ``series_step``, ``series_time_s``, ``series_alive_<name>`` — scalar time
  series recorded at each ``--log-every`` tick (plus a final row). The
  per-species ``series_alive_<name>`` is the live-particle count, computed by a
  device-side reduction so logging does not copy the full particle arrays.

Although all six C++ component allocations have the same padded size, their
physical Yee lattices do not.  Use
``quasar.pic.postprocess.yee_component_view`` to recover a field component and
``yee_component_coordinates`` to obtain its sample coordinates.  The view
retains the independent high face/node on a non-periodic axis and removes that
endpoint only when ``boundary_field`` says both sides of the axis are periodic.
For legacy archives without boundary metadata it conservatively retains all
component-valid high faces.  ``reshape_with_ghost`` remains the cell-centred
``(ny, nx)`` helper and must not be used for a general Yee component.

The reported time labels the integer-time positions and electric field. Before
any evolution, uploaded/deck particle velocities are also physical values at
``t=0``. After one or more steps, raw magnetic fields and particle velocities
are leapfrog quantities at the preceding half time
(``final_time_s - dt_last/2``); the first velocity half step was obtained from
the physical ``v(t=0)`` using a half-width Boris force interval. When the last
step is shortened to hit ``t_end_s`` exactly, the solver uses the variable-step
centred update and does not reinterpret those half-step arrays as integer-time
diagnostics. If ``t_end_s`` is shorter than the nominal first step, field
initialization and the particle startup kick both use that first clipped width,
so seeded fields and particles share the correct half-time convention.

With ``--write-every N`` the same per-step arrays are also emitted as
self-contained ``out_<step>.npz`` files alongside the end-of-run ``out.npz``.

Worked example
--------------

``examples/square_toroid_pic/`` runs a quasineutral H⁺ + μ⁻ plasma in the
equatorial slice of the square-cross-section toroidal magnet from
``examples/square_toroid``. See ``examples/square_toroid_pic/README.md``
for the physics rationale and reference numbers.

Postprocessing
--------------

``quasar.pic.postprocess`` renders field heatmaps and per-species
particle scatter from an ``out.npz``. It reads the archived ``plane`` metadata:
for Cartesian archives, ``plane: xy`` uses the ``(x, y, z)`` frame, while
``plane: xz`` labels the axes as ``(x, z)`` and the stored right-handed
component slots as ``(x, z, -y)``. Cylindrical archives retain their physical
``(r, z, phi)`` labels for either lab-plane embedding. Legacy archives without
``plane`` default to ``xy``.

.. code-block:: bash

   PYTHONPATH=build/hip-gfx942-release/python \\
     python -m quasar.pic.postprocess examples/square_toroid_pic/out.npz

matplotlib is imported lazily; install it locally if you want plots.
