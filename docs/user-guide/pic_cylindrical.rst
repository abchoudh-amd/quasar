Cylindrical (r, z) axisymmetric PIC
===================================

The PIC module can run in an **axisymmetric** ``m = 0`` mode, in which the 2-D
grid represents the meridional ``(r, z)`` plane of a body of revolution rather
than a Cartesian ``(x, y)`` slice. This is the natural setting for problems with
azimuthal symmetry: solenoids and magnetic mirrors, pillbox/resonant cavities,
and axially symmetric beams. The angular coordinate ``φ`` is assumed uniform, so
a single ``(r, z)`` plane captures the full 3-D physics at a fraction of the
cost.

Cylindrical mode is selected per deck with a single key and reuses the same
``quasar.pic`` CLI as the Cartesian workflow described in
:doc:`pic_simulation`::

   PYTHONPATH=build/hip-gfx942-release/python \\
     python -m quasar.pic.cli run examples/cyl_cavity_tm010/input.yaml

Selecting cylindrical geometry
------------------------------

The top-level ``geometry`` key chooses the coordinate system of the 2-D grid:

.. code-block:: yaml

   geometry: cartesian   # default — grid axes are (x, y) in the chosen plane
   geometry: cylindrical # axisymmetric r-z meridional plane

It accepts only ``cartesian`` (the default) or ``cylindrical``; any other value
is rejected at deck load.

Axis mapping
------------

In cylindrical mode the grid axes are remapped:

* The horizontal grid axis ``i`` is the **radius** ``r``. Its extent comes from
  the usual ``domain.nx`` / ``domain.lx_m`` / ``domain.origin_x_m`` keys, so the
  domain spans ``r ∈ [origin_x_m, origin_x_m + lx_m]``.
* The vertical grid axis ``j`` is the **axial** coordinate ``z``, from
  ``domain.ny`` / ``domain.ly_m`` / ``domain.origin_y_m``.
* The symmetry axis ``r = 0`` is the ``i = 0`` edge (the ``x_lo`` side).

Because ``origin_x_m`` is the inner radius, it must be ``>= 0`` — a negative
radius is unphysical and is rejected. Both worked examples below set
``origin_x_m: 0.0`` so the radial domain starts on the axis (the natural choice
for the axisymmetric ``m = 0`` on-axis closure); a positive value models an
annular region offset from the axis.

On-axis boundary
----------------

The ``r = 0`` side (``x_lo``) is **always** the on-axis condition in cylindrical
mode, regardless of what the deck specifies there. The solver replaces both the
field and particle ``x_lo`` boundary with its registered ``axis`` closure. If the
deck set a non-default, non-``axis`` value on ``x_lo``, the solver prints a
warning that it is being overridden; a default (periodic) ``x_lo`` — the no-op
normal case — is replaced silently. You may set ``x_lo`` to ``pec`` /
``absorbing`` purely for documentation, knowing the value is a don't-care.

The remaining sides are deck-driven:

* ``x_hi`` is the **outer radius** wall. A *periodic* outer radius is unphysical
  for an axisymmetric domain and is rejected at deck load (for both the field and
  particle boundary). Since the boundary config defaults every side to
  ``periodic``, a cylindrical deck must set ``x_hi`` (and typically all sides)
  explicitly.
* ``y_lo`` / ``y_hi`` are the two **axial** walls and take any supported field /
  particle boundary kind.

.. code-block:: yaml

   boundary:
     field:    {x_lo: pec, x_hi: pec, y_lo: pec, y_hi: pec}
     particle: {x_lo: specular, x_hi: specular, y_lo: specular, y_hi: specular}

Specifying fields by physical component (recommended)
-----------------------------------------------------

In cylindrical mode you specify fields by their **physical cylindrical axes**, so
you never have to know how the solver stores them internally. There are two
places this matters: the uniform external field and the initial field seed.

**Uniform external field — ``B_rzphi``.** Give the magnetic field in physical
``(r, z, φ)`` order — the same ordering as the velocity triad below — with the
``B_rzphi`` key. A pure **axial** field is the second (``B_z``) entry:

.. code-block:: yaml

   external_field:
     evaluator: {type: uniform, B_rzphi: [0.0, 1.0, 0.0]}   # B_z = 1.0 T, axial

``B_rzphi`` is cylindrical-only (using it on a Cartesian deck is an error) and
the deck loader translates it to the solver's storage layout in one place
(``quasar.pic.io._rzphi_to_uniform_lab``), accounting for the ``plane`` you
chose.

**Initial field seed — physical component names.** In cylindrical mode the
``fields.initial.component`` names ``Er`` / ``Ez`` / ``Ephi`` (and ``Br`` /
``Bz`` / ``Bphi``) are interpreted as physical axes: ``Ez`` means the **axial**
electric field, ``Ephi`` the **azimuthal** one. For example, to excite the axial
field of a cavity:

.. code-block:: yaml

   fields:
     initial: {type: seed_perturbation, component: Ez, mode: [1, 0], amplitude: 1.0}

**Particle velocities** are likewise physical: the seeded ``drift_v`` triad maps
as ``vx → vr``, ``vy → vz`` (axial), ``vz → vφ``, so a pure radial drift is
``drift_v: [v, 0.0, 0.0]``.

.. note::

   **For the curious / backward-compatible raw spelling.** Under the hood the
   solver stores fields in Cartesian-named slots via the grid triad
   (``i = r``, ``j = z``, out-of-plane ``= φ``), so physically the radial
   component lands in the ``ex`` / ``bx`` slot, the **axial** component in the
   ``ey`` / ``by`` slot, and the azimuthal component in the ``ez`` / ``bz`` slot.
   You can still write the raw slot vector directly — ``B_T`` for the external
   field (for ``plane: xy`` an axial field is ``B_T: [0.0, 1.0, 0.0]``, i.e. the
   ``by`` slot) and a raw component name such as ``Ey`` for the seed — and it is
   accepted unchanged for backward compatibility. But this requires knowing the
   "axial lives in the second slot" convention, which is exactly the footgun the
   physical spelling removes. Supplying both ``B_rzphi`` and ``B_T`` is an error.
   The recorded output uses the storage-slot names (see `Output`_ below).

Numerics and time step
----------------------

* ``numerics.fdtd_order`` must be ``2`` in cylindrical mode. The order-4
  cylindrical axis closure is not implemented yet, so an order-4 cylindrical deck
  is rejected at load.
* When ``time.dt_s: auto``, a cylindrical deck uses a **cylindrical** CFL limit
  (which accounts for the inner radius) rather than the Cartesian one. Supply a
  fixed ``dt_s`` if you want to override the automatic value.

Output
------

The output ``out.npz`` records the deck's coordinate system under the top-level
``geometry`` key (a 1-element array holding ``"cartesian"`` or
``"cylindrical"``), alongside the field, particle, and series keys documented in
:doc:`pic_simulation`. Field components are recorded under their storage-slot
names (``field_ey`` for the axial ``Ez``, ``field_by`` for the axial ``Bz``, and
so on), following the slot mapping noted above.

Worked example: TM010 cavity resonance
---------------------------------------

``examples/cyl_cavity_tm010/`` is a field-only deck (no particles) that recovers
the TM010 ("pillbox") mode of a cylindrical resonant cavity. The domain touches
the axis (``origin_x_m: 0.0``) with cavity radius ``R = lx_m = 0.10 m`` and axial
height ``H = ly_m = 0.10 m`` on a ``128 x 128`` grid. The inner-radius wall is
the symmetry axis; the outer radial wall and both axial walls are PEC:

.. code-block:: yaml

   geometry: cylindrical
   domain: {nx: 128, ny: 128, lx_m: 0.10, ly_m: 0.10, origin_x_m: 0.0, origin_y_m: 0.0}
   numerics: {fdtd_order: 2, shape: cic, current_filter: []}
   boundary:
     field:    {x_lo: pec, x_hi: pec, y_lo: pec, y_hi: pec}
     particle: {x_lo: specular, x_hi: specular, y_lo: specular, y_hi: specular}
   fields:
     initial: {type: seed_perturbation, component: Ez, mode: [1, 0], amplitude: 1.0}
   time: {dt_s: auto, steps: 1024}
   diagnostics: {output_path: out.npz, cadence: 4, fields: [ey], per_species: false}

The TM010 mode couples the axial ``Ez`` with the azimuthal ``Bφ``, so the deck
seeds the physical axial component ``Ez`` (which resolves to the ``ey`` storage
slot, hence the ``ey`` diagnostic). Its analytic eigenfrequency depends only on
the radius:

.. code-block:: text

   f_010 = j01 * c / (2 * pi * R),   j01 = 2.40483  (first zero of J0)

With ``R = 0.10 m`` this gives ``f_010 ≈ 1.1473 GHz``.

.. note::

   On a cylindrical grid, ``seed_perturbation`` does **not** seed a plain
   ``sin`` profile. It seeds the axisymmetric radial eigenmode
   ``J0(j_{0,mx} · r/R)`` (uniform in ``z``), where ``mx`` is ``mode[0]``. A plain
   ``sin(2π r/R)`` overlaps mostly with the TM020 mode and would ring at the
   wrong line; the Bessel seed excites the TM0,``mx``,0 mode cleanly
   (``mode: [1, 0]`` → TM010).

Run it and inspect the result:

.. code-block:: bash

   PYTHONPATH=build/hip-gfx942-release/python \\
     python -m quasar.pic.cli run examples/cyl_cavity_tm010/input.yaml

The axial-E snapshots (``snapshot_field_ey``, sampled every ``cadence = 4``
steps, with the final field in ``field_ey``) give a near-axis ``Ez(t)`` trace
whose dominant FFT peak should match ``f_010 ≈ 1.1473 GHz`` to within a few
percent, and a radial profile that matches ``J0(j01 · r/R)``. See
``examples/cyl_cavity_tm010/README.md`` for the full reference numbers and
post-processing recipe.

Worked example: electron gyro-orbit
-----------------------------------

``examples/cyl_gyro_orbit/`` pushes a small bunch of electrons in a uniform axial
magnetic field and checks the Larmor radius and gyrofrequency, including the
near-axis behaviour. The radial domain starts on the axis (``origin_x_m: 0.0``),
and the tiny Larmor radius keeps each guiding centre well inside its cell:

.. code-block:: yaml

   plane: xy
   geometry: cylindrical
   domain: {nx: 128, ny: 128, lx_m: 0.10, ly_m: 0.10, origin_x_m: 0.0, origin_y_m: -0.05}
   numerics: {fdtd_order: 2, shape: cic, current_filter: []}
   boundary:
     field:    {x_lo: pec, x_hi: pec, y_lo: pec, y_hi: pec}
     particle: {x_lo: absorbing, x_hi: absorbing, y_lo: absorbing, y_hi: absorbing}
   external_field:
     evaluator: {type: uniform, B_rzphi: [0.0, 1.0, 0.0]}   # B_z = 1.0 T, axial
   species:
     - name: electron
       charge_C: -1.602176634e-19
       mass_kg: 9.1093837015e-31
       n_particles: 64
       initial:
         distribution: maxwellian_uniform
         density_per_m3: 1.0e18
         temperature_eV: 1.0
         drift_v: [5.93e5, 0.0, 0.0]   # (vr, vz, vφ) — a pure vr perpendicular drift
   time: {dt_s: auto, steps: 512}
   diagnostics: {output_path: out.npz, cadence: 8, fields: [by], per_species: true}

The axial field is the *second* (``B_z``) entry of ``B_rzphi``, and the seeded
``drift_v`` is a pure ``vr`` velocity (``vx → vr``) perpendicular to that axial
field, giving a clean perpendicular speed ``v_perp ≈ 5.93e5 m/s``. The analytic
references are the cyclotron frequency and Larmor radius:

.. code-block:: text

   omega_c = |q| B / m  ≈ 1.7588e11 rad/s   (T_c = 2*pi/omega_c ≈ 3.573e-11 s)
   r_L     = m v_perp / (|q| B) ≈ 3.37e-6 m

Run it and check the trajectory:

.. code-block:: bash

   PYTHONPATH=build/hip-gfx942-release/python \\
     python -m quasar.pic.cli run examples/cyl_gyro_orbit/input.yaml

The per-species particle state (positions and velocities) is written under the
``species_electron_*`` keys, and the axial field is available as ``external_by``
/ ``field_by``. The orbit radius and period recovered from the trajectory should
match ``r_L`` and ``T_c`` to within a few percent, and the bunch kinetic energy
should stay bounded as guiding centres approach the axis (no spurious on-axis
energy gain). See ``examples/cyl_gyro_orbit/README.md`` for the detailed
reference signature.
