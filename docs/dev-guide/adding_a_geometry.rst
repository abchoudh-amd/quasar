Adding a new grid geometry
==========================

The EM-PIC module ships two grid geometries: the default Cartesian ``(x, y)``
grid and an axisymmetric cylindrical ``(r, z)`` (``m = 0``) grid. This page uses
the cylindrical mode as a worked example so a developer can add a third geometry
by following the same path end-to-end.

A geometry is selected by the deck's top-level ``geometry`` key
(``cartesian`` | ``cylindrical``; see ``python/quasar/pic/io.py``). Unlike the
conductor-geometry generators of the magnetostatics module (which are plain C++
free functions discovered only at the YAML boundary), a PIC grid geometry cuts
across all four architectural axes — ``physics × numerics × boundary ×
backend`` — because it changes the discrete curl operators, the deposit/gather
weights, the on-axis boundary closure, and the CFL bound. The walk-through tracks
each axis in turn.

.. note::

   Cylindrical ships at ``fdtd_order: 2`` only. The on-axis regularised closure
   is derived for the 2nd-order staggered Yee curl; an order-4 cylindrical axis
   closure is out of scope. ``PicDeck._validate_cylindrical`` rejects
   ``fdtd_order`` 4, and the field-solver name builder hard-codes
   ``"yee_cyl_o2"`` (see Steps 2 and 3 below).

Where each piece lives
----------------------

A geometry touches one tree per axis:

* **backend** — the device kernels under ``src/backend/hip/pic/``. Cylindrical
  adds ``fdtd_b_cyl_hip.hip``, ``fdtd_e_cyl_hip.hip``, ``deposit_cyl_hip.hip``,
  ``gather_push_cyl_hip.hip`` and ``boundary_axis_hip.hip``, exposing five new
  ``launch_pic_*_cyl_*`` / ``launch_pic_boundary_axis_*`` entry points declared
  once in ``include/quasar/physics/pic/kernels.hpp``.
* **numerics** — the concrete scheme classes and their registry registrations in
  ``src/physics/pic/pic_solver.cpp`` (the field solver, pusher, and deposit).
* **boundary** — the on-axis ``r = 0`` boundary condition in
  ``src/boundary/axis.{hpp,cpp}``.
* **core** — the geometry-aware grid accessors in
  ``include/quasar/core/grid.hpp`` (radius/volume helpers and the cylindrical
  CFL bound).
* **deck/CLI** — the ``geometry`` key, validation, CFL selection, and npz
  metadata in ``python/quasar/pic/io.py`` and ``python/quasar/pic/cli.py``.

Step 1 - Grid accessors in ``core/grid.hpp``
--------------------------------------------

A new geometry typically reinterprets the existing ``Grid2D`` axes rather than
introducing a new grid type. In cylindrical mode the x-axis *is* the radius
``r`` (``origin_x`` is the inner radius, ``dx()`` is ``dr``) and the y-axis is
the axial coordinate ``z``. ``Grid2D`` gains ``QUASAR_HOST_DEVICE`` helpers that
mirror the Cartesian ``x_at_*`` accessors so device kernels read the radius with
no new type:

* ``r_at_cell_center(i)`` — node radius ``origin_x + (i + 0.5) * dr``.
* ``r_at_edge(i)`` — lower-face radius ``origin_x + i * dr`` (so the ``i = 0``
  edge sits at ``r = 0`` when the domain starts on the axis).
* ``cell_volume(i)`` — the ``m = 0`` ring volume ``2*pi * r_at_cell_center(i) *
  dr * dz`` used by the deposit's radius-weighted current.

These are harmless (but meaningless) on a Cartesian run, which never calls them.

The stability bound also belongs here. ``cyl_cfl_dt(g, c)`` is phrased
separately from ``cfl_dt(g, fdtd_order, c)`` so callers select it explicitly in
cylindrical mode; for the on-axis-regularised 2nd-order scheme the bound is the
Cartesian 2nd-order Courant limit over ``(dr, dz)`` (the ``1/r`` curl terms do
not tighten it), and the separate entry point gives a future radius-dependent
refinement one home.

Step 2 - Backend kernels and the ABI
------------------------------------

Each ``launch_pic_*`` entry point is declared exactly once in
``include/quasar/physics/pic/kernels.hpp`` and included both by its ``.hip``
definition (so a signature drift is a compile error) and by every caller in the
physics/numerics/boundary layers. The cylindrical kernels reuse the Cartesian
pointer/stream ABI verbatim; only the discretisation differs.

**Component slot convention.** The stored Yee vector components map
``(x, y, z) -> (r, z, phi)``: ``ex -> Er``, ``ey -> Ez``, ``ez -> Ephi`` (and
likewise ``bx -> Br``, ``by -> Bz``, ``bz -> Bphi``). Particle slots map
``x -> r``, ``y -> z`` for position and ``vx -> vr``, ``vy -> vz``,
``vz -> vphi`` for velocity. The axial field and velocity therefore live in the
``ey`` / ``by`` / ``vy`` slots. Keeping the same six-component storage means the
gather and the Boris half-rotation are unchanged — the local
``(e_r, e_z, e_phi)`` frame is orthonormal — and only the position advance picks
up the azimuthal rotation of the ``(vr, vphi)`` pair across ``r``
(``gather_push_cyl_hip.hip``).

**On-axis closure (the subtle correctness point).** The axisymmetric ``Ez``/
``Bz`` curls use the radial-flux operator ``(1/r) d(r A_phi)/dr`` in
finite-volume form. At node ``i`` the forward flux is::

   (1 / r_c(i)) * ( r_e(i+1) * Bphi(i+1) - r_e(i) * Bphi(i) ) / dr

with ``r_c(i) = r_at_cell_center(i)`` and ``r_e(i) = r_at_edge(i)``. On the axis
(``i = 0``) the lower face sits at ``r_e(0) = 0``, so the inner-face term
*vanishes identically* — this **is** the regularised on-axis closure, with no
l'Hopital ``4*value/dr`` factor. The natural zero-radius flux converges to the
analytic Bessel (``TM0n0``) spectrum at 2nd order, whereas the earlier
``4*Bphi(0)/dr`` hack injected a resolution-independent eigenvalue shift that
broke convergence. The odd components (``Er``, ``Ephi``, ``Br``, ``Bphi``)
vanish on the axis by parity and are pinned to zero.

The deposit and Faraday updates pair a **backward** radial difference with this
**forward** flux read; the two are discrete adjoints, which is what makes the
composed operator the correct cylindrical Bessel operator and keeps the deposit
charge-consistent (``radial_flux_fwd`` / ``ddr_bwd`` in ``fdtd_e_cyl_hip.hip``).
The deposit weights are proportional to ``cell_volume(i)`` so the discrete
cylindrical continuity residual stays in tolerance.

Step 3 - Schemes and registration in ``pic_solver.cpp``
-------------------------------------------------------

The cylindrical schemes are the axisymmetric counterparts of the Cartesian
``YeeFdtd2D<2>`` / ``BorisPusher<S>`` / ``Esirkepov2D<S>``. They implement the
same ``quasar::numerics::IFieldSolver`` / ``IParticlePusher`` /
``IDepositScheme`` interfaces and forward to the cylindrical launch ABI:

* ``YeeFdtdCyl2D`` (2nd order only),
* ``BorisCylPusher<1>`` / ``BorisCylPusher<2>``,
* ``EsirkepovCyl2D<1>`` / ``EsirkepovCyl2D<2>``.

They are registered under deck-facing names next to the Cartesian ones::

   QUASAR_REGISTER_FIELD_SOLVER("yee_cyl_o2", ::quasar::numerics::YeeFdtdCyl2D)
   QUASAR_REGISTER_PUSHER("boris_cyl_cic", ::quasar::numerics::BorisCylPusher<1>)
   QUASAR_REGISTER_PUSHER("boris_cyl_tsc", ::quasar::numerics::BorisCylPusher<2>)
   QUASAR_REGISTER_DEPOSIT("esirkepov_cyl_cic", ::quasar::numerics::EsirkepovCyl2D<1>)
   QUASAR_REGISTER_DEPOSIT("esirkepov_cyl_tsc", ::quasar::numerics::EsirkepovCyl2D<2>)

**Geometry-aware name builders.** As with the order/shape vocabulary, the
deck never builds these registry strings itself. The free functions
``field_solver_name``, ``pusher_name`` and ``deposit_name`` (anonymous namespace
in ``pic_solver.cpp``) resolve a geometry + order/shape to a registry name:
``is_cylindrical(geometry)`` switches to the ``"_cyl_"`` family
(``"yee_cyl_o2"`` regardless of ``fdtd_order``, ``"boris_cyl_" + shape``,
``"esirkepov_cyl_" + shape``); a Cartesian run keeps the existing names verbatim.
``EmPic2D3V``'s constructor then calls ``Registry<...>::create(name)`` for each,
so there is no ``if/else`` ladder over geometry in the driver. A new geometry's
names appear in exactly one place: register them, then add a branch to these
three builders.

.. warning::

   **Linker gotcha — keep the registrations in this TU.** The ``pic`` module is
   plain-linked (``quasar_add_module(pic ...)`` in
   ``src/physics/pic/CMakeLists.txt``), *not* whole-archived. A namespace-scope
   static registration carries no externally referenced symbol, so a plain
   static-archive link would drop a standalone TU's registrations and
   ``Registry::create`` would throw on the deck name. The cylindrical scheme
   *definitions and registrations therefore live in*
   ``src/physics/pic/pic_solver.cpp`` — the same translation unit as the
   externally-referenced ``EmPic2D3V`` symbols — so the static initializers are
   pulled in and survive the link. Do **not** split them into a new ``.cpp``.
   The alternative is the boundary module's approach: it is declared with
   ``quasar_add_module(boundary REGISTERS ...)``, and the ``REGISTERS`` flag
   wraps the target in ``$<LINK_LIBRARY:WHOLE_ARCHIVE,...>`` (see
   ``cmake/QuasarAddModule.cmake``) so every object is forced in. Use one or the
   other; the axis BC in Step 4 relies on the latter.

Step 4 - On-axis boundary condition
-----------------------------------

A geometry with a special edge needs a boundary closure. The cylindrical
``r = 0`` axis is handled by ``AxisFieldBC`` / ``AxisParticleBC`` in
``src/boundary/axis.{hpp,cpp}``, registered under the name ``"axis"``::

   QUASAR_REGISTER_FIELD_BOUNDARY("axis", AxisFieldBC)
   QUASAR_REGISTER_PARTICLE_BOUNDARY("axis", AxisParticleBC)

Because ``boundary`` is the ``REGISTERS`` (whole-archived) module, these survive
the link from their own TU. ``AxisFieldBC`` gates its closure to the
``Side::x_lo`` (``i = 0``) face and drives ``launch_pic_boundary_axis_fields``
both as a ghost fill and after each curl; ``AxisParticleBC`` reflects particles
that cross the axis (``r -> -r``, ``vr -> -vr``) so the approach is
reflectionless.

**Auto-wiring.** A cylindrical deck does not have to name the ``axis`` BC. The
``EmPic2D3V`` constructor detects ``is_cylindrical(cfg_.geometry)`` and
overrides the ``x_lo`` field and particle boundary to ``"axis"`` regardless of
what the deck set there (warning on a non-default, non-``axis`` override; the
default periodic ``x_lo`` is replaced silently). All other sides stay
deck-driven.

Step 5 - Deck schema, validation, CFL, and metadata
---------------------------------------------------

Wire the geometry into the Python deck layer (``python/quasar/pic``):

* **Schema** — ``PicDeck`` carries ``geometry: str = "cartesian"``, parsed from
  the top-level ``geometry`` key in ``parse()``. ``EmPicConfig.geometry`` is set
  from it in ``cli._make_solver``.
* **Validation** — ``PicDeck.validate`` rejects unknown geometries;
  ``_validate_cylindrical`` enforces the geometry-specific rules: a
  non-negative inner radius (``domain.origin_x_m >= 0``), ``fdtd_order == 2``,
  and a non-periodic outer-radius (``x_hi``) wall for both field and particle
  boundaries. The inner radius (``x_lo``) is deliberately left alone so the
  default periodic side can be auto-replaced by the C++ axis condition.
* **CFL selection** — ``cli.prepare_run`` branches on ``geometry`` to pick
  ``cyl_cfl_dt`` / ``cyl_cfl_limit`` (from ``python/quasar/pic/numerics.py``)
  for both the ``"auto"`` timestep and the explicit-``dt`` stability check,
  flooring the inner radius to half a cell when the domain touches the axis.
* **Metadata** — the npz snapshot persists ``geometry`` (alongside ``plane``)
  in ``cli._snapshot`` / ``_flatten_for_npz`` so offline readers know whether to
  treat the in-plane axes as ``(x, y)`` or axisymmetric ``(r, z)``.

After this, a deck of the form

.. code-block:: yaml

   geometry: cylindrical
   domain:
     nx: 128          # radial cells (r)
     ny: 256          # axial cells (z)
     lx_m: 0.05       # outer radius R
     ly_m: 0.1        # axial length
     origin_x_m: 0.0  # inner radius (0 = domain touches the axis)
   numerics:
     fdtd_order: 2    # cylindrical ships at order 2 only
     shape: tsc
   boundary:
     field:    [periodic, pec, periodic, periodic]   # x_lo auto-replaced by 'axis'
     particle: [periodic, specular, periodic, periodic]

runs the cylindrical schemes through ``quasar pic run`` with the on-axis closure
auto-wired on the ``r = 0`` side.

Step 6 - Tests and the worked example
-------------------------------------

Anchor a new geometry with both a numerical-convergence check (cylindrical
converges to the analytic ``TM0n0`` Bessel spectrum at 2nd order, the payoff of
the natural on-axis closure) and deck round-trip / validation tests in
``tests/python``. If the geometry is worth a worked example, drop a deck into
``examples/<name>/`` with a ``README.md`` and an integration test in
``tests/python/test_examples.py``.

Checklist
---------

A merge-ready change to add a grid geometry touches all of:

#. ``include/quasar/core/grid.hpp`` (geometry-aware accessors + CFL bound)
#. ``src/backend/hip/pic/`` (the device kernels) and
   ``include/quasar/physics/pic/kernels.hpp`` (their ABI)
#. ``src/physics/pic/pic_solver.cpp`` (scheme classes, registrations, name
   builders, and any constructor auto-wiring — keep registrations in this TU)
#. ``src/boundary/axis.{hpp,cpp}`` or an equivalent edge BC (whole-archived
   ``REGISTERS`` module)
#. ``python/quasar/pic/io.py`` (schema + validation)
#. ``python/quasar/pic/cli.py`` and ``python/quasar/pic/numerics.py``
   (CFL selection + npz metadata)
#. ``tests/python`` (and ``examples/<name>/`` + ``test_examples.py`` for a
   worked example)
