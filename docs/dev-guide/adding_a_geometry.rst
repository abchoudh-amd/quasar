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

   Cylindrical supports ``fdtd_order: 2`` and ``4``. The fourth-order radial
   derivative acts on the conservative flux ``q=rA``; the even parity of ``q``
   supplies a regular axis row without evaluating ``1/r`` at zero.

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

The stability bound also belongs here. ``cyl_cfl_dt(g, fdtd_order, c)`` is
phrased separately from ``cfl_dt(g, fdtd_order, c)`` so callers select it
explicitly in cylindrical mode. Order two has the corresponding Cartesian
Courant limit over ``(dr, dz)``. At order four the regular-axis row is not
exactly Fourier-diagonal, so reusing the Cartesian ``7/6`` radial factor is
slightly unsafe. For example, the three-cell axis/outer-PEC radial operator has
``rho(-A4 B4) = 5.4444536388582545 / dr^2``, just above ``49/9``; exactly,
``det((49/9) I - (-A4 B4)) = -7/69120``.

The all-grid proof used by the helper is

.. math::

   \lVert A_4\rVert_\infty \le {5\over 2\,\Delta r},\qquad
   \lVert B_4\rVert_\infty \le {7\over 3\,\Delta r},

so ``rho(-A4 B4) <= 35/(6 dr^2)``. The axial symbol remains
``49/(9 dz^2)`` and leapfrog therefore uses the conservative bound

.. math::

   \Delta t \le {1\over c\sqrt{35/(24\,\Delta r^2)
                                  +49/(36\,\Delta z^2)}}.

This is a proved operator-norm bound, not an empirical margin. Both native and
Python helpers evaluate it after scaling by the smaller spacing so extreme
aspect ratios do not overflow intermediate inverse-spacing squares.

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

* ``YeeFdtdCyl2D<2>`` / ``YeeFdtdCyl2D<4>``,
* ``BorisCylPusher<1>`` / ``BorisCylPusher<2>``,
* ``EsirkepovCyl2D<1>`` / ``EsirkepovCyl2D<2>``.

They are registered under deck-facing names next to the Cartesian ones::

   QUASAR_REGISTER_FIELD_SOLVER("yee_cyl_o2", ::quasar::numerics::YeeFdtdCyl2D<2>)
   QUASAR_REGISTER_FIELD_SOLVER("yee_cyl_o4", ::quasar::numerics::YeeFdtdCyl2D<4>)
   QUASAR_REGISTER_PUSHER("boris_cyl_cic", ::quasar::numerics::BorisCylPusher<1>)
   QUASAR_REGISTER_PUSHER("boris_cyl_tsc", ::quasar::numerics::BorisCylPusher<2>)
   QUASAR_REGISTER_DEPOSIT("esirkepov_cyl_cic", ::quasar::numerics::EsirkepovCyl2D<1>)
   QUASAR_REGISTER_DEPOSIT("esirkepov_cyl_tsc", ::quasar::numerics::EsirkepovCyl2D<2>)

**Geometry-aware name builders.** As with the order/shape vocabulary, the
deck never builds these registry strings itself. The free functions
``resolve_scheme_family`` (anonymous namespace in ``pic_solver.cpp``) resolves a
geometry + order/shape to one set of registry names. ``is_cylindrical(geometry)``
switches to the ``"_cyl_"`` family (``"yee_cyl_o" + fdtd_order``,
``"boris_cyl_" + shape``, ``"esirkepov_cyl_" + shape``); a Cartesian run keeps
the existing names verbatim.
``EmPic2D3V``'s constructor then calls ``Registry<...>::create(name)`` for each,
so there is no ``if/else`` ladder over geometry in the driver. A new geometry's
names appear in exactly one place: register them, then add a branch to these
three builders.

.. warning::

   **Linker gotcha — keep registration sources in a ``REGISTERS`` module.** A
   namespace-scope static registration carries no externally referenced symbol,
   so a plain static-archive link can drop it and make ``Registry::create`` fail
   on a valid deck name. The ``pic`` module is therefore declared with
   ``quasar_add_module(pic REGISTERS ...)`` in
   ``src/physics/pic/CMakeLists.txt``. The helper wraps it in
   ``$<LINK_LIBRARY:WHOLE_ARCHIVE,...>`` (see
   ``cmake/QuasarAddModule.cmake``), just like the boundary module. Scheme
   registrations may share ``pic_solver.cpp`` or move to another source, but
   every such source must remain part of that registration-bearing module.

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
  ``_validate_cylindrical`` requires a non-negative inner radius and a
  non-periodic outer-radius (``x_hi``) wall for fields and particles. At
  ``origin_x_m == 0`` the default ``x_lo`` side is auto-replaced by the regular
  axis condition. At positive origin (an annulus), both ``x_lo`` boundaries must
  instead be explicitly non-periodic. Orders 2 and 4 are supported; order four
  requires at least two physical cells in each dimension because a one-cell
  non-periodic domain has overlapping two-layer ghost sources.
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
     fdtd_order: 4    # 2 and 4 are supported
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
