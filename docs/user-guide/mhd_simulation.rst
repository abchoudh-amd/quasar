Ideal-MHD simulation
====================

Quasar's ideal-MHD slice solves the 2D conservative ideal-MHD equations with a
finite-volume scheme and constrained transport (CT) that keeps the magnetic
field divergence-free. MP5/MP7 reconstruct high-order face states from
conserved cell averages before HLLD fluxes are differenced over each control
volume. Cartesian grids use uniform-measure coefficients; cylindrical grids use
radius-weighted coefficients along ``r``. It is driven from the ``quasar.mhd``
Python front-end.

Running a deck
--------------

.. code-block:: console

   $ PYTHONPATH=build/hip-gfx942-release/python \
       python -m quasar.mhd.cli run examples/orszag_tang/input.yaml

The CLI loads a YAML deck, builds the seeded solver, advances it, and writes the
diagnostics to the deck's ``diagnostics.output_path`` (``out.npz`` by default).
Useful flags: ``--print-config`` (echo the resolved deck and timestep before
running), ``--steps-override N`` (cap the step count for a smoke run),
``--log-every N`` (print progress and ``|div B|`` every ``N`` steps).

Deck structure
--------------

.. code-block:: yaml

   geometry: cartesian          # or "cylindrical" for axisymmetric (r, z)
   domain:
     nx: 256
     ny: 256
     lx_m: 1.0
     ly_m: 1.0
     origin_x_m: 0.0
     origin_y_m: 0.0
   numerics:
     gamma: 1.6666666667        # adiabatic index
     cfl: 0.4                   # CFL safety factor
     reconstruction: mp7        # high order in Cartesian or cylindrical geometry
     riemann: hlld
     integrator: ssprk3
     ct: fd_ct_christlieb
     positivity: troubled_cell
   time:
     dt_s: auto                 # "auto" = CFL-limited each step, or a fixed float
     steps: 500
   initial:
     type: orszag_tang          # brio_wu | mhd_blast | mhd_rotor | alfven_wave | ...
   boundary:
     fluid: [periodic, periodic, periodic, periodic]   # [x_lo, x_hi, y_lo, y_hi]
     field: [periodic, periodic, periodic, periodic]
   diagnostics:
     output_path: out.npz
     cadence: 50                # snapshot every N steps (0 = final only)
     fields: [rho, energy, bx, by, bz]

Every scheme axis (``reconstruction``, ``riemann``, ``integrator``, ``ct``,
``positivity``, and the per-side ``boundary`` names) is a registry string, so a
newly registered scheme is selectable without code changes. See the dev-guide
pages under ``docs/dev-guide`` for how to add one.

Positivity and configured floors
--------------------------------

The automatic ``troubled_cell`` controller makes the mathematical open
admissible set a safety condition for every state it accepts:

.. math::

   \rho > 0, \qquad p > 0.

If an SSP-RK stage leaves this set, the entire conservative stage is discarded
and retried with a smaller substep and a first-order HLL anchor. It never clamps
one cell's mass or adds energy to repair its pressure. If subdivision cannot
make a positive, representable conservative advance, the call raises an error
and restores the state from the start of the requested step.

This is a safety contract, not a guarantee of progress for every admissible but
under-resolved state. In particular, the complete cylindrical operator with
constrained transport and a spatially varying background field has no claimed
invariant-domain proof. A sufficiently sharp, very-low-beta discontinuity can
therefore exhaust representable subdivision. Refine the grid or rescale the
problem into a resolved regime rather than relying on a hidden pressure floor.

``numerics.rho_floor`` and ``numerics.p_floor`` are thresholds retained for the
explicit low-level repair API; they are **not** invariant bounds enforced by
conservative time evolution. In particular, an admissible expanding solution
may have
``0 < rho < rho_floor`` or ``0 < p < p_floor``. The CLI does not automatically
repair an initial condition to these thresholds either: initial density and gas
pressure must already be finite and strictly positive. This distinction avoids
the non-conservative mass and energy injection caused by applying a positive
floor after every stage.

Live-state solenoidality and mutable views
------------------------------------------

Every externally seeded or imported face field is checked with the strict
directional divergence predicate before the CFL or residual kernels consume
it. Cartesian DC offsets cancel before normalization. Its local 1024-face-ULP
forward-error allowance is available only when two nonzero, opposite-sign
directional contributions genuinely cancel; a one-direction or same-sign
one-ULP defect is rejected even on a very large field.

A successfully accepted internal CT/SSP-RK update has additional provenance:
in exact arithmetic it inherits the already-proved divergence, while storing
the two face components independently can leave a few ULPs of residual and can
round one directional term to zero. Only for such a solver-owned state, the
preflight admits a residual within 1024 metric-weighted face-storage ULPs
without requiring both represented directional terms to remain nonzero. This
permission is never inferred from field values. Seeding any component revokes
it until another complete internal update succeeds. Obtaining a mutable C++
``state()``, ``rk_register()``, or ``residual_register()`` view revokes it
permanently for that solver instance, because the retained device-buffer handle
can be written after any later preflight. Failed steps restore both the state
buffers and their request-start provenance.

Output
------

The ``.npz`` carries the final cell-centered state (``state_rho``,
``state_mx``, ..., ``state_bz``), the seeded ``t = 0`` profile
(``state_<name>_initial``), the per-snapshot ``div B`` series (``divb_linf``)
and the post-run scalar ``divb_linf_final``, plus the grid metadata
(``nx``, ``ny``, ``nghost``, ``geometry``, ``gamma``). With ``cadence > 0`` the
intermediate snapshots are written under the ``snapshot_*`` keys.

The evolved in-plane magnetic components are staggered CT face averages:
``bx_face(i,j)`` is the low-x face and ``by_face(i,j)`` the low-y face of a
cell. ``state_bx`` and ``state_by`` are not raw face dumps. They use the same
finite-volume face-to-cell quadrature as the equation of state and split-energy
update (fourth-, sixth-, or eighth-order on the automatically sized
MUSCL/MP5/MP7 halos, with radial weighting in cylindrical geometry).
``state_bz`` is already cell-centred. The C++
readback aliases ``bx_face`` and ``by_face`` remain available when the raw
staggered arrays are required.

Geometry
--------

``geometry: cylindrical`` runs the axisymmetric ``(r, z)`` form: the x-axis is
the radius ``r`` (with ``origin_x_m`` the inner radius, ``0`` to include the
axis) and the y-axis is ``z``. Radial fluid fluxes use exact discrete weights
chosen per equation (annular ring-volume for the mass-like variables, an
:math:`r^2` angular-momentum moment for :math:`m_\phi`, and no radial weight for
:math:`B_\phi` — see "Discrete radial measures" below); only the
radial/azimuthal curvature stresses remain as point sources.
Constrained transport uses the matching annular curl and preserves
``(1/r) d(r B_r)/dr + dB_z/dz`` at round-off. A domain with ``origin_x_m > 0``
is an annulus and uses an ordinary physical x-low boundary; only ``r=0`` uses
the ``axis`` parity boundary.

.. important::

   Cylindrical MP5/MP7 apply radius-dependent finite-volume moments to every
   stencil that runs along :math:`r`; stencils along :math:`z` retain the
   Cartesian coefficients because the volume measure factors as
   :math:`r\,dr\,dz`. This includes fluid reconstruction, magnetic collocation,
   transverse quadrature, and constrained-transport corner interpolation. The
   ``r=0`` axis uses the same weighted rows with the parity-filled ghost cells.
   Selecting MP5 or MP7 automatically expands the reconstruction halo to three
   or four cells, respectively.

Discrete radial measures
~~~~~~~~~~~~~~~~~~~~~~~~~

The cylindrical conserved variables are **not** all averages under a single
:math:`r\,dr\,dz` measure. Each equation is discretized with its own natural
measure, chosen so that the quantity the equation actually conserves telescopes
exactly (``src/backend/hip/mhd/mhd_update.hip``):

* **Mass-like variables** (:math:`\rho`, axial momentum :math:`m_z`, energy) use
  the **annular ring-volume average** with measure :math:`r\,dr`. Their radial
  residual is the ring-flux divergence
  :math:`[r_{hi}F_{hi} - r_{lo}F_{lo}] / \int r\,dr`, evaluated in the factored
  form :math:`(F_{hi}-F_{lo})/dr + (F_{hi}+F_{lo})/(2r_c)` so it avoids
  differences of squared radii and stays regular in the :math:`r=0` axis cell.

* **Azimuthal momentum** :math:`m_\phi` is stored as a piecewise-constant annular
  cell value, but the conserved integral is **angular momentum**
  :math:`\int r^2 m_\phi\,dr`. Its residual therefore uses that :math:`r^2`
  moment directly, rather than an annular divergence plus a pointwise
  :math:`-F/r` source, so it telescopes exactly under the angular-momentum
  weight.

* **Radial momentum** :math:`m_r` is excluded from the generic annular
  divergence and handled by a dedicated kernel that rounds its tensor
  derivatives, its pressure-free curvature difference, and any static-background
  stress together — the curvature cancellation must happen before rounding.

* **Toroidal field** :math:`B_\phi` obeys the **metric-free** point equation
  :math:`\partial_t B_\phi + \partial_r F_{B_\phi} = 0` and so uses an ordinary
  radial face difference with no :math:`r` weighting at all.

These component-specific measures are internally consistent, and each
cylindrical reconstruction order is conservative in every variable's own
invariant. Do not assume a single uniform :math:`r\,dr\,dz` average when
post-processing or when adding a cylindrical source term.

For a static background ("guide") field ``B = B0 + b``, see
:doc:`mhd_background_field`.
