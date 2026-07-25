Ideal-MHD simulation
====================

Quasar's ideal-MHD slice solves the 2D conservative ideal-MHD equations with a
finite-volume scheme and constrained transport (CT) that keeps the magnetic
field divergence-free. On Cartesian grids, MP5/MP7 reconstruct high-order face
states from conserved cell averages before HLLD fluxes are differenced over each
control volume. It is driven from the ``quasar.mhd`` Python front-end.

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
     reconstruction: mp7        # Cartesian; cylindrical requires muscl_minmod
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

The automatic ``troubled_cell`` controller preserves the mathematical open
admissible set

.. math::

   \rho > 0, \qquad p > 0.

If an SSP-RK stage leaves this set, the entire conservative stage is discarded
and retried with a smaller substep and a first-order HLL anchor. It never clamps
one cell's mass or adds energy to repair its pressure.

``numerics.rho_floor`` and ``numerics.p_floor`` are thresholds retained for the
explicit low-level repair API; they are **not** invariant bounds enforced by
conservative time evolution. In particular, an admissible expanding solution
may have
``0 < rho < rho_floor`` or ``0 < p < p_floor``. The CLI does not automatically
repair an initial condition to these thresholds either: initial density and gas
pressure must already be finite and strictly positive. This distinction avoids
the non-conservative mass and energy injection caused by applying a positive
floor after every stage.

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
MUSCL/MP5/MP7 Cartesian halos). ``state_bz`` is already cell-centred. The C++
readback aliases ``bx_face`` and ``by_face`` remain available when the raw
staggered arrays are required.

Geometry
--------

``geometry: cylindrical`` runs the axisymmetric ``(r, z)`` form: the x-axis is
the radius ``r`` (with ``origin_x_m`` the inner radius, ``0`` to include the
axis) and the y-axis is ``z``. Radial fluid fluxes use exact annular face/volume
weights; only the radial/azimuthal curvature stresses remain as point sources.
Constrained transport uses the matching annular curl and preserves
``(1/r) d(r B_r)/dr + dB_z/dz`` at round-off. A domain with ``origin_x_m > 0``
is an annulus and uses an ordinary physical x-low boundary; only ``r=0`` uses
the ``axis`` parity boundary.

.. important::

   Cylindrical runs currently require
   ``numerics.reconstruction: muscl_minmod`` and are second-order in space.
   Their radial conserved values are ring-volume averages with measure
   :math:`r\,dr\,dz`; the Cartesian MP5/MP7 coefficients assume the uniform
   measure :math:`dr\,dz`. Both the C++ constructor and Python deck validator
   reject MP5/MP7 in cylindrical geometry instead of silently reporting a
   design order that the radial metric treatment does not attain.

For a static background ("guide") field ``B = B0 + b``, see
:doc:`mhd_background_field`.
