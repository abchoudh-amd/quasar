Ideal-MHD simulation
====================

Quasar's ideal-MHD slice solves the 2D conservative ideal-MHD equations with a
high-order finite-difference scheme and constrained transport (CT) that keeps
the magnetic field divergence-free. It is driven from the ``quasar.mhd`` Python
front-end.

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
     reconstruction: mp7        # muscl_minmod | mp5 | mp7
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

Output
------

The ``.npz`` carries the final cell-centered state (``state_rho``,
``state_mx``, ..., ``state_bz``), the seeded ``t = 0`` profile
(``state_<name>_initial``), the per-snapshot ``div B`` series (``divb_linf``)
and the post-run scalar ``divb_linf_final``, plus the grid metadata
(``nx``, ``ny``, ``nghost``, ``geometry``, ``gamma``). With ``cadence > 0`` the
intermediate snapshots are written under the ``snapshot_*`` keys.

Geometry
--------

``geometry: cylindrical`` runs the axisymmetric ``(r, z)`` form: the x-axis is
the radius ``r`` (with ``origin_x_m`` the inner radius, ``0`` to include the
axis) and the y-axis is ``z``. The fluid update adds the ``1/r`` geometric
source terms; constrained transport holds the Cartesian ``div(B)`` at round-off
(see the "Known issues" note in the changelog for the cylindrical-CT caveat).

For a static background ("guide") field ``B = B0 + b``, see
:doc:`mhd_background_field`.
