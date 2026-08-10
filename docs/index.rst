Quasar documentation
====================

Quasar is a HIP-accelerated numerical simulation framework. It currently
ships three physics vertical slices:

* **magnetostatics** - computes magnetic flux density and its analytic
  Jacobian via the Biot-Savart law on arbitrary thin-wire current-carrying
  conductors, backed by per-segment HIP kernels and exposed through the
  ``quasar.coil`` Python front-end.
* **electromagnetic PIC** - a 2D3V particle-in-cell module (Yee FDTD fields,
  charge-conserving current deposition, Boris push) driven from the
  ``quasar.pic`` front-end.
* **ideal MHD** - a finite-volume + constrained-transport module (high-order
  MP5/MP7 reconstruction on Cartesian and axisymmetric cylindrical ``(r, z)``
  grids, HLLD Riemann solver, and SSP-RK3) driven from the ``quasar.mhd``
  front-end.

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/magnetostatics
   theory/pic
   theory/boundary_conditions
   theory/normalization
   theory/digital_filters

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user-guide/coil_design
   user-guide/pic_simulation
   user-guide/pic_cylindrical
   user-guide/mhd_simulation
   user-guide/mhd_background_field
   user-guide/distributed_simulation

.. toctree::
   :maxdepth: 2
   :caption: Developer guide

   dev-guide/adding_a_geometry
   dev-guide/adding_a_pusher
   dev-guide/adding_a_boundary
   dev-guide/adding_a_field_evaluator
   dev-guide/adding_a_field_solver
   dev-guide/adding_a_deposit_scheme
   dev-guide/adding_a_filter
   dev-guide/adding_a_background_field
   dev-guide/adding_an_mhd_scheme
   dev-guide/adding_distributed_physics

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
