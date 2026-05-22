Quasar documentation
====================

Quasar is a HIP-accelerated numerical simulation framework. The current
release ships a single physics module - **magnetostatics** - that
computes magnetic flux density and its analytic Jacobian via the
Biot-Savart law on arbitrary thin-wire current-carrying conductors,
backed by per-segment HIP kernels and exposed through a small Python
front-end (``quasar.coil``).

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

.. toctree::
   :maxdepth: 2
   :caption: Developer guide

   dev-guide/adding_a_geometry
   dev-guide/adding_a_pusher
   dev-guide/adding_a_boundary
   dev-guide/adding_a_field_evaluator
   dev-guide/adding_a_filter

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
