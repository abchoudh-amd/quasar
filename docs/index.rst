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

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user-guide/coil_design

.. toctree::
   :maxdepth: 2
   :caption: Developer guide

   dev-guide/adding_a_geometry

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
