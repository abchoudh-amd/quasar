PIC simulation workflow
=======================

The Python package exposes a lightweight ``quasar.pic`` surface for building a
2D3V PIC solver from Python and a deck schema in ``quasar.pic.io``.

Minimal deck
------------

.. code-block:: yaml

   units: SI
   normalization:
     reference_density_per_m3: 1.0e18
     reference_species: electron
   domain:
     nx: 64
     ny: 64
     lx_m: 1.0e-2
     ly_m: 1.0e-2
   numerics:
     fdtd_order: 2
     shape: cic
     current_filter: []

The C++ app ``quasar_pic`` is currently a smoke driver that verifies the solver
can be constructed from compiled code. The Python helpers provide schema
validation and initial-condition utilities used by the examples.
