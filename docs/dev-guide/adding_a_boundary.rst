Adding a PIC boundary condition
===============================

Field boundaries derive from ``IFieldBoundary`` and particle boundaries derive
from ``IParticleBoundary``. The public ``BoundarySpec`` keeps the solver
orchestrator independent of concrete boundary names.

Add a new boundary by declaring the class under ``include/quasar/boundary/``,
implementing its launch wrapper under ``src/boundary/``, adding any HIP kernels
under ``src/backend/hip/pic/``, and registering it with the appropriate
``QUASAR_REGISTER_*_BOUNDARY`` macro.

Field boundary mechanisms
-------------------------

``IFieldBoundary`` supports two ways to impose a field wall:

* **Pre-curl ghost fill** (``fill_ghosts``): populate the ghost halo before the
  FDTD curl reads it. Used by ``periodic`` (copy the opposite interior edge).
* **Post-curl node correction** (``correct_after_b`` / ``correct_after_e``):
  leave ``fill_ghosts`` a no-op and instead overwrite the boundary-node values
  the interior curl just computed, using one-sided (interior-only) stencils so no
  ghost halo participates. Used by ``pec`` (reflecting wall: pin tangential E and
  normal B, one-sided close the rest) and ``outflow`` (first-order Mur open
  wall). ``correct_after_e`` runs after the E-update so an outflow Mur step can
  read the just-updated adjacent interior node. ``configure(fdtd_order)`` lets a
  BC pick its order-dependent kernel, and ``set_corner_skip`` lets a y-face cede
  a shared corner node to the x-face that runs first.

.. note::

   The ``outflow`` (first-order Mur) wall is stable for an outflow channel
   (outflow on one axis) and when mixed with ``pec`` walls, but is **weakly
   unstable where two outflow walls meet at a corner** (an open box with outflow
   on all four sides). A dedicated corner-extrapolation closure would be needed
   for the all-sides-open case.
