Adding a PIC boundary condition
===============================

Field boundaries derive from ``IFieldBoundary`` and particle boundaries derive
from ``IParticleBoundary``. The public ``BoundarySpec`` keeps the solver
orchestrator independent of concrete boundary names.

Add a new boundary by declaring the class under ``include/quasar/boundary/``,
implementing its launch wrapper under ``src/boundary/``, adding any HIP kernels
under ``src/backend/hip/pic/``, and registering it with the appropriate
``QUASAR_REGISTER_*_BOUNDARY`` macro.
