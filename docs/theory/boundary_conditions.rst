PIC boundary conditions
=======================

PIC boundary handling is split by field and particle state. ``BoundarySpec``
stores one selection per side of the 2D domain.

Field boundaries support periodic wrapping, PEC walls, and a first-order Mur
``outflow`` condition for outgoing electromagnetic waves. Particle boundaries
support periodic wrapping, specular reflection, and absorbing removal through
an ``alive`` mask. The HIP launch wrappers are kept behind the public
``IFieldBoundary`` and ``IParticleBoundary`` interfaces so future PML or
thermalizing particle boundaries can be added without changing the PIC solver
orchestration.
