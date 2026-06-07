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

Ideal-MHD boundaries: one-sided non-periodic stencils
-----------------------------------------------------

The ideal-MHD vertical slice derives its boundaries from ``IMhdFluidBoundary``
and ``IMhdFieldBoundary`` (declared in ``include/quasar/boundary/mhd_boundary.hpp``),
which self-register under their string names via ``QUASAR_REGISTER_MHD_FLUID_BOUNDARY``
and ``QUASAR_REGISTER_MHD_FIELD_BOUNDARY`` (``periodic`` | ``outflow`` | ``reflecting``).
As with PIC, every concrete wall does its work in ``fill_ghosts`` (wrap /
zero-gradient / wall-mirror of the thin boundary halo); the fluid axis and the
magnetic-field axis carry independent per-side selections because a reflecting
wall imposes different symmetries on momentum than on **B**.

Device-path ghost fills
~~~~~~~~~~~~~~~~~~~~~~~~~

The MHD ghost fills run **on device** (HIP). Each concrete BC class is a **thin
launcher**: its ``fill_ghosts`` simply dispatches one of the two per-physics
launch ABI entry points

.. code-block:: cpp

   // include/quasar/physics/mhd/kernels.hpp
   void launch_mhd_fill_ghosts_fluid(MhdField2D<Real>& u, int side, int mode, stream_t stream);
   void launch_mhd_fill_ghosts_field(MhdField2D<Real>& u, int side, int mode, stream_t stream);

implemented as device kernels in ``src/backend/hip/mhd/mhd_boundary_hip.hip``.
The ``fluid`` variant fills the conserved fluid components (rho, momentum,
energy, cell-centered ``bz``); the ``field`` variant fills the face-staggered
in-plane magnetic components (``bx_face`` / ``by_face``). ``side`` is the
canonical order ``0 = x_lo, 1 = x_hi, 2 = y_lo, 3 = y_hi``, and ``mode`` selects
the wall behavior:

* ``mode 0`` — **periodic**: wrap from the opposite interior layer.
* ``mode 1`` — **outflow**: zero-gradient copy of the adjacent interior cell/face.
* ``mode 2`` — **reflecting**: mirror the interior; the **normal** component is
  sign-flipped (the normal momentum — ``mx`` at an x-side, ``my`` at a y-side —
  for the fluid fill, and the normal face-B — ``bx_face`` at an x-side,
  ``by_face`` at a y-side — for the field fill), while tangential momentum and
  all cell-centered scalars are copied even.

The four corner ghost blocks are filled as well: the y-side fills span the full
storage width so the corner halo shared with the x-faces is populated. Backend
isolation holds — all ``.hip`` / device code stays under
``src/backend/hip/mhd/``, and the registry BC classes reach it only through the
launch ABI in ``include/quasar/physics/mhd/kernels.hpp``.

Where the MHD axis differs from PIC is in how a *non-periodic* side reaches into
the finite-volume reconstruction. The solver classifies each side with the free
function

.. code-block:: cpp

   // include/quasar/boundary/mhd_boundary.hpp
   bool quasar::boundary::mhd_boundary_is_periodic(const std::string& name);

which returns ``true`` only for ``"periodic"`` (defined in
``src/physics/mhd/mhd_boundary.cpp``). Owning this classification in the boundary
axis keeps the solver free of any ``if/else`` chain over concrete BC names: it
simply asks "is this side periodic?" per side.

Per-side flags
~~~~~~~~~~~~~~~

From ``cfg.boundary.field`` the solver builds a small per-side descriptor

.. code-block:: cpp

   // include/quasar/physics/mhd/kernels.hpp
   struct BoundaryFlags4 {
     int side[4];  // [x_lo, x_hi, y_lo, y_hi]; 1 = non-periodic, 0 = periodic
   };

(see ``MhdSolver2D::boundary_flags()`` in ``src/physics/mhd/mhd_solver.cpp``,
which maps each name through ``mhd_boundary_is_periodic``). The flags are threaded
into the reconstruction (``launch_mhd_reconstruct``) and the constrained-transport
corner-EMF kernel (``launch_mhd_ct_emf``). An all-zero ``BoundaryFlags4`` is the
all-periodic fast path and is bit-identical to the pre-flags behavior.

The one-sided mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~

At a **non-periodic** side (``outflow`` / ``reflecting``), the boundary-face
reconstruction switches to an **interior-biased one-sided slope**: it drops its
dependence on the ghost *gradient* near the edge while **still reading the filled
ghost value** that supplies the wall closure. A **periodic** side keeps the
two-sided stencil and wraps through the ghosts, exactly as before.

.. important::

   The ghost layers are **still filled and still read** — ``fill_ghosts`` is not
   disabled at a non-periodic side. The reflecting / outflow wall closure
   continues to come from the mirrored / zero-gradient ghost **values**. The
   one-sided change removes only the dependence on the ghost **gradient** in the
   slope used at the edge interface, replacing the two-sided central slope with
   an interior-biased one.

Conservation
~~~~~~~~~~~~~

The conservative flux-difference (``launch_mhd_flux_difference``) is
**unchanged** by the boundary flags. It still differences the true adjacent
stored interface fluxes, so the discrete update telescopes and conservation holds
exactly across the domain. The one-sided behavior lives entirely in how the
boundary-face left/right interface states are *reconstructed*, never in the flux
divergence.

Relation to the PIC one-sided correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the MHD analogue of the PIC **post-curl node correction** described above
(``pec`` / ``outflow`` close the boundary node with one-sided, interior-only
stencils). The intent is the same — avoid letting a one-sided edge depend on
information that has no two-sided neighbor — but the realization differs by
scheme: PIC applies it as a post-update fix to the boundary-node field values,
whereas MHD applies it inside the finite-volume **reconstruction** of the
boundary-face states (with the ghost values still participating as the wall
closure).
