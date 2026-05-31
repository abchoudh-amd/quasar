Adding a PIC field solver
=========================

Field solvers implement ``quasar::numerics::IFieldSolver`` and advance the Yee
electric/magnetic fields from the deposited current. The default is
``YeeFdtd2D<Order>``, the staggered FDTD curl scheme, registered for 2nd- and
4th-order accuracy.

The solver is selected from the deck's ``numerics.fdtd_order`` (2 or 4): the PIC
solver derives the registry name ``"yee_o" + order`` and builds the scheme through
the registry (``src/physics/pic/pic_solver.cpp``), so there is no ``if/else`` over
the order in the driver.

To add a field solver, declare the concrete class in
``include/quasar/numerics/field_solver.hpp``, implement its launch wrapper in
``src/physics/pic`` (advancing through the ``launch_pic_fdtd_*`` kernels in
``src/backend/hip/pic``), and register it under a deck-facing name with
``QUASAR_REGISTER_FIELD_SOLVER("<name>", <Class>)`` (mirroring the
field-evaluator, pusher, boundary, and current-filter guides). Keep the
registration in the ``pic_solver`` translation unit so the static initializer is
never dropped by the linker. The order-derived name must match what the deck
maps to; if your scheme introduces a new selection vocabulary, extend the
name-derivation in ``EmPic2D3V`` accordingly.
