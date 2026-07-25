Adding a current-deposit scheme
===============================

Current-deposit schemes implement ``quasar::numerics::IDepositScheme`` and build
the grid current from the particle motion over a step. The default is
``Esirkepov2D<ShapeOrder>``, the charge-conserving density-decomposition deposit,
registered for the CIC (shape order 1) and TSC (shape order 2) particle shapes.

The scheme is selected from the deck's ``numerics.shape`` (``cic`` or ``tsc``):
the PIC solver derives the registry name ``"esirkepov_" + shape`` and builds it
through the registry (``src/physics/pic/pic_solver.cpp``), so there is no
``if/else`` over the shape in the driver. The matching Boris pusher
(``"boris_" + shape``) is selected from the same vocabulary, so a new shape order
generally needs both a pusher and a deposit registration.

To add a deposit scheme, declare the concrete class in
``include/quasar/numerics/deposit.hpp``, implement its launch wrapper in
``src/physics/pic`` (driving the ``launch_pic_deposit_*`` kernels in
``src/backend/hip/pic/deposit_hip.hip``), and register it under a deck-facing name
with ``QUASAR_REGISTER_DEPOSIT("<name>", <Class>)`` (mirroring the field-evaluator,
field-solver, pusher, boundary, and current-filter guides). Keep the registration
in the ``pic_solver`` translation unit so the static initializer is never dropped
by the linker.

A deposit must honour the per-axis periodicity set via
``set_periodic_axes(periodic_x, periodic_y)``: wrap a node only when both sides of
an axis are periodic, and otherwise deposit boundary-crossing current into the
ghost cells so the specular fold-back can reflect it into the interior. The
charge-conserving contract is that the **forward face-to-cell divergence** of
the deposited current exactly cancels the charge change. It is the same
staggered divergence used by Gauss's law and annihilates the field solver's
Ampere curl. For fourth-order FDTD, apply the compact compatibility inverse
``D4+ J = D2+ J_raw`` after the ordinary Esirkepov prefix deposit.
