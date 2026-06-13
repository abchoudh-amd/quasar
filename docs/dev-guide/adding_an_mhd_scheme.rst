Adding an MHD numerics scheme
=============================

The ideal-MHD module resolves each of its numerics axes by deck-facing string
name through the plugin registry, exactly like the PIC field-solver / pusher /
deposit guides. ``MhdSolver2D`` (``src/physics/mhd/mhd_solver.cpp``) builds every
scheme through ``make_scheme<Base>(name, what)``, so the driver carries no
``if/else`` over scheme types and a new scheme is selectable from a deck without
touching the solver.

This guide covers the four MHD-specific pluggable axes. Each has a public
interface header under ``include/quasar/numerics/``, a concrete implementation
under ``src/numerics/``, and a registration macro from
``include/quasar/core/registry.hpp``:

============================  ==============================  =========================================  ==========================
Axis                          Interface                       Registration macro                         Deck field (default)
============================  ==============================  =========================================  ==========================
Flux reconstruction           ``IFluxReconstruction``         ``QUASAR_REGISTER_FLUX_RECONSTRUCTION``     ``numerics.reconstruction`` (``mp7``)
Riemann solver                ``IRiemannSolver``              ``QUASAR_REGISTER_RIEMANN_SOLVER``          ``numerics.riemann`` (``hlld``)
Constrained-transport scheme  ``ICtScheme``                   ``QUASAR_REGISTER_CT_SCHEME``               ``numerics.ct`` (``fd_ct_christlieb``)
SSP-RK integrator             ``ISsprkIntegrator``            ``QUASAR_REGISTER_INTEGRATOR``              ``numerics.integrator`` (``ssprk3``)
Positivity limiter            ``IPositivityLimiter``          ``QUASAR_REGISTER_POSITIVITY_LIMITER``      ``numerics.positivity`` (``troubled_cell``)
============================  ==============================  =========================================  ==========================

(The flux-reconstruction axis is shared with the high-order reconstruction guide;
it is listed here for completeness because the MHD solver selects it the same way.)

Steps
-----

#. **Declare the concrete class** in the matching interface header under
   ``include/quasar/numerics/`` (e.g. a new Riemann solver subclasses
   ``quasar::numerics::IRiemannSolver`` in ``riemann_solver.hpp``). The MHD
   interfaces are phrased in ``quasar::mhd::MhdField2D`` /
   ``quasar::numerics::MhdInterfaceStates`` — see the axis-orthogonality note in
   ``CLAUDE.md`` for why these are MHD-typed today rather than templated.

#. **Implement it** in a new (or existing) translation unit under
   ``src/numerics/``. Any device work goes through the ``launch_mhd_*`` ABI
   declared in ``include/quasar/physics/mhd/kernels.hpp`` and defined under
   ``src/backend/hip/mhd/`` — do not include a HIP header from the numerics
   source directly (backend isolation). If you add a new device kernel, declare
   its ``launch_mhd_*`` wrapper in ``kernels.hpp`` and define it under
   ``src/backend/hip/mhd/`` so the signature is checked at both ends.

#. **Register it** under a deck-facing name with the macro from the table, in the
   same translation unit as the implementation, e.g.::

       QUASAR_REGISTER_RIEMANN_SOLVER("hllc", ::quasar::numerics::HllcRiemann)

   Keeping the registration in the implementation TU ensures the static
   initializer is not dropped by the linker. Add the source file to
   ``src/numerics/CMakeLists.txt`` (and mark it ``LANGUAGE HIP`` if it contains
   device code), mirroring the existing ``hlld_riemann.cpp`` / ``ct_scheme.cpp`` /
   ``ssprk_integrator.cpp`` / ``positivity_limiter.cpp`` entries.

#. **Expose it to decks.** The Python loader validates ``numerics.*`` names; add
   the new name to the relevant token list / validation in
   ``python/quasar/mhd/io.py`` so a deck selecting it is accepted, and update the
   MHD user guide if the new scheme changes deck-facing behavior.

#. **Test it.** Add a C++ unit test under ``tests/unit/numerics/`` mirroring the
   header (e.g. ``test_hlld_riemann.cpp`` for the Riemann axis). A registry-linkage
   test (``test_*_registry_linkage``) confirms the name resolves; a behavioral
   test should exercise the scheme's defining property — for a Riemann solver the
   correct intermediate states on a discontinuity, for a reconstruction the
   non-oscillatory (no-new-extremum) clip, for a CT scheme that ``div(B)`` stays
   at round-off, for the integrator the design order of accuracy, for the
   positivity limiter that a stiff state stays finite and positive.

Selection seam
--------------

``MhdSolver2D`` resolves every scheme in its constructor via
``make_scheme<Base>(cfg_.<field>, "<label>")`` (``src/physics/mhd/mhd_solver.cpp``).
An unknown deck name raises ``std::invalid_argument`` with the axis label, so a
typo in ``numerics.riemann`` reports ``unknown Riemann solver '<name>'`` rather
than a raw registry error. The reconstruction scheme is built first because its
``required_nghost()`` fixes the working-grid ghost halo that sizes every field and
register.
