Adding an MHD numerics scheme
=============================

The ideal-MHD module resolves each of its numerics axes by deck-facing string
name through the plugin registry, exactly like the PIC field-solver / pusher /
deposit guides. ``MhdSolver2D`` (``src/physics/mhd/mhd_solver.cpp``) builds every
scheme through ``make_scheme<Base>(name, what)``, so the driver carries no
``if/else`` over scheme types.

.. warning::

   **Read "What is actually pluggable today" below before writing a new scheme.**
   Registering a class is necessary but *not* sufficient to have it run. Only the
   **positivity** axis dispatches through its registry object on the hot path;
   the **integrator** axis drives the stage loop but not the coefficients. The
   **reconstruction**, **Riemann**, and **CT** axes are currently **fixed
   built-ins**: the device evolution path launches one hard-coded algorithm each,
   and ``MhdSolver2D`` rejects any other name at construction rather than
   accepting a scheme and silently ignoring it. Adding a genuinely new algorithm
   on one of those three axes requires extending the device path as well, not
   just registering a class.

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

   A positivity implementation must preserve the trailing
   ``collocation_order`` and ``RadialTablesView`` parameters on both ``apply``
   and ``admissible_fraction`` and forward them to its device launches. The
   solver passes an active, order-matched view in cylindrical runs; the inactive
   default preserves Cartesian collocation for standalone callers.

#. **Register it** under a deck-facing name with the macro from the table, in the
   same translation unit as the implementation, e.g.::

       QUASAR_REGISTER_RIEMANN_SOLVER("hllc", ::quasar::numerics::HllcRiemann)

   Keeping the registration in the implementation TU ensures the static
   initializer is not dropped by the linker. Add the source file to
   ``src/numerics/CMakeLists.txt`` (and mark it ``LANGUAGE HIP`` if it contains
   device code), mirroring the existing ``hlld_riemann.cpp`` / ``ct_scheme.cpp`` /
   ``ssprk_integrator.cpp`` / ``positivity_limiter.cpp`` entries.

#. **Expose it to decks.** The Python loader validates ``numerics.*`` names
   against the live registry, so registration alone makes the name pass Python
   validation; update the MHD user guide if the new scheme changes deck-facing
   behavior. For any axis other than positivity you must **also** add the device
   dispatch and relax the matching name check in ``validate_config`` — see
   "What is actually pluggable today" below — otherwise a deck selecting the new
   name is rejected by the C++ constructor.

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

What is actually pluggable today
--------------------------------

Constructing a scheme object is not the same as running it. The MHD evolution
path (``MhdSolver2D::compute_residual`` / ``combine_stage``) launches fused
``launch_mhd_*`` device kernels, and only some of them dispatch through the
registry object:

=============================  ===================================================
Axis                           How the constructed object is used on the hot path
=============================  ===================================================
Positivity limiter             **Fully pluggable.** ``positivity_->admissible_fraction()`` is called directly on every stage and retry.
Flux reconstruction            **Fixed built-in.** Only ``required_nghost()`` is read; it is mapped back to order 2/5/7 by ``reconstruction_order()`` and the built-in MUSCL / MP5 / MP7 device kernel is launched. ``reconstruct_faces()`` is never called during evolution.
Riemann solver                 **Fixed built-in.** ``riemann_`` is constructed but never invoked; the residual always launches the built-in HLLD kernel.
CT scheme                      **Fixed built-in.** ``ct_`` is constructed but never invoked; the face-B rate always comes from the built-in FD-CT corner-EMF kernel.
SSP-RK integrator              **Sequencing only.** ``integrator_->advance()`` *is* called and drives the stage loop, so a custom integrator can control stage ordering, retries, and error handling. But ``combine_stage()`` owns the Shu-Osher coefficients and implements exactly the 3-stage SSPRK3 tableau, so a scheme cannot supply its own weights.
=============================  ===================================================

Because a silently ignored scheme is worse than a rejected one, ``validate_config``
rejects any name outside the supported set **by name**, before construction:

* ``numerics.reconstruction`` must be ``muscl_minmod``, ``mp5``, or ``mp7`` in
  both Cartesian and cylindrical geometry;
* ``numerics.riemann`` must be ``hlld``;
* ``numerics.ct`` must be ``fd_ct_christlieb``.

The integrator axis is checked *structurally* rather than by name, because its
``advance()`` really does run: the constructor rejects any integrator whose
``n_stages()`` is not 3, since ``combine_stage`` would otherwise throw partway
through a step. A registered 3-stage integrator that drives the documented
``compute_residual`` / ``combine_stage`` sequence is therefore usable — this is
how the rollback tests in
``tests/unit/physics/mhd/test_mhd_positivity_preservation.cpp`` inject faults —
but it inherits SSPRK3's coefficients rather than supplying its own.

So registering ``HllcRiemann`` makes the name visible in
``_core.mhd.registered_riemann_solvers()`` and the Python deck validator, but a
deck selecting it is rejected by the C++ constructor rather than run as HLLD.

Because of that, the ``IRiemannSolver`` and ``ICtScheme`` surfaces are sized to
what they are actually used for rather than to the full algorithm:

* ``IRiemannSolver`` is a **host test seam**. ``MhdSolver2D`` calls
  ``launch_mhd_hlld_flux`` directly, so nothing dispatches through the interface;
  its value is that ``HlldRiemann`` reaches the same host/device-shared
  ``hlld_core.hpp`` the GPU runs, letting unit tests drive the seven-wave algebra
  from hand-built states.
* ``ICtScheme`` carries **only** ``divergence_b_linf``, the diagnostic
  ``MhdSolver2D::divergence_b_max()`` genuinely dispatches through. The corner-EMF
  construction and the face-B advance are not on it: the solver builds the EMF
  with ``launch_mhd_ct_emf_prepare`` / ``_finish`` (passing its own background,
  boundary flags, and MP5/MP7 order) and then advances face B as an ordinary
  residual component via ``launch_mhd_emf_curl_rate`` + ``rk_stage``, so it rides
  the same SSP-RK convex combination as the other seven components. Applying the
  curl directly to the field on top of the flux divergence would double-count it.

The solver's own scheme members mirror this: it owns an ``ISsprkIntegrator`` and
an ``IPositivityLimiter`` (both genuinely called) and no ``IRiemannSolver`` or
``ICtScheme`` instance at all, since constructing one that is never dereferenced
would advertise a dispatch seam that does not exist.

Adding a real scheme to one of the fixed axes therefore also requires:

#. a device kernel implementing it (declared in
   ``include/quasar/physics/mhd/kernels.hpp``, defined under
   ``src/backend/hip/mhd/``);
#. a dispatch seam in ``compute_residual`` (or ``combine_stage`` for an
   integrator) that routes to it instead of the hard-coded launcher — the
   interfaces as written do not yet carry everything the kernels need
   (background field, boundary flags, quadrature rule, stage routing); and
#. relaxing the corresponding name check in ``validate_config``.

Extending the ``IFluxReconstruction`` / ``IRiemannSolver`` / ``ICtScheme`` /
``ISsprkIntegrator`` interfaces to cover those inputs, so the hot path can
dispatch through them generically, is the outstanding work that would make these
axes pluggable in the same sense as the positivity axis.
