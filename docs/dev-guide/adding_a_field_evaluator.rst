Adding a field evaluator
========================

External PIC fields and the magnetostatics module share
``quasar::numerics::IFieldEvaluator``. A new evaluator implements the single pure
virtual ``evaluate_B``; ``evaluate_grad_B`` and ``evaluate_E`` both have default
zero implementations that may optionally be overridden (e.g. Biot-Savart,
dipole, and the gradient field override ``evaluate_grad_B``; a uniform field
with an E component overrides ``evaluate_E``). The zero-gradient default is an
API convenience, not a physical capability claim: an evaluator whose Jacobian
is trustworthy must also override ``provides_grad_B()`` to return ``true``.
It then registers with
``QUASAR_REGISTER_FIELD_EVALUATOR``. The methods take an axis-neutral
``core::IFieldSource``; an evaluator that needs a concrete source (e.g.
Biot-Savart) downcasts it, while the analytic fields ignore the source entirely.

Analytic examples live under ``physics/analytic_fields``. The same registry is
used by PIC's external-field sampler and by existing magnetic-field workflows.
Registry construction is followed by ``configure(EvaluatorParams)``. Numeric
parameters are flat vectors; the file-grid loader uses ``origin``, ``spacing``,
``dims``, and flattened x-fastest ``values`` to keep file parsing in the Python
deck boundary while leaving interpolation in the shared C++ evaluator.

The evaluator interface is device-resident
------------------------------------------

Every ``evaluate_*`` method takes a ``core::DevicePointCloud`` and returns
``core::DeviceVectorField`` (three SoA component planes) or
``core::DeviceTensorField`` (nine component-major planes, entry ``(i, j)`` of
point ``p`` at ``data()[(3 * i + j) * size() + p]``). Nothing on this interface
touches host memory:

.. code-block:: cpp

   core::DeviceVectorField evaluate_B(
       const core::IFieldSource& source,
       const core::DevicePointCloud& observations) const override;

This is not a style choice. An evaluator's output is normally consumed by
another device stage -- the PIC external-field sampler writes it straight into
a ``YeeField2D``, and the MHD background builder curls it -- so a host round
trip between them is pure cost. The Biot-Savart kernel already computed into
device SoA planes and then transposed them to host AoS purely to satisfy the
old signature; that transpose is gone.

**Writing the evaluator.** Do the arithmetic in a kernel under
``src/backend/hip/<module>/``, declared through a per-physics launch ABI header
(``include/quasar/physics/analytic_fields/kernels.hpp`` is the model). The class
in ``src/physics/`` keeps only parameter validation, the launch, and the status
check. A kernel cannot throw, so it reports failure by OR-ing bits into an
``int*`` status word with an integer atomic -- exact and order-independent,
hence independent of the launch geometry -- and the host turns that word into
the exception with ``core::throw_on_evaluator_status``. The bit meanings
(singular / not representable / non-finite point / outside grid) are documented
on that function.

**Range, not just precision.** The analytic evaluators used to carry their
intermediates in host ``long double``, whose wider exponent let
``moment_scale * mu0_over_4pi * inv_r^3`` be formed without overflowing on the
way to a representable answer. A device has no ``long double``. Carry the
extended range explicitly with ``numerics::ScaledValue`` and the exact-expansion
reducers in ``include/quasar/numerics/scaled_arithmetic.hpp``; the device-side
lift/multiply/collapse helpers are in
``src/backend/hip/analytic_fields/scaled_device.hpp``. Any module using those
reducers **must** be compiled ``-ffp-contract=off``: contraction turns the
error-free two-sum silently back into a naive sum.

**Crossing to the host.** ``.to_host()`` on either container downloads and, for
the tensor, transposes into the ``Field<Vec3>`` / ``Field<Mat3x3>`` the CLIs and
Python bindings speak. Call it at an output boundary and nowhere earlier. Tests
use the wrappers in ``tests/unit/support/host_evaluate.hpp``, and test doubles
derive from ``tests/unit/support/host_field_evaluator.hpp``, which implements
the device interface around a host hook. Both live under ``tests/`` on purpose:
on the interface they would give production code a one-token way to reintroduce
the round trip.

**Accuracy gate.** A ported evaluator ships with an equivalence test carrying
its own ``long double`` oracle, asserting the device is no worse than a naive
``double`` evaluation of the same closed form on a configuration where that
formula cancels, plus bitwise reproducibility across repeated launches. See
``tests/unit/physics/analytic_fields/test_analytic_device_accuracy.cpp``. A bare
absolute tolerance is not sufficient: it cannot distinguish a correct port from
one that is slightly wrong in a way that happens to be small on the chosen
inputs.

Deck and registry contract
--------------------------

Built-in evaluators retain their named, unit-aware deck schemas. A registered
plugin uses the generic ``params`` mapping instead:

.. code-block:: yaml

   evaluator:
     type: my_plugin
     params:
       gain: 2.0
       axis: [1.0, 0.0, -1.0]

Parameter keys must be non-empty strings. Each value must be either one finite
real scalar or a flat list/tuple of finite reals. The deck layer normalizes a
scalar such as ``gain: 2.0`` to the one-element vector ``[2.0]`` before calling
``configure``. Booleans, strings/bytes, mappings, nested sequences, and
non-finite values are rejected. These generic values are already resolved in
the deck's units; the plugin remains responsible for rejecting unknown keys and
invalid vector lengths.

The Python registry surface is live rather than mirrored in a hard-coded list:
``field_evaluator_names()`` returns every registered name in sorted order, and
``field_evaluator_provides_vector_potential(name)`` and
``field_evaluator_provides_grad_B(name)`` report the corresponding optional
capabilities. The same flags are available on an evaluator instance as
``provides_vector_potential`` and ``provides_grad_B``. ``IFieldEvaluator``
defaults to B-only with no trustworthy Jacobian. A plugin that supplies magnetic
vector potential A must override both
``provides_vector_potential()`` and ``evaluate_A()``; otherwise frontends reject
A output requests before evaluation.

A capability may depend on configuration. In that case the instance query is
authoritative after ``configure``; the name-only registry helper reports the
state of a default-constructed instance. ``FileGridEvaluator`` is the built-in
example: it reports gradient support only for a configured map with at least two
nodes on every axis. A singleton axis is one sampled geometric plane, not an
invariant direction with a known zero normal derivative.

The cylindrical PIC sampler treats prescribed ``B`` as a force-only external
field, rather than as an FDTD-evolved unknown. It therefore verifies Maxwell's
continuous constraint from ``trace(grad B)`` and requires
``provides_grad_B() == true`` for every nonzero cylindrical magnetic evaluator.
This avoids rejecting a smooth divergence-free field merely because samples of
it have the selected Yee stencil's ordinary truncation error. An evaluator that
returns exactly zero ``B`` (for example an electric-only plugin) needs no
gradient capability. The sampler also checks rotational covariance about the
configured physical symmetry axis; a single arbitrary Cartesian slice is not a
valid axisymmetric field.

Registration names must be unique within the live field-evaluator registry.
Registering an empty name, an empty factory, or a duplicate name throws
``std::invalid_argument``; a duplicate never replaces the original factory.
Registry access is synchronized, so discovery and construction may safely run
concurrently, but plugins must still choose globally distinct evaluator names
within a process.

Ideal-MHD reconstruction: the device seam
-----------------------------------------

The ideal-MHD slice does *not* use ``IFieldEvaluator`` — its in-cell field is
advanced by a finite-volume flux reconstruction rather than sampled from an
analytic evaluator — but its reconstruction schemes follow the same
plug-by-name pattern, so they are documented here for proximity. The
registry-facing classes (``MusclMinmodRecon`` / ``Mp5Recon`` / ``Mp7Recon``,
registered under ``"muscl_minmod"`` / ``"mp5"`` / ``"mp7"`` via
``QUASAR_REGISTER_FLUX_RECONSTRUCTION`` in
``src/numerics/flux_reconstruction.cpp``) are now **thin launchers**: they hold
no host compute body and simply map the scheme to its order (2 / 5 / 7) and
dispatch the device kernel. This mirrors the ``mhd_geometric_source`` wrapper.

The reconstruction hot path runs fully **on device** (HIP). MUSCL-minmod
(order 2) and the high-order characteristic monotonicity-preserving MP5
(order 5) / MP7 (order 7) reconstructions all execute in
``src/backend/hip/mhd/mhd_reconstruct.hip`` through the per-physics launch ABI
seam

.. code-block:: cpp

   // include/quasar/physics/mhd/kernels.hpp
   void launch_mhd_reconstruct(const MhdField2D<Real>& u,
                               const MhdBackgroundField<Real>& b0,
                               int dir, MhdInterfaceStates<Real>& out,
                               int scheme_order, BoundaryFlags4 flags,
                               Real gamma, stream_t stream,
                               bool rate_only = false,
                               RadialTablesView radial_tables = {});

which honors ``scheme_order``. To make the characteristic path callable from the
device kernel, the eigensystem (``include/quasar/numerics/mhd_eigensystem.hpp``)
and the characteristic projection
(``include/quasar/numerics/characteristic_projection.hpp``) are
``QUASAR_HOST_DEVICE``, and the scalar Suresh-Huynh MP limiter helpers live in a
shared ``QUASAR_HOST_DEVICE inline`` header
(``include/quasar/numerics/mp_limiter.hpp``) used by both host and device.

.. note::

   The characteristic MP5/MP7 path carries **no positivity guarantee**. The
   device reconstruct kernel therefore applies a **per-interface** robustness
   fallback: for any single interface whose high-order reconstructed state is
   non-finite or has non-positive density or pressure, that one face drops to
   2nd-order MUSCL; every smooth, positive interface keeps the full high-order
   reconstruction. The solver additionally checks every SSP-RK stage through
   the registered positivity limiter. If any cell leaves the admissible set, the
   conservative update is rolled back and CFL-subcycled using a first-order HLL
   anchor; no per-cell mass or energy floor is applied during evolution.

Backend isolation holds: all ``.hip`` / device code stays under
``src/backend/hip/mhd/``, and the registry classes reach it only through the
launch ABI in ``include/quasar/physics/mhd/kernels.hpp``.

.. important::

   MP5/MP7 use uniform finite-volume moments on Cartesian grids and along the
   axial direction of a cylindrical grid. Radial cylindrical stencils instead
   obtain radius-dependent :math:`r\,dr` rows from
   ``include/quasar/numerics/radial_moments.hpp``. ``MhdSolver2D`` owns their
   device storage through ``RadialTables`` and threads a non-owning
   ``RadialTablesView`` through every affected launch; the inactive default is
   the Cartesian path. Fluid reconstruction and staggered magnetic face-to-cell
   collocation must always use matching moment families and order when this
   device seam is extended.
