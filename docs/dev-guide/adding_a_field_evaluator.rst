Adding a field evaluator
========================

External PIC fields and the magnetostatics module share
``quasar::numerics::IFieldEvaluator``. A new evaluator implements the single pure
virtual ``evaluate_B``; ``evaluate_grad_B`` and ``evaluate_E`` both have default
zero implementations that may optionally be overridden (e.g. Biot-Savart and the
gradient field override ``evaluate_grad_B``; a uniform field with an E component
overrides ``evaluate_E``). It then registers with
``QUASAR_REGISTER_FIELD_EVALUATOR``. The methods take an axis-neutral
``core::IFieldSource`` and a ``core::PointCloud``; an evaluator that needs a
concrete source (e.g. Biot-Savart) downcasts it, while the analytic fields
ignore the source entirely.

Analytic examples live under ``physics/analytic_fields``. The same registry is
used by PIC's external-field sampler and by existing magnetic-field workflows.

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
                               Real gamma, stream_t stream);

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
   reconstruction. This is a per-interface finite/positive guard, not a
   positivity-preserving limiter.

Backend isolation holds: all ``.hip`` / device code stays under
``src/backend/hip/mhd/``, and the registry classes reach it only through the
launch ABI in ``include/quasar/physics/mhd/kernels.hpp``.
