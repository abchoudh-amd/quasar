Adding a field evaluator
========================

External PIC fields and the magnetostatics module share
``quasar::numerics::IFieldEvaluator``. A new evaluator implements the two pure
virtuals ``evaluate_B`` **and** ``evaluate_grad_B`` (only ``evaluate_E`` has a
default zero implementation that may optionally be overridden), then registers
with ``QUASAR_REGISTER_FIELD_EVALUATOR``. Both take an axis-neutral
``core::IFieldSource`` and a ``core::PointCloud``; an evaluator that needs a
concrete source (e.g. Biot-Savart) downcasts it, while the analytic fields
ignore the source entirely.

Analytic examples live under ``physics/analytic_fields``. The same registry is
used by PIC's external-field sampler and by existing magnetic-field workflows.
