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
