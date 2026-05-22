Adding a field evaluator
========================

External PIC fields and the magnetostatics module share
``quasar::numerics::IFieldEvaluator``. A new evaluator implements
``evaluate_B`` and optionally ``evaluate_E`` for a point cloud, then registers
with ``QUASAR_REGISTER_FIELD_EVALUATOR``.

Analytic examples live under ``physics/analytic_fields``. The same registry is
used by PIC's external-field sampler and by existing magnetic-field workflows.
