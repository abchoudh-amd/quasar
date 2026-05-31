#pragma once

namespace quasar::core {

// Axis-neutral base for a field source consumed by numerics::IFieldEvaluator.
// Concrete sources (e.g. magnetostatics::ConductorSystem) derive from this so
// the numerics axis depends only on this core abstraction rather than on a
// specific physics type. Evaluators that need a concrete source downcast; those
// that ignore the source (the analytic fields) never name a physics type.
class IFieldSource {
 public:
  virtual ~IFieldSource() = default;
};

}  // namespace quasar::core
