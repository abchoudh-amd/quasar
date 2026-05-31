#pragma once

#include "quasar/core/field.hpp"
#include "quasar/core/field_source.hpp"
#include "quasar/core/observations.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/core/types.hpp"

namespace quasar::numerics {

// The evaluator's observation set is the axis-neutral core::PointCloud and its
// source is the axis-neutral core::IFieldSource, so the numerics axis depends on
// no concrete physics type. An evaluator that needs a specific source (e.g.
// Biot-Savart) downcasts it; evaluators that ignore the source (the analytic
// fields) name no physics type at all.
class IFieldEvaluator {
 public:
  virtual ~IFieldEvaluator() = default;

  virtual Field<Vec3> evaluate_B(const core::IFieldSource& source,
                                 const core::PointCloud& observations) const = 0;

  // Field gradient (grad B)_{ij} = dB_i/dp_j. Defaults to zero so an evaluator
  // with no analytic Jacobian (uniform, dipole) need not restate it; evaluators
  // that model a gradient (gradient, Biot-Savart) override.
  virtual Field<Mat3x3> evaluate_grad_B(const core::IFieldSource&,
                                        const core::PointCloud& observations) const {
    return Field<Mat3x3>(observations.size());
  }

  // E field. Defaults to zero: the magnetostatic evaluators model no E field, so
  // a caller using one as a PIC external-field source contributes zero E.
  // Evaluators that model an E field (uniform with e0) override.
  virtual Field<Vec3> evaluate_E(const core::IFieldSource&,
                                 const core::PointCloud& observations) const {
    Field<Vec3> out(observations.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
      out[i] = Vec3{0, 0, 0};
    }
    return out;
  }
};

}  // namespace quasar::numerics

#define QUASAR_REGISTER_FIELD_EVALUATOR(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::numerics::IFieldEvaluator, Name, Class)
