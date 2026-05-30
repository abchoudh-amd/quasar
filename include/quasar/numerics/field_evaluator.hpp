#pragma once

#include "quasar/core/field.hpp"
#include "quasar/core/observations.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/core/types.hpp"

namespace quasar::magnetostatics {
class ConductorSystem;
}  // namespace quasar::magnetostatics

namespace quasar::numerics {

// The evaluator's observation set is the axis-neutral core::PointCloud. The
// source is a magnetostatics::ConductorSystem: it is the one named field source
// in the codebase, so it remains the source contract rather than introducing an
// abstract-source seam the analytic evaluators (which ignore the source) would
// not benefit from.
class IFieldEvaluator {
 public:
  virtual ~IFieldEvaluator() = default;

  virtual Field<Vec3> evaluate_B(const magnetostatics::ConductorSystem& conductors,
                                 const core::PointCloud& observations) const = 0;

  virtual Field<Mat3x3> evaluate_grad_B(const magnetostatics::ConductorSystem& conductors,
                                        const core::PointCloud& observations) const = 0;

  virtual Field<Vec3> evaluate_E(const magnetostatics::ConductorSystem&,
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
