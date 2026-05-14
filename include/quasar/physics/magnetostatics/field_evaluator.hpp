#pragma once

#include "quasar/core/field.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/core/types.hpp"

namespace quasar::magnetostatics {

class ConductorSystem;
class PointCloud;

// Abstract base for any magnetic-field evaluator that consumes a
// ConductorSystem + PointCloud and produces B(r) (and its Jacobian) at every
// observation point.
class IFieldEvaluator {
 public:
  virtual ~IFieldEvaluator() = default;

  virtual Field<Vec3> evaluate_B(const ConductorSystem& conductors,
                                 const PointCloud&      observations) const = 0;

  // Jacobian: result[m].r{ij} = dB_i / dp_j evaluated at observation point m.
  virtual Field<Mat3x3> evaluate_grad_B(const ConductorSystem& conductors,
                                        const PointCloud&      observations) const = 0;
};

}  // namespace quasar::magnetostatics

// Axis-specific sugar over the generic Registry macro.
#define QUASAR_REGISTER_FIELD_EVALUATOR(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::magnetostatics::IFieldEvaluator, Name, Class)
