#pragma once

#include "quasar/core/field.hpp"
#include "quasar/core/field_source.hpp"
#include "quasar/core/observations.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/core/types.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace quasar::numerics {

// Post-construction parameters for a registry-built evaluator, keyed by name.
// Each value is a flat Real list: a Vec3 is 3 elements, a Mat3x3 is 9 elements
// row-major. This lets the registry build any evaluator by name (default
// construct) and then configure() it from the deck, so a driver never branches on
// the evaluator type to hand-pick a constructor. Keys an evaluator does not know
// are ignored; an evaluator validates the arity of keys it consumes.
using EvaluatorParams = std::unordered_map<std::string, std::vector<Real>>;

// Reads a Vec3 parameter, returning `fallback` if absent. Throws if present with
// the wrong arity.
inline Vec3 param_vec3(const EvaluatorParams& p, const std::string& key,
                       Vec3 fallback = Vec3{0, 0, 0}) {
  const auto it = p.find(key);
  if (it == p.end()) return fallback;
  if (it->second.size() != 3) {
    throw std::invalid_argument{"evaluator param '" + key + "' must have 3 elements"};
  }
  return Vec3{it->second[0], it->second[1], it->second[2]};
}

// Reads a row-major Mat3x3 parameter, returning `fallback` if absent. Throws if
// present with the wrong arity.
inline Mat3x3 param_mat3x3(const EvaluatorParams& p, const std::string& key,
                           Mat3x3 fallback = Mat3x3{}) {
  const auto it = p.find(key);
  if (it == p.end()) return fallback;
  if (it->second.size() != 9) {
    throw std::invalid_argument{"evaluator param '" + key + "' must have 9 elements"};
  }
  const auto& v = it->second;
  return Mat3x3{Vec3{v[0], v[1], v[2]}, Vec3{v[3], v[4], v[5]}, Vec3{v[6], v[7], v[8]}};
}

// The evaluator's observation set is the axis-neutral core::PointCloud and its
// source is the axis-neutral core::IFieldSource, so the numerics axis depends on
// no concrete physics type. An evaluator that needs a specific source (e.g.
// Biot-Savart) downcasts it; evaluators that ignore the source (the analytic
// fields) name no physics type at all.
class IFieldEvaluator {
 public:
  virtual ~IFieldEvaluator() = default;

  // Applies deck-supplied parameters after the registry default-constructs the
  // evaluator. The base default ignores all params (so a parameterless evaluator
  // like Biot-Savart needs no override); parameterized evaluators override to read
  // the keys they consume via param_vec3 / param_mat3x3. This is the seam that
  // lets a driver do create_field_evaluator(name) + configure(params) with no
  // per-type branch.
  virtual void configure(const EvaluatorParams&) {}

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
