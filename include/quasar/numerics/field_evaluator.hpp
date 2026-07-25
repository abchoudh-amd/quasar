#pragma once

#include "quasar/core/field.hpp"
#include "quasar/core/field_source.hpp"
#include "quasar/core/observations.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/core/types.hpp"

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace quasar::numerics {

// Post-construction parameters for a registry-built evaluator, keyed by name.
// Each value is a flat Real list: a Vec3 is 3 elements, a Mat3x3 is 9 elements
// row-major. This lets the registry build any evaluator by name (default
// construct) and then configure() it from the deck, so a driver never branches on
// the evaluator type to hand-pick a constructor. Evaluators reject unknown keys
// and validate the arity of every key they consume.
using EvaluatorParams = std::unordered_map<std::string, std::vector<Real>>;

inline void reject_unknown_params(
    const EvaluatorParams& params,
    std::initializer_list<std::string_view> allowed,
    std::string_view evaluator_name) {
  for (const auto& [key, value] : params) {
    (void)value;
    bool known = false;
    for (const std::string_view candidate : allowed) {
      if (key == candidate) {
        known = true;
        break;
      }
    }
    if (!known) {
      throw std::invalid_argument{
          std::string{evaluator_name} + ": unknown evaluator parameter '" + key + "'"};
    }
  }
}

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

  // Capability query used by deck frontends before requesting an optional
  // vector potential.  New evaluator plugins remain source-compatible and
  // truthfully default to B-only until they override both this method and
  // evaluate_A().
  virtual bool provides_vector_potential() const noexcept { return false; }

  // Applies deck-supplied parameters after the registry default-constructs the
  // evaluator. The base default accepts only an empty map (so a parameterless
  // evaluator like Biot-Savart needs no override); parameterized evaluators
  // override to declare and read their keys via the helpers above. This is the seam that
  // lets a driver do create_field_evaluator(name) + configure(params) with no
  // per-type branch.
  virtual void configure(const EvaluatorParams& params) {
    reject_unknown_params(params, {}, "field evaluator");
  }

  virtual Field<Vec3> evaluate_B(const core::IFieldSource& source,
                                 const core::PointCloud& observations) const = 0;

  // Capability query for a trustworthy field gradient.  The base
  // evaluate_grad_B implementation remains the exact zero Jacobian for simple
  // callers, but a physics validator must not mistake that compatibility
  // default for evidence that an arbitrary plugin's magnetic field is uniform.
  // Evaluators that can return their actual Jacobian -- including an exactly
  // uniform evaluator -- override this together with evaluate_grad_B when
  // necessary.
  virtual bool provides_grad_B() const noexcept { return false; }

  // Field gradient (grad B)_{ij} = dB_i/dp_j.
  virtual Field<Mat3x3> evaluate_grad_B(const core::IFieldSource&,
                                        const core::PointCloud& observations) const {
    return Field<Mat3x3>(observations.size());
  }

  // Magnetic vector potential A, with B = curl A. Its gauge is evaluator- and
  // source-dependent. Defaults to
  // throwing: most evaluators expose only B, and a caller that needs A (e.g. a
  // divergence-free MHD seed built from a discrete curl of A) must select an
  // evaluator that models it. Biot-Savart overrides with the closed-form
  // line-integral A.
  virtual Field<Vec3> evaluate_A(const core::IFieldSource&,
                                 const core::PointCloud&) const {
    throw std::runtime_error{
        "this field evaluator does not provide a vector potential A"};
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
