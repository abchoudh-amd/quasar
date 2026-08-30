#pragma once

// Host-side convenience wrappers around the device-resident IFieldEvaluator
// interface, for tests and benchmarks only.
//
// `evaluate_B` and friends take a core::DevicePointCloud and return
// core::DeviceVectorField / core::DeviceTensorField, because a production
// caller chains evaluations and must not round-trip through host memory between
// them. A test is the opposite: it has a handful of points and wants to assert
// on the numbers. That is an output boundary, and .to_host() is the sanctioned
// way to cross one.
//
// These wrappers deliberately live under tests/ rather than on the interface.
// Putting them on IFieldEvaluator would give production code a one-token way to
// reintroduce exactly the host round trip this interface exists to remove; here
// they are unreachable from src/.

#include "quasar/core/device_observations.hpp"
#include "quasar/core/field.hpp"
#include "quasar/core/field_source.hpp"
#include "quasar/core/observations.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/field_evaluator.hpp"

#include <type_traits>

namespace quasar::test {

// Templated on the evaluator so a test can hand the same generic lambda both
// BiotSavartEvaluator (device SoA, fp64) and BiotSavartEvaluatorF (host, fp32)
// and compare them -- which is exactly what the precision-comparison tests do.
// The fp32 sibling is not an IFieldEvaluator and already returns host values, so
// for it these are pass-throughs.
template <class Eval>
inline constexpr bool is_device_evaluator_v =
    std::is_base_of_v<numerics::IFieldEvaluator, std::decay_t<Eval>>;

template <class Eval, class Source>
auto host_evaluate_B(const Eval& evaluator, const Source& source,
                     const core::PointCloud& points) {
  if constexpr (is_device_evaluator_v<Eval>) {
    return evaluator.evaluate_B(source, core::DevicePointCloud::upload(points))
        .to_host();
  } else {
    return evaluator.evaluate_B(source, points);
  }
}

template <class Eval, class Source>
auto host_evaluate_E(const Eval& evaluator, const Source& source,
                     const core::PointCloud& points) {
  if constexpr (is_device_evaluator_v<Eval>) {
    return evaluator.evaluate_E(source, core::DevicePointCloud::upload(points))
        .to_host();
  } else {
    return evaluator.evaluate_E(source, points);
  }
}

template <class Eval, class Source>
auto host_evaluate_A(const Eval& evaluator, const Source& source,
                     const core::PointCloud& points) {
  if constexpr (is_device_evaluator_v<Eval>) {
    return evaluator.evaluate_A(source, core::DevicePointCloud::upload(points))
        .to_host();
  } else {
    return evaluator.evaluate_A(source, points);
  }
}

template <class Eval, class Source>
auto host_evaluate_grad_B(const Eval& evaluator, const Source& source,
                          const core::PointCloud& points) {
  if constexpr (is_device_evaluator_v<Eval>) {
    return evaluator
        .evaluate_grad_B(source, core::DevicePointCloud::upload(points))
        .to_host();
  } else {
    return evaluator.evaluate_grad_B(source, points);
  }
}

}  // namespace quasar::test
