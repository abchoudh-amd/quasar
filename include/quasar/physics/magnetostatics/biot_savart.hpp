#pragma once

#include "quasar/backend/device.hpp"
#include "quasar/core/field.hpp"
#include "quasar/core/observations.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/field_evaluator.hpp"

namespace quasar::magnetostatics {

class ConductorSystem;
using core::PointCloud;

// Runtime configuration for BiotSavartEvaluator. Kernel tiling (shared-memory
// tile width and thread-block size) is a COMPILE-TIME concern: the tile sizes a
// __shared__ array, and both are tuned per-gfx in cmake/QuasarLaunchParams.cmake
// (see launch_params.hpp). The only runtime knob is the device stream.
struct BiotSavartConfig {
  quasar::backend::stream_t stream = nullptr;
};

// Double-precision evaluator. This is the only class that participates in
// Registry<IFieldEvaluator> (the registered "biot_savart" factory). Returns
// magnetic flux density and its Jacobian as host-side Field<Vec3> /
// Field<Mat3x3>, both fp64.
class BiotSavartEvaluator final : public IFieldEvaluator {
 public:
  BiotSavartEvaluator();
  explicit BiotSavartEvaluator(BiotSavartConfig cfg);

  // The IFieldEvaluator contract is axis-neutral (core::IFieldSource); this
  // evaluator downcasts to ConductorSystem, throwing std::invalid_argument if
  // handed a source of another type.
  Field<Vec3>   evaluate_B     (const core::IFieldSource& source,
                                const PointCloud&         observations) const override;

  bool provides_grad_B() const noexcept override { return true; }
  Field<Mat3x3> evaluate_grad_B(const core::IFieldSource& source,
                                const PointCloud&         observations) const override;

  // Closed-form line-integral magnetic vector potential A, with B = curl A.
  // It is in Coulomb gauge for closed/current-continuous conductor systems;
  // individual open segments retain endpoint divergence terms. Used to seed a
  // discretely divergence-free in-plane field for MHD.
  bool provides_vector_potential() const noexcept override { return true; }
  Field<Vec3>   evaluate_A     (const core::IFieldSource& source,
                                const PointCloud&         observations) const override;

  const BiotSavartConfig& config() const noexcept { return cfg_; }

 private:
  BiotSavartConfig cfg_{};
};

// Single-precision sibling of BiotSavartEvaluator. The kernel runs in fp32
// throughout - a common double-precision origin is subtracted from segments and
// observations before casting the translated coordinates to float, the kernel uses the
// fp32 instantiation of segment_B / segment_gradB, and the result is
// returned as Field<Vec3f> / Field<Mat3x3f> so callers see the precision
// difference directly. The IFieldEvaluator base is intentionally not
// implemented here because its return type pins double precision; tests
// that want to compare the two precisions construct this class directly.
class BiotSavartEvaluatorF final {
 public:
  BiotSavartEvaluatorF();
  explicit BiotSavartEvaluatorF(BiotSavartConfig cfg);

  Field<Vec3f>   evaluate_B     (const ConductorSystem& conductors,
                                 const PointCloud&      observations) const;

  Field<Mat3x3f> evaluate_grad_B(const ConductorSystem& conductors,
                                 const PointCloud&      observations) const;

  Field<Vec3f>   evaluate_A     (const ConductorSystem& conductors,
                                 const PointCloud&      observations) const;

  const BiotSavartConfig& config() const noexcept { return cfg_; }

 private:
  BiotSavartConfig cfg_{};
};

}  // namespace quasar::magnetostatics
