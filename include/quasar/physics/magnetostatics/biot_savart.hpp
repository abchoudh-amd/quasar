#pragma once

#include "quasar/backend/device.hpp"
#include "quasar/core/field.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/field_evaluator.hpp"

namespace quasar::magnetostatics {

class ConductorSystem;
class PointCloud;

// Runtime configuration knobs for BiotSavartEvaluator. The tile/block fields
// are documented hints; Phase 1 uses compile-time defaults in launch_params.hpp
// and only `stream` is consumed by the orchestrator. Per-gfx tuning of these
// hints arrives in Phase 4.
struct BiotSavartConfig {
  int                       tile_segments = 128;
  int                       block_size    = 256;
  quasar::backend::stream_t stream        = nullptr;
};

// Double-precision evaluator. This is the only class that participates in
// Registry<IFieldEvaluator> (the registered "biot_savart" factory). Returns
// magnetic flux density and its Jacobian as host-side Field<Vec3> /
// Field<Mat3x3>, both fp64.
class BiotSavartEvaluator final : public IFieldEvaluator {
 public:
  BiotSavartEvaluator();
  explicit BiotSavartEvaluator(BiotSavartConfig cfg);

  Field<Vec3>   evaluate_B     (const ConductorSystem& conductors,
                                const PointCloud&      observations) const override;

  Field<Mat3x3> evaluate_grad_B(const ConductorSystem& conductors,
                                const PointCloud&      observations) const override;

  const BiotSavartConfig& config() const noexcept { return cfg_; }

 private:
  BiotSavartConfig cfg_{};
};

// Single-precision sibling of BiotSavartEvaluator. The kernel runs in fp32
// throughout - segments and observation points are cast from the
// double-typed host containers to float on upload, the kernel uses the
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

  const BiotSavartConfig& config() const noexcept { return cfg_; }

 private:
  BiotSavartConfig cfg_{};
};

}  // namespace quasar::magnetostatics
