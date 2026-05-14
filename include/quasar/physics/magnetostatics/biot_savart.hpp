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

class BiotSavartEvaluator final : public IFieldEvaluator {
 public:
  BiotSavartEvaluator();
  explicit BiotSavartEvaluator(BiotSavartConfig cfg);

  Field<Vec3> evaluate_B(const ConductorSystem& conductors,
                         const PointCloud&      observations) const override;

  const BiotSavartConfig& config() const noexcept { return cfg_; }

 private:
  BiotSavartConfig cfg_{};
};

}  // namespace quasar::magnetostatics
