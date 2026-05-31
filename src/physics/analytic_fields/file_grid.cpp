#include "quasar/physics/analytic_fields/file_grid.hpp"

#include "quasar/core/observations.hpp"

#include <stdexcept>

namespace quasar::analytic_fields {

// The file-backed grid evaluator is registered (so the deck name "file_grid" is
// reserved for it) but not yet implemented. It is intentionally absent from the
// Python deck surface (SUPPORTED_EVALUATORS in quasar/pic/io.py and the coil
// evaluator list) until the loader lands, so a deck cannot select it. The methods
// fail loudly instead of silently returning a zero field, which would look like a
// valid-but-trivial result.
Field<Vec3> FileGridEvaluator::evaluate_B(const core::IFieldSource&,
                                          const core::PointCloud&) const {
  throw std::logic_error{"file_grid evaluator is not yet implemented"};
}

Field<Mat3x3> FileGridEvaluator::evaluate_grad_B(const core::IFieldSource&,
                                                 const core::PointCloud&) const {
  throw std::logic_error{"file_grid evaluator is not yet implemented"};
}

QUASAR_REGISTER_FIELD_EVALUATOR("file_grid", FileGridEvaluator)

}  // namespace quasar::analytic_fields
