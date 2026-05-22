#include "quasar/physics/analytic_fields/file_grid.hpp"

#include "quasar/physics/magnetostatics/observation.hpp"

namespace quasar::analytic_fields {

Field<Vec3> FileGridEvaluator::evaluate_B(const magnetostatics::ConductorSystem&,
                                          const magnetostatics::PointCloud& obs) const {
  Field<Vec3> out(obs.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = Vec3{0, 0, 0};
  return out;
}

Field<Mat3x3> FileGridEvaluator::evaluate_grad_B(const magnetostatics::ConductorSystem&,
                                                 const magnetostatics::PointCloud& obs) const {
  Field<Mat3x3> out(obs.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = Mat3x3{};
  return out;
}

QUASAR_REGISTER_FIELD_EVALUATOR("file_grid", FileGridEvaluator)

}  // namespace quasar::analytic_fields
