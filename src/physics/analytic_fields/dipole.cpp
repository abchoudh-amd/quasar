#include "quasar/physics/analytic_fields/dipole.hpp"

#include "quasar/physics/magnetostatics/observation.hpp"

namespace quasar::analytic_fields {

Field<Vec3> DipoleEvaluator::evaluate_B(const magnetostatics::ConductorSystem&,
                                        const magnetostatics::PointCloud& obs) const {
  Field<Vec3> out(obs.size());
  const auto& pts = obs.points();
  for (std::size_t i = 0; i < out.size(); ++i) {
    const Vec3 r = pts[i] - origin_;
    const Real r2 = length_squared(r);
    if (r2 <= kEps) {
      out[i] = Vec3{0, 0, 0};
      continue;
    }
    const Real rlen = std::sqrt(r2);
    const Real r5 = r2 * r2 * rlen;
    out[i] = (mu0_over_4pi / r5) * (Real{3} * dot(moment_, r) * r - r2 * moment_);
  }
  return out;
}

Field<Mat3x3> DipoleEvaluator::evaluate_grad_B(const magnetostatics::ConductorSystem&,
                                               const magnetostatics::PointCloud& obs) const {
  Field<Mat3x3> out(obs.size());
  for (std::size_t i = 0; i < out.size(); ++i) out[i] = Mat3x3{};
  return out;
}

QUASAR_REGISTER_FIELD_EVALUATOR("dipole", DipoleEvaluator)

}  // namespace quasar::analytic_fields
