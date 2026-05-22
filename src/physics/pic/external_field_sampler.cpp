#include "quasar/physics/pic/pic_solver.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <vector>

namespace quasar::pic {

namespace {

magnetostatics::PointCloud yee_points(const Grid2D& g, int component) {
  magnetostatics::PointCloud pts;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      Real x = g.x_at_cell_center(i);
      Real y = g.y_at_cell_center(j);
      if (component == 0) {
        x = g.x_at_ex(i);
        y = g.y_at_ex(j);
      } else if (component == 1) {
        x = g.x_at_ey(i);
        y = g.y_at_ey(j);
      } else if (component == 5) {
        x = g.x_at_bz(i);
        y = g.y_at_bz(j);
      }
      pts.add(Vec3{x, y, 0});
    }
  }
  return pts;
}

void copy_component(const Grid2D& g, const Field<Vec3>& values, int axis,
                    backend::DeviceBuffer<Real>& dst) {
  std::vector<Real> host(g.storage_size(), Real{0});
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const Vec3 v = values[static_cast<std::size_t>(i + g.nx * j)];
      host[g.index(i, j)] = axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
    }
  }
  dst.copy_from_host(host.data(), host.size());
}

}  // namespace

void sample_external_field(numerics::IFieldEvaluator& evaluator,
                           const magnetostatics::ConductorSystem& conductors,
                           YeeField2D<Real>& external_fields) {
  const Grid2D g = external_fields.grid;
  auto ex_pts = yee_points(g, 0);
  auto ey_pts = yee_points(g, 1);
  auto ez_pts = yee_points(g, 2);
  auto bx_pts = yee_points(g, 3);
  auto by_pts = yee_points(g, 4);
  auto bz_pts = yee_points(g, 5);

  copy_component(g, evaluator.evaluate_E(conductors, ex_pts), 0, external_fields.ex);
  copy_component(g, evaluator.evaluate_E(conductors, ey_pts), 1, external_fields.ey);
  copy_component(g, evaluator.evaluate_E(conductors, ez_pts), 2, external_fields.ez);
  copy_component(g, evaluator.evaluate_B(conductors, bx_pts), 0, external_fields.bx);
  copy_component(g, evaluator.evaluate_B(conductors, by_pts), 1, external_fields.by);
  copy_component(g, evaluator.evaluate_B(conductors, bz_pts), 2, external_fields.bz);
}

}  // namespace quasar::pic
