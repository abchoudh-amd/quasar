#include "quasar/physics/pic/pic_solver.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <vector>

namespace quasar::pic {

namespace {

// Builds the Yee-staggered sample points for one field component already scaled
// from internal length units to SI (factor length_scale), so the SI field
// evaluator sees physical coordinates. Folding the scale in here avoids a second
// full PointCloud copy (the former to_si_points pass).
magnetostatics::PointCloud yee_points(const Grid2D& g, int component,
                                      Real length_scale) {
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
      pts.add(Vec3{x * length_scale, y * length_scale, Real{0}});
    }
  }
  return pts;
}

void copy_component(const Grid2D& g, const Field<Vec3>& values, int axis,
                    Real field_scale, backend::DeviceBuffer<Real>& dst) {
  std::vector<Real> host(g.storage_size(), Real{0});
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const Vec3 v = values[static_cast<std::size_t>(i + g.nx * j)];
      const Real c = axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
      host[g.index(i, j)] = c / field_scale;
    }
  }
  // Replicate the interior edge into the ghost layer. The particle gather clamps
  // its interpolation stencil into the ghost cells on a non-periodic axis, so an
  // unfilled (zero) ghost would otherwise bias the external force on near-wall
  // particles; edge replication is the natural Neumann fill for a sampled field.
  for (int gh = 1; gh <= g.nghost; ++gh) {
    for (int j = 0; j < g.ny; ++j) {
      host[g.index(-gh, j)]          = host[g.index(0, j)];
      host[g.index(g.nx - 1 + gh, j)] = host[g.index(g.nx - 1, j)];
    }
    for (int i = -g.nghost; i < g.nx + g.nghost; ++i) {
      const int ic = i < 0 ? 0 : (i > g.nx - 1 ? g.nx - 1 : i);
      host[g.index(i, -gh)]          = host[g.index(ic, 0)];
      host[g.index(i, g.ny - 1 + gh)] = host[g.index(ic, g.ny - 1)];
    }
  }
  dst.copy_from_host(host.data(), host.size());
}

}  // namespace

void sample_external_field(numerics::IFieldEvaluator& evaluator,
                           const magnetostatics::ConductorSystem& conductors,
                           YeeField2D<Real>& external_fields,
                           Real length_scale, Real e_field_scale,
                           Real b_field_scale) {
  const Grid2D g = external_fields.grid;
  auto ex_pts = yee_points(g, 0, length_scale);
  auto ey_pts = yee_points(g, 1, length_scale);
  auto ez_pts = yee_points(g, 2, length_scale);
  auto bx_pts = yee_points(g, 3, length_scale);
  auto by_pts = yee_points(g, 4, length_scale);
  auto bz_pts = yee_points(g, 5, length_scale);

  copy_component(g, evaluator.evaluate_E(conductors, ex_pts), 0, e_field_scale, external_fields.ex);
  copy_component(g, evaluator.evaluate_E(conductors, ey_pts), 1, e_field_scale, external_fields.ey);
  copy_component(g, evaluator.evaluate_E(conductors, ez_pts), 2, e_field_scale, external_fields.ez);
  copy_component(g, evaluator.evaluate_B(conductors, bx_pts), 0, b_field_scale, external_fields.bx);
  copy_component(g, evaluator.evaluate_B(conductors, by_pts), 1, b_field_scale, external_fields.by);
  copy_component(g, evaluator.evaluate_B(conductors, bz_pts), 2, b_field_scale, external_fields.bz);
}

}  // namespace quasar::pic
