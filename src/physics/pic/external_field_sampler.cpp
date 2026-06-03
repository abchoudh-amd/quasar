#include "quasar/physics/pic/pic_solver.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/core/observations.hpp"

#include <string_view>
#include <vector>

namespace quasar::pic {

namespace {

// Builds the cell-node sample points already scaled from internal length units to
// SI (factor length_scale), so the SI field evaluator sees physical coordinates.
// Folding the scale in here avoids a second full PointCloud copy (the former
// to_si_points pass).
//
// Every component samples at the same cell-node location because the rest of the
// solver treats the stored field as node-collocated: the particle gather
// (shape_weights_2d) interpolates each component from the cell node (i,j), and
// the scalar energy/Gauss diagnostics colocate the components there too. Sampling
// the external E at a true Yee-staggered edge instead would place the value half a
// cell from where the gather reads it, biasing the external electric force on
// every particle.
core::PointCloud node_points(const Grid2D& g, Real length_scale,
                             std::string_view plane) {
  core::PointCloud pts;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const Real a = g.x_at_cell_center(i) * length_scale;  // grid-i -> lab x
      const Real b = g.y_at_cell_center(j) * length_scale;  // grid-j -> lab y or z
      // "xy": sample the lab z=0 slice at (x, y, 0).
      // "xz": sample the lab y=0 meridional slice at (x, 0, z).
      if (plane == "xz") {
        pts.add(Vec3{a, Real{0}, b});
      } else {
        pts.add(Vec3{a, b, Real{0}});
      }
    }
  }
  return pts;
}

// Writes one PIC-frame component, reading lab axis `axis` (0=x,1=y,2=z) of the
// evaluated field and scaling by `sign` (+1 / -1) for the right-handed frame map.
void copy_component(const Grid2D& g, const Field<Vec3>& values, int axis,
                    Real sign, Real field_scale,
                    backend::DeviceBuffer<Real>& dst) {
  std::vector<Real> host(g.storage_size(), Real{0});
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const Vec3 v = values[static_cast<std::size_t>(i + g.nx * j)];
      const Real c = axis == 0 ? v.x : (axis == 1 ? v.y : v.z);
      host[g.index(i, j)] = sign * c / field_scale;
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
                           const core::IFieldSource& source,
                           YeeField2D<Real>& external_fields,
                           Real length_scale, Real e_field_scale,
                           Real b_field_scale, std::string_view plane) {
  const Grid2D g = external_fields.grid;
  // All six components are node-collocated, so one point set drives every
  // evaluate. The E and B evaluations are kept separate because IFieldEvaluator
  // exposes evaluate_E / evaluate_B as distinct calls.
  const auto pts = node_points(g, length_scale, plane);
  const auto e = evaluator.evaluate_E(source, pts);
  const auto b = evaluator.evaluate_B(source, pts);

  // Map each PIC-frame component (ex/ey/ez, bx/by/bz) to a lab axis and sign.
  // "xy": identity. "xz": a right-handed 90-degree rotation about lab x so the
  // grid triad (x, z, -y) stays right-handed -> the j-axis component reads lab z
  // and the out-of-plane component reads -lab y.
  struct Map { int axis; Real sign; };
  Map mx, my, mz;
  if (plane == "xz") {
    mx = {0, Real{+1}};   // pic-x  <- lab x
    my = {2, Real{+1}};   // pic-y  <- lab z
    mz = {1, Real{-1}};   // pic-z  <- -lab y  (out-of-plane)
  } else {
    mx = {0, Real{+1}};
    my = {1, Real{+1}};
    mz = {2, Real{+1}};
  }

  copy_component(g, e, mx.axis, mx.sign, e_field_scale, external_fields.ex);
  copy_component(g, e, my.axis, my.sign, e_field_scale, external_fields.ey);
  copy_component(g, e, mz.axis, mz.sign, e_field_scale, external_fields.ez);
  copy_component(g, b, mx.axis, mx.sign, b_field_scale, external_fields.bx);
  copy_component(g, b, my.axis, my.sign, b_field_scale, external_fields.by);
  copy_component(g, b, mz.axis, mz.sign, b_field_scale, external_fields.bz);
}

}  // namespace quasar::pic
