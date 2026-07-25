#include "quasar/boundary/outflow.hpp"

#include "quasar/physics/pic/kernels.hpp"

#include <limits>
#include <stdexcept>

namespace quasar::boundary {

int OutflowFieldBC::history_stride(const Grid2D& g, Side side) const {
  return (side == Side::x_lo || side == Side::x_hi) ? g.ny + 1 : g.nx + 1;
}

void OutflowFieldBC::ensure_history(const Grid2D& g, Side side) const {
  const int stride = history_stride(g, side);
  const std::size_t stride_size = static_cast<std::size_t>(stride);
  if (stride_size > std::numeric_limits<std::size_t>::max()
                        / (4u * sizeof(Real))) {
    throw std::overflow_error{
        "OutflowFieldBC: Mur history allocation is not representable"};
  }
  const std::size_t need = 4u * stride_size;
  if (mur_.size() < need) {
    mur_ = backend::DeviceBuffer<Real>{need};
    primed_ = false;
  }
}

void OutflowFieldBC::fill_ghosts(YeeField2D<Real>& field, Side side) const {
  ::launch_pic_boundary_outflow_fill_fields(
      field.grid, field, static_cast<int>(side), cylindrical_ ? 1 : 0, nullptr);
}

void OutflowFieldBC::correct_after_b(YeeField2D<Real>& field, Side side, Real dt) const {
  fill_ghosts(field, side);
  ensure_history(field.grid, side);
  if (primed_) return;
  const int stride = history_stride(field.grid, side);
  ::launch_pic_boundary_outflow_correct_e(
      field.grid, field, static_cast<int>(side), dt, mur_.device_ptr(), stride,
      /*init=*/1, skip_lo_ ? 1 : 0, skip_hi_ ? 1 : 0,
      cylindrical_ ? 1 : 0, nullptr);
  // E is still at t^n here (Ampere has not run), so this captures the actual
  // old wall/adjacent state required by the first characteristic update.
  primed_ = true;
}

void OutflowFieldBC::correct_after_e(YeeField2D<Real>& field, Side side, Real dt) const {
  const Grid2D& g = field.grid;
  // One tangential component is face-located along the transverse axis, so the
  // history stride includes its physical high face.  The second component uses
  // the leading subset of the same strips.
  ensure_history(g, side);
  if (!primed_) {
    throw std::logic_error{
        "OutflowFieldBC: history was not primed before the E update"};
  }
  const int stride = history_stride(g, side);
  const int slo = skip_lo_ ? 1 : 0;
  const int shi = skip_hi_ ? 1 : 0;
  ::launch_pic_boundary_outflow_correct_e(
      g, field, static_cast<int>(side), dt, mur_.device_ptr(), stride,
      /*init=*/0, slo, shi, cylindrical_ ? 1 : 0, nullptr);
  // Mur changed the outer physical samples; refresh the linear continuation so
  // near-wall gathers and the next curl see the same boundary state.
  fill_ghosts(field, side);
}

QUASAR_REGISTER_FIELD_BOUNDARY("outflow", OutflowFieldBC)

}  // namespace quasar::boundary
