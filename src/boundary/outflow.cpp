#include "quasar/boundary/outflow.hpp"

#include "quasar/backend/pic_kernels.hpp"

namespace quasar::boundary {

void OutflowFieldBC::correct_after_b(YeeField2D<Real>& field, Side side, Real dt) const {
  // Order 4 falls back to the order-2 closure for now (the outer-layer 4th-order
  // outflow closure is added with the 4th-order outflow commit).
  ::launch_pic_boundary_outflow_correct_b_order2(field.grid, field,
                                                 static_cast<int>(side), dt, nullptr);
}

void OutflowFieldBC::correct_after_e(YeeField2D<Real>& field, Side side, Real dt) const {
  const Grid2D& g = field.grid;
  const int line = (side == Side::x_lo || side == Side::x_hi) ? g.ny : g.nx;
  const std::size_t need = static_cast<std::size_t>(4 * line);
  bool init = false;
  if (mur_.size() < need) {
    mur_ = backend::DeviceBuffer<Real>{need};  // zero-initialized
    primed_ = false;
  }
  if (!primed_) {
    init = true;       // first call: seed the history from the live field only
    primed_ = true;
  }
  ::launch_pic_boundary_outflow_correct_e_order2(
      g, field, static_cast<int>(side), dt, mur_.device_ptr(), init ? 1 : 0, nullptr);
}

QUASAR_REGISTER_FIELD_BOUNDARY("outflow", OutflowFieldBC)

}  // namespace quasar::boundary
