#include "quasar/boundary/outflow.hpp"

#include "quasar/physics/pic/kernels.hpp"

#include "order_dispatch.hpp"

namespace quasar::boundary {

void OutflowFieldBC::correct_after_b(YeeField2D<Real>& field, Side side, Real dt) const {
  const int s = static_cast<int>(side);
  detail::select_order(
      order_,
      [&] { ::launch_pic_boundary_outflow_correct_b_order2(field.grid, field, s, dt, nullptr); },
      [&] { ::launch_pic_boundary_outflow_correct_b_order4(field.grid, field, s, dt, nullptr); });
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
  const int slo = skip_lo_ ? 1 : 0;
  const int shi = skip_hi_ ? 1 : 0;
  const int s = static_cast<int>(side);
  Real* strips = mur_.device_ptr();
  const int init_flag = init ? 1 : 0;
  detail::select_order(
      order_,
      [&] {
        ::launch_pic_boundary_outflow_correct_e_order2(
            g, field, s, dt, strips, init_flag, slo, shi, nullptr);
      },
      [&] {
        ::launch_pic_boundary_outflow_correct_e_order4(
            g, field, s, dt, strips, init_flag, slo, shi, nullptr);
      });
}

QUASAR_REGISTER_FIELD_BOUNDARY("outflow", OutflowFieldBC)

}  // namespace quasar::boundary
