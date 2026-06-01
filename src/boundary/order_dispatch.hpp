#pragma once

namespace quasar::boundary::detail {

// Selects the 2nd- vs 4th-order field-boundary launcher in one place so the
// `order_ == 4` decision is not repeated across every correct_after_* method.
template <class F2, class F4>
inline void select_order(int order, F2&& launch_o2, F4&& launch_o4) {
  if (order == 4) {
    launch_o4();
  } else {
    launch_o2();
  }
}

}  // namespace quasar::boundary::detail
