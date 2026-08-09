#pragma once

namespace quasar::distributed::detail {

// Physics-neutral state shared by tile runtimes.  Keeping these transitions
// in one representation prevents PIC and MHD from silently growing different
// lifecycle layouts and makes collective poisoning operate on the same state.
struct RuntimeLifecycleState {
  bool seeded{false};
  bool poisoned{false};
  bool closed{false};
};

}  // namespace quasar::distributed::detail
