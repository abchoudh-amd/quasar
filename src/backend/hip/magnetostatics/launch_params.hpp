#pragma once

namespace quasar::magnetostatics::detail {

// Compile-time launch tunables for the Biot-Savart kernels on gfx942.
// Phase 4 will lift these to a per-gfx-target table.
inline constexpr int kTileSegments = 128;
inline constexpr int kBlockSize    = 256;

}  // namespace quasar::magnetostatics::detail
