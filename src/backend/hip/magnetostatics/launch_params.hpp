#pragma once

// Compile-time launch tunables for the Biot-Savart HIP kernels.
//
// Values are supplied by cmake/QuasarLaunchParams.cmake, which picks them
// from the first entry of CMAKE_HIP_ARCHITECTURES (so host launchers and
// device kernels agree). The conservative defaults below kick in only when
// the helper isn't called (e.g. a downstream consumer linking the headers
// without our CMake support).
//
// Tuning data: the per-gfx defaults in QuasarLaunchParams.cmake are baseline
// starting points; refinement against the actual benchmarks/micro/
// biot_savart_bench.cpp sweep is tracked in Phase 4.B.

#ifndef QUASAR_BS_TILE_SEGMENTS
#  define QUASAR_BS_TILE_SEGMENTS 64
#endif
#ifndef QUASAR_BS_BLOCK_SIZE
#  define QUASAR_BS_BLOCK_SIZE 128
#endif

namespace quasar::magnetostatics::detail {

inline constexpr int kTileSegments = QUASAR_BS_TILE_SEGMENTS;
inline constexpr int kBlockSize    = QUASAR_BS_BLOCK_SIZE;

}  // namespace quasar::magnetostatics::detail
