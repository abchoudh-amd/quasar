// HIP-only launch-configuration helpers shared by the PIC backend kernels.
// Included only by the .hip definitions under src/backend/hip/pic/ (it pulls in
// <hip/hip_runtime.h>); never include from non-HIP TUs.
#pragma once

#include "quasar/core/grid.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace quasar::backend::pic {

inline constexpr unsigned kLaunchBlock = 256;

// Standard 2-D thread block for full-grid (nx x ny) field/filter kernels.
inline constexpr unsigned kLaunchBlock2D = 16;

// 1-D launch grid covering n threads at the standard block size.
inline dim3 grid_1d(std::size_t n, unsigned block = kLaunchBlock) {
  return dim3(static_cast<unsigned>((n + block - 1) / block));
}

// 2-D launch grid covering the full nx x ny interior at `block` (default 16x16),
// shared by the FDTD E/B updates and the current filter so the tiling lives in
// one place.
inline dim3 grid_2d(const quasar::Grid2D& g, dim3 block = dim3(kLaunchBlock2D, kLaunchBlock2D)) {
  return dim3((static_cast<unsigned>(g.nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(g.ny) + block.y - 1) / block.y);
}

// Per-side field-BC launch. A x-face side (0/1) iterates over the ny boundary
// column; a y-face side (2/3) over the nx boundary row. Computes the 1-D grid for
// the relevant extent (skipping empty extents) and invokes the matching functor
// with (grid, block); each functor issues its own hipLaunchKernelGGL.
template <class LaunchX, class LaunchY>
inline void launch_along_side(const quasar::Grid2D& g, int side,
                              LaunchX&& launch_x, LaunchY&& launch_y) {
  const dim3 block(kLaunchBlock);
  if (side == 0 || side == 1) {
    if (g.ny <= 0) return;
    launch_x(grid_1d(static_cast<std::size_t>(g.ny)), block);
  } else {
    if (g.nx <= 0) return;
    launch_y(grid_1d(static_cast<std::size_t>(g.nx)), block);
  }
}

}  // namespace quasar::backend::pic
