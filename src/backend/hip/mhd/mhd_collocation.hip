// Face-to-cell magnetic collocation as a standalone sweep.
//
// The solver's readback path (MhdSolver2D::state_component_to_host) reports
// "bx"/"by" as the finite-volume cell average of the staggered CT faces, which
// is the same quantity the EOS reads. That quadrature used to run on the host
// after the download -- one call to mhd_staggering.hpp per padded cell. The
// data has to be copied either way, but the arithmetic belongs on the side that
// already owns the field.
//
// Compiled -ffp-contract=off: the collocation weights go through the scaled
// rational reduction in mhd_staggering.hpp, whose compensation contraction
// silently degrades.

#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

namespace {

using quasar::Grid2D;
using quasar::Real;

inline constexpr unsigned kBlock2D = 16;

__global__ void QUASAR_LAUNCH_BOUNDS_256 collocate_kernel(
    Grid2D g, int axis, quasar::numerics::RadialTablesView radial_tables,
    const double* __restrict__ face, double* __restrict__ cell) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - g.nghost;
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - g.nghost;
  if (i >= g.nx + g.nghost || j >= g.ny + g.nghost) return;
  cell[g.index(i, j)] =
      axis == 0 ? quasar::mhd::cell_bx(g, face, i, j, radial_tables)
                : quasar::mhd::cell_by(g, face, i, j, radial_tables);
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_collocate_face_to_cell(
    Grid2D grid, int axis, const backend::DeviceBuffer<Real>& face,
    quasar::numerics::RadialTablesView radial_tables,
    backend::DeviceBuffer<Real>& cell, stream_t stream) {
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 launch_grid(
      (static_cast<unsigned>(grid.pitch()) + block.x - 1) / block.x,
      (static_cast<unsigned>(grid.height()) + block.y - 1) / block.y);
  hipLaunchKernelGGL(collocate_kernel, launch_grid, block, 0,
                     static_cast<hipStream_t>(stream), grid, axis,
                     radial_tables, face.device_ptr(), cell.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
