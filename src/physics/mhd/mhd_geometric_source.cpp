#include "quasar/physics/mhd/mhd_geometric_source.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/physics/mhd/kernels.hpp"

// Host wrapper for the axisymmetric cylindrical (r,z) geometric source.
//
// This is intentionally thin: the single authoritative implementation of the
// geometric-source math lives in the device kernel in
// src/backend/hip/mhd/mhd_cylindrical_momentum.hip, declared via
// launch_mhd_geometric_source in kernels.hpp. add() forwards to it on the
// default device stream (nullptr), mirroring how the sibling MHD host paths
// pass nullptr to every launch_mhd_* wrapper. Keeping a single device
// implementation avoids host/device formula drift; the exact formulas and the
// on-axis (r -> 0) guard are documented on MhdGeometricSource in the header and
// realized in the kernel.

namespace quasar::mhd {

void MhdGeometricSource::add(const MhdField2D<Real>& u,
                             MhdField2D<Real>& dudt,
                             const MhdBackgroundField<Real>& b0,
                             const Grid2D& grid, Real gamma,
                             int collocation_order,
                             numerics::RadialTablesView radial_tables) {
  // Accumulate S(u) into dudt on the default stream. The backend selects MP
  // tensor integration when a compatible radial-table view is active, and
  // otherwise retains the cell-centred rule. Like the sibling launch wrappers,
  // this call remains asynchronous; a standalone caller synchronizes before
  // reading the accumulated result.
  launch_mhd_geometric_source(u, dudt, b0, grid, gamma, /*stream=*/nullptr,
                              collocation_order, radial_tables);
}

}  // namespace quasar::mhd
