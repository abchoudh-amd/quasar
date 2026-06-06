// Troubled-cell positivity limiter for ideal MHD, self-registered "troubled_cell".
//
// This translation unit is the HOST reference implementation exercised through the
// registry by the unit tests: a test constructs the limiter by name and calls
// apply() on an MhdField2D directly. Device buffers are staged to host, the floor
// algebra runs on the host (mirroring ct_scheme.cpp), and results are copied back
// with copy_from_host. Correctness over speed.
//
// Floor semantics (consistent with the DEVICE path launch_mhd_apply_floors used
// inside the solver's per-stage combine, see physics/mhd/kernels.hpp): for every
// cell,
//   1. density floor: if rho < rho_floor, set rho = rho_floor;
//   2. pressure floor: with momentum m and magnetic field B held FIXED, if the
//      gas pressure p = (gamma-1)*(E - 0.5*|m|^2/rho - 0.5*|B|^2) < p_floor, raise
//      total energy to E = p_floor/(gamma-1) + 0.5*|m|^2/rho + 0.5*|B|^2 so the
//      cell's pressure becomes exactly p_floor.
// A cell already above both floors is untouched to round-off (the re-derived E is
// algebraically identical to the input E when p >= p_floor and rho >= rho_floor).
// This is the "troubled cell" correction: only flagged (non-physical) cells are
// modified; positive cells pass through unchanged.
//
// The pressure() / to_conserved() energy convention here matches
// numerics/mhd_state.hpp exactly (kinetic = 0.5*|m|^2/rho, magnetic = 0.5*|B|^2),
// so the host floor and the device floor agree term-by-term.

#include "quasar/numerics/positivity_limiter.hpp"

#include "quasar/core/registry.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

#include <cstddef>
#include <vector>

namespace quasar::numerics {

namespace {

// Stage a whole DeviceBuffer to a host vector sized to the full padded storage.
inline std::vector<Real> stage(const backend::DeviceBuffer<Real>& buf,
                               std::size_t n) {
  std::vector<Real> h(n);
  buf.copy_to_host(h.data(), n);
  return h;
}

}  // namespace

// Floor-based troubled-cell limiter. Registered "troubled_cell".
class TroubledCellLimiter : public IPositivityLimiter {
 public:
  void apply(quasar::mhd::MhdField2D<Real>& u, Real rho_floor, Real p_floor,
             Real gamma) const override {
    const Grid2D& g = u.grid;
    const std::size_t storage = g.storage_size();

    // Stage the conserved field to host. Bx/By live on faces and Bz at the cell
    // center; the floor uses them only to remove magnetic energy, so reading them
    // at the same index as the cell-centered conserved variables is consistent
    // with the device floor's per-cell treatment.
    std::vector<Real> rho = stage(u.rho, storage);
    std::vector<Real> mx  = stage(u.mx, storage);
    std::vector<Real> my  = stage(u.my, storage);
    std::vector<Real> mz  = stage(u.mz, storage);
    std::vector<Real> en  = stage(u.energy, storage);
    const std::vector<Real> bx = stage(u.bx_face, storage);
    const std::vector<Real> by = stage(u.by_face, storage);
    const std::vector<Real> bz = stage(u.bz_cell, storage);

    const Real inv_gm1 = Real{1} / (gamma - Real{1});

    for (std::size_t c = 0; c < storage; ++c) {
      // 1. Density floor first, so the kinetic term below uses the floored rho.
      if (rho[c] < rho_floor) {
        rho[c] = rho_floor;
      }

      MhdState s;
      s.rho    = rho[c];
      s.mx     = mx[c];
      s.my     = my[c];
      s.mz     = mz[c];
      s.energy = en[c];
      s.bx     = bx[c];
      s.by     = by[c];
      s.bz     = bz[c];

      // 2. Pressure floor: only the flagged (sub-floor) cells are corrected.
      const Real p = pressure(s, gamma);
      if (p < p_floor) {
        const Real kinetic =
            Real{0.5} * (s.mx * s.mx + s.my * s.my + s.mz * s.mz) / s.rho;
        const Real magnetic =
            Real{0.5} * (s.bx * s.bx + s.by * s.by + s.bz * s.bz);
        en[c] = p_floor * inv_gm1 + kinetic + magnetic;
      }
    }

    // Only rho and energy can change; momentum and B are held fixed.
    u.rho.copy_from_host(rho.data(), storage);
    u.energy.copy_from_host(en.data(), storage);
  }
};

}  // namespace quasar::numerics

QUASAR_REGISTER_POSITIVITY_LIMITER("troubled_cell", ::quasar::numerics::TroubledCellLimiter)
