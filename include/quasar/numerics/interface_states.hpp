#pragma once

// Container for reconstructed CONSERVED left/right MHD states on the interfaces
// normal to `dir` (0=x, 1=y). Each of the 8 conserved components has an L and an
// R DeviceBuffer sized grid.storage_size(); the bx/by/bz components are the
// cell-reconstructed magnetic field used to form fluxes / the EMF -- they are NOT
// the staggered constrained-transport face storage. Component order matches
// MhdState: (rho, mx, my, mz, energy, bx, by, bz).

#include "quasar/backend/memory.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/numerics/mhd_state.hpp"

#include <cstddef>
#include <vector>

namespace quasar::numerics {

template <class T>
struct MhdInterfaceStates {
  Grid2D grid{};
  int    dir{0};

  backend::DeviceBuffer<T> Lrho{}, Lmx{}, Lmy{}, Lmz{}, Lenergy{}, Lbx{}, Lby{}, Lbz{};
  backend::DeviceBuffer<T> Rrho{}, Rmx{}, Rmy{}, Rmz{}, Renergy{}, Rbx{}, Rby{}, Rbz{};

  MhdInterfaceStates(Grid2D g, int direction)
    : grid{g}, dir{direction},
      Lrho{g.storage_size()}, Lmx{g.storage_size()}, Lmy{g.storage_size()},
      Lmz{g.storage_size()}, Lenergy{g.storage_size()}, Lbx{g.storage_size()},
      Lby{g.storage_size()}, Lbz{g.storage_size()},
      Rrho{g.storage_size()}, Rmx{g.storage_size()}, Rmy{g.storage_size()},
      Rmz{g.storage_size()}, Renergy{g.storage_size()}, Rbx{g.storage_size()},
      Rby{g.storage_size()}, Rbz{g.storage_size()} {}

  std::size_t component_size() const noexcept { return grid.storage_size(); }

  // Host accessors: read back the single (i,j) element of each component.
  MhdState state_left(int i, int j) const {
    const std::size_t idx = grid.index(i, j);
    MhdState s;
    s.rho    = read_element(Lrho, idx);
    s.mx     = read_element(Lmx, idx);
    s.my     = read_element(Lmy, idx);
    s.mz     = read_element(Lmz, idx);
    s.energy = read_element(Lenergy, idx);
    s.bx     = read_element(Lbx, idx);
    s.by     = read_element(Lby, idx);
    s.bz     = read_element(Lbz, idx);
    return s;
  }

  MhdState state_right(int i, int j) const {
    const std::size_t idx = grid.index(i, j);
    MhdState s;
    s.rho    = read_element(Rrho, idx);
    s.mx     = read_element(Rmx, idx);
    s.my     = read_element(Rmy, idx);
    s.mz     = read_element(Rmz, idx);
    s.energy = read_element(Renergy, idx);
    s.bx     = read_element(Rbx, idx);
    s.by     = read_element(Rby, idx);
    s.bz     = read_element(Rbz, idx);
    return s;
  }

 private:
  // Read element at `idx` by copying the [0, idx+1) prefix back to host and
  // taking the last element. DeviceBuffer::copy_to_host copies a prefix, so a
  // single-element read at an arbitrary index stages that prefix. Correctness
  // over speed -- these accessors exist only for host-side verification.
  static Real read_element(const backend::DeviceBuffer<T>& buf, std::size_t idx) {
    const std::size_t n = idx + 1;
    std::vector<T> staging(n);
    buf.copy_to_host(staging.data(), n);
    return static_cast<Real>(staging[idx]);
  }
};

}  // namespace quasar::numerics
