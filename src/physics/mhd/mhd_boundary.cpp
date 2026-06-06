#include "quasar/boundary/mhd_boundary.hpp"

#include "quasar/core/grid.hpp"
#include "quasar/core/registry.hpp"

#include <array>
#include <cstddef>
#include <vector>

// MHD fluid + magnetic-field boundary conditions for the 2D CT MHD slice.
//
// Implementation note (host staging): the conserved/CT buffers live in
// DeviceBuffer<Real>. The EM-PIC boundaries operate on device buffers through
// HIP kernels in src/backend/hip; here we instead stage each touched component
// to a host std::vector, fill its ghost layers on the host, and copy it back.
// Ghost fills touch only the thin boundary halo (nghost * line cells per
// component), so the staging cost is negligible and this keeps the MHD boundary
// axis free of any .hip translation unit / kernel-launch dependency. Correctness
// and build-safety first; a device-kernel variant can replace this later with no
// interface change.
//
// Indexing conventions (Grid2D::index(i, j)):
//   interior cells:  i in [0, nx),     j in [0, ny)
//   x ghost layers:  g in [1, nghost]  ->  i = -g (lo)  /  i = nx-1+g (hi)
//   y ghost layers:  g in [1, nghost]  ->  j = -g (lo)  /  j = ny-1+g (hi)
//
// Cell-centered components (rho, mx, my, mz, energy, bz_cell) place the
// outermost interior cell at i=0 (lo) / i=nx-1 (hi). Face-staggered components
// (bx_face on x-faces at r_at_edge(i) = origin + i*dx; by_face on y-faces) SHARE
// this same storage extent: there is no extra slot at i=nx+nghost, so every
// face ghost WRITE must stay within i in [-nghost, nx-1+nghost] (j analogous),
// exactly like the cell-centered components. The hi face ghosts therefore live
// at i = nx, nx+1, ..., nx-1+nghost (slots for layer 1..nghost), not nx+nghost.

namespace quasar::boundary {
namespace {

using mhd::MhdField2D;

// A single staged component: pull the device buffer to host, let the caller fill
// ghosts on host_, then push back. Sized to grid.storage_size() (full padded).
struct StagedComponent {
  backend::DeviceBuffer<Real>* dev{nullptr};
  std::vector<Real> host{};

  StagedComponent(backend::DeviceBuffer<Real>& d, std::size_t n) : dev{&d}, host(n) {
    dev->copy_to_host(host.data(), n);
  }
  void write_back() { dev->copy_from_host(host.data(), host.size()); }

  Real& at(const Grid2D& g, int i, int j) { return host[g.index(i, j)]; }
};

// ---------------------------------------------------------------------------
// Generic per-component ghost fillers, parameterized by Side. Each takes a
// staged component and a sign that is applied to the *source* value when writing
// the ghost (+1 = even/copy, -1 = odd/sign-flip). The reflecting wall is the
// only caller that uses sign = -1.
// ---------------------------------------------------------------------------

// Cell-centered component: ghost = sign * (mirror/copy/wrap of interior cell).
void fill_cell(StagedComponent& c, const Grid2D& g, Side side, Real sign,
               const char* mode) {
  const int nx = g.nx;
  const int ny = g.ny;
  const int ng = g.nghost;

  for (int layer = 1; layer <= ng; ++layer) {
    if (side == Side::x_lo || side == Side::x_hi) {
      const int gi = (side == Side::x_lo) ? -layer : (nx - 1 + layer);
      int si;
      if (mode[0] == 'p') {
        si = (side == Side::x_lo) ? (nx - layer) : (layer - 1);  // wrap
      } else if (mode[0] == 'o') {
        si = (side == Side::x_lo) ? 0 : (nx - 1);                // zero-gradient
      } else {
        si = (side == Side::x_lo) ? (layer - 1) : (nx - layer);  // mirror
      }
      for (int j = 0; j < ny; ++j) {
        c.at(g, gi, j) = sign * c.at(g, si, j);
      }
    } else {
      const int gj = (side == Side::y_lo) ? -layer : (ny - 1 + layer);
      int sj;
      if (mode[0] == 'p') {
        sj = (side == Side::y_lo) ? (ny - layer) : (layer - 1);
      } else if (mode[0] == 'o') {
        sj = (side == Side::y_lo) ? 0 : (ny - 1);
      } else {
        sj = (side == Side::y_lo) ? (layer - 1) : (ny - layer);
      }
      for (int i = 0; i < nx; ++i) {
        c.at(g, i, gj) = sign * c.at(g, i, sj);
      }
    }
  }
}

// Face-staggered component normal to `side` (bx_face on an x-side, by_face on a
// y-side). CRITICAL: the face arrays share the exact same storage layout/extent
// as the cell arrays -- index(i,j) with valid i in [-nghost, nx-1+nghost] (j
// analogous). There is NO extra storage slot for a face index at nx+nghost, so
// the hi ghost target MUST be gi = nx-1+layer (slots nx, nx+1, ..., nx-1+nghost
// for layer 1..nghost), identical to the cell-centered hi convention. Writing
// gi = nx+layer overflowed by one column at layer==nghost and corrupted the host
// heap. Because storage is shared with cells, the ghost WRITE indices match
// fill_cell exactly; the normal face differs from a cell only in the symmetry
// SIGN (odd at a reflecting wall), not in the index map.
//
//   periodic   : period nx/ny -- hi ghost face nx-1+layer wraps to face layer-1;
//                lo ghost face -layer wraps to face nx-layer.
//   outflow    : zero-gradient -- copy the outermost interior face (index 0 at
//                lo, nx-1 at hi) into every ghost layer.
//   reflecting : mirror about the wall with the requested (odd) sign, mirror
//                index identical to the cell mirror since storage is shared.
void fill_normal_face(StagedComponent& c, const Grid2D& g, Side side, Real sign,
                      const char* mode) {
  const int nx = g.nx;
  const int ny = g.ny;
  const int ng = g.nghost;

  for (int layer = 1; layer <= ng; ++layer) {
    if (side == Side::x_lo || side == Side::x_hi) {
      const int gi = (side == Side::x_lo) ? -layer : (nx - 1 + layer);
      int si;
      if (mode[0] == 'p') {
        si = (side == Side::x_lo) ? (nx - layer) : (layer - 1);  // face wrap
      } else if (mode[0] == 'o') {
        si = (side == Side::x_lo) ? 0 : (nx - 1);                // outermost face
      } else {
        si = (side == Side::x_lo) ? (layer - 1) : (nx - layer);  // wall mirror
      }
      for (int j = 0; j < ny; ++j) {
        c.at(g, gi, j) = sign * c.at(g, si, j);
      }
    } else {
      const int gj = (side == Side::y_lo) ? -layer : (ny - 1 + layer);
      int sj;
      if (mode[0] == 'p') {
        sj = (side == Side::y_lo) ? (ny - layer) : (layer - 1);
      } else if (mode[0] == 'o') {
        sj = (side == Side::y_lo) ? 0 : (ny - 1);
      } else {
        sj = (side == Side::y_lo) ? (layer - 1) : (ny - layer);
      }
      for (int i = 0; i < nx; ++i) {
        c.at(g, i, gj) = sign * c.at(g, i, sj);
      }
    }
  }
}

bool is_x_side(Side s) { return s == Side::x_lo || s == Side::x_hi; }

// ---------------------------------------------------------------------------
// Fluid boundaries
// ---------------------------------------------------------------------------

// Helper applying one `mode` to every fluid component. `flip_normal_momentum`
// selects the reflecting wall (normal momentum gets an odd / sign-flip mirror;
// all other components are even). The "mode" string's first char selects the
// rule: 'p'eriodic, 'o'utflow, 'r'eflecting.
void apply_fluid(MhdField2D<Real>& f, Side side, const char* mode,
                 bool flip_normal_momentum) {
  const Grid2D& g = f.grid;
  const std::size_t n = g.storage_size();

  StagedComponent rho{f.rho, n};
  StagedComponent mx{f.mx, n};
  StagedComponent my{f.my, n};
  StagedComponent mz{f.mz, n};
  StagedComponent energy{f.energy, n};

  const Real even = Real{1};
  const Real odd = Real{-1};
  // Normal momentum component: mx on an x-side, my on a y-side.
  const Real sx = (flip_normal_momentum && is_x_side(side)) ? odd : even;
  const Real sy = (flip_normal_momentum && !is_x_side(side)) ? odd : even;

  fill_cell(rho, g, side, even, mode);
  fill_cell(mx, g, side, sx, mode);
  fill_cell(my, g, side, sy, mode);
  fill_cell(mz, g, side, even, mode);
  fill_cell(energy, g, side, even, mode);

  rho.write_back();
  mx.write_back();
  my.write_back();
  mz.write_back();
  energy.write_back();
}

class PeriodicFluidBC final : public IMhdFluidBoundary {
 public:
  void fill_ghosts(MhdField2D<Real>& f, Side side) const override {
    apply_fluid(f, side, "periodic", /*flip_normal_momentum=*/false);
  }
};

class OutflowFluidBC final : public IMhdFluidBoundary {
 public:
  void fill_ghosts(MhdField2D<Real>& f, Side side) const override {
    apply_fluid(f, side, "outflow", /*flip_normal_momentum=*/false);
  }
};

class ReflectingFluidBC final : public IMhdFluidBoundary {
 public:
  void fill_ghosts(MhdField2D<Real>& f, Side side) const override {
    apply_fluid(f, side, "reflecting", /*flip_normal_momentum=*/true);
  }
};

// ---------------------------------------------------------------------------
// Magnetic-field boundaries
// ---------------------------------------------------------------------------
//
// Reflecting / conducting-wall symmetry for B: the component *normal* to the
// wall obeys odd symmetry (sign flip) and the *tangential* components obey even
// symmetry (copy). On an x-side the normal field is bx_face (a normal face) and
// the tangentials are by_face (a tangential face) and bz_cell (cell-centered);
// on a y-side the normal field is by_face and the tangentials are bx_face and
// bz_cell. Periodic and outflow apply the same rule to all components (sign +1).
void apply_field(MhdField2D<Real>& f, Side side, const char* mode,
                 bool reflecting) {
  const Grid2D& g = f.grid;
  const std::size_t n = g.storage_size();

  StagedComponent bx{f.bx_face, n};
  StagedComponent by{f.by_face, n};
  StagedComponent bz{f.bz_cell, n};

  const Real even = Real{1};
  const Real odd = Real{-1};

  if (is_x_side(side)) {
    // bx is the normal face; by is a tangential face; bz is cell-centered.
    const Real s_normal = reflecting ? odd : even;
    fill_normal_face(bx, g, side, s_normal, mode);  // normal face
    fill_cell(by, g, side, even, mode);             // tangential face -> cell-like
    fill_cell(bz, g, side, even, mode);             // tangential cell
  } else {
    // by is the normal face; bx is a tangential face; bz is cell-centered.
    const Real s_normal = reflecting ? odd : even;
    fill_normal_face(by, g, side, s_normal, mode);  // normal face
    fill_cell(bx, g, side, even, mode);             // tangential face -> cell-like
    fill_cell(bz, g, side, even, mode);             // tangential cell
  }

  bx.write_back();
  by.write_back();
  bz.write_back();
}

class PeriodicFieldBC final : public IMhdFieldBoundary {
 public:
  void fill_ghosts(MhdField2D<Real>& f, Side side) const override {
    apply_field(f, side, "periodic", /*reflecting=*/false);
  }
};

class OutflowFieldBC final : public IMhdFieldBoundary {
 public:
  void fill_ghosts(MhdField2D<Real>& f, Side side) const override {
    apply_field(f, side, "outflow", /*reflecting=*/false);
  }
};

class ReflectingFieldBC final : public IMhdFieldBoundary {
 public:
  void fill_ghosts(MhdField2D<Real>& f, Side side) const override {
    apply_field(f, side, "reflecting", /*reflecting=*/true);
  }
};

}  // namespace

QUASAR_REGISTER_MHD_FLUID_BOUNDARY("periodic", PeriodicFluidBC)
QUASAR_REGISTER_MHD_FLUID_BOUNDARY("outflow", OutflowFluidBC)
QUASAR_REGISTER_MHD_FLUID_BOUNDARY("reflecting", ReflectingFluidBC)

QUASAR_REGISTER_MHD_FIELD_BOUNDARY("periodic", PeriodicFieldBC)
QUASAR_REGISTER_MHD_FIELD_BOUNDARY("outflow", OutflowFieldBC)
QUASAR_REGISTER_MHD_FIELD_BOUNDARY("reflecting", ReflectingFieldBC)

bool mhd_boundary_is_periodic(const std::string& name) { return name == "periodic"; }

}  // namespace quasar::boundary
