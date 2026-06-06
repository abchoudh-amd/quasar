// FD-CT (Christlieb-style) constrained-transport scheme for ideal MHD.
//
// References:
//   - C. R. Evans & J. F. Hawley, "Simulation of magnetohydrodynamic flows: a
//     constrained transport method", ApJ 332 (1988) 659. (The curl-of-EMF face
//     update that keeps div(B) at round-off.)
//   - D. S. Balsara & D. S. Spicer, "A staggered mesh algorithm using high order
//     Godunov fluxes to ensure solenoidal magnetic fields in MHD simulations",
//     JCP 149 (1999) 270. (The simple arithmetic corner average of the four
//     adjacent face EMFs used here.)
//   - A. J. Christlieb et al., finite-difference CT family -- the corner EMF is
//     reconstructed from the same interface states the conservative flux uses, so
//     the induction update is consistent with the fluid update.
//
// This translation unit is the host reference implementation exercised through
// the registry by the unit tests. Device buffers are staged to host, the CT
// algebra runs on the host, and results are copied back with copy_from_host.
// Correctness (machine-epsilon div B) over speed.
//
// ---------------------------------------------------------------------------
// Staggering (matches mhd_field.hpp; see ct_scheme.hpp):
//   bx_face(i,j) = Bx on the x_lo (left)   face of cell (i,j)
//   by_face(i,j) = By on the y_lo (bottom) face of cell (i,j)
//   ez_edge(i,j) = Ez at the lower-left corner of cell (i,j)
//   divB(i,j) = (bx_face(i+1,j) - bx_face(i,j))/dx
//             + (by_face(i,j+1) - by_face(i,j))/dy
//
// ---------------------------------------------------------------------------
// EMF averaging
// -------------
// A face-centered Ez = vx*By - vy*Bx is formed at every interface from the
// arithmetic average of the reconstructed left/right CONSERVED states (the
// Balsara-Spicer simple average). The corner EMF is then the arithmetic average
// of the four face EMFs whose faces touch that corner:
//
//   ez_edge(i,j) = 1/4 * ( Ez_xface(i,j) + Ez_xface(i,j-1)      [two x-faces]
//                        + Ez_yface(i,j) + Ez_yface(i-1,j) )    [two y-faces]
//
// where Ez_xface(i,j) is the EMF on the x_lo face of cell (i,j) (from ifx) and
// Ez_yface(i,j) is the EMF on the y_lo face of cell (i,j) (from ify). The x_lo
// faces of cells (i,j) and (i,j-1) are the two vertical edges meeting the
// lower-left corner of (i,j); the y_lo faces of cells (i,j) and (i-1,j) are the
// two horizontal edges meeting it.
//
// ---------------------------------------------------------------------------
// Face-B update (dB/dt = -curl E, E = (0,0,Ez)):
//   dBx/dt = -dEz/dy,   dBy/dt = +dEz/dx
//
//   bx_face(i,j) -= dt * (ez_edge(i,j+1) - ez_edge(i,j)) / dy
//   by_face(i,j) += dt * (ez_edge(i+1,j) - ez_edge(i,j)) / dx
//
// div(B) annihilation (telescoping) -- the machine-epsilon guarantee.
// Write the change of each face touched by divB(i,j):
//   d[bx_face(i+1,j)] = -dt*( ez(i+1,j+1) - ez(i+1,j) )/dy
//   d[bx_face(i,  j)] = -dt*( ez(i,  j+1) - ez(i,  j) )/dy
//   d[by_face(i,j+1)] = +dt*( ez(i+1,j+1) - ez(i,  j+1) )/dx
//   d[by_face(i,j  )] = +dt*( ez(i+1,j  ) - ez(i,  j  ) )/dx
// Then
//   d[divB(i,j)] = ( d[bx_face(i+1,j)] - d[bx_face(i,j)] )/dx
//                + ( d[by_face(i,j+1)] - d[by_face(i,j)] )/dy
//   x-part/dx = -dt/(dx dy) * ( ez(i+1,j+1) - ez(i+1,j) - ez(i,j+1) + ez(i,j) )
//   y-part/dy = +dt/(dx dy) * ( ez(i+1,j+1) - ez(i,j+1) - ez(i+1,j) + ez(i,j) )
// The two brackets are the identical mixed second difference of ez over the
// corner stencil {(i,j),(i+1,j),(i,j+1),(i+1,j+1)}; they cancel term-by-term, so
// d[divB(i,j)] = 0 EXACTLY for any ez_edge values, independent of dt and the
// grid. div(B) is therefore preserved to round-off (a field initialized
// divergence-free stays divergence-free).
// ---------------------------------------------------------------------------

#include "quasar/numerics/ct_scheme.hpp"

#include "quasar/core/registry.hpp"
#include "quasar/numerics/mhd_state.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace quasar::numerics {

namespace {

// Face EMF Ez = vx*By - vy*Bx from the arithmetic average of the reconstructed
// left/right CONSERVED interface states (Balsara-Spicer simple average). The
// average is taken on the conserved variables, then decoded to velocities; this
// is symmetric in L/R, so it introduces no preferred direction and keeps the
// corner average even-handed.
inline Real face_emf_z(const MhdState& l, const MhdState& r) {
  MhdState a;
  a.rho    = Real{0.5} * (l.rho + r.rho);
  a.mx     = Real{0.5} * (l.mx + r.mx);
  a.my     = Real{0.5} * (l.my + r.my);
  a.mz     = Real{0.5} * (l.mz + r.mz);
  a.energy = Real{0.5} * (l.energy + r.energy);
  a.bx     = Real{0.5} * (l.bx + r.bx);
  a.by     = Real{0.5} * (l.by + r.by);
  a.bz     = Real{0.5} * (l.bz + r.bz);
  const Real inv_rho = Real{1} / a.rho;
  const Real vx = a.mx * inv_rho;
  const Real vy = a.my * inv_rho;
  return vx * a.by - vy * a.bx;  // Ez = vx*By - vy*Bx
}

// Stage a whole DeviceBuffer to a host vector sized to the full padded storage.
inline std::vector<Real> stage(const backend::DeviceBuffer<Real>& buf,
                               std::size_t n) {
  std::vector<Real> h(n);
  buf.copy_to_host(h.data(), n);
  return h;
}

}  // namespace

// Concrete FD-CT scheme, registered "fd_ct_christlieb".
class ChristliebFdCt : public ICtScheme {
 public:
  void compute_emf(const quasar::mhd::MhdField2D<Real>& u,
                   const MhdInterfaceStates<Real>& ifx,   // dir=0 faces
                   const MhdInterfaceStates<Real>& ify,   // dir=1 faces
                   quasar::mhd::EmfField2D<Real>& emf,
                   Real /*gamma*/) const override {
    const Grid2D& g = u.grid;
    const int ng = g.nghost;
    const std::size_t storage = g.storage_size();

    // Per-face Ez, laid out on the same padded storage as the corner EMF.
    std::vector<Real> ez_xface(storage, Real{0});  // Ez on x_lo face of (i,j)
    std::vector<Real> ez_yface(storage, Real{0});  // Ez on y_lo face of (i,j)

    // Face EMFs are needed one ghost layer out so every interior corner has its
    // four neighbours (corner (i,j) reads faces at (i,j-1) and (i-1,j)).
    for (int j = -ng; j < g.ny + ng; ++j) {
      for (int i = -ng; i < g.nx + ng; ++i) {
        const std::size_t idx = g.index(i, j);
        ez_xface[idx] = face_emf_z(ifx.state_left(i, j), ifx.state_right(i, j));
        ez_yface[idx] = face_emf_z(ify.state_left(i, j), ify.state_right(i, j));
      }
    }

    // Corner average: ez_edge(i,j) = 1/4 (Ez_xface(i,j) + Ez_xface(i,j-1)
    //                                   + Ez_yface(i,j) + Ez_yface(i-1,j)).
    std::vector<Real> ez_edge(storage, Real{0});
    for (int j = -ng + 1; j < g.ny + ng; ++j) {
      for (int i = -ng + 1; i < g.nx + ng; ++i) {
        const std::size_t c = g.index(i, j);
        ez_edge[c] = Real{0.25} *
            (ez_xface[g.index(i, j)] + ez_xface[g.index(i, j - 1)] +
             ez_yface[g.index(i, j)] + ez_yface[g.index(i - 1, j)]);
      }
    }

    emf.ez_edge.copy_from_host(ez_edge.data(), storage);

    // In-plane edge EMFs are not used by the poloidal (Ez) face-B update in this
    // 2.5D scheme; zero them so the EMF field is fully defined. Toroidal Bz is
    // cell-centered and is evolved by the flux-difference path, not here.
    std::vector<Real> zeros(storage, Real{0});
    emf.ex_edge.copy_from_host(zeros.data(), storage);
    emf.ey_edge.copy_from_host(zeros.data(), storage);
  }

  void update_face_b(quasar::mhd::MhdField2D<Real>& u,
                     const quasar::mhd::EmfField2D<Real>& emf,
                     Real dt) const override {
    const Grid2D& g = u.grid;
    const int ng = g.nghost;
    const std::size_t storage = g.storage_size();
    const Real dx = g.dx();
    const Real dy = g.dy();

    std::vector<Real> bx = stage(u.bx_face, storage);
    std::vector<Real> by = stage(u.by_face, storage);
    const std::vector<Real> ez = stage(emf.ez_edge, storage);

    // bx_face(i,j) -= dt*(ez(i,j+1) - ez(i,j))/dy : needs ez at (i,j) and
    // (i,j+1). by_face(i,j) += dt*(ez(i+1,j) - ez(i,j))/dx : needs (i,j),(i+1,j).
    // Update every face whose required corners exist within the padded storage.
    for (int j = -ng; j < g.ny + ng - 1; ++j) {
      for (int i = -ng; i < g.nx + ng - 1; ++i) {
        const std::size_t f = g.index(i, j);
        bx[f] -= dt * (ez[g.index(i, j + 1)] - ez[g.index(i, j)]) / dy;
        by[f] += dt * (ez[g.index(i + 1, j)] - ez[g.index(i, j)]) / dx;
      }
    }

    u.bx_face.copy_from_host(bx.data(), storage);
    u.by_face.copy_from_host(by.data(), storage);
  }

  Real divergence_b_linf(const quasar::mhd::MhdField2D<Real>& u) const override {
    const Grid2D& g = u.grid;
    const std::size_t storage = g.storage_size();
    const Real dx = g.dx();
    const Real dy = g.dy();

    const std::vector<Real> bx = stage(u.bx_face, storage);
    const std::vector<Real> by = stage(u.by_face, storage);

    // divB(i,j) = (bx_face(i+1,j) - bx_face(i,j))/dx
    //           + (by_face(i,j+1) - by_face(i,j))/dy  over interior cells.
    Real linf = Real{0};
    for (int j = 0; j < g.ny; ++j) {
      for (int i = 0; i < g.nx; ++i) {
        const Real div =
            (bx[g.index(i + 1, j)] - bx[g.index(i, j)]) / dx +
            (by[g.index(i, j + 1)] - by[g.index(i, j)]) / dy;
        linf = std::max(linf, std::abs(div));
      }
    }
    return linf;
  }
};

}  // namespace quasar::numerics

QUASAR_REGISTER_CT_SCHEME("fd_ct_christlieb", ::quasar::numerics::ChristliebFdCt)
