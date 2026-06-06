#include "quasar/numerics/flux_reconstruction.hpp"

#include "quasar/core/registry.hpp"
#include "quasar/numerics/characteristic_projection.hpp"
#include "quasar/numerics/mhd_eigensystem.hpp"
#include "quasar/numerics/mhd_state.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

// -----------------------------------------------------------------------------
// Flux reconstruction schemes (MUSCL-minmod, MP5, MP7).
//
// Conservative finite-difference reconstruction in the Shu-Osher sense: the
// stored cell values are treated as point values of the reconstructed variable,
// and each scheme produces the LEFT-biased and RIGHT-biased interface point
// values that the (sibling) HLLD Riemann solver consumes.
//
// References:
//   * MUSCL slope limiting (minmod): van Leer (1979), J. Comput. Phys. 32, 101;
//     standard 2nd-order TVD reconstruction.
//   * MP5: Suresh & Huynh (1997), "Accurate Monotonicity-Preserving Schemes with
//     Runge-Kutta Time Stepping", J. Comput. Phys. 136, 83-99 -- specifically the
//     5th-order interpolation (their Eq. 2.1) plus the monotonicity-preserving
//     bound (their Section 2.2, Eqs. 2.6-2.12). Constant alpha = 4 (their
//     recommendation) and the curvature-limited median construction.
//   * MP7: the 7th-order base interpolation (the 7-point central interpolant)
//     wrapped in the identical MP5 limiting machinery; this is the routine
//     "MP7" used e.g. in high-order WENO/MP astrophysical MHD codes.
//
// Reference state for the characteristic eigensystem (MP5/MP7): the simple
// ARITHMETIC MEAN of the two cells adjacent to the interface. (Roe averaging is
// a possible refinement; the arithmetic mean is used here for simplicity and is
// documented as such.)
//
// Normal-B handling: the magnetic component along the reconstruction normal
// (Bx for dir=0, By for dir=1) is the constrained-transport face quantity and is
// continuous across the interface. It is taken directly from the field's
// staggered face storage (bx_face / by_face) and written verbatim into BOTH the
// L and R interface states; it is never passed through the 7-wave characteristic
// projection (which excludes the normal-B degree of freedom). The transverse and
// out-of-plane B components are reconstructed like every other variable.
//
// Dimensionality: reconstruction is performed DIMENSION-BY-DIRECTION -- each
// reconstruct_faces call reconstructs only along the requested normal `dir`.
// Genuinely-multidimensional corner coupling is NOT performed here; it is
// delegated to the CT/EMF stage (sibling) that combines the per-direction
// interface states. This is a deliberate, documented v1 limitation: we do not
// claim corner-coupling that is not implemented.
// -----------------------------------------------------------------------------

namespace quasar::numerics {

namespace {

// --- scalar limiter helpers --------------------------------------------------

QUASAR_HOST_DEVICE inline Real minmod2(Real a, Real b) {
  if (a * b <= Real{0}) return Real{0};
  return (std::abs(a) < std::abs(b)) ? a : b;
}

// 4-argument minmod (Suresh-Huynh Eq. 2.9): zero unless all four share a sign,
// otherwise the smallest magnitude.
QUASAR_HOST_DEVICE inline Real minmod4(Real a, Real b, Real c, Real d) {
  const Real sa = (a > Real{0}) ? Real{1} : ((a < Real{0}) ? Real{-1} : Real{0});
  const Real sb = (b > Real{0}) ? Real{1} : ((b < Real{0}) ? Real{-1} : Real{0});
  const Real sc = (c > Real{0}) ? Real{1} : ((c < Real{0}) ? Real{-1} : Real{0});
  const Real sd = (d > Real{0}) ? Real{1} : ((d < Real{0}) ? Real{-1} : Real{0});
  if (!(sa == sb && sb == sc && sc == sd) || sa == Real{0}) {
    return Real{0};
  }
  const Real mag = std::min(std::min(std::abs(a), std::abs(b)),
                            std::min(std::abs(c), std::abs(d)));
  return sa * mag;
}

QUASAR_HOST_DEVICE inline Real median3(Real a, Real b, Real c) {
  // median(a,b,c) = a + minmod(b-a, c-a) (Suresh-Huynh Eq. 2.7).
  return a + minmod2(b - a, c - a);
}

// MP monotonicity-preserving limiter applied to an unlimited candidate `vl`
// (the high-order interpolation of the LEFT-biased interface value at the right
// face of the central cell v[0]). Stencil: v[-2..+2] for MP5 / v[-3..+3] for MP7,
// but the MP bound only needs v[-1], v[0], v[+1] plus the curvature from the
// neighbours. We pass the five-point window {vm2,vm1,v0,vp1,vp2} since both MP5
// and MP7 use the same MP machinery on the central five points.
// Suresh-Huynh 1997, Section 2.2.
QUASAR_HOST_DEVICE inline Real mp_limit(Real vl, Real vm2, Real vm1, Real v0,
                                        Real vp1, Real vp2) {
  constexpr Real alpha = Real{4};
  constexpr Real eps   = static_cast<Real>(1e-10);

  // mp = v0 + minmod(d+, alpha*d-), the monotonicity-preserving guess (Eq. 2.6).
  const Real d_minus = v0 - vm1;
  const Real d_plus  = vp1 - v0;
  const Real vmp = v0 + minmod2(d_plus, alpha * d_minus);

  // Cheap early accept: if the candidate already lies in [v0, vmp] (Eq. 2.30
  // condition), it is monotonicity preserving and no further work is needed.
  if ((vl - v0) * (vl - vmp) <= eps) {
    return vl;
  }

  // Second differences (curvature), Eq. 2.19.
  const Real dm = vm2 - Real{2} * vm1 + v0;   // d_{j-1}
  const Real d0 = vm1 - Real{2} * v0 + vp1;   // d_j
  const Real dp = v0 - Real{2} * vp1 + vp2;   // d_{j+1}

  // Curvature-limited differences (Eq. 2.27): 4-arg minmod blends.
  const Real d_m4_jph = minmod4(Real{4} * d0 - dp, Real{4} * dp - d0, d0, dp);
  const Real d_m4_jmh = minmod4(Real{4} * d0 - dm, Real{4} * dm - d0, d0, dm);

  // Candidate bounds (Eqs. 2.24-2.26).
  const Real v_ul = v0 + alpha * d_minus;                          // upper-limit
  const Real v_av = Real{0.5} * (v0 + vp1);                        // average
  const Real v_md = v_av - Real{0.5} * d_m4_jph;                   // median
  const Real v_lc = v0 + Real{0.5} * d_minus + (Real{4} / Real{3}) * d_m4_jmh;  // large-curv.

  // Allowed interval [vmin, vmax] (Eqs. 2.24a/b).
  Real vmin = std::max(std::min(std::min(v0, vp1), v_md),
                       std::min(std::min(v0, v_ul), v_lc));
  Real vmax = std::min(std::max(std::max(v0, vp1), v_md),
                       std::max(std::max(v0, v_ul), v_lc));

  // Clamp the candidate into the interval via the median construction (Eq. 2.23).
  return median3(vl, vmin, vmax);
}

// 5th-order LEFT-biased interface interpolation at the right face of cell v0,
// Suresh-Huynh Eq. 2.1: stencil {vm2,vm1,v0,vp1,vp2}.
QUASAR_HOST_DEVICE inline Real mp5_interp(Real vm2, Real vm1, Real v0,
                                          Real vp1, Real vp2) {
  return (Real{2} * vm2 - Real{13} * vm1 + Real{47} * v0
          + Real{27} * vp1 - Real{3} * vp2) / Real{60};
}

// 7th-order LEFT-biased interface interpolation at the right face of cell v0:
// the upwind-biased 7-point interpolant on {vm3,vm2,vm1,v0,vp1,vp2,vp3}.
// Coefficients (sum = 420): (-3, 25, -101, 319, 214, -38, 4)/420.
QUASAR_HOST_DEVICE inline Real mp7_interp(Real vm3, Real vm2, Real vm1, Real v0,
                                          Real vp1, Real vp2, Real vp3) {
  return (-Real{3} * vm3 + Real{25} * vm2 - Real{101} * vm1 + Real{319} * v0
          + Real{214} * vp1 - Real{38} * vp2 + Real{4} * vp3) / Real{420};
}

// MP5 reconstruction: left-biased interface value at the right face of v[0] from
// the 5-point stencil, with the MP monotonicity-preserving limiter applied.
QUASAR_HOST_DEVICE inline Real mp5_reconstruct(Real vm2, Real vm1, Real v0,
                                               Real vp1, Real vp2) {
  const Real vl = mp5_interp(vm2, vm1, v0, vp1, vp2);
  return mp_limit(vl, vm2, vm1, v0, vp1, vp2);
}

// MP7 reconstruction: 7th-order base interpolation + the same MP limiter applied
// on the central five points.
QUASAR_HOST_DEVICE inline Real mp7_reconstruct(Real vm3, Real vm2, Real vm1,
                                               Real v0, Real vp1, Real vp2,
                                               Real vp3) {
  const Real vl = mp7_interp(vm3, vm2, vm1, v0, vp1, vp2, vp3);
  return mp_limit(vl, vm2, vm1, v0, vp1, vp2);
}

// --- host-side field staging -------------------------------------------------

// Read all eight conserved components of a field into contiguous host arrays.
// The CT face/cell distinction is handled by callers: bx_face/by_face hold the
// staggered normal-B storage, bz_cell the toroidal field, the rest cell values.
struct HostField {
  std::size_t n{};
  std::vector<Real> rho, mx, my, mz, energy, bx_face, by_face, bz_cell;

  explicit HostField(const quasar::mhd::MhdField2D<Real>& u)
      : n{u.grid.storage_size()},
        rho(n), mx(n), my(n), mz(n), energy(n),
        bx_face(n), by_face(n), bz_cell(n) {
    u.rho.copy_to_host(rho.data(), n);
    u.mx.copy_to_host(mx.data(), n);
    u.my.copy_to_host(my.data(), n);
    u.mz.copy_to_host(mz.data(), n);
    u.energy.copy_to_host(energy.data(), n);
    u.bx_face.copy_to_host(bx_face.data(), n);
    u.by_face.copy_to_host(by_face.data(), n);
    u.bz_cell.copy_to_host(bz_cell.data(), n);
  }

  // Conserved cell state at (i,j). The in-plane magnetic components are taken
  // from the staggered face storage (best available cell-collocated proxy for a
  // host reconstruction); the reconstruction then refines them per interface.
  MhdState cell(const Grid2D& g, int i, int j) const {
    const std::size_t k = g.index(i, j);
    MhdState s;
    s.rho    = rho[k];
    s.mx     = mx[k];
    s.my     = my[k];
    s.mz     = mz[k];
    s.energy = energy[k];
    s.bx     = bx_face[k];
    s.by     = by_face[k];
    s.bz     = bz_cell[k];
    return s;
  }
};

// L/R staging arrays for the eight conserved components, uploaded once at the end.
struct HostInterface {
  std::size_t n{};
  std::vector<Real> Lrho, Lmx, Lmy, Lmz, Lenergy, Lbx, Lby, Lbz;
  std::vector<Real> Rrho, Rmx, Rmy, Rmz, Renergy, Rbx, Rby, Rbz;

  explicit HostInterface(std::size_t size)
      : n{size},
        Lrho(n), Lmx(n), Lmy(n), Lmz(n), Lenergy(n), Lbx(n), Lby(n), Lbz(n),
        Rrho(n), Rmx(n), Rmy(n), Rmz(n), Renergy(n), Rbx(n), Rby(n), Rbz(n) {}

  void set_left(std::size_t k, const MhdState& s) {
    Lrho[k] = s.rho; Lmx[k] = s.mx; Lmy[k] = s.my; Lmz[k] = s.mz;
    Lenergy[k] = s.energy; Lbx[k] = s.bx; Lby[k] = s.by; Lbz[k] = s.bz;
  }
  void set_right(std::size_t k, const MhdState& s) {
    Rrho[k] = s.rho; Rmx[k] = s.mx; Rmy[k] = s.my; Rmz[k] = s.mz;
    Renergy[k] = s.energy; Rbx[k] = s.bx; Rby[k] = s.by; Rbz[k] = s.bz;
  }

  void upload(MhdInterfaceStates<Real>& out) const {
    out.Lrho.copy_from_host(Lrho.data(), n);
    out.Lmx.copy_from_host(Lmx.data(), n);
    out.Lmy.copy_from_host(Lmy.data(), n);
    out.Lmz.copy_from_host(Lmz.data(), n);
    out.Lenergy.copy_from_host(Lenergy.data(), n);
    out.Lbx.copy_from_host(Lbx.data(), n);
    out.Lby.copy_from_host(Lby.data(), n);
    out.Lbz.copy_from_host(Lbz.data(), n);
    out.Rrho.copy_from_host(Rrho.data(), n);
    out.Rmx.copy_from_host(Rmx.data(), n);
    out.Rmy.copy_from_host(Rmy.data(), n);
    out.Rmz.copy_from_host(Rmz.data(), n);
    out.Renergy.copy_from_host(Renergy.data(), n);
    out.Rbx.copy_from_host(Rbx.data(), n);
    out.Rby.copy_from_host(Rby.data(), n);
    out.Rbz.copy_from_host(Rbz.data(), n);
  }
};

// Offset the (i,j) cell index by `s` cells along the normal `dir`.
inline MhdState shift_cell(const HostField& f, const Grid2D& g, int dir,
                           int i, int j, int s) {
  return (dir == 0) ? f.cell(g, i + s, j) : f.cell(g, i, j + s);
}

// Component setters/getters for the conserved normal-B (the CT face quantity).
inline Real normal_b(const MhdState& s, int dir) { return (dir == 0) ? s.bx : s.by; }
inline void set_normal_b(MhdState& s, int dir, Real v) {
  if (dir == 0) s.bx = v; else s.by = v;
}

// Linear interpolation of the staggered normal-B face value at the interface.
// For dir=0, the interface (i,j) coincides with the cell-(i-1)/cell-(i) face;
// under CT bx_face[i] already lives on that face, so the continuous interface
// normal-B is just that face value. We average the two adjacent face samples for
// a smooth high-order-consistent value; under exact CT they are equal.
inline Real interface_normal_b(const HostField& f, const Grid2D& g, int dir,
                               int i, int j) {
  const std::size_t k0 = g.index(i, j);
  const std::size_t km = (dir == 0) ? g.index(i - 1, j) : g.index(i, j - 1);
  const auto& face = (dir == 0) ? f.bx_face : f.by_face;
  return Real{0.5} * (face[k0] + face[km]);
}

}  // namespace

// =============================================================================
// MUSCL-minmod (2nd-order, primitive-variable, slope-limited)
// =============================================================================
class MusclMinmodRecon : public IFluxReconstruction {
 public:
  int  required_nghost() const override { return 2; }
  bool is_characteristic() const override { return false; }

  void reconstruct_faces(const quasar::mhd::MhdField2D<Real>& u, int dir,
                         MhdInterfaceStates<Real>& out, Real gamma) const override {
    const Grid2D g = u.grid;
    HostField f{u};
    HostInterface h{g.storage_size()};

    const int ng = g.nghost;
    // Iterate every interface normal to dir: faces are indexed by the "right"
    // cell. For dir=0 face (i,j) is between cells (i-1) and (i).
    const int i_lo = (dir == 0) ? -ng + 1 : -ng;
    const int i_hi = (dir == 0) ? g.nx + ng : g.nx + ng - 1;
    const int j_lo = (dir == 0) ? -ng : -ng + 1;
    const int j_hi = (dir == 0) ? g.ny + ng - 1 : g.ny + ng;

    for (int j = j_lo; j < j_hi; ++j) {
      for (int i = i_lo; i < i_hi; ++i) {
        // Cells adjacent to / straddling the face for the minmod slopes.
        const MhdState cL = shift_cell(f, g, dir, i, j, -1);  // left cell
        const MhdState cR = shift_cell(f, g, dir, i, j, 0);   // right cell
        const MhdState cLL = shift_cell(f, g, dir, i, j, -2);
        const MhdState cRR = shift_cell(f, g, dir, i, j, 1);

        // Reconstruct in PRIMITIVE variables for robustness.
        const MhdPrim wLL = to_primitive(cLL, gamma);
        const MhdPrim wL  = to_primitive(cL, gamma);
        const MhdPrim wR  = to_primitive(cR, gamma);
        const MhdPrim wRR = to_primitive(cRR, gamma);

        MhdPrim pL = muscl_face(wLL, wL, wR, +1);  // right face of left cell
        MhdPrim pR = muscl_face(wRR, wR, wL, -1);  // left  face of right cell

        MhdState sL = to_conserved(pL, gamma);
        MhdState sR = to_conserved(pR, gamma);

        // Normal-B: continuous CT face value on both sides.
        const Real bn = interface_normal_b(f, g, dir, i, j);
        set_normal_b(sL, dir, bn);
        set_normal_b(sR, dir, bn);

        const std::size_t k = g.index(i, j);
        h.set_left(k, sL);
        h.set_right(k, sR);
      }
    }
    h.upload(out);
  }

 private:
  // minmod-limited extrapolation to the face shared by `c` and its neighbour in
  // direction `sgn` (+1 = right face, -1 = left face). `far` is the cell on the
  // opposite side of `c` from that face. Slope = minmod(c - far, near - c).
  static MhdPrim muscl_face(const MhdPrim& far, const MhdPrim& c,
                            const MhdPrim& near, int sgn) {
    MhdPrim out;
    out.rho = limited(far.rho, c.rho, near.rho, sgn);
    out.vx  = limited(far.vx,  c.vx,  near.vx,  sgn);
    out.vy  = limited(far.vy,  c.vy,  near.vy,  sgn);
    out.vz  = limited(far.vz,  c.vz,  near.vz,  sgn);
    out.p   = limited(far.p,   c.p,   near.p,   sgn);
    out.bx  = limited(far.bx,  c.bx,  near.bx,  sgn);
    out.by  = limited(far.by,  c.by,  near.by,  sgn);
    out.bz  = limited(far.bz,  c.bz,  near.bz,  sgn);
    return out;
  }

  static Real limited(Real far, Real c, Real near, int sgn) {
    const Real slope = minmod2(c - far, near - c);
    return c + static_cast<Real>(sgn) * Real{0.5} * slope;
  }
};

// =============================================================================
// Characteristic MP reconstruction base (MP5 / MP7 share the projection driver)
// =============================================================================
class CharacteristicMpRecon : public IFluxReconstruction {
 public:
  bool is_characteristic() const override { return true; }

 protected:
  // Stencil half-width (MP5 reads 2 each side, MP7 reads 3).
  virtual int half_width() const = 0;

  // Reconstruct the LEFT-biased interface value at the right face of the central
  // cell for one scalar characteristic field, given the symmetric stencil
  // `w[0..2*half]` centered on the cell (w[half] is the central cell). The
  // returned value is the interface value extrapolated toward the right face.
  virtual Real recon_right_face(const Real* w, int half) const = 0;

  void reconstruct_faces(const quasar::mhd::MhdField2D<Real>& u, int dir,
                         MhdInterfaceStates<Real>& out, Real gamma) const override {
    const Grid2D g = u.grid;
    HostField f{u};
    HostInterface h{g.storage_size()};

    const int hw = half_width();
    const int ng = g.nghost;
    // We need hw cells on the far side of each adjacent cell, so restrict the
    // face range so every stencil index stays within [-ng, n+ng).
    const int margin = hw;
    const int i_lo = (dir == 0) ? -ng + margin : -ng;
    const int i_hi = (dir == 0) ? g.nx + ng - margin : g.nx + ng - 1;
    const int j_lo = (dir == 0) ? -ng : -ng + margin;
    const int j_hi = (dir == 0) ? g.ny + ng - 1 : g.ny + ng - margin;

    const int width = 2 * hw + 1;
    std::vector<MhdState> stencil(static_cast<std::size_t>(width) + 1);

    for (int j = j_lo; j < j_hi; ++j) {
      for (int i = i_lo; i < i_hi; ++i) {
        // Reference state = arithmetic mean of the two cells flanking the face
        // (cell i-1 and cell i for dir=0). Build the eigensystem about it.
        const MhdState cL = shift_cell(f, g, dir, i, j, -1);
        const MhdState cR = shift_cell(f, g, dir, i, j, 0);
        MhdState ref;
        ref.rho    = Real{0.5} * (cL.rho + cR.rho);
        ref.mx     = Real{0.5} * (cL.mx + cR.mx);
        ref.my     = Real{0.5} * (cL.my + cR.my);
        ref.mz     = Real{0.5} * (cL.mz + cR.mz);
        ref.energy = Real{0.5} * (cL.energy + cR.energy);
        ref.bx     = Real{0.5} * (cL.bx + cR.bx);
        ref.by     = Real{0.5} * (cL.by + cR.by);
        ref.bz     = Real{0.5} * (cL.bz + cR.bz);

        MhdEigensystem eig;
        eig.build(ref, dir, gamma);

        // Gather the stencil cells. For the L state we use the stencil centered
        // on cell i-1 and reconstruct its RIGHT face. For the R state we use the
        // stencil centered on cell i and reconstruct its LEFT face, obtained by
        // mirroring the stencil and reconstructing the right face.
        //
        // Project each stencil cell (as a delta from `ref`) into characteristic
        // space, reconstruct each of the 7 fields, then map back and add `ref`.
        const MhdState sL = recon_state(f, g, dir, i, j, /*center_off=*/-1,
                                        /*mirror=*/false, eig, ref);
        const MhdState sR = recon_state(f, g, dir, i, j, /*center_off=*/0,
                                        /*mirror=*/true, eig, ref);

        MhdState outL = sL;
        MhdState outR = sR;
        const Real bn = interface_normal_b(f, g, dir, i, j);
        set_normal_b(outL, dir, bn);
        set_normal_b(outR, dir, bn);

        const std::size_t k = g.index(i, j);
        h.set_left(k, outL);
        h.set_right(k, outR);
      }
    }
    h.upload(out);
  }

 private:
  // Reconstruct one interface state in characteristic variables. The stencil is
  // centered on cell (base + center_off) along `dir`; if `mirror` is true the
  // stencil order is reversed so the "right face" routine produces the left-face
  // value of the centered cell. Returns the reconstructed CONSERVED state
  // (normal-B set to zero by from_char; the caller overwrites it with the CT
  // face value).
  MhdState recon_state(const HostField& f, const Grid2D& g, int dir, int i,
                       int j, int center_off, bool mirror,
                       const MhdEigensystem& eig, const MhdState& ref) const {
    const int hw = half_width();
    const int width = 2 * hw + 1;

    // Project each stencil cell's delta into characteristic space.
    // chars[s][k]: s over stencil (0..width-1), k over 7 waves.
    std::vector<std::array<Real, 7>> chars(static_cast<std::size_t>(width));
    for (int s = 0; s < width; ++s) {
      const int off = center_off + (s - hw);
      const MhdState cell = shift_cell(f, g, dir, i, j, off);
      MhdState delta;
      delta.rho    = cell.rho - ref.rho;
      delta.mx     = cell.mx - ref.mx;
      delta.my     = cell.my - ref.my;
      delta.mz     = cell.mz - ref.mz;
      delta.energy = cell.energy - ref.energy;
      delta.bx     = cell.bx - ref.bx;
      delta.by     = cell.by - ref.by;
      delta.bz     = cell.bz - ref.bz;
      chars[static_cast<std::size_t>(s)] = CharacteristicProjector::to_char(delta, eig);
    }

    // Reconstruct each of the 7 characteristic fields at the interface.
    std::array<Real, 7> w_face{};
    Real scalar[16];  // scratch >= width
    for (int k = 0; k < 7; ++k) {
      for (int s = 0; s < width; ++s) {
        const int src = mirror ? (width - 1 - s) : s;
        scalar[s] = chars[static_cast<std::size_t>(src)][static_cast<std::size_t>(k)];
      }
      w_face[static_cast<std::size_t>(k)] = recon_right_face(scalar, hw);
    }

    // Map the reconstructed characteristic delta back to a conserved delta and
    // add the reference. from_char zeroes the normal-B component (carried
    // separately by the caller).
    MhdState d = CharacteristicProjector::from_char(w_face, eig);
    MhdState s;
    s.rho    = ref.rho + d.rho;
    s.mx     = ref.mx + d.mx;
    s.my     = ref.my + d.my;
    s.mz     = ref.mz + d.mz;
    s.energy = ref.energy + d.energy;
    s.bx     = ref.bx + d.bx;
    s.by     = ref.by + d.by;
    s.bz     = ref.bz + d.bz;
    return s;
  }
};

// =============================================================================
// MP5 (Suresh-Huynh 5th-order monotonicity-preserving)
// =============================================================================
class Mp5Recon : public CharacteristicMpRecon {
 public:
  int required_nghost() const override { return 3; }

 protected:
  int half_width() const override { return 2; }
  Real recon_right_face(const Real* w, int /*half*/) const override {
    return mp5_reconstruct(w[0], w[1], w[2], w[3], w[4]);
  }
};

// =============================================================================
// MP7 (7th-order base interpolation + MP limiting)
// =============================================================================
class Mp7Recon : public CharacteristicMpRecon {
 public:
  int required_nghost() const override { return 4; }

 protected:
  int half_width() const override { return 3; }
  Real recon_right_face(const Real* w, int /*half*/) const override {
    return mp7_reconstruct(w[0], w[1], w[2], w[3], w[4], w[5], w[6]);
  }
};

}  // namespace quasar::numerics

QUASAR_REGISTER_FLUX_RECONSTRUCTION("muscl_minmod",
                                    ::quasar::numerics::MusclMinmodRecon)
QUASAR_REGISTER_FLUX_RECONSTRUCTION("mp5", ::quasar::numerics::Mp5Recon)
QUASAR_REGISTER_FLUX_RECONSTRUCTION("mp7", ::quasar::numerics::Mp7Recon)
