#pragma once

// Project a Grad-Shafranov equilibrium onto the staggered MHD mesh.
//
// This is the bridge between the two grid conventions, and it is where the
// equilibrium either survives contact with the MHD solver or does not.
//
// -- The two conventions -------------------------------------------------------
// The GS solve runs on a NODE-centered EllipticGrid: psi lives on nodes and the
// domain boundary is a row of degrees of freedom (the right shape for a
// Dirichlet elliptic problem). MhdField2D is CELL-centered with a ghost halo:
// bx_face(i,j) is the x-low face average of cell (i,j), by_face(i,j) the y-low
// face average, bz_cell(i,j) a cell average.
//
// -- Why differencing psi, not interpolating B ---------------------------------
// The obvious approach -- evaluate B on the GS grid, then interpolate it onto
// the MHD faces -- does NOT preserve div(B) = 0. Interpolation of two
// independently-sampled components has no reason to satisfy a discrete
// divergence identity, and constrained transport will then propagate that
// initial divergence error forever.
//
// Instead psi is interpolated to CELL CORNERS and the face fields are obtained
// by DIFFERENCING it, exactly as examples/square_toroid_mhd does with the
// Biot-Savart vector potential:
//
//   B_r(i,j) = -(psi[j+1][i] - psi[j][i]) / dz              (x-low face)
//   B_z(i,j) =  (r_hi psi[j][i+1] - r_lo psi[j][i]) / int(r dr)   (y-low face)
//
// Because both components come from ONE scalar potential by a discrete curl,
// the discrete divergence telescopes to zero identically -- to round-off, not
// to truncation. This is the whole reason the MHD background contract is
// expressed in terms of a vector potential.
//
// psi here IS the poloidal-flux form of the toroidal vector potential:
// psi = r * A_phi. The annular r-weighting in the B_z formula is the cylindrical
// curl, matching the convention already used by the MHD cylindrical path.
//
// -- Coordinate mapping --------------------------------------------------------
//   MHD x  <-> GS r   (major radius)
//   MHD y  <-> GS z   (vertical)
//   MHD z  <-> phi    (toroidal, out of plane) -> bz_cell carries B_phi
//
// which is the same mapping examples/square_toroid_mhd documents.

#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/critical_points.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quasar::equilibrium {

using numerics::EllipticGrid;
using numerics::ScalarField;

// Staggered background field on an MHD Grid2D, in the layout MhdSolver2D's
// seed_background() expects. Each buffer is sized grid.storage_size().
struct StaggeredBackground {
  Grid2D grid{};
  std::vector<Real> b0x_face{};  // B_r on x-low faces
  std::vector<Real> b0y_face{};  // B_z on y-low faces
  std::vector<Real> b0z_cell{};  // B_phi, cell-centered
};

// Bilinear sample of psi at an arbitrary (r,z), clamped to the GS domain.
//
// Clamping rather than extrapolating is deliberate: an MHD grid slightly larger
// than the GS domain would otherwise receive extrapolated flux, which can create
// a spurious current sheet at the overhang. Clamping yields a locally uniform
// field there instead, which is benign.
inline Real sample_psi(const EllipticGrid& g, const ScalarField& psi,
                       Real r, Real z) {
  return sample_bilinear(g, psi, r, z);
}

inline void validate_projection_target(const Grid2D& grid) {
  const Real inner_sampled_face = grid.r_at_edge(-grid.nghost);
  if (!(inner_sampled_face > Real{0})) {
    throw std::invalid_argument{
        "equilibrium projection requires an annular target grid whose ghost "
        "faces remain at positive radius"};
  }
}

// Build the staggered background by differencing psi on cell corners.
//
// `mhd_grid` is interpreted cylindrically: x is the major radius r, y is the
// axial coordinate z. `f_of_psi_n` supplies F(psi_N) = r*B_phi for the
// out-of-plane component.
template <class FOfPsiN>
StaggeredBackground project_to_mhd(const EllipticGrid& g,
                                   const ScalarField& psi,
                                   const CriticalPointSet& cps,
                                   const Grid2D& mhd_grid,
                                   FOfPsiN&& f_of_psi_n) {
  validate_projection_target(mhd_grid);
  StaggeredBackground out;
  out.grid = mhd_grid;
  const std::size_t n = mhd_grid.storage_size();
  out.b0x_face.assign(n, Real{0});
  out.b0y_face.assign(n, Real{0});
  out.b0z_cell.assign(n, Real{0});

  const Real dz = mhd_grid.dy();
  const int nghost = mhd_grid.nghost;

  for (int j = -nghost; j < mhd_grid.ny + nghost; ++j) {
    for (int i = -nghost; i < mhd_grid.nx + nghost; ++i) {
      const std::size_t k = mhd_grid.index(i, j);

      // Corner coordinates of cell (i,j): the low corner is (r_edge, z_edge).
      const Real r_lo = mhd_grid.r_at_edge(i);
      const Real r_hi = mhd_grid.r_at_edge(i + 1);
      const Real z_lo = fma(static_cast<Real>(j), dz, mhd_grid.origin_y);
      const Real z_hi = z_lo + dz;

      // psi at the four corners.
      const Real p00 = sample_psi(g, psi, r_lo, z_lo);
      const Real p01 = sample_psi(g, psi, r_lo, z_hi);
      const Real p10 = sample_psi(g, psi, r_hi, z_lo);

      // B_r on the x-low face: -(1/r) dpsi/dz, discretized as a corner
      // difference along the face.
      if (r_lo > Real{0}) {
        out.b0x_face[k] = -(p01 - p00) / (dz * r_lo);
      }

      // B_z on the y-low face: (1/r) dpsi/dr in annular form. The measure
      // int(r dr) over [r_lo, r_hi] is (r_hi^2 - r_lo^2)/2.
      const Real ring = Real{0.5} * (r_hi * r_hi - r_lo * r_lo);
      if (ring > Real{0}) {
        out.b0y_face[k] = (p10 - p00) / ring;
      }

      // B_phi = F(psi_N)/r at the cell center.
      const Real rc = mhd_grid.r_at_cell_center(i);
      const Real zc = mhd_grid.y_at_cell_center(j);
      if (rc > Real{0}) {
        const Real pc = sample_psi(g, psi, rc, zc);
        const Real pn = normalized_flux(pc, cps.psi_axis, cps.psi_boundary);
        out.b0z_cell[k] = f_of_psi_n(pn) / rc;
      }
    }
  }
  return out;
}

// Discrete divergence of the staggered field, in the same annular form the MHD
// CT scheme uses:
//
//   div B = [r_hi Bx(i+1,j) - r_lo Bx(i,j)] / int(r dr)
//         + [By(i,j+1) - By(i,j)] / dz
//
// Returns the maximum over interior cells. This is the acceptance criterion for
// a background the MHD solver will take: MhdSolver2D::seed_background rejects a
// field that is not discretely solenoidal.
inline Real max_divergence(const StaggeredBackground& sb) {
  const Grid2D& g = sb.grid;
  const Real dz = g.dy();
  Real worst = Real{0};
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const Real r_lo = g.r_at_edge(i);
      const Real r_hi = g.r_at_edge(i + 1);
      const Real ring = Real{0.5} * (r_hi * r_hi - r_lo * r_lo);
      if (ring <= Real{0}) continue;
      const Real radial = (r_hi * sb.b0x_face[g.index(i + 1, j)]
                         - r_lo * sb.b0x_face[g.index(i, j)]) / ring;
      const Real axial = (sb.b0y_face[g.index(i, j + 1)]
                        - sb.b0y_face[g.index(i, j)]) / dz;
      worst = std::max(worst, std::abs(radial + axial));
    }
  }
  return worst;
}

// Scale-free field magnitude, used to turn max_divergence into a relative
// measure. A bare absolute divergence is meaningless without it.
inline Real field_scale(const StaggeredBackground& sb) {
  Real m = Real{0};
  for (const Real v : sb.b0x_face) m = std::max(m, std::abs(v));
  for (const Real v : sb.b0y_face) m = std::max(m, std::abs(v));
  return m;
}

// Pressure and density profiles sampled on MHD cell centers, for seeding the
// evolved fluid state in force balance with the projected field.
struct FluidSeed {
  std::vector<Real> rho{};
  std::vector<Real> pressure{};
};

template <class PressureOfPsiN>
FluidSeed project_fluid(const EllipticGrid& g, const ScalarField& psi,
                        const CriticalPointSet& cps, const Grid2D& mhd_grid,
                        PressureOfPsiN&& p_of_psi_n, Real rho_edge,
                        Real rho_axis) {
  validate_projection_target(mhd_grid);
  FluidSeed seed;
  const std::size_t n = mhd_grid.storage_size();
  seed.rho.assign(n, rho_edge);
  seed.pressure.assign(n, Real{0});
  const int nghost = mhd_grid.nghost;

  for (int j = -nghost; j < mhd_grid.ny + nghost; ++j) {
    for (int i = -nghost; i < mhd_grid.nx + nghost; ++i) {
      const std::size_t k = mhd_grid.index(i, j);
      const Real rc = mhd_grid.r_at_cell_center(i);
      const Real zc = mhd_grid.y_at_cell_center(j);
      const Real pc = sample_psi(g, psi, rc, zc);
      const Real pn = normalized_flux(pc, cps.psi_axis, cps.psi_boundary);
      seed.pressure[k] = p_of_psi_n(pn);
      // Linear in psi_N: peaked on axis, falling to the edge value.
      seed.rho[k] = rho_axis + (rho_edge - rho_axis) * pn;
    }
  }
  return seed;
}

}  // namespace quasar::equilibrium
