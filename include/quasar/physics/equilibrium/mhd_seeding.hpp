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
//   B_r(i,j) = -(psi[j+1][i] - psi[j][i]) / (r_i dz)        (x-low face)
//   B_z(i,j) =  (psi[j][i+1] - psi[j][i]) / int(r dr)       (y-low face)
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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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
  grid.validate();
  // The same storage array holds cell values and staggered faces. At least one
  // halo slot is therefore required to represent Bx(nx,j) and By(i,ny), the two
  // physical high faces consumed by the last interior cell's divergence.
  if (grid.nghost < 1) {
    throw std::invalid_argument{
        "equilibrium projection requires at least one target ghost cell"};
  }
  const Real inner_sampled_face = grid.r_at_edge(-grid.nghost);
  if (!(inner_sampled_face > Real{0})) {
    throw std::invalid_argument{
        "equilibrium projection requires an annular target grid whose ghost "
        "faces remain at positive radius"};
  }
  const Real outer_sampled_face = grid.r_at_edge(grid.nx + grid.nghost);
  const Real lower_sampled_z =
      std::fma(-static_cast<Real>(grid.nghost), grid.dy(), grid.origin_y);
  const Real upper_sampled_z = std::fma(
      static_cast<Real>(grid.ny + grid.nghost), grid.dy(), grid.origin_y);
  if (!std::isfinite(outer_sampled_face) ||
      !std::isfinite(lower_sampled_z) || !std::isfinite(upper_sampled_z)) {
    throw std::overflow_error{
        "equilibrium projection target ghost coordinates are not representable"};
  }
}

inline void validate_projection_source(const EllipticGrid& g,
                                       const ScalarField& psi,
                                       const CriticalPointSet& cps) {
  g.validate();
  if (psi.size() != g.size()) {
    throw std::invalid_argument{
        "equilibrium projection psi size does not match the source grid"};
  }
  if (!std::all_of(psi.begin(), psi.end(),
                   [](Real value) { return std::isfinite(value); })) {
    throw std::invalid_argument{
        "equilibrium projection psi must contain only finite values"};
  }
  const Real flux_span = cps.psi_boundary - cps.psi_axis;
  if (!cps.axis.valid || cps.axis.kind != CriticalKind::o_point ||
      !cps.has_closed_surface ||
      cps.critical_point_overflow || !std::isfinite(cps.axis.r) ||
      !std::isfinite(cps.axis.z) || !std::isfinite(cps.axis.psi) ||
      !std::isfinite(cps.psi_axis) || cps.axis.psi != cps.psi_axis ||
      !std::isfinite(cps.psi_boundary) || !std::isfinite(flux_span) ||
      flux_span == Real{0} || !(cps.axis.r > g.r_min) ||
      !(cps.axis.r < g.r_max) || !(cps.axis.z > g.z_min) ||
      !(cps.axis.z < g.z_max)) {
    throw std::invalid_argument{
        "equilibrium projection requires finite, coherent, non-overflowed "
        "closed-surface metadata with an interior O-point magnetic axis"};
  }
}

inline std::vector<int> projection_plasma_mask(
    const EllipticGrid& g, const ScalarField& psi,
    const CriticalPointSet& cps) {
  std::vector<int> mask = axis_connected_plasma_mask(
      g, psi, cps.axis.r, cps.axis.z, cps.psi_axis, cps.psi_boundary);
  if (std::none_of(mask.begin(), mask.end(),
                   [](int value) { return value != 0; })) {
    throw std::invalid_argument{
        "equilibrium projection could not identify an axis-connected plasma"};
  }
  return mask;
}

// The connectivity result lives on the source nodes. Classify a target sample
// by its nearest clamped source node: this preserves the topological exclusion
// of private-flux lobes without bilinearly smearing a binary mask across the
// separatrix. Outside the source domain the nearest node is a boundary node,
// which the mask deliberately excludes.
inline bool sample_plasma_mask(const EllipticGrid& g,
                               const std::vector<int>& mask,
                               Real r, Real z) {
  Real fi = (r - g.r_min) / g.dr();
  Real fj = (z - g.z_min) / g.dz();
  fi = std::min(std::max(fi, Real{0}), static_cast<Real>(g.nr - 1));
  fj = std::min(std::max(fj, Real{0}), static_cast<Real>(g.nz - 1));
  const int i = std::min(static_cast<int>(std::floor(fi + Real{0.5})),
                         g.nr - 1);
  const int j = std::min(static_cast<int>(std::floor(fj + Real{0.5})),
                         g.nz - 1);
  return mask[g.index(i, j)] != 0;
}

inline Real projection_profile_coordinate(
    const EllipticGrid& g, const ScalarField& psi,
    const CriticalPointSet& cps, const std::vector<int>& plasma_mask,
    Real r, Real z) {
  if (!sample_plasma_mask(g, plasma_mask, r, z)) return Real{1};
  return normalized_flux(sample_psi(g, psi, r, z),
                         cps.psi_axis, cps.psi_boundary);
}

inline void validate_staggered_background_layout(
    const StaggeredBackground& sb) {
  validate_projection_target(sb.grid);
  const std::size_t expected = sb.grid.storage_size();
  if (sb.b0x_face.size() != expected || sb.b0y_face.size() != expected ||
      sb.b0z_cell.size() != expected) {
    throw std::invalid_argument{
        "staggered background buffers do not match the grid storage size"};
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
  validate_projection_source(g, psi, cps);
  validate_projection_target(mhd_grid);
  const std::vector<int> plasma_mask = projection_plasma_mask(g, psi, cps);
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
      if (!std::isfinite(p00) || !std::isfinite(p01) || !std::isfinite(p10)) {
        throw std::overflow_error{
            "equilibrium projection produced a non-finite corner flux"};
      }

      // B_r on the x-low face: -(1/r) dpsi/dz, discretized as a corner
      // difference along the face.
      const Real br_denominator = dz * r_lo;
      if (!(std::isfinite(br_denominator) && br_denominator > Real{0})) {
        throw std::overflow_error{
            "equilibrium projection radial-face measure is not representable"};
      }
      out.b0x_face[k] = -(p01 - p00) / br_denominator;

      // B_z on the y-low face: (1/r) dpsi/dr in annular form. The measure
      // int(r dr) over [r_lo, r_hi] is (r_hi^2 - r_lo^2)/2.
      const Real ring = mhd_grid.dx() * mhd_grid.r_at_cell_center(i);
      if (!(std::isfinite(ring) && ring > Real{0})) {
        throw std::overflow_error{
            "equilibrium projection annular face measure is not representable"};
      }
      out.b0y_face[k] = (p10 - p00) / ring;
      if (!std::isfinite(out.b0x_face[k]) ||
          !std::isfinite(out.b0y_face[k])) {
        throw std::overflow_error{
            "equilibrium projection produced a non-finite poloidal field"};
      }

      // B_phi = F(psi_N)/r at the cell center.
      const Real rc = mhd_grid.r_at_cell_center(i);
      const Real zc = mhd_grid.y_at_cell_center(j);
      const Real pn = projection_profile_coordinate(
          g, psi, cps, plasma_mask, rc, zc);
      const Real f_value = f_of_psi_n(pn);
      if (!std::isfinite(f_value)) {
        throw std::invalid_argument{
            "equilibrium projection F profile produced a non-finite value"};
      }
      out.b0z_cell[k] = f_value / rc;
      if (!std::isfinite(out.b0z_cell[k])) {
        throw std::overflow_error{
            "equilibrium projection produced a non-finite toroidal field"};
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
  validate_staggered_background_layout(sb);
  const Grid2D& g = sb.grid;
  const Real dz = g.dy();
  Real worst = Real{0};
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const Real r_lo = g.r_at_edge(i);
      const Real r_hi = g.r_at_edge(i + 1);
      const Real ring = g.dx() * g.r_at_cell_center(i);
      if (!(std::isfinite(ring) && ring > Real{0})) {
        return std::numeric_limits<Real>::infinity();
      }
      const Real bx_hi = sb.b0x_face[g.index(i + 1, j)];
      const Real bx_lo = sb.b0x_face[g.index(i, j)];
      const Real by_hi = sb.b0y_face[g.index(i, j + 1)];
      const Real by_lo = sb.b0y_face[g.index(i, j)];
      if (!std::isfinite(bx_hi) || !std::isfinite(bx_lo) ||
          !std::isfinite(by_hi) || !std::isfinite(by_lo)) {
        return std::numeric_limits<Real>::infinity();
      }
      const Real radial = (r_hi * bx_hi - r_lo * bx_lo) / ring;
      const Real axial = (by_hi - by_lo) / dz;
      const Real divergence = radial + axial;
      if (!std::isfinite(divergence)) {
        return std::numeric_limits<Real>::infinity();
      }
      worst = std::max(worst, std::abs(divergence));
    }
  }
  return worst;
}

// Scale-free field magnitude, used to turn max_divergence into a relative
// measure. A bare absolute divergence is meaningless without it.
inline Real field_scale(const StaggeredBackground& sb) {
  validate_staggered_background_layout(sb);
  Real m = Real{0};
  for (const Real v : sb.b0x_face) {
    if (!std::isfinite(v)) return std::numeric_limits<Real>::infinity();
    m = std::max(m, std::abs(v));
  }
  for (const Real v : sb.b0y_face) {
    if (!std::isfinite(v)) return std::numeric_limits<Real>::infinity();
    m = std::max(m, std::abs(v));
  }
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
  validate_projection_source(g, psi, cps);
  validate_projection_target(mhd_grid);
  if (!(std::isfinite(rho_edge) && rho_edge > Real{0}) ||
      !(std::isfinite(rho_axis) && rho_axis > Real{0})) {
    throw std::invalid_argument{
        "equilibrium fluid projection densities must be finite and positive"};
  }
  const std::vector<int> plasma_mask = projection_plasma_mask(g, psi, cps);
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
      const Real pn = projection_profile_coordinate(
          g, psi, cps, plasma_mask, rc, zc);
      seed.pressure[k] = p_of_psi_n(pn);
      if (!(std::isfinite(seed.pressure[k]) && seed.pressure[k] >= Real{0})) {
        throw std::invalid_argument{
            "equilibrium pressure profile must produce finite, non-negative values"};
      }
      // Linear in psi_N: peaked on axis, falling to the edge value.
      seed.rho[k] = (Real{1} - pn) * rho_axis + pn * rho_edge;
      if (!(std::isfinite(seed.rho[k]) && seed.rho[k] > Real{0})) {
        throw std::overflow_error{
            "equilibrium fluid projection produced an invalid density"};
      }
    }
  }
  return seed;
}

}  // namespace quasar::equilibrium
