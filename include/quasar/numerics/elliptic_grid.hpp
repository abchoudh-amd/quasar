#pragma once

// Node-centered logical grid for the elliptic (boundary-value) solver path.
//
// This is deliberately NOT Grid2D. Grid2D is cell-centered with a ghost halo,
// which is the right shape for a finite-volume hyperbolic update: values live at
// cell centers and boundary conditions are imposed by filling ghosts. An
// elliptic Dirichlet problem wants the opposite -- the boundary value IS a
// degree of freedom sitting exactly on the domain edge, and the compact
// derivative closures in pade_derivative.hpp are derived for a line whose first
// and last points are the boundary. Forcing that onto a cell-centered halo grid
// would mean interpolating the boundary condition, which caps the scheme's order
// at the interpolation order and defeats the point of a sixth-order operator.
//
// The two grids coexist: the Grad-Shafranov solve runs on EllipticGrid, and the
// result is projected onto the MHD Grid2D staggering at the consumer boundary
// (see physics/equilibrium/flux_surfaces.hpp).
//
// Multigrid coarsening halves each interval, so a level with `n` nodes coarsens
// to (n-1)/2 + 1 nodes. Grids sized 2^k + 1 coarsen cleanly all the way down;
// `coarsenable_levels()` reports how far a given size actually goes.

#include "quasar/core/types.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quasar::numerics {

struct EllipticGrid {
  int  nr{0};
  int  nz{0};
  Real r_min{0};
  Real r_max{0};
  Real z_min{0};
  Real z_max{0};

  EllipticGrid() = default;

  EllipticGrid(int nr_in, int nz_in, Real r0, Real r1, Real z0, Real z1)
    : nr{nr_in}, nz{nz_in}, r_min{r0}, r_max{r1}, z_min{z0}, z_max{z1} {
    validate();
  }

  void validate() const {
    if (nr < 3 || nz < 3) {
      throw std::invalid_argument{"EllipticGrid: need at least 3 nodes per axis"};
    }
    if (!(r_max > r_min) || !(z_max > z_min)) {
      throw std::invalid_argument{"EllipticGrid: extents must be increasing"};
    }
    // The annular-domain decision: r_min > 0 removes the 1/r coordinate
    // singularity in Delta* entirely. A domain touching or crossing the axis
    // needs a regularity closure (psi ~ r^2) that this grid does not provide.
    if (!(r_min > Real{0})) {
      throw std::invalid_argument{
          "EllipticGrid: r_min must be strictly positive (annular domain); "
          "an axis-touching domain needs a regularity closure"};
    }
  }

  Real dr() const noexcept {
    return (r_max - r_min) / static_cast<Real>(nr - 1);
  }
  Real dz() const noexcept {
    return (z_max - z_min) / static_cast<Real>(nz - 1);
  }
  Real r(int i) const noexcept {
    return r_min + static_cast<Real>(i) * dr();
  }
  Real z(int j) const noexcept {
    return z_min + static_cast<Real>(j) * dz();
  }
  // Radius at the half-node between i and i+1; used by the conservative form of
  // the radial operator.
  Real r_half(int i) const noexcept {
    return r_min + (static_cast<Real>(i) + Real{0.5}) * dr();
  }

  std::size_t index(int i, int j) const noexcept {
    return static_cast<std::size_t>(i)
         + static_cast<std::size_t>(nr) * static_cast<std::size_t>(j);
  }
  std::size_t size() const noexcept {
    return static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz);
  }

  bool on_boundary(int i, int j) const noexcept {
    return i == 0 || j == 0 || i == nr - 1 || j == nz - 1;
  }

  bool can_coarsen() const noexcept {
    return (nr - 1) % 2 == 0 && (nz - 1) % 2 == 0 && nr >= 5 && nz >= 5;
  }

  EllipticGrid coarsen() const {
    if (!can_coarsen()) {
      throw std::logic_error{"EllipticGrid::coarsen: level is not coarsenable"};
    }
    return EllipticGrid{(nr - 1) / 2 + 1, (nz - 1) / 2 + 1,
                        r_min, r_max, z_min, z_max};
  }

  // Number of additional levels below this one, capped by the coarsest usable
  // grid rather than by a fixed depth.
  int coarsenable_levels() const {
    int levels = 0;
    EllipticGrid g = *this;
    while (g.can_coarsen()) {
      g = g.coarsen();
      ++levels;
    }
    return levels;
  }
};

using ScalarField = std::vector<Real>;

inline ScalarField make_field(const EllipticGrid& g) {
  return ScalarField(g.size(), Real{0});
}

// Max-norm over interior nodes only. Boundary nodes carry the Dirichlet data and
// are excluded: including them would report zero residual there by construction
// and dilute the interior measurement.
inline Real interior_max_norm(const EllipticGrid& g, const ScalarField& f) {
  Real m = Real{0};
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      const Real v = f[g.index(i, j)];
      m = std::max(m, v < Real{0} ? -v : v);
    }
  }
  return m;
}

}  // namespace quasar::numerics
