#pragma once

// Free-boundary condition for Grad-Shafranov via the exact Green's function.
//
// On a free-boundary problem psi is not known on the computational boundary: it
// is the superposition of the field from external coils and the field from the
// plasma current being solved for. Both are computed here from the same analytic
// kernel.
//
// -- The axisymmetric Green's function ----------------------------------------
// The poloidal flux at (r, z) from a unit toroidal current filament at
// (r', z') is
//
//   G(r,z; r',z') = (mu0 / 2pi) * sqrt(r r') / k * [ (2 - k^2) K(k) - 2 E(k) ]
//
// with
//
//   k^2 = 4 r r' / [ (r + r')^2 + (z - z')^2 ]
//
// and K, E the complete elliptic integrals of the first and second kind. This is
// exact -- no truncation, no series -- which is why the plan chose it over a
// boundary normal-derivative method: differentiating a numerical psi at the
// boundary would drop an order exactly where the scheme is sixth order.
//
// -- Why AGM for K and E -------------------------------------------------------
// The arithmetic-geometric mean converges quadratically (roughly 4-5 iterations
// to full double precision) and uses only add/multiply/sqrt, so it ports to a
// device kernel unchanged. std::comp_ellint_1/2 exist in C++17 but are not
// available in device code and their host performance is worse.
//
// A caution the AGM makes easy to get wrong: K(k) diverges logarithmically as
// k -> 1, which happens when the field point approaches the source filament.
// That is physically correct (the flux of a filament diverges at the filament)
// but numerically fatal if a boundary point coincides with a current-carrying
// cell. The boundary integral below therefore never evaluates a source at the
// boundary itself, and near-coincident geometry is guarded.
//
// -- Cost ----------------------------------------------------------------------
// The plasma contribution is O(N_boundary * N_interior): every boundary point
// sums over every current-carrying cell. At 256^2 this is ~1e8 fp64 operations
// per outer iteration -- negligible. At 4096^2 it is ~2.7e11 and becomes the
// dominant cost, which is why the plan defers a multipole-accelerated path. The
// interface here is deliberately shaped so that acceleration drops in behind it
// with this exact version retained as the accuracy oracle.

#include "quasar/core/types.hpp"
#include "quasar/numerics/elliptic_grid.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quasar::equilibrium {

using numerics::EllipticGrid;
using numerics::ScalarField;

inline constexpr Real kMu0 = Real{4e-7} * Real{3.14159265358979323846};

// Complete elliptic integrals K(m) and E(m) with PARAMETER m = k^2, computed
// together by the AGM (they share the same iteration).
//
// Using m rather than the modulus k avoids a redundant sqrt at every call site
// and matches the form in which k^2 naturally appears in the Green's function.
struct EllipticKE {
  Real k{0};
  Real e{0};
};

inline EllipticKE complete_elliptic_ke(Real m) {
  // Guard the endpoints. m -> 1 is the coincident-source limit where K
  // diverges; callers must avoid it, but returning a large finite value is
  // safer than a NaN propagating silently through a residual.
  if (m < Real{0}) m = Real{0};
  if (m >= Real{1}) m = Real{1} - Real{1e-16};

  Real a = Real{1};
  Real b = std::sqrt(Real{1} - m);
  Real c = std::sqrt(m);
  Real sum = Real{0};
  Real power = Real{0.5};

  // Quadratic convergence: ~5 iterations reaches double precision.
  for (int it = 0; it < 60; ++it) {
    sum += power * c * c;
    if (std::abs(c) < Real{1e-17} * std::abs(a)) break;
    const Real a_next = Real{0.5} * (a + b);
    const Real b_next = std::sqrt(a * b);
    c = Real{0.5} * (a - b);
    a = a_next;
    b = b_next;
    power *= Real{2};
  }

  EllipticKE out;
  const Real pi = Real{3.14159265358979323846};
  out.k = pi / (Real{2} * a);
  out.e = out.k * (Real{1} - sum);
  return out;
}

// Poloidal flux at (r, z) from a unit toroidal current filament at (rp, zp).
// Returns 0 for a degenerate/coincident geometry rather than a divergence.
inline Real greens_function(Real r, Real z, Real rp, Real zp) {
  if (r <= Real{0} || rp <= Real{0}) return Real{0};
  const Real dz = z - zp;
  const Real sum_r = r + rp;
  const Real denom = sum_r * sum_r + dz * dz;
  if (denom <= Real{0}) return Real{0};

  const Real m = Real{4} * r * rp / denom;
  // Coincident source and field point: the filament's own flux diverges.
  if (m >= Real{1} - Real{1e-14}) return Real{0};

  const EllipticKE ke = complete_elliptic_ke(m);
  const Real k = std::sqrt(m);
  const Real pi = Real{3.14159265358979323846};
  return kMu0 / (Real{2} * pi) * std::sqrt(r * rp) / k
       * ((Real{2} - m) * ke.k - Real{2} * ke.e);
}

// An axisymmetric circular coil: a current filament at fixed (r, z).
struct CoilFilament {
  Real r{0};
  Real z{0};
  Real current{0};  // amperes
};

// psi on the computational boundary from external coils alone.
inline void apply_coil_boundary(const EllipticGrid& g,
                                const std::vector<CoilFilament>& coils,
                                ScalarField& psi) {
  psi.resize(g.size());
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      if (!g.on_boundary(i, j)) continue;
      Real acc = Real{0};
      for (const auto& c : coils) {
        acc += c.current * greens_function(g.r(i), g.z(j), c.r, c.z);
      }
      psi[g.index(i, j)] = acc;
    }
  }
}

// psi anywhere in the domain from external coils alone. Used to seed the
// nonlinear iteration with the vacuum field, which is the standard and
// well-behaved initial guess.
inline void evaluate_coil_field(const EllipticGrid& g,
                                const std::vector<CoilFilament>& coils,
                                ScalarField& psi) {
  psi.resize(g.size());
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      Real acc = Real{0};
      for (const auto& c : coils) {
        acc += c.current * greens_function(g.r(i), g.z(j), c.r, c.z);
      }
      psi[g.index(i, j)] = acc;
    }
  }
}

// Add the plasma's own contribution to the boundary flux.
//
// `j_phi` holds the toroidal current DENSITY on interior nodes; the cell area
// weight dr*dz converts it to a filament current per node. Boundary nodes are
// skipped as sources: a boundary point would then be evaluating the Green's
// function at zero separation.
inline void add_plasma_boundary(const EllipticGrid& g,
                                const ScalarField& j_phi,
                                ScalarField& psi) {
  const Real cell_area = g.dr() * g.dz();
  for (int jb = 0; jb < g.nz; ++jb) {
    for (int ib = 0; ib < g.nr; ++ib) {
      if (!g.on_boundary(ib, jb)) continue;
      const Real rb = g.r(ib);
      const Real zb = g.z(jb);
      Real acc = Real{0};
      for (int js = 1; js < g.nz - 1; ++js) {
        for (int is = 1; is < g.nr - 1; ++is) {
          const Real src = j_phi[g.index(is, js)];
          if (src == Real{0}) continue;
          acc += src * cell_area * greens_function(rb, zb, g.r(is), g.z(js));
        }
      }
      psi[g.index(ib, jb)] += acc;
    }
  }
}

// Total toroidal plasma current, integral of j_phi over the poloidal
// cross-section. This is the quantity the profile normalization targets.
inline Real total_plasma_current(const EllipticGrid& g,
                                 const ScalarField& j_phi) {
  const Real cell_area = g.dr() * g.dz();
  Real acc = Real{0};
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) acc += j_phi[g.index(i, j)];
  }
  return acc * cell_area;
}

}  // namespace quasar::equilibrium
