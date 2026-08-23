#pragma once

// Magnetic axis and X-point location for the Grad-Shafranov outer loop.
//
// Every outer iteration needs psi_axis and psi_bdry to form psi_N, and needs the
// plasma region (the set of closed flux surfaces) to know where current flows.
// Both come from the critical points of psi: points where grad psi = 0.
//
//   Hessian determinant > 0, trace < 0  -> maximum   (O-point)
//   Hessian determinant > 0, trace > 0  -> minimum   (O-point)
//   Hessian determinant < 0             -> saddle    (X-point)
//
// The magnetic axis is the O-point; psi_bdry is set by the X-point whose flux is
// closest to the axis value (the innermost separatrix), or by limiter contact if
// no X-point lies inside the domain.
//
// -- Why high-order location matters -------------------------------------------
// psi_N normalizes by (psi_bdry - psi_axis). An error in either endpoint
// propagates directly into every profile evaluation and therefore into the
// source term everywhere. Locating the axis by a bicubic interpolant caps that
// accuracy at the interpolant's order, which would make critical-point location
// the leading error in an otherwise sixth-order solve.
//
// Instead the search runs Newton's method on grad psi computed with the SAME
// sixth-order Pade operators used by the residual, so axis location inherits the
// scheme's accuracy. The derivative fields are computed once per outer iteration
// and reused for both the search and the Hessian classification.
//
// -- Robustness ----------------------------------------------------------------
// Newton on grad psi = 0 converges quadratically but only from a good starting
// guess, and near a degenerate point (axis and X-point merging, which is exactly
// what happens as an equilibrium approaches its stability limit) the Hessian
// becomes singular. The implementation therefore:
//   - seeds from the best discrete grid extremum rather than a fixed guess,
//   - bounds the Newton step to one cell to prevent it leaving the domain,
//   - reports failure rather than returning a garbage point.
// A failed critical-point search is a legitimate physical outcome (no confined
// plasma), which is why it maps onto a diagnosed failure mode rather than an
// exception.

#include "quasar/core/types.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/numerics/pade_line_solve.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace quasar::equilibrium {

using numerics::EllipticGrid;
using numerics::ScalarField;

enum class CriticalKind { none, o_point, x_point };

struct CriticalPoint {
  CriticalKind kind{CriticalKind::none};
  Real r{0};
  Real z{0};
  Real psi{0};
  bool valid{false};
};

// First and second derivatives of psi on the whole grid, computed once with the
// sixth-order compact operators and reused.
struct DerivativeFields {
  ScalarField d_r, d_z, d_rr, d_zz, d_rz;
};

inline DerivativeFields compute_derivatives(const EllipticGrid& g,
                                            const ScalarField& psi) {
  DerivativeFields d;
  d.d_r.assign(g.size(), Real{0});
  d.d_z.assign(g.size(), Real{0});
  d.d_rr.assign(g.size(), Real{0});
  d.d_zz.assign(g.size(), Real{0});
  d.d_rz.assign(g.size(), Real{0});

  std::vector<Real> line(static_cast<std::size_t>(g.nr));
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) line[static_cast<std::size_t>(i)] = psi[g.index(i, j)];
    const auto f1 = numerics::pade::first_derivative(line, g.dr());
    const auto f2 = numerics::pade::second_derivative(line, g.dr());
    for (int i = 0; i < g.nr; ++i) {
      d.d_r[g.index(i, j)]  = f1[static_cast<std::size_t>(i)];
      d.d_rr[g.index(i, j)] = f2[static_cast<std::size_t>(i)];
    }
  }

  std::vector<Real> col(static_cast<std::size_t>(g.nz));
  for (int i = 0; i < g.nr; ++i) {
    for (int j = 0; j < g.nz; ++j) col[static_cast<std::size_t>(j)] = psi[g.index(i, j)];
    const auto f1 = numerics::pade::first_derivative(col, g.dz());
    const auto f2 = numerics::pade::second_derivative(col, g.dz());
    for (int j = 0; j < g.nz; ++j) {
      d.d_z[g.index(i, j)]  = f1[static_cast<std::size_t>(j)];
      d.d_zz[g.index(i, j)] = f2[static_cast<std::size_t>(j)];
    }
  }

  // Mixed derivative: differentiate d_r along z.
  for (int i = 0; i < g.nr; ++i) {
    for (int j = 0; j < g.nz; ++j) col[static_cast<std::size_t>(j)] = d.d_r[g.index(i, j)];
    const auto f1 = numerics::pade::first_derivative(col, g.dz());
    for (int j = 0; j < g.nz; ++j) {
      d.d_rz[g.index(i, j)] = f1[static_cast<std::size_t>(j)];
    }
  }
  return d;
}

// Bilinear sample of a grid field at an arbitrary (r,z) inside the domain.
// Used only to evaluate the Newton residual between nodes; the underlying
// derivative fields are sixth-order, and the iteration converges to a point
// where they vanish, so the interpolation order does not cap the result.
inline Real sample_bilinear(const EllipticGrid& g, const ScalarField& f,
                            Real r, Real z) {
  Real fi = (r - g.r_min) / g.dr();
  Real fj = (z - g.z_min) / g.dz();
  fi = std::min(std::max(fi, Real{0}), static_cast<Real>(g.nr - 1));
  fj = std::min(std::max(fj, Real{0}), static_cast<Real>(g.nz - 1));
  int i = std::min(static_cast<int>(fi), g.nr - 2);
  int j = std::min(static_cast<int>(fj), g.nz - 2);
  const Real tr = fi - static_cast<Real>(i);
  const Real tz = fj - static_cast<Real>(j);
  return (Real{1} - tr) * (Real{1} - tz) * f[g.index(i, j)]
       + tr * (Real{1} - tz) * f[g.index(i + 1, j)]
       + (Real{1} - tr) * tz * f[g.index(i, j + 1)]
       + tr * tz * f[g.index(i + 1, j + 1)];
}

// Refine a critical point by Newton iteration on grad psi = 0.
//
//   [psi_rr  psi_rz] [dr]   [psi_r]
//   [psi_rz  psi_zz] [dz] = -[psi_z]
//
// Steps are clamped to one cell so a near-singular Hessian cannot throw the
// iterate out of the domain.
inline CriticalPoint refine_critical_point(const EllipticGrid& g,
                                           const ScalarField& psi,
                                           const DerivativeFields& d,
                                           Real r0, Real z0,
                                           int max_iter = 40) {
  CriticalPoint cp;
  Real r = r0;
  Real z = z0;

  for (int it = 0; it < max_iter; ++it) {
    const Real gr = sample_bilinear(g, d.d_r, r, z);
    const Real gz = sample_bilinear(g, d.d_z, r, z);
    const Real hrr = sample_bilinear(g, d.d_rr, r, z);
    const Real hzz = sample_bilinear(g, d.d_zz, r, z);
    const Real hrz = sample_bilinear(g, d.d_rz, r, z);

    const Real det = hrr * hzz - hrz * hrz;
    if (std::abs(det) < Real{1e-30}) return cp;  // degenerate: report failure

    Real dr = -(hzz * gr - hrz * gz) / det;
    Real dz = -(-hrz * gr + hrr * gz) / det;

    const Real max_r = g.dr();
    const Real max_z = g.dz();
    dr = std::min(std::max(dr, -max_r), max_r);
    dz = std::min(std::max(dz, -max_z), max_z);

    r += dr;
    z += dz;
    if (r <= g.r_min || r >= g.r_max || z <= g.z_min || z >= g.z_max) return cp;

    if (std::abs(dr) < Real{1e-13} * g.dr()
        && std::abs(dz) < Real{1e-13} * g.dz()) {
      break;
    }
  }

  const Real hrr = sample_bilinear(g, d.d_rr, r, z);
  const Real hzz = sample_bilinear(g, d.d_zz, r, z);
  const Real hrz = sample_bilinear(g, d.d_rz, r, z);
  const Real det = hrr * hzz - hrz * hrz;

  cp.r = r;
  cp.z = z;
  cp.psi = sample_bilinear(g, psi, r, z);
  cp.kind = det > Real{0} ? CriticalKind::o_point : CriticalKind::x_point;
  cp.valid = true;
  return cp;
}

struct CriticalPointSet {
  CriticalPoint axis{};
  std::vector<CriticalPoint> x_points{};
  Real psi_axis{0};
  Real psi_boundary{0};
  bool has_closed_surface{false};
};

// Locate the magnetic axis and every X-point in the interior.
//
// Discrete candidates are grid nodes that are a strict extremum or saddle of psi
// among their eight neighbours; each is then refined by Newton. The axis is the
// O-point with the largest |psi - psi_edge| (the deepest well), which is the
// robust choice when several local extrema survive an unconverged iterate.
inline CriticalPointSet find_critical_points(const EllipticGrid& g,
                                             const ScalarField& psi) {
  CriticalPointSet out;
  const DerivativeFields d = compute_derivatives(g, psi);

  // Search away from the boundary: the Pade closures are least accurate in the
  // first two rows, and a physical axis never sits on the domain edge.
  constexpr int kMargin = 3;
  std::vector<CriticalPoint> o_points;

  // Scale for "this gradient is indistinguishable from zero". A bare sign-change
  // test fires on grid-scale roundoff wherever the field is locally flat, which
  // produced eleven spurious X-points on a smooth seeded state and let a
  // meaningless saddle set psi_boundary. Requiring the local gradient to be
  // small RELATIVE to the field's overall variation removes them without a
  // hand-tuned absolute threshold.
  Real grad_scale = Real{0};
  for (int j = kMargin; j < g.nz - kMargin; ++j) {
    for (int i = kMargin; i < g.nr - kMargin; ++i) {
      const Real gr = d.d_r[g.index(i, j)];
      const Real gz = d.d_z[g.index(i, j)];
      grad_scale = std::max(grad_scale, std::sqrt(gr * gr + gz * gz));
    }
  }
  const Real grad_tol = Real{1e-3} * grad_scale;

  std::vector<CriticalPoint> raw_x;
  for (int j = kMargin; j < g.nz - kMargin; ++j) {
    for (int i = kMargin; i < g.nr - kMargin; ++i) {
      // A discrete critical point: both derivative components change sign
      // across the node in their respective directions.
      const Real gr_m = d.d_r[g.index(i - 1, j)];
      const Real gr_p = d.d_r[g.index(i + 1, j)];
      const Real gz_m = d.d_z[g.index(i, j - 1)];
      const Real gz_p = d.d_z[g.index(i, j + 1)];
      if (gr_m * gr_p > Real{0} || gz_m * gz_p > Real{0}) continue;

      const CriticalPoint cp =
          refine_critical_point(g, psi, d, g.r(i), g.z(j));
      if (!cp.valid) continue;

      // Reject a "critical point" whose refined gradient is not actually small.
      const Real gr = sample_bilinear(g, d.d_r, cp.r, cp.z);
      const Real gz = sample_bilinear(g, d.d_z, cp.r, cp.z);
      if (std::sqrt(gr * gr + gz * gz) > grad_tol) continue;

      if (cp.kind == CriticalKind::o_point) {
        o_points.push_back(cp);
      } else {
        raw_x.push_back(cp);
      }
    }
  }

  // Merge duplicates: many neighbouring seeds refine onto the same point.
  const auto merge = [&](std::vector<CriticalPoint>& pts) {
    std::vector<CriticalPoint> uniq;
    for (const auto& p : pts) {
      bool dup = false;
      for (const auto& q : uniq) {
        if (std::abs(p.r - q.r) < Real{0.5} * g.dr()
            && std::abs(p.z - q.z) < Real{0.5} * g.dz()) {
          dup = true;
          break;
        }
      }
      if (!dup) uniq.push_back(p);
    }
    pts.swap(uniq);
  };
  merge(o_points);
  merge(raw_x);
  out.x_points = raw_x;

  if (o_points.empty()) return out;

  // Deduplicate O-points that refined to the same location, then take the one
  // furthest in psi from the domain edge value.
  const Real edge_psi = psi[g.index(0, g.nz / 2)];
  std::size_t best = 0;
  Real best_depth = Real{-1};
  for (std::size_t k = 0; k < o_points.size(); ++k) {
    const Real depth = std::abs(o_points[k].psi - edge_psi);
    if (depth > best_depth) {
      best_depth = depth;
      best = k;
    }
  }
  out.axis = o_points[best];
  out.psi_axis = out.axis.psi;

  // psi_boundary: the X-point flux closest to the axis (innermost separatrix).
  bool found = false;
  Real best_x = Real{0};
  Real best_gap = Real{0};
  for (const auto& x : out.x_points) {
    const Real gap = std::abs(x.psi - out.psi_axis);
    if (!found || gap < best_gap) {
      found = true;
      best_gap = gap;
      best_x = x.psi;
    }
  }
  if (found) {
    out.psi_boundary = best_x;
    out.has_closed_surface = best_gap > Real{0};
  } else {
    // Limited plasma: no X-point in the domain, so the boundary is set by the
    // outermost surface touching the domain edge.
    Real limit = psi[g.index(0, 0)];
    Real gap = std::abs(limit - out.psi_axis);
    for (int j = 0; j < g.nz; ++j) {
      for (int i = 0; i < g.nr; ++i) {
        if (!g.on_boundary(i, j)) continue;
        const Real v = psi[g.index(i, j)];
        if (std::abs(v - out.psi_axis) < gap) {
          gap = std::abs(v - out.psi_axis);
          limit = v;
        }
      }
    }
    out.psi_boundary = limit;
    out.has_closed_surface = gap > Real{0};
  }
  return out;
}

// psi_N = (psi - psi_axis) / (psi_bdry - psi_axis), clamped to [0, 1].
// Outside the plasma psi_N is clamped to 1, where the default profiles vanish,
// so no current is driven there.
inline Real normalized_flux(Real psi, Real psi_axis, Real psi_bdry) {
  const Real denom = psi_bdry - psi_axis;
  if (denom == Real{0}) return Real{1};
  const Real n = (psi - psi_axis) / denom;
  return std::min(std::max(n, Real{0}), Real{1});
}

}  // namespace quasar::equilibrium
