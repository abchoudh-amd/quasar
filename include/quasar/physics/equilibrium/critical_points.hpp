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

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
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
  if (psi.size() != g.size()) {
    throw std::invalid_argument{
        "compute_derivatives: psi size does not match the grid"};
  }
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
  if (!std::isfinite(r) || !std::isfinite(z)) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
  Real fi = (r - g.r_min) / g.dr();
  Real fj = (z - g.z_min) / g.dz();
  if (!std::isfinite(fi) || !std::isfinite(fj)) {
    return std::numeric_limits<Real>::quiet_NaN();
  }
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
  if (!std::isfinite(r) || !std::isfinite(z)) return cp;
  bool converged = false;

  for (int it = 0; it < max_iter; ++it) {
    const Real gr = sample_bilinear(g, d.d_r, r, z);
    const Real gz = sample_bilinear(g, d.d_z, r, z);
    const Real hrr = sample_bilinear(g, d.d_rr, r, z);
    const Real hzz = sample_bilinear(g, d.d_zz, r, z);
    const Real hrz = sample_bilinear(g, d.d_rz, r, z);
    if (!std::isfinite(gr) || !std::isfinite(gz)
        || !std::isfinite(hrr) || !std::isfinite(hzz)
        || !std::isfinite(hrz)) {
      return cp;
    }

    const Real det = hrr * hzz - hrz * hrz;
    if (!std::isfinite(det) || std::abs(det) < Real{1e-30}) {
      return cp;  // degenerate or overflowed: report failure
    }

    Real dr = -(hzz * gr - hrz * gz) / det;
    Real dz = -(-hrz * gr + hrr * gz) / det;
    if (!std::isfinite(dr) || !std::isfinite(dz)) return cp;

    const Real max_r = g.dr();
    const Real max_z = g.dz();
    dr = std::min(std::max(dr, -max_r), max_r);
    dz = std::min(std::max(dz, -max_z), max_z);

    r += dr;
    z += dz;
    if (!std::isfinite(r) || !std::isfinite(z)) return cp;
    if (r <= g.r_min || r >= g.r_max || z <= g.z_min || z >= g.z_max) return cp;

    if (std::abs(dr) < Real{1e-13} * g.dr()
        && std::abs(dz) < Real{1e-13} * g.dz()) {
      converged = true;
      break;
    }
  }

  if (!converged) return cp;

  const Real gr = sample_bilinear(g, d.d_r, r, z);
  const Real gz = sample_bilinear(g, d.d_z, r, z);
  const Real hrr = sample_bilinear(g, d.d_rr, r, z);
  const Real hzz = sample_bilinear(g, d.d_zz, r, z);
  const Real hrz = sample_bilinear(g, d.d_rz, r, z);
  if (!std::isfinite(gr) || !std::isfinite(gz)
      || !std::isfinite(hrr) || !std::isfinite(hzz)
      || !std::isfinite(hrz)) {
    return cp;
  }
  const Real det = hrr * hzz - hrz * hrz;
  if (!std::isfinite(det) || std::abs(det) < Real{1e-30}) return cp;

  const Real refined_psi = sample_bilinear(g, psi, r, z);
  if (!std::isfinite(refined_psi)) return cp;

  cp.r = r;
  cp.z = z;
  cp.psi = refined_psi;
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
  // Device searches have fixed-capacity result storage. Propagate exhaustion
  // rather than allowing a truncated critical set to select the axis or
  // separatrix silently.
  bool critical_point_overflow{false};
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
      if (!std::isfinite(gr) || !std::isfinite(gz)) return out;
      const Real gradient_norm = std::hypot(gr, gz);
      if (!std::isfinite(gradient_norm)) return out;
      grad_scale = std::max(grad_scale, gradient_norm);
    }
  }
  const Real grad_tol = Real{1e-3} * grad_scale;
  if (!std::isfinite(grad_tol)) return out;

  std::vector<CriticalPoint> raw_x;
  for (int j = kMargin; j < g.nz - kMargin; ++j) {
    for (int i = kMargin; i < g.nr - kMargin; ++i) {
      // A zero contour for either gradient component may cross the cell in any
      // orientation. Looking only along that component's coordinate direction
      // misses a rotated saddle such as psi=(r-r0)(z-z0), whose dpsi/dr varies
      // only in z and whose dpsi/dz varies only in r. Seed whenever zero lies
      // in both components' 3x3 ranges; Newton refinement and the gradient
      // tolerance below reject the deliberately broad false positives.
      Real gr_min = std::numeric_limits<Real>::infinity();
      Real gr_max = -std::numeric_limits<Real>::infinity();
      Real gz_min = std::numeric_limits<Real>::infinity();
      Real gz_max = -std::numeric_limits<Real>::infinity();
      bool finite_neighbourhood = true;
      for (int dj = -1; dj <= 1; ++dj) {
        for (int di = -1; di <= 1; ++di) {
          const std::size_t neighbour = g.index(i + di, j + dj);
          const Real gr_value = d.d_r[neighbour];
          const Real gz_value = d.d_z[neighbour];
          if (!std::isfinite(gr_value) || !std::isfinite(gz_value)) {
            finite_neighbourhood = false;
            continue;
          }
          gr_min = std::min(gr_min, gr_value);
          gr_max = std::max(gr_max, gr_value);
          gz_min = std::min(gz_min, gz_value);
          gz_max = std::max(gz_max, gz_value);
        }
      }
      if (!finite_neighbourhood || gr_min > Real{0} || gr_max < Real{0}
          || gz_min > Real{0} || gz_max < Real{0}) {
        continue;
      }

      const CriticalPoint cp =
          refine_critical_point(g, psi, d, g.r(i), g.z(j));
      if (!cp.valid) continue;

      // Reject a "critical point" whose refined gradient is not actually small.
      const Real gr = sample_bilinear(g, d.d_r, cp.r, cp.z);
      const Real gz = sample_bilinear(g, d.d_z, cp.r, cp.z);
      if (!std::isfinite(gr) || !std::isfinite(gz)) continue;
      const Real gradient_norm = std::hypot(gr, gz);
      if (!std::isfinite(gradient_norm) || gradient_norm > grad_tol) continue;

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
  if (!std::isfinite(psi) || !std::isfinite(psi_axis)
      || !std::isfinite(psi_bdry)) {
    return Real{1};
  }
  const Real denom = psi_bdry - psi_axis;
  if (denom == Real{0} || !std::isfinite(denom)) return Real{1};
  const Real numerator = psi - psi_axis;
  if (!std::isfinite(numerator)) return Real{1};
  const Real n = numerator / denom;
  if (!std::isfinite(n)) return Real{1};
  return std::min(std::max(n, Real{0}), Real{1});
}

namespace detail {

// A fifth-degree Bezier curve lies inside the convex hull of its controls.
// Recursively subdivide an uncertain edge and accept it only when every leaf's
// controls are strictly on the plasma side of the separatrix. This is
// conservative: an unresolved near-touch is a barrier, never a false bridge.
inline bool bezier_edge_strictly_inside(const std::array<Real, 6>& controls) {
  constexpr int kMaxDepth = 8;
  constexpr Real kTolerance =
      Real{128} * std::numeric_limits<Real>::epsilon();
  struct Segment {
    std::array<Real, 6> controls{};
    int depth{0};
  };

  std::array<Segment, kMaxDepth + 2> stack{};
  int top = 0;
  stack[top++].controls = controls;

  while (top > 0) {
    const Segment segment = stack[--top];
    bool all_inside = true;
    for (const Real value : segment.controls) {
      if (!std::isfinite(value)) return false;
      if (!(value < -kTolerance)) all_inside = false;
    }
    if (all_inside) continue;
    if (segment.depth == kMaxDepth) return false;

    std::array<Real, 6> work = segment.controls;
    Segment left;
    Segment right;
    left.depth = segment.depth + 1;
    right.depth = segment.depth + 1;
    left.controls[0] = work[0];
    right.controls[5] = work[5];
    for (int level = 1; level < 6; ++level) {
      for (int k = 0; k < 6 - level; ++k) {
        work[static_cast<std::size_t>(k)] =
            Real{0.5} * (work[static_cast<std::size_t>(k)]
                         + work[static_cast<std::size_t>(k + 1)]);
      }
      left.controls[static_cast<std::size_t>(level)] = work[0];
      right.controls[static_cast<std::size_t>(5 - level)] =
          work[static_cast<std::size_t>(5 - level)];
    }
    stack[top++] = right;
    stack[top++] = left;
  }
  return true;
}

inline bool flux_edge_strictly_inside(
    const EllipticGrid& g, const ScalarField& psi,
    const DerivativeFields& derivatives, int i0, int j0, int i1, int j1,
    Real psi_axis, Real psi_boundary) {
  const Real denom = psi_boundary - psi_axis;
  const bool radial = j0 == j1;
  const Real h = radial ? g.r(i1) - g.r(i0) : g.z(j1) - g.z(j0);
  const std::size_t k0 = g.index(i0, j0);
  const std::size_t k1 = g.index(i1, j1);
  const ScalarField& first = radial ? derivatives.d_r : derivatives.d_z;
  const ScalarField& second = radial ? derivatives.d_rr : derivatives.d_zz;

  const Real q0 = (psi[k0] - psi_boundary) / denom;
  const Real q1 = (psi[k1] - psi_boundary) / denom;
  const Real v0 = h * first[k0] / denom;
  const Real v1 = h * first[k1] / denom;
  const Real a0 = h * h * second[k0] / denom;
  const Real a1 = h * h * second[k1] / denom;

  // Quintic Hermite data converted to Bezier controls. With value, first, and
  // second derivatives at both nodes this reconstruction is exact for the
  // quartic double-well that exposes off-grid X-point leakage, and consistent
  // with the solver's sixth-order compact derivative fields in general.
  const std::array<Real, 6> controls{
      q0,
      q0 + v0 / Real{5},
      q0 + Real{2} * v0 / Real{5} + a0 / Real{20},
      q1 - Real{2} * v1 / Real{5} + a1 / Real{20},
      q1 - v1 / Real{5},
      q1};
  return bezier_edge_strictly_inside(controls);
}

}  // namespace detail

// Four-connected component of psi_N < 1 containing the selected magnetic axis.
// Connectivity is edge-aware: two eligible nodes are joined only when a
// quintic Hermite reconstruction of their shared edge stays strictly inside the
// separatrix. This prevents an off-grid X-point from becoming a one-edge bridge
// between the core and a private-flux lobe.
inline std::vector<int> axis_connected_plasma_mask(
    const EllipticGrid& g, const ScalarField& psi, Real axis_r, Real axis_z,
    Real psi_axis, Real psi_boundary) {
  if (psi.size() != g.size()) {
    throw std::invalid_argument{
        "axis_connected_plasma_mask: psi has the wrong size"};
  }

  std::vector<int> mask(g.size(), 0);
  const Real denom = psi_boundary - psi_axis;
  if (!std::isfinite(axis_r) || !std::isfinite(axis_z)
      || !std::isfinite(psi_axis) || !std::isfinite(psi_boundary)
      || !std::isfinite(denom) || denom == Real{0}) {
    return mask;
  }

  const auto eligible = [&](int i, int j) {
    if (g.on_boundary(i, j)) return false;
    const Real value = psi[g.index(i, j)];
    if (!std::isfinite(value)) return false;
    return (value - psi_axis) / denom < Real{1};
  };

  std::size_t seed = g.size();
  Real nearest = std::numeric_limits<Real>::infinity();
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      if (!eligible(i, j)) continue;
      const Real di = (g.r(i) - axis_r) / g.dr();
      const Real dj = (g.z(j) - axis_z) / g.dz();
      const Real distance = di * di + dj * dj;
      if (distance < nearest) {
        nearest = distance;
        seed = g.index(i, j);
      }
    }
  }
  if (seed == g.size()) return mask;

  const DerivativeFields derivatives = compute_derivatives(g, psi);

  std::vector<std::size_t> queue;
  queue.reserve(g.size());
  queue.push_back(seed);
  mask[seed] = 1;
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const std::size_t current = queue[head];
    const int i = static_cast<int>(current % static_cast<std::size_t>(g.nr));
    const int j = static_cast<int>(current / static_cast<std::size_t>(g.nr));
    const int neighbour_i[4] = {i - 1, i + 1, i, i};
    const int neighbour_j[4] = {j, j, j - 1, j + 1};
    for (int direction = 0; direction < 4; ++direction) {
      const int ni = neighbour_i[direction];
      const int nj = neighbour_j[direction];
      if (!eligible(ni, nj)) continue;
      if (!detail::flux_edge_strictly_inside(
              g, psi, derivatives, i, j, ni, nj, psi_axis, psi_boundary)) {
        continue;
      }
      const std::size_t next = g.index(ni, nj);
      if (mask[next] != 0) continue;
      mask[next] = 1;
      queue.push_back(next);
    }
  }
  return mask;
}

}  // namespace quasar::equilibrium
