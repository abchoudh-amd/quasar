#pragma once

// Derived equilibrium quantities: magnetic field, safety factor, and
// flux-surface geometry.
//
// These are computed HERE, inside the equilibrium module, rather than left to
// each consumer. The reason is accuracy: B = curl(psi) and q(psi) are
// derivatives of psi, and the sixth-order Pade operators that make those
// derivatives accurate are available at this point. A consumer handed only psi
// on a grid would have to re-differentiate it with whatever scheme it has --
// typically a bicubic interpolant -- and the sixth-order claim would silently
// stop being true at the module boundary.
//
// -- Field components ----------------------------------------------------------
// In (r, phi, z) with psi the poloidal flux per radian:
//
//   B_r   = -(1/r) dpsi/dz
//   B_z   =  (1/r) dpsi/dr
//   B_phi =  F(psi) / r
//
// B_r and B_z follow from psi alone. B_phi needs F(psi), which is recovered by
// integrating the profile's FF' inward from a prescribed vacuum value at the
// boundary: F^2(psi_N) = F_vac^2 + 2 * integral of FF' dpsi.
//
// -- Safety factor -------------------------------------------------------------
//   q(psi) = (1/2pi) * contour integral of [ B_phi / (r B_poloidal) ] dl
//
// evaluated on a closed flux surface. q is the single most-used equilibrium
// diagnostic and the one a stability code needs first, so it is part of the
// output contract rather than an optional extra.

#include "quasar/core/types.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/critical_points.hpp"
#include "quasar/physics/equilibrium/equilibrium_profile.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quasar::equilibrium {

using numerics::EllipticGrid;
using numerics::ScalarField;

struct MagneticField {
  ScalarField b_r;
  ScalarField b_z;
  ScalarField b_phi;
  ScalarField b_poloidal;  // sqrt(b_r^2 + b_z^2), the denominator in q
};

// B from psi using the sixth-order derivative fields.
//
// `f_of_psi_n` supplies F(psi_N) = R*B_phi; passing a constant reproduces a
// pure vacuum toroidal field, which is the common case for a first equilibrium.
template <class FOfPsiN>
MagneticField compute_field(const EllipticGrid& g, const ScalarField& psi,
                            const CriticalPointSet& cps, FOfPsiN&& f_of_psi_n) {
  const DerivativeFields d = compute_derivatives(g, psi);
  MagneticField b;
  b.b_r.assign(g.size(), Real{0});
  b.b_z.assign(g.size(), Real{0});
  b.b_phi.assign(g.size(), Real{0});
  b.b_poloidal.assign(g.size(), Real{0});

  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      const std::size_t k = g.index(i, j);
      const Real r = g.r(i);
      b.b_r[k] = -d.d_z[k] / r;
      b.b_z[k] =  d.d_r[k] / r;
      const Real pn = normalized_flux(psi[k], cps.psi_axis, cps.psi_boundary);
      b.b_phi[k] = f_of_psi_n(pn) / r;
      b.b_poloidal[k] =
          std::sqrt(b.b_r[k] * b.b_r[k] + b.b_z[k] * b.b_z[k]);
    }
  }
  return b;
}

// Recover F(psi_N) by integrating FF' inward from the boundary.
//
//   d(F^2)/dpsi_N = 2 * FF'(psi_N) * (psi_bdry - psi_axis)
//
// so F^2(psi_N) = F_vac^2 + 2 (psi_b - psi_a) * integral_{1}^{psi_N} FF' ds.
// `profile_scale` is the current-normalization amplitude reported by GsResult;
// using the raw profile would reconstruct a field inconsistent with the solved
// Grad-Shafranov source. The trapezoid rule over `samples` points is used.
inline std::vector<Real> integrate_f_profile(const IEquilibriumProfile& prof,
                                             Real f_vacuum, Real psi_axis,
                                             Real psi_bdry, Real profile_scale,
                                             int samples = 257) {
  if (samples < 2) {
    throw std::invalid_argument{
        "integrate_f_profile: samples must be at least two"};
  }
  std::vector<Real> f2(static_cast<std::size_t>(samples), Real{0});
  const Real dpn = Real{1} / static_cast<Real>(samples - 1);
  const Real scale = Real{2} * (psi_bdry - psi_axis) * profile_scale;

  // Integrate from psi_N = 1 (boundary, index samples-1) inward to 0.
  f2[static_cast<std::size_t>(samples - 1)] = f_vacuum * f_vacuum;
  for (int k = samples - 2; k >= 0; --k) {
    const Real pn_hi = static_cast<Real>(k + 1) * dpn;
    const Real pn_lo = static_cast<Real>(k) * dpn;
    const Real avg = Real{0.5} * (prof.ff_prime(pn_hi) + prof.ff_prime(pn_lo));
    f2[static_cast<std::size_t>(k)] =
        f2[static_cast<std::size_t>(k + 1)] - scale * avg * dpn;
  }

  std::vector<Real> f(static_cast<std::size_t>(samples));
  for (int k = 0; k < samples; ++k) {
    const Real v = f2[static_cast<std::size_t>(k)];
    // F^2 < 0 means the requested profile is not realizable with this vacuum
    // field; clamp rather than produce NaN, and let the caller notice via q.
    // The Grad--Shafranov equation determines F^2, so the prescribed boundary
    // value supplies the sign branch throughout the reconstructed profile.
    const Real magnitude = v > Real{0} ? std::sqrt(v) : Real{0};
    f[static_cast<std::size_t>(k)] = std::copysign(magnitude, f_vacuum);
  }
  return f;
}

struct FluxSurface {
  Real psi_n{0};
  std::vector<Real> r{};
  std::vector<Real> z{};
  Real q{0};
  Real volume{0};
  Real area{0};
  bool closed{false};
};

// Trace a closed contour of constant psi by marching outward from the axis
// along rays. For a well-behaved equilibrium each ray crosses the target
// surface exactly once, which makes this both simple and robust; a ray that
// finds no crossing marks the surface as open.
inline FluxSurface trace_surface(const EllipticGrid& g, const ScalarField& psi,
                                 const CriticalPointSet& cps, Real psi_n,
                                 int n_theta = 128) {
  FluxSurface s;
  s.psi_n = psi_n;
  s.closed = true;
  const Real target = cps.psi_axis + psi_n * (cps.psi_boundary - cps.psi_axis);

  const Real max_reach = std::max(g.r_max - g.r_min, g.z_max - g.z_min);
  const int n_steps = 4000;
  const Real ds = max_reach / static_cast<Real>(n_steps);
  const Real two_pi = Real{2} * Real{3.14159265358979323846};

  for (int t = 0; t < n_theta; ++t) {
    const Real th = two_pi * static_cast<Real>(t) / static_cast<Real>(n_theta);
    const Real ur = std::cos(th);
    const Real uz = std::sin(th);

    Real prev_val = cps.psi_axis;
    Real prev_len = Real{0};
    bool hit = false;

    for (int k = 1; k <= n_steps; ++k) {
      const Real len = static_cast<Real>(k) * ds;
      const Real rr = cps.axis.r + ur * len;
      const Real zz = cps.axis.z + uz * len;
      if (rr <= g.r_min || rr >= g.r_max || zz <= g.z_min || zz >= g.z_max) break;

      const Real val = sample_bilinear(g, psi, rr, zz);
      // Crossing detected when the target lies between successive samples.
      if ((prev_val - target) * (val - target) <= Real{0} && k > 1) {
        const Real denom = val - prev_val;
        const Real frac = denom != Real{0}
            ? (target - prev_val) / denom : Real{0};
        const Real hit_len = prev_len + frac * ds;
        s.r.push_back(cps.axis.r + ur * hit_len);
        s.z.push_back(cps.axis.z + uz * hit_len);
        hit = true;
        break;
      }
      prev_val = val;
      prev_len = len;
    }
    if (!hit) s.closed = false;
  }
  return s;
}

// Cross-sectional area and toroidal volume of a traced surface, by the shoelace
// formula and Pappus's theorem respectively.
inline void compute_surface_geometry(FluxSurface& s) {
  const std::size_t n = s.r.size();
  if (!s.closed || n < 3) {
    s.area = Real{0};
    s.volume = Real{0};
    return;
  }
  Real area2 = Real{0};
  Real cx = Real{0};
  for (std::size_t k = 0; k < n; ++k) {
    const std::size_t m = (k + 1) % n;
    const Real cross = s.r[k] * s.z[m] - s.r[m] * s.z[k];
    area2 += cross;
    cx += (s.r[k] + s.r[m]) * cross;
  }
  s.area = std::abs(area2) * Real{0.5};
  const Real centroid_r = area2 != Real{0} ? cx / (Real{3} * area2) : Real{0};
  // Pappus: V = 2 pi * R_centroid * A.
  s.volume = Real{2} * Real{3.14159265358979323846}
           * std::abs(centroid_r) * s.area;
}

// Safety factor on a traced surface:
//   q = (1/2pi) * contour integral B_phi / (r B_pol) dl
inline Real compute_q(const EllipticGrid& g, const MagneticField& b,
                      const FluxSurface& s) {
  const std::size_t n = s.r.size();
  if (!s.closed || n < 3) return Real{0};
  Real acc = Real{0};
  for (std::size_t k = 0; k < n; ++k) {
    const std::size_t m = (k + 1) % n;
    const Real dr = s.r[m] - s.r[k];
    const Real dz = s.z[m] - s.z[k];
    const Real dl = std::sqrt(dr * dr + dz * dz);
    const Real rm = Real{0.5} * (s.r[k] + s.r[m]);
    const Real zm = Real{0.5} * (s.z[k] + s.z[m]);
    const Real bp = sample_bilinear(g, b.b_poloidal, rm, zm);
    const Real bt = sample_bilinear(g, b.b_phi, rm, zm);
    if (bp <= Real{0}) continue;
    acc += bt / (rm * bp) * dl;
  }
  return acc / (Real{2} * Real{3.14159265358979323846});
}

// Elongation and triangularity of a traced surface -- the two shaping
// parameters every tokamak design conversation starts from.
struct SurfaceShape {
  Real r_major{0};
  Real r_minor{0};
  Real elongation{0};
  Real triangularity{0};
};

inline SurfaceShape compute_shape(const FluxSurface& s) {
  SurfaceShape sh;
  if (!s.closed || s.r.size() < 3) return sh;
  Real r_lo = s.r[0], r_hi = s.r[0], z_lo = s.z[0], z_hi = s.z[0];
  Real r_at_zmax = s.r[0], r_at_zmin = s.r[0];
  for (std::size_t k = 0; k < s.r.size(); ++k) {
    r_lo = std::min(r_lo, s.r[k]);
    r_hi = std::max(r_hi, s.r[k]);
    if (s.z[k] > z_hi) { z_hi = s.z[k]; r_at_zmax = s.r[k]; }
    if (s.z[k] < z_lo) { z_lo = s.z[k]; r_at_zmin = s.r[k]; }
  }
  sh.r_major = Real{0.5} * (r_hi + r_lo);
  sh.r_minor = Real{0.5} * (r_hi - r_lo);
  if (sh.r_minor <= Real{0}) return sh;
  sh.elongation = Real{0.5} * (z_hi - z_lo) / sh.r_minor;
  const Real upper = (sh.r_major - r_at_zmax) / sh.r_minor;
  const Real lower = (sh.r_major - r_at_zmin) / sh.r_minor;
  sh.triangularity = Real{0.5} * (upper + lower);
  return sh;
}

// Full derived-quantity bundle: what the four downstream consumers receive.
struct EquilibriumDiagnostics {
  MagneticField field{};
  std::vector<FluxSurface> surfaces{};
  std::vector<Real> q_profile{};
  std::vector<Real> psi_n_grid{};
  SurfaceShape boundary_shape{};
  Real q_axis{0};
  Real q_95{0};      // q at psi_N = 0.95, the standard operational metric
  Real total_volume{0};
  Real psi_n_boundary{0};  // outermost closed surface represented in aggregates
  int n_open_surfaces{0};
};

template <class FOfPsiN>
EquilibriumDiagnostics compute_diagnostics(const EllipticGrid& g,
                                           const ScalarField& psi,
                                           const CriticalPointSet& cps,
                                           FOfPsiN&& f_of_psi_n,
                                           int n_surfaces = 32) {
  EquilibriumDiagnostics diag;
  diag.field = compute_field(g, psi, cps, f_of_psi_n);
  if (!cps.axis.valid) return diag;

  std::size_t last_closed = 0;
  bool have_closed = false;
  for (int k = 1; k <= n_surfaces; ++k) {
    const Real pn = static_cast<Real>(k) / static_cast<Real>(n_surfaces + 1);
    FluxSurface s = trace_surface(g, psi, cps, pn);
    compute_surface_geometry(s);
    s.q = compute_q(g, diag.field, s);
    diag.surfaces.push_back(std::move(s));
    if (!diag.surfaces.back().closed) {
      ++diag.n_open_surfaces;
      continue;
    }
    diag.psi_n_grid.push_back(pn);
    diag.q_profile.push_back(diag.surfaces.back().q);
    last_closed = diag.surfaces.size() - 1;
    have_closed = true;
  }

  if (have_closed) {
    diag.q_axis = diag.q_profile.front();
    // q_95: nearest traced surface to psi_N = 0.95.
    std::size_t best = 0;
    Real best_gap = Real{1e30};
    for (std::size_t k = 0; k < diag.psi_n_grid.size(); ++k) {
      const Real gap = std::abs(diag.psi_n_grid[k] - Real{0.95});
      if (gap < best_gap) { best_gap = gap; best = k; }
    }
    diag.q_95 = diag.q_profile[best];
    diag.boundary_shape = compute_shape(diag.surfaces[last_closed]);
    diag.total_volume = diag.surfaces[last_closed].volume;
    diag.psi_n_boundary = diag.surfaces[last_closed].psi_n;
  }
  return diag;
}

}  // namespace quasar::equilibrium
