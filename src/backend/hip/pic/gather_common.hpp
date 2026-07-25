// Device-inline gather/push helpers shared by the Cartesian and cylindrical
// particle gather+push kernels (gather_push_hip.hip and gather_push_cyl_hip.hip).
// These were verbatim-identical in both translation units; extracting them here
// keeps the field-gather indexing and the Boris half-rotation in one place so the
// two kernels cannot drift. The cylindrical-specific position-advance coordinate
// rotation stays in gather_push_cyl_hip.hip.
//
// Header-only `__device__ inline` (no ODR/link issue). Included only by the .hip
// definitions under src/backend/hip/pic/ (it pulls in <hip/hip_runtime.h>); never
// include from a non-HIP TU.
#pragma once

#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"

#include <hip/hip_runtime.h>

#include <climits>
#include <cstddef>

namespace quasar::backend::pic {

__device__ inline void mark_particle_error(unsigned int* error) {
  // A sticky boolean cannot wrap back to zero, unlike atomicAdd on a long run.
  atomicExch(error, 1u);
}

__device__ inline bool finite_vec(quasar::Vec3 v) {
  return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}

__device__ inline bool valid_nonrelativistic_velocity(quasar::Vec3 v) {
  return finite_vec(v) && quasar::length(v) < 1.0;
}

// Shape construction floors a reduced coordinate and casts it to int. Keep a
// margin for the TSC support and the deposition window before that conversion.
__device__ inline bool stencil_coordinate_representable(
    double coordinate, double origin, double spacing) {
  if (!isfinite(coordinate)) return false;
  const double reduced = quasar::detail::scaled_difference_quotient(
      coordinate, origin, spacing);
  constexpr double margin = 16.0;
  return isfinite(reduced)
      && reduced >= static_cast<double>(INT_MIN) + margin
      && reduced <= static_cast<double>(INT_MAX) - margin;
}

__device__ inline double scaled_product4(double a, double b, double c,
                                         double d) {
  if (a == 0.0 || b == 0.0 || c == 0.0 || d == 0.0) return 0.0;
  int ea = 0, eb = 0, ec = 0, ed = 0;
  const double ma = frexp(a, &ea);
  const double mb = frexp(b, &eb);
  const double mc = frexp(c, &ec);
  const double md = frexp(d, &ed);
  double mantissa = ma * mb * mc * md;
  int adjustment = 0;
  mantissa = frexp(mantissa, &adjustment);
  return scalbn(mantissa, ea + eb + ec + ed + adjustment);
}

// Evaluate a*x+b*y with one product fused into the addition. The leapfrog
// coefficients are finite, non-negative, and no larger than one, so each
// standalone product is bounded by its field sample. Fusing the final centering
// operation preserves one more rounding's worth of cancellation between large
// opposite-signed half-step fields.
__device__ inline double weighted_sum2(double a, double x,
                                       double b, double y) {
  return fma(a, x, b * y);
}

// Per-axis gather indexing. A periodic axis wraps. A non-periodic axis reads the
// actual padded lattice so the evolved field's boundary parity/continuation and
// the prescribed field's analytic continuation participate in interpolation.
// Clamp only to the allocation itself as a last-resort memory-safety guard for
// a direct low-level pusher caller; EmPic2D3V validates enough halo for the
// selected shape, so a solver gather never reaches that guard.
__device__ inline int clamp_axis(int i, int lo, int hi) {
  return i < lo ? lo : (i > hi ? hi : i);
}

__device__ inline std::size_t gather_index(const quasar::Grid2D& g, int i, int j,
                                           bool periodic_x, bool periodic_y,
                                           int x_hi, int y_hi) {
  (void)x_hi;
  (void)y_hi;
  const int ii = periodic_x
      ? g.wrap_i(i)
      : clamp_axis(i, -g.nghost, g.nx + g.nghost - 1);
  const int jj = periodic_y
      ? g.wrap_j(j)
      : clamp_axis(j, -g.nghost, g.ny + g.nghost - 1);
  return g.index(ii, jj);
}

// Gather one component from its own Yee sub-lattice. Self and external fields
// share the same physical component location, so their weights and indices are
// identical and are accumulated together.
template <class Weights>
__device__ inline void gather_component_pair(
    const quasar::Grid2D& g, const Weights& w,
    const double* __restrict__ self, const double* __restrict__ ext,
    bool periodic_x, bool periodic_y, int x_hi, int y_hi,
    double& out_self, double& out_ext) {
  for (int jj = 0; jj < w.ny; ++jj) {
    for (int ii = 0; ii < w.nx; ++ii) {
      const double ww = w.wx[ii] * w.wy[jj];
      const std::size_t k = gather_index(g, w.ix[ii], w.iy[jj], periodic_x,
                                         periodic_y, x_hi, y_hi);
      out_self += ww * self[k];
      out_ext += ww * ext[k];
    }
  }
}

template <class Weights>
__device__ inline double gather_component(const quasar::Grid2D& g, const Weights& w,
                                          const double* __restrict__ values,
                                          bool periodic_x, bool periodic_y,
                                          int x_hi, int y_hi) {
  double out = 0.0;
  for (int jj = 0; jj < w.ny; ++jj) {
    for (int ii = 0; ii < w.nx; ++ii) {
      const std::size_t k = gather_index(g, w.ix[ii], w.iy[jj], periodic_x,
                                         periodic_y, x_hi, y_hi);
      out += w.wx[ii] * w.wy[jj] * values[k];
    }
  }
  return out;
}

// Gathers the self field and the external field of one (vector) quantity in a
// single sweep over the shape stencil, reusing the precomputed weights `w` and
// computing each node's wrapped/clamped index exactly once. `out_self` and
// `out_ext` accumulate the two contributions.
template <class Weights>
__device__ inline void gather_pair(const quasar::Grid2D& g, const Weights& w,
                                   const double* __restrict__ sx, const double* __restrict__ sy,
                                   const double* __restrict__ sz,
                                   const double* __restrict__ ex, const double* __restrict__ ey,
                                   const double* __restrict__ ez,
                                   bool periodic_x, bool periodic_y,
                                   int x_hi, int y_hi,
                                   quasar::Vec3& out_self, quasar::Vec3& out_ext) {
  for (int jj = 0; jj < w.ny; ++jj) {
    for (int ii = 0; ii < w.nx; ++ii) {
      const double ww = w.wx[ii] * w.wy[jj];
      const std::size_t k = gather_index(g, w.ix[ii], w.iy[jj], periodic_x,
                                         periodic_y, x_hi, y_hi);
      out_self.x += ww * sx[k];
      out_self.y += ww * sy[k];
      out_self.z += ww * sz[k];
      out_ext.x += ww * ex[k];
      out_ext.y += ww * ey[k];
      out_ext.z += ww * ez[k];
    }
  }
}

__device__ inline quasar::Vec3 boris(double qm, quasar::Vec3 v, quasar::Vec3 e,
                                     quasar::Vec3 b, double dt) {
  // Form q*dt*field as one exponent-scaled product. Materialising q*dt/2 first
  // can overflow (or underflow) even when the compensated final product is
  // finite and representable.
  const quasar::Vec3 kick{
      scaled_product4(qm, dt, e.x, 0.5),
      scaled_product4(qm, dt, e.y, 0.5),
      scaled_product4(qm, dt, e.z, 0.5)};
  const quasar::Vec3 v_minus = v + kick;

  // Express the Cayley rotation through a unit axis and tan(theta/2)=tau.
  // The textbook t/s form squares t and forms v x t, both of which overflow
  // for a finite, very large magnetic field even though the physical rotation
  // has the bounded theta->pi limit. Scaling B first and evaluating the tau>1
  // branch in terms of 1/tau keeps that limit finite and norm preserving.
  const double bmax = fmax(fabs(b.x), fmax(fabs(b.y), fabs(b.z)));
  if (bmax == 0.0 || qm == 0.0 || dt == 0.0) return v_minus + kick;
  const quasar::Vec3 scaled{b.x / bmax, b.y / bmax, b.z / bmax};
  const double scaled_norm = sqrt(quasar::dot(scaled, scaled));
  const quasar::Vec3 axis = (1.0 / scaled_norm) * scaled;
  const double tau = fabs(scaled_product4(qm, dt, bmax,
                                           0.5 * scaled_norm));

  double cos_theta;
  double sin_theta;
  if (tau <= 1.0) {
    const double tau2 = tau * tau;
    const double inv = 1.0 / (1.0 + tau2);
    cos_theta = (1.0 - tau2) * inv;
    sin_theta = (2.0 * tau) * inv;
  } else {
    const double inv_tau = 1.0 / tau;
    const double inv_tau2 = inv_tau * inv_tau;
    const double inv = 1.0 / (1.0 + inv_tau2);
    cos_theta = (inv_tau2 - 1.0) * inv;
    sin_theta = (2.0 * inv_tau) * inv;
  }
  if (signbit(qm) != signbit(dt)) sin_theta = -sin_theta;

  // Rotate a max-component-scaled vector. Dot/cross intermediates then stay O(1)
  // even when a low-level caller supplies a very large but finite vector; the
  // final rescale overflows only when that output component is unrepresentable.
  const double vscale = fmax(fabs(v_minus.x),
                             fmax(fabs(v_minus.y), fabs(v_minus.z)));
  if (vscale == 0.0) return kick;
  const quasar::Vec3 v_scaled = (1.0 / vscale) * v_minus;
  const quasar::Vec3 parallel = quasar::dot(v_scaled, axis) * axis;
  const quasar::Vec3 perpendicular = v_scaled - parallel;
  const quasar::Vec3 rotated = parallel + cos_theta * perpendicular
                              + sin_theta * quasar::cross(v_scaled, axis);
  const quasar::Vec3 v_plus = vscale * rotated;
  return v_plus + kick;
}

}  // namespace quasar::backend::pic
