// Device implementations of the four analytic field evaluators.
//
// Each kernel is one thread per observation point: the evaluators are
// embarrassingly parallel, with no reduction across points. What makes them
// non-trivial is range, not parallelism -- see scaled_device.hpp for why every
// intermediate is carried in (mantissa, exponent) form rather than as a plain
// Real.
//
// The host orchestrators in src/physics/analytic_fields/ keep only parameter
// validation and the status-to-exception mapping.

#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/scaled_arithmetic.hpp"
#include "quasar/physics/analytic_fields/kernels.hpp"

#include "backend/hip/analytic_fields/scaled_device.hpp"
#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

namespace {

using ::quasar::Real;
using ::quasar::backend::analytic_fields_detail::grid_for;
using ::quasar::backend::analytic_fields_detail::kBlock;
using ::quasar::backend::analytic_fields_detail::kStatusNonFiniteNode;
using ::quasar::backend::analytic_fields_detail::kStatusNonFinitePoint;
using ::quasar::backend::analytic_fields_detail::kStatusNotSolenoidal;
using ::quasar::backend::analytic_fields_detail::kStatusNotRepresentable;
using ::quasar::backend::analytic_fields_detail::kStatusOutsideGrid;
using ::quasar::backend::analytic_fields_detail::kStatusSingular;
using ::quasar::backend::analytic_fields_detail::publish_status;
using ::quasar::backend::analytic_fields_detail::sv_from;
using ::quasar::backend::analytic_fields_detail::sv_times;
using ::quasar::backend::analytic_fields_detail::sv_to_real;
using ::quasar::numerics::ScaledValue;
using ::quasar::numerics::ScaledProductQuotientAccumulator;
using ::quasar::numerics::append_scaled_difference_product_quotient;
using ::quasar::numerics::append_scaled_product_quotient;
using ::quasar::numerics::append_scaled_value_product_quotient;
using ::quasar::numerics::finish_scaled_product_quotient_sum_to_value;

__device__ inline int point_index(int M) {
  const int m = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  return m < M ? m : -1;
}

// -- Uniform ----------------------------------------------------------------

__global__ void uniform_fill_kernel(Real vx, Real vy, Real vz, int M,
                                    Real* __restrict__ ox,
                                    Real* __restrict__ oy,
                                    Real* __restrict__ oz) {
  const int m = point_index(M);
  if (m < 0) return;
  ox[m] = vx;
  oy[m] = vy;
  oz[m] = vz;
}

// -- Gradient ---------------------------------------------------------------

// B_row(p) = b0_row + sum_axis grad[row][axis] * (p_axis - origin_axis).
//
// The three displacement terms and the offset go into one exact expansion, so
// a large displacement contribution that cancels against b0 does not round the
// remainder away first. The host did this with a hand-rolled
// largest-exponent-pair merge over four `long double` terms; the expansion
// reducer in scaled_arithmetic.hpp is the exact version of the same idea, and
// it handles the overflowing-difference case (p and origin both huge, of
// opposite sign) inside scaled_difference_to_value rather than through a
// separate coordinate-rescaling branch.
__global__ void gradient_B_kernel(QuasarAfGradientParams params,
                                  const Real* __restrict__ px,
                                  const Real* __restrict__ py,
                                  const Real* __restrict__ pz,
                                  int M,
                                  Real* __restrict__ Bx,
                                  Real* __restrict__ By,
                                  Real* __restrict__ Bz,
                                  int* __restrict__ status) {
  const int m = point_index(M);
  if (m < 0) return;

  const Real point[3] = {px[m], py[m], pz[m]};
  int local = 0;
  if (!isfinite(point[0]) || !isfinite(point[1]) || !isfinite(point[2])) {
    local |= kStatusNonFinitePoint;
  }

  Real out[3] = {Real{0}, Real{0}, Real{0}};
  for (int row = 0; row < 3; ++row) {
    // Four signed terms; each may itself split into two expansion components.
    ScaledProductQuotientAccumulator<8> sum;
    append_scaled_product_quotient(sum, params.b0[row], Real{1}, Real{1},
                                   Real{1});
    for (int axis = 0; axis < 3; ++axis) {
      const Real gradient = params.grad[3 * row + axis];
      if (gradient == Real{0}) continue;
      append_scaled_difference_product_quotient(
          sum, point[axis], params.origin[axis], gradient, Real{1}, Real{1},
          Real{1});
    }
    bool overflow = false;
    out[row] = sv_to_real(finish_scaled_product_quotient_sum_to_value(sum),
                          overflow);
    if (overflow) local |= kStatusNotRepresentable;
  }

  Bx[m] = out[0];
  By[m] = out[1];
  Bz[m] = out[2];
  publish_status(status, local);
}

// -- Dipole -----------------------------------------------------------------

// Unit displacement direction plus 1/|r|, both formed without materializing a
// displacement that may overflow. Mirrors the host `radial_direction`.
struct RadialDirection {
  Real n[3];
  Real inv_r;
  bool singular;
  bool invalid;
};

__device__ inline RadialDirection radial_direction(const Real point[3],
                                                   const Real origin[3]) {
  RadialDirection out{{Real{0}, Real{0}, Real{0}}, Real{0}, false, false};
  if (point[0] == origin[0] && point[1] == origin[1]
      && point[2] == origin[2]) {
    out.singular = true;
    return out;
  }

  Real r[3] = {point[0] - origin[0], point[1] - origin[1],
               point[2] - origin[2]};
  Real distance_scale = Real{0};
  if (isfinite(r[0]) && isfinite(r[1]) && isfinite(r[2])) {
    distance_scale = fmax(fabs(r[0]), fmax(fabs(r[1]), fabs(r[2])));
    for (int k = 0; k < 3; ++k) r[k] /= distance_scale;
  } else {
    // Opposite-sign finite coordinates can have an unrepresentable direct
    // difference. Form the displacement in a common coordinate scale instead.
    distance_scale = fmax(
        fmax(fabs(point[0]), fabs(point[1])),
        fmax(fabs(point[2]),
             fmax(fabs(origin[0]), fmax(fabs(origin[1]), fabs(origin[2])))));
    for (int k = 0; k < 3; ++k) {
      r[k] = point[k] / distance_scale - origin[k] / distance_scale;
    }
  }
  // Every component is now in [-1, 1] with the largest exactly +-1, so the sum
  // of squares lies in [1, 3] and the plain square root cannot overflow.
  const Real scaled_length =
      sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
  if (!(scaled_length > Real{0}) || !isfinite(scaled_length)) {
    out.invalid = true;
    return out;
  }
  for (int k = 0; k < 3; ++k) out.n[k] = r[k] / scaled_length;
  out.inv_r = (Real{1} / distance_scale) / scaled_length;
  return out;
}

// Normalized moment: direction plus the magnitude carried separately, so an
// extreme |m| never enters the angular arithmetic.
struct NormalizedMoment {
  Real m[3];
  Real scale;
};

__device__ inline NormalizedMoment normalized_moment(const Real moment[3]) {
  const Real scale = fmax(fabs(moment[0]),
                          fmax(fabs(moment[1]), fabs(moment[2])));
  return NormalizedMoment{{moment[0] / scale, moment[1] / scale,
                           moment[2] / scale},
                          scale};
}

// Collapse `angular * moment_scale * mu0_over_4pi * inv_r^powers` to a Real,
// keeping the whole chain in the scaled frame. The angular factor arrives as an
// already-reduced exact expansion, so this is the only rounding left.
__device__ inline Real dipole_scale(ScaledValue angular, Real moment_scale,
                                    Real mu0_over_4pi, Real inv_r, int powers,
                                    bool& overflow) {
  if (angular.mantissa == Real{0}) return Real{0};
  bool invalid = false;
  ScaledValue value = sv_times(angular, moment_scale, invalid);
  value = sv_times(value, mu0_over_4pi, invalid);
  for (int k = 0; k < powers; ++k) value = sv_times(value, inv_r, invalid);
  if (invalid) {
    overflow = true;
    return Real{0};
  }
  return sv_to_real(value, overflow);
}

__global__ void dipole_B_kernel(QuasarAfDipoleParams params,
                                const Real* __restrict__ px,
                                const Real* __restrict__ py,
                                const Real* __restrict__ pz,
                                int M,
                                Real* __restrict__ Bx,
                                Real* __restrict__ By,
                                Real* __restrict__ Bz,
                                int* __restrict__ status) {
  const int m = point_index(M);
  if (m < 0) return;

  const Real point[3] = {px[m], py[m], pz[m]};
  const RadialDirection radial = radial_direction(point, params.origin);
  int local = 0;
  if (radial.singular) local |= kStatusSingular;
  if (radial.invalid)  local |= kStatusNotRepresentable;
  if (local != 0) {
    Bx[m] = Real{0};
    By[m] = Real{0};
    Bz[m] = Real{0};
    publish_status(status, local);
    return;
  }

  const NormalizedMoment moment = normalized_moment(params.moment);
  // q = m . n, both unit-ish, so a plain exact expansion suffices.
  ScaledProductQuotientAccumulator<6> q_sum;
  for (int k = 0; k < 3; ++k) {
    append_scaled_product_quotient(q_sum, moment.m[k], radial.n[k], Real{1},
                                   Real{1});
  }
  bool q_overflow = false;
  const Real q =
      sv_to_real(finish_scaled_product_quotient_sum_to_value(q_sum),
                 q_overflow);

  Real out[3];
  for (int i = 0; i < 3; ++i) {
    // 3*q*n_i - m_i, exact: this is where the far-field cancellation lives.
    ScaledProductQuotientAccumulator<6> angular;
    append_scaled_product_quotient(angular, Real{3} * q, radial.n[i], Real{1},
                                   Real{1});
    append_scaled_product_quotient(angular, -moment.m[i], Real{1}, Real{1},
                                   Real{1});
    bool overflow = false;
    out[i] = dipole_scale(finish_scaled_product_quotient_sum_to_value(angular),
                          moment.scale, params.mu0_over_4pi, radial.inv_r, 3,
                          overflow);
    if (overflow) local |= kStatusNotRepresentable;
  }

  Bx[m] = out[0];
  By[m] = out[1];
  Bz[m] = out[2];
  publish_status(status, local);
}

__global__ void dipole_gradB_kernel(QuasarAfDipoleParams params,
                                    const Real* __restrict__ px,
                                    const Real* __restrict__ py,
                                    const Real* __restrict__ pz,
                                    int M,
                                    Real* __restrict__ G,
                                    int* __restrict__ status) {
  const int m = point_index(M);
  if (m < 0) return;

  const Real point[3] = {px[m], py[m], pz[m]};
  const RadialDirection radial = radial_direction(point, params.origin);
  int local = 0;
  if (radial.singular) local |= kStatusSingular;
  if (radial.invalid)  local |= kStatusNotRepresentable;
  if (local != 0) {
    for (int e = 0; e < 9; ++e) G[static_cast<long long>(e) * M + m] = Real{0};
    publish_status(status, local);
    return;
  }

  const NormalizedMoment moment = normalized_moment(params.moment);
  ScaledProductQuotientAccumulator<6> q_sum;
  for (int k = 0; k < 3; ++k) {
    append_scaled_product_quotient(q_sum, moment.m[k], radial.n[k], Real{1},
                                   Real{1});
  }
  bool q_overflow = false;
  const Real q =
      sv_to_real(finish_scaled_product_quotient_sum_to_value(q_sum),
                 q_overflow);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // m_j n_i + m_i n_j + delta_ij q - 5 q n_i n_j
      ScaledProductQuotientAccumulator<8> angular;
      append_scaled_product_quotient(angular, moment.m[j], radial.n[i],
                                     Real{1}, Real{1});
      append_scaled_product_quotient(angular, moment.m[i], radial.n[j],
                                     Real{1}, Real{1});
      if (i == j) {
        append_scaled_product_quotient(angular, q, Real{1}, Real{1}, Real{1});
      }
      append_scaled_product_quotient(angular, Real{-5} * q,
                                     radial.n[i] * radial.n[j], Real{1},
                                     Real{1});
      bool overflow = false;
      // The leading 3 joins the scale chain rather than the angular sum so the
      // cancellation above is not first multiplied out.
      const Real value = dipole_scale(
          sv_times(finish_scaled_product_quotient_sum_to_value(angular),
                   Real{3}, overflow),
          moment.scale, params.mu0_over_4pi, radial.inv_r, 4, overflow);
      if (overflow) local |= kStatusNotRepresentable;
      G[static_cast<long long>(3 * i + j) * M + m] = value;
    }
  }
  publish_status(status, local);
}

// -- File grid --------------------------------------------------------------

// Interpolation stencil along one axis: the base node, how many nodes
// participate, and either the linear weights or their derivatives.
struct AxisWeights {
  int base;
  int count;
  ScaledValue factor[2];
  bool outside;
  bool nonfinite;
};

// +-1/spacing in scaled form, so a subnormal or enormous spacing does not turn
// the reciprocal into zero or infinity before it is used.
__device__ inline ScaledValue reciprocal_factor(Real spacing, Real sign,
                                                bool& invalid) {
  int exponent = 0;
  const Real mantissa = frexp(spacing, &exponent);
  if (mantissa == Real{0} || !isfinite(mantissa)) {
    invalid = true;
    return ScaledValue{};
  }
  int shift = 0;
  const Real value = frexp(sign / mantissa, &shift);
  return ScaledValue{value, -exponent + shift};
}

__device__ inline AxisWeights weights_for_axis(Real coordinate, Real origin,
                                               Real spacing, int n,
                                               bool derivative) {
  AxisWeights out;
  out.base = 0;
  out.count = 1;
  out.factor[0] = ScaledValue{Real{1}, 0};
  out.factor[1] = ScaledValue{};
  out.outside = false;
  out.nonfinite = false;

  if (!isfinite(coordinate)) {
    out.nonfinite = true;
    return out;
  }
  if (n == 1) {
    // A singleton axis denotes one geometric plane, not an invariant direction.
    // Accepting nearby-but-distinct values silently projects observations onto
    // the wrong plane, especially at large origins.
    if (coordinate != origin) out.outside = true;
    // A derivative normal to a singleton axis is not defined; the host refused
    // the whole gradient in that configuration, and the orchestrator still
    // does, so this branch is unreachable with derivative == true.
    return out;
  }

  // u = (coordinate - origin) / spacing, formed in the scaled frame so a large
  // pair of nearby coordinates keeps its small exact difference.
  ScaledProductQuotientAccumulator<4> u_sum;
  append_scaled_difference_product_quotient(u_sum, coordinate, origin, Real{1},
                                            Real{1}, spacing, Real{1});
  const ScaledValue scaled_u = finish_scaled_product_quotient_sum_to_value(u_sum);
  const Real u = scalbn(scaled_u.mantissa, scaled_u.exponent);

  const Real last = static_cast<Real>(n - 1);
  const Real tolerance =
      Real{64} * Real{2.220446049250313e-16} * fmax(Real{1}, last);
  if (!isfinite(u) || u < -tolerance || u > last + tolerance) {
    out.outside = true;
    return out;
  }
  const Real clamped = fmin(fmax(u, Real{0}), last);
  const int base = clamped >= last ? n - 2 : static_cast<int>(floor(clamped));
  const Real fraction = clamped - static_cast<Real>(base);

  out.base = base;
  out.count = 2;
  if (derivative) {
    bool invalid = false;
    out.factor[0] = reciprocal_factor(spacing, Real{-1}, invalid);
    out.factor[1] = reciprocal_factor(spacing, Real{1}, invalid);
    if (invalid) out.nonfinite = true;
  } else {
    bool invalid = false;
    out.factor[0] = sv_from(Real{1} - fraction, invalid);
    out.factor[1] = sv_from(fraction, invalid);
  }
  return out;
}

__global__ void file_grid_sample_kernel(QuasarAfFileGridParams params,
                                        const Real* __restrict__ vx,
                                        const Real* __restrict__ vy,
                                        const Real* __restrict__ vz,
                                        const Real* __restrict__ px,
                                        const Real* __restrict__ py,
                                        const Real* __restrict__ pz,
                                        int M,
                                        Real* __restrict__ ox,
                                        Real* __restrict__ oy,
                                        Real* __restrict__ oz,
                                        int* __restrict__ status) {
  const int m = point_index(M);
  if (m < 0) return;

  const Real point[3] = {px[m], py[m], pz[m]};
  AxisWeights axis[3];
  int local = 0;
  for (int k = 0; k < 3; ++k) {
    axis[k] = weights_for_axis(point[k], params.origin[k], params.spacing[k],
                               params.dims[k], params.derivative_axis[k] != 0);
    if (axis[k].nonfinite) local |= kStatusNonFinitePoint;
    if (axis[k].outside)   local |= kStatusOutsideGrid;
  }
  if (local != 0) {
    ox[m] = Real{0};
    oy[m] = Real{0};
    oz[m] = Real{0};
    publish_status(status, local);
    return;
  }

  // Up to eight corner contributions per component, each an exact product of a
  // stencil weight and a node value. Accumulating them as one expansion is what
  // keeps a trilinear sample of two large opposing nodes from losing its small
  // true value.
  ScaledProductQuotientAccumulator<16> sum[3];
  for (int ck = 0; ck < axis[2].count; ++ck) {
    for (int cj = 0; cj < axis[1].count; ++cj) {
      for (int ci = 0; ci < axis[0].count; ++ci) {
        bool invalid = false;
        ScaledValue weight =
            sv_times(axis[0].factor[ci], axis[1].factor[cj], invalid);
        weight = sv_times(weight, axis[2].factor[ck], invalid);
        if (invalid) local |= kStatusNotRepresentable;
        if (weight.mantissa == Real{0}) continue;

        const long long i = axis[0].base + ci;
        const long long j = axis[1].base + cj;
        const long long k = axis[2].base + ck;
        const long long node =
            i + static_cast<long long>(params.dims[0])
                    * (j + static_cast<long long>(params.dims[1]) * k);
        const Real value[3] = {vx[node], vy[node], vz[node]};
        for (int c = 0; c < 3; ++c) {
          append_scaled_value_product_quotient(sum[c], weight, value[c],
                                               Real{1}, Real{1}, Real{1});
        }
      }
    }
  }

  Real out[3];
  for (int c = 0; c < 3; ++c) {
    bool overflow = false;
    out[c] = sv_to_real(finish_scaled_product_quotient_sum_to_value(sum[c]),
                        overflow);
    if (overflow) local |= kStatusNotRepresentable;
  }
  ox[m] = out[0];
  oy[m] = out[1];
  oz[m] = out[2];
  publish_status(status, local);
}

// One-time map admissibility sweep. One thread per cell; each checks the
// vertices of its own cell, so every interior vertex is examined by more than
// one thread. That redundancy is deliberate -- it keeps the indexing trivial and
// the verdict is an idempotent OR, so a duplicated check costs nothing but
// arithmetic.
//
// Componentwise trilinear interpolation does not automatically preserve
// div(B)=0. Its divergence is multi-affine inside each cell, so its extrema
// occur at cell vertices; checking those vertices bounds the whole cell.
__global__ void file_grid_validate_kernel(QuasarAfFileGridParams params,
                                          const Real* __restrict__ vx,
                                          const Real* __restrict__ vy,
                                          const Real* __restrict__ vz,
                                          Real relative_tolerance,
                                          long long cells,
                                          long long cells_x, long long cells_y,
                                          int* __restrict__ status) {
  const long long c = static_cast<long long>(blockIdx.x) * blockDim.x
                    + threadIdx.x;
  if (c >= cells) return;

  const long long i = c % cells_x;
  const long long j = (c / cells_x) % cells_y;
  const long long k = c / (cells_x * cells_y);

  const long long nx = params.dims[0];
  const long long ny = params.dims[1];
  const auto node = [&](long long a, long long b, long long d) {
    return a + nx * (b + ny * d);
  };

  int local = 0;
  const int vertices_x = params.dims[0] > 1 ? 2 : 1;
  const int vertices_y = params.dims[1] > 1 ? 2 : 1;
  const int vertices_z = params.dims[2] > 1 ? 2 : 1;

  // Node finiteness. Each thread covers the eight (or fewer) corners of its
  // cell, which collectively covers every node of the map.
  for (int vk = 0; vk < vertices_z; ++vk) {
    for (int vj = 0; vj < vertices_y; ++vj) {
      for (int vi = 0; vi < vertices_x; ++vi) {
        const long long n = node(i + vi, j + vj, k + vk);
        if (!isfinite(vx[n]) || !isfinite(vy[n]) || !isfinite(vz[n])) {
          local |= kStatusNonFiniteNode;
        }
      }
    }
  }
  if ((local & kStatusNonFiniteNode) != 0) {
    // A non-finite node makes the divergence question meaningless; report the
    // more fundamental failure alone.
    publish_status(status, local);
    return;
  }

  // A singleton axis is one sampled geometric plane. Its missing normal
  // derivative is undetermined, so div(B) cannot be inferred by silently
  // setting that derivative to zero; such maps stay valid B-only samplers.
  if (params.dims[0] == 1 || params.dims[1] == 1 || params.dims[2] == 1) {
    publish_status(status, local);
    return;
  }

  for (int vk = 0; vk < 2; ++vk) {
    for (int vj = 0; vj < 2; ++vj) {
      for (int vi = 0; vi < 2; ++vi) {
        ScaledValue derivative[3];
        {
          ScaledProductQuotientAccumulator<4> d;
          append_scaled_difference_product_quotient(
              d, vx[node(i + 1, j + vj, k + vk)],
              vx[node(i, j + vj, k + vk)], Real{1}, Real{1},
              params.spacing[0], Real{1});
          derivative[0] = finish_scaled_product_quotient_sum_to_value(d);
        }
        {
          ScaledProductQuotientAccumulator<4> d;
          append_scaled_difference_product_quotient(
              d, vy[node(i + vi, j + 1, k + vk)],
              vy[node(i + vi, j, k + vk)], Real{1}, Real{1},
              params.spacing[1], Real{1});
          derivative[1] = finish_scaled_product_quotient_sum_to_value(d);
        }
        {
          ScaledProductQuotientAccumulator<4> d;
          append_scaled_difference_product_quotient(
              d, vz[node(i + vi, j + vj, k + 1)],
              vz[node(i + vi, j + vj, k)], Real{1}, Real{1},
              params.spacing[2], Real{1});
          derivative[2] = finish_scaled_product_quotient_sum_to_value(d);
        }

        const ScaledValue divergence =
            ::quasar::backend::analytic_fields_detail::sv_sum(derivative, 3,
                                                              false);
        const ScaledValue scale =
            ::quasar::backend::analytic_fields_detail::sv_sum(derivative, 3,
                                                              true);
        bool invalid = false;
        const ScaledValue threshold =
            sv_times(scale, relative_tolerance, invalid);
        if (::quasar::backend::analytic_fields_detail::sv_magnitude_greater(
                divergence, threshold)) {
          local |= kStatusNotSolenoidal;
        }
      }
    }
  }
  publish_status(status, local);
}

}  // namespace

extern "C" void launch_analytic_uniform_fill(
    Real vx, Real vy, Real vz, int M, Real* ox, Real* oy, Real* oz,
    ::quasar::backend::stream_t stream) {
  if (M <= 0) return;
  uniform_fill_kernel<<<grid_for(M), dim3(kBlock), 0,
                        static_cast<hipStream_t>(stream)>>>(vx, vy, vz, M, ox,
                                                            oy, oz);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_analytic_gradient_B(
    QuasarAfGradientParams params, const Real* px, const Real* py,
    const Real* pz, int M, Real* Bx, Real* By, Real* Bz, int* status,
    ::quasar::backend::stream_t stream) {
  if (M <= 0) return;
  gradient_B_kernel<<<grid_for(M), dim3(kBlock), 0,
                      static_cast<hipStream_t>(stream)>>>(
      params, px, py, pz, M, Bx, By, Bz, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_analytic_dipole_B(
    QuasarAfDipoleParams params, const Real* px, const Real* py,
    const Real* pz, int M, Real* Bx, Real* By, Real* Bz, int* status,
    ::quasar::backend::stream_t stream) {
  if (M <= 0) return;
  dipole_B_kernel<<<grid_for(M), dim3(kBlock), 0,
                    static_cast<hipStream_t>(stream)>>>(
      params, px, py, pz, M, Bx, By, Bz, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_analytic_dipole_gradB(
    QuasarAfDipoleParams params, const Real* px, const Real* py,
    const Real* pz, int M, Real* G, int* status,
    ::quasar::backend::stream_t stream) {
  if (M <= 0) return;
  dipole_gradB_kernel<<<grid_for(M), dim3(kBlock), 0,
                        static_cast<hipStream_t>(stream)>>>(
      params, px, py, pz, M, G, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_analytic_file_grid_sample(
    QuasarAfFileGridParams params, const Real* vx, const Real* vy,
    const Real* vz, const Real* px, const Real* py, const Real* pz, int M,
    Real* ox, Real* oy, Real* oz, int* status,
    ::quasar::backend::stream_t stream) {
  if (M <= 0) return;
  file_grid_sample_kernel<<<grid_for(M), dim3(kBlock), 0,
                            static_cast<hipStream_t>(stream)>>>(
      params, vx, vy, vz, px, py, pz, M, ox, oy, oz, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_analytic_file_grid_validate(
    QuasarAfFileGridParams params, const Real* vx, const Real* vy,
    const Real* vz, Real relative_tolerance, int* status,
    ::quasar::backend::stream_t stream) {
  const long long cells_x = params.dims[0] > 1 ? params.dims[0] - 1 : 1;
  const long long cells_y = params.dims[1] > 1 ? params.dims[1] - 1 : 1;
  const long long cells_z = params.dims[2] > 1 ? params.dims[2] - 1 : 1;
  const long long cells = cells_x * cells_y * cells_z;
  if (cells <= 0) return;
  const unsigned blocks =
      static_cast<unsigned>((cells + kBlock - 1) / kBlock);
  file_grid_validate_kernel<<<dim3(blocks), dim3(kBlock), 0,
                              static_cast<hipStream_t>(stream)>>>(
      params, vx, vy, vz, relative_tolerance, cells, cells_x, cells_y, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}
