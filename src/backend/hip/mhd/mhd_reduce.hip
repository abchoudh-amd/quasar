// Device reduction kernels for the ideal-MHD solver: the CFL maximum signal
// speed and the constrained-transport div(B) L-infinity diagnostic.
//
// Both quantities are a MAX over all interior cells. HIP has no native
// atomicMax for double, so each launcher uses a TWO-PASS block reduction: a
// kernel computes the per-cell value and reduces it within each block via a
// shared-memory tree to one partial per block (written to a scratch
// DeviceBuffer), then a single 256-thread block folds those partials to one
// scalar in the slot past them. The final scalar is copied to the caller's host
// pointer before the launcher returns (device_synchronize'd).
//
// The second pass used to run on the host. It was the only floating-point work
// left in this tree that both scaled with the mesh and ran several times per
// accepted step -- admissible_fraction alone is called after every SSP-RK stage
// and twice more per positivity substep. The device_synchronize is NOT what
// moved: dt, theta and the div(B) defect all gate host control flow (a throw, a
// substep accept/retry, a loop bound), so the launcher still blocks. What went
// away is the O(n_blocks) device-to-host copy and the host loop over it.

#include "quasar/backend/memory.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/numerics/mhd_background_metrics.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include "backend/hip/hip_check.hpp"
#include "mhd_face_quadrature.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <limits>
#include <vector>

namespace {

using quasar::Real;
using quasar::numerics::MhdBackground;
using quasar::numerics::MhdState;

// 16x16 = 256 threads per block, matching the other MHD update kernels.
inline constexpr unsigned kBlock2D = 16;
inline constexpr unsigned kThreadsPerBlock = kBlock2D * kBlock2D;
inline constexpr int kDiscreteSolenoidalRoundoffUlpShift = 10;
inline constexpr Real kCflOverflowRetryScale = 0x1p-64;

inline dim3 mhd_grid_2d(const quasar::Grid2D& g, dim3 block) {
  return dim3((static_cast<unsigned>(g.nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(g.ny) + block.y - 1) / block.y);
}

// Shared-memory tree reduction of `sdata` (length kThreadsPerBlock) to its max,
// left in sdata[0]. tid is the linear thread index within the block.
// The scaled-arithmetic metrics below used to be duplicated here verbatim. They
// now come from the shared numerics header, which is also what the background
// validation sweeps use -- one definition, so the divb diagnostic and the
// background check cannot drift apart.
using quasar::numerics::scaled_abs_less_equal_power_of_two;
using quasar::numerics::scaled_annular_radial_divergence;
using quasar::numerics::scaled_annular_radial_roundoff;
using quasar::numerics::scaled_directional_derivative;
using quasar::numerics::scaled_directional_roundoff;
using quasar::numerics::scaled_ulp;

__device__ inline void block_max_reduce(double* sdata, unsigned tid) {
  for (unsigned stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const double other = sdata[tid + stride];
      if (other > sdata[tid]) sdata[tid] = other;
    }
    __syncthreads();
  }
}

__device__ inline void block_min_reduce(double* sdata, unsigned tid) {
  for (unsigned stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const double other = sdata[tid + stride];
      if (other < sdata[tid]) sdata[tid] = other;
    }
    __syncthreads();
  }
}

__device__ inline void block_scaled_abs_max_reduce(
    Real* mantissa, int* exponent, unsigned tid) {
  for (unsigned stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const Real other = mantissa[tid + stride];
      const int other_exponent = exponent[tid + stride];
      const Real current = mantissa[tid];
      const bool take_other = !std::isfinite(other) ||
          (std::isfinite(current) && other != Real{0} &&
           (current == Real{0} || other_exponent > exponent[tid] ||
            (other_exponent == exponent[tid] &&
             fabs(other) > fabs(current))));
      if (take_other) {
        mantissa[tid] = other;
        exponent[tid] = other_exponent;
      }
    }
    __syncthreads();
  }
}

// Internal (gas) energy along a convex segment of conserved states. Face fields
// are first collocated at the cell centre; linear centering commutes with the
// convex interpolation. For rho>0,
//   q(U)=E-|m|^2/(2 rho)-|b|^2/2
// is concave in U, so {q >= q_floor} is convex.
__device__ inline Real convex_internal_energy(
    const MhdState& a, const MhdState& b, Real theta) {
  const Real omt = Real{1} - theta;
  const Real rho = omt * a.rho + theta * b.rho;
  if (!(rho > Real{0}) || !std::isfinite(rho)) {
    return -std::numeric_limits<Real>::infinity();
  }
  const Real mx = omt * a.mx + theta * b.mx;
  const Real my = omt * a.my + theta * b.my;
  const Real mz = omt * a.mz + theta * b.mz;
  const Real e  = omt * a.energy + theta * b.energy;
  const Real bx = omt * a.bx + theta * b.bx;
  const Real by = omt * a.by + theta * b.by;
  const Real bz = omt * a.bz + theta * b.bz;
  const Real kinetic =
      quasar::numerics::kinetic_from_momentum(mx, my, mz, rho);
  const Real magnetic =
      quasar::numerics::half_squared_norm3(bx, by, bz);
  // Match pressure(): subtract sequentially so a representable negative
  // residual is not hidden by overflow in kinetic+magnetic.
  const Real q = (e - kinetic) - magnetic;
  return std::isfinite(q) ? q : -std::numeric_limits<Real>::infinity();
}

// Per-cell admissible convex fraction, reduced to one partial minimum per block.
// Density gives an analytic linear bound. Internal energy is concave, so when
// the candidate violates its bound a bracketed bisection finds the unique edge
// of the admissible interval without modifying either state. Bounds are strict;
// the conservative solver passes zero to enforce rho>0 and q>0.
__global__ void QUASAR_LAUNCH_BOUNDS_256 admissible_fraction_kernel(
    quasar::Grid2D g, Real rho_floor, Real q_floor, int collocation_order,
    quasar::numerics::RadialTablesView radial_tables,
    const double* __restrict__ a_rho, const double* __restrict__ a_mx,
    const double* __restrict__ a_my, const double* __restrict__ a_mz,
    const double* __restrict__ a_energy, const double* __restrict__ a_bx,
    const double* __restrict__ a_by, const double* __restrict__ a_bz,
    const double* __restrict__ b_rho, const double* __restrict__ b_mx,
    const double* __restrict__ b_my, const double* __restrict__ b_mz,
    const double* __restrict__ b_energy, const double* __restrict__ b_bx,
    const double* __restrict__ b_by, const double* __restrict__ b_bz,
    double* __restrict__ block_partials) {
  __shared__ double sdata[kThreadsPerBlock];
  const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  Real local = Real{1};
  if (i < g.nx && j < g.ny) {
    const MhdState a = quasar::mhd::load_cell_state(
        g, a_rho, a_mx, a_my, a_mz, a_energy, a_bx, a_by, a_bz, i, j,
        radial_tables, collocation_order);
    const MhdState b = quasar::mhd::load_cell_state(
        g, b_rho, b_mx, b_my, b_mz, b_energy, b_bx, b_by, b_bz, i, j,
        radial_tables, collocation_order);
    const Real qa = convex_internal_energy(a, b, Real{0});
    if (!(a.rho > rho_floor) || !(qa > q_floor) ||
        !std::isfinite(a.rho) || !std::isfinite(qa)) {
      local = Real{0};
    } else if (!std::isfinite(b.rho) || !std::isfinite(b.mx) ||
               !std::isfinite(b.my) || !std::isfinite(b.mz) ||
               !std::isfinite(b.energy) || !std::isfinite(b.bx) ||
               !std::isfinite(b.by) || !std::isfinite(b.bz)) {
      local = Real{0};
    } else {
      Real hi = Real{1};
      if (b.rho <= rho_floor) {
        const Real denom = a.rho - b.rho;
        hi = (denom > Real{0}) ? (a.rho - rho_floor) / denom : Real{0};
        if (hi < Real{0}) hi = Real{0};
        if (hi > Real{1}) hi = Real{1};
      }
      if (convex_internal_energy(a, b, hi) > q_floor) {
        local = hi;
      } else {
        Real lo = Real{0};
        // 56 iterations resolve a double-precision unit interval below one ulp.
        for (int iter = 0; iter < 56; ++iter) {
          const Real mid = Real{0.5} * (lo + hi);
          if (convex_internal_energy(a, b, mid) > q_floor) {
            lo = mid;
          } else {
            hi = mid;
          }
        }
        local = lo;
      }
    }
  }

  sdata[tid] = local;
  __syncthreads();
  block_min_reduce(sdata, tid);
  if (tid == 0) {
    block_partials[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
  }
}

// a*b*c/(d0*d1), evaluated without materializing any intermediate product or
// quotient at its physical exponent. The CFL path uses this both for physical
// wave speeds and for the final face_weight*alpha/spacing metric scaling.
__device__ inline Real scaled_product3_quotient2(
    Real a, Real b, Real c, Real d0, Real d1) {
  if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) ||
      !std::isfinite(d0) || !std::isfinite(d1) ||
      d0 == Real{0} || d1 == Real{0}) {
    return (((a * b) * c) / d0) / d1;
  }
  if (a == Real{0} || b == Real{0} || c == Real{0}) return Real{0};
  int ea = 0, eb = 0, ec = 0, ed0 = 0, ed1 = 0;
  const Real ma = frexp(a, &ea);
  const Real mb = frexp(b, &eb);
  const Real mc = frexp(c, &ec);
  const Real md0 = frexp(d0, &ed0);
  const Real md1 = frexp(d1, &ed1);
  Real mantissa = (((ma * mb) * mc) / md0) / md1;
  int adjustment = 0;
  mantissa = frexp(mantissa, &adjustment);
  return scalbn(mantissa,
                ea + eb + ec - ed0 - ed1 + adjustment);
}

__device__ inline Real scaled_total_field_speed(
    Real perturbation, Real background, Real sqrt_rho) {
  const Real a[2] = {perturbation, background};
  const Real b[2] = {Real{1}, Real{1}};
  const Real d0[2] = {sqrt_rho, sqrt_rho};
  const Real d1[2] = {Real{1}, Real{1}};
  return quasar::numerics::scaled_product_quotient_sum(a, b, d0, d1, 2);
}

struct FaceSpeedParts {
  Real velocity{};
  Real fast{};
};

// One side's separate physical |v_n| and c_fast contributions. The Riemann
// solver consumes this physical wave fan before the mesh metric is applied, so
// an unrepresentable speed is not made admissible merely because a large cell
// spacing would make its quotient finite. The constituent products and field
// split still use exponent-scaled algebra so every representable speed is kept.
__device__ inline FaceSpeedParts face_state_speed_parts(
    const MhdState& s, const MhdBackground& bg, int dir, Real gamma) {
  const Real p = quasar::numerics::pressure(s, gamma);
  if (!(std::isfinite(s.rho) && s.rho > Real{0} &&
        std::isfinite(s.mx) && std::isfinite(s.my) &&
        std::isfinite(s.mz) && std::isfinite(s.energy) &&
        std::isfinite(s.bx) && std::isfinite(s.by) &&
        std::isfinite(s.bz) && std::isfinite(bg.b0x) &&
        std::isfinite(bg.b0y) && std::isfinite(bg.b0z) &&
        std::isfinite(p) && p > Real{0})) {
    const Real infinity = std::numeric_limits<Real>::infinity();
    return FaceSpeedParts{infinity, infinity};
  }

  const Real sqrt_rho = std::sqrt(s.rho);
  const Real sound_speed = scaled_product3_quotient2(
      std::sqrt(gamma), std::sqrt(p), Real{1}, sqrt_rho, Real{1});
  const Real ax_speed = scaled_total_field_speed(s.bx, bg.b0x, sqrt_rho);
  const Real ay_speed = scaled_total_field_speed(s.by, bg.b0y, sqrt_rho);
  const Real az_speed = scaled_total_field_speed(s.bz, bg.b0z, sqrt_rho);
  if (!(std::isfinite(sound_speed) && std::isfinite(ax_speed) &&
        std::isfinite(ay_speed) && std::isfinite(az_speed))) {
    const Real infinity = std::numeric_limits<Real>::infinity();
    return FaceSpeedParts{infinity, infinity};
  }

  const Real alfven_speed =
      quasar::numerics::scaled_norm3(ax_speed, ay_speed, az_speed);
  const Real alfven_normal_speed = std::abs(dir == 0 ? ax_speed : ay_speed);
  const Real scale = std::max(sound_speed, alfven_speed);
  Real fast_speed = Real{0};
  if (scale > Real{0}) {
    const Real an = sound_speed / scale;
    const Real bn = alfven_speed / scale;
    const Real can = alfven_normal_speed / scale;
    const Real a2n = an * an;
    const Real b2n = bn * bn;
    const Real sum = a2n + b2n;
    Real disc = sum * sum - Real{4} * a2n * can * can;
    if (disc < Real{0}) disc = Real{0};
    fast_speed = scale * std::sqrt(
        Real{0.5} * (sum + std::sqrt(disc)));
  }

  const Real momentum = std::abs(dir == 0 ? s.mx : s.my);
  const Real velocity_a[1] = {momentum};
  const Real velocity_b[1] = {Real{1}};
  const Real velocity_d0[1] = {s.rho};
  const Real velocity_d1[1] = {Real{1}};
  const Real velocity_speed = quasar::numerics::scaled_product_quotient_sum(
      velocity_a, velocity_b, velocity_d0, velocity_d1, 1);
  if (!(std::isfinite(velocity_speed) && std::isfinite(fast_speed))) {
    const Real infinity = std::numeric_limits<Real>::infinity();
    return FaceSpeedParts{infinity, infinity};
  }
  return FaceSpeedParts{velocity_speed, fast_speed};
}

// HLLD uses one common fast speed across the two states:
//   S_L=min(v_L,v_R)-max(c_f,L,c_f,R),
//   S_R=max(v_L,v_R)+max(c_f,L,c_f,R).
// Its absolute outer-fan bound is therefore max(|v_L|,|v_R|)+max(c_f,L,c_f,R),
// not max_side(|v|+c_f). The piecewise-constant positivity retry uses the LF
// flux, whose alpha is the latter expression, so retain that distinct bound.
__device__ inline Real interface_speed(
    const FaceSpeedParts& left, const FaceSpeedParts& right,
    int scheme_order) {
  const Real velocity = std::max(left.velocity, right.velocity);
  const Real fast = std::max(left.fast, right.fast);
  const Real rate = scheme_order <= 1
      ? std::max(left.velocity + left.fast,
                 right.velocity + right.fast)
      : velocity + fast;
  return std::isfinite(rate) ? rate
                             : std::numeric_limits<Real>::infinity();
}

// Apply the finite-volume metric only after the physical Riemann fan has been
// proved representable. ``face_weight`` includes the homogeneous retry scale;
// exponent factoring prevents either its product or the spacing quotient from
// overflowing before the final (possibly still out-of-range) rate is known.
__device__ inline Real scaled_face_rate(
    Real physical_speed, Real face_weight, Real spacing) {
  if (!(std::isfinite(physical_speed) && physical_speed >= Real{0} &&
        std::isfinite(face_weight) && face_weight > Real{0} &&
        std::isfinite(spacing) && spacing > Real{0})) {
    return std::numeric_limits<Real>::infinity();
  }
  return scaled_product3_quotient2(
      physical_speed, face_weight, Real{1}, spacing, Real{1});
}

// Maximum weighted Courant rate on one numerical face, evaluated from the
// exact reconstructed L/R point states consumed by HLLD. MP5/MP7 reconstruct
// face averages and then recover transverse Gauss-point states; bounding only
// the face average is unsafe because that second polynomial can have a larger
// wave speed. The duplicate periodic high face uses the same face-0 data and
// background mapping as the production Riemann kernel.
__device__ inline Real reconstructed_face_rate(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags,
    int b0_active, Real gamma, int scheme_order, int dir,
    int face_i, int face_j, Real face_weight, Real spacing,
    const quasar::mhd::detail::FaceInterfaceView& iface,
    const double* b0x, const double* b0y, const double* b0z,
    quasar::numerics::RadialTablesView radial_tables) {
  if (face_weight == Real{0}) return Real{0};
  if (dir == 0 && face_i == g.nx &&
      flags.side[0] == 0 && flags.side[1] == 0) {
    face_i = 0;
  }
  if (dir == 1 && face_j == g.ny &&
      flags.side[2] == 0 && flags.side[3] == 0) {
    face_j = 0;
  }
  const std::size_t k = g.index(face_i, face_j);
  const MhdState base_left =
      quasar::mhd::detail::load_interface_side(iface, k, true);
  const MhdState base_right =
      quasar::mhd::detail::load_interface_side(iface, k, false);
  const quasar::mhd::detail::FaceBackgroundView background{
      b0_active, b0x, b0y, b0z};
  const MhdBackground base_bg = quasar::mhd::detail::average_background(
      g, dir, face_i, face_j, background, scheme_order <= 1 ? 1 : 0,
      radial_tables);
  const FaceSpeedParts base_left_speed = face_state_speed_parts(
      base_left, base_bg, dir, gamma);
  const FaceSpeedParts base_right_speed = face_state_speed_parts(
      base_right, base_bg, dir, gamma);
  Real maximum = scaled_face_rate(
      interface_speed(base_left_speed, base_right_speed, scheme_order),
      face_weight, spacing);
  if (scheme_order != 5 && scheme_order != 7) return maximum;

  const int count = scheme_order == 5 ? 3 : 4;
  for (int q = 0; q < count; ++q) {
    const MhdState left = quasar::mhd::detail::point_interface_state(
        g, dir, face_i, face_j, scheme_order, q, iface, true, radial_tables);
    const MhdState right = quasar::mhd::detail::point_interface_state(
        g, dir, face_i, face_j, scheme_order, q, iface, false, radial_tables);
    // Production quadrature falls back to the admissible face-average solve if
    // any recovered point is inadmissible, so its base rate is the matching
    // bound rather than an unrelated infinite rejection.
    if (!quasar::mhd::detail::state_is_admissible(left, gamma) ||
        !quasar::mhd::detail::state_is_admissible(right, gamma)) {
      return maximum;
    }
    const MhdBackground point_bg = quasar::mhd::detail::point_background(
        g, dir, face_i, face_j, scheme_order, q, background, radial_tables);
    const FaceSpeedParts left_speed = face_state_speed_parts(
        left, point_bg, dir, gamma);
    const FaceSpeedParts right_speed = face_state_speed_parts(
        right, point_bg, dir, gamma);
    maximum = std::max(
        maximum, scaled_face_rate(
            interface_speed(left_speed, right_speed, scheme_order),
            face_weight, spacing));
  }
  return maximum;
}

// Per-cell finite-volume CFL coefficient from the incident numerical faces.
// The half-sums are the exact self-weights of the local LF updates. Cylindrical
// mode takes the largest radial self-weight among the annular fluid, exact
// angular-momentum, and metric-free B_phi operators.
__global__ void QUASAR_LAUNCH_BOUNDS_256 cfl_max_rate_kernel(
    quasar::Grid2D g, int b0_active, Real gamma, Real dx, Real dy,
    Real rate_scale, int scheme_order, int cylindrical,
    quasar::numerics::RadialTablesView radial_tables,
    quasar::mhd::BoundaryFlags4 flags,
    quasar::mhd::detail::FaceInterfaceView ifx,
    quasar::mhd::detail::FaceInterfaceView ify,
    const double* __restrict__ b0x, const double* __restrict__ b0y,
    const double* __restrict__ b0z,
    double* __restrict__ block_partials) {
  __shared__ double sdata[kThreadsPerBlock];
  const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  double local = 0.0;
  if (i < g.nx && j < g.ny) {
    // Compute each expensive physical face eigenvalue once, then apply the
    // finite-volume half weight together with the homogeneous retry scale.
    // Physical-fan failures stay infinite; mesh/aggregate overflow becomes
    // finite on the retry whenever a positive binary64 timestep exists.
    const Real half_weight = Real{0.5} * rate_scale;
    const Real ax_lo = reconstructed_face_rate(
        g, flags, b0_active, gamma, scheme_order, 0, i, j,
        half_weight, dx, ifx, b0x, b0y, b0z, radial_tables);
    const Real ax_hi = reconstructed_face_rate(
        g, flags, b0_active, gamma, scheme_order, 0, i + 1, j,
        half_weight, dx, ifx, b0x, b0y, b0z, radial_tables);
    const Real ay_lo = reconstructed_face_rate(
        g, flags, b0_active, gamma, scheme_order, 1, i, j,
        half_weight, dy, ify, b0x, b0y, b0z, radial_tables);
    const Real ay_hi = reconstructed_face_rate(
        g, flags, b0_active, gamma, scheme_order, 1, i, j + 1,
        half_weight, dy, ify, b0x, b0y, b0z, radial_tables);

    const Real radial_bphi = ax_lo + ax_hi;
    Real radial = radial_bphi;
    if (cylindrical != 0) {
      const Real r_c = g.r_at_cell_center(i);
      Real q = Real{0.5} * (dx / r_c);
      // q=1 on the axis cell. Clamp only roundoff above the physical [0,1]
      // interval so the zero-area low face remains exactly zero weighted.
      if (q > Real{1}) q = Real{1};
      const Real lo = Real{1} - q;
      const Real hi = Real{1} + q;
      const Real moment_norm = Real{1} + q * q / Real{3};
      const Real annular_a[2] = {ax_lo, ax_hi};
      const Real annular_b[2] = {lo, hi};
      const Real one[2] = {Real{1}, Real{1}};
      const Real radial_annular =
          quasar::numerics::scaled_product_quotient_sum(
              annular_a, annular_b, one, one, 2);

      // m_phi uses the exact piecewise-constant angular-momentum moment
      // int(r^2 dr), not the ordinary annular int(r dr) mass matrix. Its
      // axis-cell LF self-weight is 1.5*alpha/dr and is therefore the limiting
      // radial coefficient there.
      const Real angular_b[2] = {lo * lo, hi * hi};
      const Real angular_d0[2] = {moment_norm, moment_norm};
      const Real radial_angular =
          quasar::numerics::scaled_product_quotient_sum(
              annular_a, angular_b, angular_d0, one, 2);
      // B_phi obeys a metric-free radial induction equation, so it retains the
      // Cartesian half-sum even though the fluid variables are annular. This
      // is distinct at the axis and whenever the two incident alphas differ.
      radial = std::max(radial_annular,
                        std::max(radial_angular, radial_bphi));
    }
    const Real axial = ay_lo + ay_hi;
    local = radial + axial;
    if (!std::isfinite(local)) {
      local = std::numeric_limits<Real>::infinity();
    } else {
      local = std::max(Real{0}, local);
    }
  }

  sdata[tid] = local;
  __syncthreads();
  block_max_reduce(sdata, tid);
  if (tid == 0) {
    block_partials[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
  }
}

// Per-cell |div B| from the face-staggered in-plane field, reduced to one partial
// max per block. Same stencil as the host ChristliebFdCt::divergence_b_linf.
__global__ void QUASAR_LAUNCH_BOUNDS_256 ct_divb_linf_kernel(
    quasar::Grid2D g, int cylindrical,
    const double* __restrict__ bx, const double* __restrict__ by,
    double* __restrict__ block_partials) {
  __shared__ double sdata[kThreadsPerBlock];
  const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  double local = 0.0;
  if (i < g.nx && j < g.ny) {
    const Real b_lo = bx[g.index(i, j)];
    const Real b_hi = bx[g.index(i + 1, j)];
    const Real by_lo = by[g.index(i, j)];
    const Real by_hi = by[g.index(i, j + 1)];
    Real div = Real{0};
    if (cylindrical != 0) {
      const Real dr = g.dx();
      const Real r_c = g.r_at_cell_center(i);
      // Algebraically identical to
      //   (r_hi*b_hi-r_lo*b_lo)/(r_c*dr),
      // but it neither forms coordinate*field products nor subtracts two
      // nearly equal O(r*B) values on a thin, large-radius annulus.
      const Real a[6] = {b_hi, -b_lo, b_hi, b_lo, by_hi, -by_lo};
      const Real b[6] = {Real{1}, Real{1}, Real{1}, Real{1},
                         Real{1}, Real{1}};
      const Real d0[6] = {dr, dr, r_c, r_c, g.dy(), g.dy()};
      const Real d1[6] = {Real{1}, Real{1}, Real{2}, Real{2},
                          Real{1}, Real{1}};
      div = quasar::numerics::scaled_product_quotient_sum(a, b, d0, d1, 6);
    } else {
      const Real a[4] = {b_hi, -b_lo, by_hi, -by_lo};
      const Real b[4] = {Real{1}, Real{1}, Real{1}, Real{1}};
      const Real d0[4] = {g.dx(), g.dx(), g.dy(), g.dy()};
      const Real d1[4] = {Real{1}, Real{1}, Real{1}, Real{1}};
      div = quasar::numerics::scaled_product_quotient_sum(a, b, d0, d1, 4);
    }
    local = std::isfinite(div)
        ? std::abs(div)
        : std::numeric_limits<Real>::infinity();
  }

  sdata[tid] = local;
  __syncthreads();
  block_max_reduce(sdata, tid);
  if (tid == 0) {
    block_partials[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
  }
}






struct DirectionalNormParts {
  quasar::numerics::ScaledValue residual;
  quasar::numerics::ScaledValue scale;
};

// Return the signed residual and sum of absolute directional contributions as
// separate scaled values. The inputs have already cancelled Cartesian field
// offsets, so neither norm can be diluted by a large DC field.
// NOT the shared numerics::residual_is_roundoff_explained. The live-field
// divergence diagnostic takes two extra arguments -- the directional scale and
// whether the solver owns the state -- because it must distinguish a defect in
// a field the solver produced itself from one in a field an external caller
// wrote. Same name, deliberately different question.
__device__ inline bool residual_is_roundoff_explained(
    const quasar::numerics::ScaledValue& lhs,
    const quasar::numerics::ScaledValue& rhs,
    const quasar::numerics::ScaledValue& residual,
    const quasar::numerics::ScaledValue& uncertainty,
    const quasar::numerics::ScaledValue& scale,
    bool solver_owned) {
  if (solver_owned) {
    return scaled_abs_less_equal_power_of_two(
        residual, uncertainty, kDiscreteSolenoidalRoundoffUlpShift);
  }
  if (lhs.mantissa == Real{0} || rhs.mantissa == Real{0} ||
      signbit(lhs.mantissa) == signbit(rhs.mantissa)) {
    return false;
  }
  if (!scaled_abs_less_equal_power_of_two(residual, scale, -1)) {
    return false;
  }
  return scaled_abs_less_equal_power_of_two(
      residual, uncertainty, kDiscreteSolenoidalRoundoffUlpShift);
}

__device__ inline DirectionalNormParts directional_norm_parts(
    const quasar::numerics::ScaledValue& lhs,
    const quasar::numerics::ScaledValue& rhs) {
  if (!(std::isfinite(lhs.mantissa) && std::isfinite(rhs.mantissa))) {
    const quasar::numerics::ScaledValue invalid{
        std::numeric_limits<Real>::infinity(), 0};
    return DirectionalNormParts{invalid, invalid};
  }

  quasar::numerics::ScaledProductQuotientAccumulator<2> residual_sum;
  quasar::numerics::append_scaled_value_quotient(
      residual_sum, lhs, Real{1}, Real{1}, Real{1});
  quasar::numerics::append_scaled_value_quotient(
      residual_sum, rhs, Real{1}, Real{1}, Real{1});
  const quasar::numerics::ScaledValue residual =
      quasar::numerics::finish_scaled_product_quotient_sum_to_value(
          residual_sum);

  const quasar::numerics::ScaledValue lhs_abs{
      fabs(lhs.mantissa), lhs.exponent};
  const quasar::numerics::ScaledValue rhs_abs{
      fabs(rhs.mantissa), rhs.exponent};
  quasar::numerics::ScaledProductQuotientAccumulator<2> scale_sum;
  quasar::numerics::append_scaled_value_quotient(
      scale_sum, lhs_abs, Real{1}, Real{1}, Real{1});
  quasar::numerics::append_scaled_value_quotient(
      scale_sum, rhs_abs, Real{1}, Real{1}, Real{1});
  const quasar::numerics::ScaledValue scale =
      quasar::numerics::finish_scaled_product_quotient_sum_to_value(scale_sum);
  return DirectionalNormParts{residual, scale};
}



// Accumulate the two global L-infinity norms needed by
//   ||d_1+d_2||_inf / || |d_1|+|d_2| ||_inf.
// Mantissa/exponent maxima retain values outside binary64's materialized range
// and let the host make the threshold decision without a log2/exp2 round trip.
__global__ void QUASAR_LAUNCH_BOUNDS_256 ct_divb_relative_linf_kernel(
    quasar::Grid2D g, int cylindrical, int solver_owned,
    const double* __restrict__ bx, const double* __restrict__ by,
    double* __restrict__ block_partials) {
  __shared__ Real shared_mantissa[kThreadsPerBlock];
  __shared__ int shared_exponent[kThreadsPerBlock];
  const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  quasar::numerics::ScaledValue residual{};
  quasar::numerics::ScaledValue scale{};
  if (i < g.nx && j < g.ny) {
    const Real bx_lo = bx[g.index(i, j)];
    const Real bx_hi = bx[g.index(i + 1, j)];
    const Real by_lo = by[g.index(i, j)];
    const Real by_hi = by[g.index(i, j + 1)];
    const quasar::numerics::ScaledValue radial = cylindrical != 0
        ? scaled_annular_radial_divergence(
              bx_hi, bx_lo, g.dx(), g.r_at_cell_center(i))
        : scaled_directional_derivative(bx_hi, bx_lo, g.dx());
    const quasar::numerics::ScaledValue axial =
        scaled_directional_derivative(by_hi, by_lo, g.dy());
    const quasar::numerics::ScaledValue radial_roundoff = cylindrical != 0
        ? scaled_annular_radial_roundoff(
              bx_hi, bx_lo, g.dx(), g.r_at_cell_center(i))
        : scaled_directional_roundoff(bx_hi, bx_lo, g.dx());
    const quasar::numerics::ScaledValue axial_roundoff =
        scaled_directional_roundoff(by_hi, by_lo, g.dy());
    const DirectionalNormParts norms = directional_norm_parts(radial, axial);
    quasar::numerics::ScaledProductQuotientAccumulator<2> roundoff_sum;
    quasar::numerics::append_scaled_value_quotient(
        roundoff_sum, radial_roundoff, Real{1}, Real{1}, Real{1});
    quasar::numerics::append_scaled_value_quotient(
        roundoff_sum, axial_roundoff, Real{1}, Real{1}, Real{1});
    const quasar::numerics::ScaledValue roundoff =
        quasar::numerics::finish_scaled_product_quotient_sum_to_value(
            roundoff_sum);
    const bool explained = residual_is_roundoff_explained(
        radial, axial, norms.residual, roundoff, norms.scale,
        solver_owned != 0);
    residual = explained ? quasar::numerics::ScaledValue{} : norms.residual;
    scale = norms.scale;
  }

  const std::size_t block_count =
      static_cast<std::size_t>(gridDim.x) * gridDim.y;
  const std::size_t block_index =
      static_cast<std::size_t>(blockIdx.y) * gridDim.x + blockIdx.x;
  shared_mantissa[tid] = residual.mantissa;
  shared_exponent[tid] = residual.exponent;
  __syncthreads();
  block_scaled_abs_max_reduce(shared_mantissa, shared_exponent, tid);
  if (tid == 0) {
    block_partials[block_index] = shared_mantissa[0];
    block_partials[block_count + block_index] =
        static_cast<Real>(shared_exponent[0]);
  }
  shared_mantissa[tid] = scale.mantissa;
  shared_exponent[tid] = scale.exponent;
  __syncthreads();
  block_scaled_abs_max_reduce(shared_mantissa, shared_exponent, tid);
  if (tid == 0) {
    block_partials[2 * block_count + block_index] = shared_mantissa[0];
    block_partials[3 * block_count + block_index] =
        static_cast<Real>(shared_exponent[0]);
  }
}

// -- Second-stage folds -------------------------------------------------------
//
// Each kernel is launched as ONE block of kThreadsPerBlock threads and folds the
// per-block partials to a single scalar written to `out`. `blocks` is the number
// of 16x16 tiles over the grid, so a single block striding over it is ample.

// Fold a partials array to its maximum, with any non-finite partial poisoning
// the result to +inf.
//
// The host loop this replaces returned +inf on the first non-finite entry.
// Mapping every non-finite input to +inf up front reaches the same answer in a
// tree, because +inf compares greater than every finite partial and the fold is
// seeded at 0 exactly as the host was. Comparison-only, so the result is
// bit-identical to the loop it replaces.
__global__ void max_finish_kernel(unsigned blocks,
                                  const double* __restrict__ partials,
                                  double* __restrict__ out) {
  __shared__ double sdata[kThreadsPerBlock];
  const unsigned tid = threadIdx.x;
  double best = 0.0;
  for (unsigned k = tid; k < blocks; k += kThreadsPerBlock) {
    const double value = partials[k];
    const double mapped = std::isfinite(value)
        ? value
        : std::numeric_limits<double>::infinity();
    if (mapped > best) best = mapped;
  }
  sdata[tid] = best;
  __syncthreads();
  block_max_reduce(sdata, tid);
  if (tid == 0) out[0] = sdata[0];
}

// Fold a partials array to its minimum, clamped to [0,1], with any non-finite
// partial forcing 0.
//
// Same argument as above in the other direction: 0 is both the host loop's early
// return and the lower clamp bound, so mapping non-finite to 0 survives the
// min-fold and the clamp unchanged. Comparison-only, hence bit-identical.
__global__ void min_finish_clamped_kernel(unsigned blocks,
                                          const double* __restrict__ partials,
                                          double* __restrict__ out) {
  __shared__ double sdata[kThreadsPerBlock];
  const unsigned tid = threadIdx.x;
  double best = 1.0;
  for (unsigned k = tid; k < blocks; k += kThreadsPerBlock) {
    const double value = partials[k];
    const double mapped = std::isfinite(value) ? value : 0.0;
    if (mapped < best) best = mapped;
  }
  sdata[tid] = best;
  __syncthreads();
  block_min_reduce(sdata, tid);
  if (tid == 0) {
    const double folded = sdata[0];
    out[0] = fmax(0.0, fmin(1.0, folded));
  }
}

// One step of the scaled-maximum fold, in the exact form the host loop used.
//
// This is deliberately NOT block_scaled_abs_max_reduce: that helper keeps the
// raw mantissa where this one keeps its magnitude, and it does not screen a
// non-finite EXPONENT. The relative div(B) acceptance threshold is measured
// against this value, so the rule has to be the one the threshold was
// calibrated against.
//
// The rule is a commutative monoid -- identity {0,0}, absorbing {inf,0}, and
// otherwise a lexicographic max on (exponent, |mantissa|) -- which is why
// folding it as a tree lands on the same value as the sequential host loop.
__device__ inline void retain_scaled_max(Real mantissa, Real exponent_value,
                                         Real& best_mantissa,
                                         int& best_exponent) {
  if (!std::isfinite(mantissa) || !std::isfinite(exponent_value)) {
    best_mantissa = std::numeric_limits<Real>::infinity();
    best_exponent = 0;
    return;
  }
  if (!std::isfinite(best_mantissa) || mantissa == Real{0}) return;
  const int exponent = static_cast<int>(exponent_value);
  if (best_mantissa == Real{0} || exponent > best_exponent ||
      (exponent == best_exponent && fabs(mantissa) > fabs(best_mantissa))) {
    best_mantissa = fabs(mantissa);
    best_exponent = exponent;
  }
}

// Fold the four residual/scale mantissa+exponent planes and form the relative
// defect. `partials` is laid out as [residual mantissa | residual exponent |
// scale mantissa | scale exponent], each `blocks` long.
//
// The two scalbn calls and the division are the only arithmetic here, and
// neither is a multiply-add, so this kernel does not need -ffp-contract=off to
// reproduce the host result exactly.
__global__ void scaled_ratio_finish_kernel(unsigned blocks,
                                           const double* __restrict__ partials,
                                           double* __restrict__ out) {
  __shared__ Real residual_mantissa[kThreadsPerBlock];
  __shared__ int residual_exponent[kThreadsPerBlock];
  __shared__ Real scale_mantissa[kThreadsPerBlock];
  __shared__ int scale_exponent[kThreadsPerBlock];
  const unsigned tid = threadIdx.x;

  Real res_m = Real{0};
  int res_e = 0;
  Real scl_m = Real{0};
  int scl_e = 0;
  for (unsigned k = tid; k < blocks; k += kThreadsPerBlock) {
    retain_scaled_max(partials[k], partials[blocks + k], res_m, res_e);
    retain_scaled_max(partials[2 * blocks + k], partials[3 * blocks + k],
                      scl_m, scl_e);
  }
  residual_mantissa[tid] = res_m;
  residual_exponent[tid] = res_e;
  scale_mantissa[tid] = scl_m;
  scale_exponent[tid] = scl_e;
  __syncthreads();

  for (unsigned stride = kThreadsPerBlock / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      retain_scaled_max(residual_mantissa[tid + stride],
                        static_cast<Real>(residual_exponent[tid + stride]),
                        residual_mantissa[tid], residual_exponent[tid]);
      retain_scaled_max(scale_mantissa[tid + stride],
                        static_cast<Real>(scale_exponent[tid + stride]),
                        scale_mantissa[tid], scale_exponent[tid]);
    }
    __syncthreads();
  }
  if (tid != 0) return;

  const Real folded_residual = residual_mantissa[0];
  const Real folded_scale = scale_mantissa[0];
  const int folded_residual_exponent = residual_exponent[0];
  const int folded_scale_exponent = scale_exponent[0];
  const Real infinity = std::numeric_limits<Real>::infinity();

  if (!(std::isfinite(folded_residual) && std::isfinite(folded_scale))) {
    out[0] = infinity;
    return;
  }
  if (folded_residual == Real{0}) {
    out[0] = Real{0};
    return;
  }
  if (!(folded_scale > Real{0})) {
    out[0] = infinity;
    return;
  }
  const int common_exponent = folded_residual_exponent > folded_scale_exponent
      ? folded_residual_exponent
      : folded_scale_exponent;
  const Real numerator = fabs(
      ::scalbn(folded_residual, folded_residual_exponent - common_exponent));
  const Real denominator = fabs(
      ::scalbn(folded_scale, folded_scale_exponent - common_exponent));
  if (!(denominator > Real{0}) || !std::isfinite(numerator) ||
      !std::isfinite(denominator)) {
    out[0] = infinity;
    return;
  }
  out[0] = numerator / denominator;
}

// Ensure `scratch` holds at least n_blocks elements; (re)allocate only when it is
// too small. The reduction kernels write every slot before reading, so an
// uninitialized (possibly larger) buffer is safe to reuse across calls.
//
// Callers size this to the partials PLUS one trailing slot, which the finish
// kernel above writes its single scalar into. Folding in place into partial 0
// would work too, but the extra slot keeps the partials readable for a failing
// test to inspect.
inline void ensure_scratch(quasar::backend::DeviceBuffer<Real>& scratch,
                           std::size_t n_blocks) {
  if (scratch.size() < n_blocks) {
    scratch = quasar::backend::DeviceBuffer<Real>{n_blocks, quasar::backend::uninitialized};
  }
}

// Launch one of the finish kernels above over `n_blocks` partials and return the
// folded scalar. `result_slot` is the element just past the partials.
template <typename Kernel>
inline Real finish_on_device(Kernel kernel,
                             quasar::backend::DeviceBuffer<Real>& scratch,
                             std::size_t n_blocks, std::size_t result_slot,
                             hipStream_t stream) {
  Real* const base = scratch.device_ptr();
  hipLaunchKernelGGL(kernel, dim3(1), dim3(kThreadsPerBlock), 0, stream,
                     static_cast<unsigned>(n_blocks), base, base + result_slot);
  QUASAR_HIP_CHECK(::hipGetLastError());
  Real host = Real{0};
  QUASAR_HIP_CHECK(::hipMemcpyAsync(&host, base + result_slot, sizeof(Real),
                                    hipMemcpyDeviceToHost, stream));
  QUASAR_HIP_CHECK(::hipStreamSynchronize(stream));
  return host;
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_cfl_max_rate(
    const quasar::numerics::MhdInterfaceStates<Real>& ifx,
    const quasar::numerics::MhdInterfaceStates<Real>& ify,
    const MhdBackgroundField<Real>& b0, Real gamma,
    backend::DeviceBuffer<Real>& scratch, ScaledCflRate* host_rate,
    stream_t stream, int scheme_order, bool cylindrical,
    BoundaryFlags4 flags,
    quasar::numerics::RadialTablesView radial_tables) {
  const Grid2D g = ifx.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  const std::size_t n_blocks = static_cast<std::size_t>(grid.x) * grid.y;

  // Caller-owned scratch reused across calls (the auto-dt loop calls this every
  // step): grow it only when too small so there is no per-step hipMalloc/hipFree.
  // One slot past the partials holds the folded scalar.
  ensure_scratch(scratch, n_blocks + 1);

  const Real dx = g.dx();
  const Real dy = g.dy();

  const int b0_active = b0.active ? 1 : 0;
  const Real* b0x = b0.active ? b0.b0x_face.device_ptr() : nullptr;
  const Real* b0y = b0.active ? b0.b0y_face.device_ptr() : nullptr;
  const Real* b0z = b0.active ? b0.b0z_cell.device_ptr() : nullptr;

  // Keep the kernel POD construction explicit. This is host-only pointer
  // packaging; avoiding a generic lambda here also keeps older HIP frontends
  // from trying to infer device attributes for a host closure.
  const detail::FaceInterfaceView x_view{
      ifx.Lrho.device_ptr(), ifx.Lmx.device_ptr(),
      ifx.Lmy.device_ptr(), ifx.Lmz.device_ptr(),
      ifx.Lenergy.device_ptr(), ifx.Lbx.device_ptr(),
      ifx.Lby.device_ptr(), ifx.Lbz.device_ptr(),
      ifx.Rrho.device_ptr(), ifx.Rmx.device_ptr(),
      ifx.Rmy.device_ptr(), ifx.Rmz.device_ptr(),
      ifx.Renergy.device_ptr(), ifx.Rbx.device_ptr(),
      ifx.Rby.device_ptr(), ifx.Rbz.device_ptr()};
  const detail::FaceInterfaceView y_view{
      ify.Lrho.device_ptr(), ify.Lmx.device_ptr(),
      ify.Lmy.device_ptr(), ify.Lmz.device_ptr(),
      ify.Lenergy.device_ptr(), ify.Lbx.device_ptr(),
      ify.Lby.device_ptr(), ify.Lbz.device_ptr(),
      ify.Rrho.device_ptr(), ify.Rmx.device_ptr(),
      ify.Rmy.device_ptr(), ify.Rmz.device_ptr(),
      ify.Renergy.device_ptr(), ify.Rbx.device_ptr(),
      ify.Rby.device_ptr(), ify.Rbz.device_ptr()};

  hipLaunchKernelGGL(cfl_max_rate_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, b0_active, gamma, dx, dy, Real{1},
                     scheme_order, cylindrical ? 1 : 0, radial_tables, flags,
                     x_view, y_view,
                     b0x, b0y, b0z, scratch.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  backend::device_synchronize(stream);

  Real rate_scale = Real{1};
  Real scaled_max_rate = finish_on_device(
      max_finish_kernel, scratch, n_blocks, n_blocks,
      static_cast<hipStream_t>(stream));
  if (!std::isfinite(scaled_max_rate)) {
    // Every valid physical face alpha is finite before its mesh factor is
    // applied. An infinite first reduction can therefore be either a genuine
    // state/fan failure (which remains infinite) or only an out-of-range
    // metric/incident-face sum. Retry once with an exact power-of-two scale.
    // If that retry still overflows, the true rate exceeds roughly 2^1088;
    // because cfl<=1, its reciprocal lies below binary64's 2^-1074 minimum.
    rate_scale = kCflOverflowRetryScale;
    hipLaunchKernelGGL(
        cfl_max_rate_kernel, grid, block, 0,
        static_cast<hipStream_t>(stream), g, b0_active, gamma, dx, dy,
        rate_scale, scheme_order, cylindrical ? 1 : 0, radial_tables, flags,
        x_view, y_view, b0x, b0y, b0z, scratch.device_ptr());
    QUASAR_HIP_CHECK(::hipGetLastError());
    backend::device_synchronize(stream);
    scaled_max_rate = finish_on_device(
        max_finish_kernel, scratch, n_blocks, n_blocks,
        static_cast<hipStream_t>(stream));
  }
  *host_rate = ScaledCflRate{scaled_max_rate, rate_scale};
}

void launch_mhd_ct_divb_linf(const MhdField2D<Real>& u,
                             backend::DeviceBuffer<Real>& scratch,
                             Real* host_linf, stream_t stream,
                             bool cylindrical) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  const std::size_t n_blocks = static_cast<std::size_t>(grid.x) * grid.y;

  // Caller-owned scratch reused across calls (same rationale as the CFL
  // reduction); grow it only when too small, plus the folded-scalar slot.
  ensure_scratch(scratch, n_blocks + 1);

  hipLaunchKernelGGL(ct_divb_linf_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, cylindrical ? 1 : 0,
                     u.bx_face.device_ptr(), u.by_face.device_ptr(),
                     scratch.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  backend::device_synchronize(stream);

  *host_linf = finish_on_device(max_finish_kernel, scratch, n_blocks, n_blocks,
                                static_cast<hipStream_t>(stream));
}

void launch_mhd_ct_divb_relative_linf(
    const MhdField2D<Real>& u, backend::DeviceBuffer<Real>& scratch,
    Real* host_linf, stream_t stream, bool cylindrical,
    bool solver_owned) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  const std::size_t n_blocks = static_cast<std::size_t>(grid.x) * grid.y;
  ensure_scratch(scratch, 4 * n_blocks + 1);

  hipLaunchKernelGGL(
      ct_divb_relative_linf_kernel, grid, block, 0,
      static_cast<hipStream_t>(stream), g, cylindrical ? 1 : 0,
      solver_owned ? 1 : 0,
      u.bx_face.device_ptr(), u.by_face.device_ptr(), scratch.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  backend::device_synchronize(stream);

  *host_linf = finish_on_device(scaled_ratio_finish_kernel, scratch, n_blocks,
                                4 * n_blocks,
                                static_cast<hipStream_t>(stream));
}

void launch_mhd_admissible_fraction(
    const MhdField2D<Real>& base, const MhdField2D<Real>& candidate,
    Real rho_floor, Real p_floor, Real gamma,
    backend::DeviceBuffer<Real>& scratch, Real* host_theta, stream_t stream,
    int collocation_order,
    quasar::numerics::RadialTablesView radial_tables) {
  const Grid2D g = base.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  const std::size_t n_blocks = static_cast<std::size_t>(grid.x) * grid.y;
  ensure_scratch(scratch, n_blocks + 1);

  const Real q_floor = p_floor / (gamma - Real{1});
  hipLaunchKernelGGL(
      admissible_fraction_kernel, grid, block, 0,
      static_cast<hipStream_t>(stream), g, rho_floor, q_floor,
      collocation_order, radial_tables,
      base.rho.device_ptr(), base.mx.device_ptr(), base.my.device_ptr(),
      base.mz.device_ptr(), base.energy.device_ptr(), base.bx_face.device_ptr(),
      base.by_face.device_ptr(), base.bz_cell.device_ptr(),
      candidate.rho.device_ptr(), candidate.mx.device_ptr(),
      candidate.my.device_ptr(), candidate.mz.device_ptr(),
      candidate.energy.device_ptr(), candidate.bx_face.device_ptr(),
      candidate.by_face.device_ptr(), candidate.bz_cell.device_ptr(),
      scratch.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  backend::device_synchronize(stream);
  *host_theta = finish_on_device(min_finish_clamped_kernel, scratch, n_blocks,
                                 n_blocks, static_cast<hipStream_t>(stream));
}

}  // namespace quasar::mhd
