// Device reduction kernels for the ideal-MHD solver: the CFL maximum signal
// speed and the constrained-transport div(B) L-infinity diagnostic.
//
// Both quantities are a MAX over all interior cells. HIP has no native
// atomicMax for double, so each launcher uses a TWO-PASS block reduction: a
// kernel computes the per-cell value, reduces it within each block via a
// shared-memory tree to one partial max per block (written to a scratch
// DeviceBuffer), and the host finishes the reduction over the (small)
// block-partials array. The final scalar is then written to the caller's host
// pointer before the launcher returns (device_synchronize'd).

#include "quasar/backend/memory.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_background.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include "backend/hip/hip_check.hpp"

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

inline dim3 mhd_grid_2d(const quasar::Grid2D& g, dim3 block) {
  return dim3((static_cast<unsigned>(g.nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(g.ny) + block.y - 1) / block.y);
}

// Shared-memory tree reduction of `sdata` (length kThreadsPerBlock) to its max,
// left in sdata[0]. tid is the linear thread index within the block.
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
__global__ void admissible_fraction_kernel(
    quasar::Grid2D g, Real rho_floor, Real q_floor, int collocation_order,
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
        collocation_order);
    const MhdState b = quasar::mhd::load_cell_state(
        g, b_rho, b_mx, b_my, b_mz, b_energy, b_bx, b_by, b_bz, i, j,
        collocation_order);
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
// quotient at its physical exponent.  The CFL path needs the third numerator
// factor for sqrt(gamma)*sqrt(p)*face_weight/(sqrt(rho)*spacing).
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

__device__ inline Real scaled_field_rate(
    Real perturbation, Real background, Real face_weight,
    Real sqrt_rho, Real spacing) {
  const Real a[2] = {perturbation, background};
  const Real b[2] = {face_weight, face_weight};
  const Real d0[2] = {sqrt_rho, sqrt_rho};
  const Real d1[2] = {spacing, spacing};
  return quasar::numerics::scaled_product_quotient_sum(a, b, d0, d1, 2);
}

// One side's contribution face_weight*(|v_n|+c_fast)/spacing.  Compute the
// RATE directly: a perfectly finite conservative state can have an
// unrepresentable physical velocity or magnetosonic speed while division by a
// large cell spacing makes the Courant rate finite.  Magnetosonic eigenvalues
// are homogeneous in their constituent speeds, so evaluating the normalized
// quadratic from already-scaled sound/Alfven rates is algebraically identical.
__device__ inline Real face_state_rate(
    const MhdState& s, const MhdBackground& bg, int dir, Real gamma,
    Real face_weight, Real spacing) {
  if (face_weight == Real{0}) return Real{0};
  const Real p = quasar::numerics::pressure(s, gamma);
  if (!(std::isfinite(s.rho) && s.rho > Real{0} &&
        std::isfinite(s.mx) && std::isfinite(s.my) &&
        std::isfinite(s.mz) && std::isfinite(s.energy) &&
        std::isfinite(s.bx) && std::isfinite(s.by) &&
        std::isfinite(s.bz) && std::isfinite(bg.b0x) &&
        std::isfinite(bg.b0y) && std::isfinite(bg.b0z) &&
        std::isfinite(p) && p > Real{0} &&
        std::isfinite(face_weight) && face_weight > Real{0} &&
        std::isfinite(spacing) && spacing > Real{0})) {
    return std::numeric_limits<Real>::infinity();
  }

  const Real sqrt_rho = std::sqrt(s.rho);
  const Real sound_rate = scaled_product3_quotient2(
      std::sqrt(gamma), std::sqrt(p), face_weight, sqrt_rho, spacing);
  const Real ax_rate = scaled_field_rate(
      s.bx, bg.b0x, face_weight, sqrt_rho, spacing);
  const Real ay_rate = scaled_field_rate(
      s.by, bg.b0y, face_weight, sqrt_rho, spacing);
  const Real az_rate = scaled_field_rate(
      s.bz, bg.b0z, face_weight, sqrt_rho, spacing);
  if (!(std::isfinite(sound_rate) && std::isfinite(ax_rate) &&
        std::isfinite(ay_rate) && std::isfinite(az_rate))) {
    return std::numeric_limits<Real>::infinity();
  }

  const Real alfven_rate =
      quasar::numerics::scaled_norm3(ax_rate, ay_rate, az_rate);
  const Real alfven_normal_rate = std::abs(dir == 0 ? ax_rate : ay_rate);
  const Real scale = std::max(sound_rate, alfven_rate);
  Real fast_rate = Real{0};
  if (scale > Real{0}) {
    const Real an = sound_rate / scale;
    const Real bn = alfven_rate / scale;
    const Real can = alfven_normal_rate / scale;
    const Real a2n = an * an;
    const Real b2n = bn * bn;
    const Real sum = a2n + b2n;
    Real disc = sum * sum - Real{4} * a2n * can * can;
    if (disc < Real{0}) disc = Real{0};
    fast_rate = scale * std::sqrt(
        Real{0.5} * (sum + std::sqrt(disc)));
  }

  const Real momentum = std::abs(dir == 0 ? s.mx : s.my);
  const Real velocity_a[1] = {momentum};
  const Real velocity_b[1] = {face_weight};
  const Real velocity_d0[1] = {s.rho};
  const Real velocity_d1[1] = {spacing};
  const Real velocity_rate = quasar::numerics::scaled_product_quotient_sum(
      velocity_a, velocity_b, velocity_d0, velocity_d1, 1);
  const Real rate = velocity_rate + fast_rate;
  return std::isfinite(rate) ? rate
                             : std::numeric_limits<Real>::infinity();
}

struct DeviceInterfaceStates {
  const double* lrho;
  const double* lmx;
  const double* lmy;
  const double* lmz;
  const double* lenergy;
  const double* lbx;
  const double* lby;
  const double* lbz;
  const double* rrho;
  const double* rmx;
  const double* rmy;
  const double* rmz;
  const double* renergy;
  const double* rbx;
  const double* rby;
  const double* rbz;
};

__device__ inline MhdState load_interface_state(
    const DeviceInterfaceStates& iface, std::size_t k, bool left) {
  MhdState state;
  if (left) {
    state.rho = iface.lrho[k];
    state.mx = iface.lmx[k];
    state.my = iface.lmy[k];
    state.mz = iface.lmz[k];
    state.energy = iface.lenergy[k];
    state.bx = iface.lbx[k];
    state.by = iface.lby[k];
    state.bz = iface.lbz[k];
  } else {
    state.rho = iface.rrho[k];
    state.mx = iface.rmx[k];
    state.my = iface.rmy[k];
    state.mz = iface.rmz[k];
    state.energy = iface.renergy[k];
    state.bx = iface.rbx[k];
    state.by = iface.rby[k];
    state.bz = iface.rbz[k];
  }
  return state;
}

// Maximum weighted Courant rate on one numerical face, evaluated from the
// exact reconstructed L/R states consumed by HLLD. This closes the amplitude
// gap in which an MP face overshoot has a larger signal speed than either
// incident cell average. The duplicate periodic high face uses the same face-0
// data/background mapping as the production Riemann kernel.
__device__ inline Real reconstructed_face_rate(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags,
    int b0_active, Real gamma, int low_order, int dir,
    int face_i, int face_j, Real face_weight, Real spacing,
    const DeviceInterfaceStates& iface,
    const double* b0x, const double* b0y, const double* b0z) {
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
  const MhdState left = load_interface_state(iface, k, true);
  const MhdState right = load_interface_state(iface, k, false);

  MhdBackground bg{};
  if (b0_active != 0) {
    bg = quasar::mhd::load_interface_background(
        g, dir, b0x, b0y, b0z, face_i, face_j,
        low_order != 0 ? 1 : 0);
  }
  const Real left_rate = face_state_rate(
      left, bg, dir, gamma, face_weight, spacing);
  const Real right_rate = face_state_rate(
      right, bg, dir, gamma, face_weight, spacing);
  return left_rate > right_rate ? left_rate : right_rate;
}

// Per-cell finite-volume CFL coefficient from the incident numerical faces.
// The half-sums are the exact self-weights of the local LF updates. Cylindrical
// mode takes the largest radial self-weight among the annular fluid, exact
// angular-momentum, and metric-free B_phi operators.
__global__ void cfl_max_rate_kernel(
    quasar::Grid2D g, int b0_active, Real gamma, Real dx, Real dy,
    int low_order, int cylindrical,
    quasar::mhd::BoundaryFlags4 flags,
    DeviceInterfaceStates ifx, DeviceInterfaceStates ify,
    const double* __restrict__ b0x, const double* __restrict__ b0y,
    const double* __restrict__ b0z,
    double* __restrict__ block_partials) {
  __shared__ double sdata[kThreadsPerBlock];
  const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  double local = 0.0;
  if (i < g.nx && j < g.ny) {
    // Compute each expensive face state/eigenvalue once at the metric-free
    // half weight. This quantity is itself required by B_phi, so if it is not
    // representable then no later smaller metric factor can make the complete
    // local CFL coefficient representable.
    const Real ax_lo = reconstructed_face_rate(
        g, flags, b0_active, gamma, low_order, 0, i, j,
        Real{0.5}, dx, ifx, b0x, b0y, b0z);
    const Real ax_hi = reconstructed_face_rate(
        g, flags, b0_active, gamma, low_order, 0, i + 1, j,
        Real{0.5}, dx, ifx, b0x, b0y, b0z);
    const Real ay_lo = reconstructed_face_rate(
        g, flags, b0_active, gamma, low_order, 1, i, j,
        Real{0.5}, dy, ify, b0x, b0y, b0z);
    const Real ay_hi = reconstructed_face_rate(
        g, flags, b0_active, gamma, low_order, 1, i, j + 1,
        Real{0.5}, dy, ify, b0x, b0y, b0z);

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
__global__ void ct_divb_linf_kernel(
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

// Return the cancellation defect of a short signed product/quotient sum,
// normalized by the corresponding absolute term sum.  Every nonzero term is
// first represented as mantissa*2^exponent; scaling to the largest exponent
// keeps both numerator and denominator in an O(1) range even when the physical
// divergence terms overflow, underflow, or cancel across hundreds of decades.
__device__ inline Real normalized_divergence_defect(
    const Real* numerator, const Real* coefficient,
    const Real* denominator, int count) {
  constexpr int kMaxTerms = 4;
  Real mantissa[kMaxTerms]{};
  int exponent[kMaxTerms]{};
  int max_exponent = std::numeric_limits<int>::min();

  for (int k = 0; k < count; ++k) {
    if (!std::isfinite(numerator[k]) ||
        !std::isfinite(coefficient[k]) ||
        !std::isfinite(denominator[k]) ||
        !(denominator[k] > Real{0})) {
      return std::numeric_limits<Real>::infinity();
    }
    if (numerator[k] == Real{0} || coefficient[k] == Real{0}) continue;

    int en = 0, ec = 0, ed = 0;
    const Real mn = frexp(numerator[k], &en);
    const Real mc = frexp(coefficient[k], &ec);
    const Real md = frexp(denominator[k], &ed);
    int shift = 0;
    mantissa[k] = frexp((mn * mc) / md, &shift);
    exponent[k] = en + ec - ed + shift;
    if (exponent[k] > max_exponent) max_exponent = exponent[k];
  }

  if (max_exponent == std::numeric_limits<int>::min()) return Real{0};

  // Kahan accumulation avoids making the diagnostic itself appreciably less
  // accurate than the roundoff envelope it is intended to measure.
  Real signed_sum = Real{0};
  Real compensation = Real{0};
  Real absolute_sum = Real{0};
  for (int k = 0; k < count; ++k) {
    if (mantissa[k] == Real{0}) continue;
    const Real term = scalbn(mantissa[k], exponent[k] - max_exponent);
    const Real corrected = term - compensation;
    const Real updated = signed_sum + corrected;
    compensation = (updated - signed_sum) - corrected;
    signed_sum = updated;
    absolute_sum += std::fabs(term);
  }
  if (!(absolute_sum > Real{0}) || !std::isfinite(absolute_sum)) {
    return std::numeric_limits<Real>::infinity();
  }
  return std::fabs(signed_sum) / absolute_sum;
}

// Per-cell dimensionless defect in the exact CT divergence stencil.  The
// cylindrical radial terms use their actual annular weights (1+q,1-q), rather
// than an algebraic expansion containing cancelling copies of the zero-area
// axis face; an arbitrary axis-face value therefore cannot mask a real defect.
__global__ void ct_divb_relative_linf_kernel(
    quasar::Grid2D g, int cylindrical,
    const double* __restrict__ bx, const double* __restrict__ by,
    double* __restrict__ block_partials) {
  __shared__ double sdata[kThreadsPerBlock];
  const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;

  Real local = Real{0};
  if (i < g.nx && j < g.ny) {
    const Real bx_lo = bx[g.index(i, j)];
    const Real bx_hi = bx[g.index(i + 1, j)];
    const Real by_lo = by[g.index(i, j)];
    const Real by_hi = by[g.index(i, j + 1)];
    const Real numerator[4] = {bx_hi, bx_lo, by_hi, by_lo};
    Real coefficient[4] = {Real{1}, Real{-1}, Real{1}, Real{-1}};
    const Real denominator[4] = {g.dx(), g.dx(), g.dy(), g.dy()};
    if (cylindrical != 0) {
      Real q = Real{0.5} * (g.dx() / g.r_at_cell_center(i));
      if (q > Real{1}) q = Real{1};
      coefficient[0] = Real{1} + q;
      coefficient[1] = -(Real{1} - q);
    }
    local = normalized_divergence_defect(
        numerator, coefficient, denominator, 4);
  }

  sdata[tid] = local;
  __syncthreads();
  block_max_reduce(sdata, tid);
  if (tid == 0) {
    block_partials[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
  }
}

// Finish a block-partials max on the host. Copies the partials to host and folds
// them into a single scalar. n_blocks is the number of 16x16 tiles over the grid
// (small), so a per-call host vector is negligible next to the device reduction.
inline Real finish_host_max(const quasar::backend::DeviceBuffer<Real>& partials,
                            std::size_t n_blocks) {
  std::vector<Real> host(n_blocks);
  partials.copy_to_host(host.data(), n_blocks);
  Real m = Real{0};
  for (std::size_t b = 0; b < n_blocks; ++b) {
    if (!std::isfinite(host[b])) {
      return std::numeric_limits<Real>::infinity();
    }
    if (host[b] > m) m = host[b];
  }
  return m;
}

inline Real finish_host_min(const quasar::backend::DeviceBuffer<Real>& partials,
                            std::size_t n_blocks) {
  std::vector<Real> host(n_blocks);
  partials.copy_to_host(host.data(), n_blocks);
  Real m = Real{1};
  for (std::size_t b = 0; b < n_blocks; ++b) {
    if (!std::isfinite(host[b])) return Real{0};
    if (host[b] < m) m = host[b];
  }
  return std::max(Real{0}, std::min(Real{1}, m));
}

// Ensure `scratch` holds at least n_blocks elements; (re)allocate only when it is
// too small. The reduction kernels write every slot before reading, so an
// uninitialized (possibly larger) buffer is safe to reuse across calls.
inline void ensure_scratch(quasar::backend::DeviceBuffer<Real>& scratch,
                           std::size_t n_blocks) {
  if (scratch.size() < n_blocks) {
    scratch = quasar::backend::DeviceBuffer<Real>{n_blocks, quasar::backend::uninitialized};
  }
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_cfl_max_rate(
    const quasar::numerics::MhdInterfaceStates<Real>& ifx,
    const quasar::numerics::MhdInterfaceStates<Real>& ify,
    const MhdBackgroundField<Real>& b0, Real gamma,
    backend::DeviceBuffer<Real>& scratch, Real* host_max_rate,
    stream_t stream, bool low_order, bool cylindrical,
    BoundaryFlags4 flags) {
  const Grid2D g = ifx.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  const std::size_t n_blocks = static_cast<std::size_t>(grid.x) * grid.y;

  // Caller-owned scratch reused across calls (the auto-dt loop calls this every
  // step): grow it only when too small so there is no per-step hipMalloc/hipFree.
  ensure_scratch(scratch, n_blocks);

  const Real dx = g.dx();
  const Real dy = g.dy();

  const int b0_active = b0.active ? 1 : 0;
  const Real* b0x = b0.active ? b0.b0x_face.device_ptr() : nullptr;
  const Real* b0y = b0.active ? b0.b0y_face.device_ptr() : nullptr;
  const Real* b0z = b0.active ? b0.b0z_cell.device_ptr() : nullptr;

  // Keep the kernel POD construction explicit. This is host-only pointer
  // packaging; avoiding a generic lambda here also keeps older HIP frontends
  // from trying to infer device attributes for a host closure.
  const DeviceInterfaceStates x_view{
      ifx.Lrho.device_ptr(), ifx.Lmx.device_ptr(),
      ifx.Lmy.device_ptr(), ifx.Lmz.device_ptr(),
      ifx.Lenergy.device_ptr(), ifx.Lbx.device_ptr(),
      ifx.Lby.device_ptr(), ifx.Lbz.device_ptr(),
      ifx.Rrho.device_ptr(), ifx.Rmx.device_ptr(),
      ifx.Rmy.device_ptr(), ifx.Rmz.device_ptr(),
      ifx.Renergy.device_ptr(), ifx.Rbx.device_ptr(),
      ifx.Rby.device_ptr(), ifx.Rbz.device_ptr()};
  const DeviceInterfaceStates y_view{
      ify.Lrho.device_ptr(), ify.Lmx.device_ptr(),
      ify.Lmy.device_ptr(), ify.Lmz.device_ptr(),
      ify.Lenergy.device_ptr(), ify.Lbx.device_ptr(),
      ify.Lby.device_ptr(), ify.Lbz.device_ptr(),
      ify.Rrho.device_ptr(), ify.Rmx.device_ptr(),
      ify.Rmy.device_ptr(), ify.Rmz.device_ptr(),
      ify.Renergy.device_ptr(), ify.Rbx.device_ptr(),
      ify.Rby.device_ptr(), ify.Rbz.device_ptr()};

  hipLaunchKernelGGL(cfl_max_rate_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, b0_active, gamma, dx, dy,
                     low_order ? 1 : 0, cylindrical ? 1 : 0, flags,
                     x_view, y_view,
                     b0x, b0y, b0z, scratch.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  backend::device_synchronize(stream);

  *host_max_rate = finish_host_max(scratch, n_blocks);
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
  // reduction); grow it only when too small.
  ensure_scratch(scratch, n_blocks);

  hipLaunchKernelGGL(ct_divb_linf_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, cylindrical ? 1 : 0,
                     u.bx_face.device_ptr(), u.by_face.device_ptr(),
                     scratch.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  backend::device_synchronize(stream);

  *host_linf = finish_host_max(scratch, n_blocks);
}

void launch_mhd_ct_divb_relative_linf(
    const MhdField2D<Real>& u, backend::DeviceBuffer<Real>& scratch,
    Real* host_linf, stream_t stream, bool cylindrical) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  const std::size_t n_blocks = static_cast<std::size_t>(grid.x) * grid.y;
  ensure_scratch(scratch, n_blocks);

  hipLaunchKernelGGL(
      ct_divb_relative_linf_kernel, grid, block, 0,
      static_cast<hipStream_t>(stream), g, cylindrical ? 1 : 0,
      u.bx_face.device_ptr(), u.by_face.device_ptr(), scratch.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  backend::device_synchronize(stream);

  *host_linf = finish_host_max(scratch, n_blocks);
}

void launch_mhd_admissible_fraction(
    const MhdField2D<Real>& base, const MhdField2D<Real>& candidate,
    Real rho_floor, Real p_floor, Real gamma,
    backend::DeviceBuffer<Real>& scratch, Real* host_theta, stream_t stream,
    int collocation_order) {
  const Grid2D g = base.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  const std::size_t n_blocks = static_cast<std::size_t>(grid.x) * grid.y;
  ensure_scratch(scratch, n_blocks);

  const Real q_floor = p_floor / (gamma - Real{1});
  hipLaunchKernelGGL(
      admissible_fraction_kernel, grid, block, 0,
      static_cast<hipStream_t>(stream), g, rho_floor, q_floor,
      collocation_order,
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
  *host_theta = finish_host_min(scratch, n_blocks);
}

}  // namespace quasar::mhd
