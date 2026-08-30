// Device reductions for the PIC diagnostics.
//
// These replace host loops that downloaded all six field components and every
// particle record on each diagnostic interval and reduced them in long double
// (the former pic/diagnostics.cpp). long double has no device equivalent, so
// the port is not bit-for-bit; it is held to the standard set by the
// Grad-Shafranov port instead -- deterministic, compensated, and measured
// against a long-double oracle in the equivalence tests, which assert the
// device error does not exceed a naive host sum's error.
//
// The dynamic-range problem is the interesting part. A kinetic-energy term is
// 0.5*m*|v|^2*w with m ~ 1e-31 and w ~ 1e10; a cylindrical field-energy term
// carries a 2*pi*r*dr*dz volume. Multiplying those out in double underflows
// before the sum ever happens. Every term is therefore decomposed with frexp
// and accumulated in a shared binary exponent frame -- see
// backend/hip/scaled_reduction_detail.hpp, which is the device form of the
// host ScaledPositiveSum this replaces.
//
// What stays on the host: turning the returned (sum, correction, exponent)
// triple into a Real, and choosing which exception to throw. That epilogue
// reads three scalars regardless of grid or particle count.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/yee_field.hpp"
#include "quasar/numerics/stencil.hpp"
#include "quasar/physics/pic/kernels.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/reduction_detail.hpp"
#include "backend/hip/scaled_reduction_detail.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace quasar::pic {
namespace {

namespace hd = ::quasar::backend::hip_detail;

using hd::ScaledAcc;
using hd::ScaledTerm;
using hd::scaled_acc_add;
using hd::scaled_acc_merge;
using hd::scaled_product;
using hd::scaled_square_times;

inline constexpr unsigned kThreads = hd::kReduceThreads;
inline constexpr unsigned kMaxBlocks = hd::kReduceMaxBlocks;

// Convert the device accumulator to the public POD. Field-for-field rather
// than a reinterpret_cast: the two types are deliberately independent so
// neither header has to know the other's layout.
PicScaledSum to_public(const ScaledAcc& a) {
  PicScaledSum out;
  out.sum = a.sum;
  out.correction = a.correction;
  out.exponent = a.exponent;
  out.initialized = a.initialized;
  out.invalid = a.invalid;
  out.nonfinite_input = a.nonfinite_input;
  return out;
}

// Second pass: one block folds the first pass's block partials. Shared by all
// three reductions.
__global__ QUASAR_LAUNCH_BOUNDS_256 void scaled_pass2(
    const ScaledAcc* __restrict__ partial, ScaledAcc* __restrict__ out,
    unsigned n_partial) {
  __shared__ ScaledAcc sdata[kThreads];
  const unsigned tid = threadIdx.x;

  ScaledAcc a{};
  for (unsigned k = tid; k < n_partial; k += kThreads) {
    scaled_acc_merge(a, partial[k]);
  }
  sdata[tid] = a;
  __syncthreads();

  hd::block_scaled_acc_reduce<kThreads>(sdata, tid);
  if (tid == 0) out[0] = sdata[0];
}

// -- Kinetic energy ----------------------------------------------------------

__global__ QUASAR_LAUNCH_BOUNDS_256 void kinetic_energy_pass1(
    std::size_t n, const Real* __restrict__ vx, const Real* __restrict__ vy,
    const Real* __restrict__ vz, const Real* __restrict__ weight,
    const std::uint8_t* __restrict__ alive, Real mass,
    ScaledAcc* __restrict__ partial) {
  __shared__ ScaledAcc sdata[kThreads];
  const unsigned tid = threadIdx.x;

  ScaledAcc a{};
  for (std::size_t p = blockIdx.x * blockDim.x + tid; p < n;
       p += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (alive[p] == 0) continue;
    // hypot(hypot(vx,vy),vz) rather than sqrt(vx^2+vy^2+vz^2): the squares of
    // a relativistic-normalized velocity are fine, but this matches the host
    // formula exactly and does not overflow for large gamma*v.
    const Real speed = hypot(hypot(vx[p], vy[p]), vz[p]);
    const Real factors[] = {Real{0.5}, mass, speed, speed, weight[p]};
    scaled_acc_add(a, scaled_product(factors, 5));
  }
  sdata[tid] = a;
  __syncthreads();

  hd::block_scaled_acc_reduce<kThreads>(sdata, tid);
  if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// -- Net and absolute charge -------------------------------------------------

__global__ QUASAR_LAUNCH_BOUNDS_256 void total_charge_pass1(
    std::size_t n, const Real* __restrict__ weight,
    const std::uint8_t* __restrict__ alive, Real charge,
    ScaledAcc* __restrict__ partial_net,
    ScaledAcc* __restrict__ partial_absolute) {
  __shared__ ScaledAcc s_net[kThreads];
  __shared__ ScaledAcc s_absolute[kThreads];
  const unsigned tid = threadIdx.x;

  ScaledAcc net{};
  ScaledAcc absolute{};
  for (std::size_t p = blockIdx.x * blockDim.x + tid; p < n;
       p += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    if (alive[p] == 0) continue;
    const Real factors[] = {charge, weight[p]};
    ScaledTerm term = hd::scaled_signed_product(factors, 2);
    scaled_acc_add(net, term);
    term.mantissa = fabs(term.mantissa);
    scaled_acc_add(absolute, term);
  }
  s_net[tid] = net;
  s_absolute[tid] = absolute;
  __syncthreads();

  hd::block_scaled_acc_reduce<kThreads>(s_net, tid);
  hd::block_scaled_acc_reduce<kThreads>(s_absolute, tid);
  if (tid == 0) {
    partial_net[blockIdx.x] = s_net[0];
    partial_absolute[blockIdx.x] = s_absolute[0];
  }
}

// -- Field energy ------------------------------------------------------------

// Region layout for the staggered sweep. Each staggered component owns a
// different index range, so the reduction enumerates a concatenation of four
// regions and decodes the region from the flat index. The concatenation order
// is fixed, which is what keeps the fold deterministic.
struct FieldEnergyPlan {
  std::size_t end_a{0};
  std::size_t end_b{0};
  std::size_t end_c{0};
  std::size_t end_d{0};
  int nx{0};
  int ny{0};
  int x_faces{0};
  int y_faces{0};
  int width_a{0};
  int width_b{0};
  int width_c{0};
  int width_d{0};
  int periodic_x{0};
  int periodic_y{0};
  int cylindrical{0};
};

struct FieldPlanes {
  const Real* ex;
  const Real* ey;
  const Real* ez;
  const Real* bx;
  const Real* by;
  const Real* bz;
};

// Half-width control volume on a non-periodic edge face, full width otherwise.
__device__ inline Real cart_x(const FieldEnergyPlan& p, Real dx, int i) {
  if (p.periodic_x) return dx;
  return (i == 0 || i == p.nx) ? Real{0.5} * dx : dx;
}

__device__ inline Real cart_y(const FieldEnergyPlan& p, Real dy, int j) {
  if (p.periodic_y) return dy;
  return (j == 0 || j == p.ny) ? Real{0.5} * dy : dy;
}

__device__ inline void add_checked(ScaledAcc& a, Real value,
                                   const Real* factors, int n) {
  if (!isfinite(value)) {
    a.nonfinite_input = true;
    return;
  }
  scaled_acc_add(a, scaled_square_times(value, factors, n));
}

// Radial face dual annulus [midpoint_left, midpoint_right], clipped to the
// physical domain. pi*width*(2*r0 + radial_offset) is kept as two positive
// products so neither a large 2*r0 nor the full area is ever materialized --
// that is what preserves a thin annulus at a large radius.
__device__ inline void add_radial_face(ScaledAcc& a, const FieldEnergyPlan& p,
                                       Real value, int i, Real dx, Real r0,
                                       Real dz) {
  if (!isfinite(value)) {
    a.nonfinite_input = true;
    return;
  }
  const Real pi = pi_v<Real>;
  const Real width = (i == 0 || i == p.nx) ? Real{0.5} * dx : dx;
  const Real radial_offset =
      (i == 0) ? Real{0.5} * dx
               : (i == p.nx)
                     ? (Real{2} * static_cast<Real>(p.nx) - Real{0.5}) * dx
                     : Real{2} * static_cast<Real>(i) * dx;
  const Real first[] = {pi, width, Real{2}, r0, dz};
  scaled_acc_add(a, scaled_square_times(value, first, 5));
  const Real second[] = {pi, width, radial_offset, dz};
  scaled_acc_add(a, scaled_square_times(value, second, 4));
}

// 2*pi*dx*(r0 + offset*dx), likewise split into positive scaled products.
__device__ inline void add_radial_cell(ScaledAcc& a, Real value, int i, Real dx,
                                       Real r0, Real dz) {
  if (!isfinite(value)) {
    a.nonfinite_input = true;
    return;
  }
  const Real pi = pi_v<Real>;
  const Real offset = static_cast<Real>(i) + Real{0.5};
  const Real first[] = {Real{2}, pi, dx, r0, dz};
  scaled_acc_add(a, scaled_square_times(value, first, 5));
  const Real second[] = {Real{2}, pi, dx, offset, dx, dz};
  scaled_acc_add(a, scaled_square_times(value, second, 6));
}

__global__ QUASAR_LAUNCH_BOUNDS_256 void field_energy_pass1(
    Grid2D grid, FieldEnergyPlan plan, FieldPlanes f,
    ScaledAcc* __restrict__ partial) {
  __shared__ ScaledAcc sdata[kThreads];
  const unsigned tid = threadIdx.x;

  const Real dx = grid.dx();
  const Real dy = grid.dy();
  const Real r0 = grid.origin_x;

  ScaledAcc a{};
  for (std::size_t t = blockIdx.x * blockDim.x + tid; t < plan.end_d;
       t += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    int i = 0;
    int j = 0;
    int region = 0;
    if (t < plan.end_a) {
      region = 0;
      i = static_cast<int>(t % static_cast<std::size_t>(plan.width_a));
      j = static_cast<int>(t / static_cast<std::size_t>(plan.width_a));
    } else if (t < plan.end_b) {
      const std::size_t u = t - plan.end_a;
      region = 1;
      i = static_cast<int>(u % static_cast<std::size_t>(plan.width_b));
      j = static_cast<int>(u / static_cast<std::size_t>(plan.width_b));
    } else if (t < plan.end_c) {
      const std::size_t u = t - plan.end_b;
      region = 2;
      i = static_cast<int>(u % static_cast<std::size_t>(plan.width_c));
      j = static_cast<int>(u / static_cast<std::size_t>(plan.width_c));
    } else {
      const std::size_t u = t - plan.end_c;
      region = 3;
      i = static_cast<int>(u % static_cast<std::size_t>(plan.width_d));
      j = static_cast<int>(u / static_cast<std::size_t>(plan.width_d));
    }
    const std::size_t idx = grid.index(i, j);

    if (!plan.cylindrical) {
      switch (region) {
        case 0: {  // Ex, By on radial faces: j in [0,ny), i in [0,x_faces)
          const Real w[] = {cart_x(plan, dx, i), dy};
          add_checked(a, f.ex[idx], w, 2);
          add_checked(a, f.by[idx], w, 2);
          break;
        }
        case 1: {  // Ey, Bx: j in [0,y_faces), i in [0,nx)
          const Real w[] = {dx, cart_y(plan, dy, j)};
          add_checked(a, f.ey[idx], w, 2);
          add_checked(a, f.bx[idx], w, 2);
          break;
        }
        case 2: {  // Ez at cell centres
          const Real w[] = {dx, dy};
          add_checked(a, f.ez[idx], w, 2);
          break;
        }
        default: {  // Bz at nodes
          const Real w[] = {cart_x(plan, dx, i), cart_y(plan, dy, j)};
          add_checked(a, f.bz[idx], w, 2);
          break;
        }
      }
    } else {
      switch (region) {
        case 0:  // Er, Ephi on all nx+1 radial faces (never periodic)
          add_radial_face(a, plan, f.ex[idx], i, dx, r0, dy);
          add_radial_face(a, plan, f.ez[idx], i, dx, r0, dy);
          break;
        case 1:  // Ez at cell centres
          add_radial_cell(a, f.ey[idx], i, dx, r0, cart_y(plan, dy, j));
          break;
        case 2:  // Br, Bphi on radial faces
          add_radial_face(a, plan, f.bx[idx], i, dx, r0, cart_y(plan, dy, j));
          add_radial_face(a, plan, f.bz[idx], i, dx, r0, cart_y(plan, dy, j));
          break;
        default:  // Bz at cell centres
          add_radial_cell(a, f.by[idx], i, dx, r0, dy);
          break;
      }
    }
  }
  sdata[tid] = a;
  __syncthreads();

  hd::block_scaled_acc_reduce<kThreads>(sdata, tid);
  if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// -- Gauss-law residual ------------------------------------------------------

struct GaussPlan {
  int fdtd_order{2};
  int periodic_x{0};
  int periodic_y{0};
  int cylindrical{0};
};

// Periodic axes wrap; wall axes read their maintained halo, clamped to the
// padded storage exactly as the host `at` lambda did.
__device__ inline Real sample(const Real* f, const Grid2D& g,
                              const GaussPlan& p, int i, int j) {
  const int ii = p.periodic_x
                     ? g.wrap_i(i)
                     : max(-g.nghost, min(g.nx - 1 + g.nghost, i));
  const int jj = p.periodic_y
                     ? g.wrap_j(j)
                     : max(-g.nghost, min(g.ny - 1 + g.nghost, j));
  return f[g.index(ii, jj)];
}

__device__ inline numerics::detail::ScaledValue dx_fwd(
    const Real* f, const Grid2D& g, const GaussPlan& p, int i, int j) {
  if (p.fdtd_order == 4) {
    return numerics::staggered_derivative_scaled_values<4>(
        sample(f, g, p, i - 1, j), sample(f, g, p, i, j),
        sample(f, g, p, i + 1, j), sample(f, g, p, i + 2, j), g.dx());
  }
  return numerics::staggered_derivative_scaled_values<2>(
      Real{0}, sample(f, g, p, i, j), sample(f, g, p, i + 1, j), Real{0},
      g.dx());
}

__device__ inline numerics::detail::ScaledValue dy_fwd(
    const Real* f, const Grid2D& g, const GaussPlan& p, int i, int j) {
  if (p.fdtd_order == 4) {
    return numerics::staggered_derivative_scaled_values<4>(
        sample(f, g, p, i, j - 1), sample(f, g, p, i, j),
        sample(f, g, p, i, j + 1), sample(f, g, p, i, j + 2), g.dy());
  }
  return numerics::staggered_derivative_scaled_values<2>(
      Real{0}, sample(f, g, p, i, j), sample(f, g, p, i, j + 1), Real{0},
      g.dy());
}

__global__ QUASAR_LAUNCH_BOUNDS_256 void gauss_residual_pass1(
    Grid2D grid, GaussPlan plan, const Real* __restrict__ ex,
    const Real* __restrict__ ey, const Real* __restrict__ rho,
    ScaledAcc* __restrict__ partial_square,
    ScaledAcc* __restrict__ partial_volume) {
  __shared__ ScaledAcc s_square[kThreads];
  __shared__ ScaledAcc s_volume[kThreads];
  const unsigned tid = threadIdx.x;

  const std::size_t n_cells = static_cast<std::size_t>(grid.nx)
                            * static_cast<std::size_t>(grid.ny);
  const Real dx = grid.dx();
  const Real dy = grid.dy();
  const Real r0 = grid.origin_x;
  const Real pi = pi_v<Real>;

  ScaledAcc square{};
  ScaledAcc volume{};
  for (std::size_t t = blockIdx.x * blockDim.x + tid; t < n_cells;
       t += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const int i = static_cast<int>(t % static_cast<std::size_t>(grid.nx));
    const int j = static_cast<int>(t / static_cast<std::size_t>(grid.nx));

    Real residual;
    if (plan.cylindrical) {
      numerics::detail::ScaledValue radial;
      if (plan.fdtd_order == 4) {
        radial = numerics::cylindrical_radial_flux_scaled_values<4>(
            sample(ex, grid, plan, i - 1, j), sample(ex, grid, plan, i, j),
            sample(ex, grid, plan, i + 1, j), sample(ex, grid, plan, i + 2, j),
            grid.r_at_cell_center(i), grid.dx());
      } else {
        radial = numerics::cylindrical_radial_flux_scaled_values<2>(
            Real{0}, sample(ex, grid, plan, i, j),
            sample(ex, grid, plan, i + 1, j), Real{0},
            grid.r_at_cell_center(i), grid.dx());
      }
      residual = numerics::detail::scaled_signed_sum(
          radial, dy_fwd(ey, grid, plan, i, j),
          numerics::detail::negate(
              numerics::detail::scaled_value(rho[grid.index(i, j)])));
    } else {
      residual = numerics::detail::scaled_signed_sum(
          dx_fwd(ex, grid, plan, i, j), dy_fwd(ey, grid, plan, i, j),
          numerics::detail::negate(
              numerics::detail::scaled_value(rho[grid.index(i, j)])));
    }

    if (!isfinite(residual)) {
      square.nonfinite_input = true;
      continue;
    }

    if (plan.cylindrical) {
      const Real offset = static_cast<Real>(i) + Real{0.5};
      // 2*pi*dx*dy*(r0 + offset*dx), split so a thin annulus at a large
      // radius survives without forming either the radius or the volume.
      const Real inner[] = {Real{2}, pi, dx, dy, r0};
      const Real outer[] = {Real{2}, pi, dx, dy, offset, dx};
      scaled_acc_add(square, scaled_square_times(residual, inner, 5));
      scaled_acc_add(square, scaled_square_times(residual, outer, 6));
      scaled_acc_add(volume, scaled_product(inner, 5));
      scaled_acc_add(volume, scaled_product(outer, 6));
    } else {
      const Real w[] = {dx, dy};
      scaled_acc_add(square, scaled_square_times(residual, w, 2));
      scaled_acc_add(volume, scaled_product(w, 2));
    }
  }
  s_square[tid] = square;
  s_volume[tid] = volume;
  __syncthreads();

  hd::block_scaled_acc_reduce<kThreads>(s_square, tid);
  hd::block_scaled_acc_reduce<kThreads>(s_volume, tid);
  if (tid == 0) {
    partial_square[blockIdx.x] = s_square[0];
    partial_volume[blockIdx.x] = s_volume[0];
  }
}

// Diagnostics run at output intervals, not per step, so a per-call scratch
// allocation is cheaper than holding kMaxBlocks accumulators resident for the
// life of every solver.
struct ReduceScratch {
  explicit ReduceScratch(unsigned blocks)
      : partial{blocks}, result{1} {}
  backend::DeviceBuffer<ScaledAcc> partial;
  backend::DeviceBuffer<ScaledAcc> result;
};

PicScaledSum finish(ReduceScratch& scratch, unsigned blocks,
                    hipStream_t stream) {
  hipLaunchKernelGGL(scaled_pass2, dim3{1}, dim3{kThreads}, 0, stream,
                     scratch.partial.device_ptr(), scratch.result.device_ptr(),
                     blocks);
  QUASAR_HIP_CHECK(hipGetLastError());
  QUASAR_HIP_CHECK(hipStreamSynchronize(stream));
  ScaledAcc host{};
  scratch.result.copy_to_host(&host, 1);
  return to_public(host);
}

}  // namespace

PicScaledSum launch_pic_kinetic_energy(const ParticleSpecies& species,
                                       backend::stream_t stream_in) {
  const auto stream = static_cast<hipStream_t>(stream_in);
  const std::size_t n = species.size();
  if (n == 0) return PicScaledSum{};

  const unsigned blocks = hd::reduce_block_count(n);
  ReduceScratch scratch{blocks};
  hipLaunchKernelGGL(kinetic_energy_pass1, dim3{blocks}, dim3{kThreads}, 0,
                     stream, n, species.vx(), species.vy(), species.vz(),
                     species.weight(), species.alive(), species.mass(),
                     scratch.partial.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());
  return finish(scratch, blocks, stream);
}

PicChargeTotals launch_pic_total_charge(const ParticleSpecies& species,
                                        backend::stream_t stream_in) {
  const auto stream = static_cast<hipStream_t>(stream_in);
  const std::size_t n = species.size();
  if (n == 0) return PicChargeTotals{};

  const unsigned blocks = hd::reduce_block_count(n);
  ReduceScratch net{blocks};
  ReduceScratch absolute{blocks};
  hipLaunchKernelGGL(total_charge_pass1, dim3{blocks}, dim3{kThreads}, 0,
                     stream, n, species.weight(), species.alive(),
                     species.charge(), net.partial.device_ptr(),
                     absolute.partial.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  PicChargeTotals out;
  out.net = finish(net, blocks, stream);
  out.absolute = finish(absolute, blocks, stream);
  return out;
}

PicScaledSum launch_pic_field_energy(const YeeField2D<Real>& fields,
                                     Grid2D grid, int periodic_x,
                                     int periodic_y, int cylindrical,
                                     backend::stream_t stream_in) {
  const auto stream = static_cast<hipStream_t>(stream_in);

  FieldEnergyPlan plan{};
  plan.nx = grid.nx;
  plan.ny = grid.ny;
  plan.periodic_x = periodic_x;
  plan.periodic_y = periodic_y;
  plan.cylindrical = cylindrical;
  plan.x_faces = periodic_x ? grid.nx : grid.nx + 1;
  plan.y_faces = periodic_y ? grid.ny : grid.ny + 1;

  int height_a = 0;
  int height_b = 0;
  int height_c = 0;
  int height_d = 0;
  if (!cylindrical) {
    plan.width_a = plan.x_faces;  height_a = grid.ny;
    plan.width_b = grid.nx;       height_b = plan.y_faces;
    plan.width_c = grid.nx;       height_c = grid.ny;
    plan.width_d = plan.x_faces;  height_d = plan.y_faces;
  } else {
    // The radial axis is never periodic in cylindrical geometry: all nx+1
    // radial faces are physical, the axis face included.
    plan.width_a = grid.nx + 1;   height_a = grid.ny;
    plan.width_b = grid.nx;       height_b = plan.y_faces;
    plan.width_c = grid.nx + 1;   height_c = plan.y_faces;
    plan.width_d = grid.nx;       height_d = grid.ny;
  }
  const auto extent = [](int w, int h) {
    return static_cast<std::size_t>(w > 0 ? w : 0)
         * static_cast<std::size_t>(h > 0 ? h : 0);
  };
  plan.end_a = extent(plan.width_a, height_a);
  plan.end_b = plan.end_a + extent(plan.width_b, height_b);
  plan.end_c = plan.end_b + extent(plan.width_c, height_c);
  plan.end_d = plan.end_c + extent(plan.width_d, height_d);
  if (plan.end_d == 0) return PicScaledSum{};

  FieldPlanes planes{fields.ex.device_ptr(), fields.ey.device_ptr(),
                     fields.ez.device_ptr(), fields.bx.device_ptr(),
                     fields.by.device_ptr(), fields.bz.device_ptr()};

  const unsigned blocks = hd::reduce_block_count(plan.end_d);
  ReduceScratch scratch{blocks};
  hipLaunchKernelGGL(field_energy_pass1, dim3{blocks}, dim3{kThreads}, 0,
                     stream, grid, plan, planes, scratch.partial.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());
  return finish(scratch, blocks, stream);
}

PicGaussResidualSums launch_pic_gauss_residual(
    const YeeField2D<Real>& fields, const ScalarGrid2D<Real>& charge,
    int fdtd_order, int periodic_x, int periodic_y, int cylindrical,
    backend::stream_t stream_in) {
  const auto stream = static_cast<hipStream_t>(stream_in);
  const Grid2D& grid = fields.grid;
  const std::size_t n_cells = static_cast<std::size_t>(grid.nx)
                            * static_cast<std::size_t>(grid.ny);
  if (n_cells == 0) return PicGaussResidualSums{};

  GaussPlan plan{fdtd_order, periodic_x, periodic_y, cylindrical};
  const unsigned blocks = hd::reduce_block_count(n_cells);
  ReduceScratch square{blocks};
  ReduceScratch volume{blocks};

  hipLaunchKernelGGL(gauss_residual_pass1, dim3{blocks}, dim3{kThreads}, 0,
                     stream, grid, plan, fields.ex.device_ptr(),
                     fields.ey.device_ptr(), charge.values.device_ptr(),
                     square.partial.device_ptr(), volume.partial.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  PicGaussResidualSums out;
  out.weighted_square = finish(square, blocks, stream);
  out.volume = finish(volume, blocks, stream);
  return out;
}

}  // namespace quasar::pic
