#include "quasar/core/yee_field.hpp"
#include "quasar/numerics/shape.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"

#include <hip/hip_runtime.h>

#include <stdexcept>

// Charge-conserving (Esirkepov density-decomposition) current deposition.
//
// Given a macro-particle's pre-move (x_prev) and post-move (x) positions, the
// in-plane current (Jx, Jy) is built so that the discrete continuity equation
//   d rho/dt + (Jx[i]-Jx[i-1])/dx + (Jy[j]-Jy[j-1])/dy = 0
// holds exactly on the grid (backward-difference divergence, matching the
// forward-difference curl used by the FDTD E-update so Gauss's law is preserved).
// J is a current *density*: the per-cell charge flux is divided by the cell area
// dx*dy, which the old direct scheme omitted. Jz (out-of-plane, no grid axis) is
// deposited with the time-centred shape.

namespace {

// Local stencil window sized per shape order. CFL keeps |displacement| < ~1 cell,
// so the window must span the lowest support node of the min endpoint to the
// highest support node of the max endpoint, plus a one-node margin each side.
// CIC (support 2) needs a 5-node window with base floor(min)-1; TSC (support 3)
// needs a 6-node window with base floor(min)-2. Both follow from:
//   window = ShapeOrder + 4,  base margin = ShapeOrder.
// (The TSC values match the historical fixed 6 / -2, so that path is unchanged.)
template <int ShapeOrder>
inline constexpr int kWindowFor = ShapeOrder + 4;

// Per-axis deposit indexing. On a periodic axis (both sides periodic) the node is
// wrapped exactly as before (bit-identical to the historical behaviour). On a
// non-periodic (wall) axis the node is NOT wrapped: out-of-domain shape tails land
// in the ghost layer via Grid2D::index, and are later reflected into the interior
// by the specular fold-back kernel. The clamp keeps a rare deep overshoot inside
// allocated ghost storage instead of wrapping it to the far edge.
__device__ inline int clamp_axis(int i, int lo, int hi) {
  return i < lo ? lo : (i > hi ? hi : i);
}

__device__ inline std::size_t deposit_index(const quasar::Grid2D& g, int i, int j,
                                            bool periodic_x, bool periodic_y) {
  const int ii = periodic_x ? g.wrap_i(i) : clamp_axis(i, -g.nghost, g.nx - 1 + g.nghost);
  const int jj = periodic_y ? g.wrap_j(j) : clamp_axis(j, -g.nghost, g.ny - 1 + g.nghost);
  return g.index(ii, jj);
}

template <int ShapeOrder>
__device__ inline double weight_1d(double r) {
  if constexpr (ShapeOrder == 1) {
    const double a = r < 0.0 ? -r : r;
    return a < 1.0 ? 1.0 - a : 0.0;
  } else {
    return quasar::numerics::tsc_weight_1d(r);
  }
}

template <int ShapeOrder>
__global__ void deposit_kernel(quasar::Grid2D g, std::size_t n, double q, double dt,
                               bool periodic_x, bool periodic_y,
                               const double* __restrict__ x, const double* __restrict__ y,
                               const double* __restrict__ x_prev, const double* __restrict__ y_prev,
                               const double* __restrict__ vz, const double* __restrict__ weight,
                               const unsigned char* __restrict__ alive,
                               double* __restrict__ jx, double* __restrict__ jy,
                               double* __restrict__ jz,
                               unsigned int* __restrict__ overflow) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;

  // Reduced (cell-centred) coordinates, consistent with shape_weights_2d.
  const double gx0 = (x_prev[p] - g.origin_x) / g.dx() - 0.5;
  const double gy0 = (y_prev[p] - g.origin_y) / g.dy() - 0.5;
  const double gx1 = (x[p] - g.origin_x) / g.dx() - 0.5;
  const double gy1 = (y[p] - g.origin_y) / g.dy() - 0.5;

  // Guard against non-finite reduced coords (e.g. a particle pushed to NaN by a
  // diverging field): static_cast<int>(NaN) is UB and would seed a garbage
  // window base. Skip the deposit for such particles.
  if (!isfinite(gx0) || !isfinite(gx1) || !isfinite(gy0) || !isfinite(gy1)) return;
  // Window base sits one support-radius (= ShapeOrder nodes) below the lower
  // endpoint; the window then reaches one node past the upper endpoint's support.
  constexpr int kWindow = kWindowFor<ShapeOrder>;
  const int bx = static_cast<int>(floor(gx0 < gx1 ? gx0 : gx1)) - ShapeOrder;
  const int by = static_cast<int>(floor(gy0 < gy1 ? gy0 : gy1)) - ShapeOrder;

  double Sx0[kWindow], Sx1[kWindow], Sy0[kWindow], Sy1[kWindow];
  for (int a = 0; a < kWindow; ++a) {
    const int ix = bx + a;
    const int iy = by + a;
    Sx0[a] = weight_1d<ShapeOrder>(gx0 - ix);
    Sx1[a] = weight_1d<ShapeOrder>(gx1 - ix);
    Sy0[a] = weight_1d<ShapeOrder>(gy0 - iy);
    Sy1[a] = weight_1d<ShapeOrder>(gy1 - iy);
  }

  // kWindow is sized for sub-CFL particle motion. If either endpoint shape spills
  // outside the window, continuing would silently truncate the current path and
  // violate discrete continuity. Flag it and let the host throw a clear
  // "reduce dt / split the step" error instead of returning corrupt current.
  double sx0_sum = 0.0, sx1_sum = 0.0, sy0_sum = 0.0, sy1_sum = 0.0;
  for (int a = 0; a < kWindow; ++a) {
    sx0_sum += Sx0[a];
    sx1_sum += Sx1[a];
    sy0_sum += Sy0[a];
    sy1_sum += Sy1[a];
  }
  constexpr double kWeightTol = 1.0e-10;
  if (fabs(sx0_sum - 1.0) > kWeightTol || fabs(sx1_sum - 1.0) > kWeightTol ||
      fabs(sy0_sum - 1.0) > kWeightTol || fabs(sy1_sum - 1.0) > kWeightTol) {
    atomicAdd(overflow, 1u);
    return;
  }

  // The destination index for window node (bx+a, by+b) is identical across the
  // Jx/Jy/Jz accumulation loops, so compute the wrap/clamp + linear index once.
  std::size_t off[kWindow][kWindow];
  for (int a = 0; a < kWindow; ++a) {
    for (int b = 0; b < kWindow; ++b) {
      off[a][b] = deposit_index(g, bx + a, by + b, periodic_x, periodic_y);
    }
  }

  // NOTE(perf): current is accumulated with scattered global atomicAdd into a
  // 6x6 window per particle (~100 atomics/particle). Particles sharing a cell
  // collide on the same addresses, so this is the typical PIC bottleneck. The
  // `run != 0.0` / `smid != 0.0` guards skip zero writes, which helps; a future
  // optimization is per-block shared-memory partial accumulation or cell-binning
  // to reduce cross-thread address contention. Left as a follow-up.
  const double qw = q * weight[p];
  const double inv_area = 1.0 / (g.dx() * g.dy());

  // Jx: prefix sum along x for each row j. Increment per node uses the
  // x-shape change weighted by the time-centred y-shape.
  const double kx = -qw / (g.dy() * dt);
  for (int b = 0; b < kWindow; ++b) {
    const double sy_mid = Sy0[b] + 0.5 * (Sy1[b] - Sy0[b]);
    double run = 0.0;
    for (int a = 0; a < kWindow; ++a) {
      const double dsx = Sx1[a] - Sx0[a];
      run += kx * dsx * sy_mid;
      if (run != 0.0) {
        atomicAdd(jx + off[a][b], run);
      }
    }
  }

  // Jy: prefix sum along y for each column i.
  const double ky = -qw / (g.dx() * dt);
  for (int a = 0; a < kWindow; ++a) {
    const double sx_mid = Sx0[a] + 0.5 * (Sx1[a] - Sx0[a]);
    double run = 0.0;
    for (int b = 0; b < kWindow; ++b) {
      const double dsy = Sy1[b] - Sy0[b];
      run += ky * dsy * sx_mid;
      if (run != 0.0) {
        atomicAdd(jy + off[a][b], run);
      }
    }
  }

  // Jz: out-of-plane, deposited with the time-centred 2D shape as a density.
  const double qwz = qw * vz[p] * inv_area;
  for (int b = 0; b < kWindow; ++b) {
    for (int a = 0; a < kWindow; ++a) {
      const double smid = 0.5 * (Sx0[a] * Sy0[b] + Sx1[a] * Sy1[b]);
      if (smid != 0.0) {
        atomicAdd(jz + off[a][b], qwz * smid);
      }
    }
  }
}

template <int ShapeOrder>
void launch(const quasar::Grid2D& g, const quasar::pic::ParticleSpecies& s,
            quasar::JField2D<double>& j, double dt, bool periodic_x, bool periodic_y,
            hipStream_t stream) {
  if (s.size() == 0) return;
  const dim3 block(256);
  const dim3 grid((s.size() + block.x - 1) / block.x);
  QUASAR_HIP_CHECK(::hipMemsetAsync(s.compact_counter(), 0, sizeof(unsigned int), stream));
  hipLaunchKernelGGL((deposit_kernel<ShapeOrder>), grid, block, 0, stream,
                     g, s.size(), s.charge(), dt, periodic_x, periodic_y,
                     s.x(), s.y(), s.x_prev(), s.y_prev(),
                     s.vz(), s.weight(), s.alive(), j.jx.device_ptr(), j.jy.device_ptr(),
                     j.jz.device_ptr(), s.compact_counter());
  QUASAR_HIP_CHECK(::hipGetLastError());
  unsigned int overflow = 0;
  QUASAR_HIP_CHECK(::hipMemcpyAsync(&overflow, s.compact_counter(), sizeof(unsigned int),
                                    ::hipMemcpyDeviceToHost, stream));
  QUASAR_HIP_CHECK(::hipStreamSynchronize(stream));
  if (overflow != 0) {
    throw std::runtime_error{
        "PIC deposit displacement exceeded the fixed current-deposition window; "
        "reduce dt or subcycle/split the particle step"};
  }
}

}  // namespace

extern "C" void launch_pic_deposit_shape1(const quasar::Grid2D& g,
                                          const quasar::pic::ParticleSpecies& s,
                                          quasar::JField2D<double>& j,
                                          double dt, int periodic_x, int periodic_y,
                                          quasar_stream_t stream) {
  launch<1>(g, s, j, dt, periodic_x != 0, periodic_y != 0,
            static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_deposit_shape2(const quasar::Grid2D& g,
                                          const quasar::pic::ParticleSpecies& s,
                                          quasar::JField2D<double>& j,
                                          double dt, int periodic_x, int periodic_y,
                                          quasar_stream_t stream) {
  launch<2>(g, s, j, dt, periodic_x != 0, periodic_y != 0,
            static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}
