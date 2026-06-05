#include "quasar/core/yee_field.hpp"
#include "quasar/numerics/shape.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/deposit_common.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

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

// kWindowFor, deposit_axis_x/y, weight_1d, and the sum-to-1 deposition-window
// guard are shared with the cylindrical deposit and live in deposit_common.hpp.
using namespace quasar::backend::pic;

// Cartesian-only fused node index (the cylindrical deposit uses the per-axis
// decomposition directly and does not need this).
__device__ inline std::size_t deposit_index(const quasar::Grid2D& g, int i, int j,
                                            bool periodic_x, bool periodic_y) {
  return g.index(deposit_axis_x(g, i, periodic_x), deposit_axis_y(g, j, periodic_y));
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

  // Reciprocal spacings computed once: g.dx()/g.dy() are themselves divisions
  // (lx/nx), and the reduced coords plus the kx/ky/inv_area scales below would
  // otherwise each re-divide by them per particle.
  const double inv_dx = 1.0 / g.dx();
  const double inv_dy = 1.0 / g.dy();

  // Reduced (cell-centred) coordinates, consistent with shape_weights_2d.
  const double gx0 = (x_prev[p] - g.origin_x) * inv_dx - 0.5;
  const double gy0 = (y_prev[p] - g.origin_y) * inv_dy - 0.5;
  const double gx1 = (x[p] - g.origin_x) * inv_dx - 0.5;
  const double gy1 = (y[p] - g.origin_y) * inv_dy - 0.5;

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
  if (!shape_strip_sums_to_one(Sx0, kWindow) || !shape_strip_sums_to_one(Sx1, kWindow) ||
      !shape_strip_sums_to_one(Sy0, kWindow) || !shape_strip_sums_to_one(Sy1, kWindow)) {
    atomicAdd(overflow, 1u);
    return;
  }

  // The destination index for window node (bx+a, by+b) is identical across the
  // Jx/Jy/Jz accumulation loops. The 2D storage index separates exactly into a
  // per-column term and a per-row term:
  //   index(ii, jj) = (ii + nghost) + pitch * (jj + nghost),
  // so precompute the wrap/clamp once per axis (2*kWindow calls, not kWindow^2)
  // and form each node index as colx[a] + rowy[b]. This shrinks the per-thread
  // table from kWindow^2 to 2*kWindow ints and removes the kWindow^2 wrap/clamp
  // recomputation, while reproducing deposit_index() bit-for-bit on both periodic
  // and wall axes. storage_size stays well under INT_MAX, so int offsets are safe.
  const int pitch = g.pitch();
  int colx[kWindow];
  int rowy[kWindow];
  for (int a = 0; a < kWindow; ++a) {
    colx[a] = deposit_axis_x(g, bx + a, periodic_x) + g.nghost;
    rowy[a] = pitch * (deposit_axis_y(g, by + a, periodic_y) + g.nghost);
  }

  // NOTE(perf): current is accumulated with scattered global atomicAdd into a
  // 6x6 window per particle (~100 atomics/particle). Particles sharing a cell
  // collide on the same addresses, so this is the typical PIC bottleneck. The
  // `run != 0.0` / `smid != 0.0` guards skip zero writes, which helps. Cutting the
  // contention itself needs per-block LDS partial accumulation or cell-binning so
  // co-located particles reduce in shared memory first — both require a
  // particle-sort/bin pass this module does not yet have, so that remains a
  // follow-up (separate from the per-axis index decomposition above, applied).
  const double qw = q * weight[p];
  const double inv_area = inv_dx * inv_dy;
  const double inv_dt = 1.0 / dt;

  // Jx: prefix sum along x for each row j. Increment per node uses the
  // x-shape change weighted by the time-centred y-shape.
  const double kx = -qw * inv_dy * inv_dt;
  for (int b = 0; b < kWindow; ++b) {
    const double sy_mid = Sy0[b] + 0.5 * (Sy1[b] - Sy0[b]);
    double run = 0.0;
    for (int a = 0; a < kWindow; ++a) {
      const double dsx = Sx1[a] - Sx0[a];
      run += kx * dsx * sy_mid;
      if (run != 0.0) {
        atomicAdd(jx + colx[a] + rowy[b], run);
      }
    }
  }

  // Jy: prefix sum along y for each column i.
  const double ky = -qw * inv_dx * inv_dt;
  for (int a = 0; a < kWindow; ++a) {
    const double sx_mid = Sx0[a] + 0.5 * (Sx1[a] - Sx0[a]);
    double run = 0.0;
    for (int b = 0; b < kWindow; ++b) {
      const double dsy = Sy1[b] - Sy0[b];
      run += ky * dsy * sx_mid;
      if (run != 0.0) {
        atomicAdd(jy + colx[a] + rowy[b], run);
      }
    }
  }

  // Jz: out-of-plane, deposited with the time-centred 2D shape as a density.
  const double qwz = qw * vz[p] * inv_area;
  for (int b = 0; b < kWindow; ++b) {
    for (int a = 0; a < kWindow; ++a) {
      const double smid = 0.5 * (Sx0[a] * Sy0[b] + Sx1[a] * Sy1[b]);
      if (smid != 0.0) {
        atomicAdd(jz + colx[a] + rowy[b], qwz * smid);
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
  // The kernel atomically bumps the persistent deposit-overflow flag if any
  // particle's displacement spills outside the deposition window. The flag is
  // accumulated across steps and read back by the solver on a cadence (and at
  // end-of-run), so there is no per-step host synchronize on the hot path — the
  // old unconditional hipStreamSynchronize here drained the GPU pipeline every
  // timestep. Overflow is a fatal "reduce dt" condition, so a delayed throw is
  // acceptable.
  hipLaunchKernelGGL((deposit_kernel<ShapeOrder>), grid, block, 0, stream,
                     g, s.size(), s.charge(), dt, periodic_x, periodic_y,
                     s.x(), s.y(), s.x_prev(), s.y_prev(),
                     s.vz(), s.weight(), s.alive(), j.jx.device_ptr(), j.jy.device_ptr(),
                     j.jz.device_ptr(), s.deposit_overflow());
  // The launch error is checked once in the extern-C wrapper below, matching the
  // single-check convention of the fdtd_b / gather_push launchers.
}

}  // namespace

QUASAR_PIC_DEPOSIT_ABI(launch_pic_deposit_shape1, 1)
QUASAR_PIC_DEPOSIT_ABI(launch_pic_deposit_shape2, 2)

// Reads back the persistent deposit-overflow flag, clears it, and throws if any
// overflow accumulated since the last check. Called by the solver on a cadence
// and at end-of-run rather than every step, keeping the host-device synchronize
// off the per-step hot path.
extern "C" void launch_pic_deposit_overflow_check(const quasar::pic::ParticleSpecies& s,
                                                  quasar_stream_t stream) {
  auto hip_stream = static_cast<hipStream_t>(stream);
  unsigned int overflow = 0;
  QUASAR_HIP_CHECK(::hipMemcpyAsync(&overflow, s.deposit_overflow(), sizeof(unsigned int),
                                    ::hipMemcpyDeviceToHost, hip_stream));
  QUASAR_HIP_CHECK(::hipStreamSynchronize(hip_stream));
  if (overflow != 0) {
    QUASAR_HIP_CHECK(::hipMemsetAsync(s.deposit_overflow(), 0, sizeof(unsigned int),
                                      hip_stream));
    QUASAR_HIP_CHECK(::hipStreamSynchronize(hip_stream));
    throw std::runtime_error{
        "PIC deposit displacement exceeded the fixed current-deposition window; "
        "reduce dt or subcycle/split the particle step"};
  }
}
