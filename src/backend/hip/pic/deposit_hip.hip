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
//   d rho/dt + (Jx[i+1]-Jx[i])/dx + (Jy[j+1]-Jy[j])/dy = 0
// holds exactly on the charge grid. This is the forward Yee divergence that
// annihilates the forward-difference Ampere curl. A prefix associated with cell
// `a` is therefore stored on its upper face `a+1`.
// J is a current *density*: the per-cell charge flux is divided by the cell area
// dx*dy, which the old direct scheme omitted. Jz (out-of-plane, no grid axis) is
// deposited with the time-centred shape.

namespace {

// kWindowFor, deposit_axis_x/y, weight_1d, and the sum-to-1 deposition-window
// guard are shared with the cylindrical deposit and live in deposit_common.hpp.
using namespace quasar::backend::pic;

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
  const double gx0 = quasar::detail::scaled_difference_quotient(
      x_prev[p], g.origin_x, g.dx()) - 0.5;
  const double gy0 = quasar::detail::scaled_difference_quotient(
      y_prev[p], g.origin_y, g.dy()) - 0.5;
  const double gx1 = quasar::detail::scaled_difference_quotient(
      x[p], g.origin_x, g.dx()) - 0.5;
  const double gy1 = quasar::detail::scaled_difference_quotient(
      y[p], g.origin_y, g.dy()) - 0.5;

  // Casting a non-finite or out-of-int-range floor is undefined. Signal the
  // persistent fatal flag rather than silently dropping charge/current.
  if (!reduced_coordinate_representable(gx0)
      || !reduced_coordinate_representable(gx1)
      || !reduced_coordinate_representable(gy0)
      || !reduced_coordinate_representable(gy1)) {
    mark_deposit_error(overflow);
    return;
  }
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
    mark_deposit_error(overflow);
    return;
  }

  // The destination index for window node (bx+a, by+b) is identical across the
  // Jx/Jy/Jz accumulation loops. The 2D storage index separates exactly into a
  // per-column term and a per-row term:
  //   index(ii, jj) = (ii + nghost) + pitch * (jj + nghost),
  // so precompute the wrapped or storage-bounded index once per axis
  // and form each node index as colx[a] + rowy[b]. This shrinks the per-thread
  // table from kWindow^2 to 2*kWindow ints and removes the kWindow^2 wrap/clamp
  // recomputation; colx[a] + rowy[b] equals g.index(deposit_axis_x(g, bx+a, ...),
  // on both periodic and wall axes. Validity bits prevent a shape tail beyond
  // allocated ghost storage from being spuriously accumulated into a clamped
  // edge slot. storage_size stays well under INT_MAX, so int offsets are safe.
  const int pitch = g.pitch();
  int colx[kWindow];
  int colx_face[kWindow];
  int rowy[kWindow];
  int rowy_face[kWindow];
  bool colx_valid[kWindow];
  bool colx_face_valid[kWindow];
  bool rowy_valid[kWindow];
  bool rowy_face_valid[kWindow];
  for (int a = 0; a < kWindow; ++a) {
    const int ix = bx + a;
    colx_valid[a] = periodic_x
        || (ix >= -g.nghost && ix <= g.nx - 1 + g.nghost);
    colx[a] = (periodic_x ? g.wrap_i(ix)
                          : deposit_clamp_axis(ix, -g.nghost,
                                               g.nx - 1 + g.nghost))
            + g.nghost;
    const int fx = bx + a + 1;
    colx_face_valid[a] = periodic_x
        || (fx >= -g.nghost && fx <= g.nx + g.nghost - 1);
    colx_face[a] = (periodic_x ? g.wrap_i(fx)
                               : deposit_clamp_axis(fx, -g.nghost,
                                                    g.nx + g.nghost - 1))
                   + g.nghost;
    const int iy = by + a;
    rowy_valid[a] = periodic_y
        || (iy >= -g.nghost && iy <= g.ny - 1 + g.nghost);
    rowy[a] = pitch * ((periodic_y ? g.wrap_j(iy)
                                   : deposit_clamp_axis(iy, -g.nghost,
                                                        g.ny - 1 + g.nghost))
                       + g.nghost);
    const int fy = by + a + 1;
    rowy_face_valid[a] = periodic_y
        || (fy >= -g.nghost && fy <= g.ny + g.nghost - 1);
    rowy_face[a] = pitch * ((periodic_y ? g.wrap_j(fy)
                                        : deposit_clamp_axis(
                                              fy, -g.nghost,
                                              g.ny + g.nghost - 1))
                            + g.nghost);
  }

  // NOTE(perf): current is accumulated with scattered checked atomic additions
  // into a 6x6 window per particle (~100 atomics/particle). Particles sharing a
  // cell collide on the same addresses, so this is the typical PIC bottleneck. The
  // `run != 0.0` / `smid != 0.0` guards skip zero writes, which helps. Cutting the
  // contention itself needs per-block LDS partial accumulation or cell-binning so
  // co-located particles reduce in shared memory first — both require a
  // particle-sort/bin pass this module does not yet have, so that remains a
  // follow-up (separate from the per-axis index decomposition above, applied).
  ScaledProduct kx_scaled;
  scaled_multiply(kx_scaled, -q);
  scaled_multiply(kx_scaled, weight[p]);
  scaled_divide(kx_scaled, g.dy());
  scaled_divide(kx_scaled, dt);

  ScaledProduct ky_scaled;
  scaled_multiply(ky_scaled, -q);
  scaled_multiply(ky_scaled, weight[p]);
  scaled_divide(ky_scaled, g.dx());
  scaled_divide(ky_scaled, dt);

  ScaledProduct jz_scaled;
  scaled_multiply(jz_scaled, q);
  scaled_multiply(jz_scaled, weight[p]);
  scaled_multiply(jz_scaled, vz[p]);
  scaled_divide(jz_scaled, g.dx());
  scaled_divide(jz_scaled, g.dy());

  // Jx: prefix sum along x for each row j. Increment per node uses the
  // x-shape change weighted by the time-centred y-shape.
  for (int b = 0; b < kWindow; ++b) {
    const double sy_mid = Sy0[b] + 0.5 * (Sy1[b] - Sy0[b]);
    double run = 0.0;
    for (int a = 0; a < kWindow; ++a) {
      const double dsx = Sx1[a] - Sx0[a];
      ScaledProduct increment_scaled = kx_scaled;
      scaled_multiply(increment_scaled, dsx);
      scaled_multiply(increment_scaled, sy_mid);
      const double increment = scaled_value(increment_scaled);
      if (!scaled_value_representable(increment_scaled, increment)) {
        mark_deposit_error(overflow);
        return;
      }
      run += increment;
      if (!isfinite(run)) {
        mark_deposit_error(overflow);
        return;
      }
      if (run != 0.0 && colx_face_valid[a] && rowy_valid[b]) {
        checked_atomic_add(jx + colx_face[a] + rowy[b], run, overflow);
      }
    }
  }

  // Jy: prefix sum along y for each column i.
  for (int a = 0; a < kWindow; ++a) {
    const double sx_mid = Sx0[a] + 0.5 * (Sx1[a] - Sx0[a]);
    double run = 0.0;
    for (int b = 0; b < kWindow; ++b) {
      const double dsy = Sy1[b] - Sy0[b];
      ScaledProduct increment_scaled = ky_scaled;
      scaled_multiply(increment_scaled, dsy);
      scaled_multiply(increment_scaled, sx_mid);
      const double increment = scaled_value(increment_scaled);
      if (!scaled_value_representable(increment_scaled, increment)) {
        mark_deposit_error(overflow);
        return;
      }
      run += increment;
      if (!isfinite(run)) {
        mark_deposit_error(overflow);
        return;
      }
      if (run != 0.0 && colx_valid[a] && rowy_face_valid[b]) {
        checked_atomic_add(jy + colx[a] + rowy_face[b], run, overflow);
      }
    }
  }

  // Jz: out-of-plane, deposited with the time-centred 2D shape as a density.
  for (int b = 0; b < kWindow; ++b) {
    for (int a = 0; a < kWindow; ++a) {
      const double smid = 0.5 * (Sx0[a] * Sy0[b] + Sx1[a] * Sy1[b]);
      ScaledProduct value_scaled = jz_scaled;
      scaled_multiply(value_scaled, smid);
      const double value = scaled_value(value_scaled);
      if (!scaled_value_representable(value_scaled, value)) {
        mark_deposit_error(overflow);
        return;
      }
      if (value != 0.0 && colx_valid[a] && rowy_valid[b]) {
        checked_atomic_add(jz + colx[a] + rowy[b], value, overflow);
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
  // The kernel atomically sets the persistent deposit-error flag if any
  // particle's displacement spills outside the deposition window or any node
  // accumulation becomes non-finite. EmPic2D3V drains the flag after the
  // current and charge launches, before Ampere; keeping the synchronization at
  // that ownership boundary avoids one host stall per species launch.
  hipLaunchKernelGGL((deposit_kernel<ShapeOrder>), grid, block, 0, stream,
                     g, s.size(), s.charge(), dt, periodic_x, periodic_y,
                     s.x(), s.y(), s.x_prev(), s.y_prev(),
                     s.vz(), s.weight(), s.alive(), j.jx.device_ptr(), j.jy.device_ptr(),
                     j.jz.device_ptr(), s.deposit_overflow());
  // The launch error is checked once in the extern-C wrapper below, matching the
  // single-check convention of the fdtd_b / gather_push launchers.
}

template <int ShapeOrder>
__global__ void charge_kernel(quasar::Grid2D g, std::size_t n, double q,
                              bool periodic_x, bool periodic_y,
                              const double* __restrict__ x,
                              const double* __restrict__ y,
                              const double* __restrict__ weight,
                              const unsigned char* __restrict__ alive,
                              double* __restrict__ rho,
                              unsigned int* __restrict__ overflow) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;
  const double gx = quasar::detail::scaled_difference_quotient(
      x[p], g.origin_x, g.dx()) - 0.5;
  const double gy = quasar::detail::scaled_difference_quotient(
      y[p], g.origin_y, g.dy()) - 0.5;
  if (!reduced_coordinate_representable(gx)
      || !reduced_coordinate_representable(gy)) {
    mark_deposit_error(overflow);
    return;
  }
  const auto w = quasar::numerics::shape_weights_2d<ShapeOrder>(x[p], y[p], g);
  ScaledProduct scale_product;
  scaled_multiply(scale_product, q);
  scaled_multiply(scale_product, weight[p]);
  scaled_divide(scale_product, g.dx());
  scaled_divide(scale_product, g.dy());
  for (int b = 0; b < w.ny; ++b) {
    int jj = w.iy[b];
    if (periodic_y) jj = g.wrap_j(jj);
    else if (jj < -g.nghost || jj > g.ny - 1 + g.nghost) continue;
    for (int a = 0; a < w.nx; ++a) {
      int ii = w.ix[a];
      if (periodic_x) ii = g.wrap_i(ii);
      else if (ii < -g.nghost || ii > g.nx - 1 + g.nghost) continue;
      ScaledProduct value_scaled = scale_product;
      scaled_multiply(value_scaled, w.wx[a]);
      scaled_multiply(value_scaled, w.wy[b]);
      const double value = scaled_value(value_scaled);
      if (!scaled_value_representable(value_scaled, value)) {
        mark_deposit_error(overflow);
        return;
      }
      if (value != 0.0) {
        checked_atomic_add(rho + g.index(ii, jj), value, overflow);
      }
    }
  }
}

template <int ShapeOrder>
void launch_charge(const quasar::Grid2D& g, const quasar::pic::ParticleSpecies& s,
                   quasar::ScalarGrid2D<double>& rho, bool periodic_x,
                   bool periodic_y, hipStream_t stream) {
  if (s.size() == 0) return;
  const dim3 block(quasar::backend::pic::kLaunchBlock);
  const dim3 grid = quasar::backend::pic::grid_1d(s.size());
  hipLaunchKernelGGL((charge_kernel<ShapeOrder>), grid, block, 0, stream,
                     g, s.size(), s.charge(), periodic_x, periodic_y,
                     s.x(), s.y(), s.weight(), s.alive(), rho.values.device_ptr(),
                     s.deposit_overflow());
}

}  // namespace

QUASAR_PIC_DEPOSIT_ABI(launch_pic_deposit_shape1, 1)
QUASAR_PIC_DEPOSIT_ABI(launch_pic_deposit_shape2, 2)

#define QUASAR_PIC_CHARGE_ABI(fn_name, shape_order)                              \
  extern "C" void fn_name(const quasar::Grid2D& g,                              \
                          const quasar::pic::ParticleSpecies& s,                 \
                          quasar::ScalarGrid2D<quasar::Real>& rho,                \
                          int periodic_x, int periodic_y,                         \
                          quasar_stream_t stream) {                               \
    launch_charge<shape_order>(g, s, rho, periodic_x != 0, periodic_y != 0,      \
                               static_cast<hipStream_t>(stream));                 \
    QUASAR_HIP_CHECK(::hipGetLastError());                                        \
  }

QUASAR_PIC_CHARGE_ABI(launch_pic_charge_shape1, 1)
QUASAR_PIC_CHARGE_ABI(launch_pic_charge_shape2, 2)
#undef QUASAR_PIC_CHARGE_ABI

namespace {
__global__ void add_uniform_charge_kernel(quasar::Grid2D g, double density,
                                          double* rho) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  rho[g.index(i, j)] += density;
}
}  // namespace

extern "C" void launch_pic_add_uniform_charge(
    const quasar::Grid2D& g, quasar::ScalarGrid2D<quasar::Real>& rho,
    quasar::Real density, quasar_stream_t stream) {
  if (density == 0.0) return;
  const dim3 block(quasar::backend::pic::kLaunchBlock2D,
                   quasar::backend::pic::kLaunchBlock2D);
  const dim3 grid = quasar::backend::pic::grid_2d(g, block);
  hipLaunchKernelGGL(add_uniform_charge_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream), g, density,
                     rho.values.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

namespace {

__global__ void validate_finite_sources_kernel(
    std::size_t n, const double* jx, const double* jy, const double* jz,
    const double* charge, unsigned int* error) {
  const std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  if ((jx != nullptr
       && (!isfinite(jx[k]) || !isfinite(jy[k]) || !isfinite(jz[k])))
      || (charge != nullptr && !isfinite(charge[k]))) {
    atomicExch(error, 1u);
  }
}

}  // namespace

extern "C" void launch_pic_validate_finite_sources(
    const quasar::Grid2D& g,
    const quasar::JField2D<quasar::Real>* current,
    const quasar::ScalarGrid2D<quasar::Real>* charge,
    unsigned int* error, quasar_stream_t stream) {
  if (current == nullptr && charge == nullptr) {
    throw std::invalid_argument{
        "launch_pic_validate_finite_sources: no source was supplied"};
  }
  if (error == nullptr) {
    throw std::invalid_argument{
        "launch_pic_validate_finite_sources: error flag is null"};
  }

  const std::size_t n = g.storage_size();
  if (current != nullptr
      && (current->jx.size() < n || current->jy.size() < n
          || current->jz.size() < n)) {
    throw std::invalid_argument{
        "launch_pic_validate_finite_sources: current storage is too small"};
  }
  if (charge != nullptr && charge->values.size() < n) {
    throw std::invalid_argument{
        "launch_pic_validate_finite_sources: charge storage is too small"};
  }

  auto hip_stream = static_cast<hipStream_t>(stream);
  QUASAR_HIP_CHECK(::hipMemsetAsync(error, 0, sizeof(unsigned int), hip_stream));
  const dim3 block(quasar::backend::pic::kLaunchBlock);
  const dim3 grid = quasar::backend::pic::grid_1d(n);
  hipLaunchKernelGGL(
      validate_finite_sources_kernel, grid, block, 0, hip_stream, n,
      current != nullptr ? current->jx.device_ptr() : nullptr,
      current != nullptr ? current->jy.device_ptr() : nullptr,
      current != nullptr ? current->jz.device_ptr() : nullptr,
      charge != nullptr ? charge->values.device_ptr() : nullptr, error);
  QUASAR_HIP_CHECK(::hipGetLastError());

  unsigned int found_non_finite = 0;
  QUASAR_HIP_CHECK(::hipMemcpyAsync(
      &found_non_finite, error, sizeof(unsigned int),
      ::hipMemcpyDeviceToHost, hip_stream));
  QUASAR_HIP_CHECK(::hipStreamSynchronize(hip_stream));
  if (found_non_finite != 0u) {
    throw std::overflow_error{
        "PIC source construction produced a non-finite charge/current value "
        "during boundary foldback, filtering/correction, periodic restoration, "
        "or background addition"};
  }
}

namespace {

__device__ inline int correction_axis_index(int i, int n) {
  int r = i % n;
  return r < 0 ? r + n : r;
}

__device__ inline std::size_t correction_storage_index(
    const quasar::Grid2D& g, int normal, int line, bool along_x) {
  return along_x ? g.index(normal, line) : g.index(line, normal);
}

// One row of S^{-1}, where
//   S(J)_i = 13 J_i/12 - (J_{i-1}+J_{i+1})/24.
// The boundary row must substitute the SAME ghost continuation later applied to
// normal E after Ampere. PEC is even (J_-1=J_1); outflow is linear
// (J_-1=2J_0-J_1). A self-clamped neighbour is neither continuation and leaves
// an O((J1-J0)/24h) Gauss-law defect in the boundary cell.
__device__ inline double correction_row(
    quasar::Grid2D g, const double* rhs, const double* in,
    int face, int line, int n_faces, bool along_x,
    int lo_mode, int hi_mode) {
  const std::size_t k = correction_storage_index(g, face, line, along_x);
  const bool periodic = lo_mode == 0 && hi_mode == 0;
  if (periodic) {
    const int im = correction_axis_index(face - 1, n_faces);
    const int ip = correction_axis_index(face + 1, n_faces);
    return (12.0 / 13.0) * rhs[k]
         + (1.0 / 26.0)
             * (in[correction_storage_index(g, im, line, along_x)]
                + in[correction_storage_index(g, ip, line, along_x)]);
  }
  if (face == 0) {
    if (lo_mode == 2) return rhs[k];  // linear ghost => S(J)_0=J_0
    return (12.0 / 13.0) * rhs[k]
         + (1.0 / 13.0)
             * in[correction_storage_index(g, 1, line, along_x)];
  }
  if (face == n_faces - 1) {
    if (hi_mode == 2) return rhs[k];
    return (12.0 / 13.0) * rhs[k]
         + (1.0 / 13.0)
             * in[correction_storage_index(g, n_faces - 2, line, along_x)];
  }
  return (12.0 / 13.0) * rhs[k]
       + (1.0 / 26.0) *
             (in[correction_storage_index(g, face - 1, line, along_x)]
              + in[correction_storage_index(g, face + 1, line, along_x)]);
}

// One Jacobi step for the two independent compact solves
//   (13/12) J_i - (J_{i-1}+J_{i+1})/24 = J_raw_i.
// Jx couples only along x and Jy only along y. The diagonal dominance gives a
// contraction factor <=1/13, so seventeen fixed iterations reduce the initial
// error below double roundoff without a device/host convergence reduction.
__global__ void current_correct_o4_kernel(
    quasar::Grid2D g, const double* __restrict__ rhs_x,
    const double* __restrict__ rhs_y, const double* __restrict__ in_x,
    const double* __restrict__ in_y, double* __restrict__ out_x,
    double* __restrict__ out_y, int x_lo_mode, int x_hi_mode,
    int y_lo_mode, int y_hi_mode) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > g.nx || j > g.ny) return;

  // Jx and Jy live on the complete electric face lattices.  A periodic high
  // face is a duplicate and is restored after the solve; a non-periodic high
  // face is an independent physical flux and must participate in A^{-1}.
  const bool periodic_x = x_lo_mode == 0 && x_hi_mode == 0;
  const bool periodic_y = y_lo_mode == 0 && y_hi_mode == 0;
  const int nx_faces = periodic_x ? g.nx : g.nx + 1;
  const int ny_faces = periodic_y ? g.ny : g.ny + 1;
  if (j < g.ny && i < nx_faces) {
    const std::size_t k = g.index(i, j);
    out_x[k] = correction_row(g, rhs_x, in_x, i, j, nx_faces,
                              true, x_lo_mode, x_hi_mode);
  }
  if (i < g.nx && j < ny_faces) {
    const std::size_t k = g.index(i, j);
    out_y[k] = correction_row(g, rhs_y, in_y, j, i, ny_faces,
                              false, y_lo_mode, y_hi_mode);
  }
}

__global__ void current_correct_periodic_high_faces_kernel(
    quasar::Grid2D g, double* jx, double* jy, bool periodic_x,
    bool periodic_y) {
  const int line = blockIdx.x * blockDim.x + threadIdx.x;
  if (periodic_x && line < g.ny) {
    jx[g.index(g.nx, line)] = jx[g.index(0, line)];
  }
  if (periodic_y && line < g.nx) {
    jy[g.index(line, g.ny)] = jy[g.index(line, 0)];
  }
}

}  // namespace

extern "C" void launch_pic_current_correct_order4(
    const quasar::Grid2D& g, quasar::JField2D<quasar::Real>& j,
    quasar::Real* rhs_x, quasar::Real* rhs_y,
    quasar::Real* iter_x, quasar::Real* iter_y,
    int x_lo_mode, int x_hi_mode, int y_lo_mode, int y_hi_mode,
    quasar_stream_t stream) {
  auto hip_stream = static_cast<hipStream_t>(stream);
  const std::size_t bytes = g.storage_size() * sizeof(quasar::Real);
  QUASAR_HIP_CHECK(::hipMemcpyAsync(rhs_x, j.jx.device_ptr(), bytes,
                                    ::hipMemcpyDeviceToDevice, hip_stream));
  QUASAR_HIP_CHECK(::hipMemcpyAsync(rhs_y, j.jy.device_ptr(), bytes,
                                    ::hipMemcpyDeviceToDevice, hip_stream));
  QUASAR_HIP_CHECK(::hipMemcpyAsync(iter_x, j.jx.device_ptr(), bytes,
                                    ::hipMemcpyDeviceToDevice, hip_stream));
  QUASAR_HIP_CHECK(::hipMemcpyAsync(iter_y, j.jy.device_ptr(), bytes,
                                    ::hipMemcpyDeviceToDevice, hip_stream));

  const dim3 block(quasar::backend::pic::kLaunchBlock2D,
                   quasar::backend::pic::kLaunchBlock2D);
  const dim3 grid((static_cast<unsigned int>(g.nx + 1) + block.x - 1) / block.x,
                  (static_cast<unsigned int>(g.ny + 1) + block.y - 1) / block.y);
  constexpr int kIterations = 17;
  for (int it = 0; it < kIterations; ++it) {
    const bool even = (it & 1) == 0;
    const double* in_x = even ? iter_x : j.jx.device_ptr();
    const double* in_y = even ? iter_y : j.jy.device_ptr();
    double* out_x = even ? j.jx.device_ptr() : iter_x;
    double* out_y = even ? j.jy.device_ptr() : iter_y;
    hipLaunchKernelGGL(current_correct_o4_kernel, grid, block, 0, hip_stream,
                       g, rhs_x, rhs_y, in_x, in_y, out_x, out_y,
                       x_lo_mode, x_hi_mode, y_lo_mode, y_hi_mode);
  }
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_current_periodic_high_faces(
    const quasar::Grid2D& g, quasar::JField2D<quasar::Real>& j,
    int periodic_x, int periodic_y, quasar_stream_t stream) {
  if (periodic_x == 0 && periodic_y == 0) return;
  const std::size_t max_line = static_cast<std::size_t>(
      g.nx > g.ny ? g.nx : g.ny);
  const dim3 block(quasar::backend::pic::kLaunchBlock);
  const dim3 grid = quasar::backend::pic::grid_1d(max_line);
  hipLaunchKernelGGL(current_correct_periodic_high_faces_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream), g, j.jx.device_ptr(),
                     j.jy.device_ptr(), periodic_x != 0, periodic_y != 0);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

// Reads back the sticky deposit-error flag, clears it, and throws if any invalid
// coordinate, range error, or window overflow occurred. EmPic2D3V drains this
// synchronously before Ampere so an incomplete current never evolves E.
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
        "PIC deposit encountered a non-finite/nonrepresentable coordinate or "
        "value, or displacement exceeded the fixed current-deposition window; "
        "reduce dt or correct the particle/normalization scale"};
  }
}
