#include "quasar/core/yee_field.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

namespace {

__device__ inline int clamp_axis(int i, int lo, int hi) {
  return i < lo ? lo : (i > hi ? hi : i);
}

__device__ inline std::size_t filter_index(const quasar::Grid2D& g, int i, int j,
                                           bool periodic_x, bool periodic_y,
                                           int x_hi) {
  const int ii = periodic_x ? g.wrap_i(i) : clamp_axis(i, 0, x_hi);
  const int jj = periodic_y ? g.wrap_j(j) : clamp_axis(j, 0, g.ny - 1);
  return g.index(ii, jj);
}

// Separable digital filter pass: out[k] = a*left + b*center + a*right applied
// first along x (Axis == 0) then along y (Axis == 1). (a, b) = (0.25, 0.5) is the
// binomial (1-2-1)/4 smoother; the compensator pass uses (a, b) = (-0.5, 2.0) =
// (1-2-1) with a sharpening sign so that smoother∘compensator has flat response at
// low k. Periodic axes wrap; wall axes clamp at the edge, avoiding far-edge
// current coupling on mixed/non-periodic domains.
//
// In 2D, Jz does not enter charge continuity and may be smoothed directly. Jx/Jy
// are copied unchanged: convolving them without applying the identical operator
// to both endpoint charge densities changes div(J) and violates Gauss's law.
// A future longitudinal/transverse projection may smooth the in-plane pair, but
// silently breaking continuity is not an acceptable current-filter semantics.
template <int Axis>
__global__ void stencil_kernel(quasar::Grid2D g,
                               const double* __restrict__ in_z,
                               double* __restrict__ out_z,
                               double a, double b, bool periodic_x,
                               bool periodic_y, bool cylindrical) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int logical_nx = cylindrical ? g.nx + 1 : g.nx;
  if (i >= logical_nx || j >= g.ny) return;
  const int di = (Axis == 0) ? 1 : 0;
  const int dj = (Axis == 0) ? 0 : 1;
  const int x_hi = logical_nx - 1;
  const std::size_t kl = filter_index(g, i - di, j - dj, periodic_x,
                                      periodic_y, x_hi);
  const std::size_t kc = filter_index(g, i, j, periodic_x, periodic_y, x_hi);
  const std::size_t kr = filter_index(g, i + di, j + dj, periodic_x,
                                      periodic_y, x_hi);
  const std::size_t ko = g.index(i, j);
  // Jphi is odd on the r=0 axis. Preserve regularity explicitly; a clamped
  // smoother would otherwise manufacture a non-zero axis current.
  if (cylindrical && g.origin_x == 0.0 && i == 0) {
    out_z[ko] = 0.0;
  } else {
    out_z[ko] = a * in_z[kl] + b * in_z[kc] + a * in_z[kr];
  }
}

// One separable (x then y) pass over Jz, using the single `scratch` strip as
// ping-pong storage between the two axis sweeps. Jx/Jy are not read or written.
// Leaves the result back in the JField2D buffer.
void filter_pass(const quasar::Grid2D& g, quasar::JField2D<double>& j,
                 double* sz,
                 double a, double b, bool periodic_x, bool periodic_y,
                 bool cylindrical, hipStream_t stream) {
  const dim3 block(quasar::backend::pic::kLaunchBlock2D,
                   quasar::backend::pic::kLaunchBlock2D);
  const unsigned int logical_nx = static_cast<unsigned int>(
      cylindrical ? g.nx + 1 : g.nx);
  const dim3 grid((logical_nx + block.x - 1) / block.x,
                  (static_cast<unsigned int>(g.ny) + block.y - 1) / block.y);
  hipLaunchKernelGGL(stencil_kernel<0>, grid, block, 0, stream,
                     g, j.jz.device_ptr(), sz, a, b, periodic_x, periodic_y,
                     cylindrical);
  hipLaunchKernelGGL(stencil_kernel<1>, grid, block, 0, stream,
                     g, sz, j.jz.device_ptr(), a, b, periodic_x, periodic_y,
                     cylindrical);
}

void run_passes(const quasar::Grid2D& g, quasar::JField2D<double>& j, double* scratch,
                int passes, double a, double b, bool periodic_x, bool periodic_y,
                bool cylindrical, hipStream_t stream) {
  if (passes <= 0) return;
  // `scratch` is one storage_size() strip: only Jz is smoothed.
  double* sz = scratch;
  for (int p = 0; p < passes; ++p) {
    filter_pass(g, j, sz, a, b, periodic_x, periodic_y, cylindrical, stream);
  }
}

}  // namespace

// `scratch` is caller-owned ping-pong storage of at least g.storage_size() Real
// values, hoisted out of the per-step hot path (see numerics/filter.cpp). Only
// Jz is smoothed, so one strip suffices; Jx/Jy are left untouched to preserve
// discrete charge continuity.
extern "C" void launch_pic_filter_binomial(const quasar::Grid2D& g,
                                           quasar::JField2D<quasar::Real>& j,
                                           quasar::Real* scratch, int passes, int periodic_x,
                                           int periodic_y, int cylindrical,
                                           quasar_stream_t stream) {
  run_passes(g, j, scratch, passes, 0.25, 0.5,
             periodic_x != 0, periodic_y != 0, cylindrical != 0,
             static_cast<hipStream_t>(stream));
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_filter_compensated(const quasar::Grid2D& g,
                                              quasar::JField2D<quasar::Real>& j,
                                              quasar::Real* scratch, int passes, int periodic_x,
                                              int periodic_y, int cylindrical,
                                              quasar_stream_t stream) {
  // For H_s=1-s, `passes` smooths have low-k response 1-passes*s. The one
  // compensator must therefore have response 1+passes*s, corresponding to
  // a=-passes/4, b=1+passes/2. The historical fixed (-1/2,2) coefficients only
  // compensate two passes and amplify resolved modes when the default is one.
  const auto hip_stream = static_cast<hipStream_t>(stream);
  const bool px = periodic_x != 0;
  const bool py = periodic_y != 0;
  const bool cyl = cylindrical != 0;
  run_passes(g, j, scratch, passes, 0.25, 0.5, px, py, cyl, hip_stream);
  const double a_comp = -0.25 * static_cast<double>(passes);
  const double b_comp = 1.0 + 0.5 * static_cast<double>(passes);
  run_passes(g, j, scratch, 1, a_comp, b_comp, px, py, cyl, hip_stream);
  QUASAR_HIP_CHECK(::hipGetLastError());
}
