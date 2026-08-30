// Device expansion of the structured observation sets.
//
// One thread per lattice site, evaluating the same affine formula the host
// `point_at` accessors use. Those accessors stay on the host: they are scalar
// queries against a description, not a sweep, and tests compare the device
// expansion against them.
//
// Bit-exactness with `point_at` is a requirement, not a nicety, which is why
// `device_lerp` below reproduces the C++20 std::lerp algorithm step for step
// rather than using the obvious `a + t * (b - a)`. std::lerp is specified to be
// exact at both endpoints and monotonic; the naive form is neither, and a line
// probe whose last point drifts off `end` by an ulp would quietly disagree with
// the accessor it is supposed to match.

#include "quasar/backend/device.hpp"
#include "quasar/core/observation_kernels.hpp"
#include "quasar/core/types.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

namespace {

using ::quasar::Real;

constexpr unsigned kBlock = 256;

dim3 grid_for(long long n) {
  return dim3(static_cast<unsigned>((n + kBlock - 1) / kBlock));
}

__device__ inline void record(Real value, Real* slot, int* local) {
  if (!isfinite(value)) *local |= 1;
  *slot = value;
}

// C++20 std::lerp, transcribed. The endpoint and monotonicity guarantees are
// what make this agree with LineProbe::point_at bit for bit.
__device__ inline Real device_lerp(Real a, Real b, Real t) {
  if ((a <= Real{0} && b >= Real{0}) || (a >= Real{0} && b <= Real{0})) {
    return t * b + (Real{1} - t) * a;
  }
  if (t == Real{1}) return b;
  const Real x = a + t * (b - a);
  if ((t > Real{1}) == (b > a)) return b < x ? x : b;
  return x < b ? x : b;
}

__global__ void grid_points_kernel(QuasarObsGridParams params,
                                   long long total,
                                   Real* __restrict__ px,
                                   Real* __restrict__ py,
                                   Real* __restrict__ pz,
                                   int* __restrict__ status) {
  const long long s = static_cast<long long>(blockIdx.x) * blockDim.x
                    + threadIdx.x;
  if (s >= total) return;
  const long long nx = params.dims[0];
  const long long ny = params.dims[1];
  const long long i = s % nx;
  const long long j = (s / nx) % ny;
  const long long k = s / (nx * ny);

  int local = 0;
  record(fma(static_cast<Real>(i), params.spacing[0], params.origin[0]),
         &px[s], &local);
  record(fma(static_cast<Real>(j), params.spacing[1], params.origin[1]),
         &py[s], &local);
  record(fma(static_cast<Real>(k), params.spacing[2], params.origin[2]),
         &pz[s], &local);
  if (local != 0) atomicOr(status, local);
}

__global__ void plane_points_kernel(QuasarObsPlaneParams params,
                                    long long total,
                                    Real* __restrict__ px,
                                    Real* __restrict__ py,
                                    Real* __restrict__ pz,
                                    int* __restrict__ status) {
  const long long s = static_cast<long long>(blockIdx.x) * blockDim.x
                    + threadIdx.x;
  if (s >= total) return;
  const long long nu = params.nu;
  const long long i = s % nu;
  const long long j = s / nu;

  int local = 0;
  Real* const out[3] = {&px[s], &py[s], &pz[s]};
  for (int c = 0; c < 3; ++c) {
    const Real value = fma(static_cast<Real>(j), params.v_step[c],
                           fma(static_cast<Real>(i), params.u_step[c],
                               params.origin[c]));
    record(value, out[c], &local);
  }
  if (local != 0) atomicOr(status, local);
}

__global__ void line_points_kernel(QuasarObsLineParams params,
                                   Real* __restrict__ px,
                                   Real* __restrict__ py,
                                   Real* __restrict__ pz,
                                   int* __restrict__ status) {
  const int s = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (s >= params.n_points) return;

  const Real t = static_cast<Real>(s)
               / static_cast<Real>(params.n_points - 1);
  int local = 0;
  Real* const out[3] = {&px[s], &py[s], &pz[s]};
  for (int c = 0; c < 3; ++c) {
    record(device_lerp(params.start[c], params.end[c], t), out[c], &local);
  }
  if (local != 0) atomicOr(status, local);
}

}  // namespace

extern "C" void launch_observation_grid_points(
    QuasarObsGridParams params, Real* px, Real* py, Real* pz, int* status,
    ::quasar::backend::stream_t stream) {
  const long long total = static_cast<long long>(params.dims[0])
                        * params.dims[1] * params.dims[2];
  if (total <= 0) return;
  grid_points_kernel<<<grid_for(total), dim3(kBlock), 0,
                       static_cast<hipStream_t>(stream)>>>(
      params, total, px, py, pz, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_observation_plane_points(
    QuasarObsPlaneParams params, Real* px, Real* py, Real* pz, int* status,
    ::quasar::backend::stream_t stream) {
  const long long total = static_cast<long long>(params.nu) * params.nv;
  if (total <= 0) return;
  plane_points_kernel<<<grid_for(total), dim3(kBlock), 0,
                        static_cast<hipStream_t>(stream)>>>(
      params, total, px, py, pz, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_observation_line_points(
    QuasarObsLineParams params, Real* px, Real* py, Real* pz, int* status,
    ::quasar::backend::stream_t stream) {
  if (params.n_points <= 0) return;
  line_points_kernel<<<grid_for(params.n_points), dim3(kBlock), 0,
                       static_cast<hipStream_t>(stream)>>>(
      params, px, py, pz, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}
