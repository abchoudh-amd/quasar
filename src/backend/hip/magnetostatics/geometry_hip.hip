// Device generation, validation and flattening of filament geometry.
//
// The generators in src/physics/magnetostatics/geometry_generators.cpp used to
// walk every vertex on the host: a sin/cos pair, a scaled local displacement,
// and two resolvability checks per point. That is the last per-point host
// arithmetic in the magnetostatics slice, so it moved here.
//
// The resolvability checks are what make these kernels more than a trig call.
// A local displacement requested at a large centre can lose an entire dimension
// to rounding while the surviving coordinates still form non-zero segments --
// a coil at x = 1e300 whose radius silently vanishes still looks like a valid
// polyline. Each vertex therefore verifies that every *material* component of
// its offset survives both the scaling and the translation with at least one
// useful binary digit. Components below a few ulp of the unit direction are
// immaterial by construction, which is what admits the expected sin(pi)
// residue without special-casing it.

#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

namespace {

using ::quasar::Vec3;

constexpr unsigned kBlock = 256;

// Status bits.
constexpr int kRadialUnresolved = 1;
constexpr int kAxialUnresolved = 2;
constexpr int kOverflow = 4;

dim3 grid_for(long long n) {
  return dim3(static_cast<unsigned>((n + kBlock - 1) / kBlock));
}

__device__ inline Vec3 load3(const double* v) {
  return Vec3{v[0], v[1], v[2]};
}

// Form `distance * direction` and confirm no material component was lost.
// Mirrors the host `scaled_direction`.
__device__ inline Vec3 scaled_direction(double distance, Vec3 direction,
                                        double material_component,
                                        bool& unresolved) {
  const Vec3 offset = distance * direction;
  if (distance == 0.0) return offset;

  const double magnitude = fabs(distance);
  const double direction_components[3] = {direction.x, direction.y, direction.z};
  const double offset_components[3] = {offset.x, offset.y, offset.z};
  for (int component = 0; component < 3; ++component) {
    const double unit_component = fabs(direction_components[component]);
    if (unit_component <= material_component) continue;
    const double actual = offset_components[component];
    if (actual == 0.0 || !isfinite(actual)
        || signbit(actual)
            != signbit(distance * direction_components[component])) {
      unresolved = true;
      return offset;
    }
    // Dividing back by the requested distance avoids forming an exact
    // distance*direction reference that may itself be subnormal. A ratio
    // outside [1/2, 3/2] means even the first significant bit of this local
    // component was lost.
    const double ratio = fabs((actual / magnitude) / unit_component);
    if (!isfinite(ratio) || ratio < 0.5 || ratio > 1.5) {
      unresolved = true;
      return offset;
    }
  }
  return offset;
}

// Translate a locally meaningful displacement into global coordinates, then
// round-trip it back. Mirrors the host `translated_point`.
__device__ inline Vec3 translated_point(Vec3 origin, Vec3 offset,
                                        double local_scale,
                                        double material_component,
                                        bool& unresolved, bool& overflow) {
  const Vec3 point = origin + offset;
  if (!isfinite(point.x) || !isfinite(point.y) || !isfinite(point.z)) {
    overflow = true;
    return point;
  }
  if (!(local_scale > 0.0)) return point;

  const Vec3 recovered = point - origin;
  const double intended_components[3] = {offset.x, offset.y, offset.z};
  const double recovered_components[3] = {recovered.x, recovered.y,
                                          recovered.z};
  for (int component = 0; component < 3; ++component) {
    const double intended = intended_components[component];
    if (fabs(intended) <= material_component * local_scale) continue;
    const double actual = recovered_components[component];
    if (actual == 0.0 || !isfinite(actual)
        || signbit(actual) != signbit(intended)) {
      unresolved = true;
      return point;
    }
    const double ratio = fabs(actual / intended);
    if (!isfinite(ratio) || ratio < 0.5 || ratio > 1.5) {
      unresolved = true;
      return point;
    }
  }
  return point;
}

__device__ inline void store3(int k, Vec3 p, double* px, double* py,
                              double* pz) {
  px[k] = p.x;
  py[k] = p.y;
  pz[k] = p.z;
}

__device__ inline void publish(int* status, int local) {
  if (local != 0) atomicOr(status, local);
}

// The angle is rebuilt from the integer index every time rather than
// accumulated, so a many-turn helix cannot drift through argument reduction.
__device__ inline double arc_angle(const QuasarMsArcParams& params, int k) {
  int index = k + params.index_offset;
  if (params.index_modulus > 0) index %= params.index_modulus;
  return params.theta_bias
       + params.theta_scale * static_cast<double>(index)
             / static_cast<double>(params.theta_divisor);
}

__global__ void arc_points_kernel(QuasarMsArcParams params, int count,
                                  double* __restrict__ px,
                                  double* __restrict__ py,
                                  double* __restrict__ pz,
                                  int* __restrict__ status) {
  const int k = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (k >= count) return;

  const double theta = arc_angle(params, k);
  const Vec3 radial_direction =
      cos(theta) * load3(params.u) + sin(theta) * load3(params.v);

  bool unresolved = false;
  bool overflow = false;
  const Vec3 radial = scaled_direction(params.radius, radial_direction,
                                       params.material_component, unresolved);
  const Vec3 point = translated_point(load3(params.centre), radial,
                                      params.radius,
                                      params.material_component, unresolved,
                                      overflow);
  store3(k, point, px, py, pz);

  int local = 0;
  if (unresolved) local |= kRadialUnresolved;
  if (overflow) local |= kOverflow;
  publish(status, local);
}

__global__ void helix_points_kernel(QuasarMsHelixParams params, int count,
                                    double* __restrict__ px,
                                    double* __restrict__ py,
                                    double* __restrict__ pz,
                                    int* __restrict__ status) {
  const int k = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (k >= count) return;

  bool axial_unresolved = false;
  bool radial_unresolved = false;
  bool overflow = false;

  const double fraction = static_cast<double>(k)
                        / static_cast<double>(params.n_total);
  // std::lerp(-half_length, half_length, fraction). The endpoints straddle zero
  // whenever half_length > 0, which is std::lerp's first branch; the degenerate
  // half_length == 0 case lands there too.
  const double axial_distance =
      fraction * params.half_length + (1.0 - fraction) * (-params.half_length);
  const Vec3 axial_offset = scaled_direction(
      axial_distance, load3(params.axis), params.material_component,
      axial_unresolved);
  const Vec3 turn_centre = translated_point(
      load3(params.centre), axial_offset, fabs(axial_distance),
      params.material_component, axial_unresolved, overflow);

  // Radial phase is periodic; reduce the integer index before forming the
  // angle so many turns do not accumulate argument-reduction drift.
  const int phase = k % params.segments_per_turn;
  const double theta = 2.0 * ::quasar::pi_v<double> * static_cast<double>(phase)
                     / static_cast<double>(params.segments_per_turn);
  const Vec3 radial_direction =
      cos(theta) * load3(params.u) + sin(theta) * load3(params.v);
  const Vec3 radial = scaled_direction(params.radius, radial_direction,
                                       params.material_component,
                                       radial_unresolved);
  const Vec3 point = translated_point(turn_centre, radial, params.radius,
                                      params.material_component,
                                      radial_unresolved, overflow);
  store3(k, point, px, py, pz);

  int local = 0;
  if (radial_unresolved) local |= kRadialUnresolved;
  if (axial_unresolved) local |= kAxialUnresolved;
  if (overflow) local |= kOverflow;
  publish(status, local);
}

__global__ void validate_segments_kernel(const double* __restrict__ px,
                                         const double* __restrict__ py,
                                         const double* __restrict__ pz,
                                         int n_points,
                                         int* __restrict__ status) {
  const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x) + 1;
  if (i >= n_points) return;
  if (px[i] == px[i - 1] && py[i] == py[i - 1] && pz[i] == pz[i - 1]) {
    atomicOr(status, 1);
  }
}

__global__ void flatten_kernel(const double* __restrict__ px,
                               const double* __restrict__ py,
                               const double* __restrict__ pz,
                               int n_points, double current, long long offset,
                               double* __restrict__ ax,
                               double* __restrict__ ay,
                               double* __restrict__ az,
                               double* __restrict__ bx,
                               double* __restrict__ by,
                               double* __restrict__ bz,
                               double* __restrict__ I_,
                               int* __restrict__ status) {
  const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (i + 1 >= n_points) return;

  const Vec3 a{px[i], py[i], pz[i]};
  const Vec3 b{px[i + 1], py[i + 1], pz[i + 1]};
  int local = 0;
  if (!(isfinite(a.x) && isfinite(a.y) && isfinite(a.z) && isfinite(b.x)
        && isfinite(b.y) && isfinite(b.z))) {
    local |= 1;
  } else {
    const Vec3 d = b - a;
    if (!(isfinite(d.x) && isfinite(d.y) && isfinite(d.z))) {
      local |= 2;
    } else if (d.x == 0.0 && d.y == 0.0 && d.z == 0.0) {
      local |= 4;
    }
  }

  const long long s = offset + i;
  ax[s] = a.x; ay[s] = a.y; az[s] = a.z;
  bx[s] = b.x; by[s] = b.y; bz[s] = b.z;
  I_[s] = current;
  publish(status, local);
}

__device__ inline float narrow(double value, double origin, int* local) {
  const float narrowed = static_cast<float>(value - origin);
  if (!isfinite(narrowed)) *local |= 1;
  return narrowed;
}

__global__ void narrow_segments_kernel(
    const double* __restrict__ ax, const double* __restrict__ ay,
    const double* __restrict__ az, const double* __restrict__ bx,
    const double* __restrict__ by, const double* __restrict__ bz,
    const double* __restrict__ I_, int n,
    double ox, double oy, double oz,
    float* __restrict__ fax, float* __restrict__ fay,
    float* __restrict__ faz, float* __restrict__ fbx,
    float* __restrict__ fby, float* __restrict__ fbz,
    float* __restrict__ fI, int* __restrict__ status) {
  const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (i >= n) return;

  int local = 0;
  const float a0 = narrow(ax[i], ox, &local);
  const float a1 = narrow(ay[i], oy, &local);
  const float a2 = narrow(az[i], oz, &local);
  const float b0 = narrow(bx[i], ox, &local);
  const float b1 = narrow(by[i], oy, &local);
  const float b2 = narrow(bz[i], oz, &local);
  const float current = narrow(I_[i], 0.0, &local);
  if (a0 == b0 && a1 == b1 && a2 == b2) local |= 2;

  fax[i] = a0; fay[i] = a1; faz[i] = a2;
  fbx[i] = b0; fby[i] = b1; fbz[i] = b2;
  fI[i] = current;
  publish(status, local);
}

__global__ void narrow_points_kernel(
    const double* __restrict__ px, const double* __restrict__ py,
    const double* __restrict__ pz, int n,
    double ox, double oy, double oz,
    float* __restrict__ fpx, float* __restrict__ fpy,
    float* __restrict__ fpz, int* __restrict__ status) {
  const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (i >= n) return;
  int local = 0;
  fpx[i] = narrow(px[i], ox, &local);
  fpy[i] = narrow(py[i], oy, &local);
  fpz[i] = narrow(pz[i], oz, &local);
  publish(status, local);
}

}  // namespace

extern "C" void launch_ms_arc_points(
    QuasarMsArcParams params, int count, double* px, double* py, double* pz,
    int* status, ::quasar::backend::stream_t stream) {
  if (count <= 0) return;
  arc_points_kernel<<<grid_for(count), dim3(kBlock), 0,
                      static_cast<hipStream_t>(stream)>>>(
      params, count, px, py, pz, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_ms_helix_points(
    QuasarMsHelixParams params, int count, double* px, double* py, double* pz,
    int* status, ::quasar::backend::stream_t stream) {
  if (count <= 0) return;
  helix_points_kernel<<<grid_for(count), dim3(kBlock), 0,
                        static_cast<hipStream_t>(stream)>>>(
      params, count, px, py, pz, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_ms_validate_segments(
    const double* px, const double* py, const double* pz, int n_points,
    int* status, ::quasar::backend::stream_t stream) {
  if (n_points < 2) return;
  validate_segments_kernel<<<grid_for(n_points - 1), dim3(kBlock), 0,
                             static_cast<hipStream_t>(stream)>>>(
      px, py, pz, n_points, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_ms_flatten_filament(
    const double* px, const double* py, const double* pz, int n_points,
    double current, long long offset,
    double* ax, double* ay, double* az,
    double* bx, double* by, double* bz, double* I_,
    int* status, ::quasar::backend::stream_t stream) {
  if (n_points < 2) return;
  flatten_kernel<<<grid_for(n_points - 1), dim3(kBlock), 0,
                   static_cast<hipStream_t>(stream)>>>(
      px, py, pz, n_points, current, offset, ax, ay, az, bx, by, bz, I_,
      status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_ms_narrow_segments(
    const double* ax, const double* ay, const double* az,
    const double* bx, const double* by, const double* bz,
    const double* I_, int n, double ox, double oy, double oz,
    float* fax, float* fay, float* faz,
    float* fbx, float* fby, float* fbz, float* fI,
    int* status, ::quasar::backend::stream_t stream) {
  if (n <= 0) return;
  narrow_segments_kernel<<<grid_for(n), dim3(kBlock), 0,
                           static_cast<hipStream_t>(stream)>>>(
      ax, ay, az, bx, by, bz, I_, n, ox, oy, oz, fax, fay, faz, fbx, fby, fbz,
      fI, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_ms_narrow_points(
    const double* px, const double* py, const double* pz, int n,
    double ox, double oy, double oz,
    float* fpx, float* fpy, float* fpz, int* status,
    ::quasar::backend::stream_t stream) {
  if (n <= 0) return;
  narrow_points_kernel<<<grid_for(n), dim3(kBlock), 0,
                         static_cast<hipStream_t>(stream)>>>(
      px, py, pz, n, ox, oy, oz, fpx, fpy, fpz, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}
