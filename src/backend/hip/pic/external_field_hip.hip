// Device kernels for prescribing an external field on the Yee lattice.
//
// `pic::sample_external_field` used to build the sample coordinates, check the
// evaluator's answers, and map them into solver units entirely on the host, over
// the full padded grid, six components deep, and up to three times per component
// for the cylindrical covariance probes. Every one of those passes is a
// per-lattice-point map or an all-points predicate, so they belong here.
//
// The checks report through `int*` status words rather than throwing: a boolean
// verdict OR-reduced with an integer atomic is exact and order-independent, so
// what the host sees does not depend on the launch geometry. The host keeps the
// scalar threshold configuration and the decision of which exception to raise,
// which is what makes the messages identical to the ones these kernels replace.
//
// Point ordering is `Grid2D::index(i, j)` throughout: the host loop that built
// the point cloud walked j outer, i inner over the padded extent, which is
// exactly that layout. Keeping it means materializing a component is a flat
// elementwise map with no index arithmetic at all.

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/scaled_arithmetic.hpp"
#include "quasar/physics/pic/kernels.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

namespace {

using ::quasar::Grid2D;
using ::quasar::Real;
using ::quasar::backend::pic::grid_1d;
using ::quasar::backend::pic::kLaunchBlock;
using ::quasar::numerics::ScaledProductQuotientAccumulator;
using ::quasar::numerics::ScaledValue;
using ::quasar::numerics::append_scaled_product_quotient;
using ::quasar::numerics::finish_scaled_product_quotient_sum_to_value;

__device__ inline void publish(int* status, int local) {
  if (local != 0) atomicOr(status, local);
}

// -- Sample-point construction ----------------------------------------------

__global__ void yee_points_kernel(Grid2D g, Real offset_x, Real offset_y,
                                  Real length_scale, int plane_is_xz,
                                  Real* __restrict__ px,
                                  Real* __restrict__ py,
                                  Real* __restrict__ pz,
                                  int* __restrict__ status) {
  const int pitch = g.pitch();
  const int total = pitch * g.height();
  const int s = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (s >= total) return;
  const int i = s % pitch - g.nghost;
  const int j = s / pitch - g.nghost;

  // Keep the affine coordinate as one rounded operation. On a translated grid
  // the product can lie outside Real even though cancellation with the origin
  // leaves a representable coordinate (for example -DBL_MAX + 1.5*DBL_MAX); a
  // multiply followed by an add spuriously turns that valid Yee point into
  // infinity.
  const Real internal_a = fma(static_cast<Real>(i) + offset_x, g.dx(),
                              g.origin_x);
  const Real internal_b = fma(static_cast<Real>(j) + offset_y, g.dy(),
                              g.origin_y);
  const Real a = internal_a * length_scale;
  const Real b = internal_b * length_scale;

  int local = 0;
  if (!(isfinite(a) && isfinite(b))) local |= 1;
  if ((internal_a != Real{0} && a == Real{0})
      || (internal_b != Real{0} && b == Real{0})) {
    local |= 2;
  }

  // "xy": sample the lab z=0 slice at (x, y, 0).
  // "xz": sample the lab y=0 meridional slice at (x, 0, z).
  if (plane_is_xz != 0) {
    px[s] = a;
    py[s] = Real{0};
    pz[s] = b;
  } else {
    px[s] = a;
    py[s] = b;
    pz[s] = Real{0};
  }
  publish(status, local);
}

// -- Cylindrical rotation ----------------------------------------------------

struct Rotation {
  Real cosine;
  Real sine;
  int exact_quarter_turn;
  int plane_is_xz;
};

__device__ inline void rotate(const Rotation& r, Real vx, Real vy, Real vz,
                              Real& ox, Real& oy, Real& oz) {
  // Preserve the quarter-turn as an exact signed permutation. Besides avoiding
  // trigonometric noise, this makes each output component depend on exactly one
  // input component, so an unrelated large axial field cannot hide a bad small
  // transverse component.
  if (r.exact_quarter_turn != 0) {
    if (r.plane_is_xz != 0) {
      ox = -vy; oy = vx; oz = vz;
    } else {
      ox = vz; oy = vy; oz = -vx;
    }
    return;
  }
  const Real c = r.cosine;
  const Real s = r.sine;
  if (r.plane_is_xz != 0) {
    // Active right-handed rotation about lab z.
    ox = fma(-s, vy, c * vx);
    oy = fma(s, vx, c * vy);
    oz = vz;
    return;
  }
  // Active right-handed rotation about lab y. At phi=0 this takes +lab x toward
  // -lab z, the cylindrical +phi convention for the lab x-y slice.
  ox = fma(s, vz, c * vx);
  oy = vy;
  oz = fma(-s, vx, c * vz);
}

// The roundoff floor of a rotated component. A general rotated component is a
// two-term sum, so its floor may scale with either contributing component --
// but never with the unrelated symmetry-axis component.
__device__ inline void rotation_scales(const Rotation& r, Real vx, Real vy,
                                       Real vz, Real& sx, Real& sy, Real& sz) {
  if (r.exact_quarter_turn != 0) {
    if (r.plane_is_xz != 0) {
      sx = fabs(vy); sy = fabs(vx); sz = fabs(vz);
    } else {
      sx = fabs(vz); sy = fabs(vy); sz = fabs(vx);
    }
    return;
  }
  if (r.plane_is_xz != 0) {
    const Real transverse = fmax(fabs(vx), fabs(vy));
    sx = transverse; sy = transverse; sz = fabs(vz);
    return;
  }
  const Real transverse = fmax(fabs(vx), fabs(vz));
  sx = transverse; sy = fabs(vy); sz = transverse;
}

__global__ void rotate_points_kernel(Rotation r,
                                     const Real* __restrict__ px,
                                     const Real* __restrict__ py,
                                     const Real* __restrict__ pz,
                                     int M,
                                     Real* __restrict__ ox,
                                     Real* __restrict__ oy,
                                     Real* __restrict__ oz,
                                     int* __restrict__ status) {
  const int s = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (s >= M) return;
  Real a, b, c;
  rotate(r, px[s], py[s], pz[s], a, b, c);
  if (!(isfinite(a) && isfinite(b) && isfinite(c))) publish(status, 1);
  ox[s] = a;
  oy[s] = b;
  oz[s] = c;
}

__device__ inline bool component_close(Real actual, Real expected,
                                       Real operation_scale,
                                       Real relative_tolerance,
                                       Real absolute_floor) {
  if (actual == expected) return true;
  const Real direct_error = fabs(actual - expected);
  if (isfinite(direct_error) && direct_error <= absolute_floor) return true;
  const Real scale = fmax(fmax(fabs(actual), fabs(expected)), operation_scale);
  if (scale == Real{0}) return true;
  return fabs(actual / scale - expected / scale) <= relative_tolerance;
}

__global__ void covariance_kernel(Rotation r,
                                  const Real* __restrict__ mx,
                                  const Real* __restrict__ my,
                                  const Real* __restrict__ mz,
                                  const Real* __restrict__ rx,
                                  const Real* __restrict__ ry,
                                  const Real* __restrict__ rz,
                                  int M, Real relative_tolerance,
                                  Real absolute_floor,
                                  int* __restrict__ status) {
  const int s = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (s >= M) return;

  Real ex, ey, ez;
  rotate(r, mx[s], my[s], mz[s], ex, ey, ez);
  Real sx, sy, sz;
  rotation_scales(r, mx[s], my[s], mz[s], sx, sy, sz);
  const Real ax = rx[s], ay = ry[s], az = rz[s];

  int local = 0;
  if (!(isfinite(ax) && isfinite(ay) && isfinite(az) && isfinite(ex)
        && isfinite(ey) && isfinite(ez))) {
    local |= 1;
  } else if (!(component_close(ax, ex, sx, relative_tolerance, absolute_floor)
               && component_close(ay, ey, sy, relative_tolerance,
                                  absolute_floor)
               && component_close(az, ez, sz, relative_tolerance,
                                  absolute_floor))) {
    local |= 2;
  }
  publish(status, local);
}

// -- Component materialization ----------------------------------------------

// Reads lab axis `axis` (0=x, 1=y, 2=z) and scales by `sign` for the
// right-handed frame map. The evaluator planes are already in Grid2D::index
// order, so this is a flat map over the padded storage.
__global__ void materialize_kernel(const Real* __restrict__ vx,
                                   const Real* __restrict__ vy,
                                   const Real* __restrict__ vz,
                                   int total, int axis, Real sign,
                                   Real field_scale,
                                   Real* __restrict__ out,
                                   int* __restrict__ status) {
  const int s = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (s >= total) return;

  const Real x = vx[s], y = vy[s], z = vz[s];
  int local = 0;
  if (!(isfinite(x) && isfinite(y) && isfinite(z))) {
    local |= 1;
    out[s] = Real{0};
    publish(status, local);
    return;
  }
  const Real c = axis == 0 ? x : (axis == 1 ? y : z);
  const Real mapped = sign * c / field_scale;
  if (!isfinite(mapped)) local |= 2;
  if (c != Real{0} && mapped == Real{0}) local |= 4;
  out[s] = mapped;
  publish(status, local);
}

// -- Cylindrical axis parity -------------------------------------------------

__device__ inline bool parity_close(Real actual, Real expected,
                                    Real relative_tolerance) {
  if (actual == expected) return true;
  const Real scale = fmax(fabs(actual), fabs(expected));
  if (scale == Real{0}) return true;
  return fabs(actual / scale - expected / scale) <= relative_tolerance;
}

// One thread per padded row j. Within a row the writes are strictly to the
// i<0 ghosts (plus the i=0 face under odd parity) and the reads are strictly
// from i>=1 under odd parity and i>=0 under even parity, so no thread reads a
// cell another thread writes.
__global__ void axis_parity_kernel(Grid2D g, Real* __restrict__ plane,
                                   int face_odd, Real relative_tolerance,
                                   int* __restrict__ status) {
  const int rows = g.height();
  const int t = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (t >= rows) return;
  const int j = t - g.nghost;

  int local = 0;
  if (face_odd != 0) {
    if (!parity_close(plane[g.index(0, j)], Real{0}, relative_tolerance)) {
      local |= 1;
    }
    plane[g.index(0, j)] = Real{0};
  }
  for (int gh = 1; gh <= g.nghost; ++gh) {
    const int mirror = face_odd != 0 ? gh : gh - 1;
    const Real expected = face_odd != 0 ? -plane[g.index(mirror, j)]
                                        : plane[g.index(mirror, j)];
    if (!parity_close(plane[g.index(-gh, j)], expected, relative_tolerance)) {
      local |= 2;
    }
    // Make the prescribed ghosts bit-for-bit consistent with the closure used
    // by the evolved field after accepting evaluator roundoff at the tolerance
    // above.
    plane[g.index(-gh, j)] = expected;
  }
  publish(status, local);
}

// -- Continuous solenoidality ------------------------------------------------

// trace(grad B) must vanish for a prescribed force-only magnetic field. The
// three diagonal entries are normalized by the largest of them and summed as an
// exact expansion, so a trace that cancels identically collapses to exactly
// zero rather than to one ulp of an intermediate -- which is what the host's
// long double sum was buying, and this is strictly tighter.
__global__ void solenoidal_kernel(const Real* __restrict__ G, int M,
                                  Real relative_tolerance,
                                  int* __restrict__ status) {
  const int s = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (s >= M) return;

  // Component-major 9*M layout: entry (i, j) at (3*i + j) * M + s.
  const long long stride = M;
  int local = 0;
  for (int e = 0; e < 9; ++e) {
    if (!isfinite(G[static_cast<long long>(e) * stride + s])) local |= 1;
  }
  if (local != 0) {
    publish(status, local);
    return;
  }

  const Real dxx = G[0 * stride + s];
  const Real dyy = G[4 * stride + s];
  const Real dzz = G[8 * stride + s];
  const Real scale = fmax(fabs(dxx), fmax(fabs(dyy), fabs(dzz)));
  if (scale == Real{0}) return;

  ScaledProductQuotientAccumulator<6> trace;
  append_scaled_product_quotient(trace, dxx, Real{1}, scale, Real{1});
  append_scaled_product_quotient(trace, dyy, Real{1}, scale, Real{1});
  append_scaled_product_quotient(trace, dzz, Real{1}, scale, Real{1});
  const ScaledValue residual =
      finish_scaled_product_quotient_sum_to_value(trace);

  const Real magnitude = fabs(dxx / scale) + fabs(dyy / scale)
                       + fabs(dzz / scale);
  const Real value = scalbn(fabs(residual.mantissa), residual.exponent);
  if (value > relative_tolerance * magnitude) local |= 2;
  publish(status, local);
}

// -- Identically-zero predicate ----------------------------------------------

__global__ void nonzero_kernel(const Real* __restrict__ x,
                               const Real* __restrict__ y,
                               const Real* __restrict__ z, int M,
                               int* __restrict__ flag) {
  const int s = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
              + static_cast<int>(threadIdx.x);
  if (s >= M) return;
  if (x[s] != Real{0} || y[s] != Real{0} || z[s] != Real{0}) {
    atomicOr(flag, 1);
  }
}

dim3 grid_for(int n) { return grid_1d(static_cast<std::size_t>(n)); }

}  // namespace

extern "C" void launch_pic_yee_points(
    const Grid2D& g, Real offset_x, Real offset_y, Real length_scale,
    int plane_is_xz, Real* px, Real* py, Real* pz, int* status,
    ::quasar::backend::stream_t stream) {
  const int total = g.pitch() * g.height();
  if (total <= 0) return;
  yee_points_kernel<<<grid_for(total), dim3(kLaunchBlock), 0,
                      static_cast<hipStream_t>(stream)>>>(
      g, offset_x, offset_y, length_scale, plane_is_xz, px, py, pz, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_rotate_points(
    int plane_is_xz, Real cosine, Real sine, int exact_quarter_turn,
    const Real* px, const Real* py, const Real* pz, int M,
    Real* ox, Real* oy, Real* oz, int* status,
    ::quasar::backend::stream_t stream) {
  if (M <= 0) return;
  const Rotation r{cosine, sine, exact_quarter_turn, plane_is_xz};
  rotate_points_kernel<<<grid_for(M), dim3(kLaunchBlock), 0,
                         static_cast<hipStream_t>(stream)>>>(
      r, px, py, pz, M, ox, oy, oz, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_check_rotational_covariance(
    int plane_is_xz, Real cosine, Real sine, int exact_quarter_turn,
    const Real* mx, const Real* my, const Real* mz,
    const Real* rx, const Real* ry, const Real* rz, int M,
    Real relative_tolerance, Real absolute_floor, int* status,
    ::quasar::backend::stream_t stream) {
  if (M <= 0) return;
  const Rotation r{cosine, sine, exact_quarter_turn, plane_is_xz};
  covariance_kernel<<<grid_for(M), dim3(kLaunchBlock), 0,
                      static_cast<hipStream_t>(stream)>>>(
      r, mx, my, mz, rx, ry, rz, M, relative_tolerance, absolute_floor, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_materialize_external_component(
    const Real* vx, const Real* vy, const Real* vz, int total, int axis,
    Real sign, Real field_scale, Real* out, int* status,
    ::quasar::backend::stream_t stream) {
  if (total <= 0) return;
  materialize_kernel<<<grid_for(total), dim3(kLaunchBlock), 0,
                       static_cast<hipStream_t>(stream)>>>(
      vx, vy, vz, total, axis, sign, field_scale, out, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_enforce_axis_parity(
    const Grid2D& g, Real* plane, int face_odd, Real relative_tolerance,
    int* status, ::quasar::backend::stream_t stream) {
  const int rows = g.height();
  if (rows <= 0) return;
  axis_parity_kernel<<<grid_for(rows), dim3(kLaunchBlock), 0,
                       static_cast<hipStream_t>(stream)>>>(
      g, plane, face_odd, relative_tolerance, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_check_continuous_solenoidality(
    const Real* G, int M, Real relative_tolerance, int* status,
    ::quasar::backend::stream_t stream) {
  if (M <= 0) return;
  solenoidal_kernel<<<grid_for(M), dim3(kLaunchBlock), 0,
                      static_cast<hipStream_t>(stream)>>>(
      G, M, relative_tolerance, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_field_has_nonzero(
    const Real* x, const Real* y, const Real* z, int M, int* flag,
    ::quasar::backend::stream_t stream) {
  if (M <= 0) return;
  nonzero_kernel<<<grid_for(M), dim3(kLaunchBlock), 0,
                   static_cast<hipStream_t>(stream)>>>(x, y, z, M, flag);
  QUASAR_HIP_CHECK(::hipGetLastError());
}
