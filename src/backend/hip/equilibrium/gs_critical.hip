// Device critical-point search: magnetic axis and X-points.
//
// This step decides psi_N, which decides the current, which creates the axis,
// so it runs every Picard iteration and its tie-breaking is load-bearing rather
// than incidental.
//
// -- Two halves, parallelized differently --------------------------------------
// The candidate scan and Newton refinement are one thread per interior node and
// bit-exact: each Newton iterate is serial within its own thread, so the
// arithmetic is identical to the host's.
//
// The duplicate merge and the axis/boundary selection are NOT parallelized.
// They are order-dependent in a way that changes the answer, not just the last
// bit: the host keeps the FIRST candidate in row-major scan order when merging
// near-duplicates, and picks the deepest O-point with a strict `>` so ties go to
// the earliest. On a symmetric equilibrium -- an up-down symmetric double-null,
// say -- two X-points sit at exactly equal |psi - psi_axis|, and a reordered
// merge silently selects the other one and shifts psi_boundary.
//
// So that half runs single-threaded on device. That sounds wasteful and is not:
// the candidate count is tens, the pass is microseconds, and it keeps the result
// resident for the rest of the iteration. Parallelizing it would be a
// correctness change disguised as an optimization.

#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace quasar::equilibrium {
namespace {

inline constexpr unsigned kBlock2D = 16;
inline constexpr unsigned kThreads = 256;

// Search inset. The Pade closures are least accurate in the first rows, and a
// physical axis never sits on the domain edge.
inline constexpr int kMargin = 3;

struct GridView {
  int nr;
  int nz;
  Real r_min;
  Real z_min;
  Real r_max;
  Real z_max;
  Real dr;
  Real dz;

  __device__ Real r(int i) const { return r_min + static_cast<Real>(i) * dr; }
  __device__ Real z(int j) const { return z_min + static_cast<Real>(j) * dz; }
  __device__ bool on_boundary(int i, int j) const {
    return i == 0 || j == 0 || i == nr - 1 || j == nz - 1;
  }
  __device__ std::size_t index(int i, int j) const {
    return static_cast<std::size_t>(i) + static_cast<std::size_t>(nr) * j;
  }
};

GridView view_of(const EllipticGrid& g) {
  return GridView{g.nr,  g.nz,  g.r_min, g.z_min,
                  g.r_max, g.z_max, g.dr(), g.dz()};
}

__device__ inline Real sample_bilinear_dev(const GridView& g, const Real* f,
                                           Real r, Real z) {
  Real fi = (r - g.r_min) / g.dr;
  Real fj = (z - g.z_min) / g.dz;
  fi = fmin(fmax(fi, Real{0}), static_cast<Real>(g.nr - 1));
  fj = fmin(fmax(fj, Real{0}), static_cast<Real>(g.nz - 1));
  int i = min(static_cast<int>(fi), g.nr - 2);
  int j = min(static_cast<int>(fj), g.nz - 2);
  const Real tr = fi - static_cast<Real>(i);
  const Real tz = fj - static_cast<Real>(j);
  return (Real{1} - tr) * (Real{1} - tz) * f[g.index(i, j)]
       + tr * (Real{1} - tz) * f[g.index(i + 1, j)]
       + (Real{1} - tr) * tz * f[g.index(i, j + 1)]
       + tr * tz * f[g.index(i + 1, j + 1)];
}

struct DerivView {
  const Real* d_r;
  const Real* d_z;
  const Real* d_rr;
  const Real* d_zz;
  const Real* d_rz;
};

// Newton on grad psi = 0, with steps clamped to one cell so a near-singular
// Hessian cannot throw the iterate out of the domain.
__device__ inline CriticalPoint refine_dev(const GridView& g, const Real* psi,
                                           const DerivView& d, Real r0,
                                           Real z0, int max_iter) {
  CriticalPoint cp{};
  Real r = r0;
  Real z = z0;

  for (int it = 0; it < max_iter; ++it) {
    const Real gr  = sample_bilinear_dev(g, d.d_r, r, z);
    const Real gz  = sample_bilinear_dev(g, d.d_z, r, z);
    const Real hrr = sample_bilinear_dev(g, d.d_rr, r, z);
    const Real hzz = sample_bilinear_dev(g, d.d_zz, r, z);
    const Real hrz = sample_bilinear_dev(g, d.d_rz, r, z);

    const Real det = hrr * hzz - hrz * hrz;
    if (fabs(det) < Real{1e-30}) return cp;  // degenerate: report failure

    Real dr = -(hzz * gr - hrz * gz) / det;
    Real dz = -(-hrz * gr + hrr * gz) / det;

    dr = fmin(fmax(dr, -g.dr), g.dr);
    dz = fmin(fmax(dz, -g.dz), g.dz);

    r += dr;
    z += dz;
    if (r <= g.r_min || r >= g.r_max || z <= g.z_min || z >= g.z_max) return cp;

    if (fabs(dr) < Real{1e-13} * g.dr && fabs(dz) < Real{1e-13} * g.dz) break;
  }

  const Real hrr = sample_bilinear_dev(g, d.d_rr, r, z);
  const Real hzz = sample_bilinear_dev(g, d.d_zz, r, z);
  const Real hrz = sample_bilinear_dev(g, d.d_rz, r, z);
  const Real det = hrr * hzz - hrz * hrz;

  cp.r = r;
  cp.z = z;
  cp.psi = sample_bilinear_dev(g, psi, r, z);
  cp.kind = det > Real{0} ? CriticalKind::o_point : CriticalKind::x_point;
  cp.valid = true;
  return cp;
}

// Max |grad psi| over the inset region. This sets the scale for "indistinguishable
// from zero": a bare sign-change test fires on roundoff wherever the field is
// locally flat, which produced eleven spurious X-points on a smooth seeded state
// in the host implementation.
__global__ void grad_scale_kernel(GridView g, const Real* __restrict__ d_r,
                                  const Real* __restrict__ d_z,
                                  Real* __restrict__ partial) {
  __shared__ Real sdata[kThreads];
  const unsigned tid = threadIdx.x;

  const int w = g.nr - 2 * kMargin;
  const int h = g.nz - 2 * kMargin;
  const std::size_t n = static_cast<std::size_t>(w > 0 ? w : 0)
                      * static_cast<std::size_t>(h > 0 ? h : 0);

  Real acc = Real{0};
  for (std::size_t t = blockIdx.x * blockDim.x + tid; t < n;
       t += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const int i = kMargin + static_cast<int>(t % static_cast<std::size_t>(w));
    const int j = kMargin + static_cast<int>(t / static_cast<std::size_t>(w));
    const Real gr = d_r[g.index(i, j)];
    const Real gz = d_z[g.index(i, j)];
    acc = fmax(acc, sqrt(gr * gr + gz * gz));
  }
  sdata[tid] = acc;
  __syncthreads();

  for (unsigned s = kThreads / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0) partial[blockIdx.x] = sdata[0];
}

__global__ void grad_scale_finalize(const Real* __restrict__ partial,
                                    Real* __restrict__ out, unsigned n) {
  __shared__ Real sdata[kThreads];
  const unsigned tid = threadIdx.x;
  Real acc = Real{0};
  for (unsigned k = tid; k < n; k += kThreads) acc = fmax(acc, partial[k]);
  sdata[tid] = acc;
  __syncthreads();
  for (unsigned s = kThreads / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0) out[0] = sdata[0];
}

// One thread per interior node. Writes a valid CriticalPoint into candidates[k]
// if the node is a discrete critical point that survives refinement and the
// gradient test; otherwise leaves candidates[k] invalid.
__global__ void candidate_kernel(GridView g, const Real* __restrict__ psi,
                                 DerivView d, const Real* __restrict__ grad_scale,
                                 CriticalPoint* __restrict__ candidates) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= g.nr || j >= g.nz) return;

  const std::size_t k = g.index(i, j);
  candidates[k] = CriticalPoint{};
  if (i < kMargin || j < kMargin || i >= g.nr - kMargin || j >= g.nz - kMargin) {
    return;
  }

  // Discrete critical point: both derivative components change sign across the
  // node in their respective directions.
  const Real gr_m = d.d_r[g.index(i - 1, j)];
  const Real gr_p = d.d_r[g.index(i + 1, j)];
  const Real gz_m = d.d_z[g.index(i, j - 1)];
  const Real gz_p = d.d_z[g.index(i, j + 1)];
  if (gr_m * gr_p > Real{0} || gz_m * gz_p > Real{0}) return;

  const CriticalPoint cp = refine_dev(g, psi, d, g.r(i), g.z(j), 40);
  if (!cp.valid) return;

  // Reject a refined point whose gradient is not actually small relative to the
  // field's overall variation.
  const Real gr = sample_bilinear_dev(g, d.d_r, cp.r, cp.z);
  const Real gz = sample_bilinear_dev(g, d.d_z, cp.r, cp.z);
  if (sqrt(gr * gr + gz * gz) > Real{1e-3} * grad_scale[0]) return;

  candidates[k] = cp;
}

// Single-threaded, deliberately. Walks the candidate array in the host's
// row-major scan order, merges near-duplicates keeping the first, then selects
// the axis and psi_boundary with the host's tie-breaking.
__global__ void finalize_kernel(GridView g, const Real* __restrict__ psi,
                                const CriticalPoint* __restrict__ candidates,
                                GsCriticalResult* __restrict__ out) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  GsCriticalResult res{};

  // O-points are merged into a local buffer; X-points go straight into the
  // result, which is what the host's `out.x_points = raw_x` does.
  constexpr int kMaxO = GsCriticalResult::kMaxXPoints;
  CriticalPoint o_points[kMaxO];
  int n_o = 0;
  bool o_overflow = false;

  for (int j = kMargin; j < g.nz - kMargin; ++j) {
    for (int i = kMargin; i < g.nr - kMargin; ++i) {
      const CriticalPoint cp = candidates[g.index(i, j)];
      if (!cp.valid) continue;

      const bool is_o = cp.kind == CriticalKind::o_point;
      CriticalPoint* list = is_o ? o_points : res.x_points;
      int& count = is_o ? n_o : res.n_x;

      // First-wins duplicate merge: many neighbouring seeds refine onto the
      // same point.
      bool dup = false;
      for (int q = 0; q < count; ++q) {
        if (fabs(cp.r - list[q].r) < Real{0.5} * g.dr
            && fabs(cp.z - list[q].z) < Real{0.5} * g.dz) {
          dup = true;
          break;
        }
      }
      if (dup) continue;

      if (count >= kMaxO) {
        if (is_o) {
          o_overflow = true;
        } else {
          res.x_point_overflow = true;
        }
        continue;
      }
      list[count++] = cp;
    }
  }
  res.x_point_overflow = res.x_point_overflow || o_overflow;

  if (n_o == 0) {
    *out = res;
    return;
  }

  // Axis: the O-point furthest in psi from the domain edge value. Strict `>`
  // so ties go to the earliest in scan order, matching the host.
  const Real edge_psi = psi[g.index(0, g.nz / 2)];
  int best = 0;
  Real best_depth = Real{-1};
  for (int k = 0; k < n_o; ++k) {
    const Real depth = fabs(o_points[k].psi - edge_psi);
    if (depth > best_depth) {
      best_depth = depth;
      best = k;
    }
  }
  res.axis = o_points[best];
  res.psi_axis = res.axis.psi;

  // psi_boundary: the X-point flux closest to the axis (innermost separatrix).
  bool found = false;
  Real best_x = Real{0};
  Real best_gap = Real{0};
  for (int k = 0; k < res.n_x; ++k) {
    const Real gap = fabs(res.x_points[k].psi - res.psi_axis);
    if (!found || gap < best_gap) {
      found = true;
      best_gap = gap;
      best_x = res.x_points[k].psi;
    }
  }
  if (found) {
    res.psi_boundary = best_x;
    res.has_closed_surface = best_gap > Real{0};
  } else {
    // Limited plasma: no X-point in the domain, so the boundary is set by the
    // outermost surface touching the domain edge. Scanned in the host's
    // row-major order, with strict `<`, so ties again go to the earliest.
    Real limit = psi[g.index(0, 0)];
    Real gap = fabs(limit - res.psi_axis);
    for (int j = 0; j < g.nz; ++j) {
      for (int i = 0; i < g.nr; ++i) {
        if (!g.on_boundary(i, j)) continue;
        const Real v = psi[g.index(i, j)];
        if (fabs(v - res.psi_axis) < gap) {
          gap = fabs(v - res.psi_axis);
          limit = v;
        }
      }
    }
    res.psi_boundary = limit;
    res.has_closed_surface = gap > Real{0};
  }

  *out = res;
}

dim3 grid_2d(int nx, int ny, dim3 block) {
  return dim3((static_cast<unsigned>(nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(ny) + block.y - 1) / block.y);
}

}  // namespace

void GsCriticalScratch::resize(const EllipticGrid& g) {
  grad_scale = backend::DeviceBuffer<Real>{1024};  // block partials, then result
  candidates = backend::DeviceBuffer<CriticalPoint>{g.size()};
  result     = backend::DeviceBuffer<GsCriticalResult>{1};
}

void launch_gs_find_critical_points(const EllipticGrid& g, const Real* d_psi,
                                    const GsDerivativeFields& derivatives,
                                    GsCriticalScratch& scratch,
                                    stream_t stream) {
  auto* s = static_cast<hipStream_t>(stream);
  const GridView gv = view_of(g);

  const int w = g.nr - 2 * kMargin;
  const int h = g.nz - 2 * kMargin;
  const std::size_t n_inset = static_cast<std::size_t>(w > 0 ? w : 0)
                            * static_cast<std::size_t>(h > 0 ? h : 0);
  unsigned blocks = static_cast<unsigned>((n_inset + kThreads - 1) / kThreads);
  if (blocks == 0) blocks = 1;
  if (blocks > 512) blocks = 512;  // partial buffer is 1024 wide

  hipLaunchKernelGGL(grad_scale_kernel, dim3{blocks}, dim3{kThreads}, 0, s, gv,
                     derivatives.d_r.device_ptr(), derivatives.d_z.device_ptr(),
                     scratch.grad_scale.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  // Result lands at the far end of the same buffer, past the block partials.
  Real* grad_scale_out = scratch.grad_scale.device_ptr() + 1023;
  hipLaunchKernelGGL(grad_scale_finalize, dim3{1}, dim3{kThreads}, 0, s,
                     scratch.grad_scale.device_ptr(), grad_scale_out, blocks);
  QUASAR_HIP_CHECK(hipGetLastError());

  const DerivView dv{derivatives.d_r.device_ptr(), derivatives.d_z.device_ptr(),
                     derivatives.d_rr.device_ptr(),
                     derivatives.d_zz.device_ptr(),
                     derivatives.d_rz.device_ptr()};

  const dim3 block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(candidate_kernel, grid_2d(g.nr, g.nz, block), block, 0, s,
                     gv, d_psi, dv, grad_scale_out,
                     scratch.candidates.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  hipLaunchKernelGGL(finalize_kernel, dim3{1}, dim3{1}, 0, s, gv, d_psi,
                     scratch.candidates.device_ptr(),
                     scratch.result.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());
}

GsCriticalResult copy_critical_to_host(const GsCriticalScratch& scratch,
                                       stream_t stream) {
  backend::device_synchronize(stream);
  GsCriticalResult out{};
  scratch.result.copy_to_host(&out, 1);
  return out;
}

}  // namespace quasar::equilibrium
