// Device geometric multigrid and defect correction for Delta* psi = b.
//
// Second-order multigrid is the preconditioner; sixth-order accuracy comes from
// iterating the L6 residual against it. The two operators deliberately disagree
// at O(h^2) -- that discrepancy IS the defect being corrected.
//
// -- Why all of this is bit-exact ----------------------------------------------
// This was the stage most likely to force a tolerance, and it did not, because
// the host implementation was already written in parallel-friendly form:
//
//   * The smoother is RED-BLACK Gauss-Seidel. Within one colour every node
//     reads only the other colour, which is frozen for the duration of the
//     half-sweep, so one thread per node produces exactly the sequential
//     result. Lexicographic GS would have been genuinely sequential and would
//     have forced either a different smoother or a tolerance.
//   * Bilinear prolongation appears to accumulate -- four passes each doing
//     `+=` into the fine field -- but the passes are disjoint by the parity of
//     (i, j): even/even, odd/even, even/odd, odd/odd. Every fine node receives
//     exactly one contribution, so there is no accumulation order to preserve
//     and one thread per fine node can select its own case by parity.
//   * Restriction accumulates nine terms per coarse node, and the kernel keeps
//     the host's (dj, di) traversal so the nine additions round identically.
//
// -- What stays on the host ----------------------------------------------------
// The level recursion. It performs no arithmetic -- it launches kernels and
// picks buffers -- so there is nothing to move to the device. The same applies
// to the defect-correction outer loop, whose convergence test does read a
// scalar back each iteration; that synchronization is kept because dropping it
// would change the iteration count and therefore the result.

#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace quasar::equilibrium {
namespace {

inline constexpr unsigned kBlock2D = 16;

struct GridView {
  int nr;
  int nz;
  Real r_min;
  Real dr;
  Real dz;

  __device__ Real r(int i) const { return r_min + static_cast<Real>(i) * dr; }
  __device__ Real r_half(int i) const {
    return r_min + (static_cast<Real>(i) + Real{0.5}) * dr;
  }
  __device__ bool on_boundary(int i, int j) const {
    return i == 0 || j == 0 || i == nr - 1 || j == nz - 1;
  }
  __device__ std::size_t index(int i, int j) const {
    return static_cast<std::size_t>(i) + static_cast<std::size_t>(nr) * j;
  }
};

GridView view_of(const EllipticGrid& g) {
  return GridView{g.nr, g.nz, g.r_min, g.dr(), g.dz()};
}

struct Stencil {
  Real w_center, w_rm, w_rp, w_zm, w_zp;
};

// Conservative half-node form. Delta* is not the Poisson operator: expanding it
// naively gives a nonsymmetric matrix whose symmetry defect is O(dr/r), which
// costs red-black smoothing its predictable rate. This form is symmetric in the
// 1/r-weighted inner product, which is the correct one for the operator.
__device__ inline Stencil stencil_at(const GridView& g, int i) {
  const Real inv_dr2 = Real{1} / (g.dr * g.dr);
  const Real inv_dz2 = Real{1} / (g.dz * g.dz);
  const Real r_i = g.r(i);

  const Real a_p = r_i / g.r_half(i) * inv_dr2;
  const Real a_m = r_i / g.r_half(i - 1) * inv_dr2;

  Stencil s;
  s.w_rp = a_p;
  s.w_rm = a_m;
  s.w_zp = inv_dz2;
  s.w_zm = inv_dz2;
  s.w_center = -(a_p + a_m + Real{2} * inv_dz2);
  return s;
}

// One colour of a red-black sweep. `x` is read and written, but every thread
// touches only its own node and reads only the opposite colour, so the aliasing
// is benign and the result matches a sequential sweep exactly.
__global__ void smooth_colour_kernel(GridView g, Real* x,
                                     const Real* __restrict__ b, int colour) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + 1;
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) + 1;
  if (i >= g.nr - 1 || j >= g.nz - 1) return;
  if (((i + j) & 1) != colour) return;

  const Stencil s = stencil_at(g, i);
  const std::size_t k = g.index(i, j);
  const Real off = s.w_rm * x[g.index(i - 1, j)]
                 + s.w_rp * x[g.index(i + 1, j)]
                 + s.w_zm * x[g.index(i, j - 1)]
                 + s.w_zp * x[g.index(i, j + 1)];
  x[k] = (b[k] - off) / s.w_center;
}

__global__ void residual_l2_kernel(GridView g, const Real* __restrict__ x,
                                   const Real* __restrict__ b,
                                   Real* __restrict__ r) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= g.nr || j >= g.nz) return;
  const std::size_t k = g.index(i, j);
  if (g.on_boundary(i, j)) {
    r[k] = Real{0};
    return;
  }
  const Stencil s = stencil_at(g, i);
  const Real ax = s.w_center * x[k]
                + s.w_rm * x[g.index(i - 1, j)]
                + s.w_rp * x[g.index(i + 1, j)]
                + s.w_zm * x[g.index(i, j - 1)]
                + s.w_zp * x[g.index(i, j + 1)];
  r[k] = b[k] - ax;
}

// Full-weighting restriction with the 1/r measure. The nine contributions are
// accumulated in the host's (dj, di) order so the roundings match.
__global__ void restrict_kernel(GridView fine, GridView coarse,
                                const Real* __restrict__ rf,
                                Real* __restrict__ rc) {
  const int ic = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int jc = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (ic >= coarse.nr || jc >= coarse.nz) return;

  const std::size_t kc = coarse.index(ic, jc);
  if (ic == 0 || jc == 0 || ic >= coarse.nr - 1 || jc >= coarse.nz - 1) {
    rc[kc] = Real{0};
    return;
  }

  constexpr Real kStencil[3][3] = {
      {Real{1} / Real{16}, Real{1} / Real{8}, Real{1} / Real{16}},
      {Real{1} / Real{8},  Real{1} / Real{4}, Real{1} / Real{8}},
      {Real{1} / Real{16}, Real{1} / Real{8}, Real{1} / Real{16}},
  };

  const int i2 = 2 * ic;
  const int j2 = 2 * jc;
  Real acc = Real{0};
  Real wsum = Real{0};
  for (int dj = -1; dj <= 1; ++dj) {
    for (int di = -1; di <= 1; ++di) {
      const Real w = kStencil[dj + 1][di + 1] / fine.r(i2 + di);
      acc  += w * rf[fine.index(i2 + di, j2 + dj)];
      wsum += w;
    }
  }
  // Renormalizing by the accumulated weight is what makes a constant residual
  // restrict to itself; without it the 1/r weighting introduces an O(dr/r)
  // scaling error per level that compounds through the hierarchy.
  rc[kc] = acc / wsum;
}

// Bilinear prolongation, accumulated into the fine field.
//
// One thread per FINE node, selecting its case by parity. The host writes this
// as four separate passes over the coarse grid, but those passes are disjoint
// by (i, j) parity, so this inversion is exact rather than merely equivalent.
__global__ void prolong_add_kernel(GridView coarse, GridView fine,
                                   const Real* __restrict__ ec,
                                   Real* __restrict__ xf) {
  const int i2 = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j2 = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i2 >= fine.nr || j2 >= fine.nz) return;
  if (fine.on_boundary(i2, j2)) return;  // correction is zero on the boundary

  const int ic = i2 / 2;
  const int jc = j2 / 2;
  const bool odd_i = (i2 & 1) != 0;
  const bool odd_j = (j2 & 1) != 0;

  Real v;
  if (!odd_i && !odd_j) {
    v = ec[coarse.index(ic, jc)];
  } else if (odd_i && !odd_j) {
    if (ic + 1 >= coarse.nr) return;
    v = Real{0.5} * (ec[coarse.index(ic, jc)] + ec[coarse.index(ic + 1, jc)]);
  } else if (!odd_i && odd_j) {
    if (jc + 1 >= coarse.nz) return;
    v = Real{0.5} * (ec[coarse.index(ic, jc)] + ec[coarse.index(ic, jc + 1)]);
  } else {
    if (ic + 1 >= coarse.nr || jc + 1 >= coarse.nz) return;
    v = Real{0.25} * (ec[coarse.index(ic, jc)]
                    + ec[coarse.index(ic + 1, jc)]
                    + ec[coarse.index(ic, jc + 1)]
                    + ec[coarse.index(ic + 1, jc + 1)]);
  }
  xf[fine.index(i2, j2)] += v;
}

// x += relaxation * e, interior only. The correction is zero on the boundary,
// so the Dirichlet data in x is preserved.
__global__ void apply_correction_kernel(GridView g, Real* __restrict__ x,
                                        const Real* __restrict__ e,
                                        Real relaxation) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + 1;
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) + 1;
  if (i >= g.nr - 1 || j >= g.nz - 1) return;
  const std::size_t k = g.index(i, j);
  x[k] += relaxation * e[k];
}

// rhs_shift = residual + jac * delta, interior only. This is the right-hand
// side of the Newton system after the diagonal shift has been moved across.
__global__ void newton_shift_kernel(GridView g,
                                    const Real* __restrict__ residual,
                                    const Real* __restrict__ jac,
                                    const Real* __restrict__ delta,
                                    Real* __restrict__ shift) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + 1;
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) + 1;
  if (i >= g.nr - 1 || j >= g.nz - 1) return;
  const std::size_t k = g.index(i, j);
  shift[k] = residual[k] + jac[k] * delta[k];
}

dim3 grid_2d(int nx, int ny, dim3 block) {
  return dim3((static_cast<unsigned>(nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(ny) + block.y - 1) / block.y);
}

void smooth(const EllipticGrid& g, Real* d_x, const Real* d_b, int sweeps,
            hipStream_t s) {
  const dim3 block{kBlock2D, kBlock2D};
  const dim3 grid = grid_2d(g.nr - 2, g.nz - 2, block);
  for (int n = 0; n < sweeps; ++n) {
    for (int colour = 0; colour < 2; ++colour) {
      hipLaunchKernelGGL(smooth_colour_kernel, grid, block, 0, s, view_of(g),
                         d_x, d_b, colour);
      QUASAR_HIP_CHECK(hipGetLastError());
    }
  }
}

}  // namespace

GsDeviceMultigrid::GsDeviceMultigrid(const EllipticGrid& finest,
                                     GsMultigridConfig cfg)
  : cfg_{cfg} {
  levels_.push_back(finest);
  while (static_cast<int>(levels_.size()) < cfg_.max_levels
         && levels_.back().can_coarsen()) {
    levels_.push_back(levels_.back().coarsen());
  }
  x_.resize(levels_.size());
  b_.resize(levels_.size());
  r_.resize(levels_.size());
  for (std::size_t l = 0; l < levels_.size(); ++l) {
    x_[l] = backend::DeviceBuffer<Real>{levels_[l].size()};
    b_[l] = backend::DeviceBuffer<Real>{levels_[l].size()};
    r_[l] = backend::DeviceBuffer<Real>{levels_[l].size()};
  }
}

void GsDeviceMultigrid::v_cycle(Real* d_x, const Real* d_b, stream_t stream) {
  if (levels_.empty()) {
    throw std::logic_error{"GsDeviceMultigrid::v_cycle: hierarchy not built"};
  }
  vcycle_level(0, d_x, d_b, stream);
}

void GsDeviceMultigrid::vcycle_level(int l, Real* d_x, const Real* d_b,
                                     stream_t stream) {
  auto* s = static_cast<hipStream_t>(stream);
  const auto ll = static_cast<std::size_t>(l);
  const EllipticGrid& g = levels_[ll];

  // Coarsest level: smooth to convergence rather than solve directly. At the
  // sizes this bottoms out at (5x5 to 9x9) that is a few microseconds and
  // avoids carrying a direct solver for the sake of the smallest grid.
  if (l + 1 >= static_cast<int>(levels_.size())) {
    smooth(g, d_x, d_b, cfg_.coarse_sweeps, s);
    return;
  }

  smooth(g, d_x, d_b, cfg_.pre_smooth, s);

  const dim3 block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(residual_l2_kernel, grid_2d(g.nr, g.nz, block), block, 0, s,
                     view_of(g), d_x, d_b, r_[ll].device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  const EllipticGrid& gc = levels_[ll + 1];
  hipLaunchKernelGGL(restrict_kernel, grid_2d(gc.nr, gc.nz, block), block, 0, s,
                     view_of(g), view_of(gc), r_[ll].device_ptr(),
                     b_[ll + 1].device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  // The coarse correction starts from zero, with zero boundary data, so it
  // cannot perturb the Dirichlet values held on the fine level.
  QUASAR_HIP_CHECK(hipMemsetAsync(x_[ll + 1].device_ptr(), 0,
                                  x_[ll + 1].bytes(), s));

  vcycle_level(l + 1, x_[ll + 1].device_ptr(), b_[ll + 1].device_ptr(), stream);

  hipLaunchKernelGGL(prolong_add_kernel, grid_2d(g.nr, g.nz, block), block, 0, s,
                     view_of(gc), view_of(g), x_[ll + 1].device_ptr(), d_x);
  QUASAR_HIP_CHECK(hipGetLastError());

  smooth(g, d_x, d_b, cfg_.post_smooth, s);
}

void launch_gs_newton_correction(const EllipticGrid& g, const Real* d_jac,
                                 const Real* d_residual, Real* d_delta,
                                 GsDeviceMultigrid& mg, stream_t stream) {
  auto* s = static_cast<hipStream_t>(stream);

  backend::DeviceBuffer<Real> d_work{g.size()};
  backend::DeviceBuffer<Real> d_shift{g.size()};
  QUASAR_HIP_CHECK(hipMemsetAsync(d_delta, 0, g.size() * sizeof(Real), s));

  const dim3 block{kBlock2D, kBlock2D};

  // The shifted operator [Delta* - diag(jac)] is not the plain Delta* the
  // multigrid hierarchy was built for, so the shift is carried as a
  // defect-corrected outer iteration around the existing V-cycle rather than by
  // rebuilding the hierarchy. The shift is small against the operator diagonal,
  // so a handful of passes suffices.
  for (int pass = 0; pass < 4; ++pass) {
    // Residual convention here is r = b - L6(psi), so L6(psi) - b = -r and the
    // Newton system [L6 - J] delta = -(L6(psi) - b) has right-hand side +r.
    // Moving the diagonal shift across gives Delta* delta = r + jac*delta.
    hipLaunchKernelGGL(newton_shift_kernel, grid_2d(g.nr - 2, g.nz - 2, block),
                       block, 0, s, view_of(g), d_residual, d_jac, d_delta,
                       d_shift.device_ptr());
    QUASAR_HIP_CHECK(hipGetLastError());

    QUASAR_HIP_CHECK(hipMemsetAsync(d_work.device_ptr(), 0, d_work.bytes(), s));
    for (int c = 0; c < 3; ++c) {
      mg.v_cycle(d_work.device_ptr(), d_shift.device_ptr(), stream);
    }
    QUASAR_HIP_CHECK(hipMemcpyAsync(d_delta, d_work.device_ptr(),
                                    g.size() * sizeof(Real),
                                    hipMemcpyDeviceToDevice, s));
  }
}

Real launch_gs_newton_line_search(const EllipticGrid& g, const Real* d_current,
                                  const Real* d_delta, const Real* d_rhs,
                                  Real* d_trial, GsOperatorScratch& op_scratch,
                                  GsReduceScratch& reduce_scratch,
                                  stream_t stream) {
  auto* s = static_cast<hipStream_t>(stream);
  backend::DeviceBuffer<Real> d_resid{g.size()};

  launch_gs_residual_l6(g, d_current, d_rhs, d_resid.device_ptr(), op_scratch,
                        stream);
  launch_gs_interior_max_norm(g, d_resid.device_ptr(), reduce_scratch, stream);
  const Real base = copy_scalar_to_host(reduce_scratch, stream);

  for (const Real alpha : {Real{1}, Real{0.5}, Real{0.25}, Real{0.125}}) {
    QUASAR_HIP_CHECK(hipMemcpyAsync(d_trial, d_current, g.size() * sizeof(Real),
                                    hipMemcpyDeviceToDevice, s));
    launch_gs_axpy_interior(g, d_trial, d_delta, alpha, stream);
    launch_gs_residual_l6(g, d_trial, d_rhs, d_resid.device_ptr(), op_scratch,
                          stream);
    launch_gs_interior_max_norm(g, d_resid.device_ptr(), reduce_scratch,
                                stream);
    if (copy_scalar_to_host(reduce_scratch, stream) < base) return alpha;
  }
  return Real{0};
}

GsDefectCorrectionReport launch_gs_solve_defect_corrected(
    const EllipticGrid& g, Real* d_x, const Real* d_b, GsDeviceMultigrid& mg,
    GsOperatorScratch& op_scratch, GsReduceScratch& reduce_scratch,
    GsDefectCorrectionConfig cfg, stream_t stream) {
  auto* s = static_cast<hipStream_t>(stream);
  GsDefectCorrectionReport rep;

  // Correction and residual scratch. Allocated per call rather than held in a
  // *Scratch struct because the defect-correction solve is once per Picard
  // step, not once per node -- two grid-sized allocations there are noise
  // against the Green's-function integral in the same step.
  backend::DeviceBuffer<Real> d_r{g.size()};
  backend::DeviceBuffer<Real> d_e{g.size()};

  launch_gs_residual_l6(g, d_x, d_b, d_r.device_ptr(), op_scratch, stream);
  launch_gs_interior_max_norm(g, d_r.device_ptr(), reduce_scratch, stream);
  rep.initial_residual = copy_scalar_to_host(reduce_scratch, stream);
  rep.final_residual = rep.initial_residual;

  if (!std::isfinite(rep.initial_residual)
      || rep.initial_residual < Real{0}) {
    return rep;
  }
  if (rep.initial_residual == Real{0}) {
    rep.converged = true;
    return rep;
  }

  const dim3 block{kBlock2D, kBlock2D};
  for (int it = 1; it <= cfg.max_iterations; ++it) {
    QUASAR_HIP_CHECK(hipMemsetAsync(d_e.device_ptr(), 0, d_e.bytes(), s));
    for (int c = 0; c < cfg.inner_cycles; ++c) {
      mg.v_cycle(d_e.device_ptr(), d_r.device_ptr(), stream);
    }

    hipLaunchKernelGGL(apply_correction_kernel, grid_2d(g.nr - 2, g.nz - 2, block),
                       block, 0, s, view_of(g), d_x, d_e.device_ptr(),
                       cfg.relaxation);
    QUASAR_HIP_CHECK(hipGetLastError());

    launch_gs_residual_l6(g, d_x, d_b, d_r.device_ptr(), op_scratch, stream);
    launch_gs_interior_max_norm(g, d_r.device_ptr(), reduce_scratch, stream);
    rep.final_residual = copy_scalar_to_host(reduce_scratch, stream);
    rep.iterations = it;

    if (!std::isfinite(rep.final_residual)
        || rep.final_residual < Real{0}) {
      break;
    }
    if (rep.final_residual <= cfg.tolerance * rep.initial_residual) {
      rep.converged = true;
      break;
    }
  }
  return rep;
}

}  // namespace quasar::equilibrium
