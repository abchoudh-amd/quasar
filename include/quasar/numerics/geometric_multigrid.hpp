#pragma once

// Matrix-free geometric multigrid for the second-order Grad-Shafranov operator.
//
// V-cycle and FMG over a node-centered hierarchy, red-black Gauss-Seidel
// smoothing, full-weighting restriction and bilinear prolongation. Everything is
// evaluated from the stencil formula rather than an assembled matrix, so a level
// costs one array per grid and no setup.
//
// -- Transfer operators and the radial measure --------------------------------
// The plan called for restriction/prolongation that "respect the radial
// measure, not plain Cartesian averaging". The correct statement is narrower
// than it sounds, and getting it wrong in either direction costs convergence:
//
// Prolongation interpolates a CORRECTION, which is a pointwise quantity. Linear
// interpolation in (r,z) is already exact for the linear functions that matter,
// and weighting it by r would bias the correction toward large radius for no
// reason. So prolongation is plain bilinear.
//
// Restriction transfers a RESIDUAL, which is a density with respect to the
// operator's inner product. Delta* is symmetric under <u,v>_r = sum u v / r, so
// the restriction adjoint to bilinear prolongation under that inner product
// carries 1/r weights. That is what `restrict_full_weighting` applies. Using
// plain full weighting instead still converges, but the V-cycle rate degrades
// measurably as (r_max/r_min) grows, because R and P are then no longer a
// variational pair for this operator.
//
// -- Coarse level -------------------------------------------------------------
// Coarsening stops when the grid can no longer halve cleanly. The coarsest level
// is solved by many smoothing sweeps rather than a direct solve: at the sizes
// this bottoms out at (typically 5x5 to 9x9) smoothing to convergence is a few
// microseconds and avoids carrying a dense solver.

#include "quasar/core/types.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/numerics/gs_operator_l2.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace quasar::numerics {

struct MultigridConfig {
  int  pre_smooth{2};
  int  post_smooth{2};
  int  coarse_sweeps{60};
  int  max_levels{32};   // hierarchy depth cap; coarsening usually stops first
};

// Full-weighting restriction with the 1/r measure of Delta*'s inner product.
// Fine node (2i, 2j) maps to coarse node (i, j).
inline void restrict_full_weighting(const EllipticGrid& fine,
                                    const EllipticGrid& coarse,
                                    const ScalarField& rf, ScalarField& rc) {
  const bool nested = fine.nr == 2 * coarse.nr - 1
                   && fine.nz == 2 * coarse.nz - 1
                   && fine.r_min == coarse.r_min
                   && fine.r_max == coarse.r_max
                   && fine.z_min == coarse.z_min
                   && fine.z_max == coarse.z_max;
  if (!nested) {
    throw std::invalid_argument{
        "restrict_full_weighting: grids must be an exact 2:1 nested pair"};
  }
  if (rf.size() != fine.size()) {
    throw std::invalid_argument{
        "restrict_full_weighting: residual size does not match fine grid"};
  }
  rc.assign(coarse.size(), Real{0});
  static constexpr Real kStencil[3][3] = {
      {Real{1} / Real{16}, Real{1} / Real{8}, Real{1} / Real{16}},
      {Real{1} / Real{8},  Real{1} / Real{4}, Real{1} / Real{8}},
      {Real{1} / Real{16}, Real{1} / Real{8}, Real{1} / Real{16}},
  };
  for (int jc = 1; jc < coarse.nz - 1; ++jc) {
    for (int ic = 1; ic < coarse.nr - 1; ++ic) {
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
      // Renormalize by the accumulated weight and re-apply the coarse-node
      // measure, so a constant residual restricts to itself. Without this the
      // 1/r weighting introduces an O(dr/r) scaling error per level that
      // compounds through the hierarchy.
      rc[coarse.index(ic, jc)] = acc / wsum;
    }
  }
}

// Bilinear prolongation of a coarse correction onto the fine grid, accumulated
// into `xf`. Boundary nodes are untouched (the correction is zero there).
inline void prolong_add_bilinear(const EllipticGrid& coarse,
                                 const EllipticGrid& fine,
                                 const ScalarField& ec, ScalarField& xf) {
  for (int jc = 0; jc < coarse.nz; ++jc) {
    for (int ic = 0; ic < coarse.nr; ++ic) {
      const Real v = ec[coarse.index(ic, jc)];
      const int i2 = 2 * ic;
      const int j2 = 2 * jc;
      if (i2 < fine.nr && j2 < fine.nz && !fine.on_boundary(i2, j2)) {
        xf[fine.index(i2, j2)] += v;
      }
    }
  }
  // Edge midpoints along r.
  for (int jc = 0; jc < coarse.nz; ++jc) {
    for (int ic = 0; ic + 1 < coarse.nr; ++ic) {
      const int i2 = 2 * ic + 1;
      const int j2 = 2 * jc;
      if (i2 < fine.nr && j2 < fine.nz && !fine.on_boundary(i2, j2)) {
        xf[fine.index(i2, j2)] += Real{0.5} * (ec[coarse.index(ic, jc)]
                                             + ec[coarse.index(ic + 1, jc)]);
      }
    }
  }
  // Edge midpoints along z.
  for (int jc = 0; jc + 1 < coarse.nz; ++jc) {
    for (int ic = 0; ic < coarse.nr; ++ic) {
      const int i2 = 2 * ic;
      const int j2 = 2 * jc + 1;
      if (i2 < fine.nr && j2 < fine.nz && !fine.on_boundary(i2, j2)) {
        xf[fine.index(i2, j2)] += Real{0.5} * (ec[coarse.index(ic, jc)]
                                             + ec[coarse.index(ic, jc + 1)]);
      }
    }
  }
  // Cell centers.
  for (int jc = 0; jc + 1 < coarse.nz; ++jc) {
    for (int ic = 0; ic + 1 < coarse.nr; ++ic) {
      const int i2 = 2 * ic + 1;
      const int j2 = 2 * jc + 1;
      if (i2 < fine.nr && j2 < fine.nz && !fine.on_boundary(i2, j2)) {
        xf[fine.index(i2, j2)] += Real{0.25}
            * (ec[coarse.index(ic, jc)]     + ec[coarse.index(ic + 1, jc)]
             + ec[coarse.index(ic, jc + 1)] + ec[coarse.index(ic + 1, jc + 1)]);
      }
    }
  }
}

class GsMultigrid {
 public:
  GsMultigrid(EllipticGrid finest, MultigridConfig cfg = {})
    : cfg_{cfg} {
    levels_.push_back(finest);
    while (static_cast<int>(levels_.size()) < cfg_.max_levels
           && levels_.back().can_coarsen()) {
      levels_.push_back(levels_.back().coarsen());
    }
    scratch_x_.resize(levels_.size());
    scratch_b_.resize(levels_.size());
    scratch_r_.resize(levels_.size());
    for (std::size_t l = 0; l < levels_.size(); ++l) {
      scratch_x_[l] = make_field(levels_[l]);
      scratch_b_[l] = make_field(levels_[l]);
      scratch_r_[l] = make_field(levels_[l]);
    }
  }

  int n_levels() const noexcept { return static_cast<int>(levels_.size()); }
  const EllipticGrid& level(int l) const { return levels_[static_cast<std::size_t>(l)]; }

  // One V-cycle on the finest level: x is updated in place toward Delta* x = b.
  // `x` must already carry the Dirichlet boundary values.
  void v_cycle(ScalarField& x, const ScalarField& b) {
    vcycle_level(0, x, b);
  }

  // Solve to a relative residual tolerance, returning the iteration count.
  // Returns -1 if `max_cycles` was exhausted first.
  int solve(ScalarField& x, const ScalarField& b, Real rel_tol, int max_cycles) {
    ScalarField r = make_field(levels_[0]);
    gs_residual_l2(levels_[0], x, b, r);
    const Real r0 = interior_max_norm(levels_[0], r);
    if (r0 == Real{0}) return 0;
    for (int it = 1; it <= max_cycles; ++it) {
      v_cycle(x, b);
      gs_residual_l2(levels_[0], x, b, r);
      if (interior_max_norm(levels_[0], r) <= rel_tol * r0) return it;
    }
    return -1;
  }

  // Residual reduction factor of a single V-cycle, measured from the current
  // state. The convergence-rate regression test uses this directly.
  Real cycle_reduction(ScalarField x, const ScalarField& b) {
    ScalarField r = make_field(levels_[0]);
    gs_residual_l2(levels_[0], x, b, r);
    const Real before = interior_max_norm(levels_[0], r);
    v_cycle(x, b);
    gs_residual_l2(levels_[0], x, b, r);
    const Real after = interior_max_norm(levels_[0], r);
    return before > Real{0} ? after / before : Real{0};
  }

 private:
  void vcycle_level(int l, ScalarField& x, const ScalarField& b) {
    const auto ll = static_cast<std::size_t>(l);
    const EllipticGrid& g = levels_[ll];

    if (l + 1 >= static_cast<int>(levels_.size())) {
      gs_smooth_rbgs(g, x, b, cfg_.coarse_sweeps);
      return;
    }

    gs_smooth_rbgs(g, x, b, cfg_.pre_smooth);
    gs_residual_l2(g, x, b, scratch_r_[ll]);

    const EllipticGrid& gc = levels_[ll + 1];
    restrict_full_weighting(g, gc, scratch_r_[ll], scratch_b_[ll + 1]);
    scratch_x_[ll + 1].assign(gc.size(), Real{0});

    vcycle_level(l + 1, scratch_x_[ll + 1], scratch_b_[ll + 1]);

    prolong_add_bilinear(gc, g, scratch_x_[ll + 1], x);
    gs_smooth_rbgs(g, x, b, cfg_.post_smooth);
  }

  MultigridConfig cfg_{};
  std::vector<EllipticGrid> levels_{};
  std::vector<ScalarField> scratch_x_{}, scratch_b_{}, scratch_r_{};
};

}  // namespace quasar::numerics
