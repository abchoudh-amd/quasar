#pragma once

// Tensor finite-volume point recovery for nonlinear cell-volume MHD terms.
// MP5/MP7 conserved variables and prescribed backgrounds are cell averages;
// nonlinear EMFs and field-split sources therefore have to be evaluated at
// tensor Gauss points and integrated, just like nonlinear face fluxes.

#include "quasar/core/grid.hpp"
#include "quasar/numerics/finite_volume_quadrature.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

namespace quasar::mhd::detail {

using numerics::MhdBackground;

struct CellMagneticRateView {
  const double *bx, *by, *bz;
};

template <int Count, class Sample>
__device__ inline Real cell_quadrature_weighted_value(
    const Real (&weights)[Count], const Sample& sample) {
  const Real reference = sample(Count / 2);
  bool constant = true;
  numerics::ScaledProductQuotientAccumulator<Count> sum;
  for (int q = 0; q < Count; ++q) {
    const Real value = sample(q);
    constant = constant && value == reference;
    numerics::append_scaled_product_quotient(
        sum, weights[q], value, Real{1}, Real{1});
  }
  return constant ? reference
                  : numerics::finish_scaled_product_quotient_sum(sum);
}

template <class Sample>
__device__ inline Real cell_quadrature_point_value(
    int order, int node, const Sample& sample) {
  if (order >= 7) {
    return cell_quadrature_weighted_value(
        numerics::kMp7TransversePointWeights[node],
        [&](int k) { return sample(k - 3); });
  }
  return cell_quadrature_weighted_value(
      numerics::kMp5TransversePointWeights[node],
      [&](int k) { return sample(k - 2); });
}

template <class Sample>
__device__ inline Real cell_tensor_point_value(
    int order, int node_x, int node_y, const Sample& sample) {
  return cell_quadrature_point_value(order, node_y, [&](int oy) {
    return cell_quadrature_point_value(order, node_x, [&](int ox) {
      return sample(ox, oy);
    });
  });
}

// Recover one component of a cell-volume magnetic rate at a tensor Gauss
// point.  The in-plane components first use the same face-to-cell collocation
// as the evolved magnetic field; the out-of-plane component is already a cell
// average.  This makes the split-energy change of variables use the final CT
// rate on its native staggering without silently dropping high-order moments.
__device__ inline Real point_cell_magnetic_rate_component(
    const Grid2D& g, int i, int j, int order, int node_x, int node_y,
    const CellMagneticRateView& rate, int component) {
  return cell_tensor_point_value(order, node_x, node_y, [&](int ox, int oy) {
    if (component == 0) {
      return cell_bx(g, rate.bx, i + ox, j + oy);
    }
    if (component == 1) {
      return cell_by(g, rate.by, i + ox, j + oy);
    }
    return static_cast<Real>(rate.bz[g.index(i + ox, j + oy)]);
  });
}

template <class Component>
__device__ inline Real point_cell_background_component(
    const Grid2D& g, int i, int j, int order, int node_x, int node_y,
    const double* b0x, const double* b0y, const double* b0z,
    const Component& component) {
  return cell_tensor_point_value(order, node_x, node_y, [&](int ox, int oy) {
    return component(load_cell_background(
        g, b0x, b0y, b0z, i + ox, j + oy));
  });
}

__device__ inline MhdBackground point_cell_background(
    const Grid2D& g, int i, int j, int order, int node_x, int node_y,
    const double* b0x, const double* b0y, const double* b0z) {
  return MhdBackground{
      point_cell_background_component(
          g, i, j, order, node_x, node_y, b0x, b0y, b0z,
          [](const MhdBackground& value) { return value.b0x; }),
      point_cell_background_component(
          g, i, j, order, node_x, node_y, b0x, b0y, b0z,
          [](const MhdBackground& value) { return value.b0y; }),
      point_cell_background_component(
          g, i, j, order, node_x, node_y, b0x, b0y, b0z,
          [](const MhdBackground& value) { return value.b0z; })};
}

// Return <a*b>-a_mean*b_mean over one cell tensor quadrature.  Constant-factor
// detection is an exact invariant, not just an optimization: a uniform CT rate
// beside a dominant varying background must have bit-zero covariance even when
// the stored Gauss weights do not sum to one in the evaluation order.
template <class ASample, class BSample>
__device__ inline numerics::ScaledValue cell_tensor_product_correction_scaled(
    int order, Real a_mean, Real b_mean,
    const ASample& a, const BSample& b) {
  const int count = order == 5 ? 3 : 4;
  const Real a_reference = a(count / 2, count / 2);
  const Real b_reference = b(count / 2, count / 2);
  bool a_constant = true;
  bool b_constant = true;
  numerics::ScaledQuaternaryAccumulator sum;
  for (int qy = 0; qy < count; ++qy) {
    const Real wy = order == 5
        ? numerics::kMp5TransverseGaussWeights[qy]
        : numerics::kMp7TransverseGaussWeights[qy];
    for (int qx = 0; qx < count; ++qx) {
      const Real wx = order == 5
          ? numerics::kMp5TransverseGaussWeights[qx]
          : numerics::kMp7TransverseGaussWeights[qx];
      const Real aq = a(qx, qy);
      const Real bq = b(qx, qy);
      a_constant = a_constant && aq == a_reference;
      b_constant = b_constant && bq == b_reference;
      numerics::append_scaled_quaternary_product(sum, aq, bq, wx, wy);
    }
  }
  if (a_constant || b_constant) return {};
  numerics::append_scaled_quaternary_product(
      sum, Real{-1}, a_mean, b_mean, Real{1});
  return numerics::finish_scaled_quaternary_sum_to_value(sum);
}

}  // namespace quasar::mhd::detail
