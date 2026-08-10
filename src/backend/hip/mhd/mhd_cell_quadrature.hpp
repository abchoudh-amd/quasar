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

struct CellStateView {
  const double *rho, *mx, *my, *mz, *energy, *bx, *by, *bz;
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
    int order, int node_x, int node_y, int radial_index,
    numerics::RadialTablesView radial_tables, const Sample& sample) {
  return cell_quadrature_point_value(order, node_y, [&](int oy) {
    if (radial_tables.active != 0 &&
        radial_tables.contains(radial_index)) {
      const int width = order >= 7 ? 7 : 5;
      const Real* weights = radial_tables.r2_row(radial_index, node_x);
      const Real reference = sample(0, oy);
      bool constant = true;
      numerics::ScaledProductQuotientAccumulator<8> sum;
      for (int k = 0; k < width; ++k) {
        const Real value = sample(k - width / 2, oy);
        constant = constant && value == reference;
        numerics::append_scaled_product_quotient(
            sum, weights[k], value, Real{1}, Real{1});
      }
      return constant ? reference
          : numerics::finish_scaled_product_quotient_sum(sum);
    }
    return cell_quadrature_point_value(order, node_x, [&](int ox) {
      return sample(ox, oy);
    });
  });
}

template <class Sample>
__device__ inline Real cell_tensor_point_value(
    int order, int node_x, int node_y, const Sample& sample) {
  return cell_tensor_point_value(
      order, node_x, node_y, /*radial_index=*/0,
      numerics::RadialTablesView{}, sample);
}

// Recover one component of a cell-volume magnetic rate at a tensor Gauss
// point.  The in-plane components first use the same face-to-cell collocation
// as the evolved magnetic field; the out-of-plane component is already a cell
// average.  This makes the split-energy change of variables use the final CT
// rate on its native staggering without silently dropping high-order moments.
__device__ inline Real point_cell_magnetic_rate_component(
    const Grid2D& g, int i, int j, int order, int node_x, int node_y,
    const CellMagneticRateView& rate, int component,
    numerics::RadialTablesView radial_tables) {
  return cell_tensor_point_value(
      order, node_x, node_y, i, radial_tables, [&](int ox, int oy) {
    if (component == 0) {
      return cell_bx(g, rate.bx, i + ox, j + oy, radial_tables);
    }
    if (component == 1) {
      return cell_by(g, rate.by, i + ox, j + oy, radial_tables);
    }
    return static_cast<Real>(rate.bz[g.index(i + ox, j + oy)]);
  });
}

// Recover one complete conserved state at a tensor Gauss point. Cell-centred
// components use the radial R2 rows and the Cartesian axial point rows; the CT
// components are first collocated to ring-cell averages with their native R4 /
// Cartesian face-to-cell rules. This is the volume counterpart of the point
// face states used by transverse HLLD quadrature.
__device__ inline numerics::MhdState point_cell_state(
    const Grid2D& g, int i, int j, int order, int node_x, int node_y,
    const CellStateView& state, numerics::RadialTablesView radial_tables,
    int collocation_order = 0) {
  const auto cell_component = [&](const double* values) {
    return cell_tensor_point_value(
        order, node_x, node_y, i, radial_tables, [&](int ox, int oy) {
          return static_cast<Real>(values[g.index(i + ox, j + oy)]);
        });
  };
  const auto magnetic_component = [&](int component) {
    return cell_tensor_point_value(
        order, node_x, node_y, i, radial_tables, [&](int ox, int oy) {
          if (component == 0) {
            return cell_bx(g, state.bx, i + ox, j + oy, radial_tables,
                           collocation_order);
          }
          if (component == 1) {
            return cell_by(g, state.by, i + ox, j + oy, radial_tables,
                           collocation_order);
          }
          return static_cast<Real>(state.bz[g.index(i + ox, j + oy)]);
        });
  };
  return numerics::MhdState{
      cell_component(state.rho), cell_component(state.mx),
      cell_component(state.my), cell_component(state.mz),
      cell_component(state.energy), magnetic_component(0),
      magnetic_component(1), magnetic_component(2)};
}

template <class Component>
__device__ inline Real point_cell_background_component(
    const Grid2D& g, int i, int j, int order, int node_x, int node_y,
    const double* b0x, const double* b0y, const double* b0z,
    const Component& component,
    numerics::RadialTablesView radial_tables) {
  return cell_tensor_point_value(
      order, node_x, node_y, i, radial_tables, [&](int ox, int oy) {
    return component(load_cell_background(
        g, b0x, b0y, b0z, i + ox, j + oy, radial_tables));
  });
}

__device__ inline MhdBackground point_cell_background(
    const Grid2D& g, int i, int j, int order, int node_x, int node_y,
    const double* b0x, const double* b0y, const double* b0z,
    numerics::RadialTablesView radial_tables) {
  return MhdBackground{
      point_cell_background_component(
          g, i, j, order, node_x, node_y, b0x, b0y, b0z,
          [](const MhdBackground& value) { return value.b0x; }, radial_tables),
      point_cell_background_component(
          g, i, j, order, node_x, node_y, b0x, b0y, b0z,
          [](const MhdBackground& value) { return value.b0y; }, radial_tables),
      point_cell_background_component(
          g, i, j, order, node_x, node_y, b0x, b0y, b0z,
          [](const MhdBackground& value) { return value.b0z; }, radial_tables)};
}

// Return <a*b>-a_mean*b_mean over one cell tensor quadrature.  Constant-factor
// detection is an exact invariant, not just an optimization: a uniform CT rate
// beside a dominant varying background must have bit-zero covariance even when
// the stored Gauss weights do not sum to one in the evaluation order.
template <class ASample, class BSample>
__device__ inline numerics::ScaledValue cell_tensor_product_correction_scaled(
    int order, int radial_index, numerics::RadialTablesView radial_tables,
    Real a_mean, Real b_mean,
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
      const Real wx = radial_tables.active != 0 &&
              radial_tables.contains(radial_index)
          ? radial_tables.r3_row(radial_index)[qx]
          : order == 5 ? numerics::kMp5TransverseGaussWeights[qx]
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

template <class ASample, class BSample>
__device__ inline numerics::ScaledValue cell_tensor_product_correction_scaled(
    int order, Real a_mean, Real b_mean,
    const ASample& a, const BSample& b) {
  return cell_tensor_product_correction_scaled(
      order, /*radial_index=*/0, numerics::RadialTablesView{},
      a_mean, b_mean, a, b);
}

}  // namespace quasar::mhd::detail
