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

template <class Sample>
__device__ inline Real cell_quadrature_point_value(
    int order, int node, const Sample& sample) {
  if (order >= 7) {
    return numerics::weighted_quadrature_value(
        numerics::kMp7TransversePointWeights[node],
        [&](int k) { return sample(k - 3); });
  }
  return numerics::weighted_quadrature_value(
      numerics::kMp5TransversePointWeights[node],
      [&](int k) { return sample(k - 2); });
}

template <bool UseRadialTables, class Sample>
__device__ inline Real cell_tensor_point_value_with_radial_row(
    int order, int node_x, int node_y, int radial_index,
    numerics::RadialTablesView radial_tables,
    const Real* component_radial_row, const Sample& sample) {
  return cell_quadrature_point_value(order, node_y, [&](int oy) {
    if constexpr (UseRadialTables) {
      if (radial_tables.active != 0 &&
          radial_tables.contains(radial_index)) {
        const int width = order >= 7 ? 7 : 5;
        const Real* weights = component_radial_row != nullptr
            ? component_radial_row
            : radial_tables.r2_row(radial_index, node_x);
        return numerics::weighted_quadrature_value(
            weights, width,
            [&](int k) { return sample(k - width / 2, oy); });
      }
    }
    return cell_quadrature_point_value(order, node_x, [&](int ox) {
      return sample(ox, oy);
    });
  });
}

template <class Sample>
__device__ inline Real cell_tensor_point_value(
    int order, int node_x, int node_y, int radial_index,
    numerics::RadialTablesView radial_tables, const Sample& sample) {
  return cell_tensor_point_value_with_radial_row<true>(
      order, node_x, node_y, radial_index, radial_tables,
      /*component_radial_row=*/nullptr, sample);
}

template <class Sample>
__device__ inline Real cell_tensor_point_value(
    int order, int node_x, int node_y, const Sample& sample) {
  return cell_tensor_point_value_with_radial_row<false>(
      order, node_x, node_y, /*radial_index=*/0,
      numerics::RadialTablesView{}, /*component_radial_row=*/nullptr,
      sample);
}

// Recover one component of a cell-volume magnetic rate at a tensor Gauss
// point.  The in-plane components first use the same face-to-cell collocation
// as the evolved magnetic field; the out-of-plane component is already a cell
// average.  This makes the split-energy change of variables use the final CT
// rate on its native staggering without silently dropping high-order moments.
template <bool UseRadialTables>
__device__ inline Real point_cell_magnetic_rate_component(
    const Grid2D& g, int i, int j, int order, int node_x, int node_y,
    const CellMagneticRateView& rate, int component,
    numerics::RadialTablesView radial_tables) {
  const Real* radial_row = UseRadialTables && component == 2 &&
          radial_tables.active != 0 && radial_tables.contains(i)
      ? radial_tables.r2_bphi_row(i, node_x) : nullptr;
  return cell_tensor_point_value_with_radial_row<UseRadialTables>(
      order, node_x, node_y, i, radial_tables, radial_row,
      [&](int ox, int oy) {
    if (component == 0) {
      if constexpr (UseRadialTables) {
        return cell_bx(g, rate.bx, i + ox, j + oy, radial_tables);
      }
      return cell_bx(g, rate.bx, i + ox, j + oy);
    }
    if (component == 1) {
      if constexpr (UseRadialTables) {
        return cell_by(g, rate.by, i + ox, j + oy, radial_tables);
      }
      return cell_by(g, rate.by, i + ox, j + oy);
    }
    return static_cast<Real>(rate.bz[g.index(i + ox, j + oy)]);
  });
}

__device__ inline Real point_cell_magnetic_rate_component(
    const Grid2D& g, int i, int j, int order, int node_x, int node_y,
    const CellMagneticRateView& rate, int component,
    numerics::RadialTablesView radial_tables) {
  return point_cell_magnetic_rate_component<true>(
      g, i, j, order, node_x, node_y, rate, component, radial_tables);
}

__device__ inline Real point_cell_state_component(
    const Grid2D& g, int i, int j, int order, int node_x, int node_y,
    const CellStateView& state, int component,
    numerics::RadialTablesView radial_tables,
    int collocation_order = 0) {
  const bool radial_recovery =
      radial_tables.active != 0 && radial_tables.contains(i);
  const Real* radial_row = radial_recovery
      ? component == 3 ? radial_tables.r2_mphi_row(i, node_x)
                       : component == 7
                             ? radial_tables.r2_bphi_row(i, node_x) : nullptr
      : nullptr;
  const double* cell_values = component == 0 ? state.rho
      : component == 1 ? state.mx
      : component == 2 ? state.my
      : component == 3 ? state.mz
      : component == 4 ? state.energy
      : component == 7 ? state.bz : nullptr;
  return cell_tensor_point_value_with_radial_row<true>(
      order, node_x, node_y, i, radial_tables, radial_row,
      [&](int ox, int oy) {
        if (component == 5) {
          return cell_bx(g, state.bx, i + ox, j + oy, radial_tables,
                         collocation_order);
        }
        if (component == 6) {
          return cell_by(g, state.by, i + ox, j + oy, radial_tables,
                         collocation_order);
        }
        return static_cast<Real>(cell_values[g.index(i + ox, j + oy)]);
      });
}

// Recover one complete conserved state at a tensor Gauss point.  Cell-centred
// m_phi and B_phi use the measures of their evolution equations (r^2 dr and
// dr); all other components retain annular |r| dr recovery.
__device__ inline numerics::MhdState point_cell_state(
    const Grid2D& g, int i, int j, int order, int node_x, int node_y,
    const CellStateView& state, numerics::RadialTablesView radial_tables,
    int collocation_order = 0) {
  return numerics::MhdState{
      point_cell_state_component(
          g, i, j, order, node_x, node_y, state, 0, radial_tables,
          collocation_order),
      point_cell_state_component(
          g, i, j, order, node_x, node_y, state, 1, radial_tables,
          collocation_order),
      point_cell_state_component(
          g, i, j, order, node_x, node_y, state, 2, radial_tables,
          collocation_order),
      point_cell_state_component(
          g, i, j, order, node_x, node_y, state, 3, radial_tables,
          collocation_order),
      point_cell_state_component(
          g, i, j, order, node_x, node_y, state, 4, radial_tables,
          collocation_order),
      point_cell_state_component(
          g, i, j, order, node_x, node_y, state, 5, radial_tables,
          collocation_order),
      point_cell_state_component(
          g, i, j, order, node_x, node_y, state, 6, radial_tables,
          collocation_order),
      point_cell_state_component(
          g, i, j, order, node_x, node_y, state, 7, radial_tables,
          collocation_order)};
}

// Reduce all three background components in one pass.  Besides sharing the
// stencil traversal, reuse the centre sample used by the constant-field check;
// the scalar helper otherwise evaluates that expensive collocation twice.
template <class Sample>
__device__ inline MhdBackground weighted_background_value(
    const Real* weights, const Real* bphi_weights, int count,
    const Sample& sample) {
  const int centre = count / 2;
  const MhdBackground reference = sample(centre);
  bool constant[3] = {true, true, true};
  numerics::ScaledProductQuotientAccumulator<8> sums[3];
  for (int q = 0; q < count; ++q) {
    const MhdBackground value = q == centre ? reference : sample(q);
    const Real components[3] = {value.b0x, value.b0y, value.b0z};
    const Real reference_components[3] = {
        reference.b0x, reference.b0y, reference.b0z};
    for (int component = 0; component < 3; ++component) {
      constant[component] =
          constant[component] && components[component] ==
                                     reference_components[component];
      numerics::append_scaled_product_quotient(
          sums[component],
          component == 2 && bphi_weights != nullptr
              ? bphi_weights[q] : weights[q],
          components[component],
          Real{1}, Real{1});
    }
  }
  return MhdBackground{
      constant[0] ? reference.b0x
                  : numerics::finish_scaled_product_quotient_sum(sums[0]),
      constant[1] ? reference.b0y
                  : numerics::finish_scaled_product_quotient_sum(sums[1]),
      constant[2] ? reference.b0z
                  : numerics::finish_scaled_product_quotient_sum(sums[2])};
}

template <class Sample>
__device__ inline MhdBackground weighted_background_value(
    const Real* weights, int count, const Sample& sample) {
  return weighted_background_value(
      weights, /*bphi_weights=*/nullptr, count, sample);
}

__device__ inline MhdBackground point_cell_background(
    const Grid2D& g, int i, int j, int order, int node_x, int node_y,
    const double* b0x, const double* b0y, const double* b0z,
    numerics::RadialTablesView radial_tables) {
  const auto radial_line = [&](int oy) {
    const auto background = [&](int ox) {
      return load_cell_background(
          g, b0x, b0y, b0z, i + ox, j + oy, radial_tables);
    };
    if (radial_tables.active != 0 && radial_tables.contains(i)) {
      const int width = order >= 7 ? 7 : 5;
      return weighted_background_value(
          radial_tables.r2_row(i, node_x),
          radial_tables.r2_bphi_row(i, node_x), width,
          [&](int q) { return background(q - width / 2); });
    }
    if (order >= 7) {
      return weighted_background_value(
          numerics::kMp7TransversePointWeights[node_x], 7,
          [&](int q) { return background(q - 3); });
    }
    return weighted_background_value(
        numerics::kMp5TransversePointWeights[node_x], 5,
        [&](int q) { return background(q - 2); });
  };
  if (order >= 7) {
    return weighted_background_value(
        numerics::kMp7TransversePointWeights[node_y], 7,
        [&](int q) { return radial_line(q - 3); });
  }
  return weighted_background_value(
      numerics::kMp5TransversePointWeights[node_y], 5,
      [&](int q) { return radial_line(q - 2); });
}

template <bool UseRadialTables>
__device__ inline Real point_cell_background_component(
    const Grid2D& g, int i, int j, int order, int node_x, int node_y,
    const double* b0x, const double* b0y, const double* b0z,
    int component, numerics::RadialTablesView radial_tables) {
  const Real* radial_row = UseRadialTables && component == 2 &&
          radial_tables.active != 0 && radial_tables.contains(i)
      ? radial_tables.r2_bphi_row(i, node_x) : nullptr;
  return cell_tensor_point_value_with_radial_row<UseRadialTables>(
      order, node_x, node_y, i, radial_tables, radial_row,
      [&](int ox, int oy) {
        if (component == 0) {
          if constexpr (UseRadialTables) {
            return cell_bx(g, b0x, i + ox, j + oy, radial_tables);
          }
          return cell_bx(g, b0x, i + ox, j + oy);
        }
        if (component == 1) {
          if constexpr (UseRadialTables) {
            return cell_by(g, b0y, i + ox, j + oy, radial_tables);
          }
          return cell_by(g, b0y, i + ox, j + oy);
        }
        return static_cast<Real>(b0z[g.index(i + ox, j + oy)]);
      });
}

__device__ inline Real point_cell_background_component(
    const Grid2D& g, int i, int j, int order, int node_x, int node_y,
    const double* b0x, const double* b0y, const double* b0z,
    int component, numerics::RadialTablesView radial_tables) {
  return point_cell_background_component<true>(
      g, i, j, order, node_x, node_y, b0x, b0y, b0z, component,
      radial_tables);
}

// Return <a*b>-a_mean*b_mean over one cell tensor quadrature.  Constant-factor
// detection is an exact invariant, not just an optimization: a uniform CT rate
// beside a dominant varying background must have bit-zero covariance even when
// the stored Gauss weights do not sum to one in the evaluation order.
template <bool UseRadialTables, class ASample, class BSample>
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
      Real wx = order == 5 ? numerics::kMp5TransverseGaussWeights[qx]
                           : numerics::kMp7TransverseGaussWeights[qx];
      if constexpr (UseRadialTables) {
        if (radial_tables.active != 0 &&
            radial_tables.contains(radial_index)) {
          wx = radial_tables.r3_row(radial_index)[qx];
        }
      }
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
    int order, int radial_index, numerics::RadialTablesView radial_tables,
    Real a_mean, Real b_mean,
    const ASample& a, const BSample& b) {
  return cell_tensor_product_correction_scaled<true>(
      order, radial_index, radial_tables, a_mean, b_mean, a, b);
}

// Return <a*b>_annular-a_native*b_native when the factors' stored cell means
// use a different radial measure from the target energy moment.  Unlike an
// ordinary covariance, one constant factor does not make this difference
// vanish: for example, constant B0_phi times a varying flat-stored B_phi rate
// still contains B0_phi*(<rate>_annular-<rate>_flat).  Only cases that make the
// full difference provably zero bypass the scaled reduction.
template <class ASample, class BSample>
__device__ inline numerics::ScaledValue cell_tensor_product_difference_scaled(
    int order, int radial_index, numerics::RadialTablesView radial_tables,
    Real a_native_mean, Real b_native_mean,
    const ASample& a, const BSample& b) {
  const int count = order == 5 ? 3 : 4;
  const Real a_reference = a(count / 2, count / 2);
  const Real b_reference = b(count / 2, count / 2);
  bool a_constant = true;
  bool b_constant = true;
  bool a_zero = a_native_mean == Real{0};
  bool b_zero = b_native_mean == Real{0};
  numerics::ScaledQuaternaryAccumulator sum;
  for (int qy = 0; qy < count; ++qy) {
    const Real wy = order == 5
        ? numerics::kMp5TransverseGaussWeights[qy]
        : numerics::kMp7TransverseGaussWeights[qy];
    for (int qx = 0; qx < count; ++qx) {
      const Real wx = radial_tables.r3_row(radial_index)[qx];
      const Real aq = a(qx, qy);
      const Real bq = b(qx, qy);
      a_constant = a_constant && aq == a_reference;
      b_constant = b_constant && bq == b_reference;
      a_zero = a_zero && aq == Real{0};
      b_zero = b_zero && bq == Real{0};
      numerics::append_scaled_quaternary_product(sum, aq, bq, wx, wy);
    }
  }
  if (a_zero || b_zero ||
      (a_constant && b_constant && a_native_mean == a_reference &&
       b_native_mean == b_reference)) {
    return {};
  }
  numerics::append_scaled_quaternary_product(
      sum, Real{-1}, a_native_mean, b_native_mean, Real{1});
  return numerics::finish_scaled_quaternary_sum_to_value(sum);
}

template <class ASample, class BSample>
__device__ inline numerics::ScaledValue cell_tensor_product_correction_scaled(
    int order, Real a_mean, Real b_mean,
    const ASample& a, const BSample& b) {
  return cell_tensor_product_correction_scaled<false>(
      order, /*radial_index=*/0, numerics::RadialTablesView{},
      a_mean, b_mean, a, b);
}

}  // namespace quasar::mhd::detail
