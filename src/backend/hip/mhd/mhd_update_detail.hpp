#pragma once

// Private inline helpers shared by the MHD update translation units.
//
// The HIP target does not enable relocatable device code. Keep every shared
// device helper inline or templated here so kernels never call device functions
// defined in another translation unit.

#include "quasar/core/grid.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include "mhd_cell_quadrature.hpp"

#include <hip/hip_runtime.h>

namespace quasar::mhd::update_detail {

using quasar::Real;
using quasar::numerics::MhdBackground;
using quasar::numerics::MhdState;
using quasar::numerics::pressure;

inline constexpr unsigned kBlock2D = 16;
inline dim3 mhd_grid_2d(const quasar::Grid2D& g, dim3 block) {
  return dim3((static_cast<unsigned>(g.nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(g.ny) + block.y - 1) / block.y);
}

struct CylindricalAngularMetric {
  Real lo_weight{};
  Real hi_weight{};
  Real moment_norm{};
};

__device__ inline CylindricalAngularMetric cylindrical_angular_metric(
    Real spacing, Real radius) {
  Real q = Real{0.5} * (spacing / radius);
  if (q > Real{1}) q = Real{1};
  const Real lo_ratio = Real{1} - q;
  const Real hi_ratio = Real{1} + q;
  return CylindricalAngularMetric{
      lo_ratio * lo_ratio, hi_ratio * hi_ratio,
      Real{1} + q * q / Real{3}};
}

__device__ inline Real background_component(const MhdBackground& bg, int q) {
  return q == 0 ? bg.b0x : q == 1 ? bg.b0y : bg.b0z;
}

template <bool UseRadialTables>
__device__ inline MhdBackground mapped_face_background(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags, int dir,
    const double* b0x, const double* b0y, const double* b0z,
    int i, int j, int collocation_order,
    quasar::numerics::RadialTablesView radial_tables = {}) {
  if (dir == 0 && i == g.nx && flags.side[0] == 0 && flags.side[1] == 0) {
    i = 0;
  }
  if (dir == 1 && j == g.ny && flags.side[2] == 0 && flags.side[3] == 0) {
    j = 0;
  }
  if constexpr (UseRadialTables) {
    return quasar::mhd::load_interface_background(
        g, dir, b0x, b0y, b0z, i, j, radial_tables,
        collocation_order);
  }
  return quasar::mhd::load_interface_background(
      g, dir, b0x, b0y, b0z, i, j, collocation_order);
}

template <int Capacity>
__device__ inline void append_normal_background_stress(
    quasar::numerics::ScaledProductQuotientAccumulator<Capacity>& sum,
    const MhdBackground& lo, const MhdBackground& hi, int normal,
    Real spacing) {
  for (int q = 0; q < 3; ++q) {
    const Real hi_value = background_component(hi, q);
    const Real lo_value = background_component(lo, q);
    const Real hi_coefficient = q == normal ? Real{0.5} : Real{-0.5};
    quasar::numerics::append_scaled_product_quotient(
        sum, hi_value, hi_coefficient * hi_value, spacing, Real{1});
    quasar::numerics::append_scaled_product_quotient(
        sum, lo_value, -hi_coefficient * lo_value, spacing, Real{1});
  }
}

template <int Capacity>
__device__ inline void append_tangential_background_stress(
    quasar::numerics::ScaledProductQuotientAccumulator<Capacity>& sum,
    const MhdBackground& lo, const MhdBackground& hi, int normal,
    int tangent, Real spacing) {
  quasar::numerics::append_scaled_product_quotient(
      sum, background_component(hi, normal),
      background_component(hi, tangent), spacing, Real{1});
  quasar::numerics::append_scaled_product_quotient(
      sum, background_component(lo, normal),
      -background_component(lo, tangent), spacing, Real{1});
}

template <int Capacity>
__device__ inline void append_flux_part(
    quasar::numerics::ScaledProductQuotientAccumulator<Capacity>& sum,
    Real value, Real coefficient, Real d0, Real d1) {
  quasar::numerics::append_scaled_product_quotient(
      sum, value, coefficient, d0, d1);
}

struct CrossBPointView {
  const double* x[4]{};
  const double* y[4]{};
  const double* z[4]{};
  const int* quadrature_valid{};
};

struct CrossStressFace {
  MhdBackground average_background{};
  MhdBackground average_cross_b{};
  MhdBackground point_background[4]{};
  MhdBackground point_cross_b[4]{};
  Real point_weight[2][4]{};
  int point_count{};
};

__device__ inline MhdBackground load_cross_b_point(
    const CrossBPointView& view, int q, std::size_t k) {
  return MhdBackground{view.x[q][k], view.y[q][k], view.z[q][k]};
}

template <bool UseRadialTables>
__device__ inline MhdBackground mapped_point_face_background(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags, int dir,
    const double* b0x, const double* b0y, const double* b0z,
    int i, int j, int scheme_order, int node,
    quasar::numerics::RadialTablesView radial_tables = {}) {
  if (dir == 0 && i == g.nx && flags.side[0] == 0 && flags.side[1] == 0) {
    i = 0;
  }
  if (dir == 1 && j == g.ny && flags.side[2] == 0 && flags.side[3] == 0) {
    j = 0;
  }
  const int width = scheme_order >= 7 ? 7 : 5;
  bool radial_recovery = false;
  if constexpr (UseRadialTables) {
    radial_recovery =
        dir == 1 && radial_tables.active != 0 && radial_tables.contains(i);
  }
  const Real* weights = radial_recovery
      ? radial_tables.r2_row(i, node)
      : scheme_order >= 7
            ? quasar::numerics::kMp7TransversePointWeights[node]
            : quasar::numerics::kMp5TransversePointWeights[node];
  const Real* bphi_weights = radial_recovery
      ? radial_tables.r2_bphi_row(i, node) : weights;
  MhdBackground reference{};
  bool constant_x = true;
  bool constant_y = true;
  bool constant_z = true;
  quasar::numerics::ScaledProductQuotientAccumulator<8> x_sum;
  quasar::numerics::ScaledProductQuotientAccumulator<8> y_sum;
  quasar::numerics::ScaledProductQuotientAccumulator<8> z_sum;
  for (int k = 0; k < width; ++k) {
    const int offset = k - width / 2;
    const int is = dir == 0 ? i : i + offset;
    const int js = dir == 0 ? j + offset : j;
    MhdBackground value;
    if constexpr (UseRadialTables) {
      value = quasar::mhd::load_interface_background(
          g, dir, b0x, b0y, b0z, is, js,
          radial_tables, /*collocation_order=*/0);
    } else {
      value = quasar::mhd::load_interface_background(
          g, dir, b0x, b0y, b0z, is, js,
          /*collocation_order=*/0);
    }
    if (k == 0) {
      reference = value;
    } else {
      constant_x = constant_x && value.b0x == reference.b0x;
      constant_y = constant_y && value.b0y == reference.b0y;
      constant_z = constant_z && value.b0z == reference.b0z;
    }
    quasar::numerics::append_scaled_product_quotient(
        x_sum, weights[k], value.b0x, Real{1}, Real{1});
    quasar::numerics::append_scaled_product_quotient(
        y_sum, weights[k], value.b0y, Real{1}, Real{1});
    quasar::numerics::append_scaled_product_quotient(
        z_sum, bphi_weights[k], value.b0z, Real{1}, Real{1});
  }
  return MhdBackground{
      constant_x ? reference.b0x
                 : quasar::numerics::finish_scaled_product_quotient_sum(x_sum),
      constant_y ? reference.b0y
                 : quasar::numerics::finish_scaled_product_quotient_sum(y_sum),
      constant_z ? reference.b0z
                 : quasar::numerics::finish_scaled_product_quotient_sum(z_sum)};
}

template <bool UseRadialTables>
__device__ inline const Real* face_quadrature_weight_row(
    int dir, int radial_index, int point_count,
    quasar::numerics::RadialTablesView radial_tables,
    bool angular_momentum = false) {
  if constexpr (UseRadialTables) {
    if (dir == 1 && radial_tables.active != 0 &&
        radial_tables.contains(radial_index)) {
      return angular_momentum
          ? radial_tables.r3_mphi_row(radial_index)
          : radial_tables.r3_row(radial_index);
    }
  }
  return point_count == 3
      ? quasar::numerics::kMp5TransverseGaussWeights
      : quasar::numerics::kMp7TransverseGaussWeights;
}

template <bool UseRadialTables>
__device__ inline CrossStressFace load_cross_stress_face(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags, int dir,
    const double* b0x, const double* b0y, const double* b0z,
    int i, int j, int collocation_order, int scheme_order,
    const CrossBPointView& cross_b, std::size_t k,
    quasar::numerics::RadialTablesView radial_tables = {}) {
  CrossStressFace face;
  face.average_background = mapped_face_background<UseRadialTables>(
      g, flags, dir, b0x, b0y, b0z, i, j, collocation_order,
      radial_tables);
  face.average_cross_b = load_cross_b_point(cross_b, 0, k);
  if ((scheme_order != 5 && scheme_order != 7) ||
      cross_b.quadrature_valid[k] == 0) {
    return face;
  }
  face.point_count = scheme_order == 5 ? 3 : 4;
  const Real* point_weights = face_quadrature_weight_row<UseRadialTables>(
      dir, i, face.point_count, radial_tables);
  const Real* mphi_weights = face_quadrature_weight_row<UseRadialTables>(
      dir, i, face.point_count, radial_tables,
      /*angular_momentum=*/true);
  for (int q = 0; q < face.point_count; ++q) {
    face.point_background[q] = mapped_point_face_background<UseRadialTables>(
        g, flags, dir, b0x, b0y, b0z, i, j, scheme_order, q,
        radial_tables);
    face.point_cross_b[q] = load_cross_b_point(cross_b, q, k);
    face.point_weight[0][q] = point_weights[q];
    face.point_weight[1][q] = mphi_weights[q];
  }
  return face;
}

template <int Capacity>
__device__ inline void append_scaled_flux_part(
    quasar::numerics::ScaledProductQuotientAccumulator<Capacity>& sum,
    const quasar::numerics::ScaledValue& value, Real coefficient,
    Real d0, Real d1) {
  quasar::numerics::append_scaled_value_quotient(
      sum, value, coefficient, d0, d1);
}

// Append coefficient*<C_{normal,momentum}(B0,cross_b)>/(d0*d1) without
// forming a face stress.  A valid MP face contributes every transverse
// w_q*B0(q)*cross_b(q) product directly.  Lower-order and inadmissible
// recovered faces retain the conservative base-face factorization.
template <int Capacity>
__device__ inline void append_cross_momentum_stress(
    quasar::numerics::ScaledProductQuotientAccumulator<Capacity>& sum,
    const CrossStressFace& face,
    int normal, int momentum, Real coefficient, Real d0, Real d1) {
  if (face.point_count != 0) {
    for (int q = 0; q < face.point_count; ++q) {
      const Real weight = face.point_weight[momentum == 2 ? 1 : 0][q];
      if (normal == momentum) {
        for (int component = 0; component < 3; ++component) {
          const Real sign = component == normal ? Real{-1} : Real{1};
          quasar::numerics::append_scaled_triple_product_quotient(
              sum, background_component(face.point_background[q], component),
              background_component(face.point_cross_b[q], component),
              coefficient * sign * weight, d0, d1);
        }
      } else {
        quasar::numerics::append_scaled_triple_product_quotient(
            sum, background_component(face.point_background[q], normal),
            background_component(face.point_cross_b[q], momentum),
            -coefficient * weight, d0, d1);
        quasar::numerics::append_scaled_triple_product_quotient(
            sum, background_component(face.point_background[q], momentum),
            background_component(face.point_cross_b[q], normal),
            -coefficient * weight, d0, d1);
      }
    }
    return;
  }

  if (normal == momentum) {
    for (int component = 0; component < 3; ++component) {
      const Real sign = component == normal ? Real{-1} : Real{1};
      quasar::numerics::append_scaled_product_quotient(
          sum, background_component(face.average_background, component),
          (coefficient * sign) *
              background_component(face.average_cross_b, component),
          d0, d1);
    }
    return;
  }
  quasar::numerics::append_scaled_product_quotient(
      sum, background_component(face.average_background, normal),
      -coefficient * background_component(face.average_cross_b, momentum),
      d0, d1);
  quasar::numerics::append_scaled_product_quotient(
      sum, background_component(face.average_background, momentum),
      -coefficient * background_component(face.average_cross_b, normal),
      d0, d1);
}

struct BackgroundStressFace {
  MhdBackground average{};
  MhdBackground point[4]{};
  Real point_weight[2][4]{};
  int point_count{};
};

template <bool UseRadialTables>
__device__ inline BackgroundStressFace load_background_stress_face(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags, int dir,
    const double* b0x, const double* b0y, const double* b0z,
    int i, int j, int collocation_order, int scheme_order,
    quasar::numerics::RadialTablesView radial_tables = {}) {
  BackgroundStressFace face;
  face.average = mapped_face_background<UseRadialTables>(
      g, flags, dir, b0x, b0y, b0z, i, j, collocation_order,
      radial_tables);
  if (scheme_order != 5 && scheme_order != 7) return face;
  face.point_count = scheme_order == 5 ? 3 : 4;
  const Real* point_weights = face_quadrature_weight_row<UseRadialTables>(
      dir, i, face.point_count, radial_tables);
  const Real* mphi_weights = face_quadrature_weight_row<UseRadialTables>(
      dir, i, face.point_count, radial_tables,
      /*angular_momentum=*/true);
  for (int q = 0; q < face.point_count; ++q) {
    face.point[q] = mapped_point_face_background<UseRadialTables>(
        g, flags, dir, b0x, b0y, b0z, i, j, scheme_order, q,
        radial_tables);
    face.point_weight[0][q] = point_weights[q];
    face.point_weight[1][q] = mphi_weights[q];
  }
  return face;
}

// Reuse the background half of a cross-stress face.  When cross quadrature is
// valid this is a pure copy; when it is invalid, CrossStressFace deliberately
// keeps point_count=0 and this helper alone recovers the static-background
// points needed by the separately valid background stress.
template <bool UseRadialTables>
__device__ inline BackgroundStressFace background_stress_from_cross_face(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags, int dir,
    const double* b0x, const double* b0y, const double* b0z,
    int i, int j, int scheme_order, const CrossStressFace& cross,
    quasar::numerics::RadialTablesView radial_tables = {}) {
  BackgroundStressFace face;
  face.average = cross.average_background;
  if (scheme_order != 5 && scheme_order != 7) return face;
  face.point_count = scheme_order == 5 ? 3 : 4;
  const Real* point_weights = face_quadrature_weight_row<UseRadialTables>(
      dir, i, face.point_count, radial_tables);
  const Real* mphi_weights = face_quadrature_weight_row<UseRadialTables>(
      dir, i, face.point_count, radial_tables,
      /*angular_momentum=*/true);
  for (int q = 0; q < face.point_count; ++q) {
    face.point[q] = cross.point_count == face.point_count
        ? cross.point_background[q]
        : mapped_point_face_background<UseRadialTables>(
              g, flags, dir, b0x, b0y, b0z, i, j, scheme_order, q,
              radial_tables);
    face.point_weight[0][q] = point_weights[q];
    face.point_weight[1][q] = mphi_weights[q];
  }
  return face;
}

// Append one static Maxwell-stress face contribution.  MP5/MP7 integrate the
// nonlinear product at transverse Gauss points, but retain both B0 factors
// until the cell reduction.  Shared face factors make the periodic static
// force conservative; a cell-volume J0 x B0 source is formally accurate but
// does not telescope across cells.
template <int Capacity>
__device__ inline void append_background_momentum_stress(
    quasar::numerics::ScaledProductQuotientAccumulator<Capacity>& sum,
    const BackgroundStressFace& face, int normal, int momentum,
    Real coefficient, Real d0, Real d1) {
  if (face.point_count != 0) {
    for (int q = 0; q < face.point_count; ++q) {
      const Real weight = face.point_weight[momentum == 2 ? 1 : 0][q];
      if (normal == momentum) {
        for (int component = 0; component < 3; ++component) {
          const Real sign = component == normal ? Real{-0.5} : Real{0.5};
          const Real value = background_component(face.point[q], component);
          quasar::numerics::append_scaled_triple_product_quotient(
              sum, value, value, coefficient * sign * weight, d0, d1);
        }
      } else {
        quasar::numerics::append_scaled_triple_product_quotient(
            sum, background_component(face.point[q], normal),
            background_component(face.point[q], momentum),
            -coefficient * weight, d0, d1);
      }
    }
    return;
  }
  if (normal == momentum) {
    for (int component = 0; component < 3; ++component) {
      const Real sign = component == normal ? Real{-0.5} : Real{0.5};
      const Real value = background_component(face.average, component);
      quasar::numerics::append_scaled_product_quotient(
          sum, value, coefficient * sign * value, d0, d1);
    }
    return;
  }
  quasar::numerics::append_scaled_product_quotient(
      sum, background_component(face.average, normal),
      -coefficient * background_component(face.average, momentum), d0, d1);
}

template <int Capacity>
__device__ inline void append_cylindrical_background_axial_residual(
    quasar::numerics::ScaledProductQuotientAccumulator<Capacity>& sum,
    const BackgroundStressFace& radial_lo,
    const BackgroundStressFace& radial_hi,
    const BackgroundStressFace& axial_lo,
    const BackgroundStressFace& axial_hi,
    int scheme_order, Real dr, Real dz, Real radius) {
  if (scheme_order == 5 || scheme_order == 7) {
    append_background_momentum_stress(
        sum, radial_hi, 0, 1, Real{-1}, dr, Real{1});
    append_background_momentum_stress(
        sum, radial_lo, 0, 1, Real{1}, dr, Real{1});
    append_background_momentum_stress(
        sum, radial_hi, 0, 1, Real{-1}, radius, Real{2});
    append_background_momentum_stress(
        sum, radial_lo, 0, 1, Real{-1}, radius, Real{2});
    append_background_momentum_stress(
        sum, axial_hi, 1, 1, Real{-1}, dz, Real{1});
    append_background_momentum_stress(
        sum, axial_lo, 1, 1, Real{1}, dz, Real{1});
    return;
  }
  const MhdBackground& radial_low = radial_lo.average;
  const MhdBackground& radial_high = radial_hi.average;
  quasar::numerics::append_scaled_product_quotient(
      sum, radial_high.b0x, radial_high.b0y, dr, Real{1});
  quasar::numerics::append_scaled_product_quotient(
      sum, radial_low.b0x, -radial_low.b0y, dr, Real{1});
  quasar::numerics::append_scaled_product_quotient(
      sum, radial_high.b0x, radial_high.b0y, radius, Real{2});
  quasar::numerics::append_scaled_product_quotient(
      sum, radial_low.b0x, radial_low.b0y, radius, Real{2});
  append_normal_background_stress(
      sum, axial_lo.average, axial_hi.average, 1, dz);
}

template <int Capacity>
__device__ inline void append_cylindrical_background_azimuthal_residual(
    quasar::numerics::ScaledProductQuotientAccumulator<Capacity>& sum,
    const BackgroundStressFace& radial_lo,
    const BackgroundStressFace& radial_hi,
    const BackgroundStressFace& axial_lo,
    const BackgroundStressFace& axial_hi,
    int scheme_order, Real dr, Real dz,
    const CylindricalAngularMetric& angular) {
  if (scheme_order == 5 || scheme_order == 7) {
    append_background_momentum_stress(
        sum, radial_hi, 0, 2, -angular.hi_weight,
        dr, angular.moment_norm);
    if (angular.lo_weight != Real{0}) {
      append_background_momentum_stress(
          sum, radial_lo, 0, 2, angular.lo_weight,
          dr, angular.moment_norm);
    }
    append_background_momentum_stress(
        sum, axial_hi, 1, 2, Real{-1}, dz, Real{1});
    append_background_momentum_stress(
        sum, axial_lo, 1, 2, Real{1}, dz, Real{1});
    return;
  }
  quasar::numerics::append_scaled_product_quotient(
      sum, radial_hi.average.b0x, radial_hi.average.b0z, dr,
      angular.moment_norm / angular.hi_weight);
  if (angular.lo_weight != Real{0}) {
    quasar::numerics::append_scaled_product_quotient(
        sum, radial_lo.average.b0x, -radial_lo.average.b0z, dr,
        angular.moment_norm / angular.lo_weight);
  }
  append_tangential_background_stress(
      sum, axial_lo.average, axial_hi.average, 1, 2, dz);
}

// Return <value^2>_dr,dz - mean^2 over the tensor Gauss rule. The point
// recovery still uses the active radial R2 rows, but the radial integration is
// deliberately uniform: the cylindrical curvature source carries a 1/r that
// cancels the ring measure. A constant field returns bit-zero so adding the
// correction preserves the existing free-stream arithmetic exactly.
template <class Sample>
__device__ inline quasar::numerics::ScaledValue
cell_uniform_square_correction_scaled(
    int order, Real mean, const Sample& sample) {
  const int count = order == 5 ? 3 : 4;
  const Real reference = sample(count / 2, count / 2);
  bool constant = reference == mean;
  quasar::numerics::ScaledQuaternaryAccumulator sum;
  for (int qy = 0; qy < count; ++qy) {
    const Real wy = order == 5
        ? quasar::numerics::kMp5TransverseGaussWeights[qy]
        : quasar::numerics::kMp7TransverseGaussWeights[qy];
    for (int qx = 0; qx < count; ++qx) {
      const Real wx = order == 5
          ? quasar::numerics::kMp5TransverseGaussWeights[qx]
          : quasar::numerics::kMp7TransverseGaussWeights[qx];
      const Real value = sample(qx, qy);
      constant = constant && value == reference;
      quasar::numerics::append_scaled_quaternary_product(
          sum, value, value, wx, wy);
    }
  }
  if (constant) return {};
  quasar::numerics::append_scaled_quaternary_product(
      sum, Real{-1}, mean, mean, Real{1});
  return quasar::numerics::finish_scaled_quaternary_sum_to_value(sum);
}

// Append <M0_phiphi>_dr,dz/r_center, where
// M0_phiphi = 0.5*(B0_r^2+B0_z^2-B0_phi^2).  The 1/r source cancels the ring
// measure, so both tensor integration weights are Cartesian even though point
// recovery along r still uses R2.  Mean-square terms plus conditioned
// corrections preserve the constant-field path exactly.
template <int Capacity>
__device__ inline void append_background_azimuthal_stress(
    quasar::numerics::ScaledProductQuotientAccumulator<Capacity>& sum,
    quasar::Grid2D g, int i, int j,
    const double* b0x, const double* b0y, const double* b0z,
    const MhdBackground& mean, int scheme_order,
    quasar::numerics::RadialTablesView radial_tables, Real radius) {
  const Real coefficient[3] = {Real{0.5}, Real{0.5}, Real{-0.5}};
  for (int component = 0; component < 3; ++component) {
    const Real mean_value = background_component(mean, component);
    quasar::numerics::append_scaled_product_quotient(
        sum, mean_value, coefficient[component] * mean_value,
        radius, Real{1});
    if (scheme_order == 5 || scheme_order == 7) {
      const auto point_value = [&](int qx, int qy) {
        return quasar::mhd::detail::point_cell_background_component(
            g, i, j, scheme_order, qx, qy, b0x, b0y, b0z,
            component, radial_tables);
      };
      const quasar::numerics::ScaledValue correction =
          cell_uniform_square_correction_scaled(
              scheme_order, mean_value, point_value);
      quasar::numerics::append_scaled_value_quotient(
          sum, correction, coefficient[component], radius, Real{1});
    }
  }
}

}  // namespace quasar::mhd::update_detail
