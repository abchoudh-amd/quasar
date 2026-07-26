// Device update kernels for the ideal-MHD solver: conservative flux difference,
// SSP-RK stage combine, explicit standalone positivity repair, and the
// cylindrical geometric source.

#include "quasar/core/grid.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include "backend/hip/hip_check.hpp"
#include "mhd_cell_quadrature.hpp"

#include <hip/hip_runtime.h>

namespace {

using quasar::Real;
using quasar::numerics::MhdBackground;
using quasar::numerics::MhdState;
using quasar::numerics::pressure;

inline constexpr unsigned kBlock2D = 16;
inline dim3 mhd_grid_2d(const quasar::Grid2D& g, dim3 block) {
  return dim3((static_cast<unsigned>(g.nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(g.ny) + block.y - 1) / block.y);
}

// dudt += -(F_{i+1/2} - F_{i-1/2}) / d(dir). The flux at face (i,j) is the
// x_lo/y_lo face of cell (i,j); the i+1/2 face of cell (i,j) is therefore the
// stored flux of cell (i+di, j+dj). Accumulates so the caller can sum both
// directions into a single dudt (zeroed before the first direction).
__global__ void flux_difference_kernel(
    quasar::Grid2D g, int dir, int cylindrical,
    const double* __restrict__ Frho, const double* __restrict__ Fmx,
    const double* __restrict__ Fmy, const double* __restrict__ Fmz,
    const double* __restrict__ Fenergy, const double* __restrict__ Fbx,
    const double* __restrict__ Fby, const double* __restrict__ Fbz,
    double* __restrict__ drho, double* __restrict__ dmx, double* __restrict__ dmy,
    double* __restrict__ dmz, double* __restrict__ denergy, double* __restrict__ dbx,
    double* __restrict__ dby, double* __restrict__ dbz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const int di = (dir == 0) ? 1 : 0;
  const int dj = (dir == 0) ? 0 : 1;
  const std::size_t k  = g.index(i, j);            // F at i-1/2 face
  const std::size_t kp = g.index(i + di, j + dj);  // F at i+1/2 face
  const Real spacing_value = (dir == 0) ? g.dx() : g.dy();
  const auto difference_quotient = [&](Real flo, Real fhi) {
    const Real a[2] = {fhi, -flo};
    const Real one[2] = {Real{1}, Real{1}};
    const Real spacing[2] = {spacing_value, spacing_value};
    return quasar::numerics::scaled_product_quotient_sum(
        a, one, spacing, one, 2);
  };
  if (cylindrical != 0 && dir == 0) {
    // Ring-volume finite-volume divergence. On a uniform radial mesh,
    //   [r_hi F_hi-r_lo F_lo]/int(r dr)
    // = (F_hi-F_lo)/dr + (F_hi/2+F_lo/2)/r_c.
    // The factored form avoids differences of squared radii and overflowing
    // coordinate*flux products, and is regular in the r_lo=0 axis cell.
    const Real dr = g.dx();
    const Real r_center = g.r_at_cell_center(i);
    const auto annular_div = [&](Real flo, Real fhi) {
      const Real a[4] = {fhi, -flo, fhi, flo};
      const Real b[4] = {Real{1}, Real{1}, Real{1}, Real{1}};
      const Real d0[4] = {dr, dr, r_center, r_center};
      const Real d1[4] = {Real{1}, Real{1}, Real{2}, Real{2}};
      return quasar::numerics::scaled_product_quotient_sum(a, b, d0, d1, 4);
    };
    drho[k]    += -annular_div(Frho[k], Frho[kp]);
    // Cylindrical radial momentum is deliberately omitted here. Its ordinary
    // tensor derivatives, pressure-free curvature difference, and optional
    // static-background stress are rounded together by the dedicated kernel.
    dmy[k]     += -annular_div(Fmy[k], Fmy[kp]);
    // m_phi is stored as a piecewise-constant annular cell value, but its
    // conserved integral is angular momentum int(r^2 m_phi dr). Use that exact
    // cell moment rather than an annular divergence plus a pointwise -F/r
    // source. With q=dr/(2*r_c),
    //
    //   D_J = [(1+q)^2 F_hi - (1-q)^2 F_lo]
    //         / [dr*(1+q^2/3)].
    //
    // This normalized form never constructs r^2 or W=int(r^2 dr), remains
    // regular at the axis (q=1), and telescopes exactly when weighted by W.
    Real q = Real{0.5} * (dr / r_center);
    if (q > Real{1}) q = Real{1};
    const Real lo_ratio = Real{1} - q;
    const Real hi_ratio = Real{1} + q;
    const Real moment_norm = Real{1} + q * q / Real{3};
    const Real angular_a[2] = {Fmz[kp], -Fmz[k]};
    const Real angular_b[2] = {
        hi_ratio * hi_ratio, lo_ratio * lo_ratio};
    const Real angular_d0[2] = {dr, dr};
    const Real angular_d1[2] = {moment_norm, moment_norm};
    dmz[k] += -quasar::numerics::scaled_product_quotient_sum(
        angular_a, angular_b, angular_d0, angular_d1, 2);
    denergy[k] += -annular_div(Fenergy[k], Fenergy[kp]);

    // B_phi obeys the metric-free point equation
    // d_t B_phi + d_r F_{Bphi}=0, so use the ordinary radial face difference.
    const Real a_bphi[2] = {-Fbz[kp], Fbz[k]};
    const Real one[2] = {Real{1}, Real{1}};
    const Real spacing[2] = {dr, dr};
    dbz[k] += quasar::numerics::scaled_product_quotient_sum(
        a_bphi, one, spacing, one, 2);
  } else {
    drho[k]    += -difference_quotient(Frho[k], Frho[kp]);
    if (cylindrical == 0) {
      dmx[k] += -difference_quotient(Fmx[k], Fmx[kp]);
    }
    dmy[k]     += -difference_quotient(Fmy[k], Fmy[kp]);
    dmz[k]     += -difference_quotient(Fmz[k], Fmz[kp]);
    denergy[k] += -difference_quotient(Fenergy[k], Fenergy[kp]);
    dbz[k]     += -difference_quotient(Fbz[k], Fbz[kp]);
  }
  dbx[k] += -difference_quotient(Fbx[k], Fbx[kp]);
  dby[k] += -difference_quotient(Fby[k], Fby[kp]);
}

__device__ inline Real background_component(const MhdBackground& bg, int q) {
  return q == 0 ? bg.b0x : q == 1 ? bg.b0y : bg.b0z;
}

__device__ inline MhdBackground mapped_face_background(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags, int dir,
    const double* b0x, const double* b0y, const double* b0z,
    int i, int j, int collocation_order) {
  if (dir == 0 && i == g.nx && flags.side[0] == 0 && flags.side[1] == 0) {
    i = 0;
  }
  if (dir == 1 && j == g.ny && flags.side[2] == 0 && flags.side[3] == 0) {
    j = 0;
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
  int point_count{};
};

__device__ inline MhdBackground load_cross_b_point(
    const CrossBPointView& view, int q, std::size_t k) {
  return MhdBackground{view.x[q][k], view.y[q][k], view.z[q][k]};
}

__device__ inline MhdBackground mapped_point_face_background(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags, int dir,
    const double* b0x, const double* b0y, const double* b0z,
    int i, int j, int scheme_order, int node) {
  if (dir == 0 && i == g.nx && flags.side[0] == 0 && flags.side[1] == 0) {
    i = 0;
  }
  if (dir == 1 && j == g.ny && flags.side[2] == 0 && flags.side[3] == 0) {
    j = 0;
  }
  const auto component = [&](int component_index) {
    return quasar::mhd::detail::cell_quadrature_point_value(
        scheme_order, node, [&](int offset) {
          const int is = dir == 0 ? i : i + offset;
          const int js = dir == 0 ? j + offset : j;
          const MhdBackground value = quasar::mhd::load_interface_background(
              g, dir, b0x, b0y, b0z, is, js,
              /*collocation_order=*/0);
          return background_component(value, component_index);
        });
  };
  return MhdBackground{component(0), component(1), component(2)};
}

__device__ inline CrossStressFace load_cross_stress_face(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags, int dir,
    const double* b0x, const double* b0y, const double* b0z,
    int i, int j, int collocation_order, int scheme_order,
    const CrossBPointView& cross_b, std::size_t k) {
  CrossStressFace face;
  face.average_background = mapped_face_background(
      g, flags, dir, b0x, b0y, b0z, i, j, collocation_order);
  face.average_cross_b = load_cross_b_point(cross_b, 0, k);
  if ((scheme_order != 5 && scheme_order != 7) ||
      cross_b.quadrature_valid[k] == 0) {
    return face;
  }
  face.point_count = scheme_order == 5 ? 3 : 4;
  for (int q = 0; q < face.point_count; ++q) {
    face.point_background[q] = mapped_point_face_background(
        g, flags, dir, b0x, b0y, b0z, i, j, scheme_order, q);
    face.point_cross_b[q] = load_cross_b_point(cross_b, q, k);
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
      const Real weight = face.point_count == 3
          ? quasar::numerics::kMp5TransverseGaussWeights[q]
          : quasar::numerics::kMp7TransverseGaussWeights[q];
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
  int point_count{};
};

__device__ inline BackgroundStressFace load_background_stress_face(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags, int dir,
    const double* b0x, const double* b0y, const double* b0z,
    int i, int j, int collocation_order, int scheme_order) {
  BackgroundStressFace face;
  face.average = mapped_face_background(
      g, flags, dir, b0x, b0y, b0z, i, j, collocation_order);
  if (scheme_order != 5 && scheme_order != 7) return face;
  face.point_count = scheme_order == 5 ? 3 : 4;
  for (int q = 0; q < face.point_count; ++q) {
    face.point[q] = mapped_point_face_background(
        g, flags, dir, b0x, b0y, b0z, i, j, scheme_order, q);
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
      const Real weight = face.point_count == 3
          ? quasar::numerics::kMp5TransverseGaussWeights[q]
          : quasar::numerics::kMp7TransverseGaussWeights[q];
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

// Add -div(F_m(B0)) once per cell.  Both directions, and in cylindrical
// geometry the radial curvature source, enter the same common-exponent sum for
// each momentum component.  This prevents a curl-free dominant B0 from leaving
// an O(epsilon*B0^2/h) force merely because x/r and y/z were rounded first.
__global__ void background_stress_correction_kernel(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags, int cylindrical,
    int collocation_order, int scheme_order,
    const double* __restrict__ b0x,
    const double* __restrict__ b0y, const double* __restrict__ b0z,
    double* __restrict__ dmx, double* __restrict__ dmy,
    double* __restrict__ dmz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);
  const MhdBackground xlo = mapped_face_background(
      g, flags, 0, b0x, b0y, b0z, i, j, collocation_order);
  const MhdBackground xhi = mapped_face_background(
      g, flags, 0, b0x, b0y, b0z, i + 1, j, collocation_order);
  const MhdBackground ylo = mapped_face_background(
      g, flags, 1, b0x, b0y, b0z, i, j, collocation_order);
  const MhdBackground yhi = mapped_face_background(
      g, flags, 1, b0x, b0y, b0z, i, j + 1, collocation_order);

  if (cylindrical == 0) {
    const BackgroundStressFace xface_lo = load_background_stress_face(
        g, flags, 0, b0x, b0y, b0z, i, j,
        collocation_order, scheme_order);
    const BackgroundStressFace xface_hi = load_background_stress_face(
        g, flags, 0, b0x, b0y, b0z, i + 1, j,
        collocation_order, scheme_order);
    const BackgroundStressFace yface_lo = load_background_stress_face(
        g, flags, 1, b0x, b0y, b0z, i, j,
        collocation_order, scheme_order);
    const BackgroundStressFace yface_hi = load_background_stress_face(
        g, flags, 1, b0x, b0y, b0z, i, j + 1,
        collocation_order, scheme_order);
    const auto component = [&](int momentum) {
      quasar::numerics::ScaledProductQuotientAccumulator<32> sum;
      append_background_momentum_stress(
          sum, xface_hi, 0, momentum, Real{-1}, g.dx(), Real{1});
      append_background_momentum_stress(
          sum, xface_lo, 0, momentum, Real{1}, g.dx(), Real{1});
      append_background_momentum_stress(
          sum, yface_hi, 1, momentum, Real{-1}, g.dy(), Real{1});
      append_background_momentum_stress(
          sum, yface_lo, 1, momentum, Real{1}, g.dy(), Real{1});
      return quasar::numerics::finish_scaled_product_quotient_sum(sum);
    };
    dmx[k] += component(0);
    dmy[k] += component(1);
    dmz[k] += component(2);
    return;
  }

  // Radial momentum is completed by the fused dynamic/background kernel. Keep
  // this launch responsible only for the axial and azimuthal static stresses.
  const Real r = g.r_at_cell_center(i);
  quasar::numerics::ScaledProductQuotientAccumulator<10> axial_sum;
  quasar::numerics::append_scaled_product_quotient(
      axial_sum, xhi.b0x, xhi.b0y, g.dx(), Real{1});
  quasar::numerics::append_scaled_product_quotient(
      axial_sum, xlo.b0x, -xlo.b0y, g.dx(), Real{1});
  quasar::numerics::append_scaled_product_quotient(
      axial_sum, xhi.b0x, xhi.b0y, r, Real{2});
  quasar::numerics::append_scaled_product_quotient(
      axial_sum, xlo.b0x, xlo.b0y, r, Real{2});
  append_normal_background_stress(axial_sum, ylo, yhi, 1, g.dy());
  dmy[k] += quasar::numerics::finish_scaled_product_quotient_sum(axial_sum);

  Real qmetric = Real{0.5} * (g.dx() / r);
  if (qmetric > Real{1}) qmetric = Real{1};
  const Real lo_ratio = Real{1} - qmetric;
  const Real hi_ratio = Real{1} + qmetric;
  const Real moment_norm = Real{1} + qmetric * qmetric / Real{3};
  quasar::numerics::ScaledProductQuotientAccumulator<4> phi_sum;
  quasar::numerics::append_scaled_product_quotient(
      phi_sum, xhi.b0x, xhi.b0z, g.dx(),
      moment_norm / (hi_ratio * hi_ratio));
  if (lo_ratio != Real{0}) {
    quasar::numerics::append_scaled_product_quotient(
        phi_sum, xlo.b0x, -xlo.b0z, g.dx(),
        moment_norm / (lo_ratio * lo_ratio));
  } else {
    // Preserve the four-term accumulator shape at the axis without evaluating
    // moment_norm/0; this face has exactly zero angular-momentum weight.
    quasar::numerics::append_scaled_product_quotient(
        phi_sum, Real{0}, Real{0}, Real{1}, Real{1});
  }
  append_tangential_background_stress(
      phi_sum, ylo, yhi, 1, 2, g.dy());
  dmz[k] += quasar::numerics::finish_scaled_product_quotient_sum(phi_sum);
}

// Complete active-background momentum residual.  Every face contribution from
// both directions is kept as material stress H, factored cross stress C, and
// Riemann correction W until this single reduction.  Static T0 terms share the
// same accumulator, so neither a directional update nor a later correction can
// erase a small survivor beside a dominant cancelling stress.
__global__ void split_momentum_residual_kernel(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags, int cylindrical,
    int collocation_order, int scheme_order, int globally_curl_free,
    const double* __restrict__ b0x, const double* __restrict__ b0y,
    const double* __restrict__ b0z,
    const double* __restrict__ x_hx, const double* __restrict__ x_hy,
    const double* __restrict__ x_hz,
    const double* __restrict__ x_wx, const double* __restrict__ x_wy,
    const double* __restrict__ x_wz,
    CrossBPointView x_cross,
    const double* __restrict__ y_hx, const double* __restrict__ y_hy,
    const double* __restrict__ y_hz,
    const double* __restrict__ y_wx, const double* __restrict__ y_wy,
    const double* __restrict__ y_wz,
    CrossBPointView y_cross,
    double* __restrict__ dmx, double* __restrict__ dmy,
    double* __restrict__ dmz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);
  const std::size_t kx = g.index(i + 1, j);
  const std::size_t ky = g.index(i, j + 1);

  const CrossStressFace xface_lo = load_cross_stress_face(
      g, flags, 0, b0x, b0y, b0z, i, j, collocation_order,
      scheme_order, x_cross, k);
  const CrossStressFace xface_hi = load_cross_stress_face(
      g, flags, 0, b0x, b0y, b0z, i + 1, j, collocation_order,
      scheme_order, x_cross, kx);
  const CrossStressFace yface_lo = load_cross_stress_face(
      g, flags, 1, b0x, b0y, b0z, i, j, collocation_order,
      scheme_order, y_cross, k);
  const CrossStressFace yface_hi = load_cross_stress_face(
      g, flags, 1, b0x, b0y, b0z, i, j + 1, collocation_order,
      scheme_order, y_cross, ky);
  const MhdBackground& xlo = xface_lo.average_background;
  const MhdBackground& xhi = xface_hi.average_background;
  const MhdBackground& ylo = yface_lo.average_background;
  const MhdBackground& yhi = yface_hi.average_background;

  if (cylindrical == 0) {
    const BackgroundStressFace xbackground_lo =
        load_background_stress_face(
            g, flags, 0, b0x, b0y, b0z, i, j,
            collocation_order, scheme_order);
    const BackgroundStressFace xbackground_hi =
        load_background_stress_face(
            g, flags, 0, b0x, b0y, b0z, i + 1, j,
            collocation_order, scheme_order);
    const BackgroundStressFace ybackground_lo =
        load_background_stress_face(
            g, flags, 1, b0x, b0y, b0z, i, j,
            collocation_order, scheme_order);
    const BackgroundStressFace ybackground_hi =
        load_background_stress_face(
            g, flags, 1, b0x, b0y, b0z, i, j + 1,
            collocation_order, scheme_order);
    const auto cartesian_component = [&](int momentum,
                                         const double* x_h,
                                         const double* x_w,
                                         const double* y_h,
                                         const double* y_w) {
      quasar::numerics::ScaledProductQuotientAccumulator<128> sum;
      append_flux_part(sum, x_h[kx], Real{-1}, g.dx(), Real{1});
      append_flux_part(sum, x_w[kx], Real{-1}, g.dx(), Real{1});
      append_flux_part(sum, x_h[k], Real{1}, g.dx(), Real{1});
      append_flux_part(sum, x_w[k], Real{1}, g.dx(), Real{1});
      append_cross_momentum_stress(
          sum, xface_hi, 0, momentum, Real{-1},
          g.dx(), Real{1});
      append_cross_momentum_stress(
          sum, xface_lo, 0, momentum, Real{1},
          g.dx(), Real{1});

      append_flux_part(sum, y_h[ky], Real{-1}, g.dy(), Real{1});
      append_flux_part(sum, y_w[ky], Real{-1}, g.dy(), Real{1});
      append_flux_part(sum, y_h[k], Real{1}, g.dy(), Real{1});
      append_flux_part(sum, y_w[k], Real{1}, g.dy(), Real{1});
      append_cross_momentum_stress(
          sum, yface_hi, 1, momentum, Real{-1},
          g.dy(), Real{1});
      append_cross_momentum_stress(
          sum, yface_lo, 1, momentum, Real{1},
          g.dy(), Real{1});

      if (globally_curl_free == 0) {
        append_background_momentum_stress(
            sum, xbackground_hi, 0, momentum, Real{-1},
            g.dx(), Real{1});
        append_background_momentum_stress(
            sum, xbackground_lo, 0, momentum, Real{1},
            g.dx(), Real{1});
        append_background_momentum_stress(
            sum, ybackground_hi, 1, momentum, Real{-1},
            g.dy(), Real{1});
        append_background_momentum_stress(
            sum, ybackground_lo, 1, momentum, Real{1},
            g.dy(), Real{1});
      }
      return quasar::numerics::finish_scaled_product_quotient_sum(sum);
    };
    dmx[k] = cartesian_component(0, x_hx, x_wx, y_hx, y_wx);
    dmy[k] = cartesian_component(1, x_hy, x_wy, y_hy, y_wy);
    dmz[k] = cartesian_component(2, x_hz, x_wz, y_hz, y_wz);
    return;
  }

  // Cylindrical axial momentum: annular radial divergence plus ordinary axial
  // divergence, with the matching static stresses in the same reduction.
  const Real r = g.r_at_cell_center(i);
  quasar::numerics::ScaledProductQuotientAccumulator<96> axial_sum;
  for (int channel = 0; channel < 2; ++channel) {
    const double* radial = channel == 0 ? x_hy : x_wy;
    append_flux_part(axial_sum, radial[kx], Real{-1}, g.dx(), Real{1});
    append_flux_part(axial_sum, radial[k], Real{1}, g.dx(), Real{1});
    append_flux_part(axial_sum, radial[kx], Real{-1}, r, Real{2});
    append_flux_part(axial_sum, radial[k], Real{-1}, r, Real{2});
    const double* axial = channel == 0 ? y_hy : y_wy;
    append_flux_part(axial_sum, axial[ky], Real{-1}, g.dy(), Real{1});
    append_flux_part(axial_sum, axial[k], Real{1}, g.dy(), Real{1});
  }
  append_cross_momentum_stress(
      axial_sum, xface_hi, 0, 1, Real{-1},
      g.dx(), Real{1});
  append_cross_momentum_stress(
      axial_sum, xface_lo, 0, 1, Real{1},
      g.dx(), Real{1});
  append_cross_momentum_stress(
      axial_sum, xface_hi, 0, 1, Real{-1},
      r, Real{2});
  append_cross_momentum_stress(
      axial_sum, xface_lo, 0, 1, Real{-1},
      r, Real{2});
  append_cross_momentum_stress(
      axial_sum, yface_hi, 1, 1, Real{-1},
      g.dy(), Real{1});
  append_cross_momentum_stress(
      axial_sum, yface_lo, 1, 1, Real{1},
      g.dy(), Real{1});
  if (globally_curl_free == 0) {
    quasar::numerics::append_scaled_product_quotient(
        axial_sum, xhi.b0x, xhi.b0y, g.dx(), Real{1});
    quasar::numerics::append_scaled_product_quotient(
        axial_sum, xlo.b0x, -xlo.b0y, g.dx(), Real{1});
    quasar::numerics::append_scaled_product_quotient(
        axial_sum, xhi.b0x, xhi.b0y, r, Real{2});
    quasar::numerics::append_scaled_product_quotient(
        axial_sum, xlo.b0x, xlo.b0y, r, Real{2});
    append_normal_background_stress(axial_sum, ylo, yhi, 1, g.dy());
  }
  dmy[k] = quasar::numerics::finish_scaled_product_quotient_sum(axial_sum);

  // Cylindrical azimuthal momentum uses the exact angular-momentum cell moment
  // in r and an ordinary axial face difference.
  Real qmetric = Real{0.5} * (g.dx() / r);
  if (qmetric > Real{1}) qmetric = Real{1};
  const Real lo_ratio = Real{1} - qmetric;
  const Real hi_ratio = Real{1} + qmetric;
  const Real moment_norm = Real{1} + qmetric * qmetric / Real{3};
  const Real hi_weight = hi_ratio * hi_ratio;
  const Real lo_weight = lo_ratio * lo_ratio;
  quasar::numerics::ScaledProductQuotientAccumulator<64> phi_sum;
  for (int channel = 0; channel < 2; ++channel) {
    const double* radial = channel == 0 ? x_hz : x_wz;
    append_flux_part(
        phi_sum, radial[kx], -hi_weight, g.dx(), moment_norm);
    append_flux_part(
        phi_sum, radial[k], lo_weight, g.dx(), moment_norm);
    const double* axial = channel == 0 ? y_hz : y_wz;
    append_flux_part(phi_sum, axial[ky], Real{-1}, g.dy(), Real{1});
    append_flux_part(phi_sum, axial[k], Real{1}, g.dy(), Real{1});
  }
  append_cross_momentum_stress(
      phi_sum, xface_hi, 0, 2, -hi_weight,
      g.dx(), moment_norm);
  if (lo_weight != Real{0}) {
    append_cross_momentum_stress(
        phi_sum, xface_lo, 0, 2, lo_weight,
        g.dx(), moment_norm);
  }
  append_cross_momentum_stress(
      phi_sum, yface_hi, 1, 2, Real{-1},
      g.dy(), Real{1});
  append_cross_momentum_stress(
      phi_sum, yface_lo, 1, 2, Real{1},
      g.dy(), Real{1});
  if (globally_curl_free == 0) {
    quasar::numerics::append_scaled_product_quotient(
        phi_sum, xhi.b0x, xhi.b0z, g.dx(), moment_norm / hi_weight);
    if (lo_weight != Real{0}) {
      quasar::numerics::append_scaled_product_quotient(
          phi_sum, xlo.b0x, -xlo.b0z, g.dx(), moment_norm / lo_weight);
    }
    append_tangential_background_stress(
        phi_sum, ylo, yhi, 1, 2, g.dy());
  }
  dmz[k] = quasar::numerics::finish_scaled_product_quotient_sum(phi_sum);
}

// Complete cylindrical radial-momentum residual in the tensor-difference form
//
//   R_mr = -d_r F_rr - d_z F_zr + (T_phiphi-T_rr)/r.
//
// Unlike an annular aggregate F_rr plus a separately expanded T_phiphi/r,
// this form cancels gas pressure and axial-field terms analytically before
// rounding. That matters when the already-rounded face flux cannot retain a
// small pressure beside a dominant toroidal stress. The split curvature is
//
//   rho(vphi^2-vr^2) + 2 B0r br + br^2
//                         - 2 B0phi bphi - bphi^2,
//
// and the static curvature is B0r^2-B0phi^2. The maximum active path is
// 2 radial flux + 2 axial flux + 6 reduced curvature + 6 radial static stress
// + 2 axial static stress + 2 static curvature = 20 terms.
__global__ void cylindrical_radial_momentum_residual_kernel(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags,
    int background_active, int globally_curl_free, int decomposed_flux,
    int collocation_order, int scheme_order,
    const double* __restrict__ rho, const double* __restrict__ mr,
    const double* __restrict__ mphi, const double* __restrict__ br,
    const double* __restrict__ bphi,
    const double* __restrict__ b0r, const double* __restrict__ b0z,
    const double* __restrict__ b0phi,
    const double* __restrict__ radial_momentum_flux,
    const double* __restrict__ axial_momentum_flux,
    const double* __restrict__ radial_wave,
    const double* __restrict__ axial_wave,
    CrossBPointView radial_cross,
    CrossBPointView axial_cross,
    double* __restrict__ radial_momentum_rate) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;

  const std::size_t k = g.index(i, j);
  const std::size_t kr = g.index(i + 1, j);
  const std::size_t kz = g.index(i, j + 1);
  const Real r = g.r_at_cell_center(i);

  quasar::numerics::ScaledProductQuotientAccumulator<64> sum;

  // Ordinary radial tensor derivative. Curvature is supplied explicitly by
  // T_phiphi-T_rr below; using the annular derivative here would double-count
  // F_rr/r and recreate the aggregate-pressure cancellation defect.
  quasar::numerics::append_scaled_product_quotient(
      sum, Real{1}, -radial_momentum_flux[kr], g.dx(), Real{1});
  quasar::numerics::append_scaled_product_quotient(
      sum, Real{1}, radial_momentum_flux[k], g.dx(), Real{1});

  // -d_z F_{z,mr}.
  quasar::numerics::append_scaled_product_quotient(
      sum, Real{1}, -axial_momentum_flux[kz], g.dy(), Real{1});
  quasar::numerics::append_scaled_product_quotient(
      sum, Real{1}, axial_momentum_flux[k], g.dy(), Real{1});

  if (decomposed_flux != 0) {
    append_flux_part(sum, radial_wave[kr], Real{-1}, g.dx(), Real{1});
    append_flux_part(sum, radial_wave[k], Real{1}, g.dx(), Real{1});
    append_flux_part(sum, axial_wave[kz], Real{-1}, g.dy(), Real{1});
    append_flux_part(sum, axial_wave[k], Real{1}, g.dy(), Real{1});

    const CrossStressFace radial_face_lo = load_cross_stress_face(
        g, flags, 0, b0r, b0z, b0phi, i, j, collocation_order,
        scheme_order, radial_cross, k);
    const CrossStressFace radial_face_hi = load_cross_stress_face(
        g, flags, 0, b0r, b0z, b0phi, i + 1, j, collocation_order,
        scheme_order, radial_cross, kr);
    const CrossStressFace axial_face_lo = load_cross_stress_face(
        g, flags, 1, b0r, b0z, b0phi, i, j, collocation_order,
        scheme_order, axial_cross, k);
    const CrossStressFace axial_face_hi = load_cross_stress_face(
        g, flags, 1, b0r, b0z, b0phi, i, j + 1, collocation_order,
        scheme_order, axial_cross, kz);
    append_cross_momentum_stress(
        sum, radial_face_hi, 0, 0, Real{-1},
        g.dx(), Real{1});
    append_cross_momentum_stress(
        sum, radial_face_lo, 0, 0, Real{1},
        g.dx(), Real{1});
    append_cross_momentum_stress(
        sum, axial_face_hi, 1, 0, Real{-1},
        g.dy(), Real{1});
    append_cross_momentum_stress(
        sum, axial_face_lo, 1, 0, Real{1},
        g.dy(), Real{1});
  }

  const Real br_pert = quasar::mhd::cell_bx(
      g, br, i, j, collocation_order);
  const Real bphi_pert = bphi[k];
  MhdBackground centre_background{};
  if (background_active != 0) {
    centre_background = quasar::mhd::load_cell_background(
        g, b0r, b0z, b0phi, i, j, collocation_order);
  }
  // Reduced tensor difference. Keep each quotient/product separate so
  // individually overflowing rotation and toroidal terms can cancel before a
  // much smaller radial-field survivor is rounded.
  quasar::numerics::append_scaled_product_quotient(
      sum, mphi[k], mphi[k], rho[k], r);
  quasar::numerics::append_scaled_product_quotient(
      sum, mr[k], -mr[k], rho[k], r);
  quasar::numerics::append_scaled_product_quotient(
      sum, centre_background.b0x, br_pert, Real{0.5}, r);
  quasar::numerics::append_scaled_product_quotient(
      sum, br_pert, br_pert, Real{1}, r);
  quasar::numerics::append_scaled_product_quotient(
      sum, centre_background.b0z, -bphi_pert, Real{0.5}, r);
  quasar::numerics::append_scaled_product_quotient(
      sum, bphi_pert, -bphi_pert, Real{1}, r);

  if (background_active != 0 && globally_curl_free == 0) {
    const MhdBackground radial_lo = mapped_face_background(
        g, flags, 0, b0r, b0z, b0phi, i, j, collocation_order);
    const MhdBackground radial_hi = mapped_face_background(
        g, flags, 0, b0r, b0z, b0phi, i + 1, j,
        collocation_order);
    const MhdBackground axial_lo = mapped_face_background(
        g, flags, 1, b0r, b0z, b0phi, i, j, collocation_order);
    const MhdBackground axial_hi = mapped_face_background(
        g, flags, 1, b0r, b0z, b0phi, i, j + 1,
        collocation_order);

    // -d_r(T0_rr) and -d_z(T0_zr), followed by
    // (T0_phiphi-T0_rr)/r = (B0r^2-B0phi^2)/r.
    append_normal_background_stress(
        sum, radial_lo, radial_hi, /*normal=*/0, g.dx());
    append_tangential_background_stress(
        sum, axial_lo, axial_hi, /*normal=*/1, /*tangent=*/0,
        g.dy());
    quasar::numerics::append_scaled_product_quotient(
        sum, centre_background.b0x, centre_background.b0x,
        Real{1}, r);
    quasar::numerics::append_scaled_product_quotient(
        sum, centre_background.b0z, -centre_background.b0z,
        Real{1}, r);
  }

  radial_momentum_rate[k] =
      quasar::numerics::finish_scaled_product_quotient_sum(sum);
}

// Append coefficient*[Cov_face + (B0_face-B0_cell).F_B]/(d0*d1).
// This is <B0.F_B>-B0_cell.<F_B>, but its dominant cell-reference product has
// been cancelled before multiplication.  The transverse covariance was formed
// in the Riemann quadrature and is retained as a scaled value.
template <int Capacity>
__device__ inline void append_conditioned_induction_flux(
    quasar::numerics::ScaledProductQuotientAccumulator<Capacity>& sum,
    const MhdBackground& face_background,
    const MhdBackground& cell_background,
    const MhdBackground& induction_flux,
    const quasar::numerics::ScaledValue& covariance,
    Real coefficient, Real d0, Real d1) {
  quasar::numerics::append_scaled_value_quotient(
      sum, covariance, coefficient, d0, d1);
  for (int component = 0; component < 3; ++component) {
    quasar::numerics::append_scaled_difference_product_quotient(
        sum, background_component(face_background, component),
        background_component(cell_background, component),
        background_component(induction_flux, component),
        coefficient, d0, d1);
  }
}

// Discretely CT-consistent active-background energy equation.  With
// E'=E-B0.b-|B0|^2/2 and static B0,
//
//   d_t E' = -div(F_E' + B0.F_B) - <B0 . db/dt>.
//
// Directly expanding that identity still exposes cancelling O(B0^2) products.
// Choose the cell background Bc as a reference and evaluate the equivalent
// conditioned correction
//
//   -D[Cov_f + (Bf-Bc).Ff] - Bc.(Rct-Rg) - Cov_V,
//
// where Rg=-D(F_B), Cov_f=<B0.F_B>-Bf.Ff, and
// Cov_V=<B0.Rct>-Bc.Rct.  Every multiplication therefore contains either a
// background difference, a rate difference, or a covariance.  Rct is read
// only after the in-plane slots have been overwritten by the final CT curl.
__global__ void split_energy_residual_kernel(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags, int cylindrical,
    int collocation_order, int scheme_order,
    const double* __restrict__ b0x,
    const double* __restrict__ b0y, const double* __restrict__ b0z,
    const double* __restrict__ x_energy,
    const double* __restrict__ y_energy,
    const double* __restrict__ x_bx, const double* __restrict__ x_by,
    const double* __restrict__ x_bz,
    const double* __restrict__ y_bx, const double* __restrict__ y_by,
    const double* __restrict__ y_bz,
    const quasar::numerics::ScaledValue* __restrict__ x_covariance,
    const quasar::numerics::ScaledValue* __restrict__ y_covariance,
    const double* __restrict__ dbx, const double* __restrict__ dby,
    const double* __restrict__ dbz,
    double* __restrict__ denergy) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);
  const std::size_t kx = g.index(i + 1, j);
  const std::size_t ky = g.index(i, j + 1);
  const MhdBackground cell_background = quasar::mhd::load_cell_background(
      g, b0x, b0y, b0z, i, j, collocation_order);
  const MhdBackground x_background_lo = mapped_face_background(
      g, flags, 0, b0x, b0y, b0z, i, j, collocation_order);
  const MhdBackground x_background_hi = mapped_face_background(
      g, flags, 0, b0x, b0y, b0z, i + 1, j, collocation_order);
  const MhdBackground y_background_lo = mapped_face_background(
      g, flags, 1, b0x, b0y, b0z, i, j, collocation_order);
  const MhdBackground y_background_hi = mapped_face_background(
      g, flags, 1, b0x, b0y, b0z, i, j + 1, collocation_order);
  const MhdBackground x_flux_lo{x_bx[k], x_by[k], x_bz[k]};
  const MhdBackground x_flux_hi{x_bx[kx], x_by[kx], x_bz[kx]};
  const MhdBackground y_flux_lo{y_bx[k], y_by[k], y_bz[k]};
  const MhdBackground y_flux_hi{y_bx[ky], y_by[ky], y_bz[ky]};

  // The conditioned MP paths use at most 33 entries: 6 F_E terms, 24
  // cylindrical face terms, and 3 rate-difference products. Cartesian MP7
  // additionally uses three volume covariances but only four face occurrences.
  quasar::numerics::ScaledProductQuotientAccumulator<40> sum;
  const auto append_flux = [&](Real flux, Real sign, Real spacing) {
    quasar::numerics::append_scaled_product_quotient(
        sum, flux, sign, spacing, Real{1});
  };

  if (cylindrical != 0) {
    const Real r = g.r_at_cell_center(i);
    append_flux(x_energy[kx], Real{-1}, g.dx());
    append_flux(x_energy[k], Real{1}, g.dx());
    append_flux(x_energy[kx], Real{-0.5}, r);
    append_flux(x_energy[k], Real{-0.5}, r);
    append_conditioned_induction_flux(
        sum, x_background_hi, cell_background, x_flux_hi,
        x_covariance[kx], Real{-1}, g.dx(), Real{1});
    append_conditioned_induction_flux(
        sum, x_background_lo, cell_background, x_flux_lo,
        x_covariance[k], Real{1}, g.dx(), Real{1});
    append_conditioned_induction_flux(
        sum, x_background_hi, cell_background, x_flux_hi,
        x_covariance[kx], Real{-0.5}, r, Real{1});
    append_conditioned_induction_flux(
        sum, x_background_lo, cell_background, x_flux_lo,
        x_covariance[k], Real{-0.5}, r, Real{1});
  } else {
    append_flux(x_energy[kx], Real{-1}, g.dx());
    append_flux(x_energy[k], Real{1}, g.dx());
    append_conditioned_induction_flux(
        sum, x_background_hi, cell_background, x_flux_hi,
        x_covariance[kx], Real{-1}, g.dx(), Real{1});
    append_conditioned_induction_flux(
        sum, x_background_lo, cell_background, x_flux_lo,
        x_covariance[k], Real{1}, g.dx(), Real{1});
  }
  append_flux(y_energy[ky], Real{-1}, g.dy());
  append_flux(y_energy[k], Real{1}, g.dy());
  append_conditioned_induction_flux(
      sum, y_background_hi, cell_background, y_flux_hi,
      y_covariance[ky], Real{-1}, g.dy(), Real{1});
  append_conditioned_induction_flux(
      sum, y_background_lo, cell_background, y_flux_lo,
      y_covariance[k], Real{1}, g.dy(), Real{1});

  const quasar::mhd::detail::CellMagneticRateView rate{dbx, dby, dbz};
  const MhdBackground rate_mean{
      quasar::mhd::cell_bx(g, dbx, i, j, collocation_order),
      quasar::mhd::cell_by(g, dby, i, j, collocation_order), dbz[k]};
  const double* x_flux_component[3] = {x_bx, x_by, x_bz};
  const double* y_flux_component[3] = {y_bx, y_by, y_bz};
  for (int component = 0; component < 3; ++component) {
    quasar::numerics::ScaledProductQuotientAccumulator<8> mismatch_sum;
    const double* fx = x_flux_component[component];
    const double* fy = y_flux_component[component];
    quasar::numerics::append_scaled_product_quotient(
        mismatch_sum, background_component(rate_mean, component),
        Real{1}, Real{1}, Real{1});
    // Rct-Rg = Rct + D_E(F_B).  Form that small mismatch in scaled form
    // before multiplying by the potentially dominant cell background.
    quasar::numerics::append_scaled_product_quotient(
        mismatch_sum, fx[kx], Real{1}, g.dx(), Real{1});
    quasar::numerics::append_scaled_product_quotient(
        mismatch_sum, fx[k], Real{-1}, g.dx(), Real{1});
    if (cylindrical != 0) {
      const Real r = g.r_at_cell_center(i);
      quasar::numerics::append_scaled_product_quotient(
          mismatch_sum, fx[kx], Real{0.5}, r, Real{1});
      quasar::numerics::append_scaled_product_quotient(
          mismatch_sum, fx[k], Real{0.5}, r, Real{1});
    }
    quasar::numerics::append_scaled_product_quotient(
        mismatch_sum, fy[ky], Real{1}, g.dy(), Real{1});
    quasar::numerics::append_scaled_product_quotient(
        mismatch_sum, fy[k], Real{-1}, g.dy(), Real{1});
    const quasar::numerics::ScaledValue mismatch =
        quasar::numerics::finish_scaled_product_quotient_sum_to_value(
            mismatch_sum);
    quasar::numerics::append_scaled_value_quotient(
        sum, mismatch,
        -background_component(cell_background, component),
        Real{1}, Real{1});
  }

  if (cylindrical == 0 && (scheme_order == 5 || scheme_order == 7)) {
    for (int component = 0; component < 3; ++component) {
      const Real background_mean =
          background_component(cell_background, component);
      const Real component_rate_mean =
          background_component(rate_mean, component);
      const quasar::numerics::ScaledValue covariance =
          quasar::mhd::detail::cell_tensor_product_correction_scaled(
              scheme_order, background_mean, component_rate_mean,
              [&](int qx, int qy) {
                const MhdBackground background =
                    quasar::mhd::detail::point_cell_background(
                        g, i, j, scheme_order, qx, qy,
                        b0x, b0y, b0z);
                return background_component(background, component);
              },
              [&](int qx, int qy) {
                return quasar::mhd::detail::
                    point_cell_magnetic_rate_component(
                        g, i, j, scheme_order, qx, qy, rate, component);
              });
      quasar::numerics::append_scaled_value_quotient(
          sum, covariance, Real{-1}, Real{1}, Real{1});
    }
  }
  denergy[k] =
      quasar::numerics::finish_scaled_product_quotient_sum(sum);
}

// Evaluate the Shu--Osher linear combination without reading an operand whose
// coefficient is exactly zero.  This is not merely an optimization: after a
// rejected stage a scratch register may legitimately contain NaN, and the
// mathematical zero-weight term must not become IEEE 0*NaN and poison the
// retry.  Nonzero terms still share the exponent-scaled reduction, preserving
// finite survivors when individually overflowing products cancel.
__device__ inline Real rk_weighted_sum(Real a, Real un, Real b, Real ustage,
                                       Real c_dt, Real dudt) {
  const bool use_a = a != Real{0};
  const bool use_b = b != Real{0};
  const bool use_c = c_dt != Real{0};
  if (use_a) {
    if (use_b) {
      return use_c
          ? quasar::numerics::product_sum3(
                a, un, b, ustage, c_dt, dudt)
          : quasar::numerics::product_sum2(a, un, b, ustage);
    }
    return use_c
        ? quasar::numerics::product_sum2(a, un, c_dt, dudt)
        : a * un;
  }
  if (use_b) {
    return use_c
        ? quasar::numerics::product_sum2(b, ustage, c_dt, dudt)
        : b * ustage;
  }
  return use_c ? c_dt * dudt : Real{0};
}

// out = a*un + b*ustage + c_dt*dudt. Cell-centred components update over the
// nx-by-ny cell interior. The staggered poloidal components also update their
// physical high-face layer: bx on i=nx and by on j=ny. Those faces participate
// in the divergence of the last cell and must receive the same SSP-RK combine as
// the other CT faces; only face slots beyond them are derived ghosts.
__global__ void rk_stage_kernel(
    quasar::Grid2D g, Real a, Real b, Real c_dt,
    const double* __restrict__ un_rho, const double* __restrict__ un_mx,
    const double* __restrict__ un_my, const double* __restrict__ un_mz,
    const double* __restrict__ un_energy, const double* __restrict__ un_bx,
    const double* __restrict__ un_by, const double* __restrict__ un_bz,
    const double* __restrict__ us_rho, const double* __restrict__ us_mx,
    const double* __restrict__ us_my, const double* __restrict__ us_mz,
    const double* __restrict__ us_energy, const double* __restrict__ us_bx,
    const double* __restrict__ us_by, const double* __restrict__ us_bz,
    const double* __restrict__ d_rho, const double* __restrict__ d_mx,
    const double* __restrict__ d_my, const double* __restrict__ d_mz,
    const double* __restrict__ d_energy, const double* __restrict__ d_bx,
    const double* __restrict__ d_by, const double* __restrict__ d_bz,
    double* __restrict__ o_rho, double* __restrict__ o_mx, double* __restrict__ o_my,
    double* __restrict__ o_mz, double* __restrict__ o_energy, double* __restrict__ o_bx,
    double* __restrict__ o_by, double* __restrict__ o_bz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i > g.nx || j > g.ny) return;
  const bool cell = i < g.nx && j < g.ny;
  const bool bx_physical = i <= g.nx && j < g.ny;
  const bool by_physical = i < g.nx && j <= g.ny;
  if (!(cell || bx_physical || by_physical)) return;
  const std::size_t k = g.index(i, j);
  // copy_state() deliberately reuses this launcher with (a,b,c)=(1,0,0).
  // Take an exact copy path rather than evaluating 0*dudt: a rejected stage may
  // have left NaN in its residual, and IEEE 0*NaN is NaN rather than zero. This
  // branch gives rollback a strong bitwise state-restoration guarantee.
  if (a == Real{1} && b == Real{0} && c_dt == Real{0}) {
    if (cell) {
      o_rho[k] = un_rho[k];
      o_mx[k] = un_mx[k];
      o_my[k] = un_my[k];
      o_mz[k] = un_mz[k];
      o_energy[k] = un_energy[k];
      o_bz[k] = un_bz[k];
    }
    if (bx_physical) o_bx[k] = un_bx[k];
    if (by_physical) o_by[k] = un_by[k];
    return;
  }
  if (cell) {
    o_rho[k] = rk_weighted_sum(
        a, un_rho[k], b, us_rho[k], c_dt, d_rho[k]);
    o_mx[k] = rk_weighted_sum(
        a, un_mx[k], b, us_mx[k], c_dt, d_mx[k]);
    o_my[k] = rk_weighted_sum(
        a, un_my[k], b, us_my[k], c_dt, d_my[k]);
    o_mz[k] = rk_weighted_sum(
        a, un_mz[k], b, us_mz[k], c_dt, d_mz[k]);
    o_energy[k] = rk_weighted_sum(
        a, un_energy[k], b, us_energy[k], c_dt, d_energy[k]);
    o_bz[k] = rk_weighted_sum(
        a, un_bz[k], b, us_bz[k], c_dt, d_bz[k]);
  }
  if (bx_physical) {
    o_bx[k] = rk_weighted_sum(
        a, un_bx[k], b, us_bx[k], c_dt, d_bx[k]);
  }
  if (by_physical) {
    o_by[k] = rk_weighted_sum(
        a, un_by[k], b, us_by[k], c_dt, d_by[k]);
  }
}

// Clamp rho >= rho_floor and gas pressure >= p_floor, re-deriving total energy
// from the floored pressure (momentum and B held fixed).
//
// Field-split convention (B = B0 + b): the stored conserved state carries the
// perturbation b in bx/by/bz and the perturbation-only magnetic energy 0.5|b|^2.
// Under this convention the gas pressure has NO B0 cross term -- the background
// is B0-independent for the floor -- so the floored energy is re-derived with the
// perturbation-only magnetic energy 0.5|b|^2 and the perturbation-only gas
// pressure. The `MhdBackground` argument to pressure() exists only to share the
// EOS call surface; it is intentionally unread (pressure(s, b0, gamma) is
// bit-for-bit pressure(s, gamma)), so with B0 inactive this is bit-identical to
// the original body. `b0_active` is accepted for ABI/signature parity; the floor
// re-derivation does not depend on it.
__global__ void apply_floors_kernel(
    quasar::Grid2D g, int /*b0_active*/, Real rho_floor, Real p_floor, Real gamma,
    double* __restrict__ rho, double* __restrict__ mx, double* __restrict__ my,
    double* __restrict__ mz, double* __restrict__ energy, double* __restrict__ bx,
    double* __restrict__ by, double* __restrict__ bz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);
  if (!isfinite(rho[k]) || rho[k] < rho_floor) rho[k] = rho_floor;
  if (!isfinite(mx[k])) mx[k] = Real{0};
  if (!isfinite(my[k])) my[k] = Real{0};
  if (!isfinite(mz[k])) mz[k] = Real{0};
  MhdState s = quasar::mhd::load_cell_state(
      g, rho, mx, my, mz, energy, bx, by, bz, i, j);
  // Perturbation-only gas pressure via the background-aware overload (B0 is
  // unread under the field-split EOS, so this equals pressure(s, gamma)).
  const Real p = pressure(s, MhdBackground{}, gamma);
  if (!isfinite(p) || p < p_floor) {
    const Real kinetic = quasar::numerics::kinetic_from_momentum(
        s.mx, s.my, s.mz, s.rho);
    // Stored magnetic energy stays the perturbation-only 0.5|b|^2 (b = bx/by/bz).
    const Real magnetic = quasar::numerics::half_squared_norm3(s.bx, s.by, s.bz);
    energy[k] = p_floor / (gamma - Real{1}) + kinetic + magnetic;
  }
}

// Axisymmetric (r,z) geometric source for the m=0 cylindrical update.
//
// In the conventional annular formulation, the only terms left outside the
// radial face/volume-weighted momentum divergence are
//
//   S_{m_r} = [rho v_phi^2 + p + |B0+b|^2/2
//                                  -(B0_phi+b_phi)^2] / r.
//
// The expression below expands every static, cross, and perturbation product
// into one common-exponent sum. This standalone API is the full geometric
// source; the solver's flux-difference calls deliberately leave radial momentum
// untouched and its cancellation-safe fused residual instead uses the
// analytically pressure-free tensor-difference kernel above. Continuity, axial
// momentum, and energy have no pointwise source. Angular momentum is handled
// conservatively inside flux_difference_kernel with the exact int(r^2 dr) cell
// moment, so no pointwise m_phi source remains. Toroidal induction remains
// metric-free, while B_r/B_z are advanced by the matching annular CT curl.
//
// On-axis handling: the radius is read at the CELL CENTER, r = (i+0.5)*dr (for a
// domain that starts on the axis, origin_x = 0), so the innermost cell i=0 has
// r = 0.5*dr > 0 and the 1/r factor is finite (= 2/dr) -- the geometric source is
// NOT singular at any cell center and the i=0 column must receive it like every
// other column. Only a non-positive or non-finite radius (which cannot occur for
// a cell center on a valid grid, but is guarded defensively) is skipped to avoid
// a division by zero.
__global__ void geometric_source_kernel(
    quasar::Grid2D g, int b0_active, Real gamma, int collocation_order,
    const double* __restrict__ rho, const double* __restrict__ mx,
    const double* __restrict__ my, const double* __restrict__ mz,
    const double* __restrict__ energy, const double* __restrict__ bx,
    const double* __restrict__ by, const double* __restrict__ bz,
    const double* __restrict__ b0x, const double* __restrict__ b0y,
    const double* __restrict__ b0z,
    double* __restrict__ drho, double* __restrict__ dmr,
    double* __restrict__ dmz_axial, double* __restrict__ dmphi,
    double* __restrict__ denergy) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const Real r = g.r_at_cell_center(i);
  if (!(r > Real{0})) return;  // defensive: skip only a non-positive cell-center radius
  const std::size_t k = g.index(i, j);
  // Slot mapping: mx = m_r, my = m_z (axial), mz = m_phi (azimuthal);
  // bx = B_r, by = B_z, bz = B_phi.
  const Real br_pert = quasar::mhd::cell_bx(
      g, bx, i, j, collocation_order);
  const Real bz_pert = quasar::mhd::cell_by(
      g, by, i, j, collocation_order);
  const Real bphi_pert = bz[k];
  MhdBackground background{};
  if (b0_active != 0) {
    background = quasar::mhd::load_cell_background(
        g, b0x, b0y, b0z, i, j, collocation_order);
  }
  MhdState s;
  s.rho = rho[k]; s.mx = mx[k]; s.my = my[k]; s.mz = mz[k];
  s.energy = energy[k]; s.bx = br_pert; s.by = bz_pert; s.bz = bphi_pert;
  const Real p_gas = pressure(s, background, gamma);
  // Fluid rotation/pressure and all total-field magnetic curvature pieces form
  // one physical source. Accumulate all eleven signed quotients before rounding
  // so dominant centrifugal/static/toroidal terms can cancel without erasing a
  // representable survivor.
  quasar::numerics::ScaledProductQuotientAccumulator<11> source_sum;
  quasar::numerics::append_scaled_product_quotient(
      source_sum, mz[k], mz[k], rho[k], r);
  quasar::numerics::append_scaled_product_quotient(
      source_sum, p_gas, Real{1}, Real{1}, r);
  quasar::numerics::append_scaled_product_quotient(
      source_sum, background.b0x, br_pert, Real{1}, r);
  quasar::numerics::append_scaled_product_quotient(
      source_sum, background.b0y, bz_pert, Real{1}, r);
  quasar::numerics::append_scaled_product_quotient(
      source_sum, background.b0z, -bphi_pert, Real{1}, r);
  quasar::numerics::append_scaled_product_quotient(
      source_sum, background.b0x, Real{0.5} * background.b0x,
      Real{1}, r);
  quasar::numerics::append_scaled_product_quotient(
      source_sum, background.b0y, Real{0.5} * background.b0y,
      Real{1}, r);
  quasar::numerics::append_scaled_product_quotient(
      source_sum, background.b0z, Real{-0.5} * background.b0z,
      Real{1}, r);
  quasar::numerics::append_scaled_product_quotient(
      source_sum, br_pert, Real{0.5} * br_pert, Real{1}, r);
  quasar::numerics::append_scaled_product_quotient(
      source_sum, bz_pert, Real{0.5} * bz_pert, Real{1}, r);
  quasar::numerics::append_scaled_product_quotient(
      source_sum, bphi_pert, Real{-0.5} * bphi_pert, Real{1}, r);
  dmr[k] +=
      quasar::numerics::finish_scaled_product_quotient_sum(source_sum);

  (void)drho;
  (void)dmz_axial;
  (void)dmphi;
  (void)denergy;
  // S_{B_phi} = 0: toroidal induction is metric-free, handled by the Cartesian
  // flux difference; the poloidal faces B_r/B_z are advanced by CT.
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_flux_difference(const MhdField2D<Real>& flux, int dir,
                                MhdField2D<Real>& dudt, stream_t stream,
                                bool cylindrical) {
  const Grid2D g = flux.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  hipLaunchKernelGGL(flux_difference_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, dir, cylindrical ? 1 : 0,
                     flux.rho.device_ptr(), flux.mx.device_ptr(), flux.my.device_ptr(),
                     flux.mz.device_ptr(), flux.energy.device_ptr(), flux.bx_face.device_ptr(),
                     flux.by_face.device_ptr(), flux.bz_cell.device_ptr(),
                     dudt.rho.device_ptr(), dudt.mx.device_ptr(), dudt.my.device_ptr(),
                     dudt.mz.device_ptr(), dudt.energy.device_ptr(), dudt.bx_face.device_ptr(),
                     dudt.by_face.device_ptr(), dudt.bz_cell.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_background_stress_correction(
    const MhdBackgroundField<Real>& b0, MhdField2D<Real>& dudt,
    BoundaryFlags4 flags, stream_t stream, bool cylindrical,
    int collocation_order, int scheme_order) {
  if (!b0.active) return;
  if (b0.globally_curl_free) return;
  const Grid2D g = dudt.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  hipLaunchKernelGGL(
      background_stress_correction_kernel, grid, block, 0,
      static_cast<hipStream_t>(stream), g, flags,
      cylindrical ? 1 : 0, collocation_order, scheme_order,
      b0.b0x_face.device_ptr(), b0.b0y_face.device_ptr(),
      b0.b0z_cell.device_ptr(), dudt.mx.device_ptr(),
      dudt.my.device_ptr(), dudt.mz.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_split_momentum_residual(
    const MhdBackgroundField<Real>& b0,
    const MhdField2D<Real>& flux_x,
    const MhdMomentumFluxParts2D<Real>& parts_x,
    const MhdField2D<Real>& flux_y,
    const MhdMomentumFluxParts2D<Real>& parts_y,
    MhdField2D<Real>& dudt, BoundaryFlags4 flags, stream_t stream,
    bool cylindrical, int collocation_order, int scheme_order) {
  if (!b0.active) return;
  const Grid2D g = dudt.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  const auto point_view = [](const MhdMomentumFluxParts2D<Real>& parts) {
    CrossBPointView view;
    for (int q = 0; q < 4; ++q) {
      view.x[q] = parts.cross_b_point[q].x.device_ptr();
      view.y[q] = parts.cross_b_point[q].y.device_ptr();
      view.z[q] = parts.cross_b_point[q].z.device_ptr();
    }
    view.quadrature_valid = parts.quadrature_valid.device_ptr();
    return view;
  };
  const CrossBPointView x_cross = point_view(parts_x);
  const CrossBPointView y_cross = point_view(parts_y);
  hipLaunchKernelGGL(
      split_momentum_residual_kernel, grid, block, 0,
      static_cast<hipStream_t>(stream), g, flags, cylindrical ? 1 : 0,
      collocation_order, scheme_order, b0.globally_curl_free ? 1 : 0,
      b0.b0x_face.device_ptr(),
      b0.b0y_face.device_ptr(), b0.b0z_cell.device_ptr(),
      flux_x.mx.device_ptr(), flux_x.my.device_ptr(), flux_x.mz.device_ptr(),
      parts_x.wave_x.device_ptr(), parts_x.wave_y.device_ptr(),
      parts_x.wave_z.device_ptr(), x_cross,
      flux_y.mx.device_ptr(), flux_y.my.device_ptr(), flux_y.mz.device_ptr(),
      parts_y.wave_x.device_ptr(), parts_y.wave_y.device_ptr(),
      parts_y.wave_z.device_ptr(), y_cross,
      dudt.mx.device_ptr(), dudt.my.device_ptr(), dudt.mz.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_cylindrical_radial_momentum_residual(
    const MhdField2D<Real>& u, const MhdBackgroundField<Real>& b0,
    const MhdField2D<Real>& flux_r, const MhdField2D<Real>& flux_z,
    MhdField2D<Real>& dudt, BoundaryFlags4 flags, stream_t stream,
    int collocation_order, int scheme_order,
    const MhdMomentumFluxParts2D<Real>* parts_r,
    const MhdMomentumFluxParts2D<Real>* parts_z) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  const int decomposed_flux =
      (b0.active && parts_r != nullptr && parts_z != nullptr) ? 1 : 0;
  const auto point_view = [](const MhdMomentumFluxParts2D<Real>* parts) {
    CrossBPointView view;
    if (parts == nullptr) return view;
    for (int q = 0; q < 4; ++q) {
      view.x[q] = parts->cross_b_point[q].x.device_ptr();
      view.y[q] = parts->cross_b_point[q].y.device_ptr();
      view.z[q] = parts->cross_b_point[q].z.device_ptr();
    }
    view.quadrature_valid = parts->quadrature_valid.device_ptr();
    return view;
  };
  const CrossBPointView radial_cross = point_view(parts_r);
  const CrossBPointView axial_cross = point_view(parts_z);
  hipLaunchKernelGGL(
      cylindrical_radial_momentum_residual_kernel, grid, block, 0,
      static_cast<hipStream_t>(stream), g, flags, b0.active ? 1 : 0,
      b0.globally_curl_free ? 1 : 0, decomposed_flux, collocation_order,
      scheme_order,
      u.rho.device_ptr(), u.mx.device_ptr(),
      u.mz.device_ptr(), u.bx_face.device_ptr(), u.bz_cell.device_ptr(),
      b0.active ? b0.b0x_face.device_ptr() : nullptr,
      b0.active ? b0.b0y_face.device_ptr() : nullptr,
      b0.active ? b0.b0z_cell.device_ptr() : nullptr,
      flux_r.mx.device_ptr(), flux_z.mx.device_ptr(),
      decomposed_flux != 0 ? parts_r->wave_x.device_ptr() : nullptr,
      decomposed_flux != 0 ? parts_z->wave_x.device_ptr() : nullptr,
      radial_cross, axial_cross,
      dudt.mx.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_rk_stage(MhdField2D<Real>& out, const MhdField2D<Real>& un,
                         const MhdField2D<Real>& ustage, const MhdField2D<Real>& dudt,
                         Real a, Real b, Real c_dt, stream_t stream) {
  const Grid2D g = out.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid((static_cast<unsigned>(g.nx) + 1 + block.x - 1) / block.x,
                  (static_cast<unsigned>(g.ny) + 1 + block.y - 1) / block.y);
  hipLaunchKernelGGL(rk_stage_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, a, b, c_dt,
                     un.rho.device_ptr(), un.mx.device_ptr(), un.my.device_ptr(),
                     un.mz.device_ptr(), un.energy.device_ptr(), un.bx_face.device_ptr(),
                     un.by_face.device_ptr(), un.bz_cell.device_ptr(),
                     ustage.rho.device_ptr(), ustage.mx.device_ptr(), ustage.my.device_ptr(),
                     ustage.mz.device_ptr(), ustage.energy.device_ptr(), ustage.bx_face.device_ptr(),
                     ustage.by_face.device_ptr(), ustage.bz_cell.device_ptr(),
                     dudt.rho.device_ptr(), dudt.mx.device_ptr(), dudt.my.device_ptr(),
                     dudt.mz.device_ptr(), dudt.energy.device_ptr(), dudt.bx_face.device_ptr(),
                     dudt.by_face.device_ptr(), dudt.bz_cell.device_ptr(),
                     out.rho.device_ptr(), out.mx.device_ptr(), out.my.device_ptr(),
                     out.mz.device_ptr(), out.energy.device_ptr(), out.bx_face.device_ptr(),
                     out.by_face.device_ptr(), out.bz_cell.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_split_energy_residual(
    const MhdBackgroundField<Real>& b0,
    const MhdField2D<Real>& flux_x,
    const MhdMomentumFluxParts2D<Real>& parts_x,
    const MhdField2D<Real>& flux_y,
    const MhdMomentumFluxParts2D<Real>& parts_y,
    MhdField2D<Real>& actual_rate, BoundaryFlags4 flags, stream_t stream,
    bool cylindrical, int collocation_order, int scheme_order) {
  if (!b0.active) return;
  const Grid2D g = actual_rate.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  hipLaunchKernelGGL(
      split_energy_residual_kernel, grid, block, 0,
      static_cast<hipStream_t>(stream), g, flags, cylindrical ? 1 : 0,
      collocation_order, scheme_order, b0.b0x_face.device_ptr(),
      b0.b0y_face.device_ptr(), b0.b0z_cell.device_ptr(),
      flux_x.energy.device_ptr(), flux_y.energy.device_ptr(),
      flux_x.bx_face.device_ptr(), flux_x.by_face.device_ptr(),
      flux_x.bz_cell.device_ptr(), flux_y.bx_face.device_ptr(),
      flux_y.by_face.device_ptr(), flux_y.bz_cell.device_ptr(),
      parts_x.b0_induction_covariance.device_ptr(),
      parts_y.b0_induction_covariance.device_ptr(),
      actual_rate.bx_face.device_ptr(), actual_rate.by_face.device_ptr(),
      actual_rate.bz_cell.device_ptr(),
      actual_rate.energy.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_apply_floors(MhdField2D<Real>& u, const MhdBackgroundField<Real>& b0,
                             Real rho_floor, Real p_floor, Real gamma, stream_t stream) {
  const Grid2D g = u.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  // The floor re-derivation is B0-independent under the field-split EOS; the
  // background-active flag is threaded only for ABI parity with the field-split
  // kernel family. No B0 buffers are read.
  const int b0_active = b0.active ? 1 : 0;
  hipLaunchKernelGGL(apply_floors_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, b0_active, rho_floor, p_floor, gamma,
                     u.rho.device_ptr(), u.mx.device_ptr(), u.my.device_ptr(),
                     u.mz.device_ptr(), u.energy.device_ptr(), u.bx_face.device_ptr(),
                     u.by_face.device_ptr(), u.bz_cell.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_geometric_source(const MhdField2D<Real>& u, MhdField2D<Real>& dudt,
                                 const MhdBackgroundField<Real>& b0,
                                 Grid2D grid, Real gamma, stream_t stream,
                                 int collocation_order) {
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 launch_grid = mhd_grid_2d(grid, block);
  hipLaunchKernelGGL(geometric_source_kernel, launch_grid, block, 0,
                     static_cast<hipStream_t>(stream), grid, b0.active ? 1 : 0,
                     gamma, collocation_order,
                     u.rho.device_ptr(), u.mx.device_ptr(), u.my.device_ptr(),
                     u.mz.device_ptr(), u.energy.device_ptr(), u.bx_face.device_ptr(),
                     u.by_face.device_ptr(), u.bz_cell.device_ptr(),
                     b0.active ? b0.b0x_face.device_ptr() : nullptr,
                     b0.active ? b0.b0y_face.device_ptr() : nullptr,
                     b0.active ? b0.b0z_cell.device_ptr() : nullptr,
                     dudt.rho.device_ptr(), dudt.mx.device_ptr(), dudt.my.device_ptr(),
                     dudt.mz.device_ptr(), dudt.energy.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
