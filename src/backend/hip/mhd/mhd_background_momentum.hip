// Background and split-momentum residual kernels for ideal MHD.

#include "mhd_update_detail.hpp"

#include "backend/hip/hip_check.hpp"

namespace {

using quasar::Real;
using quasar::numerics::MhdBackground;
using quasar::numerics::MhdState;
using quasar::numerics::pressure;

using namespace quasar::mhd::update_detail;

// Add -div(F_m(B0)) once per cell.  Both directions, and in cylindrical
// geometry the radial curvature source, enter the same common-exponent sum for
// each momentum component.  This prevents a curl-free dominant B0 from leaving
// an O(epsilon*B0^2/h) force merely because x/r and y/z were rounded first.
template <bool Cylindrical>
__global__ void QUASAR_LAUNCH_BOUNDS_256 background_stress_correction_kernel(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags,
    int collocation_order, int scheme_order,
    quasar::numerics::RadialTablesView radial_tables,
    const double* __restrict__ b0x,
    const double* __restrict__ b0y, const double* __restrict__ b0z,
    double* __restrict__ dmx, double* __restrict__ dmy,
    double* __restrict__ dmz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= g.nx || j >= g.ny) return;
  const std::size_t k = g.index(i, j);
  const quasar::numerics::RadialTablesView tables = Cylindrical
      ? radial_tables : quasar::numerics::RadialTablesView{};
  const BackgroundStressFace xface_lo = load_background_stress_face<Cylindrical>(
      g, flags, 0, b0x, b0y, b0z, i, j,
      collocation_order, scheme_order, tables);
  const BackgroundStressFace xface_hi = load_background_stress_face<Cylindrical>(
      g, flags, 0, b0x, b0y, b0z, i + 1, j,
      collocation_order, scheme_order, tables);
  const BackgroundStressFace yface_lo = load_background_stress_face<Cylindrical>(
      g, flags, 1, b0x, b0y, b0z, i, j,
      collocation_order, scheme_order, tables);
  const BackgroundStressFace yface_hi = load_background_stress_face<Cylindrical>(
      g, flags, 1, b0x, b0y, b0z, i, j + 1,
      collocation_order, scheme_order, tables);
  if constexpr (!Cylindrical) {
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
  quasar::numerics::ScaledProductQuotientAccumulator<48> axial_sum;
  append_cylindrical_background_axial_residual(
      axial_sum, xface_lo, xface_hi, yface_lo, yface_hi,
      scheme_order, g.dx(), g.dy(), r);
  dmy[k] += quasar::numerics::finish_scaled_product_quotient_sum(axial_sum);

  const CylindricalAngularMetric angular =
      cylindrical_angular_metric(g.dx(), r);
  quasar::numerics::ScaledProductQuotientAccumulator<16> phi_sum;
  append_cylindrical_background_azimuthal_residual(
      phi_sum, xface_lo, xface_hi, yface_lo, yface_hi,
      scheme_order, g.dx(), g.dy(), angular);
  dmz[k] += quasar::numerics::finish_scaled_product_quotient_sum(phi_sum);
}

// Complete active-background momentum residual.  Every face contribution from
// both directions is kept as material stress H, factored cross stress C, and
// Riemann correction W until this single reduction.  Static T0 terms share the
// same accumulator, so neither a directional update nor a later correction can
// erase a small survivor beside a dominant cancelling stress.
template <bool Cylindrical>
__global__ void QUASAR_LAUNCH_BOUNDS_256 split_momentum_residual_kernel(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags,
    int collocation_order, int scheme_order, int globally_curl_free,
    quasar::numerics::RadialTablesView radial_tables,
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
  const quasar::numerics::RadialTablesView tables = Cylindrical
      ? radial_tables : quasar::numerics::RadialTablesView{};

  const CrossStressFace xface_lo = load_cross_stress_face<Cylindrical>(
      g, flags, 0, b0x, b0y, b0z, i, j, collocation_order,
      scheme_order, x_cross, k, tables);
  const CrossStressFace xface_hi = load_cross_stress_face<Cylindrical>(
      g, flags, 0, b0x, b0y, b0z, i + 1, j, collocation_order,
      scheme_order, x_cross, kx, tables);
  const CrossStressFace yface_lo = load_cross_stress_face<Cylindrical>(
      g, flags, 1, b0x, b0y, b0z, i, j, collocation_order,
      scheme_order, y_cross, k, tables);
  const CrossStressFace yface_hi = load_cross_stress_face<Cylindrical>(
      g, flags, 1, b0x, b0y, b0z, i, j + 1, collocation_order,
      scheme_order, y_cross, ky, tables);
  BackgroundStressFace xbackground_lo{};
  BackgroundStressFace xbackground_hi{};
  BackgroundStressFace ybackground_lo{};
  BackgroundStressFace ybackground_hi{};
  if (globally_curl_free == 0) {
    xbackground_lo = background_stress_from_cross_face<Cylindrical>(
        g, flags, 0, b0x, b0y, b0z, i, j, scheme_order, xface_lo,
        tables);
    xbackground_hi = background_stress_from_cross_face<Cylindrical>(
        g, flags, 0, b0x, b0y, b0z, i + 1, j, scheme_order, xface_hi,
        tables);
    ybackground_lo = background_stress_from_cross_face<Cylindrical>(
        g, flags, 1, b0x, b0y, b0z, i, j, scheme_order, yface_lo,
        tables);
    ybackground_hi = background_stress_from_cross_face<Cylindrical>(
        g, flags, 1, b0x, b0y, b0z, i, j + 1, scheme_order, yface_hi,
        tables);
  }

  if constexpr (!Cylindrical) {
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
  quasar::numerics::ScaledProductQuotientAccumulator<128> axial_sum;
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
    append_cylindrical_background_axial_residual(
        axial_sum, xbackground_lo, xbackground_hi,
        ybackground_lo, ybackground_hi,
        scheme_order, g.dx(), g.dy(), r);
  }
  dmy[k] = quasar::numerics::finish_scaled_product_quotient_sum(axial_sum);

  // Cylindrical azimuthal momentum uses the exact angular-momentum cell moment
  // in r and an ordinary axial face difference.
  const CylindricalAngularMetric angular =
      cylindrical_angular_metric(g.dx(), r);
  quasar::numerics::ScaledProductQuotientAccumulator<64> phi_sum;
  for (int channel = 0; channel < 2; ++channel) {
    const double* radial = channel == 0 ? x_hz : x_wz;
    append_flux_part(
        phi_sum, radial[kx], -angular.hi_weight,
        g.dx(), angular.moment_norm);
    append_flux_part(
        phi_sum, radial[k], angular.lo_weight,
        g.dx(), angular.moment_norm);
    const double* axial = channel == 0 ? y_hz : y_wz;
    append_flux_part(phi_sum, axial[ky], Real{-1}, g.dy(), Real{1});
    append_flux_part(phi_sum, axial[k], Real{1}, g.dy(), Real{1});
  }
  append_cross_momentum_stress(
      phi_sum, xface_hi, 0, 2, -angular.hi_weight,
      g.dx(), angular.moment_norm);
  if (angular.lo_weight != Real{0}) {
    append_cross_momentum_stress(
        phi_sum, xface_lo, 0, 2, angular.lo_weight,
        g.dx(), angular.moment_norm);
  }
  append_cross_momentum_stress(
      phi_sum, yface_hi, 1, 2, Real{-1},
      g.dy(), Real{1});
  append_cross_momentum_stress(
      phi_sum, yface_lo, 1, 2, Real{1},
      g.dy(), Real{1});
  if (globally_curl_free == 0) {
    append_cylindrical_background_azimuthal_residual(
        phi_sum, xbackground_lo, xbackground_hi,
        ybackground_lo, ybackground_hi,
        scheme_order, g.dx(), g.dy(), angular);
  }
  dmz[k] = quasar::numerics::finish_scaled_product_quotient_sum(phi_sum);
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_background_stress_correction(
    const MhdBackgroundField<Real>& b0, MhdField2D<Real>& dudt,
    BoundaryFlags4 flags, stream_t stream, bool cylindrical,
    int collocation_order, int scheme_order,
    quasar::numerics::RadialTablesView radial_tables) {
  if (!b0.active) return;
  if (b0.globally_curl_free) return;
  const Grid2D g = dudt.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  const auto launch = [&](auto cylindrical_tag) {
    constexpr bool kCylindrical = decltype(cylindrical_tag)::value;
    const auto tables = kCylindrical
        ? radial_tables : quasar::numerics::RadialTablesView{};
    hipLaunchKernelGGL(
        (background_stress_correction_kernel<kCylindrical>),
        grid, block, 0, static_cast<hipStream_t>(stream), g, flags,
        collocation_order, scheme_order, tables,
        b0.b0x_face.device_ptr(), b0.b0y_face.device_ptr(),
        b0.b0z_cell.device_ptr(), dudt.mx.device_ptr(),
        dudt.my.device_ptr(), dudt.mz.device_ptr());
  };
  if (cylindrical) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_mhd_split_momentum_residual(
    const MhdBackgroundField<Real>& b0,
    const MhdField2D<Real>& flux_x,
    const MhdMomentumFluxParts2D<Real>& parts_x,
    const MhdField2D<Real>& flux_y,
    const MhdMomentumFluxParts2D<Real>& parts_y,
    MhdField2D<Real>& dudt, BoundaryFlags4 flags, stream_t stream,
    bool cylindrical, int collocation_order, int scheme_order,
    quasar::numerics::RadialTablesView radial_tables) {
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
  const auto launch = [&](auto cylindrical_tag) {
    constexpr bool kCylindrical = decltype(cylindrical_tag)::value;
    const auto tables = kCylindrical
        ? radial_tables : quasar::numerics::RadialTablesView{};
    hipLaunchKernelGGL(
        (split_momentum_residual_kernel<kCylindrical>), grid, block, 0,
        static_cast<hipStream_t>(stream), g, flags,
        collocation_order, scheme_order, b0.globally_curl_free ? 1 : 0,
        tables, b0.b0x_face.device_ptr(), b0.b0y_face.device_ptr(),
        b0.b0z_cell.device_ptr(), flux_x.mx.device_ptr(),
        flux_x.my.device_ptr(), flux_x.mz.device_ptr(),
        parts_x.wave_x.device_ptr(), parts_x.wave_y.device_ptr(),
        parts_x.wave_z.device_ptr(), x_cross, flux_y.mx.device_ptr(),
        flux_y.my.device_ptr(), flux_y.mz.device_ptr(),
        parts_y.wave_x.device_ptr(), parts_y.wave_y.device_ptr(),
        parts_y.wave_z.device_ptr(), y_cross, dudt.mx.device_ptr(),
        dudt.my.device_ptr(), dudt.mz.device_ptr());
  };
  if (cylindrical) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
