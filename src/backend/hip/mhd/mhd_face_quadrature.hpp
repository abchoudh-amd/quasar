#pragma once

// Transverse finite-volume quadrature shared by the MHD Riemann and CT kernels.
//
// Directional reconstruction returns a face AVERAGE for every transverse cell.
// Applying a nonlinear Riemann solver directly to that average is only second
// order in multiple dimensions.  MP5/MP7 instead recover point states at the
// three-/four-point Gauss--Legendre nodes, solve the Riemann problem there, and
// integrate the point fluxes back to one face average.  The recovery polynomial
// is built from five/seven neighbouring face averages, respectively.

#include "quasar/core/grid.hpp"
#include "quasar/numerics/finite_volume_quadrature.hpp"
#include "quasar/numerics/hlld_core.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include <cmath>

namespace quasar::mhd::detail {

using numerics::MhdBackground;
using numerics::MhdFlux;
using numerics::MhdState;
using numerics::hlld::MhdMomentumFluxParts;

struct FaceInterfaceView {
  const double *Lrho, *Lmx, *Lmy, *Lmz, *Lenergy, *Lbx, *Lby, *Lbz;
  const double *Rrho, *Rmx, *Rmy, *Rmz, *Renergy, *Rbx, *Rby, *Rbz;
};

struct FaceBackgroundView {
  int active;
  const double *bx, *by, *bz;
};

struct FaceMomentumFluxParts {
  MhdMomentumFluxParts flux{};
  MhdBackground cross_b_point[4]{};
  int quadrature_valid{};
  numerics::ScaledValue b0_induction_covariance{};
};

using numerics::kMp5TransverseGaussWeights;
using numerics::kMp5TransversePointWeights;
using numerics::kMp7TransverseGaussWeights;
using numerics::kMp7TransversePointWeights;

__device__ inline MhdState load_interface_side(
    const FaceInterfaceView& v, std::size_t k, bool left) {
  MhdState s;
  if (left) {
    s.rho = v.Lrho[k]; s.mx = v.Lmx[k]; s.my = v.Lmy[k]; s.mz = v.Lmz[k];
    s.energy = v.Lenergy[k]; s.bx = v.Lbx[k]; s.by = v.Lby[k]; s.bz = v.Lbz[k];
  } else {
    s.rho = v.Rrho[k]; s.mx = v.Rmx[k]; s.my = v.Rmy[k]; s.mz = v.Rmz[k];
    s.energy = v.Renergy[k]; s.bx = v.Rbx[k]; s.by = v.Rby[k]; s.bz = v.Rbz[k];
  }
  return s;
}

__device__ inline bool state_is_admissible(const MhdState& s, Real gamma) {
  if (!(isfinite(s.rho) && s.rho > Real{0} && isfinite(s.mx) &&
        isfinite(s.my) && isfinite(s.mz) && isfinite(s.energy) &&
        isfinite(s.bx) && isfinite(s.by) && isfinite(s.bz))) {
    return false;
  }
  const Real p = numerics::pressure(s, gamma);
  return isfinite(p) && p > Real{0};
}

template <int Count, class Sample>
__device__ inline Real weighted_value(const Real (&weights)[Count],
                                         const Sample& sample) {
  const Real reference = sample(Count / 2);
  bool constant = true;
  numerics::ScaledProductQuotientAccumulator<Count> sum;
  for (int k = 0; k < Count; ++k) {
    const Real value = sample(k);
    constant = constant && value == reference;
    numerics::append_scaled_product_quotient(
        sum, weights[k], value, Real{1}, Real{1});
  }
  // Besides avoiding unnecessary cancellation, this branch makes a truly 1-D
  // state pass through transverse quadrature bit-for-bit unchanged.
  return constant ? reference
                  : numerics::finish_scaled_product_quotient_sum(sum);
}

template <class Sample>
__device__ inline Real weighted_value(const Real* weights, int count,
                                      const Sample& sample) {
  const Real reference = sample(count / 2);
  bool constant = true;
  numerics::ScaledProductQuotientAccumulator<8> sum;
  for (int k = 0; k < count; ++k) {
    const Real value = sample(k);
    constant = constant && value == reference;
    numerics::append_scaled_product_quotient(
        sum, weights[k], value, Real{1}, Real{1});
  }
  return constant ? reference
                  : numerics::finish_scaled_product_quotient_sum(sum);
}

template <class Sample>
__device__ inline Real point_value(
    int order, int node, int dir, int radial_index,
    numerics::RadialTablesView radial_tables, const Sample& sample) {
  if (dir == 1 && radial_tables.active != 0 &&
      radial_tables.contains(radial_index) &&
      radial_tables.reconstruction_width == (order >= 7 ? 7 : 5)) {
    const int width = radial_tables.reconstruction_width;
    return weighted_value(
        radial_tables.r2_row(radial_index, node), width,
        [&](int k) { return sample(k - width / 2); });
  }
  if (order >= 7) {
    return weighted_value(kMp7TransversePointWeights[node], [&](int k) {
      return sample(k - 3);
    });
  }
  return weighted_value(kMp5TransversePointWeights[node], [&](int k) {
    return sample(k - 2);
  });
}

template <class Sample>
__device__ inline Real point_value(int order, int node,
                                   const Sample& sample) {
  return point_value(
      order, node, /*dir=*/0, /*radial_index=*/0,
      numerics::RadialTablesView{}, sample);
}

__device__ inline std::size_t transverse_index(
    const Grid2D& g, int dir, int i, int j, int offset) {
  return dir == 0 ? g.index(i, j + offset) : g.index(i + offset, j);
}

__device__ inline MhdState point_interface_state(
    const Grid2D& g, int dir, int i, int j, int order, int node,
    const FaceInterfaceView& v, bool left,
    numerics::RadialTablesView radial_tables) {
  const double* rho = left ? v.Lrho : v.Rrho;
  const double* mx = left ? v.Lmx : v.Rmx;
  const double* my = left ? v.Lmy : v.Rmy;
  const double* mz = left ? v.Lmz : v.Rmz;
  const double* energy = left ? v.Lenergy : v.Renergy;
  const double* bx = left ? v.Lbx : v.Rbx;
  const double* by = left ? v.Lby : v.Rby;
  const double* bz = left ? v.Lbz : v.Rbz;
  const auto component = [&](const double* values) {
    return point_value(order, node, dir, i, radial_tables, [&](int offset) {
      return static_cast<Real>(values[transverse_index(g, dir, i, j, offset)]);
    });
  };
  return MhdState{component(rho), component(mx), component(my), component(mz),
                  component(energy), component(bx), component(by), component(bz)};
}

__device__ inline MhdBackground average_background(
    const Grid2D& g, int dir, int i, int j, const FaceBackgroundView& bg,
    int collocation_order,
    numerics::RadialTablesView radial_tables) {
  if (bg.active == 0) return {};
  return load_interface_background(g, dir, bg.bx, bg.by, bg.bz, i, j,
                                   radial_tables, collocation_order);
}

__device__ inline MhdBackground point_background(
    const Grid2D& g, int dir, int i, int j, int order, int node,
    const FaceBackgroundView& bg,
    numerics::RadialTablesView radial_tables) {
  if (bg.active == 0) return {};
  const auto component = [&](int component_index) {
    return point_value(order, node, dir, i, radial_tables, [&](int offset) {
      const int is = dir == 0 ? i : i + offset;
      const int js = dir == 0 ? j + offset : j;
      const MhdBackground value =
          average_background(g, dir, is, js, bg, /*collocation_order=*/0,
                             radial_tables);
      return component_index == 0 ? value.b0x
           : component_index == 1 ? value.b0y : value.b0z;
    });
  };
  return MhdBackground{component(0), component(1), component(2)};
}

__device__ inline Real background_component(
    const MhdBackground& value, int component) {
  return component == 0 ? value.b0x
       : component == 1 ? value.b0y : value.b0z;
}

__device__ inline Real magnetic_flux_component(
    const MhdFlux& value, int component) {
  return component == 0 ? value.bx
       : component == 1 ? value.by : value.bz;
}

__device__ inline MhdFlux riemann_flux(
    const MhdState& L, const MhdState& R, const MhdBackground& bg,
    int dir, int hll_only, Real gamma) {
  const MhdState Lr = numerics::hlld::rotate_in(L, dir);
  const MhdState Rr = numerics::hlld::rotate_in(R, dir);
  const MhdBackground bgr = numerics::hlld::rotate_in(bg, dir);
  const MhdFlux fr = hll_only != 0
      ? numerics::hlld::lax_friedrichs_flux_split_x(Lr, Rr, bgr, gamma)
      : numerics::hlld::hlld_flux_split_x(Lr, Rr, bgr, gamma);
  return numerics::hlld::rotate_out(fr, dir);
}

__device__ inline MhdMomentumFluxParts riemann_flux_parts(
    const MhdState& L, const MhdState& R, const MhdBackground& bg,
    int dir, int hll_only, Real gamma) {
  const MhdState Lr = numerics::hlld::rotate_in(L, dir);
  const MhdState Rr = numerics::hlld::rotate_in(R, dir);
  const MhdBackground bgr = numerics::hlld::rotate_in(bg, dir);
  const MhdMomentumFluxParts fr = hll_only != 0
      ? numerics::hlld::lax_friedrichs_flux_split_parts_x(Lr, Rr, bgr, gamma)
      : numerics::hlld::hlld_flux_split_parts_x(Lr, Rr, bgr, gamma);
  return numerics::hlld::rotate_out(fr, dir);
}

__device__ inline MhdFlux average_face_flux(
    const Grid2D& g, int dir, int i, int j, int scheme_order, int hll_only,
    Real gamma, const FaceInterfaceView& v, const FaceBackgroundView& bg,
    numerics::RadialTablesView radial_tables = {}) {
  const std::size_t k = g.index(i, j);
  const MhdState base_l = load_interface_side(v, k, true);
  const MhdState base_r = load_interface_side(v, k, false);
  const MhdBackground base_bg =
      average_background(g, dir, i, j, bg, hll_only != 0 ? 1 : 0,
                         radial_tables);
  if (scheme_order != 5 && scheme_order != 7) {
    return riemann_flux(base_l, base_r, base_bg, dir, hll_only, gamma);
  }

  MhdFlux flux[4]{};
  const int count = scheme_order == 5 ? 3 : 4;
  for (int q = 0; q < count; ++q) {
    const MhdState l = point_interface_state(
        g, dir, i, j, scheme_order, q, v, true, radial_tables);
    const MhdState r = point_interface_state(
        g, dir, i, j, scheme_order, q, v, false, radial_tables);
    if (!state_is_admissible(l, gamma) || !state_is_admissible(r, gamma)) {
      // A transverse polynomial has no invariant-domain guarantee.  Falling
      // back to the already-admissible face-average problem is conservative and
      // local, matching the normal MP reconstruction's troubled-face policy.
      return riemann_flux(base_l, base_r, base_bg, dir, hll_only, gamma);
    }
    const MhdBackground point_bg =
        point_background(g, dir, i, j, scheme_order, q, bg, radial_tables);
    flux[q] = riemann_flux(l, r, point_bg, dir, hll_only, gamma);
  }

  const auto component = [&](Real MhdFlux::*member) {
    if (dir == 1 && radial_tables.active != 0 &&
        radial_tables.contains(i)) {
      return weighted_value(radial_tables.r3_row(i), count, [&](int q) {
        return flux[q].*member;
      });
    }
    if (scheme_order == 5) {
      return weighted_value(kMp5TransverseGaussWeights, [&](int q) {
        return flux[q].*member;
      });
    }
    return weighted_value(kMp7TransverseGaussWeights, [&](int q) {
      return flux[q].*member;
    });
  };
  return MhdFlux{component(&MhdFlux::rho), component(&MhdFlux::mx),
                 component(&MhdFlux::my), component(&MhdFlux::mz),
                 component(&MhdFlux::energy), component(&MhdFlux::bx),
                 component(&MhdFlux::by), component(&MhdFlux::bz)};
}

__device__ inline FaceMomentumFluxParts average_face_flux_parts(
    const Grid2D& g, int dir, int i, int j, int scheme_order, int hll_only,
    Real gamma, const FaceInterfaceView& v, const FaceBackgroundView& bg,
    numerics::RadialTablesView radial_tables = {}) {
  const std::size_t k = g.index(i, j);
  const MhdState base_l = load_interface_side(v, k, true);
  const MhdState base_r = load_interface_side(v, k, false);
  const MhdBackground base_bg =
      average_background(g, dir, i, j, bg, hll_only != 0 ? 1 : 0,
                         radial_tables);
  if (scheme_order != 5 && scheme_order != 7) {
    FaceMomentumFluxParts result;
    result.flux =
        riemann_flux_parts(base_l, base_r, base_bg, dir, hll_only, gamma);
    result.cross_b_point[0] = result.flux.cross_b;
    return result;
  }

  MhdMomentumFluxParts flux[4]{};
  MhdBackground point_bg[4]{};
  const int count = scheme_order == 5 ? 3 : 4;
  for (int q = 0; q < count; ++q) {
    const MhdState l = point_interface_state(
        g, dir, i, j, scheme_order, q, v, true, radial_tables);
    const MhdState r = point_interface_state(
        g, dir, i, j, scheme_order, q, v, false, radial_tables);
    if (!state_is_admissible(l, gamma) || !state_is_admissible(r, gamma)) {
      FaceMomentumFluxParts result;
      result.flux =
          riemann_flux_parts(base_l, base_r, base_bg, dir, hll_only, gamma);
      result.cross_b_point[0] = result.flux.cross_b;
      return result;
    }
    point_bg[q] =
        point_background(g, dir, i, j, scheme_order, q, bg, radial_tables);
    flux[q] = riemann_flux_parts(l, r, point_bg[q], dir, hll_only, gamma);
  }

  const auto average = [&](const auto& sample) {
    if (dir == 1 && radial_tables.active != 0 &&
        radial_tables.contains(i)) {
      return weighted_value(radial_tables.r3_row(i), count, sample);
    }
    if (scheme_order == 5) {
      return weighted_value(kMp5TransverseGaussWeights, sample);
    }
    return weighted_value(kMp7TransverseGaussWeights, sample);
  };
  const auto material = [&](Real MhdFlux::*member) {
    return average([&](int q) { return flux[q].material.*member; });
  };
  const auto wave = [&](Real MhdFlux::*member) {
    return average([&](int q) { return flux[q].wave.*member; });
  };
  const auto cross = [&](Real MhdBackground::*member) {
    return average([&](int q) { return flux[q].cross_b.*member; });
  };
  FaceMomentumFluxParts result;
  result.flux.material = MhdFlux{
      material(&MhdFlux::rho), material(&MhdFlux::mx),
      material(&MhdFlux::my), material(&MhdFlux::mz),
      material(&MhdFlux::energy), material(&MhdFlux::bx),
      material(&MhdFlux::by), material(&MhdFlux::bz)};
  result.flux.wave = MhdFlux{
      wave(&MhdFlux::rho), wave(&MhdFlux::mx), wave(&MhdFlux::my),
      wave(&MhdFlux::mz), wave(&MhdFlux::energy), wave(&MhdFlux::bx),
      wave(&MhdFlux::by), wave(&MhdFlux::bz)};
  result.flux.cross_b = MhdBackground{
      cross(&MhdBackground::b0x), cross(&MhdBackground::b0y),
      cross(&MhdBackground::b0z)};
  for (int q = 0; q < count; ++q) {
    result.cross_b_point[q] = flux[q].cross_b;
  }
  numerics::ScaledQuaternaryAccumulator covariance_sum;
  for (int component = 0; component < 3; ++component) {
    const Real background_mean =
        background_component(base_bg, component);
    const Real flux_mean =
        magnetic_flux_component(result.flux.material, component);
    const auto background_sample = [&](int q) {
      return background_component(point_bg[q], component);
    };
    const auto flux_sample = [&](int q) {
      return magnetic_flux_component(flux[q].material, component);
    };
    numerics::ScaledValue covariance;
    if (dir == 1 && radial_tables.active != 0 &&
        radial_tables.contains(i)) {
      covariance = numerics::transverse_product_correction_scaled(
          radial_tables.r3_row(i), count, background_mean, flux_mean,
          background_sample, flux_sample);
    } else if (scheme_order == 5) {
      covariance = numerics::transverse_product_correction_scaled(
          kMp5TransverseGaussWeights, background_mean, flux_mean,
          background_sample, flux_sample);
    } else {
      covariance = numerics::transverse_product_correction_scaled(
          kMp7TransverseGaussWeights, background_mean, flux_mean,
          background_sample, flux_sample);
    }
    numerics::append_scaled_value_product(
        covariance_sum, covariance, Real{1}, Real{1}, Real{1});
  }
  result.b0_induction_covariance =
      numerics::finish_scaled_quaternary_sum_to_value(covariance_sum);
  result.quadrature_valid = 1;
  return result;
}

}  // namespace quasar::mhd::detail
