// Conditioned active-background split-energy residual kernel.

#include "mhd_update_detail.hpp"

#include "backend/hip/hip_check.hpp"

namespace {

using quasar::Real;
using quasar::numerics::MhdBackground;
using quasar::numerics::MhdState;
using quasar::numerics::pressure;

using namespace quasar::mhd::update_detail;

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
template <bool Cylindrical>
__global__ void QUASAR_LAUNCH_BOUNDS_256 split_energy_residual_kernel(
    quasar::Grid2D g, quasar::mhd::BoundaryFlags4 flags,
    int collocation_order, int scheme_order,
    quasar::numerics::RadialTablesView radial_tables,
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
  const quasar::numerics::RadialTablesView tables = Cylindrical
      ? radial_tables : quasar::numerics::RadialTablesView{};
  MhdBackground cell_background;
  if constexpr (Cylindrical) {
    cell_background = quasar::mhd::load_cell_background(
        g, b0x, b0y, b0z, i, j, tables, collocation_order);
  } else {
    cell_background = quasar::mhd::load_cell_background(
        g, b0x, b0y, b0z, i, j, collocation_order);
  }
  const MhdBackground x_background_lo = mapped_face_background<Cylindrical>(
      g, flags, 0, b0x, b0y, b0z, i, j, collocation_order, tables);
  const MhdBackground x_background_hi = mapped_face_background<Cylindrical>(
      g, flags, 0, b0x, b0y, b0z, i + 1, j, collocation_order,
      tables);
  const MhdBackground y_background_lo = mapped_face_background<Cylindrical>(
      g, flags, 1, b0x, b0y, b0z, i, j, collocation_order, tables);
  const MhdBackground y_background_hi = mapped_face_background<Cylindrical>(
      g, flags, 1, b0x, b0y, b0z, i, j + 1, collocation_order,
      tables);
  const MhdBackground x_flux_lo{x_bx[k], x_by[k], x_bz[k]};
  const MhdBackground x_flux_hi{x_bx[kx], x_by[kx], x_bz[kx]};
  const MhdBackground y_flux_lo{y_bx[k], y_by[k], y_bz[k]};
  const MhdBackground y_flux_hi{y_bx[ky], y_by[ky], y_bz[ky]};

  // The conditioned MP paths use at most 36 entries: 6 F_E terms, 24
  // cylindrical face terms, 3 rate-difference products, and 3 volume
  // covariances. Cartesian paths have only four face occurrences.
  quasar::numerics::ScaledProductQuotientAccumulator<40> sum;
  const auto append_flux = [&](Real flux, Real sign, Real spacing) {
    quasar::numerics::append_scaled_product_quotient(
        sum, flux, sign, spacing, Real{1});
  };

  if constexpr (Cylindrical) {
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
  MhdBackground rate_mean;
  if constexpr (Cylindrical) {
    rate_mean = MhdBackground{
        quasar::mhd::cell_bx(g, dbx, i, j, tables, collocation_order),
        quasar::mhd::cell_by(g, dby, i, j, tables, collocation_order),
        dbz[k]};
  } else {
    rate_mean = MhdBackground{
        quasar::mhd::cell_bx(g, dbx, i, j, collocation_order),
        quasar::mhd::cell_by(g, dby, i, j, collocation_order),
        dbz[k]};
  }
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
    if constexpr (Cylindrical) {
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

  if (scheme_order == 5 || scheme_order == 7) {
    for (int component = 0; component < 3; ++component) {
      const Real background_mean =
          background_component(cell_background, component);
      const Real component_rate_mean =
          background_component(rate_mean, component);
      const auto background_sample = [&](int qx, int qy) {
        return quasar::mhd::detail::
            point_cell_background_component<Cylindrical>(
            g, i, j, scheme_order, qx, qy, b0x, b0y, b0z, component,
            tables);
      };
      const auto rate_sample = [&](int qx, int qy) {
        return quasar::mhd::detail::
            point_cell_magnetic_rate_component<Cylindrical>(
            g, i, j, scheme_order, qx, qy, rate, component,
            tables);
      };
      quasar::numerics::ScaledValue covariance;
      if (Cylindrical && component == 2 && tables.active != 0 &&
          tables.contains(i)) {
        covariance =
            quasar::mhd::detail::cell_tensor_product_difference_scaled(
                scheme_order, i, tables,
                background_mean, component_rate_mean,
                background_sample, rate_sample);
      } else {
        covariance =
            quasar::mhd::detail::
                cell_tensor_product_correction_scaled<Cylindrical>(
                scheme_order, i, tables,
                background_mean, component_rate_mean,
                background_sample, rate_sample);
      }
      quasar::numerics::append_scaled_value_quotient(
          sum, covariance, Real{-1}, Real{1}, Real{1});
    }
  }
  denergy[k] =
      quasar::numerics::finish_scaled_product_quotient_sum(sum);
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_split_energy_residual(
    const MhdBackgroundField<Real>& b0,
    const MhdField2D<Real>& flux_x,
    const MhdMomentumFluxParts2D<Real>& parts_x,
    const MhdField2D<Real>& flux_y,
    const MhdMomentumFluxParts2D<Real>& parts_y,
    MhdField2D<Real>& actual_rate, BoundaryFlags4 flags, stream_t stream,
    bool cylindrical, int collocation_order, int scheme_order,
    quasar::numerics::RadialTablesView radial_tables) {
  if (!b0.active) return;
  const Grid2D g = actual_rate.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = mhd_grid_2d(g, block);
  const auto launch = [&](auto cylindrical_tag) {
    constexpr bool kCylindrical = decltype(cylindrical_tag)::value;
    const auto tables = kCylindrical
        ? radial_tables : quasar::numerics::RadialTablesView{};
    hipLaunchKernelGGL(
        (split_energy_residual_kernel<kCylindrical>), grid, block, 0,
        static_cast<hipStream_t>(stream), g, flags, collocation_order,
        scheme_order, tables, b0.b0x_face.device_ptr(),
        b0.b0y_face.device_ptr(), b0.b0z_cell.device_ptr(),
        flux_x.energy.device_ptr(), flux_y.energy.device_ptr(),
        flux_x.bx_face.device_ptr(), flux_x.by_face.device_ptr(),
        flux_x.bz_cell.device_ptr(), flux_y.bx_face.device_ptr(),
        flux_y.by_face.device_ptr(), flux_y.bz_cell.device_ptr(),
        parts_x.b0_induction_covariance.device_ptr(),
        parts_y.b0_induction_covariance.device_ptr(),
        actual_rate.bx_face.device_ptr(), actual_rate.by_face.device_ptr(),
        actual_rate.bz_cell.device_ptr(), actual_rate.energy.device_ptr());
  };
  if (cylindrical) {
    launch(std::true_type{});
  } else {
    launch(std::false_type{});
  }
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
