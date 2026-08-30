// PIC diagnostics.
//
// Every reduction here runs on the device. What remains on the host is a
// three-scalar epilogue: the kernels return a sum in a (mantissa, exponent)
// frame, and this file converts that to a Real and decides which exception to
// raise. That epilogue costs the same whether the grid has a thousand cells or
// a billion.
//
// The previous implementation downloaded all six field components and every
// particle record on each call and reduced them in long double. The device
// reductions cannot reproduce long double, so they are held to the standard the
// Grad-Shafranov port established: deterministic, compensated, and measured
// against a long-double oracle in the equivalence tests. See
// src/backend/hip/pic/diagnostics_hip.hip.
//
// Why the (mantissa, exponent) frame survives into the host epilogue rather
// than the kernel just returning a Real: the terms being summed span an
// enormous dynamic range, and the final multiplier (0.5 for the Yee energy, a
// square root for the RMS) has to be applied before deciding whether the answer
// is representable at all. Collapsing to a Real inside the kernel would throw
// away the exponent that makes that decision possible.

#include "quasar/physics/pic/diagnostics.hpp"

#include "quasar/physics/pic/kernels.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace quasar::pic {

namespace {

// A positive total as a normalized mantissa in [0.5,1) and a binary exponent.
// An infinite mantissa marks a poisoned sum.
struct ScaledLongValue {
  long double mantissa{0.0L};
  int exponent{0};
};

// Collapse a device accumulator into the normalized frame the conversions
// below expect. The sum and its Kahan correction are added in long double so
// the compensation the kernel accumulated is not discarded at the boundary.
ScaledLongValue normalized(const PicScaledSum& sum) {
  if (sum.invalid) {
    return {std::numeric_limits<long double>::infinity(), 0};
  }
  if (!sum.initialized) return {};
  const long double total =
      static_cast<long double>(sum.sum) + static_cast<long double>(sum.correction);
  if (!(total > 0.0L)) return {};
  int adjustment = 0;
  return {std::frexp(total, &adjustment), sum.exponent + adjustment};
}

Real scaled_sum_to_real(const PicScaledSum& sum, long double multiplier,
                        const char* error) {
  const auto value = normalized(sum);
  if (value.mantissa == 0.0L) return Real{0};
  long double mantissa = value.mantissa * multiplier;
  int adjustment = 0;
  mantissa = std::frexp(mantissa, &adjustment);
  const long double result = std::scalbn(
      mantissa, value.exponent + adjustment);
  if (!std::isfinite(result)
      || result > static_cast<long double>(std::numeric_limits<Real>::max())) {
    throw std::overflow_error{error};
  }
  const Real converted = static_cast<Real>(result);
  if (result != 0.0L && converted == Real{0}) {
    throw std::underflow_error{error};
  }
  return converted;
}

Real scaled_rms(const PicScaledSum& numerator, const PicScaledSum& denominator,
                const char* error) {
  const auto num = normalized(numerator);
  const auto den = normalized(denominator);
  if (!(den.mantissa > 0.0L) || !std::isfinite(den.mantissa)
      || !std::isfinite(num.mantissa)) {
    throw std::overflow_error{error};
  }
  if (num.mantissa == 0.0L) return Real{0};
  int exponent = num.exponent - den.exponent;
  long double mantissa = num.mantissa / den.mantissa;
  // The square root halves the exponent, so an odd exponent is folded into the
  // mantissa first rather than rounded away.
  if (exponent % 2 != 0) {
    mantissa *= 2.0L;
    --exponent;
  }
  const long double rms = std::scalbn(std::sqrt(mantissa), exponent / 2);
  if (!std::isfinite(rms)
      || rms > static_cast<long double>(std::numeric_limits<Real>::max())) {
    throw std::overflow_error{error};
  }
  const Real converted = static_cast<Real>(rms);
  if (rms != 0.0L && converted == Real{0}) {
    throw std::underflow_error{error};
  }
  return converted;
}

bool both_faces_periodic(const boundary::BoundarySpec& boundary, int low,
                         int high) {
  return boundary.field[low] == "periodic" && boundary.field[high] == "periodic";
}

}  // namespace

std::size_t alive_count(const ParticleSpecies& species) {
  // Device-side reduction over the alive flags — avoids copying all seven
  // particle arrays to the host just to count survivors.
  return ::launch_pic_alive_count(species, nullptr);
}

Real total_kinetic_energy(const ParticleSpecies& species) {
  const PicScaledSum ke = launch_pic_kinetic_energy(species, nullptr);
  return scaled_sum_to_real(
      ke, 1.0L, "total_kinetic_energy: energy is not representable");
}

Real total_em_energy(const YeeField2D<Real>& fields, const Grid2D& grid,
                     const boundary::BoundarySpec& boundary,
                     bool cylindrical) {
  // Normalized natural units (c=eps0=mu0=1). Each staggered sample owns its
  // primal/dual control volume. This is positive definite even at the Nyquist
  // mode; collocating by averaging before squaring would incorrectly erase a
  // checkerboard field and is not the Yee energy norm.
  if (fields.ex.size() < grid.storage_size()) return Real{0};

  const PicScaledSum twice_energy = launch_pic_field_energy(
      fields, grid, both_faces_periodic(boundary, 0, 1) ? 1 : 0,
      both_faces_periodic(boundary, 2, 3) ? 1 : 0, cylindrical ? 1 : 0,
      nullptr);
  if (twice_energy.nonfinite_input) {
    throw std::domain_error{"total_em_energy: field contains a non-finite value"};
  }
  return scaled_sum_to_real(
      twice_energy, 0.5L, "total_em_energy: energy is not representable");
}

Real total_em_energy(const YeeField2D<Real>& fields, const Grid2D& grid,
                     bool cylindrical) {
  return total_em_energy(fields, grid, boundary::BoundarySpec{}, cylindrical);
}

Real total_em_energy(const EmPic2D3V& solver) {
  return total_em_energy(solver.fields(), solver.grid(), solver.config().boundary,
                         solver.config().geometry == "cylindrical");
}

Real gauss_residual(const YeeField2D<Real>& fields,
                    const ScalarGrid2D<Real>& charge, int fdtd_order,
                    const boundary::BoundarySpec& boundary, bool cylindrical) {
  // Gauss's law residual ‖div(E)-rho‖_2 using the same forward staggered
  // derivative that annihilates Ampere's curl. Periodic axes wrap explicitly;
  // wall axes read their maintained high-face halo.
  const auto& grid = fields.grid;
  if (fields.ex.size() < grid.storage_size() ||
      charge.values.size() < grid.storage_size()) return Real{0};

  const PicGaussResidualSums sums = launch_pic_gauss_residual(
      fields, charge, fdtd_order, both_faces_periodic(boundary, 0, 1) ? 1 : 0,
      both_faces_periodic(boundary, 2, 3) ? 1 : 0, cylindrical ? 1 : 0,
      nullptr);
  if (sums.weighted_square.nonfinite_input) {
    throw std::domain_error{
        "gauss_residual: divergence or charge contains a non-finite value"};
  }
  return scaled_rms(
      sums.weighted_square, sums.volume,
      "gauss_residual: residual norm or domain volume is not representable");
}

Real gauss_residual(EmPic2D3V& solver) {
  return gauss_residual(solver.fields(), solver.charge_density(),
                        solver.config().fdtd_order, solver.config().boundary,
                        solver.config().geometry == "cylindrical");
}

Real electric_divergence_norm(const YeeField2D<Real>& fields, int fdtd_order,
                              const boundary::BoundarySpec& boundary,
                              bool cylindrical) {
  ScalarGrid2D<Real> zero{fields.grid};
  return gauss_residual(fields, zero, fdtd_order, boundary, cylindrical);
}

}  // namespace quasar::pic
