#include "quasar/physics/pic/particle_sampling.hpp"

#include "quasar/backend/memory.hpp"
#include "quasar/numerics/scaled_arithmetic.hpp"
#include "quasar/physics/pic/kernels.hpp"

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace quasar::pic {
namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument{std::string{message}};
}

// A finite, strictly positive interval whose midpoint strata are still
// resolvable at the requested count. The host helper this replaces made the
// same three checks, and they belong here rather than in a kernel: they are
// properties of the configuration, not of any particle.
void validate_extent(Real lower, Real upper, std::size_t count,
                     const char* label) {
  const Real width = upper - lower;
  if (!(std::isfinite(width) && width > Real{0})) {
    throw std::invalid_argument{std::string{"quiet-start "} + label +
                                " extent is not representable"};
  }
  const Real half_cell = Real{0.5} * (width / static_cast<Real>(count));
  if (!(std::isfinite(half_cell) && half_cell > Real{0})) {
    throw std::invalid_argument{std::string{"quiet-start "} + label +
                                " strata are not representable"};
  }
  if (lower + half_cell == lower || upper - half_cell == upper) {
    throw std::invalid_argument{
        std::string{"quiet-start "} + label +
        " strata collapse in floating-point precision"};
  }
}

// Positive product divided by a positive integer, carried in a mantissa and an
// exponent so a representable answer is not lost to an intermediate that is
// not. Same reason the analytic evaluators carry ScaledValue.
Real scaled_product_over(const Real* factors, std::size_t factor_count,
                         std::size_t divisor, const char* label) {
  Real mantissa = Real{1};
  int exponent = 0;
  for (std::size_t i = 0; i < factor_count; ++i) {
    if (!(std::isfinite(factors[i]) && factors[i] > Real{0})) {
      throw std::invalid_argument{std::string{label} +
                                  " is not representable"};
    }
    int part_exponent = 0;
    mantissa *= std::frexp(factors[i], &part_exponent);
    exponent += part_exponent;
    int adjustment = 0;
    mantissa = std::frexp(mantissa, &adjustment);
    exponent += adjustment;
  }
  int divisor_exponent = 0;
  const Real divisor_mantissa =
      std::frexp(static_cast<Real>(divisor), &divisor_exponent);
  mantissa /= divisor_mantissa;
  exponent -= divisor_exponent;
  int adjustment = 0;
  mantissa = std::frexp(mantissa, &adjustment);
  exponent += adjustment;
  const Real result = std::ldexp(mantissa, exponent);
  if (!(std::isfinite(result) && result > Real{0})) {
    throw std::invalid_argument{std::string{label} + " is not representable"};
  }
  return result;
}

void throw_on_sample_status(int status) {
  if ((status & kPicSampleNonFinitePosition) != 0) {
    throw std::invalid_argument{
        "quiet-start bounds produce non-finite coordinates"};
  }
  if ((status & kPicSampleNonFiniteVelocity) != 0) {
    throw std::invalid_argument{
        "thermal speed, drift and perturbation produce non-finite velocities"};
  }
  if ((status & kPicSampleOutsideDomain) != 0) {
    throw std::invalid_argument{
        "every initial particle must lie inside the physical domain"};
  }
  if ((status & kPicSampleSuperluminal) != 0) {
    throw std::invalid_argument{
        "sampled |v|/c >= 1, outside the nonrelativistic Boris model; lower "
        "temperature/drift or use a relativistic pusher"};
  }
}

}  // namespace

std::uint64_t quiet_start_stride(std::uint64_t count) {
  if (count == 0) return 1;
  // (sqrt(5) - 1) / 2. One scalar, so the extra precision of a host evaluation
  // costs nothing and every particle sees the same integer answer.
  const Real golden = (std::sqrt(Real{5}) - Real{1}) / Real{2};
  std::uint64_t stride = static_cast<std::uint64_t>(
      std::llround(golden * static_cast<Real>(count)));
  if (stride < 1) stride = 1;
  while (std::gcd(stride, count) != 1) ++stride;
  return stride;
}

Real quiet_block_measure(const ParticleSampleConfig& config) {
  require(config.count > 0, "quiet-start requires a positive particle count");
  validate_extent(config.x_min, config.x_max, config.count, "x");
  validate_extent(config.y_min, config.y_max, config.count, "y");
  if (!config.cylindrical) {
    const Real factors[2] = {config.x_max - config.x_min,
                             config.y_max - config.y_min};
    return scaled_product_over(factors, 2, config.count,
                               "quiet-start area per particle");
  }
  require(config.x_min >= Real{0},
          "cylindrical quiet-start radius must be non-negative");
  // pi * (r_max^2 - r_min^2) * (z_max - z_min), with the difference of squares
  // factored so it neither cancels catastrophically for a thin annulus nor
  // overflows in r_max + r_min at a large radius.
  const Real radial_difference = config.x_max - config.x_min;
  const Real radial_sum_factor = Real{1} + config.x_min / config.x_max;
  const Real factors[5] = {Real{3.14159265358979323846},
                           radial_difference,
                           config.x_max,
                           radial_sum_factor,
                           config.y_max - config.y_min};
  return scaled_product_over(factors, 5, config.count,
                             "quiet-start ring volume");
}

void sample_species(ParticleSpecies& species,
                    const ParticleSampleConfig& config,
                    backend::stream_t stream) {
  require(config.count > 0, "quiet-start requires a positive particle count");
  require(config.count <= species.capacity(),
          "quiet-start particle count exceeds the species capacity");
  require(std::isfinite(config.thermal_speed) &&
              config.thermal_speed >= Real{0},
          "thermal speed must be finite and non-negative");
  require(std::isfinite(config.drift_x) && std::isfinite(config.drift_y) &&
              std::isfinite(config.drift_z),
          "drift must contain exactly three finite components");
  require(std::isfinite(config.weight) && config.weight >= Real{0},
          "macro weight must be finite and non-negative");
  validate_extent(config.x_min, config.x_max, config.count, "x");
  validate_extent(config.y_min, config.y_max, config.count, "y");
  if (config.cylindrical) {
    require(config.x_min >= Real{0},
            "cylindrical quiet-start radius must be non-negative");
  }

  const auto count = static_cast<std::uint64_t>(config.count);
  backend::DeviceBuffer<int> status(1);

  PicQuietStartSpec positions{};
  positions.count = count;
  positions.stride = quiet_start_stride(count);
  positions.x_min = config.x_min;
  positions.x_max = config.x_max;
  positions.y_min = config.y_min;
  positions.y_max = config.y_max;
  positions.cylindrical = config.cylindrical ? 1 : 0;
  positions.domain_x_min = config.domain_origin_x;
  positions.domain_x_max = config.domain_origin_x + config.domain_lx;
  positions.domain_y_min = config.domain_origin_y;
  positions.domain_y_max = config.domain_origin_y + config.domain_ly;
  launch_pic_quiet_positions(positions, species.x(), species.y(),
                             status.device_ptr(), stream);

  PicMaxwellianSpec velocities{};
  velocities.thermal_speed = config.thermal_speed;
  velocities.drift_x = config.drift_x;
  velocities.drift_y = config.drift_y;
  velocities.drift_z = config.drift_z;
  velocities.seed = config.seed;
  velocities.species_key = config.species_key;
  launch_pic_maxwellian_velocities(count, velocities, species.vx(),
                                   species.vy(), species.vz(),
                                   status.device_ptr(), stream);

  if (config.perturb) {
    PicVelocityPerturbationSpec perturbation{};
    perturbation.active = 1;
    perturbation.mode_x = config.mode_x;
    perturbation.mode_y = config.mode_y;
    perturbation.phase = config.phase;
    perturbation.amplitude_x = config.amplitude_x;
    perturbation.amplitude_y = config.amplitude_y;
    perturbation.amplitude_z = config.amplitude_z;
    perturbation.origin_x = config.domain_origin_x;
    perturbation.origin_y = config.domain_origin_y;
    perturbation.lx = config.domain_lx;
    perturbation.ly = config.domain_ly;
    launch_pic_velocity_perturbation(count, perturbation, species.x(),
                                     species.y(), species.vx(), species.vy(),
                                     species.vz(), status.device_ptr(),
                                     stream);
  }

  launch_pic_check_subluminal(count, species.vx(), species.vy(), species.vz(),
                              status.device_ptr(), stream);
  launch_pic_finalize_seed(count, config.weight, species.x(), species.y(),
                           species.vz(), species.x_prev(), species.y_prev(),
                           species.vphi_deposit(), species.weight(),
                           species.alive(), species.id(), stream);

  int host_status = 0;
  status.copy_to_host(&host_status, 1);
  backend::device_synchronize(stream);
  // The species is only published once the sample is known good: a rejected
  // configuration must not leave a partially seeded population behind.
  throw_on_sample_status(host_status);
  species.set_count(config.count);
}

}  // namespace quasar::pic
