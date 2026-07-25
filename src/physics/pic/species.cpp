#include "quasar/physics/pic/species.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace quasar::pic {

namespace {

SpeciesConfig validate_species_config(SpeciesConfig cfg) {
  if (cfg.name.empty()
      || std::all_of(cfg.name.begin(), cfg.name.end(), [](unsigned char ch) {
           return std::isspace(ch) != 0;
         })) {
    throw std::invalid_argument{
        "ParticleSpecies: name must contain a non-whitespace character"};
  }
  if (!(std::isfinite(cfg.mass) && cfg.mass > Real{0})) {
    throw std::invalid_argument{
        "ParticleSpecies: mass must be finite and positive"};
  }
  if (!std::isfinite(cfg.charge)) {
    throw std::invalid_argument{"ParticleSpecies: charge must be finite"};
  }
  const Real charge_to_mass = cfg.charge / cfg.mass;
  if (!std::isfinite(charge_to_mass)
      || (cfg.charge != Real{0} && charge_to_mass == Real{0})) {
    throw std::invalid_argument{
        "ParticleSpecies: charge-to-mass ratio is not representable"};
  }
  if (cfg.capacity > std::numeric_limits<unsigned int>::max()) {
    throw std::invalid_argument{
        "ParticleSpecies: capacity exceeds the unsigned device-counter range"};
  }
  return cfg;
}

}  // namespace

ParticleSpecies::ParticleSpecies(SpeciesConfig cfg)
  : ParticleSpecies(validate_species_config(std::move(cfg)),
                    ValidatedConfigTag{}) {}

ParticleSpecies::ParticleSpecies(SpeciesConfig cfg, ValidatedConfigTag)
  : name_{std::move(cfg.name)},
    charge_{cfg.charge},
    mass_{cfg.mass},
    capacity_{cfg.capacity},
    x_{capacity_},
    y_{capacity_},
    x_prev_{capacity_},
    y_prev_{capacity_},
    vx_{capacity_},
    vy_{capacity_},
    vz_{capacity_},
    vphi_deposit_{capacity_},
    weight_{capacity_},
    alive_{capacity_},
    c_x_{capacity_},
    c_y_{capacity_},
    c_x_prev_{capacity_},
    c_y_prev_{capacity_},
    c_vx_{capacity_},
    c_vy_{capacity_},
    c_vz_{capacity_},
    c_vphi_deposit_{capacity_},
    c_weight_{capacity_},
    c_alive_{capacity_},
    c_counter_{1},
    deposit_overflow_{1},
    particle_error_{1} {}

void ParticleSpecies::set_count(std::size_t n) {
  if (n > capacity_) {
    throw std::out_of_range{"ParticleSpecies::set_count exceeds capacity"};
  }
  count_ = n;
}

void ParticleSpecies::set_host_particles(const std::vector<Real>& x,
                                         const std::vector<Real>& y,
                                         const std::vector<Real>& vx,
                                         const std::vector<Real>& vy,
                                         const std::vector<Real>& vz,
                                         const std::vector<Real>& weight) {
  const std::size_t n = x.size();
  if (y.size() != n || vx.size() != n || vy.size() != n || vz.size() != n || weight.size() != n) {
    throw std::invalid_argument{"ParticleSpecies::set_host_particles: size mismatch"};
  }
  if (n > capacity_) {
    throw std::out_of_range{"ParticleSpecies::set_host_particles exceeds capacity"};
  }
  for (std::size_t p = 0; p < n; ++p) {
    if (!(std::isfinite(x[p]) && std::isfinite(y[p]) &&
          std::isfinite(vx[p]) && std::isfinite(vy[p]) &&
          std::isfinite(vz[p]) && std::isfinite(weight[p]))) {
      throw std::invalid_argument{
          "ParticleSpecies::set_host_particles: all values must be finite"};
    }
    if (weight[p] < Real{0}) {
      throw std::invalid_argument{
          "ParticleSpecies::set_host_particles: weights must be non-negative"};
    }
    const Real speed = std::hypot(std::hypot(vx[p], vy[p]), vz[p]);
    if (speed >= Real{1}) {
      throw std::invalid_argument{
          "ParticleSpecies::set_host_particles: |v| must be < c=1 for the "
          "nonrelativistic Boris pusher"};
    }
  }
  std::vector<std::uint8_t> alive(n, 1);
  x_.copy_from_host(x.data(), n);
  y_.copy_from_host(y.data(), n);
  // Seed previous positions to the initial positions so the first deposit sees
  // zero displacement (no spurious startup current).
  x_prev_.copy_from_host(x.data(), n);
  y_prev_.copy_from_host(y.data(), n);
  vx_.copy_from_host(vx.data(), n);
  vy_.copy_from_host(vy.data(), n);
  vz_.copy_from_host(vz.data(), n);
  vphi_deposit_.copy_from_host(vz.data(), n);
  weight_.copy_from_host(weight.data(), n);
  alive_.copy_from_host(alive.data(), n);
  count_ = n;
}

ParticleSpecies::HostSnapshot ParticleSpecies::to_host() const {
  HostSnapshot s;
  s.x.resize(count_);
  s.y.resize(count_);
  s.vx.resize(count_);
  s.vy.resize(count_);
  s.vz.resize(count_);
  s.weight.resize(count_);
  s.alive.resize(count_);
  if (count_ == 0) return s;
  x_.copy_to_host(s.x.data(), count_);
  y_.copy_to_host(s.y.data(), count_);
  vx_.copy_to_host(s.vx.data(), count_);
  vy_.copy_to_host(s.vy.data(), count_);
  vz_.copy_to_host(s.vz.data(), count_);
  weight_.copy_to_host(s.weight.data(), count_);
  alive_.copy_to_host(s.alive.data(), count_);
  return s;
}

}  // namespace quasar::pic
