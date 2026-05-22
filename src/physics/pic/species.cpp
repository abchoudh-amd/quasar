#include "quasar/physics/pic/species.hpp"

#include <algorithm>
#include <stdexcept>

namespace quasar::pic {

ParticleSpecies::ParticleSpecies(SpeciesConfig cfg)
  : name_{std::move(cfg.name)},
    charge_{cfg.charge},
    mass_{cfg.mass},
    capacity_{cfg.capacity},
    x_{capacity_},
    y_{capacity_},
    vx_{capacity_},
    vy_{capacity_},
    vz_{capacity_},
    weight_{capacity_},
    alive_{capacity_} {
  if (mass_ <= Real{0}) {
    throw std::invalid_argument{"ParticleSpecies: mass must be positive"};
  }
}

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
  std::vector<std::uint8_t> alive(n, 1);
  x_.copy_from_host(x.data(), n);
  y_.copy_from_host(y.data(), n);
  vx_.copy_from_host(vx.data(), n);
  vy_.copy_from_host(vy.data(), n);
  vz_.copy_from_host(vz.data(), n);
  weight_.copy_from_host(weight.data(), n);
  alive_.copy_from_host(alive.data(), n);
  count_ = n;
}

}  // namespace quasar::pic
