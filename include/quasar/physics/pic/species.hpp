#pragma once

#include "quasar/backend/memory.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace quasar::pic {

struct SpeciesConfig {
  std::string name{"species"};
  Real charge{Real{-1}};
  Real mass{Real{1}};
  std::size_t capacity{0};
};

class ParticleSpecies {
 public:
  ParticleSpecies() = default;
  explicit ParticleSpecies(SpeciesConfig cfg);

  const std::string& name() const noexcept { return name_; }
  Real charge() const noexcept { return charge_; }
  Real mass() const noexcept { return mass_; }
  Real charge_to_mass() const noexcept { return charge_ / mass_; }
  std::size_t size() const noexcept { return count_; }
  std::size_t capacity() const noexcept { return capacity_; }

  void set_count(std::size_t n);
  void set_host_particles(const std::vector<Real>& x, const std::vector<Real>& y,
                          const std::vector<Real>& vx, const std::vector<Real>& vy,
                          const std::vector<Real>& vz,
                          const std::vector<Real>& weight);

  Real* x() noexcept { return x_.device_ptr(); }
  Real* y() noexcept { return y_.device_ptr(); }
  Real* vx() noexcept { return vx_.device_ptr(); }
  Real* vy() noexcept { return vy_.device_ptr(); }
  Real* vz() noexcept { return vz_.device_ptr(); }
  Real* weight() noexcept { return weight_.device_ptr(); }
  std::uint8_t* alive() noexcept { return alive_.device_ptr(); }

  const Real* x() const noexcept { return x_.device_ptr(); }
  const Real* y() const noexcept { return y_.device_ptr(); }
  const Real* vx() const noexcept { return vx_.device_ptr(); }
  const Real* vy() const noexcept { return vy_.device_ptr(); }
  const Real* vz() const noexcept { return vz_.device_ptr(); }
  const Real* weight() const noexcept { return weight_.device_ptr(); }
  const std::uint8_t* alive() const noexcept { return alive_.device_ptr(); }

  const Grid2D& grid() const noexcept { return grid_; }
  void set_grid(Grid2D g) noexcept { grid_ = g; }

 private:
  std::string name_{"species"};
  Real charge_{Real{-1}};
  Real mass_{Real{1}};
  std::size_t count_{0};
  std::size_t capacity_{0};
  Grid2D grid_{};
  backend::DeviceBuffer<Real> x_{};
  backend::DeviceBuffer<Real> y_{};
  backend::DeviceBuffer<Real> vx_{};
  backend::DeviceBuffer<Real> vy_{};
  backend::DeviceBuffer<Real> vz_{};
  backend::DeviceBuffer<Real> weight_{};
  backend::DeviceBuffer<std::uint8_t> alive_{};
};

}  // namespace quasar::pic
