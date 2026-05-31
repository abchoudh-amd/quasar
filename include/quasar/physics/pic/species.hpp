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

  struct HostSnapshot {
    std::vector<Real> x, y, vx, vy, vz, weight;
    std::vector<std::uint8_t> alive;
  };
  HostSnapshot to_host() const;

  Real* x() noexcept { return x_.device_ptr(); }
  Real* y() noexcept { return y_.device_ptr(); }
  Real* x_prev() noexcept { return x_prev_.device_ptr(); }
  Real* y_prev() noexcept { return y_prev_.device_ptr(); }
  Real* vx() noexcept { return vx_.device_ptr(); }
  Real* vy() noexcept { return vy_.device_ptr(); }
  Real* vz() noexcept { return vz_.device_ptr(); }
  Real* weight() noexcept { return weight_.device_ptr(); }
  std::uint8_t* alive() noexcept { return alive_.device_ptr(); }

  const Real* x() const noexcept { return x_.device_ptr(); }
  const Real* y() const noexcept { return y_.device_ptr(); }
  const Real* x_prev() const noexcept { return x_prev_.device_ptr(); }
  const Real* y_prev() const noexcept { return y_prev_.device_ptr(); }
  const Real* vx() const noexcept { return vx_.device_ptr(); }
  const Real* vy() const noexcept { return vy_.device_ptr(); }
  const Real* vz() const noexcept { return vz_.device_ptr(); }
  const Real* weight() const noexcept { return weight_.device_ptr(); }
  const std::uint8_t* alive() const noexcept { return alive_.device_ptr(); }

  const Grid2D& grid() const noexcept { return grid_; }
  void set_grid(Grid2D g) noexcept { grid_ = g; }

  // Compaction scratch: capacity-sized double buffers + an alive flag buffer and
  // a single counter, allocated once with the species so the periodic compaction
  // op does not reallocate every call. Accessors are non-const (mutated by the
  // backend launch).
  Real* compact_x() noexcept { return c_x_.device_ptr(); }
  Real* compact_y() noexcept { return c_y_.device_ptr(); }
  Real* compact_x_prev() noexcept { return c_x_prev_.device_ptr(); }
  Real* compact_y_prev() noexcept { return c_y_prev_.device_ptr(); }
  Real* compact_vx() noexcept { return c_vx_.device_ptr(); }
  Real* compact_vy() noexcept { return c_vy_.device_ptr(); }
  Real* compact_vz() noexcept { return c_vz_.device_ptr(); }
  Real* compact_weight() noexcept { return c_weight_.device_ptr(); }
  std::uint8_t* compact_alive() noexcept { return c_alive_.device_ptr(); }
  unsigned int* compact_counter() noexcept { return c_counter_.device_ptr(); }
  // const overload: the counter is device scratch (mutable), so a logically-const
  // read-only reduction (alive_count) can reuse it without copying the species.
  unsigned int* compact_counter() const noexcept { return c_counter_.device_ptr(); }

  // Persistent device flag the charge-conserving deposit atomically bumps when a
  // particle's displacement spills outside the fixed deposition window. Kept
  // separate from compact_counter (which the deposit memsets to 0 each call) so
  // the host can read it on a cadence instead of synchronizing every step. The
  // deposit accumulates into it across steps; the solver copies it back
  // periodically and at end-of-run, then clears it. mutable: device scratch read
  // by the logically-const consume path.
  unsigned int* deposit_overflow() const noexcept { return deposit_overflow_.device_ptr(); }

 private:
  std::string name_{"species"};
  Real charge_{Real{-1}};
  Real mass_{Real{1}};
  std::size_t count_{0};
  std::size_t capacity_{0};
  Grid2D grid_{};
  backend::DeviceBuffer<Real> x_{};
  backend::DeviceBuffer<Real> y_{};
  backend::DeviceBuffer<Real> x_prev_{};
  backend::DeviceBuffer<Real> y_prev_{};
  backend::DeviceBuffer<Real> vx_{};
  backend::DeviceBuffer<Real> vy_{};
  backend::DeviceBuffer<Real> vz_{};
  backend::DeviceBuffer<Real> weight_{};
  backend::DeviceBuffer<std::uint8_t> alive_{};
  // Compaction scratch (capacity-sized, allocated alongside the particle arrays).
  backend::DeviceBuffer<Real> c_x_{};
  backend::DeviceBuffer<Real> c_y_{};
  backend::DeviceBuffer<Real> c_x_prev_{};
  backend::DeviceBuffer<Real> c_y_prev_{};
  backend::DeviceBuffer<Real> c_vx_{};
  backend::DeviceBuffer<Real> c_vy_{};
  backend::DeviceBuffer<Real> c_vz_{};
  backend::DeviceBuffer<Real> c_weight_{};
  backend::DeviceBuffer<std::uint8_t> c_alive_{};
  // mutable: device scratch reused by the logically-const alive_count reduction.
  mutable backend::DeviceBuffer<unsigned int> c_counter_{};
  // mutable: persistent deposit-overflow flag, accumulated by the deposit kernel
  // and consumed (copied + cleared) by the solver on a cadence.
  mutable backend::DeviceBuffer<unsigned int> deposit_overflow_{};
};

}  // namespace quasar::pic
