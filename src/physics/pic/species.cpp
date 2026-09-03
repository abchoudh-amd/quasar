#include "quasar/physics/pic/species.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
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

void validate_snapshot_layout(const ParticleSpecies::HostSnapshot& snapshot,
                              std::string_view operation) {
  const std::size_t n = snapshot.x.size();
  if (snapshot.y.size() != n
      || snapshot.x_prev.size() != n || snapshot.y_prev.size() != n
      || snapshot.vx.size() != n || snapshot.vy.size() != n
      || snapshot.vz.size() != n || snapshot.vphi_deposit.size() != n
      || snapshot.weight.size() != n || snapshot.alive.size() != n
      || snapshot.id.size() != n) {
    throw std::invalid_argument{
        std::string{operation} + ": complete particle-state size mismatch"};
  }
}

// Admissibility of a snapshot that arrived on the HOST -- from a caller's
// vectors, an HDF5 checkpoint, or a migration gather.
//
// This deliberately stays on the host, unlike the sampled path, whose identical
// |v| < c test runs as subluminal_kernel. The two are not gratuitous
// duplication: they guard the same predicate at two places where the data
// genuinely lives differently. A sampled population is born on the device, so
// checking it there avoids a download. A restored population is born on the
// host, and checking it there buys the property the callers below depend on --
// every value is screened BEFORE the first copy_from_host, so a rejected
// snapshot leaves the existing device state untouched.
//
// Moving this check after the upload would not remove a transfer (the H2D
// happens either way), would add a kernel launch and a synchronize, and would
// cost that transactional guarantee on the two paths that need it most: the
// distributed seed and the checkpoint restore. Staging into a scratch device
// buffer to keep the guarantee would double the residency of exactly the
// population size that motivates worrying about this in the first place.
//
// What the two DO share is a formula that must not drift, hence the nested
// hypot in both and the note in subluminal_kernel naming this function.
void validate_snapshot(const ParticleSpecies::HostSnapshot& snapshot,
                       std::string_view operation) {
  validate_snapshot_layout(snapshot, operation);
  const std::size_t n = snapshot.x.size();

  std::unordered_set<std::uint64_t> ids;
  ids.reserve(n);
  for (std::size_t p = 0; p < n; ++p) {
    if (!(std::isfinite(snapshot.x[p]) && std::isfinite(snapshot.y[p])
          && std::isfinite(snapshot.x_prev[p])
          && std::isfinite(snapshot.y_prev[p])
          && std::isfinite(snapshot.vx[p]) && std::isfinite(snapshot.vy[p])
          && std::isfinite(snapshot.vz[p])
          && std::isfinite(snapshot.vphi_deposit[p])
          && std::isfinite(snapshot.weight[p]))) {
      throw std::invalid_argument{
          std::string{operation} + ": all particle values must be finite"};
    }
    if (snapshot.weight[p] < Real{0}) {
      throw std::invalid_argument{
          std::string{operation} + ": weights must be non-negative"};
    }
    const Real speed = std::hypot(
        std::hypot(snapshot.vx[p], snapshot.vy[p]), snapshot.vz[p]);
    if (speed >= Real{1}) {
      throw std::invalid_argument{
          std::string{operation}
          + ": |v| must be < c=1 for the nonrelativistic Boris pusher"};
    }
    if (snapshot.alive[p] > 1) {
      throw std::invalid_argument{
          std::string{operation} + ": alive flags must be zero or one"};
    }
    if (!ids.insert(snapshot.id[p]).second) {
      throw std::invalid_argument{
          std::string{operation} + ": particle IDs must be unique"};
    }
  }
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
    id_{capacity_},
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
    c_id_{capacity_},
    c_counter_{2},
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
  std::vector<std::uint64_t> id(x.size());
  std::iota(id.begin(), id.end(), std::uint64_t{0});
  set_host_particles(x, y, vx, vy, vz, weight, id);
}

void ParticleSpecies::set_host_particles(
    const std::vector<Real>& x, const std::vector<Real>& y,
    const std::vector<Real>& vx, const std::vector<Real>& vy,
    const std::vector<Real>& vz, const std::vector<Real>& weight,
    const std::vector<std::uint64_t>& id) {
  const std::size_t n = x.size();
  if (n > capacity_) {
    throw std::out_of_range{"ParticleSpecies::set_host_particles exceeds capacity"};
  }
  HostSnapshot snapshot;
  snapshot.x = x;
  snapshot.y = y;
  snapshot.x_prev = x;
  snapshot.y_prev = y;
  snapshot.vx = vx;
  snapshot.vy = vy;
  snapshot.vz = vz;
  snapshot.vphi_deposit = vz;
  snapshot.weight = weight;
  snapshot.alive.assign(n, 1);
  snapshot.id = id;
  validate_snapshot(snapshot, "ParticleSpecies::set_host_particles");

  x_.copy_from_host(snapshot.x.data(), n);
  y_.copy_from_host(snapshot.y.data(), n);
  // Seed previous positions to the initial positions so the first deposit sees
  // zero displacement (no spurious startup current).
  x_prev_.copy_from_host(snapshot.x_prev.data(), n);
  y_prev_.copy_from_host(snapshot.y_prev.data(), n);
  vx_.copy_from_host(snapshot.vx.data(), n);
  vy_.copy_from_host(snapshot.vy.data(), n);
  vz_.copy_from_host(snapshot.vz.data(), n);
  vphi_deposit_.copy_from_host(snapshot.vphi_deposit.data(), n);
  weight_.copy_from_host(snapshot.weight.data(), n);
  alive_.copy_from_host(snapshot.alive.data(), n);
  id_.copy_from_host(snapshot.id.data(), n);
  count_ = n;
}

ParticleSpecies::HostSnapshot ParticleSpecies::to_host() const {
  HostSnapshot s;
  s.x.resize(count_);
  s.y.resize(count_);
  s.x_prev.resize(count_);
  s.y_prev.resize(count_);
  s.vx.resize(count_);
  s.vy.resize(count_);
  s.vz.resize(count_);
  s.vphi_deposit.resize(count_);
  s.weight.resize(count_);
  s.alive.resize(count_);
  s.id.resize(count_);
  if (count_ == 0) return s;
  x_.copy_to_host(s.x.data(), count_);
  y_.copy_to_host(s.y.data(), count_);
  x_prev_.copy_to_host(s.x_prev.data(), count_);
  y_prev_.copy_to_host(s.y_prev.data(), count_);
  vx_.copy_to_host(s.vx.data(), count_);
  vy_.copy_to_host(s.vy.data(), count_);
  vz_.copy_to_host(s.vz.data(), count_);
  vphi_deposit_.copy_to_host(s.vphi_deposit.data(), count_);
  weight_.copy_to_host(s.weight.data(), count_);
  alive_.copy_to_host(s.alive.data(), count_);
  id_.copy_to_host(s.id.data(), count_);
  return s;
}

void ParticleSpecies::reserve(std::size_t new_capacity) {
  if (new_capacity <= capacity_) return;
  if (new_capacity > std::numeric_limits<unsigned int>::max()) {
    throw std::invalid_argument{
        "ParticleSpecies::reserve exceeds the unsigned device-counter range"};
  }

  int owner = c_counter_.owner_device();
  if (owner < 0) owner = backend::current_device();
  const auto device = backend::on_device(owner);

  backend::DeviceBuffer<Real> new_x{new_capacity, device};
  backend::DeviceBuffer<Real> new_y{new_capacity, device};
  backend::DeviceBuffer<Real> new_x_prev{new_capacity, device};
  backend::DeviceBuffer<Real> new_y_prev{new_capacity, device};
  backend::DeviceBuffer<Real> new_vx{new_capacity, device};
  backend::DeviceBuffer<Real> new_vy{new_capacity, device};
  backend::DeviceBuffer<Real> new_vz{new_capacity, device};
  backend::DeviceBuffer<Real> new_vphi_deposit{new_capacity, device};
  backend::DeviceBuffer<Real> new_weight{new_capacity, device};
  backend::DeviceBuffer<std::uint8_t> new_alive{new_capacity, device};
  backend::DeviceBuffer<std::uint64_t> new_id{new_capacity, device};

  backend::DeviceBuffer<Real> new_c_x{new_capacity, device};
  backend::DeviceBuffer<Real> new_c_y{new_capacity, device};
  backend::DeviceBuffer<Real> new_c_x_prev{new_capacity, device};
  backend::DeviceBuffer<Real> new_c_y_prev{new_capacity, device};
  backend::DeviceBuffer<Real> new_c_vx{new_capacity, device};
  backend::DeviceBuffer<Real> new_c_vy{new_capacity, device};
  backend::DeviceBuffer<Real> new_c_vz{new_capacity, device};
  backend::DeviceBuffer<Real> new_c_vphi_deposit{new_capacity, device};
  backend::DeviceBuffer<Real> new_c_weight{new_capacity, device};
  backend::DeviceBuffer<std::uint8_t> new_c_alive{new_capacity, device};
  backend::DeviceBuffer<std::uint64_t> new_c_id{new_capacity, device};

  const std::size_t n = count_;
  if (n != 0) {
    backend::DeviceGuard guard{owner};
    const auto copy = [owner](void* destination, const void* source,
                              std::size_t bytes) {
      backend::device_memcpy_peer_async(destination, owner, source, owner,
                                        bytes, nullptr);
    };
    const std::size_t real_bytes = n * sizeof(Real);
    copy(new_x.device_ptr(), x(), real_bytes);
    copy(new_y.device_ptr(), y(), real_bytes);
    copy(new_x_prev.device_ptr(), x_prev(), real_bytes);
    copy(new_y_prev.device_ptr(), y_prev(), real_bytes);
    copy(new_vx.device_ptr(), vx(), real_bytes);
    copy(new_vy.device_ptr(), vy(), real_bytes);
    copy(new_vz.device_ptr(), vz(), real_bytes);
    copy(new_vphi_deposit.device_ptr(), vphi_deposit(), real_bytes);
    copy(new_weight.device_ptr(), weight(), real_bytes);
    copy(new_alive.device_ptr(), alive(), n * sizeof(std::uint8_t));
    copy(new_id.device_ptr(), id(), n * sizeof(std::uint64_t));
    backend::device_synchronize(nullptr);
  }

  // Every potentially throwing allocation/copy completed. Move-only buffer
  // replacement is noexcept, so the active state changes atomically here.
  x_ = std::move(new_x);
  y_ = std::move(new_y);
  x_prev_ = std::move(new_x_prev);
  y_prev_ = std::move(new_y_prev);
  vx_ = std::move(new_vx);
  vy_ = std::move(new_vy);
  vz_ = std::move(new_vz);
  vphi_deposit_ = std::move(new_vphi_deposit);
  weight_ = std::move(new_weight);
  alive_ = std::move(new_alive);
  id_ = std::move(new_id);
  c_x_ = std::move(new_c_x);
  c_y_ = std::move(new_c_y);
  c_x_prev_ = std::move(new_c_x_prev);
  c_y_prev_ = std::move(new_c_y_prev);
  c_vx_ = std::move(new_c_vx);
  c_vy_ = std::move(new_c_vy);
  c_vz_ = std::move(new_c_vz);
  c_vphi_deposit_ = std::move(new_c_vphi_deposit);
  c_weight_ = std::move(new_c_weight);
  c_alive_ = std::move(new_c_alive);
  c_id_ = std::move(new_c_id);
  capacity_ = new_capacity;
}

void ParticleSpecies::replace_host_particles(const HostSnapshot& particles) {
  validate_snapshot(particles, "ParticleSpecies::replace_host_particles");
  replace_migrated_particles(particles, nullptr);
  backend::device_synchronize(nullptr);
}

void ParticleSpecies::replace_migrated_particles(
    const HostSnapshot& particles, backend::stream_t stream) {
  validate_snapshot_layout(
      particles, "ParticleSpecies::replace_migrated_particles");
  const std::size_t incoming = particles.x.size();
  if (incoming > std::numeric_limits<unsigned int>::max()) {
    throw std::length_error{
        "ParticleSpecies::replace_migrated_particles exceeds the "
        "device-counter range"};
  }
  if (incoming > capacity_) reserve(incoming);
  if (incoming == 0) {
    count_ = 0;
    return;
  }

  int owner = id_.owner_device();
  if (owner < 0) owner = c_counter_.owner_device();
  backend::DeviceGuard guard{owner};
  const auto copy_real = [incoming, stream](
                             Real* destination,
                             const std::vector<Real>& source) {
    backend::device_memcpy_h2d_async(destination, source.data(),
                                     incoming * sizeof(Real), stream);
  };
  copy_real(x(), particles.x);
  copy_real(y(), particles.y);
  copy_real(x_prev(), particles.x_prev);
  copy_real(y_prev(), particles.y_prev);
  copy_real(vx(), particles.vx);
  copy_real(vy(), particles.vy);
  copy_real(vz(), particles.vz);
  copy_real(vphi_deposit(), particles.vphi_deposit);
  copy_real(weight(), particles.weight);
  backend::device_memcpy_h2d_async(alive(), particles.alive.data(),
                                   incoming * sizeof(std::uint8_t), stream);
  backend::device_memcpy_h2d_async(id(), particles.id.data(),
                                   incoming * sizeof(std::uint64_t), stream);
  count_ = incoming;
}

void ParticleSpecies::append_host_particles(const HostSnapshot& particles) {
  validate_snapshot(particles, "ParticleSpecies::append_host_particles");
  const std::size_t incoming = particles.x.size();
  if (incoming == 0) return;
  if (incoming > std::numeric_limits<unsigned int>::max() - count_) {
    throw std::length_error{
        "ParticleSpecies::append_host_particles exceeds the device-counter range"};
  }

  std::vector<std::uint64_t> existing_ids(count_);
  id_.copy_to_host(existing_ids.data(), count_);
  std::unordered_set<std::uint64_t> all_ids(
      existing_ids.begin(), existing_ids.end());
  for (const std::uint64_t id : particles.id) {
    if (!all_ids.insert(id).second) {
      throw std::invalid_argument{
          "ParticleSpecies::append_host_particles: particle ID already exists"};
    }
  }

  append_migrated_particles(particles, nullptr);
  backend::device_synchronize(nullptr);
}

void ParticleSpecies::append_migrated_particles(
    const HostSnapshot& particles, backend::stream_t stream) {
  validate_snapshot_layout(
      particles, "ParticleSpecies::append_migrated_particles");
  const std::size_t incoming = particles.x.size();
  if (incoming == 0) return;
  if (incoming > std::numeric_limits<unsigned int>::max() - count_) {
    throw std::length_error{
        "ParticleSpecies::append_migrated_particles exceeds the "
        "device-counter range"};
  }
  const std::size_t required = count_ + incoming;
  if (required > capacity_) {
    const std::size_t maximum = std::numeric_limits<unsigned int>::max();
    const std::size_t increment = std::max<std::size_t>(capacity_ / 2, 1);
    const std::size_t grown = capacity_ > maximum - increment
                            ? maximum : capacity_ + increment;
    reserve(std::max(required, grown));
  }

  const std::size_t offset = count_;
  int owner = id_.owner_device();
  if (owner < 0) owner = c_counter_.owner_device();
  backend::DeviceGuard guard{owner};
  const auto copy_real = [offset, incoming, stream](
                             Real* destination,
                             const std::vector<Real>& source) {
    backend::device_memcpy_h2d_async(destination + offset, source.data(),
                                     incoming * sizeof(Real), stream);
  };
  copy_real(x(), particles.x);
  copy_real(y(), particles.y);
  copy_real(x_prev(), particles.x_prev);
  copy_real(y_prev(), particles.y_prev);
  copy_real(vx(), particles.vx);
  copy_real(vy(), particles.vy);
  copy_real(vz(), particles.vz);
  copy_real(vphi_deposit(), particles.vphi_deposit);
  copy_real(weight(), particles.weight);
  backend::device_memcpy_h2d_async(alive() + offset, particles.alive.data(),
                                   incoming * sizeof(std::uint8_t), stream);
  backend::device_memcpy_h2d_async(id() + offset, particles.id.data(),
                                   incoming * sizeof(std::uint64_t), stream);
  count_ = required;
}

}  // namespace quasar::pic
