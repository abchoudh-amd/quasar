#include "quasar/physics/magnetostatics/conductor.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/magnetostatics/kernels.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace quasar::magnetostatics {

namespace {

bool is_finite(Vec3 v) noexcept {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

[[noreturn]] void throw_bad_filament(const std::string& name, const std::string& why) {
  throw std::invalid_argument{"quasar::magnetostatics::ConductorSystem: filament '"
                              + name + "' " + why};
}

int checked_count(std::size_t n, const char* what) {
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error{std::string{"quasar::magnetostatics: "} + what
                            + " exceeds the signed kernel-index limit"};
  }
  return static_cast<int>(n);
}

}  // namespace

// -- FilamentPoints ---------------------------------------------------------

FilamentPoints::FilamentPoints(std::size_t n)
    : px_(n, backend::uninitialized), py_(n, backend::uninitialized),
      pz_(n, backend::uninitialized), n_points_{n} {}

FilamentPoints FilamentPoints::upload(const std::vector<Vec3>& points) {
  FilamentPoints out(points.size());
  const std::size_t n = points.size();
  if (n == 0) return out;

  std::vector<Real> plane(n);
  for (std::size_t i = 0; i < n; ++i) plane[i] = points[i].x;
  out.px_.copy_from_host(plane.data(), n);
  for (std::size_t i = 0; i < n; ++i) plane[i] = points[i].y;
  out.py_.copy_from_host(plane.data(), n);
  for (std::size_t i = 0; i < n; ++i) plane[i] = points[i].z;
  out.pz_.copy_from_host(plane.data(), n);
  return out;
}

FilamentPoints FilamentPoints::clone() const {
  FilamentPoints out(n_points_);
  if (n_points_ == 0) return out;
  const std::size_t bytes = n_points_ * sizeof(Real);
  backend::device_memcpy_d2d_async(out.px_.device_ptr(), px_.device_ptr(),
                                   bytes, nullptr);
  backend::device_memcpy_d2d_async(out.py_.device_ptr(), py_.device_ptr(),
                                   bytes, nullptr);
  backend::device_memcpy_d2d_async(out.pz_.device_ptr(), pz_.device_ptr(),
                                   bytes, nullptr);
  backend::device_synchronize(nullptr);
  return out;
}

std::vector<Vec3> FilamentPoints::to_host() const {
  std::vector<Vec3> out(n_points_);
  if (n_points_ == 0) return out;
  std::vector<Real> plane(n_points_);
  px_.copy_to_host(plane.data(), n_points_);
  for (std::size_t i = 0; i < n_points_; ++i) out[i].x = plane[i];
  py_.copy_to_host(plane.data(), n_points_);
  for (std::size_t i = 0; i < n_points_; ++i) out[i].y = plane[i];
  pz_.copy_to_host(plane.data(), n_points_);
  for (std::size_t i = 0; i < n_points_; ++i) out[i].z = plane[i];
  return out;
}

void FilamentPoints::copy_vertex(std::size_t from, std::size_t to) {
  if (from >= n_points_ || to >= n_points_) {
    throw std::out_of_range{
        "quasar::magnetostatics::FilamentPoints: vertex index out of range"};
  }
  const std::size_t bytes = sizeof(Real);
  backend::device_memcpy_d2d_async(px_.device_ptr() + to,
                                   px_.device_ptr() + from, bytes, nullptr);
  backend::device_memcpy_d2d_async(py_.device_ptr() + to,
                                   py_.device_ptr() + from, bytes, nullptr);
  backend::device_memcpy_d2d_async(pz_.device_ptr() + to,
                                   pz_.device_ptr() + from, bytes, nullptr);
  backend::device_synchronize(nullptr);
}

// -- ConductorSystem special members ----------------------------------------

namespace {

std::vector<Filament> clone_filaments(const std::vector<Filament>& source) {
  std::vector<Filament> out;
  out.reserve(source.size());
  for (const Filament& f : source) {
    out.push_back(Filament{f.name, f.current_A, f.points.clone()});
  }
  return out;
}

}  // namespace

ConductorSystem::ConductorSystem(const ConductorSystem& other) {
  const std::scoped_lock lock{other.soa_cache_mutex_};
  filaments_ = clone_filaments(other.filaments_);
}

ConductorSystem& ConductorSystem::operator=(const ConductorSystem& other) {
  if (this == &other) return *this;
  const std::scoped_lock lock{soa_cache_mutex_, other.soa_cache_mutex_};
  filaments_ = clone_filaments(other.filaments_);
  soa_cache_ = DeviceSegmentSoA{};
  soa_cache_valid_ = false;
  return *this;
}

ConductorSystem::ConductorSystem(ConductorSystem&& other) noexcept {
  // Moving is itself a mutation and, like add(), requires external exclusion
  // from readers.  Avoid locking here so the class retains its original
  // nothrow-move property for value-return and container use.
  filaments_ = std::move(other.filaments_);
  other.filaments_.clear();
  // A moved-from object remains a valid empty ConductorSystem.  In particular,
  // it must not retain a cache describing geometry it no longer owns.
  other.soa_cache_ = DeviceSegmentSoA{};
  other.soa_cache_valid_ = false;
}

ConductorSystem& ConductorSystem::operator=(ConductorSystem&& other) noexcept {
  if (this == &other) return *this;
  filaments_ = std::move(other.filaments_);
  other.filaments_.clear();
  soa_cache_ = DeviceSegmentSoA{};
  soa_cache_valid_ = false;
  other.soa_cache_ = DeviceSegmentSoA{};
  other.soa_cache_valid_ = false;
  return *this;
}

void SegmentSoA::validate() const {
  const std::size_t n = ax.size();
  if (ay.size() != n || az.size() != n || bx.size() != n || by.size() != n
      || bz.size() != n || I.size() != n) {
    throw std::invalid_argument{
        "quasar::magnetostatics::SegmentSoA: component vectors have "
        "inconsistent lengths"};
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (!(std::isfinite(ax[i]) && std::isfinite(ay[i])
          && std::isfinite(az[i]) && std::isfinite(bx[i])
          && std::isfinite(by[i]) && std::isfinite(bz[i])
          && std::isfinite(I[i]))) {
      throw std::invalid_argument{
          "quasar::magnetostatics::SegmentSoA: payload has a non-finite "
          "component"};
    }
  }
}

SegmentSoA DeviceSegmentSoA::to_host() const {
  SegmentSoA out;
  out.ax.resize(n);
  out.ay.resize(n);
  out.az.resize(n);
  out.bx.resize(n);
  out.by.resize(n);
  out.bz.resize(n);
  out.I.resize(n);
  if (n == 0) return out;
  ax.copy_to_host(out.ax.data(), n);
  ay.copy_to_host(out.ay.data(), n);
  az.copy_to_host(out.az.data(), n);
  bx.copy_to_host(out.bx.data(), n);
  by.copy_to_host(out.by.data(), n);
  bz.copy_to_host(out.bz.data(), n);
  I.copy_to_host(out.I.data(), n);
  return out;
}

void ConductorSystem::add(Filament f) {
  const std::scoped_lock lock{soa_cache_mutex_};
  filaments_.push_back(std::move(f));
  soa_cache_valid_ = false;  // geometry changed; drop the stale flattened SoA
}

const DeviceSegmentSoA& ConductorSystem::device_segments() const {
  const std::scoped_lock lock{soa_cache_mutex_};
  if (soa_cache_valid_) return soa_cache_;

  // Segment counting is integer arithmetic over a handful of filaments, so it
  // stays on the host; only the per-segment work is a kernel.
  std::size_t total_segments = 0;
  for (const auto& fil : filaments_) {
    if (fil.points.size() >= 2) {
      const std::size_t add = fil.points.size() - 1;
      if (add > std::numeric_limits<std::size_t>::max() - total_segments) {
        throw_bad_filament(fil.name, "has too many segments to flatten");
      }
      total_segments += add;
    }
  }
  (void)checked_count(total_segments, "segment count");

  // Construct off to the side so a validation or allocation failure leaves the
  // existing cache object untouched and the validity flag false.
  DeviceSegmentSoA rebuilt;
  rebuilt.n = total_segments;
  rebuilt.ax = backend::DeviceBuffer<Real>(total_segments, backend::uninitialized);
  rebuilt.ay = backend::DeviceBuffer<Real>(total_segments, backend::uninitialized);
  rebuilt.az = backend::DeviceBuffer<Real>(total_segments, backend::uninitialized);
  rebuilt.bx = backend::DeviceBuffer<Real>(total_segments, backend::uninitialized);
  rebuilt.by = backend::DeviceBuffer<Real>(total_segments, backend::uninitialized);
  rebuilt.bz = backend::DeviceBuffer<Real>(total_segments, backend::uninitialized);
  rebuilt.I  = backend::DeviceBuffer<Real>(total_segments, backend::uninitialized);

  // One launch and one status word per filament, so a rejection can still name
  // the filament the way the host loop did. The segment index within it is not
  // recoverable from an OR-reduced predicate; the message drops it.
  long long offset = 0;
  for (const auto& fil : filaments_) {
    if (fil.points.size() < 2) {
      throw_bad_filament(fil.name, "needs at least 2 points");
    }
    if (!std::isfinite(fil.current_A)) {
      throw_bad_filament(fil.name, "has non-finite current_A");
    }
    backend::DeviceBuffer<int> status(1);
    ::launch_ms_flatten_filament(
        fil.points.x(), fil.points.y(), fil.points.z(),
        checked_count(fil.points.size(), "filament vertex count"),
        fil.current_A, offset,
        rebuilt.ax.device_ptr(), rebuilt.ay.device_ptr(),
        rebuilt.az.device_ptr(), rebuilt.bx.device_ptr(),
        rebuilt.by.device_ptr(), rebuilt.bz.device_ptr(),
        rebuilt.I.device_ptr(), status.device_ptr(), nullptr);
    int flags = 0;
    status.copy_to_host(&flags, 1);
    if ((flags & 1) != 0) {
      throw_bad_filament(fil.name, "contains a non-finite coordinate");
    }
    if ((flags & 2) != 0) {
      throw_bad_filament(fil.name,
                         "contains endpoints whose displacement overflows");
    }
    if ((flags & 4) != 0) {
      throw_bad_filament(fil.name, "contains a zero-length segment");
    }
    offset += static_cast<long long>(fil.points.size()) - 1;
  }

  soa_cache_ = std::move(rebuilt);
  soa_cache_valid_ = true;
  return soa_cache_;
}

SegmentSoA ConductorSystem::to_segments_soa() const {
  return device_segments().to_host();
}

}  // namespace quasar::magnetostatics
