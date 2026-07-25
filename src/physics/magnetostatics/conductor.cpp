#include "quasar/physics/magnetostatics/conductor.hpp"

#include "quasar/core/types.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace quasar::magnetostatics {

namespace {

bool is_finite(Vec3 v) noexcept {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

[[noreturn]] void throw_bad_filament(const std::string& name, const std::string& why) {
  throw std::invalid_argument{"quasar::magnetostatics::ConductorSystem: filament '"
                              + name + "' " + why};
}

}  // namespace

ConductorSystem::ConductorSystem(const ConductorSystem& other) {
  const std::scoped_lock lock{other.soa_cache_mutex_};
  filaments_ = other.filaments_;
}

ConductorSystem& ConductorSystem::operator=(const ConductorSystem& other) {
  if (this == &other) return *this;
  const std::scoped_lock lock{soa_cache_mutex_, other.soa_cache_mutex_};
  filaments_ = other.filaments_;
  soa_cache_ = SegmentSoA{};
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
  other.soa_cache_ = SegmentSoA{};
  other.soa_cache_valid_ = false;
}

ConductorSystem& ConductorSystem::operator=(ConductorSystem&& other) noexcept {
  if (this == &other) return *this;
  filaments_ = std::move(other.filaments_);
  other.filaments_.clear();
  soa_cache_ = SegmentSoA{};
  soa_cache_valid_ = false;
  other.soa_cache_ = SegmentSoA{};
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

void ConductorSystem::add(Filament f) {
  const std::scoped_lock lock{soa_cache_mutex_};
  filaments_.push_back(std::move(f));
  soa_cache_valid_ = false;  // geometry changed; drop the stale flattened SoA
}

const SegmentSoA& ConductorSystem::segments_soa() const {
  const std::scoped_lock lock{soa_cache_mutex_};
  if (!soa_cache_valid_) {
    // Construct off to the side so a validation/allocation exception leaves
    // the existing cache object untouched and the validity flag false.
    SegmentSoA rebuilt = to_segments_soa();
    soa_cache_ = std::move(rebuilt);
    soa_cache_valid_ = true;
  }
  return soa_cache_;
}

SegmentSoA ConductorSystem::to_segments_soa() const {
  SegmentSoA soa;

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
  soa.ax.reserve(total_segments);
  soa.ay.reserve(total_segments);
  soa.az.reserve(total_segments);
  soa.bx.reserve(total_segments);
  soa.by.reserve(total_segments);
  soa.bz.reserve(total_segments);
  soa.I .reserve(total_segments);

  for (const auto& fil : filaments_) {
    if (fil.points.size() < 2) {
      throw_bad_filament(fil.name, "needs at least 2 points");
    }
    if (!std::isfinite(fil.current_A)) {
      throw_bad_filament(fil.name, "has non-finite current_A");
    }
    for (std::size_t i = 0; i + 1 < fil.points.size(); ++i) {
      const Vec3 a = fil.points[i];
      const Vec3 b = fil.points[i + 1];
      if (!is_finite(a) || !is_finite(b)) {
        throw_bad_filament(fil.name, "contains a non-finite coordinate");
      }
      const Vec3 d = b - a;
      if (!is_finite(d)) {
        throw_bad_filament(fil.name,
                           "contains endpoints whose displacement overflows");
      }
      if (d.x == Real{0} && d.y == Real{0} && d.z == Real{0}) {
        throw_bad_filament(fil.name,
                           "contains a zero-length segment at index "
                           + std::to_string(i));
      }
      soa.ax.push_back(a.x);
      soa.ay.push_back(a.y);
      soa.az.push_back(a.z);
      soa.bx.push_back(b.x);
      soa.by.push_back(b.y);
      soa.bz.push_back(b.z);
      soa.I .push_back(fil.current_A);
    }
  }

  return soa;
}

}  // namespace quasar::magnetostatics
