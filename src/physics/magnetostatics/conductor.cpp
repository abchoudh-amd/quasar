#include "quasar/physics/magnetostatics/conductor.hpp"

#include "quasar/core/types.hpp"

#include <cmath>
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

void ConductorSystem::add(Filament f) {
  filaments_.push_back(std::move(f));
}

SegmentSoA ConductorSystem::to_segments_soa() const {
  SegmentSoA soa;

  std::size_t total_segments = 0;
  for (const auto& fil : filaments_) {
    if (fil.points.size() >= 2) {
      total_segments += fil.points.size() - 1;
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
      if (length_squared(d) < kEps) {
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
