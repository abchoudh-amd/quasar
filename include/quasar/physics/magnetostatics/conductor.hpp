#pragma once

#include "quasar/core/field_source.hpp"
#include "quasar/core/types.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace quasar::magnetostatics {

// A single filamentary conductor: an open polyline with a constant current.
struct Filament {
  std::string       name;
  Real              current_A{0};
  std::vector<Vec3> points;
};

// Structure-of-arrays representation of every straight segment in a
// ConductorSystem, laid out for coalesced device loads. One entry per segment.
struct SegmentSoA {
  std::vector<Real> ax, ay, az;  // segment start  (a)
  std::vector<Real> bx, by, bz;  // segment end    (b)
  std::vector<Real> I;           // current per segment

  std::size_t n_segments() const noexcept { return ax.size(); }
};

class ConductorSystem : public core::IFieldSource {
 public:
  void add(Filament f);

  std::size_t                  size()  const noexcept { return filaments_.size(); }
  bool                         empty() const noexcept { return filaments_.empty(); }
  const Filament&              operator[](std::size_t i) const { return filaments_[i]; }
  const std::vector<Filament>& filaments() const noexcept { return filaments_; }

  // Flattens every consecutive-vertex segment in every filament into the
  // per-segment SoA. Throws std::invalid_argument if any filament has fewer
  // than two points, contains a non-finite coordinate, or contains a segment
  // of length below kEps.
  SegmentSoA to_segments_soa() const;

  // Cached view of to_segments_soa(): the geometry is invariant between add()
  // calls, so repeated evaluations (e.g. evaluate_B then evaluate_grad_B, or an
  // observation-set sweep) reuse one flatten+validate instead of redoing it each
  // call. Invalidated by add(). Same throw conditions as to_segments_soa().
  const SegmentSoA& segments_soa() const;

 private:
  std::vector<Filament> filaments_{};
  mutable SegmentSoA    soa_cache_{};
  mutable bool          soa_cache_valid_{false};
};

}  // namespace quasar::magnetostatics
