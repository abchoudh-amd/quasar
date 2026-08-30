#pragma once

#include "quasar/backend/memory.hpp"
#include "quasar/core/field_source.hpp"
#include "quasar/core/types.hpp"

#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace quasar::magnetostatics {

// A filament's vertices, held on the device as three SoA planes.
//
// Device-resident because that is where they are produced and where they are
// consumed: the generators write them with a kernel, and the flatten that feeds
// Biot-Savart reads them with a kernel. A deck that lists coordinates literally
// uploads them once through upload(); those are data from a file, not the
// result of a calculation.
//
// Move-only, following DeviceBuffer. Filament and ConductorSystem inherit that,
// which is a behaviour change from the previous std::vector<Vec3> storage:
// copying a ConductorSystem now duplicates device allocations, so the copy
// constructor is explicit about doing so.
class FilamentPoints {
 public:
  FilamentPoints() = default;

  // Uninitialized planes for `n` vertices, to be filled by a generator kernel.
  explicit FilamentPoints(std::size_t n);

  FilamentPoints(const FilamentPoints&) = delete;
  FilamentPoints& operator=(const FilamentPoints&) = delete;
  FilamentPoints(FilamentPoints&&) noexcept = default;
  FilamentPoints& operator=(FilamentPoints&&) noexcept = default;

  // Validates finiteness on the host (these coordinates come from a deck, so
  // there is nothing to gain by checking them on the device) and uploads.
  static FilamentPoints upload(const std::vector<Vec3>& points);

  // Deep copy of the device planes. Explicit because it allocates.
  [[nodiscard]] FilamentPoints clone() const;

  std::size_t size()  const noexcept { return n_points_; }
  bool        empty() const noexcept { return n_points_ == 0; }

  Real* x() noexcept { return px_.device_ptr(); }
  Real* y() noexcept { return py_.device_ptr(); }
  Real* z() noexcept { return pz_.device_ptr(); }
  const Real* x() const noexcept { return px_.device_ptr(); }
  const Real* y() const noexcept { return py_.device_ptr(); }
  const Real* z() const noexcept { return pz_.device_ptr(); }

  // Output boundary only: a deck echo, a Python binding, a test assertion.
  std::vector<Vec3> to_host() const;

  // Copies vertex `from` onto vertex `to`, device to device. Closing a loop
  // this way is bitwise exact and cannot manufacture a microscopic extra
  // segment out of sin(2*pi) round-off, which recomputing the angle would.
  void copy_vertex(std::size_t from, std::size_t to);

 private:
  backend::DeviceBuffer<Real> px_{}, py_{}, pz_{};
  std::size_t n_points_{0};
};

// A single filamentary conductor: an open polyline with a constant current.
struct Filament {
  std::string    name;
  Real           current_A{0};
  FilamentPoints points;
};

// Structure-of-arrays representation of every straight segment in a
// ConductorSystem, laid out for coalesced device loads. One entry per segment.
// This is the host staging form, produced by downloading the device planes; a
// consumer that runs on the device wants DeviceSegmentSoA below.
struct SegmentSoA {
  std::vector<Real> ax, ay, az;  // segment start  (a)
  std::vector<Real> bx, by, bz;  // segment end    (b)
  std::vector<Real> I;           // current per segment

  std::size_t n_segments() const noexcept { return ax.size(); }
  // Reject mismatched component planes and non-finite payloads before a
  // consumer uses n_segments() to size or upload every plane.
  void validate() const;
};

// Device-resident per-segment planes: what the Biot-Savart kernels read.
struct DeviceSegmentSoA {
  backend::DeviceBuffer<Real> ax, ay, az;
  backend::DeviceBuffer<Real> bx, by, bz;
  backend::DeviceBuffer<Real> I;
  std::size_t n{0};

  std::size_t n_segments() const noexcept { return n; }
  bool        empty()      const noexcept { return n == 0; }

  // Output boundary only.
  SegmentSoA to_host() const;
};

class ConductorSystem : public core::IFieldSource {
 public:
  ConductorSystem() = default;

  // The flattened representation is a derived, invalidatable cache.  Copies
  // receive only the filament geometry and build an independent cache on first
  // use; moves likewise leave both objects with invalid caches.  Explicit
  // special members preserve the value semantics that an inline mutex would
  // otherwise delete.  Copying now duplicates device allocations, since the
  // filament vertices themselves live on the device.
  ConductorSystem(const ConductorSystem& other);
  ConductorSystem& operator=(const ConductorSystem& other);
  ConductorSystem(ConductorSystem&& other) noexcept;
  ConductorSystem& operator=(ConductorSystem&& other) noexcept;

  void add(Filament f);

  std::size_t                  size()  const noexcept { return filaments_.size(); }
  bool                         empty() const noexcept { return filaments_.empty(); }
  const Filament&              operator[](std::size_t i) const { return filaments_[i]; }
  const std::vector<Filament>& filaments() const noexcept { return filaments_; }

  // Cached device flatten of every consecutive-vertex segment in every
  // filament. The geometry is invariant between add() calls, so repeated
  // evaluations (evaluate_B then evaluate_grad_B, or an observation-set sweep)
  // reuse one flatten+validate instead of redoing it. Concurrent const calls
  // are serialized during the first cache fill; mutation still requires
  // external exclusion from readers. Invalidated by add().
  //
  // Throws std::invalid_argument if any filament has fewer than two points,
  // contains a non-finite coordinate, has an endpoint displacement that
  // overflows, or contains a segment whose endpoints coincide exactly.
  const DeviceSegmentSoA& device_segments() const;

  // Host staging view of device_segments(), for a deck echo or a test. Downloads
  // on every call; a device consumer must use device_segments().
  SegmentSoA to_segments_soa() const;

 private:
  std::vector<Filament>    filaments_{};
  mutable std::mutex       soa_cache_mutex_{};
  mutable DeviceSegmentSoA soa_cache_{};
  mutable bool             soa_cache_valid_{false};
};

}  // namespace quasar::magnetostatics
