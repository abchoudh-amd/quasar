#pragma once

#include "quasar/core/types.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

// Axis-neutral observation-point sets. These describe *where* a field is
// sampled and carry no physics-specific data, so they live in core and form the
// shared currency of the numerics field-evaluator interface (instead of coupling
// numerics upward to a specific physics module).

namespace quasar::core {

// Structure-of-arrays representation of the observation points.
struct PointSoA {
  std::vector<Real> px, py, pz;

  std::size_t n_points() const noexcept { return px.size(); }
  // Reject mismatched component planes and non-finite coordinates before a
  // consumer uses n_points() to size or upload every plane.
  void validate() const;
};

// Unstructured list of observation points. Other observation-set variants
// (ObservationGrid, PlaneSlice, LineProbe) materialize through this type.
class PointCloud {
 public:
  PointCloud() = default;

  void add(Vec3 p);
  void add(std::span<const Vec3> ps);

  std::size_t              size()   const noexcept { return points_.size(); }
  bool                     empty()  const noexcept { return points_.empty(); }
  const std::vector<Vec3>& points() const noexcept { return points_; }

  PointSoA to_point_soa() const;

 private:
  std::vector<Vec3> points_{};
};

// Regular 3D rectilinear grid of observation points. Cell (i,j,k) lives at
// `origin + (i * spacing.x, j * spacing.y, k * spacing.z)` for
// 0 <= i < dims[0], 0 <= j < dims[1], 0 <= k < dims[2].
// Flat layout is x-fastest: linear index = i + dims[0] * (j + dims[1] * k).
struct ObservationGrid {
  Vec3               origin{};
  Vec3               spacing{Real{1}, Real{1}, Real{1}};
  std::array<int, 3> dims{1, 1, 1};

  std::size_t size() const;
  Vec3       point_at(int i, int j, int k) const;
  PointCloud to_point_cloud() const;
  PointSoA   to_point_soa()   const;

  static void validate(const ObservationGrid& g);
};

// 2D regular slice through space. Cell (i,j) lives at
// `origin + i * u_step + j * v_step` for 0 <= i < nu, 0 <= j < nv.
// Flat layout is u-fastest: linear index = i + nu * j.
struct PlaneSlice {
  Vec3 origin{};
  Vec3 u_step{Real{1}, Real{0}, Real{0}};
  Vec3 v_step{Real{0}, Real{1}, Real{0}};
  int  nu{1};
  int  nv{1};

  std::size_t size() const;
  Vec3       point_at(int i, int j) const;
  PointCloud to_point_cloud() const;
  PointSoA   to_point_soa()   const;

  static void validate(const PlaneSlice& s);
};

// 1D probe with `n_points` samples linearly interpolated from `start` to `end`
// (inclusive at both ends; requires n_points >= 2).
struct LineProbe {
  Vec3 start{};
  Vec3 end{Real{1}, Real{0}, Real{0}};
  int  n_points{2};

  std::size_t size() const;
  Vec3       point_at(int i) const;
  PointCloud to_point_cloud() const;
  PointSoA   to_point_soa()   const;

  static void validate(const LineProbe& l);
};

}  // namespace quasar::core
