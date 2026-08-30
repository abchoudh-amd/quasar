#include "quasar/core/observations.hpp"

#include "quasar/backend/memory.hpp"
#include "quasar/core/device_observations.hpp"
#include "quasar/core/observation_kernels.hpp"
#include "quasar/core/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace quasar::core {

namespace {

bool is_finite(Vec3 v) noexcept {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

[[noreturn]] void throw_bad(const std::string& kind, const std::string& why) {
  throw std::invalid_argument{"quasar::core::" + kind + ": " + why};
}

PointSoA points_to_soa(const std::vector<Vec3>& pts) {
  PointSoA soa;
  const std::size_t n = pts.size();
  soa.px.resize(n);
  soa.py.resize(n);
  soa.pz.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    soa.px[i] = pts[i].x;
    soa.py[i] = pts[i].y;
    soa.pz[i] = pts[i].z;
  }
  return soa;
}

std::size_t checked_product(const std::string& kind,
                            std::span<const int> dimensions) {
  constexpr std::size_t max_points =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  std::size_t total = 1;
  for (int d : dimensions) {
    if (d < 1) throw_bad(kind, "each dimension must be >= 1");
    const auto ud = static_cast<std::size_t>(d);
    if (total > max_points / ud) {
      throw_bad(kind, "point count exceeds the signed evaluator-index limit");
    }
    total *= ud;
  }
  return total;
}

Real checked_coordinate(const std::string& kind, Real value) {
  if (!std::isfinite(value)) {
    throw_bad(kind, "generated coordinate overflowed");
  }
  return value;
}

Vec3 grid_point_unchecked(const ObservationGrid& g, int i, int j, int k) {
  return Vec3{
      checked_coordinate("ObservationGrid",
                         std::fma(static_cast<Real>(i), g.spacing.x, g.origin.x)),
      checked_coordinate("ObservationGrid",
                         std::fma(static_cast<Real>(j), g.spacing.y, g.origin.y)),
      checked_coordinate("ObservationGrid",
                         std::fma(static_cast<Real>(k), g.spacing.z, g.origin.z))};
}

Vec3 slice_point_unchecked(const PlaneSlice& s, int i, int j) {
  const auto coordinate = [i, j](Real origin, Real u, Real v) {
    return std::fma(static_cast<Real>(j), v,
                    std::fma(static_cast<Real>(i), u, origin));
  };
  return Vec3{
      checked_coordinate(
          "PlaneSlice", coordinate(s.origin.x, s.u_step.x, s.v_step.x)),
      checked_coordinate(
          "PlaneSlice", coordinate(s.origin.y, s.u_step.y, s.v_step.y)),
      checked_coordinate(
          "PlaneSlice", coordinate(s.origin.z, s.u_step.z, s.v_step.z))};
}

Vec3 line_point_unchecked(const LineProbe& l, int i) {
  const Real t = static_cast<Real>(i) / static_cast<Real>(l.n_points - 1);
  return Vec3{
      checked_coordinate("LineProbe", std::lerp(l.start.x, l.end.x, t)),
      checked_coordinate("LineProbe", std::lerp(l.start.y, l.end.y, t)),
      checked_coordinate("LineProbe", std::lerp(l.start.z, l.end.z, t))};
}

}  // namespace

void PointSoA::validate() const {
  const std::size_t n = px.size();
  if (py.size() != n || pz.size() != n) {
    throw_bad("PointSoA", "component vectors have inconsistent lengths");
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (!(std::isfinite(px[i]) && std::isfinite(py[i])
          && std::isfinite(pz[i]))) {
      throw_bad("PointSoA", "coordinate has a non-finite component");
    }
  }
}

// ---------------------------------------------------------------------------
// PointCloud
// ---------------------------------------------------------------------------

void PointCloud::add(Vec3 p) {
  if (!is_finite(p)) {
    throw_bad("PointCloud", "point has a non-finite component");
  }
  points_.push_back(p);
}

void PointCloud::add(std::span<const Vec3> ps) {
  for (Vec3 p : ps) {
    if (!is_finite(p)) {
      throw_bad("PointCloud", "point has a non-finite component");
    }
  }
  if (!ps.empty() && !points_.empty()) {
    const Vec3* const source_begin = ps.data();
    const Vec3* const source_end = source_begin + ps.size();
    const Vec3* const own_begin = points_.data();
    const Vec3* const own_end = own_begin + points_.size();
    const std::less<const Vec3*> before;
    if (before(source_begin, own_end) && before(own_begin, source_end)) {
      // vector::insert does not permit a source range from the same vector;
      // reallocation would invalidate the span mid-copy. Snapshot overlapping
      // input first while keeping the no-copy path for ordinary large uploads.
      const std::vector<Vec3> snapshot{ps.begin(), ps.end()};
      points_.insert(points_.end(), snapshot.begin(), snapshot.end());
      return;
    }
  }
  points_.insert(points_.end(), ps.begin(), ps.end());
}

PointSoA PointCloud::to_point_soa() const {
  return points_to_soa(points_);
}

// ---------------------------------------------------------------------------
// ObservationGrid
// ---------------------------------------------------------------------------

void ObservationGrid::validate(const ObservationGrid& g) {
  if (!is_finite(g.origin) || !is_finite(g.spacing)) {
    throw_bad("ObservationGrid", "origin or spacing has non-finite component");
  }
  for (int d : g.dims) {
    if (d < 1) {
      throw_bad("ObservationGrid", "each dim must be >= 1");
    }
  }
  (void)checked_product("ObservationGrid",
                        std::span<const int>{g.dims.data(), g.dims.size()});
  const Real origins[] = {g.origin.x, g.origin.y, g.origin.z};
  const Real spacings[] = {g.spacing.x, g.spacing.y, g.spacing.z};
  for (int axis = 0; axis < 3; ++axis) {
    if (g.dims[axis] <= 1) continue;
    const Real first_next = std::fma(Real{1}, spacings[axis], origins[axis]);
    const Real last = std::fma(static_cast<Real>(g.dims[axis] - 1),
                               spacings[axis], origins[axis]);
    const Real last_previous = std::fma(
        static_cast<Real>(g.dims[axis] - 2), spacings[axis], origins[axis]);
    if (!std::isfinite(first_next) || !std::isfinite(last)
        || first_next == origins[axis] || last == last_previous) {
      throw_bad("ObservationGrid",
                "adjacent coordinates collapse or overflow in host precision");
    }
  }
}

std::size_t ObservationGrid::size() const {
  validate(*this);
  return checked_product("ObservationGrid",
                         std::span<const int>{dims.data(), dims.size()});
}

Vec3 ObservationGrid::point_at(int i, int j, int k) const {
  if (i < 0 || i >= dims[0] || j < 0 || j >= dims[1]
      || k < 0 || k >= dims[2]) {
    throw_bad("ObservationGrid", "sample index is outside the declared dimensions");
  }
  validate(*this);
  return grid_point_unchecked(*this, i, j, k);
}

PointCloud ObservationGrid::to_point_cloud() const {
  validate(*this);
  PointCloud pc;
  std::vector<Vec3> buf;
  buf.reserve(size());
  for (int k = 0; k < dims[2]; ++k) {
    for (int j = 0; j < dims[1]; ++j) {
      for (int i = 0; i < dims[0]; ++i) {
        buf.push_back(grid_point_unchecked(*this, i, j, k));
      }
    }
  }
  pc.add(std::span<const Vec3>{buf.data(), buf.size()});
  return pc;
}

PointSoA ObservationGrid::to_point_soa() const {
  validate(*this);
  PointSoA soa;
  const std::size_t n = size();
  soa.px.resize(n);
  soa.py.resize(n);
  soa.pz.resize(n);
  std::size_t flat = 0;
  for (int k = 0; k < dims[2]; ++k) {
    for (int j = 0; j < dims[1]; ++j) {
      for (int i = 0; i < dims[0]; ++i) {
        const Vec3 p = grid_point_unchecked(*this, i, j, k);
        soa.px[flat] = p.x;
        soa.py[flat] = p.y;
        soa.pz[flat] = p.z;
        ++flat;
      }
    }
  }
  return soa;
}

// ---------------------------------------------------------------------------
// PlaneSlice
// ---------------------------------------------------------------------------

void PlaneSlice::validate(const PlaneSlice& s) {
  if (!is_finite(s.origin) || !is_finite(s.u_step) || !is_finite(s.v_step)) {
    throw_bad("PlaneSlice", "origin or step has non-finite component");
  }
  if (s.nu < 1 || s.nv < 1) {
    throw_bad("PlaneSlice", "nu and nv must be >= 1");
  }
  const int dims[] = {s.nu, s.nv};
  (void)checked_product("PlaneSlice", std::span<const int>{dims, 2});
  if (s.nu > 1 && s.nv > 1) {
    const auto normalized = [](Vec3 step) {
      const Real scale = std::max(
          {std::abs(step.x), std::abs(step.y), std::abs(step.z)});
      if (scale == Real{0}) return Vec3{};
      const Vec3 scaled{step.x / scale, step.y / scale, step.z / scale};
      const Real norm = std::hypot(scaled.x, scaled.y, scaled.z);
      return Vec3{scaled.x / norm, scaled.y / norm, scaled.z / norm};
    };
    const Vec3 u = normalized(s.u_step);
    const Vec3 v = normalized(s.v_step);
    const Vec3 cross{u.y * v.z - u.z * v.y,
                     u.z * v.x - u.x * v.z,
                     u.x * v.y - u.y * v.x};
    const Real sine = std::hypot(cross.x, cross.y, cross.z);
    if (sine <= Real{64} * std::numeric_limits<Real>::epsilon()) {
      throw_bad("PlaneSlice", "u_step and v_step must be linearly independent");
    }
  }

  const auto point = [&s](int i, int j) {
    const auto coordinate = [i, j](Real origin, Real u, Real v) {
      return std::fma(static_cast<Real>(j), v,
                      std::fma(static_cast<Real>(i), u, origin));
    };
    return Vec3{coordinate(s.origin.x, s.u_step.x, s.v_step.x),
                coordinate(s.origin.y, s.u_step.y, s.v_step.y),
                coordinate(s.origin.z, s.u_step.z, s.v_step.z)};
  };
  const auto distinct_finite = [](Vec3 a, Vec3 b) {
    return is_finite(a) && is_finite(b)
        && (a.x != b.x || a.y != b.y || a.z != b.z);
  };

  // Check both ends of every boundary row/column.  An individually resolvable
  // step can still collapse after the other axis has translated the samples to
  // a much larger coordinate, and the combined corner can overflow even when
  // each isolated endpoint is finite.
  if (s.nu > 1) {
    for (const int j : {0, s.nv - 1}) {
      if (!distinct_finite(point(0, j), point(1, j))
          || !distinct_finite(point(s.nu - 2, j), point(s.nu - 1, j))) {
        throw_bad("PlaneSlice",
                  "adjacent u samples collapse or overflow in host precision");
      }
    }
  }
  if (s.nv > 1) {
    for (const int i : {0, s.nu - 1}) {
      if (!distinct_finite(point(i, 0), point(i, 1))
          || !distinct_finite(point(i, s.nv - 2), point(i, s.nv - 1))) {
        throw_bad("PlaneSlice",
                  "adjacent v samples collapse or overflow in host precision");
      }
    }
  }
  for (const int i : {0, s.nu - 1}) {
    for (const int j : {0, s.nv - 1}) {
      if (!is_finite(point(i, j))) {
        throw_bad("PlaneSlice", "sample corner overflowed in host precision");
      }
    }
  }
}

std::size_t PlaneSlice::size() const {
  validate(*this);
  const int dims[] = {nu, nv};
  return checked_product("PlaneSlice", std::span<const int>{dims, 2});
}

Vec3 PlaneSlice::point_at(int i, int j) const {
  if (i < 0 || i >= nu || j < 0 || j >= nv) {
    throw_bad("PlaneSlice", "sample index is outside the declared dimensions");
  }
  validate(*this);
  // Each multiply-add is rounded only once. In particular, i*u may exceed
  // Real even when o+i*u is representable; std::fma evaluates that affine
  // combination without exposing the overflowing intermediate product.
  return slice_point_unchecked(*this, i, j);
}

PointCloud PlaneSlice::to_point_cloud() const {
  validate(*this);
  PointCloud pc;
  std::vector<Vec3> buf;
  buf.reserve(size());
  for (int j = 0; j < nv; ++j) {
    for (int i = 0; i < nu; ++i) {
      buf.push_back(slice_point_unchecked(*this, i, j));
    }
  }
  pc.add(std::span<const Vec3>{buf.data(), buf.size()});
  return pc;
}

PointSoA PlaneSlice::to_point_soa() const {
  validate(*this);
  PointSoA soa;
  const std::size_t n = size();
  soa.px.resize(n);
  soa.py.resize(n);
  soa.pz.resize(n);
  std::size_t flat = 0;
  for (int j = 0; j < nv; ++j) {
    for (int i = 0; i < nu; ++i) {
      const Vec3 p = slice_point_unchecked(*this, i, j);
      soa.px[flat] = p.x;
      soa.py[flat] = p.y;
      soa.pz[flat] = p.z;
      ++flat;
    }
  }
  return soa;
}

// ---------------------------------------------------------------------------
// LineProbe
// ---------------------------------------------------------------------------

void LineProbe::validate(const LineProbe& l) {
  if (!is_finite(l.start) || !is_finite(l.end)) {
    throw_bad("LineProbe", "start or end has non-finite component");
  }
  if (l.n_points < 2) {
    throw_bad("LineProbe", "n_points must be >= 2");
  }
  if (l.start.x == l.end.x && l.start.y == l.end.y && l.start.z == l.end.z) {
    throw_bad("LineProbe", "start and end must be distinct");
  }
  const Real t = Real{1} / static_cast<Real>(l.n_points - 1);
  const Vec3 next{std::lerp(l.start.x, l.end.x, t),
                  std::lerp(l.start.y, l.end.y, t),
                  std::lerp(l.start.z, l.end.z, t)};
  const Real previous_t = static_cast<Real>(l.n_points - 2)
                        / static_cast<Real>(l.n_points - 1);
  const Vec3 previous{std::lerp(l.start.x, l.end.x, previous_t),
                      std::lerp(l.start.y, l.end.y, previous_t),
                      std::lerp(l.start.z, l.end.z, previous_t)};
  if ((next.x == l.start.x && next.y == l.start.y && next.z == l.start.z)
      || (previous.x == l.end.x && previous.y == l.end.y
          && previous.z == l.end.z)) {
    throw_bad("LineProbe", "adjacent samples collapse in host precision");
  }
}

std::size_t LineProbe::size() const {
  validate(*this);
  return static_cast<std::size_t>(n_points);
}

Vec3 LineProbe::point_at(int i) const {
  if (i < 0 || i >= n_points) {
    throw_bad("LineProbe", "sample index is outside the declared dimensions");
  }
  validate(*this);
  // C++20 lerp is specified to avoid overflow for finite endpoints and keeps
  // the exact endpoints at t=0/1.
  return line_point_unchecked(*this, i);
}

PointCloud LineProbe::to_point_cloud() const {
  validate(*this);
  PointCloud pc;
  std::vector<Vec3> buf;
  buf.reserve(size());
  for (int i = 0; i < n_points; ++i) {
    buf.push_back(line_point_unchecked(*this, i));
  }
  pc.add(std::span<const Vec3>{buf.data(), buf.size()});
  return pc;
}

PointSoA LineProbe::to_point_soa() const {
  validate(*this);
  PointSoA soa;
  const std::size_t n = size();
  soa.px.resize(n);
  soa.py.resize(n);
  soa.pz.resize(n);
  for (int i = 0; i < n_points; ++i) {
    const Vec3 p = line_point_unchecked(*this, i);
    soa.px[static_cast<std::size_t>(i)] = p.x;
    soa.py[static_cast<std::size_t>(i)] = p.y;
    soa.pz[static_cast<std::size_t>(i)] = p.z;
  }
  return soa;
}

// ---------------------------------------------------------------------------
// Device expansion
// ---------------------------------------------------------------------------
//
// A structured observation set is a description, not data: it expands to one
// point per lattice site through a short affine formula. Doing that on the host
// and uploading the result was the last per-point host arithmetic in front of
// the device-resident field evaluators, so it happens on the device instead.
//
// The host to_point_cloud()/to_point_soa() forms remain because a caller may
// genuinely want host coordinates (a deck echo, a test), but a caller feeding an
// evaluator should use these.

namespace {

// The three launchers share their whole epilogue: allocate the planes, zero a
// status word, launch, and turn an overflow bit into the same exception the
// host checked_coordinate() raised.
template <class Launch>
DevicePointCloud expand_on_device(std::size_t n, const std::string& kind,
                                  Launch&& launch) {
  DevicePointCloud points(n);
  if (n == 0) return points;
  backend::DeviceBuffer<int> status(1);
  launch(points, status);
  int flags = 0;
  status.copy_to_host(&flags, 1);
  if ((flags & 1) != 0) throw_bad(kind, "generated coordinate overflowed");
  return points;
}

}  // namespace

DevicePointCloud ObservationGrid::to_device_point_cloud() const {
  validate(*this);
  QuasarObsGridParams params{};
  const Real origin_components[3] = {origin.x, origin.y, origin.z};
  const Real spacing_components[3] = {spacing.x, spacing.y, spacing.z};
  for (int axis = 0; axis < 3; ++axis) {
    params.origin[axis] = origin_components[axis];
    params.spacing[axis] = spacing_components[axis];
    params.dims[axis] = dims[static_cast<std::size_t>(axis)];
  }
  return expand_on_device(
      size(), "ObservationGrid",
      [&](DevicePointCloud& points, backend::DeviceBuffer<int>& status) {
        ::launch_observation_grid_points(params, points.x(), points.y(),
                                         points.z(), status.device_ptr(),
                                         nullptr);
      });
}

DevicePointCloud PlaneSlice::to_device_point_cloud() const {
  validate(*this);
  QuasarObsPlaneParams params{};
  const Real origin_components[3] = {origin.x, origin.y, origin.z};
  const Real u_components[3] = {u_step.x, u_step.y, u_step.z};
  const Real v_components[3] = {v_step.x, v_step.y, v_step.z};
  for (int axis = 0; axis < 3; ++axis) {
    params.origin[axis] = origin_components[axis];
    params.u_step[axis] = u_components[axis];
    params.v_step[axis] = v_components[axis];
  }
  params.nu = nu;
  params.nv = nv;
  return expand_on_device(
      size(), "PlaneSlice",
      [&](DevicePointCloud& points, backend::DeviceBuffer<int>& status) {
        ::launch_observation_plane_points(params, points.x(), points.y(),
                                          points.z(), status.device_ptr(),
                                          nullptr);
      });
}

DevicePointCloud LineProbe::to_device_point_cloud() const {
  validate(*this);
  QuasarObsLineParams params{};
  const Real start_components[3] = {start.x, start.y, start.z};
  const Real end_components[3] = {end.x, end.y, end.z};
  for (int axis = 0; axis < 3; ++axis) {
    params.start[axis] = start_components[axis];
    params.end[axis] = end_components[axis];
  }
  params.n_points = n_points;
  return expand_on_device(
      size(), "LineProbe",
      [&](DevicePointCloud& points, backend::DeviceBuffer<int>& status) {
        ::launch_observation_line_points(params, points.x(), points.y(),
                                         points.z(), status.device_ptr(),
                                         nullptr);
      });
}

}  // namespace quasar::core
