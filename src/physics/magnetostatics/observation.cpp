#include "quasar/physics/magnetostatics/observation.hpp"

#include "quasar/core/types.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace quasar::magnetostatics {

namespace {

bool is_finite(Vec3 v) noexcept {
  return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

[[noreturn]] void throw_bad(const std::string& kind, const std::string& why) {
  throw std::invalid_argument{"quasar::magnetostatics::" + kind + ": " + why};
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

}  // namespace

// ---------------------------------------------------------------------------
// PointCloud
// ---------------------------------------------------------------------------

void PointCloud::add(Vec3 p) {
  points_.push_back(p);
}

void PointCloud::add(std::span<const Vec3> ps) {
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
}

Vec3 ObservationGrid::point_at(int i, int j, int k) const {
  return Vec3{origin.x + static_cast<Real>(i) * spacing.x,
              origin.y + static_cast<Real>(j) * spacing.y,
              origin.z + static_cast<Real>(k) * spacing.z};
}

PointCloud ObservationGrid::to_point_cloud() const {
  validate(*this);
  PointCloud pc;
  std::vector<Vec3> buf;
  buf.reserve(size());
  for (int k = 0; k < dims[2]; ++k) {
    for (int j = 0; j < dims[1]; ++j) {
      for (int i = 0; i < dims[0]; ++i) {
        buf.push_back(point_at(i, j, k));
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
        const Vec3 p = point_at(i, j, k);
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
}

Vec3 PlaneSlice::point_at(int i, int j) const {
  return origin + static_cast<Real>(i) * u_step + static_cast<Real>(j) * v_step;
}

PointCloud PlaneSlice::to_point_cloud() const {
  validate(*this);
  PointCloud pc;
  std::vector<Vec3> buf;
  buf.reserve(size());
  for (int j = 0; j < nv; ++j) {
    for (int i = 0; i < nu; ++i) {
      buf.push_back(point_at(i, j));
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
      const Vec3 p = point_at(i, j);
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
}

Vec3 LineProbe::point_at(int i) const {
  const Real t = static_cast<Real>(i) / static_cast<Real>(n_points - 1);
  return start + t * (end - start);
}

PointCloud LineProbe::to_point_cloud() const {
  validate(*this);
  PointCloud pc;
  std::vector<Vec3> buf;
  buf.reserve(size());
  for (int i = 0; i < n_points; ++i) {
    buf.push_back(point_at(i));
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
    const Vec3 p = point_at(i);
    soa.px[static_cast<std::size_t>(i)] = p.x;
    soa.py[static_cast<std::size_t>(i)] = p.y;
    soa.pz[static_cast<std::size_t>(i)] = p.z;
  }
  return soa;
}

}  // namespace quasar::magnetostatics
