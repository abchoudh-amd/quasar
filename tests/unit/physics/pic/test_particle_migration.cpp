// Accuracy and ordering gate for the device-resident distributed migration
// routing.
//
// The routing decision used to be a host loop: wrap a periodic coordinate,
// divide by the cell width, floor, and map the global cell through the balanced
// tile partition. This file holds an independent host reference for that whole
// chain -- written from the definitions rather than copied from the kernel --
// and requires the device to agree with it exactly. Exactly, not approximately:
// every step is either integer arithmetic or a single correctly-rounded
// operation, so any disagreement is a transcription error rather than
// round-off, and a tolerance would hide it.
//
// The ordering claims are gated too, because they are the reason the sort
// exists: records must come back grouped by destination rank and sorted by
// stable id inside each group, whatever order the departing set was in.

#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/physics/pic/kernels.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::pic::PicDepartingParticles;
using quasar::pic::PicMigrationTopology;
using quasar::pic::PicParticleMigrationRecord;

// -- Host reference ----------------------------------------------------------

std::uint64_t reference_owner_coordinate(std::uint64_t index,
                                         std::uint64_t global_size,
                                         std::uint64_t partitions) {
  const std::uint64_t base = global_size / partitions;
  const std::uint64_t remainder = global_size % partitions;
  const std::uint64_t large = (base + 1) * remainder;
  if (index < large) return index / (base + 1);
  return remainder + (index - large) / base;
}

Real reference_wrap(Real value, Real low, Real length) {
  Real reduced = std::fmod(value - low, length);
  if (reduced < Real{0}) reduced += length;
  Real wrapped = low + reduced;
  const Real high = low + length;
  if (!(wrapped < high)) wrapped = low;
  return wrapped;
}

struct ReferenceRoute {
  Real x{}, y{}, x_prev{}, y_prev{};
  std::uint64_t owner{};
  bool valid{true};
};

ReferenceRoute reference_route(const PicMigrationTopology& t, Real x, Real y,
                               Real x_prev, Real y_prev,
                               std::uint64_t source_endpoint) {
  ReferenceRoute out{x, y, x_prev, y_prev, source_endpoint, true};
  if (t.periodic_x != 0) {
    const Real wrapped = reference_wrap(out.x, t.grid.origin_x, t.grid.lx);
    out.x_prev += wrapped - out.x;
    out.x = wrapped;
  }
  if (t.periodic_y != 0) {
    const Real wrapped = reference_wrap(out.y, t.grid.origin_y, t.grid.ly);
    out.y_prev += wrapped - out.y;
    out.y = wrapped;
  }

  const Real x_hi = t.grid.origin_x + t.grid.lx;
  const Real y_hi = t.grid.origin_y + t.grid.ly;
  std::uint64_t ix = 0;
  std::uint64_t iy = 0;
  if (out.x == x_hi) {
    ix = t.global_nx - 1;
  } else {
    const Real c = (out.x - t.grid.origin_x) / t.grid.dx();
    if (!(std::isfinite(c) && c >= Real{0}) ||
        std::floor(c) >= static_cast<Real>(t.global_nx)) {
      out.valid = false;
      return out;
    }
    ix = static_cast<std::uint64_t>(std::floor(c));
  }
  if (out.y == y_hi) {
    iy = t.global_ny - 1;
  } else {
    const Real c = (out.y - t.grid.origin_y) / t.grid.dy();
    if (!(std::isfinite(c) && c >= Real{0}) ||
        std::floor(c) >= static_cast<Real>(t.global_ny)) {
      out.valid = false;
      return out;
    }
    iy = static_cast<std::uint64_t>(std::floor(c));
  }
  const std::uint64_t tx = reference_owner_coordinate(ix, t.global_nx, t.px);
  const std::uint64_t ty = reference_owner_coordinate(iy, t.global_ny, t.py);
  out.owner = ty * t.px + tx;
  return out;
}

// -- Fixtures ----------------------------------------------------------------

PicMigrationTopology make_topology(std::uint64_t px, std::uint64_t py,
                                   int periodic_x, int periodic_y) {
  // 61 x 37: both prime, so neither axis divides evenly by the tile count and
  // the balanced partition's remainder branch is actually exercised.
  Grid2D grid(61, 37, Real{3.25}, Real{1.75}, Real{-1.5}, Real{0.75});
  PicMigrationTopology topology{};
  topology.grid = grid;
  topology.global_nx = 61;
  topology.global_ny = 37;
  topology.px = px;
  topology.py = py;
  topology.endpoint_count = px * py;
  topology.periodic_x = periodic_x;
  topology.periodic_y = periodic_y;
  return topology;
}

struct HostParticles {
  std::vector<Real> x, y, x_prev, y_prev, vx, vy, vz, vphi, weight;
  std::vector<std::uint8_t> alive;
  std::vector<std::uint64_t> id;
  std::size_t size() const { return id.size(); }
};

PicDepartingParticles upload(const HostParticles& host) {
  PicDepartingParticles out;
  const std::size_t n = host.size();
  out.count = n;
  if (n == 0) return out;
  const auto real_plane = [n](const std::vector<Real>& values) {
    quasar::backend::DeviceBuffer<Real> buffer(n,
                                               quasar::backend::uninitialized);
    buffer.copy_from_host(values.data(), n);
    return buffer;
  };
  out.x = real_plane(host.x);
  out.y = real_plane(host.y);
  out.x_prev = real_plane(host.x_prev);
  out.y_prev = real_plane(host.y_prev);
  out.vx = real_plane(host.vx);
  out.vy = real_plane(host.vy);
  out.vz = real_plane(host.vz);
  out.vphi_deposit = real_plane(host.vphi);
  out.weight = real_plane(host.weight);
  out.alive = quasar::backend::DeviceBuffer<std::uint8_t>(
      n, quasar::backend::uninitialized);
  out.alive.copy_from_host(host.alive.data(), n);
  out.id = quasar::backend::DeviceBuffer<std::uint64_t>(
      n, quasar::backend::uninitialized);
  out.id.copy_from_host(host.id.data(), n);
  return out;
}

quasar::backend::DeviceBuffer<std::uint64_t> rank_table(
    std::uint64_t endpoint_count, std::uint64_t rank_count) {
  std::vector<std::uint64_t> host(endpoint_count);
  for (std::uint64_t e = 0; e < endpoint_count; ++e) {
    host[e] = e % rank_count;
  }
  quasar::backend::DeviceBuffer<std::uint64_t> table(
      endpoint_count, quasar::backend::uninitialized);
  table.copy_from_host(host.data(), endpoint_count);
  return table;
}

// A spread of departing particles: inside the domain, past both faces, and --
// crucially -- exactly on the closed upper edges, which is the one position the
// half-open cell map has to special-case.
HostParticles make_particles(const PicMigrationTopology& t, std::size_t n,
                             unsigned seed) {
  std::mt19937_64 rng(seed);
  const Real x_lo = t.grid.origin_x;
  const Real y_lo = t.grid.origin_y;
  const Real x_hi = x_lo + t.grid.lx;
  const Real y_hi = y_lo + t.grid.ly;
  std::uniform_real_distribution<Real> px(x_lo - t.grid.lx, x_hi + t.grid.lx);
  std::uniform_real_distribution<Real> py(y_lo - t.grid.ly, y_hi + t.grid.ly);

  HostParticles host;
  for (std::size_t i = 0; i < n; ++i) {
    Real x = px(rng);
    Real y = py(rng);
    if (i % 17 == 0) x = x_hi;
    if (i % 23 == 0) y = y_hi;
    host.x.push_back(x);
    host.y.push_back(y);
    host.x_prev.push_back(x - Real{0.001});
    host.y_prev.push_back(y + Real{0.002});
    host.vx.push_back(static_cast<Real>(i) * Real{0.5});
    host.vy.push_back(static_cast<Real>(i) * Real{-0.25});
    host.vz.push_back(Real{0.125});
    host.vphi.push_back(Real{0.0625});
    host.weight.push_back(Real{1} + static_cast<Real>(i));
    host.alive.push_back(1);
    // Ids deliberately descending, so a result in ascending id order cannot
    // have come from the input order.
    host.id.push_back(static_cast<std::uint64_t>(n - i));
  }
  return host;
}

quasar::pic::PicMigrationRouting route(const HostParticles& host,
                                       const PicMigrationTopology& topology,
                                       std::uint64_t source_endpoint,
                                       std::size_t rank_count, int* status_out) {
  const PicDepartingParticles departing = upload(host);
  const auto table = rank_table(topology.endpoint_count, rank_count);
  quasar::backend::DeviceBuffer<int> status(1);
  auto routing = quasar::pic::launch_pic_route_departing_particles(
      departing, topology, source_endpoint, /*species_index=*/3, table,
      rank_count, status.device_ptr(), nullptr);
  status.copy_to_host(status_out, 1);
  return routing;
}

// -- Tests -------------------------------------------------------------------

TEST(PicParticleMigration, RoutingMatchesTheHostReferenceExactly) {
  const PicMigrationTopology topology = make_topology(4, 3, 1, 1);
  const HostParticles host = make_particles(topology, 2048, 20260830u);
  int status = 0;
  const auto routing = route(host, topology, /*source_endpoint=*/5, 4, &status);
  ASSERT_EQ(status, 0);
  ASSERT_EQ(routing.records.size(), host.size());

  // Index the result by id: the device reorders, the reference does not.
  std::vector<const PicParticleMigrationRecord*> by_id(host.size() + 1,
                                                       nullptr);
  for (const auto& record : routing.records) {
    ASSERT_LE(record.particle.id, host.size());
    ASSERT_EQ(by_id[record.particle.id], nullptr)
        << "id " << record.particle.id << " appeared twice";
    by_id[record.particle.id] = &record;
  }

  for (std::size_t i = 0; i < host.size(); ++i) {
    const ReferenceRoute expected =
        reference_route(topology, host.x[i], host.y[i], host.x_prev[i],
                        host.y_prev[i], 5);
    ASSERT_TRUE(expected.valid)
        << "reference rejected a particle the fixture intends to be routable";
    const PicParticleMigrationRecord* actual = by_id[host.id[i]];
    ASSERT_NE(actual, nullptr);
    EXPECT_EQ(actual->destination_endpoint, expected.owner) << "particle " << i;
    EXPECT_EQ(actual->particle.x, expected.x) << "particle " << i;
    EXPECT_EQ(actual->particle.y, expected.y) << "particle " << i;
    EXPECT_EQ(actual->particle.x_prev, expected.x_prev) << "particle " << i;
    EXPECT_EQ(actual->particle.y_prev, expected.y_prev) << "particle " << i;
    // The rest of the state is carried, not computed, and must be bit-identical.
    EXPECT_EQ(actual->particle.vx, host.vx[i]);
    EXPECT_EQ(actual->particle.vy, host.vy[i]);
    EXPECT_EQ(actual->particle.vz, host.vz[i]);
    EXPECT_EQ(actual->particle.vphi_deposit, host.vphi[i]);
    EXPECT_EQ(actual->particle.weight, host.weight[i]);
    EXPECT_EQ(actual->particle.alive, host.alive[i]);
    EXPECT_EQ(actual->particle.source_endpoint, 5u);
    EXPECT_EQ(actual->species, 3u);
  }
}

TEST(PicParticleMigration, NonPeriodicAxisIsNotWrapped) {
  // Periodic in x only. A particle past the y face has no owner, and the
  // kernel must say so rather than folding it back.
  const PicMigrationTopology topology = make_topology(2, 2, 1, 0);
  HostParticles host;
  host.x = {topology.grid.origin_x + topology.grid.lx * Real{1.5}};
  host.y = {topology.grid.origin_y + topology.grid.ly * Real{2}};
  host.x_prev = {host.x[0]};
  host.y_prev = {host.y[0]};
  host.vx = {Real{0}};
  host.vy = {Real{0}};
  host.vz = {Real{0}};
  host.vphi = {Real{0}};
  host.weight = {Real{1}};
  host.alive = {1};
  host.id = {7};

  int status = 0;
  const auto routing = route(host, topology, /*source_endpoint=*/0, 1, &status);
  EXPECT_NE(status & quasar::pic::kPicMigrationCoordinateOutsideMesh, 0)
      << "a particle past a non-periodic face must be reported, not wrapped";
  ASSERT_EQ(routing.records.size(), 1u);
  // x was wrapped because that axis is periodic; y was left alone.
  EXPECT_EQ(routing.records[0].particle.x,
            reference_wrap(host.x[0], topology.grid.origin_x,
                           topology.grid.lx));
  EXPECT_EQ(routing.records[0].particle.y, host.y[0]);
}

TEST(PicParticleMigration, RecordsAreGroupedByRankAndOrderedById) {
  const PicMigrationTopology topology = make_topology(4, 3, 1, 1);
  const HostParticles host = make_particles(topology, 1500, 99u);
  constexpr std::size_t kRanks = 5;
  int status = 0;
  const auto routing = route(host, topology, 1, kRanks, &status);
  ASSERT_EQ(status, 0);

  ASSERT_EQ(routing.rank_offsets.size(), kRanks + 1);
  EXPECT_EQ(routing.rank_offsets.front(), 0u);
  EXPECT_EQ(routing.rank_offsets.back(), routing.records.size());
  for (std::size_t r = 0; r + 1 < routing.rank_offsets.size(); ++r) {
    EXPECT_LE(routing.rank_offsets[r], routing.rank_offsets[r + 1]);
  }

  for (std::size_t r = 0; r < kRanks; ++r) {
    const std::size_t begin = routing.rank_offsets[r];
    const std::size_t end = routing.rank_offsets[r + 1];
    for (std::size_t i = begin; i < end; ++i) {
      // Every record in group r really is destined for rank r.
      EXPECT_EQ(routing.records[i].destination_endpoint % kRanks, r);
      if (i > begin) {
        EXPECT_LT(routing.records[i - 1].particle.id,
                  routing.records[i].particle.id)
            << "group " << r << " is not in ascending id order";
      }
    }
  }
}

TEST(PicParticleMigration, MigratedCountCountsOnlyOwnerChanges) {
  const PicMigrationTopology topology = make_topology(4, 3, 1, 1);
  const HostParticles host = make_particles(topology, 512, 4242u);
  const std::uint64_t source = 6;
  int status = 0;
  const auto routing = route(host, topology, source, 3, &status);
  ASSERT_EQ(status, 0);

  std::uint64_t expected = 0;
  for (const auto& record : routing.records) {
    if (record.destination_endpoint != source) ++expected;
  }
  EXPECT_EQ(routing.migrated, expected);
}

TEST(PicParticleMigration, DuplicateIdsAreDetectedAndUniqueIdsAreNot) {
  std::vector<std::uint64_t> unique(4096);
  std::iota(unique.begin(), unique.end(), std::uint64_t{1});
  std::mt19937_64 rng(7u);
  std::shuffle(unique.begin(), unique.end(), rng);
  EXPECT_FALSE(quasar::pic::launch_pic_ids_have_duplicate(unique, nullptr));

  // A repeat far from its twin, so an adjacent-compare only finds it after a
  // correct full sort.
  std::vector<std::uint64_t> repeated = unique;
  repeated[10] = repeated[4000];
  EXPECT_TRUE(quasar::pic::launch_pic_ids_have_duplicate(repeated, nullptr));

  // The 64-bit key is exercised across every radix pass, not just the low byte.
  std::vector<std::uint64_t> wide = {1ull, 1ull << 8, 1ull << 16, 1ull << 24,
                                     1ull << 32, 1ull << 40, 1ull << 48,
                                     1ull << 56};
  EXPECT_FALSE(quasar::pic::launch_pic_ids_have_duplicate(wide, nullptr));
  wide.push_back(1ull << 56);
  EXPECT_TRUE(quasar::pic::launch_pic_ids_have_duplicate(wide, nullptr));
}

TEST(PicParticleMigration, AppendSortsArrivalsByStableId) {
  quasar::pic::SpeciesConfig config;
  config.name = "electron";
  config.capacity = 64;
  quasar::pic::ParticleSpecies species{config};
  species.set_grid(Grid2D(8, 8, Real{1}, Real{1}));

  // Arrivals in descending id, the order MPI is free to deliver them in.
  std::vector<PicParticleMigrationRecord> records;
  for (int i = 20; i >= 1; --i) {
    PicParticleMigrationRecord record{};
    record.particle.id = static_cast<std::uint64_t>(i);
    record.particle.x = static_cast<Real>(i) * Real{0.03125};
    record.particle.y = static_cast<Real>(i) * Real{0.0625};
    record.particle.weight = static_cast<Real>(i);
    record.particle.alive = 1;
    record.destination_endpoint = 0;
    record.species = 0;
    records.push_back(record);
  }
  quasar::pic::launch_pic_append_migrated_records(species, records, nullptr);

  ASSERT_EQ(species.size(), records.size());
  const auto snapshot = species.to_host();
  for (std::size_t i = 0; i < snapshot.id.size(); ++i) {
    EXPECT_EQ(snapshot.id[i], static_cast<std::uint64_t>(i + 1));
    // The payload must travel with its id, not merely be sorted alongside it.
    EXPECT_EQ(snapshot.weight[i], static_cast<Real>(i + 1));
    EXPECT_EQ(snapshot.x[i], static_cast<Real>(i + 1) * Real{0.03125});
    EXPECT_EQ(snapshot.y[i], static_cast<Real>(i + 1) * Real{0.0625});
    EXPECT_EQ(snapshot.alive[i], 1);
  }
}

TEST(PicParticleMigration, AppendPreservesExistingResidents) {
  quasar::pic::SpeciesConfig config;
  config.name = "ion";
  config.capacity = 8;
  quasar::pic::ParticleSpecies species{config};
  species.set_grid(Grid2D(8, 8, Real{1}, Real{1}));
  species.set_host_particles({Real{0.5}}, {Real{0.5}}, {Real{0}}, {Real{0}},
                             {Real{0}}, {Real{2}}, {std::uint64_t{100}});
  ASSERT_EQ(species.size(), 1u);

  std::vector<PicParticleMigrationRecord> records(2);
  records[0].particle.id = 60;
  records[0].particle.weight = Real{6};
  records[0].particle.alive = 1;
  records[1].particle.id = 40;
  records[1].particle.weight = Real{4};
  records[1].particle.alive = 1;
  quasar::pic::launch_pic_append_migrated_records(species, records, nullptr);

  ASSERT_EQ(species.size(), 3u);
  const auto snapshot = species.to_host();
  EXPECT_EQ(snapshot.id[0], 100u);
  EXPECT_EQ(snapshot.weight[0], Real{2});
  // Appended records are sorted among themselves, after the residents.
  EXPECT_EQ(snapshot.id[1], 40u);
  EXPECT_EQ(snapshot.id[2], 60u);
}

}  // namespace
