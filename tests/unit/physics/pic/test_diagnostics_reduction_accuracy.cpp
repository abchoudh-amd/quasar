// Accuracy and determinism of the device PIC diagnostic reductions.
//
// The host implementations these kernels replaced have been deleted, so there
// is no shipped reference to compare against. Each test therefore builds its
// OWN long-double oracle over the same inputs. That is the whole point: an
// oracle written here is a few lines of obvious sequential arithmetic that a
// reader can check by eye, whereas keeping the old production code alive just
// to test against it would mean maintaining two implementations forever.
//
// Three distinct properties are asserted, because any one of them alone can be
// satisfied by a broken kernel:
//
//   * ACCURACY  -- the device result is within a couple of ulps of the
//                  exactly-rounded answer, measured against the oracle.
//   * TEETH     -- it is dramatically better than a naive double sum over the
//                  same terms. Without this, a kernel whose compensation had
//                  been silently disabled (a stray -ffp-contract=fast, say)
//                  would still pass the accuracy check on friendly data.
//   * DETERMINISM -- repeated launches are bit-identical. This is the property
//                  the port actually promises, since bit-equality with any
//                  host ordering is unattainable for a parallel sum.
//
// A note on what changed numerically. The deleted host code accumulated in
// long double with Kahan compensation. These kernels accumulate in double with
// Kahan compensation in a scaled exponent frame, which is worth roughly 106
// bits of effective mantissa against long double's 64 -- so the device is not
// a downgrade despite the narrower base type, and both are far beyond what a
// diagnostic needs. The tests below pin that down rather than assuming it.

#include "quasar/backend/memory.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/yee_field.hpp"
#include "quasar/physics/pic/diagnostics.hpp"
#include "quasar/physics/pic/kernels.hpp"
#include "quasar/physics/pic/species.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::ScalarGrid2D;
using quasar::YeeField2D;
using quasar::pic::ParticleSpecies;
using quasar::pic::SpeciesConfig;

constexpr long double kEps =
    static_cast<long double>(std::numeric_limits<Real>::epsilon());

// Kahan-compensated long-double reference. Carries ~11 more mantissa bits than
// the quantity under test even before compensation, which is enough to rank
// two double-precision results against each other.
class Oracle {
 public:
  void add(long double term) {
    const long double next = sum_ + term;
    if (std::fabs(sum_) >= std::fabs(term)) {
      correction_ += (sum_ - next) + term;
    } else {
      correction_ += (term - next) + sum_;
    }
    sum_ = next;
  }
  long double value() const { return sum_ + correction_; }

 private:
  long double sum_{0.0L};
  long double correction_{0.0L};
};

// BoundarySpec defaults to periodic on all four faces, so a default-constructed
// spec is emphatically NOT a wall. Say which one is wanted explicitly: the
// difference decides whether a face-centred axis has nx or nx+1 samples and
// whether its edge samples carry half-width control volumes.
quasar::boundary::BoundarySpec wall_spec() {
  quasar::boundary::BoundarySpec spec;
  for (auto& face : spec.field) face = "wall";
  for (auto& face : spec.particle) face = "wall";
  return spec;
}

quasar::boundary::BoundarySpec periodic_spec() {
  return quasar::boundary::BoundarySpec{};
}

long double relative_error(Real actual, long double reference) {
  if (reference == 0.0L) return std::fabs(static_cast<long double>(actual));
  return std::fabs((static_cast<long double>(actual) - reference) / reference);
}

// -- Particle fixtures -------------------------------------------------------

struct ParticleData {
  std::vector<Real> x, y, vx, vy, vz, weight;
};

// One dominant term plus a long tail of terms far below its ulp. This is the
// arrangement a naive double sum gets badly wrong and a compensated one does
// not; a set of similar-magnitude terms would let both look perfect.
ParticleData wide_dynamic_range_particles(std::size_t n, unsigned seed) {
  std::mt19937 rng{seed};
  std::uniform_real_distribution<Real> unit{Real{0.1}, Real{1.0}};
  std::uniform_int_distribution<int> exponent{-24, 0};

  ParticleData d;
  d.x.resize(n, Real{0.5});
  d.y.resize(n, Real{0.5});
  d.vx.resize(n);
  d.vy.resize(n, Real{0});
  d.vz.resize(n, Real{0});
  d.weight.resize(n);
  for (std::size_t p = 0; p < n; ++p) {
    d.vx[p] = unit(rng);
    // Particle 0 carries a weight ~1e7 times every other, so its energy term
    // dominates and the rest live below the running sum's ulp.
    d.weight[p] = (p == 0)
        ? Real{1e7}
        : unit(rng) * std::pow(Real{2}, static_cast<Real>(exponent(rng)));
  }
  return d;
}

ParticleSpecies make_species(const ParticleData& d, Real mass) {
  SpeciesConfig cfg;
  cfg.name = "test";
  cfg.charge = Real{-1};
  cfg.mass = mass;
  cfg.capacity = d.x.size();
  ParticleSpecies s{cfg};
  s.set_host_particles(d.x, d.y, d.vx, d.vy, d.vz, d.weight);
  return s;
}

long double oracle_kinetic_energy(const ParticleData& d, Real mass) {
  Oracle acc;
  for (std::size_t p = 0; p < d.x.size(); ++p) {
    const long double vx = d.vx[p];
    const long double vy = d.vy[p];
    const long double vz = d.vz[p];
    const long double speed = std::hypot(std::hypot(vx, vy), vz);
    acc.add(0.5L * static_cast<long double>(mass) * speed * speed
            * static_cast<long double>(d.weight[p]));
  }
  return acc.value();
}

// Deliberately uncompensated, in double, in index order. Stands in for "what
// the kernel would produce if the two-sum stopped working".
Real naive_kinetic_energy(const ParticleData& d, Real mass) {
  Real sum{0};
  for (std::size_t p = 0; p < d.x.size(); ++p) {
    const Real speed = std::hypot(std::hypot(d.vx[p], d.vy[p]), d.vz[p]);
    sum += Real{0.5} * mass * speed * speed * d.weight[p];
  }
  return sum;
}

// -- Field fixtures ----------------------------------------------------------

void upload(quasar::backend::DeviceBuffer<Real>& dst,
            const std::vector<Real>& src) {
  dst.copy_from_host(src.data(), src.size());
}

struct FieldData {
  std::vector<Real> ex, ey, ez, bx, by, bz;
};

FieldData wide_dynamic_range_field(const Grid2D& g, unsigned seed) {
  std::mt19937 rng{seed};
  std::uniform_real_distribution<Real> mantissa{Real{0.25}, Real{1.0}};
  std::uniform_int_distribution<int> exponent{-20, 4};

  const std::size_t n = g.storage_size();
  FieldData f;
  for (std::vector<Real>* plane : {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz}) {
    plane->assign(n, Real{0});
    for (std::size_t k = 0; k < n; ++k) {
      (*plane)[k] = mantissa(rng)
                  * std::pow(Real{2}, static_cast<Real>(exponent(rng)));
    }
  }
  return f;
}

YeeField2D<Real> make_fields(const Grid2D& g, const FieldData& f) {
  YeeField2D<Real> fields{g};
  upload(fields.ex, f.ex);
  upload(fields.ey, f.ey);
  upload(fields.ez, f.ez);
  upload(fields.bx, f.bx);
  upload(fields.by, f.by);
  upload(fields.bz, f.bz);
  return fields;
}

// Independent restatement of the Yee energy norm: every staggered sample owns
// its own primal/dual control volume and is squared in place, never collocated
// first. Non-periodic edge faces get a half-width cell.
long double oracle_field_energy_cartesian(const Grid2D& g, const FieldData& f,
                                          bool periodic_x, bool periodic_y) {
  const long double dx = g.dx();
  const long double dy = g.dy();
  const auto wx = [&](int i) -> long double {
    if (periodic_x) return dx;
    return (i == 0 || i == g.nx) ? 0.5L * dx : dx;
  };
  const auto wy = [&](int j) -> long double {
    if (periodic_y) return dy;
    return (j == 0 || j == g.ny) ? 0.5L * dy : dy;
  };
  const int x_faces = periodic_x ? g.nx : g.nx + 1;
  const int y_faces = periodic_y ? g.ny : g.ny + 1;

  Oracle acc;
  const auto add = [&](Real value, long double volume) {
    const long double v = value;
    acc.add(v * v * volume);
  };
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < x_faces; ++i) {
      add(f.ex[g.index(i, j)], wx(i) * dy);
      add(f.by[g.index(i, j)], wx(i) * dy);
    }
  }
  for (int j = 0; j < y_faces; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      add(f.ey[g.index(i, j)], dx * wy(j));
      add(f.bx[g.index(i, j)], dx * wy(j));
    }
  }
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      add(f.ez[g.index(i, j)], dx * dy);
    }
  }
  for (int j = 0; j < y_faces; ++j) {
    for (int i = 0; i < x_faces; ++i) {
      add(f.bz[g.index(i, j)], wx(i) * wy(j));
    }
  }
  return 0.5L * acc.value();
}

// Cylindrical counterpart. The radial axis is never periodic, so all nx+1
// radial faces are physical. Face samples own the dual annulus between
// neighbouring midpoints; cell samples own the full annulus of their column.
long double oracle_field_energy_cylindrical(const Grid2D& g, const FieldData& f,
                                            bool periodic_y) {
  const long double dx = g.dx();
  const long double dy = g.dy();
  const long double r0 = g.origin_x;
  const long double pi = static_cast<long double>(quasar::pi_v<Real>);
  const auto wy = [&](int j) -> long double {
    if (periodic_y) return dy;
    return (j == 0 || j == g.ny) ? 0.5L * dy : dy;
  };
  const int y_faces = periodic_y ? g.ny : g.ny + 1;

  Oracle acc;
  const auto face = [&](Real value, int i, long double dz) {
    const long double width = (i == 0 || i == g.nx) ? 0.5L * dx : dx;
    const long double offset =
        (i == 0) ? 0.5L * dx
                 : (i == g.nx) ? (2.0L * g.nx - 0.5L) * dx : 2.0L * i * dx;
    const long double v = value;
    acc.add(v * v * pi * width * (2.0L * r0 + offset) * dz);
  };
  const auto cell = [&](Real value, int i, long double dz) {
    const long double v = value;
    acc.add(v * v * 2.0L * pi * dx * (r0 + (i + 0.5L) * dx) * dz);
  };

  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i <= g.nx; ++i) {
      face(f.ex[g.index(i, j)], i, dy);
      face(f.ez[g.index(i, j)], i, dy);
    }
  }
  for (int j = 0; j < y_faces; ++j) {
    for (int i = 0; i < g.nx; ++i) cell(f.ey[g.index(i, j)], i, wy(j));
    for (int i = 0; i <= g.nx; ++i) {
      face(f.bx[g.index(i, j)], i, wy(j));
      face(f.bz[g.index(i, j)], i, wy(j));
    }
  }
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) cell(f.by[g.index(i, j)], i, dy);
  }
  return 0.5L * acc.value();
}

}  // namespace

// -- Kinetic energy ----------------------------------------------------------

TEST(PicDiagnosticsReduction, KineticEnergyMatchesLongDoubleOracle) {
  const Real mass = Real{9.1093837015e-31};
  const auto data = wide_dynamic_range_particles(20000, 1234u);
  ParticleSpecies species = make_species(data, mass);

  const Real device = quasar::pic::total_kinetic_energy(species);
  const long double reference = oracle_kinetic_energy(data, mass);

  ASSERT_GT(reference, 0.0L) << "oracle is trivially zero";
  EXPECT_LE(relative_error(device, reference), 4.0L * kEps)
      << "device " << device << " vs oracle "
      << static_cast<double>(reference);
}

TEST(PicDiagnosticsReduction, KineticEnergyBeatsNaiveDoubleSum) {
  const Real mass = Real{9.1093837015e-31};

  long double worst_naive = 0.0L;
  long double worst_device = 0.0L;
  for (unsigned seed : {1u, 7u, 99u, 2024u}) {
    const auto data = wide_dynamic_range_particles(20000, seed);
    ParticleSpecies species = make_species(data, mass);

    const long double reference = oracle_kinetic_energy(data, mass);
    const long double naive_err =
        relative_error(naive_kinetic_energy(data, mass), reference);
    const long double device_err =
        relative_error(quasar::pic::total_kinetic_energy(species), reference);

    EXPECT_LE(device_err, naive_err) << "seed " << seed;
    worst_naive = std::max(worst_naive, naive_err);
    worst_device = std::max(worst_device, device_err);
  }

  // Teeth. Without this the LE assertions above would still pass if the
  // compensation had become a no-op and both paths were equally inaccurate,
  // which is exactly what a stray -ffp-contract=fast would cause.
  ASSERT_GT(worst_naive, 0.0L) << "naive sum is exact here; fixture is too easy "
                                  "to distinguish a compensated sum";
  // Measured margin on this fixture is ~32x (2.0e-16 device vs 6.6e-15 naive).
  // The 10x threshold leaves room for the exact figure to move with particle
  // count or seed while still failing outright if compensation stops working.
  EXPECT_LT(worst_device, worst_naive * 0.1L)
      << "compensated device sum is not measurably better than a naive one: "
      << "device " << static_cast<double>(worst_device) << " vs naive "
      << static_cast<double>(worst_naive);
}

TEST(PicDiagnosticsReduction, KineticEnergyIsBitwiseReproducible) {
  const Real mass = Real{9.1093837015e-31};
  const auto data = wide_dynamic_range_particles(20000, 555u);
  ParticleSpecies species = make_species(data, mass);

  const Real first = quasar::pic::total_kinetic_energy(species);
  for (int trial = 0; trial < 8; ++trial) {
    const Real again = quasar::pic::total_kinetic_energy(species);
    EXPECT_EQ(std::memcmp(&first, &again, sizeof(Real)), 0)
        << "launch " << trial << " differed";
  }
}

// Dead slots must not contribute. A fixture where the dead particles carry the
// largest weights is the only arrangement that catches an ignored alive flag.
TEST(PicDiagnosticsReduction, KineticEnergySkipsDeadParticles) {
  const Real mass = Real{1};
  ParticleData data;
  data.x = {Real{0.5}, Real{0.5}};
  data.y = {Real{0.5}, Real{0.5}};
  // Velocities are normalized to c=1 and ParticleSpecies rejects |v| >= 1.
  data.vx = {Real{0.5}, Real{0.5}};
  data.vy = {Real{0}, Real{0}};
  data.vz = {Real{0}, Real{0}};
  data.weight = {Real{1}, Real{1}};
  ParticleSpecies species = make_species(data, mass);

  // 2 particles * 0.5 * 1 * 0.25 * 1 = 0.25
  EXPECT_NEAR(quasar::pic::total_kinetic_energy(species), Real{0.25},
              Real{1e-15});

  species.set_count(1);
  EXPECT_NEAR(quasar::pic::total_kinetic_energy(species), Real{0.125},
              Real{1e-15});
}

// -- Field energy ------------------------------------------------------------

// Every plane set to 1 makes the answer purely geometric, and the geometry has
// a closed form: summing the half-width edge weights over a face-centred axis
// telescopes to nx*dx, so each of the six planes contributes exactly Lx*Ly and
// the energy is 3*Lx*Ly. Independent of the reduction, this pins the control
// volumes themselves -- a mis-weighted edge or a mis-decoded region shows up
// here as a clean percentage rather than hiding inside random data.
TEST(PicDiagnosticsReduction, FieldEnergyUniformFieldMatchesAnalyticVolume) {
  const Grid2D g = Grid2D::from_cell_spacing(48, 32, Real{0.01}, Real{0.02},
                                             Real{0}, Real{0}, 2);
  FieldData f;
  for (std::vector<Real>* plane : {&f.ex, &f.ey, &f.ez, &f.bx, &f.by, &f.bz}) {
    plane->assign(g.storage_size(), Real{1});
  }
  YeeField2D<Real> fields = make_fields(g, f);

  const auto wall = wall_spec();
  const Real device = quasar::pic::total_em_energy(fields, g, wall, false);
  const Real expected = Real{3} * g.lx * g.ly;
  EXPECT_NEAR(device, expected, expected * Real{1e-14})
      << "device " << device << " vs analytic " << expected;
}

TEST(PicDiagnosticsReduction, FieldEnergyMatchesLongDoubleOracleCartesian) {
  const Grid2D g = Grid2D::from_cell_spacing(48, 32, Real{0.01}, Real{0.02},
                                             Real{0}, Real{0}, 2);
  const FieldData f = wide_dynamic_range_field(g, 4242u);
  YeeField2D<Real> fields = make_fields(g, f);

  const auto wall = wall_spec();
  const Real device =
      quasar::pic::total_em_energy(fields, g, wall, /*cylindrical=*/false);
  const long double reference =
      oracle_field_energy_cartesian(g, f, false, false);

  ASSERT_GT(reference, 0.0L);
  EXPECT_LE(relative_error(device, reference), 4.0L * kEps)
      << "device " << device << " vs oracle "
      << static_cast<double>(reference);
}

TEST(PicDiagnosticsReduction, FieldEnergyMatchesLongDoubleOraclePeriodic) {
  const Grid2D g = Grid2D::from_cell_spacing(32, 32, Real{0.05}, Real{0.05},
                                             Real{0}, Real{0}, 2);
  const FieldData f = wide_dynamic_range_field(g, 777u);
  YeeField2D<Real> fields = make_fields(g, f);

  const auto periodic = periodic_spec();

  const Real device =
      quasar::pic::total_em_energy(fields, g, periodic, /*cylindrical=*/false);
  const long double reference = oracle_field_energy_cartesian(g, f, true, true);

  ASSERT_GT(reference, 0.0L);
  EXPECT_LE(relative_error(device, reference), 4.0L * kEps);
}

// The annular volume weights are where a cylindrical port most easily goes
// wrong: the axis face and the outermost face are half-width and have their own
// radial offsets.
TEST(PicDiagnosticsReduction, FieldEnergyMatchesLongDoubleOracleCylindrical) {
  const Grid2D g = Grid2D::from_cell_spacing(40, 24, Real{0.005}, Real{0.01},
                                             Real{0}, Real{0}, 2);
  const FieldData f = wide_dynamic_range_field(g, 31337u);
  YeeField2D<Real> fields = make_fields(g, f);

  const auto wall = wall_spec();
  const Real device =
      quasar::pic::total_em_energy(fields, g, wall, /*cylindrical=*/true);
  const long double reference = oracle_field_energy_cylindrical(g, f, false);

  ASSERT_GT(reference, 0.0L);
  EXPECT_LE(relative_error(device, reference), 8.0L * kEps)
      << "device " << device << " vs oracle "
      << static_cast<double>(reference);
}

// An off-axis annulus (r0 > 0) takes the other branch of the split product,
// where the 2*r0 term is the dominant one.
TEST(PicDiagnosticsReduction, FieldEnergyCylindricalOffAxis) {
  const Grid2D g = Grid2D::from_cell_spacing(24, 16, Real{0.002}, Real{0.01},
                                             Real{0.4}, Real{0}, 2);
  const FieldData f = wide_dynamic_range_field(g, 9001u);
  YeeField2D<Real> fields = make_fields(g, f);

  const auto wall = wall_spec();
  const Real device =
      quasar::pic::total_em_energy(fields, g, wall, /*cylindrical=*/true);
  const long double reference = oracle_field_energy_cylindrical(g, f, false);

  ASSERT_GT(reference, 0.0L);
  EXPECT_LE(relative_error(device, reference), 8.0L * kEps);
}

TEST(PicDiagnosticsReduction, FieldEnergyIsBitwiseReproducible) {
  const Grid2D g = Grid2D::from_cell_spacing(48, 32, Real{0.01}, Real{0.02},
                                             Real{0}, Real{0}, 2);
  const FieldData f = wide_dynamic_range_field(g, 4242u);
  YeeField2D<Real> fields = make_fields(g, f);
  const auto wall = wall_spec();

  const Real first = quasar::pic::total_em_energy(fields, g, wall, false);
  for (int trial = 0; trial < 8; ++trial) {
    const Real again = quasar::pic::total_em_energy(fields, g, wall, false);
    EXPECT_EQ(std::memcmp(&first, &again, sizeof(Real)), 0)
        << "launch " << trial << " differed";
  }
}

// A grid whose region sizes are not multiples of the block size exercises the
// grid-stride tail and the region-decode boundaries at once.
TEST(PicDiagnosticsReduction, FieldEnergyHandlesNonMultipleOfBlockSize) {
  const Grid2D g = Grid2D::from_cell_spacing(37, 29, Real{0.013}, Real{0.017},
                                             Real{0}, Real{0}, 2);
  const FieldData f = wide_dynamic_range_field(g, 606u);
  YeeField2D<Real> fields = make_fields(g, f);

  const auto wall = wall_spec();
  const Real device = quasar::pic::total_em_energy(fields, g, wall, false);
  const long double reference =
      oracle_field_energy_cartesian(g, f, false, false);

  ASSERT_GT(reference, 0.0L);
  EXPECT_LE(relative_error(device, reference), 4.0L * kEps);
}

TEST(PicDiagnosticsReduction, FieldEnergyRejectsNonFiniteSample) {
  const Grid2D g = Grid2D::from_cell_spacing(16, 16, Real{0.01}, Real{0.01},
                                             Real{0}, Real{0}, 2);
  FieldData f = wide_dynamic_range_field(g, 11u);
  f.ez[g.index(5, 7)] = std::numeric_limits<Real>::quiet_NaN();
  YeeField2D<Real> fields = make_fields(g, f);

  const auto wall = wall_spec();
  EXPECT_THROW(quasar::pic::total_em_energy(fields, g, wall, false),
               std::domain_error);
}

// -- Gauss-law residual ------------------------------------------------------

// A divergence-free field with zero charge has an identically zero residual.
// The RMS must be exactly zero, not merely small: every term is zero, so any
// non-zero answer means the volume weighting leaked into the numerator.
TEST(PicDiagnosticsReduction, GaussResidualIsZeroForConsistentState) {
  const Grid2D g = Grid2D::from_cell_spacing(32, 32, Real{0.01}, Real{0.01},
                                             Real{0}, Real{0}, 2);
  YeeField2D<Real> fields{g};
  ScalarGrid2D<Real> charge{g};

  const auto wall = wall_spec();
  EXPECT_EQ(quasar::pic::gauss_residual(fields, charge, 2, wall, false),
            Real{0});
}

// A uniform charge with no field gives a residual whose RMS is exactly that
// charge -- an analytic value the reduction and its volume weighting must
// reproduce independently of grid spacing.
TEST(PicDiagnosticsReduction, GaussResidualMatchesAnalyticUniformCharge) {
  const Grid2D g = Grid2D::from_cell_spacing(24, 18, Real{0.02}, Real{0.03},
                                             Real{0}, Real{0}, 2);
  YeeField2D<Real> fields{g};
  ScalarGrid2D<Real> charge{g};

  const Real rho = Real{3.25};
  std::vector<Real> host(g.storage_size(), rho);
  charge.values.copy_from_host(host.data(), host.size());

  const auto wall = wall_spec();
  const Real residual =
      quasar::pic::gauss_residual(fields, charge, 2, wall, false);
  EXPECT_NEAR(residual, rho, rho * Real{1e-14});
}

// Same statement in cylindrical geometry, where the volume weight varies by
// column. The RMS of a constant is still that constant only if the numerator
// and the total volume use the same annular weights.
TEST(PicDiagnosticsReduction, GaussResidualCylindricalVolumeWeightsAreConsistent) {
  const Grid2D g = Grid2D::from_cell_spacing(20, 12, Real{0.004}, Real{0.01},
                                             Real{0}, Real{0}, 2);
  YeeField2D<Real> fields{g};
  ScalarGrid2D<Real> charge{g};

  const Real rho = Real{-1.75};
  std::vector<Real> host(g.storage_size(), rho);
  charge.values.copy_from_host(host.data(), host.size());

  const auto wall = wall_spec();
  const Real residual =
      quasar::pic::gauss_residual(fields, charge, 2, wall, true);
  EXPECT_NEAR(residual, std::abs(rho), std::abs(rho) * Real{1e-14});
}

TEST(PicDiagnosticsReduction, GaussResidualIsBitwiseReproducible) {
  const Grid2D g = Grid2D::from_cell_spacing(37, 29, Real{0.013}, Real{0.017},
                                             Real{0}, Real{0}, 2);
  const FieldData f = wide_dynamic_range_field(g, 8080u);
  YeeField2D<Real> fields = make_fields(g, f);
  ScalarGrid2D<Real> charge{g};
  const auto wall = wall_spec();

  const Real first = quasar::pic::gauss_residual(fields, charge, 2, wall, false);
  for (int trial = 0; trial < 8; ++trial) {
    const Real again =
        quasar::pic::gauss_residual(fields, charge, 2, wall, false);
    EXPECT_EQ(std::memcmp(&first, &again, sizeof(Real)), 0)
        << "launch " << trial << " differed";
  }
}

TEST(PicDiagnosticsReduction, GaussResidualRejectsNonFiniteSample) {
  const Grid2D g = Grid2D::from_cell_spacing(16, 16, Real{0.01}, Real{0.01},
                                             Real{0}, Real{0}, 2);
  YeeField2D<Real> fields{g};
  ScalarGrid2D<Real> charge{g};
  std::vector<Real> host(g.storage_size(), Real{0});
  host[g.index(4, 4)] = std::numeric_limits<Real>::quiet_NaN();
  charge.values.copy_from_host(host.data(), host.size());

  const auto wall = wall_spec();
  EXPECT_THROW(quasar::pic::gauss_residual(fields, charge, 2, wall, false),
               std::domain_error);
}
