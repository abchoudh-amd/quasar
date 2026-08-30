// Host-vs-device equivalence for the Grad-Shafranov reductions.
//
// The two reductions are held to deliberately different standards, and the
// tests encode that difference rather than papering over it:
//
//   * max-norm  -> EQUALITY. Max is associative and rounds nothing, so the tree
//                  and the sequential loop must agree bit-for-bit.
//   * sum       -> ACCURACY ORDERING. Summation is not associative, so the
//                  device tree cannot equal the host's naive sequential sum.
//                  The test instead measures both against a long-double
//                  reference and requires the device to be no worse. Asserting
//                  equality here would be asserting something false; asserting
//                  a loose tolerance would let a genuinely broken compensated
//                  sum pass.

#include "quasar/backend/memory.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/free_boundary.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::numerics::EllipticGrid;
using quasar::numerics::ScalarField;

DeviceBuffer<Real> upload(const ScalarField& f) {
  DeviceBuffer<Real> d{f.size()};
  d.copy_from_host(f.data(), f.size());
  return d;
}

Real device_max_norm(const EllipticGrid& g, const ScalarField& f) {
  auto d = upload(f);
  quasar::equilibrium::GsReduceScratch scratch{g};
  quasar::equilibrium::launch_gs_interior_max_norm(g, d.device_ptr(), scratch,
                                                   nullptr);
  return quasar::equilibrium::copy_scalar_to_host(scratch, nullptr);
}

Real device_current(const EllipticGrid& g, const ScalarField& f) {
  auto d = upload(f);
  quasar::equilibrium::GsReduceScratch scratch{g};
  quasar::equilibrium::launch_gs_total_plasma_current(g, d.device_ptr(),
                                                      scratch, nullptr);
  return quasar::equilibrium::copy_scalar_to_host(scratch, nullptr);
}

// Long-double sequential reference. Not exact, but carries ~11 more mantissa
// bits than the quantity under test, which is enough to rank two double results
// against each other.
long double reference_current(const EllipticGrid& g, const ScalarField& f) {
  long double acc = 0.0L;
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      acc += static_cast<long double>(f[g.index(i, j)]);
    }
  }
  return acc * static_cast<long double>(g.dr())
             * static_cast<long double>(g.dz());
}

// Values spanning many orders of magnitude with mixed signs. This is the
// regime that separates a compensated sum from a naive one; a field of similar
// positive values would let both look perfect.
ScalarField cancelling_field(const EllipticGrid& g, unsigned seed) {
  std::mt19937 rng{seed};
  std::uniform_real_distribution<Real> mantissa{Real{-1}, Real{1}};
  std::uniform_int_distribution<int> exponent{-30, 30};

  ScalarField f = quasar::numerics::make_field(g);
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      f[g.index(i, j)] = mantissa(rng) * std::pow(Real{2}, exponent(rng));
    }
  }
  return f;
}

ScalarField physical_field(const EllipticGrid& g) {
  ScalarField f = quasar::numerics::make_field(g);
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      const Real r = g.r(i);
      const Real z = g.z(j);
      f[g.index(i, j)] = Real{1e6} * std::exp(-Real{4} * ((r - Real{1.2}) * (r - Real{1.2})
                                                          + z * z));
    }
  }
  return f;
}

TEST(GsReduceDevice, MaxNormMatchesHostBitExactly) {
  const EllipticGrid g{65, 65, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  const ScalarField f = cancelling_field(g, 12345u);

  const Real host = quasar::numerics::interior_max_norm(g, f);
  const Real dev  = device_max_norm(g, f);

  ASSERT_GT(host, Real{0}) << "reference is trivially zero";
  EXPECT_EQ(std::memcmp(&host, &dev, sizeof(Real)), 0)
      << "host = " << host << ", device = " << dev;
}

// The boundary must be excluded. A field that is huge on the boundary and small
// inside is the only arrangement that catches an off-by-one in the interior
// index mapping; a uniformly-scaled field would not.
TEST(GsReduceDevice, MaxNormExcludesBoundary) {
  const EllipticGrid g{65, 33, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  ScalarField f = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      f[g.index(i, j)] = g.on_boundary(i, j) ? Real{1e9} : Real{2.5};
    }
  }

  EXPECT_EQ(device_max_norm(g, f), Real{2.5});
  EXPECT_EQ(quasar::numerics::interior_max_norm(g, f), device_max_norm(g, f));
}

TEST(GsReduceDevice, MaxNormPropagatesInteriorNaN) {
  const EllipticGrid g{37, 29, Real{0.5}, Real{2.0}, Real{-0.6}, Real{0.6}};
  ScalarField f(g.size(), Real{2.5});
  f[g.index(17, 13)] = std::numeric_limits<Real>::quiet_NaN();

  EXPECT_TRUE(std::isnan(quasar::numerics::interior_max_norm(g, f)));
  EXPECT_TRUE(std::isnan(device_max_norm(g, f)));

  // Boundary data is excluded from this norm, even when it is non-finite.
  f[g.index(17, 13)] = Real{2.5};
  f[g.index(0, 0)] = std::numeric_limits<Real>::quiet_NaN();
  EXPECT_EQ(quasar::numerics::interior_max_norm(g, f), Real{2.5});
  EXPECT_EQ(device_max_norm(g, f), Real{2.5});
}

TEST(GsReduceDevice, CurrentIsNoLessAccurateThanHost) {
  const EllipticGrid g{129, 129, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};

  long double worst_host = 0.0L;
  long double worst_dev  = 0.0L;

  for (unsigned seed : {1u, 7u, 99u, 2024u}) {
    const ScalarField f = cancelling_field(g, seed);

    const long double ref  = reference_current(g, f);
    const Real host = quasar::equilibrium::total_plasma_current(g, f);
    const Real dev  = device_current(g, f);

    const long double host_err = std::abs(static_cast<long double>(host) - ref);
    const long double dev_err  = std::abs(static_cast<long double>(dev) - ref);

    EXPECT_LE(dev_err, host_err)
        << "seed " << seed << ": device error " << static_cast<double>(dev_err)
        << " exceeds host error " << static_cast<double>(host_err);

    worst_host = std::max(worst_host, host_err);
    worst_dev  = std::max(worst_dev, dev_err);
  }

  // Teeth. Without this the LE assertions above would also pass if the
  // compensation were a no-op and both paths were equally (in)accurate, which
  // is exactly the regression a future edit to the two-sum -- or a stray
  // -ffp-contract change -- would introduce.
  // Measured margin on this case is ~84x (7.2e-11 device vs 6.0e-9 host); the
  // 10x threshold leaves room for the exact figure to move with grid size or
  // seed without becoming brittle.
  EXPECT_LT(worst_dev, worst_host * 0.1L)
      << "compensated sum is not measurably better than the naive host sum: "
      << "device " << static_cast<double>(worst_dev) << " vs host "
      << static_cast<double>(worst_host);
}

// On a well-conditioned physical field the two must agree closely in the
// ordinary sense. This is the check that the device sum is the SAME QUANTITY as
// the host sum, not merely a reproducible number.
TEST(GsReduceDevice, CurrentAgreesWithHostOnPhysicalField) {
  const EllipticGrid g{65, 65, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  const ScalarField f = physical_field(g);

  const Real host = quasar::equilibrium::total_plasma_current(g, f);
  const Real dev  = device_current(g, f);

  ASSERT_GT(std::abs(host), Real{1}) << "reference is trivially small";
  EXPECT_NEAR(dev, host, std::abs(host) * Real{1e-13});
}

// Determinism is the property the port actually promises, since bit-equality
// with the host is unattainable for a sum. Repeated launches must be identical.
TEST(GsReduceDevice, CurrentIsBitwiseReproducible) {
  const EllipticGrid g{129, 129, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  const ScalarField f = cancelling_field(g, 555u);

  const Real first = device_current(g, f);
  for (int trial = 0; trial < 8; ++trial) {
    const Real again = device_current(g, f);
    EXPECT_EQ(std::memcmp(&first, &again, sizeof(Real)), 0)
        << "launch " << trial << " differed";
  }
}

// A grid whose interior node count is not a multiple of the block size
// exercises the grid-stride tail in both passes.
TEST(GsReduceDevice, HandlesNonMultipleOfBlockSize) {
  const EllipticGrid g{37, 29, Real{0.5}, Real{2.0}, Real{-0.6}, Real{0.6}};
  const ScalarField f = physical_field(g);

  const Real host = quasar::equilibrium::total_plasma_current(g, f);
  const Real dev  = device_current(g, f);
  ASSERT_GT(std::abs(host), Real{1});
  EXPECT_NEAR(dev, host, std::abs(host) * Real{1e-13});

  EXPECT_EQ(quasar::numerics::interior_max_norm(g, f), device_max_norm(g, f));
}

}  // namespace
