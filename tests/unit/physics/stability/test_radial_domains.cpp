// Rational-surface location and Chebyshev domain layout.
//
// These run against a SYNTHETIC q profile rather than a solved equilibrium,
// deliberately. The property under test is "does this find every psi where
// q = m/n, and only those", which is a statement about root finding on a known
// function. Driving it from a real equilibrium would make the expected answer
// something the test has to compute for itself -- at which point the test is
// just the implementation again.
//
// A real equilibrium is used in one place, at the end, to confirm the two
// compose.

#include "quasar/backend/memory.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/equilibrium_profile.hpp"
#include "quasar/physics/equilibrium/gs_solver.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"
#include "quasar/physics/stability/kernels.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::stability::FluxCoordinateGrid;
using quasar::stability::RadialDomains;
using quasar::stability::RationalSurfaces;

// A coordinate grid carrying only what the domain layout reads: psi_n, q, and
// the validity flags.
FluxCoordinateGrid synthetic(const std::vector<Real>& q_values,
                             const std::vector<int>& valid = {}) {
  const int n = static_cast<int>(q_values.size());
  FluxCoordinateGrid c{n, 4};

  std::vector<Real> psi_n(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    psi_n[static_cast<std::size_t>(i)] =
        static_cast<Real>(i) / static_cast<Real>(n - 1);
  }
  std::vector<int> ok = valid.empty() ? std::vector<int>(static_cast<std::size_t>(n), 1)
                                      : valid;

  c.psi_n.copy_from_host(psi_n.data(), psi_n.size());
  c.q.copy_from_host(q_values.data(), q_values.size());
  c.valid.copy_from_host(ok.data(), ok.size());
  return c;
}

RationalSurfaces locate(
    const FluxCoordinateGrid& c, int n_toroidal,
    int m_max = RationalSurfaces::kMaxRational) {
  DeviceBuffer<RationalSurfaces> d{1};
  quasar::stability::launch_locate_rational_surfaces(
      c, n_toroidal, m_max, d.device_ptr(), nullptr);
  quasar::backend::device_synchronize(nullptr);
  RationalSurfaces h{};
  d.copy_to_host(&h, 1);
  return h;
}

RadialDomains layout(const RationalSurfaces& r, Real lo, Real hi, int min_dom,
                     Real min_width) {
  DeviceBuffer<RationalSurfaces> d_r{1};
  d_r.copy_from_host(&r, 1);
  DeviceBuffer<RadialDomains> d_out{1};
  quasar::stability::launch_build_radial_domains(d_r.device_ptr(), lo, hi,
                                                 min_dom, min_width,
                                                 d_out.device_ptr(), nullptr);
  quasar::backend::device_synchronize(nullptr);
  RadialDomains h{};
  d_out.copy_to_host(&h, 1);
  return h;
}

// q rising linearly from 1 to 4 across the plasma. For n = 1 the resonances are
// q = 2 and q = 3, at known locations, so the expected answer is exact.
std::vector<Real> linear_q(int n, Real q0, Real q1) {
  std::vector<Real> q(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    q[static_cast<std::size_t>(i)] =
        q0 + (q1 - q0) * static_cast<Real>(i) / static_cast<Real>(n - 1);
  }
  return q;
}

TEST(RadialDomains, RejectsMalformedLaunchContractsBeforeDeviceAccess) {
  FluxCoordinateGrid coords{};
  RationalSurfaces rational{};
  RadialDomains domains{};

  EXPECT_THROW(quasar::stability::launch_locate_rational_surfaces(
                   coords, 1, 4, nullptr, nullptr),
               std::invalid_argument);
  EXPECT_THROW(quasar::stability::launch_locate_rational_surfaces(
                   coords, 1, -1, &rational, nullptr),
               std::invalid_argument);
  EXPECT_THROW(quasar::stability::launch_locate_rational_surfaces(
                   coords, 1, 4, &rational, nullptr),
               std::invalid_argument);

  coords.n_psi = 1;
  EXPECT_THROW(quasar::stability::launch_locate_rational_surfaces(
                   coords, 1, 4, &rational, nullptr),
               std::invalid_argument);

  // The sample count is now useful, but the three arrays read by the kernel
  // are still empty. Exact-size validation must fail before device discovery.
  coords.n_psi = 2;
  EXPECT_THROW(quasar::stability::launch_locate_rational_surfaces(
                   coords, 1, 4, &rational, nullptr),
               std::invalid_argument);

  EXPECT_THROW(quasar::stability::launch_build_radial_domains(
                   nullptr, Real{0}, Real{1}, 1, Real{0}, &domains, nullptr),
               std::invalid_argument);
  EXPECT_THROW(quasar::stability::launch_build_radial_domains(
                   &rational, Real{0}, Real{1}, 1, Real{0}, nullptr, nullptr),
               std::invalid_argument);

  const auto launch_domains = [&](Real lo, Real hi, int count, Real width) {
    quasar::stability::launch_build_radial_domains(
        &rational, lo, hi, count, width, &domains, nullptr);
  };
  EXPECT_THROW(launch_domains(std::numeric_limits<Real>::quiet_NaN(), Real{1},
                              1, Real{0}),
               std::invalid_argument);
  EXPECT_THROW(launch_domains(Real{0}, std::numeric_limits<Real>::infinity(),
                              1, Real{0}),
               std::invalid_argument);
  EXPECT_THROW(launch_domains(Real{1}, Real{1}, 1, Real{0}),
               std::invalid_argument);
  EXPECT_THROW(launch_domains(Real{1}, Real{0}, 1, Real{0}),
               std::invalid_argument);
  EXPECT_THROW(launch_domains(Real{0}, Real{1}, 0, Real{0}),
               std::invalid_argument);
  EXPECT_THROW(launch_domains(Real{0}, Real{1},
                              RadialDomains::kMaxDomains + 1, Real{0}),
               std::invalid_argument);
  EXPECT_THROW(launch_domains(Real{0}, Real{1}, 1, Real{-1}),
               std::invalid_argument);
  EXPECT_THROW(launch_domains(Real{0}, Real{1}, 1,
                              std::numeric_limits<Real>::quiet_NaN()),
               std::invalid_argument);
  EXPECT_THROW(launch_domains(Real{0}, Real{1}, 1, Real{1}),
               std::invalid_argument);
}

TEST(RadialDomains, LocatesIntegerResonancesOnALinearProfile) {
  const auto c = synthetic(linear_q(101, Real{1}, Real{4}));
  const RationalSurfaces r = locate(c, 1);

  // q goes from 1 to 4. The half-open convention claims q = 1 at the innermost
  // surface and excludes q = 4 at the outermost, so m = 1, 2, 3.
  ASSERT_EQ(r.count, 3) << "expected resonances at q = 1, 2, 3";
  EXPECT_FALSE(r.overflow);

  EXPECT_EQ(r.m[0], 1);
  EXPECT_EQ(r.m[1], 2);
  EXPECT_EQ(r.m[2], 3);

  // q = q0 + 3*psi_n, so q = m at psi_n = (m - 1)/3.
  EXPECT_NEAR(r.psi_n[0], Real{0}, Real{1e-12});
  EXPECT_NEAR(r.psi_n[1], Real{1} / Real{3}, Real{1e-3});
  EXPECT_NEAR(r.psi_n[2], Real{2} / Real{3}, Real{1e-3});
}

// n = 2 halves the spacing between resonances, so the same profile must yield
// roughly twice as many. This is what catches an implementation that ignores
// the toroidal mode number.
TEST(RadialDomains, ResonanceCountScalesWithToroidalModeNumber) {
  const auto c = synthetic(linear_q(201, Real{1}, Real{4}));

  const RationalSurfaces r1 = locate(c, 1);
  const RationalSurfaces r2 = locate(c, 2);
  const RationalSurfaces r3 = locate(c, 3);

  EXPECT_EQ(r1.count, 3);   // q = 1, 2, 3
  EXPECT_EQ(r2.count, 6);   // q = 1, 1.5, ... 3.5
  EXPECT_EQ(r3.count, 9);

  for (int k = 0; k < r2.count; ++k) {
    EXPECT_NEAR(static_cast<Real>(r2.m[k]) / Real{2},
                Real{1} + Real{3} * r2.psi_n[k], Real{1e-3})
        << "resonance " << k << " is not at q = m/n";
  }
}

TEST(RadialDomains, NegativeToroidalModeKeepsSurfacesAndFlipsHarmonics) {
  const auto c = synthetic(linear_q(201, Real{1}, Real{4}));
  const RationalSurfaces positive = locate(c, 2);
  const RationalSurfaces negative = locate(c, -2);

  ASSERT_EQ(negative.count, positive.count);
  EXPECT_FALSE(negative.overflow);
  for (int k = 0; k < positive.count; ++k) {
    EXPECT_NEAR(negative.psi_n[k], positive.psi_n[k], Real{1e-14});
    EXPECT_EQ(negative.m[k], -positive.m[k]);
  }
}

// Reversed shear: q dips and comes back, so the same m resonates twice. A scan
// that assumed monotonicity would find only one of each pair.
TEST(RadialDomains, HandlesReversedShearWithRepeatedResonances) {
  const int n = 201;
  std::vector<Real> q(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const Real x = static_cast<Real>(i) / static_cast<Real>(n - 1);
    // Minimum of 1.5 at x = 0.5, rising to 3.5 at both ends.
    q[static_cast<std::size_t>(i)] =
        Real{1.5} + Real{8} * (x - Real{0.5}) * (x - Real{0.5});
  }
  const auto c = synthetic(q);
  const RationalSurfaces r = locate(c, 1);

  // q spans [1.5, 3.5], so m = 2 and m = 3 each resonate twice.
  int count_m2 = 0;
  int count_m3 = 0;
  for (int k = 0; k < r.count; ++k) {
    if (r.m[k] == 2) ++count_m2;
    if (r.m[k] == 3) ++count_m3;
  }
  EXPECT_EQ(count_m2, 2) << "reversed shear should give two q = 2 surfaces";
  EXPECT_EQ(count_m3, 2) << "reversed shear should give two q = 3 surfaces";

  // Output must still be ascending in psi_n even though the m loop within an
  // interval runs ascending in m.
  for (int k = 1; k < r.count; ++k) {
    EXPECT_LE(r.psi_n[k - 1], r.psi_n[k]) << "list is not sorted at " << k;
  }
}

TEST(RadialDomains, ToroidalModeZeroHasNoResonances) {
  const auto c = synthetic(linear_q(51, Real{1}, Real{4}));
  const RationalSurfaces r = locate(c, 0);
  EXPECT_EQ(r.count, 0);
  EXPECT_FALSE(r.overflow);
}

TEST(RadialDomains, ReportsConstantRationalIntervalAsUnsupportedTopology) {
  const auto c = synthetic({Real{2}, Real{2}});
  const RationalSurfaces r = locate(c, 1, 2);

  EXPECT_EQ(r.count, 0);
  EXPECT_FALSE(r.overflow);
  EXPECT_TRUE(r.has_rational_interval);

  const RadialDomains d = layout(r, Real{0}, Real{1}, 2, Real{1e-3});
  EXPECT_TRUE(d.overflow);
  EXPECT_EQ(d.n_domains, 0);
}

TEST(RadialDomains, IgnoresConstantRationalIntervalOutsideHarmonicBasis) {
  const auto c = synthetic({Real{2}, Real{2}});
  const RationalSurfaces r = locate(c, 1, 1);

  EXPECT_EQ(r.count, 0);
  EXPECT_FALSE(r.overflow);
  EXPECT_FALSE(r.has_rational_interval);
}

TEST(RadialDomains, IrrelevantHighHarmonicsDoNotConsumeLocatorCapacity) {
  const auto c = synthetic(linear_q(101, Real{1}, Real{2}));
  const RationalSurfaces r = locate(c, 100, 4);

  EXPECT_EQ(r.count, 0);
  EXPECT_FALSE(r.overflow);
  EXPECT_FALSE(r.has_rational_interval);
}

TEST(RadialDomains, InvalidSurfacesAreSkipped) {
  const int n = 51;
  std::vector<int> valid(static_cast<std::size_t>(n), 1);
  // Invalidate the band that would contain the q = 2 crossing.
  for (int i = 15; i <= 20; ++i) valid[static_cast<std::size_t>(i)] = 0;

  const auto c = synthetic(linear_q(n, Real{1}, Real{4}), valid);
  const RationalSurfaces r = locate(c, 1);

  for (int k = 0; k < r.count; ++k) {
    const Real p = r.psi_n[k];
    EXPECT_FALSE(p > Real{14.0 / 50.0} && p < Real{21.0 / 50.0})
        << "found a resonance inside the invalidated band";
  }
}

TEST(RadialDomains, BreakpointsAreSortedAndSpanTheRange) {
  const auto c = synthetic(linear_q(201, Real{1}, Real{4}));
  const RationalSurfaces r = locate(c, 2);
  const RadialDomains d = layout(r, Real{0.05}, Real{0.95}, 4, Real{0.01});

  ASSERT_GT(d.n_domains, 0);
  EXPECT_FALSE(d.overflow);
  EXPECT_EQ(d.breakpoints[0], Real{0.05});
  EXPECT_EQ(d.breakpoints[d.n_domains], Real{0.95});
  for (int k = 1; k <= d.n_domains; ++k) {
    EXPECT_GT(d.breakpoints[k], d.breakpoints[k - 1])
        << "breakpoints not strictly ascending at " << k;
  }
}

// Every in-range resonance must actually appear as a breakpoint -- that is the
// entire purpose of the layout.
TEST(RadialDomains, EveryInRangeResonanceBecomesABreakpoint) {
  const auto c = synthetic(linear_q(401, Real{1}, Real{4}));
  const RationalSurfaces r = locate(c, 1);
  const Real lo = Real{0.05};
  const Real hi = Real{0.95};
  const RadialDomains d = layout(r, lo, hi, 3, Real{1e-4});

  int matched = 0;
  for (int k = 0; k < r.count; ++k) {
    if (r.psi_n[k] <= lo || r.psi_n[k] >= hi) continue;
    bool found = false;
    for (int b = 0; b <= d.n_domains; ++b) {
      if (std::abs(d.breakpoints[b] - r.psi_n[k]) < Real{1e-9}) found = true;
    }
    EXPECT_TRUE(found) << "resonance at psi_n = " << r.psi_n[k]
                       << " is not a breakpoint";
    if (found) ++matched;
  }
  EXPECT_GT(matched, 0) << "no in-range resonances: the test is vacuous";
}

// Resonances landing almost together must share one cut rather than creating a
// sliver subinterval.  Both harmonic tags and both source locations must
// survive on that cut so downstream layouts split only the affected harmonics.
TEST(RadialDomains, ResonancesTooCloseTogetherAreMerged) {
  RationalSurfaces r{};
  r.count = 3;
  r.psi_n[0] = Real{0.5};
  r.psi_n[1] = Real{0.5000001};  // a hair away
  r.psi_n[2] = Real{0.7};
  r.m[0] = 2;
  r.m[1] = 3;
  r.m[2] = 4;

  const RadialDomains d = layout(r, Real{0}, Real{1}, 1, Real{1e-3});

  int near_half = 0;
  for (int b = 0; b <= d.n_domains; ++b) {
    if (std::abs(d.breakpoints[b] - Real{0.5}) < Real{1e-3}) ++near_half;
  }
  EXPECT_EQ(near_half, 1) << "the near-duplicate resonance was not merged";

  for (int b = 1; b <= d.n_domains; ++b) {
    EXPECT_GE(d.breakpoints[b] - d.breakpoints[b - 1], Real{1e-3})
        << "a sliver subinterval survived at " << b;
  }

  int half_breakpoint = -1;
  for (int b = 0; b <= d.n_domains; ++b) {
    if (std::abs(d.breakpoints[b] - Real{0.5}) < Real{1e-12}) {
      half_breakpoint = b;
    }
  }
  ASSERT_GE(half_breakpoint, 0);
  const int begin = d.resonance_offsets[half_breakpoint];
  const int end = d.resonance_offsets[half_breakpoint + 1];
  ASSERT_EQ(end - begin, 2);
  EXPECT_EQ(d.resonant_m[begin], 2);
  EXPECT_EQ(d.resonant_m[begin + 1], 3);
  EXPECT_DOUBLE_EQ(d.resonant_psi_n[begin], Real{0.5});
  EXPECT_DOUBLE_EQ(d.resonant_psi_n[begin + 1], Real{0.5000001});
}

TEST(RadialDomains, InteriorResonancesNeverSnapOntoPhysicalEndpoints) {
  RationalSurfaces r{};
  r.count = 2;
  r.psi_n[0] = Real{0.001};
  r.psi_n[1] = Real{0.999};
  r.m[0] = 1;
  r.m[1] = 2;

  const RadialDomains d = layout(r, Real{0}, Real{1}, 1, Real{0.01});
  ASSERT_FALSE(d.overflow);
  ASSERT_EQ(d.n_domains, 3);
  EXPECT_EQ(d.breakpoints[0], Real{0});
  EXPECT_EQ(d.breakpoints[1], r.psi_n[0]);
  EXPECT_EQ(d.breakpoints[2], r.psi_n[1]);
  EXPECT_EQ(d.breakpoints[3], Real{1});
  EXPECT_EQ(d.resonance_count, 2);
  EXPECT_EQ(d.resonance_offsets[1], 0);
  EXPECT_EQ(d.resonance_offsets[2], 1);
  EXPECT_EQ(d.resonance_offsets[3], 2);
  EXPECT_EQ(d.resonant_m[0], 1);
  EXPECT_EQ(d.resonant_m[1], 2);
}

TEST(RadialDomains, ResonanceSnappedToRegularBreakpointKeepsProvenance) {
  RationalSurfaces r{};
  r.count = 1;
  r.psi_n[0] = Real{0.5002};
  r.m[0] = 7;

  // min_domains=4 creates an ordinary breakpoint at exactly 0.5.  The nearby
  // resonance must mark that existing cut rational rather than disappearing.
  const RadialDomains d = layout(r, Real{0}, Real{1}, 4, Real{1e-3});
  ASSERT_EQ(d.n_domains, 4);
  EXPECT_DOUBLE_EQ(d.breakpoints[2], Real{0.5});
  ASSERT_EQ(d.resonance_offsets[3] - d.resonance_offsets[2], 1);
  const int tag = d.resonance_offsets[2];
  EXPECT_EQ(d.resonant_m[tag], 7);
  EXPECT_DOUBLE_EQ(d.resonant_psi_n[tag], Real{0.5002});
  EXPECT_EQ(d.resonance_count, 1);
}

TEST(RadialDomains, LaterInsertionDoesNotStealSnappedProvenance) {
  RationalSurfaces r{};
  r.count = 2;
  r.psi_n[0] = Real{0.5009};  // snaps to the regular point at 0.5
  r.psi_n[1] = Real{0.5011};  // far enough from 0.5 to become its own cut
  r.m[0] = 2;
  r.m[1] = 3;

  const RadialDomains d = layout(r, Real{0}, Real{1}, 2, Real{1e-3});
  ASSERT_EQ(d.n_domains, 3);
  EXPECT_DOUBLE_EQ(d.breakpoints[1], Real{0.5});
  EXPECT_DOUBLE_EQ(d.breakpoints[2], Real{0.5011});

  ASSERT_EQ(d.resonance_offsets[2] - d.resonance_offsets[1], 1);
  ASSERT_EQ(d.resonance_offsets[3] - d.resonance_offsets[2], 1);
  EXPECT_EQ(d.resonant_m[d.resonance_offsets[1]], 2);
  EXPECT_DOUBLE_EQ(d.resonant_psi_n[d.resonance_offsets[1]], Real{0.5009});
  EXPECT_EQ(d.resonant_m[d.resonance_offsets[2]], 3);
  EXPECT_DOUBLE_EQ(d.resonant_psi_n[d.resonance_offsets[2]], Real{0.5011});
}

TEST(RadialDomains, MinimumSubdivisionAppliesWithNoResonances) {
  // q well below 1 everywhere: nothing resonates for n = 1.
  const auto c = synthetic(linear_q(51, Real{0.2}, Real{0.8}));
  const RationalSurfaces r = locate(c, 1);
  ASSERT_EQ(r.count, 0);

  const RadialDomains d = layout(r, Real{0}, Real{1}, 6, Real{1e-3});
  EXPECT_EQ(d.n_domains, 6) << "minimum subdivision was not applied";
}

// Composition check against a real equilibrium: the located resonances must sit
// where the computed q profile actually crosses m/n.
TEST(RadialDomains, ComposesWithASolvedEquilibrium) {
  using quasar::equilibrium::CoilFilament;
  using quasar::equilibrium::GsConfig;
  using quasar::equilibrium::GsSolver;
  using quasar::equilibrium::GsStatus;
  using quasar::equilibrium::PolynomialProfile;
  using quasar::numerics::EllipticGrid;

  constexpr int kFSamples = 257;
  constexpr int kNSurfaces = 24;
  constexpr int kNContour = 256;

  const EllipticGrid grid{33, 33, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  GsConfig cfg;
  cfg.grid = grid;
  cfg.coils = {
      CoilFilament{Real{2.4}, Real{0.9}, Real{-3.0e5}},
      CoilFilament{Real{2.4}, Real{-0.9}, Real{-3.0e5}},
      CoilFilament{Real{0.28}, Real{0.0}, Real{1.0e5}},
  };
  cfg.plasma_current = Real{1e6};
  cfg.max_iterations = 400;
  PolynomialProfile profile{std::vector<Real>{Real{1}, Real{-1}},
                            std::vector<Real>{Real{1}, Real{-1}}};

  const auto res =
      GsSolver{cfg, std::make_shared<PolynomialProfile>(profile)}.solve();
  ASSERT_EQ(res.status, GsStatus::converged);

  DeviceBuffer<Real> d_psi{res.psi.size()};
  d_psi.copy_from_host(res.psi.data(), res.psi.size());
  DeviceBuffer<Real> d_f{static_cast<std::size_t>(kFSamples)};
  quasar::equilibrium::launch_gs_integrate_f_profile(
      quasar::equilibrium::to_coefficients(profile), Real{5},
      res.critical.psi_axis, res.critical.psi_boundary, res.profile_scale,
      kFSamples, d_f.device_ptr(), nullptr);

  quasar::equilibrium::GsOperatorScratch op{grid};
  quasar::equilibrium::GsDerivativeFields deriv{grid};
  quasar::equilibrium::launch_gs_compute_derivatives(grid, d_psi.device_ptr(),
                                                     deriv, op, nullptr);
  quasar::equilibrium::GsMagneticField field{grid};
  quasar::equilibrium::launch_gs_compute_field(
      grid, d_psi.device_ptr(), deriv, d_f.device_ptr(), kFSamples,
      res.critical.psi_axis, res.critical.psi_boundary, field, nullptr);
  quasar::equilibrium::GsFluxSurfaces surfaces{kNSurfaces, kNContour};
  quasar::equilibrium::launch_gs_trace_surfaces(
      grid, d_psi.device_ptr(), field, res.critical.axis.r,
      res.critical.axis.z, res.critical.psi_axis, res.critical.psi_boundary,
      surfaces, nullptr);

  FluxCoordinateGrid coords{kNSurfaces, 64};
  quasar::stability::launch_build_flux_coordinates(
      grid, surfaces, field, res.critical.axis.r, res.critical.axis.z, coords,
      nullptr);
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> q(kNSurfaces), psi_n(kNSurfaces);
  std::vector<int> valid(kNSurfaces);
  coords.q.copy_to_host(q.data(), q.size());
  coords.psi_n.copy_to_host(psi_n.data(), psi_n.size());
  coords.valid.copy_to_host(valid.data(), valid.size());

  // This equilibrium has q_axis near 3.8 and rising, so pick a mode number that
  // certainly resonates rather than assuming one does.
  Real q_min = Real{1e30}, q_max = Real{0};
  for (int i = 0; i < kNSurfaces; ++i) {
    if (!valid[i]) continue;
    q_min = std::min(q_min, q[i]);
    q_max = std::max(q_max, q[i]);
  }
  ASSERT_LT(q_min, q_max);

  const int n_toroidal = 1;
  const RationalSurfaces r = locate(coords, n_toroidal);
  ASSERT_GT(r.count, 0) << "q spans [" << q_min << ", " << q_max
                        << "] but no n=1 resonance was found";

  for (int k = 0; k < r.count; ++k) {
    const Real target = static_cast<Real>(r.m[k]) / static_cast<Real>(n_toroidal);
    EXPECT_GE(target, q_min - Real{1e-9});
    EXPECT_LE(target, q_max + Real{1e-9});
    EXPECT_GT(r.psi_n[k], Real{0});
    EXPECT_LT(r.psi_n[k], Real{1});
  }
}

}  // namespace
