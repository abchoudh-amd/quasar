// Host-vs-device equivalence for multigrid and defect correction.
//
// All of these are EQUALITY assertions, which is a stronger outcome than
// expected going in. The host smoother is red-black rather than lexicographic
// Gauss-Seidel, so the colours are independent within a sweep; and the four
// bilinear-prolongation passes are disjoint by (i, j) parity, so there is no
// accumulation order to preserve. Had either been otherwise, this file would
// have had to fall back to a tolerance and the port would have lost its oracle
// for the elliptic solve.
//
// The V-cycle test is the important one: a single smoothing sweep can agree
// while the hierarchy diverges, because restriction and prolongation only enter
// below the finest level.

#include "quasar/backend/memory.hpp"
#include "quasar/numerics/defect_correction.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/numerics/geometric_multigrid.hpp"
#include "quasar/numerics/gs_operator_l2.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::numerics::EllipticGrid;
using quasar::numerics::ScalarField;

std::size_t bitwise_mismatches(const ScalarField& a, const ScalarField& b) {
  std::size_t n = 0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    if (std::memcmp(&a[k], &b[k], sizeof(Real)) != 0) ++n;
  }
  return n;
}

Real max_abs_difference(const ScalarField& a, const ScalarField& b) {
  Real m = Real{0};
  for (std::size_t k = 0; k < a.size(); ++k) {
    m = std::max(m, std::abs(a[k] - b[k]));
  }
  return m;
}

DeviceBuffer<Real> upload(const ScalarField& f) {
  DeviceBuffer<Real> d{f.size()};
  d.copy_from_host(f.data(), f.size());
  return d;
}

ScalarField download(const DeviceBuffer<Real>& d, std::size_t n) {
  ScalarField h(n, Real{0});
  d.copy_to_host(h.data(), n);
  return h;
}

// Nonzero on the boundary, so the Dirichlet data is genuinely carried through
// rather than being trivially zero everywhere.
ScalarField initial_guess(const EllipticGrid& g) {
  ScalarField x = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      x[g.index(i, j)] = Real{0.15} * g.r(i) - Real{0.08} * g.z(j) * g.z(j);
    }
  }
  return x;
}

ScalarField rhs_field(const EllipticGrid& g) {
  ScalarField b = quasar::numerics::make_field(g);
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      b[g.index(i, j)] = std::sin(Real{3.1} * g.r(i))
                       * std::cos(Real{2.7} * g.z(j));
    }
  }
  return b;
}

TEST(GsMultigridDevice, SmootherMatchesHostBitExactly) {
  const EllipticGrid g{65, 33, Real{0.3}, Real{2.0}, Real{-0.8}, Real{0.8}};
  const ScalarField b = rhs_field(g);

  ScalarField host = initial_guess(g);
  quasar::numerics::gs_smooth_rbgs(g, host, b, 3);

  // A one-level hierarchy reduces the V-cycle to its coarsest-level branch,
  // which is exactly `coarse_sweeps` smoothing sweeps and nothing else. That
  // isolates the smoother without exposing it as a separate entry point.
  auto d_x = upload(initial_guess(g));
  auto d_b = upload(b);
  quasar::equilibrium::GsMultigridConfig cfg;
  cfg.max_levels = 1;
  cfg.coarse_sweeps = 3;
  quasar::equilibrium::GsDeviceMultigrid single{g, cfg};
  ASSERT_EQ(single.n_levels(), 1);
  single.v_cycle(d_x.device_ptr(), d_b.device_ptr(), nullptr);
  quasar::backend::device_synchronize(nullptr);

  const ScalarField dev = download(d_x, g.size());
  EXPECT_EQ(bitwise_mismatches(host, dev), 0u)
      << "max |host - device| = " << max_abs_difference(host, dev);
}

TEST(GsMultigridDevice, VCycleMatchesHostBitExactly) {
  const EllipticGrid g{65, 65, Real{0.3}, Real{2.0}, Real{-0.8}, Real{0.8}};
  const ScalarField b = rhs_field(g);

  quasar::numerics::GsMultigrid host_mg{g};
  ASSERT_GT(host_mg.n_levels(), 2)
      << "hierarchy is too shallow to exercise restriction and prolongation";

  ScalarField host = initial_guess(g);
  host_mg.v_cycle(host, b);

  auto d_x = upload(initial_guess(g));
  auto d_b = upload(b);
  quasar::equilibrium::GsDeviceMultigrid dev_mg{g};
  ASSERT_EQ(dev_mg.n_levels(), host_mg.n_levels());

  dev_mg.v_cycle(d_x.device_ptr(), d_b.device_ptr(), nullptr);
  quasar::backend::device_synchronize(nullptr);

  const ScalarField dev = download(d_x, g.size());
  EXPECT_EQ(bitwise_mismatches(host, dev), 0u)
      << "max |host - device| = " << max_abs_difference(host, dev);
}

// Several V-cycles compound any per-cycle discrepancy, and also confirm the
// per-level scratch is correctly reset between cycles.
TEST(GsMultigridDevice, RepeatedVCyclesMatchHost) {
  const EllipticGrid g{65, 65, Real{0.3}, Real{2.0}, Real{-0.8}, Real{0.8}};
  const ScalarField b = rhs_field(g);

  quasar::numerics::GsMultigrid host_mg{g};
  ScalarField host = initial_guess(g);

  auto d_x = upload(initial_guess(g));
  auto d_b = upload(b);
  quasar::equilibrium::GsDeviceMultigrid dev_mg{g};

  for (int cycle = 0; cycle < 5; ++cycle) {
    host_mg.v_cycle(host, b);
    dev_mg.v_cycle(d_x.device_ptr(), d_b.device_ptr(), nullptr);
  }
  quasar::backend::device_synchronize(nullptr);

  const ScalarField dev = download(d_x, g.size());
  EXPECT_EQ(bitwise_mismatches(host, dev), 0u)
      << "max |host - device| = " << max_abs_difference(host, dev);
}

// The V-cycle must not disturb the Dirichlet data. The coarse correction starts
// from zero and prolongation skips boundary nodes, so the boundary should come
// back exactly as it went in.
TEST(GsMultigridDevice, VCyclePreservesBoundaryData) {
  const EllipticGrid g{65, 65, Real{0.3}, Real{2.0}, Real{-0.8}, Real{0.8}};
  const ScalarField b = rhs_field(g);
  const ScalarField seed = initial_guess(g);

  auto d_x = upload(seed);
  auto d_b = upload(b);
  quasar::equilibrium::GsDeviceMultigrid dev_mg{g};
  dev_mg.v_cycle(d_x.device_ptr(), d_b.device_ptr(), nullptr);
  quasar::backend::device_synchronize(nullptr);

  const ScalarField dev = download(d_x, g.size());
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      if (g.on_boundary(i, j)) {
        ASSERT_EQ(dev[g.index(i, j)], seed[g.index(i, j)])
            << "boundary node (" << i << ", " << j << ") was modified";
      }
    }
  }
}

TEST(GsMultigridDevice, DefectCorrectionMatchesHost) {
  const EllipticGrid g{65, 65, Real{0.3}, Real{2.0}, Real{-0.8}, Real{0.8}};
  const ScalarField b = rhs_field(g);

  quasar::numerics::DefectCorrectionConfig host_cfg;
  host_cfg.max_iterations = 40;
  host_cfg.tolerance = Real{1e-8};

  quasar::numerics::GsMultigrid host_mg{g};
  ScalarField host = initial_guess(g);
  const auto host_rep = quasar::numerics::solve_defect_corrected(
      g, host, b, host_mg, host_cfg);

  quasar::equilibrium::GsDefectCorrectionConfig dev_cfg;
  dev_cfg.max_iterations = 40;
  dev_cfg.tolerance = Real{1e-8};

  auto d_x = upload(initial_guess(g));
  auto d_b = upload(b);
  quasar::equilibrium::GsOperatorScratch op{g};
  quasar::equilibrium::GsReduceScratch red{g};
  quasar::equilibrium::GsDeviceMultigrid dev_mg{g};

  const auto dev_rep = quasar::equilibrium::launch_gs_solve_defect_corrected(
      g, d_x.device_ptr(), d_b.device_ptr(), dev_mg, op, red, dev_cfg, nullptr);

  // The iteration count must match, not merely the final state: a different
  // count means the convergence test saw a different residual.
  EXPECT_EQ(host_rep.converged, dev_rep.converged);
  EXPECT_EQ(host_rep.iterations, dev_rep.iterations);
  EXPECT_EQ(host_rep.initial_residual, dev_rep.initial_residual);
  EXPECT_EQ(host_rep.final_residual, dev_rep.final_residual);

  ASSERT_TRUE(dev_rep.converged) << "solve did not converge: test is vacuous";
  ASSERT_GT(dev_rep.iterations, 1)
      << "converged immediately: test is not exercising the loop";

  const ScalarField dev = download(d_x, g.size());
  EXPECT_EQ(bitwise_mismatches(host, dev), 0u)
      << "max |host - device| = " << max_abs_difference(host, dev);
}

}  // namespace
