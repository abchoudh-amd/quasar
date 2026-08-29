// Host-vs-device equivalence for the Green's-function boundary coupling.
//
// All three kernels assign one thread per output node and loop over sources
// serially in the host's order, so every assertion here is an EQUALITY. That is
// the whole reason the plasma boundary kernel is written the low-occupancy way
// it is; if a future change reduces over sources cooperatively for speed, these
// tests are what will notice, and the trade should be made deliberately rather
// than by loosening them.

#include "quasar/backend/memory.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/free_boundary.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::equilibrium::CoilFilament;
using quasar::equilibrium::GsCoilSet;
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

// An asymmetric coil set: unequal currents, off-midplane placement, and one
// reversed coil. A symmetric pair would mask an r/z transposition in the
// device Green's function.
std::vector<CoilFilament> test_coils() {
  return {
      CoilFilament{Real{1.6}, Real{0.55}, Real{4.0e5}},
      CoilFilament{Real{1.6}, Real{-0.55}, Real{3.1e5}},
      CoilFilament{Real{0.45}, Real{0.20}, Real{-1.7e5}},
      CoilFilament{Real{2.05}, Real{-0.05}, Real{9.0e4}},
  };
}

ScalarField download(const DeviceBuffer<Real>& d, std::size_t n) {
  ScalarField h(n, Real{0});
  d.copy_to_host(h.data(), n);
  return h;
}

TEST(GsGreensDevice, CoilFieldMatchesHostBitExactly) {
  const EllipticGrid g{65, 33, Real{0.3}, Real{2.2}, Real{-0.8}, Real{0.8}};
  const auto coils = test_coils();

  ScalarField host = quasar::numerics::make_field(g);
  quasar::equilibrium::evaluate_coil_field(g, coils, host);

  DeviceBuffer<Real> d_psi{g.size()};
  GsCoilSet d_coils{coils};
  quasar::equilibrium::launch_gs_evaluate_coil_field(g, d_coils,
                                                     d_psi.device_ptr(),
                                                     nullptr);
  quasar::backend::device_synchronize(nullptr);
  const ScalarField dev = download(d_psi, g.size());

  Real peak = Real{0};
  for (const Real v : dev) peak = std::max(peak, std::abs(v));
  ASSERT_GT(peak, Real{1e-6}) << "coil field is trivially small";

  EXPECT_EQ(bitwise_mismatches(host, dev), 0u)
      << "max |host - device| = " << max_abs_difference(host, dev);
}

// apply_coil_boundary writes ONLY boundary nodes and must leave the interior
// alone. Pre-filling with a sentinel is what catches a kernel that writes
// everywhere.
TEST(GsGreensDevice, CoilBoundaryMatchesHostAndPreservesInterior) {
  const EllipticGrid g{65, 33, Real{0.3}, Real{2.2}, Real{-0.8}, Real{0.8}};
  const auto coils = test_coils();

  const Real sentinel = Real{-12345.75};
  ScalarField host(g.size(), sentinel);
  quasar::equilibrium::apply_coil_boundary(g, coils, host);

  ScalarField seed(g.size(), sentinel);
  DeviceBuffer<Real> d_psi{g.size()};
  d_psi.copy_from_host(seed.data(), seed.size());

  GsCoilSet d_coils{coils};
  quasar::equilibrium::launch_gs_apply_coil_boundary(g, d_coils,
                                                     d_psi.device_ptr(),
                                                     nullptr);
  quasar::backend::device_synchronize(nullptr);
  const ScalarField dev = download(d_psi, g.size());

  EXPECT_EQ(bitwise_mismatches(host, dev), 0u)
      << "max |host - device| = " << max_abs_difference(host, dev);

  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      if (!g.on_boundary(i, j)) {
        ASSERT_EQ(dev[g.index(i, j)], sentinel)
            << "interior node (" << i << ", " << j << ") was overwritten";
      }
    }
  }
}

// The hot path. A localized current blob leaves most sources exactly zero,
// which also exercises the zero-source skip.
TEST(GsGreensDevice, PlasmaBoundaryMatchesHostBitExactly) {
  const EllipticGrid g{65, 33, Real{0.3}, Real{2.2}, Real{-0.8}, Real{0.8}};

  ScalarField j_phi = quasar::numerics::make_field(g);
  for (int j = 1; j < g.nz - 1; ++j) {
    for (int i = 1; i < g.nr - 1; ++i) {
      const Real dr = (g.r(i) - Real{1.15}) / Real{0.45};
      const Real dz = (g.z(j) - Real{0.05}) / Real{0.40};
      const Real s2 = dr * dr + dz * dz;
      if (s2 >= Real{1}) continue;  // exact zeros outside the blob
      j_phi[g.index(i, j)] = Real{2.4e6} * (Real{1} - s2) * (Real{1} - s2);
    }
  }

  // Start from a nonzero boundary so the kernel's ADD semantics are tested,
  // not just an assignment that happens to look right against a zero seed.
  ScalarField host = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      if (g.on_boundary(i, j)) host[g.index(i, j)] = Real{0.37} * g.r(i);
    }
  }
  ScalarField seed = host;
  quasar::equilibrium::add_plasma_boundary(g, j_phi, host);

  DeviceBuffer<Real> d_j{j_phi.size()};
  d_j.copy_from_host(j_phi.data(), j_phi.size());
  DeviceBuffer<Real> d_psi{seed.size()};
  d_psi.copy_from_host(seed.data(), seed.size());

  quasar::equilibrium::launch_gs_add_plasma_boundary(g, d_j.device_ptr(),
                                                     d_psi.device_ptr(),
                                                     nullptr);
  quasar::backend::device_synchronize(nullptr);
  const ScalarField dev = download(d_psi, g.size());

  // The contribution must actually be nonzero somewhere, or the equality below
  // is only comparing the seed with itself.
  ASSERT_GT(max_abs_difference(seed, dev), Real{1e-9})
      << "plasma boundary contribution is trivially small";

  EXPECT_EQ(bitwise_mismatches(host, dev), 0u)
      << "max |host - device| = " << max_abs_difference(host, dev);
}

// Every perimeter node must be visited exactly once. A duplicated node would
// double-count its contribution and a missed one would keep its seed value;
// both are invisible on a square grid with a symmetric source, so this uses
// neither.
TEST(GsGreensDevice, PlasmaBoundaryCoversEveryPerimeterNodeOnce) {
  const EllipticGrid g{41, 27, Real{0.35}, Real{1.85}, Real{-0.7}, Real{0.9}};

  // Unit source at a single asymmetric interior node: every boundary node then
  // receives a distinct, strictly nonzero contribution.
  ScalarField j_phi = quasar::numerics::make_field(g);
  j_phi[g.index(13, 7)] = Real{1.0e6};

  ScalarField host = quasar::numerics::make_field(g);
  quasar::equilibrium::add_plasma_boundary(g, j_phi, host);

  DeviceBuffer<Real> d_j{j_phi.size()};
  d_j.copy_from_host(j_phi.data(), j_phi.size());
  DeviceBuffer<Real> d_psi{g.size()};

  quasar::equilibrium::launch_gs_add_plasma_boundary(g, d_j.device_ptr(),
                                                     d_psi.device_ptr(),
                                                     nullptr);
  quasar::backend::device_synchronize(nullptr);
  const ScalarField dev = download(d_psi, g.size());

  int visited = 0;
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      const std::size_t k = g.index(i, j);
      if (g.on_boundary(i, j)) {
        ASSERT_NE(dev[k], Real{0}) << "perimeter node (" << i << ", " << j
                                   << ") received nothing";
        ++visited;
      } else {
        ASSERT_EQ(dev[k], Real{0}) << "interior node (" << i << ", " << j
                                   << ") was written";
      }
    }
  }
  EXPECT_EQ(visited, 2 * g.nr + 2 * (g.nz - 2));
  EXPECT_EQ(bitwise_mismatches(host, dev), 0u)
      << "max |host - device| = " << max_abs_difference(host, dev);
}

}  // namespace
