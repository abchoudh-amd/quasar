// Host-vs-device equivalence for the sixth-order Grad-Shafranov operator.
//
// This test is the licence to delete the host implementation. The equilibrium
// port is a hard replace, so once numerics/gs_operator_l6.hpp is gone there is
// no oracle left except the manufactured-solution order study -- which pins the
// SCHEME but not the port. This test pins the port.
//
// It asserts BIT-EXACT equality, not a tolerance. That is achievable because
// the device solve is one thread per line running the same serial pivoting
// elimination in the same order as the host, and because the equilibrium HIP
// module is compiled with -ffp-contract=off (see its CMakeLists). A tolerance
// here would be strictly weaker: at 65x65 a genuinely wrong port can easily sit
// inside any tolerance loose enough to absorb an FMA difference.

#include "quasar/backend/memory.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/numerics/gs_operator_l6.hpp"
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

// A smooth, non-polynomial field. Deliberately NOT a Solov'ev polynomial: a
// low-degree polynomial is reproduced exactly by the sixth-order closures, so
// every near-boundary row would return the same trivially correct answer and a
// pivoting bug in the line solve could hide.
ScalarField smooth_field(const EllipticGrid& g) {
  ScalarField f = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      const Real r = g.r(i);
      const Real z = g.z(j);
      f[g.index(i, j)] = std::sin(Real{2.3} * r) * std::exp(Real{-0.7} * z * z)
                       + Real{0.4} * std::log(r) * std::cos(Real{1.9} * z);
    }
  }
  return f;
}

// Count differing bits so a failure reports how far off the port is, rather
// than only that it is off.
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

ScalarField device_apply(const EllipticGrid& g, const ScalarField& x) {
  DeviceBuffer<Real> d_x{x.size()};
  d_x.copy_from_host(x.data(), x.size());
  DeviceBuffer<Real> d_y{g.size()};

  quasar::equilibrium::GsOperatorScratch scratch{g};
  quasar::equilibrium::launch_gs_apply_l6(g, d_x.device_ptr(),
                                          d_y.device_ptr(), scratch, nullptr);
  quasar::backend::device_synchronize(nullptr);

  ScalarField y(g.size(), Real{0});
  d_y.copy_to_host(y.data(), y.size());
  return y;
}

ScalarField device_residual(const EllipticGrid& g, const ScalarField& x,
                            const ScalarField& b) {
  DeviceBuffer<Real> d_x{x.size()};
  d_x.copy_from_host(x.data(), x.size());
  DeviceBuffer<Real> d_b{b.size()};
  d_b.copy_from_host(b.data(), b.size());
  DeviceBuffer<Real> d_r{g.size()};

  quasar::equilibrium::GsOperatorScratch scratch{g};
  quasar::equilibrium::launch_gs_residual_l6(g, d_x.device_ptr(),
                                             d_b.device_ptr(),
                                             d_r.device_ptr(), scratch,
                                             nullptr);
  quasar::backend::device_synchronize(nullptr);

  ScalarField r(g.size(), Real{0});
  d_r.copy_to_host(r.data(), r.size());
  return r;
}

TEST(GsOperatorDevice, ApplyMatchesHostBitExactly) {
  const EllipticGrid g{65, 65, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  const ScalarField x = smooth_field(g);

  ScalarField host_y;
  quasar::numerics::gs_apply_l6(g, x, host_y);
  const ScalarField dev_y = device_apply(g, x);

  ASSERT_EQ(host_y.size(), dev_y.size());

  // Guard against a vacuous pass: two all-zero fields are bit-identical too.
  // Delta* of this field is O(1), so anything near zero means a launch silently
  // did nothing.
  Real peak = Real{0};
  for (const Real v : dev_y) peak = std::max(peak, std::abs(v));
  ASSERT_GT(peak, Real{1e-3}) << "device operator output is trivially small";

  EXPECT_EQ(bitwise_mismatches(host_y, dev_y), 0u)
      << "max |host - device| = " << max_abs_difference(host_y, dev_y);
}

// A non-square grid catches an nr/nz transposition in the sweep strides, which
// a square grid cannot: on 65x65 the radial and axial layouts coincide.
TEST(GsOperatorDevice, ApplyMatchesHostOnNonSquareGrid) {
  const EllipticGrid g{65, 33, Real{0.4}, Real{2.1}, Real{-0.9}, Real{0.7}};
  const ScalarField x = smooth_field(g);

  ScalarField host_y;
  quasar::numerics::gs_apply_l6(g, x, host_y);
  const ScalarField dev_y = device_apply(g, x);

  EXPECT_EQ(bitwise_mismatches(host_y, dev_y), 0u)
      << "max |host - device| = " << max_abs_difference(host_y, dev_y);
}

TEST(GsOperatorDevice, ResidualMatchesHostBitExactly) {
  const EllipticGrid g{65, 65, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  const ScalarField x = smooth_field(g);

  // A right-hand side of the same magnitude as Delta* x, so the subtraction
  // actually cancels significant digits rather than being dominated by one term.
  ScalarField b = quasar::numerics::make_field(g);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      b[g.index(i, j)] = std::cos(Real{1.7} * g.r(i)) * std::sin(Real{2.1} * g.z(j));
    }
  }

  ScalarField host_r;
  quasar::numerics::gs_residual_l6(g, x, b, host_r);
  const ScalarField dev_r = device_residual(g, x, b);

  EXPECT_EQ(bitwise_mismatches(host_r, dev_r), 0u)
      << "max |host - device| = " << max_abs_difference(host_r, dev_r);
}

// The boundary carries Dirichlet data, so both paths must report exactly zero
// there -- not a small number. A nonzero entry would pollute the max-norm the
// Picard loop converges on.
TEST(GsOperatorDevice, ResidualIsExactlyZeroOnBoundary) {
  const EllipticGrid g{65, 65, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  const ScalarField x = smooth_field(g);
  const ScalarField b = quasar::numerics::make_field(g);

  const ScalarField dev_r = device_residual(g, x, b);
  for (int j = 0; j < g.nz; ++j) {
    for (int i = 0; i < g.nr; ++i) {
      if (g.on_boundary(i, j)) {
        EXPECT_EQ(dev_r[g.index(i, j)], Real{0})
            << "boundary node (" << i << ", " << j << ")";
      }
    }
  }
}

// Reusing one scratch across launches must not change the answer. The band
// arrays are allocated uninitialized and every slot is written before it is
// read, so a stale value from a previous launch leaking through would show up
// here as a difference between the first and second call.
TEST(GsOperatorDevice, ScratchIsReusableAcrossLaunches) {
  const EllipticGrid g{65, 65, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  const ScalarField x = smooth_field(g);

  DeviceBuffer<Real> d_x{x.size()};
  d_x.copy_from_host(x.data(), x.size());
  DeviceBuffer<Real> d_y{g.size()};
  quasar::equilibrium::GsOperatorScratch scratch{g};

  ScalarField first(g.size(), Real{0});
  quasar::equilibrium::launch_gs_apply_l6(g, d_x.device_ptr(),
                                          d_y.device_ptr(), scratch, nullptr);
  quasar::backend::device_synchronize(nullptr);
  d_y.copy_to_host(first.data(), first.size());

  ScalarField second(g.size(), Real{0});
  quasar::equilibrium::launch_gs_apply_l6(g, d_x.device_ptr(),
                                          d_y.device_ptr(), scratch, nullptr);
  quasar::backend::device_synchronize(nullptr);
  d_y.copy_to_host(second.data(), second.size());

  EXPECT_EQ(bitwise_mismatches(first, second), 0u);
}

}  // namespace
