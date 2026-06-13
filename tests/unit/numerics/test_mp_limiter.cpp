// Monotonicity-preserving (MP) limiter contract.
//
// The MP5/MP7 reconstruction is high-order-accurate on smooth data, but its
// DEFINING property is non-oscillatory capture at discontinuities: the limited
// interface value must introduce no new local extremum. The smooth convergence
// tests in test_flux_reconstruction.cpp exercise only the high-order interpolation
// (where the limiter early-accepts and does nothing); this file pins the limiter's
// clip on stencils where the unlimited interpolation OVERSHOOTS, so a broken or
// disabled clamp is caught.
//
// The helpers (mp5_reconstruct / mp7_reconstruct / mp5_interp / mp7_interp /
// mp_limit) are QUASAR_HOST_DEVICE inline and depend only on core/types.hpp, so
// these are pure host checks with no HIP runtime.

#include "quasar/numerics/mp_limiter.hpp"

#include <algorithm>
#include <array>

#include <gtest/gtest.h>

namespace {

using quasar::Real;
using quasar::numerics::mp5_interp;
using quasar::numerics::mp5_reconstruct;
using quasar::numerics::mp7_interp;
using quasar::numerics::mp7_reconstruct;

constexpr Real kTol = 1e-12;

// A constant field must reconstruct to the same constant exactly (no spurious
// extremum injected by the high-order stencil or the limiter).
TEST(MpLimiter, ConstantFieldIsReproducedExactly) {
  const Real c = 3.5;
  EXPECT_NEAR(mp5_reconstruct(c, c, c, c, c), c, kTol);
  EXPECT_NEAR(mp7_reconstruct(c, c, c, c, c, c, c), c, kTol);
}

// MP5: on a step where the left-biased interpolation overshoots ABOVE the cells
// straddling the face, the raw interpolation must exceed the stencil max (proving
// the limiter has real work to do) and the limited value must NOT (no new
// extremum). Stencil {0,0,1,1,1}: straddling cells v0=vp1=1, stencil max=1, but
// the 5th-order interpolant rings up to ~1.13.
TEST(MpLimiter, Mp5ClampsOvershootAtStep) {
  const Real vm2 = 0, vm1 = 0, v0 = 1, vp1 = 1, vp2 = 1;
  const Real raw = mp5_interp(vm2, vm1, v0, vp1, vp2);
  const Real lim = mp5_reconstruct(vm2, vm1, v0, vp1, vp2);

  const Real smax = std::max({vm2, vm1, v0, vp1, vp2});
  const Real smin = std::min({vm2, vm1, v0, vp1, vp2});

  // The raw high-order interpolation genuinely overshoots above the stencil max.
  EXPECT_GT(raw, smax + 1e-6);
  // The limiter removes the new extremum: the result stays within the stencil
  // range (and in fact clamps back to the straddling value 1).
  EXPECT_LE(lim, smax + kTol);
  EXPECT_GE(lim, smin - kTol);
  EXPECT_NEAR(lim, 1.0, 1e-9);
}

// MP5: undershoot below the stencil min. Stencil {0,0,0,0,1}: straddling cells
// v0=vp1=0, stencil min=0, but the interpolant dips to ~-0.039.
TEST(MpLimiter, Mp5ClampsUndershootAtStep) {
  const Real vm2 = 0, vm1 = 0, v0 = 0, vp1 = 0, vp2 = 1;
  const Real raw = mp5_interp(vm2, vm1, v0, vp1, vp2);
  const Real lim = mp5_reconstruct(vm2, vm1, v0, vp1, vp2);

  EXPECT_LT(raw, 0.0 - 1e-6);          // raw undershoots below the stencil min
  EXPECT_GE(lim, 0.0 - kTol);          // limiter removes the undershoot
  EXPECT_LE(lim, 1.0 + kTol);
  EXPECT_NEAR(lim, 0.0, 1e-9);
}

// MP7: same overshoot contract with the 7-point stencil. {0,0,0,1,1,1,1} rings
// the 7th-order interpolant above the straddling value 1.
TEST(MpLimiter, Mp7ClampsOvershootAtStep) {
  const Real vm3 = 0, vm2 = 0, vm1 = 0, v0 = 1, vp1 = 1, vp2 = 1, vp3 = 1;
  const Real raw = mp7_interp(vm3, vm2, vm1, v0, vp1, vp2, vp3);
  const Real lim = mp7_reconstruct(vm3, vm2, vm1, v0, vp1, vp2, vp3);

  const Real smax = std::max({vm3, vm2, vm1, v0, vp1, vp2, vp3});
  EXPECT_GT(raw, smax + 1e-6);
  EXPECT_LE(lim, smax + kTol);
  EXPECT_NEAR(lim, 1.0, 1e-9);
}

// No-new-extremum sweep: reconstruct every interior right face of a monotone
// step profile and assert the limited value never leaves the local stencil's
// [min, max]. A disabled clamp would let the high-order interpolant overshoot at
// the jump and fail this.
TEST(MpLimiter, Mp5NoNewExtremumOnMonotoneStep) {
  // Monotone non-increasing Heaviside step at the middle of the array.
  std::array<Real, 12> v{};
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = (i < v.size() / 2) ? 1.0 : 0.0;

  for (std::size_t i = 2; i + 2 < v.size(); ++i) {
    const Real lim = mp5_reconstruct(v[i - 2], v[i - 1], v[i], v[i + 1], v[i + 2]);
    const Real lo = std::min({v[i - 2], v[i - 1], v[i], v[i + 1], v[i + 2]});
    const Real hi = std::max({v[i - 2], v[i - 1], v[i], v[i + 1], v[i + 2]});
    EXPECT_GE(lim, lo - kTol) << "undershoot at i=" << i;
    EXPECT_LE(lim, hi + kTol) << "overshoot at i=" << i;
  }
}

}  // namespace
