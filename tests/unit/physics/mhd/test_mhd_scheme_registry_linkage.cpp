// Guards the registry-driven scheme construction the high-order ideal-MHD solver
// relies on. The MHD Riemann solvers, reconstructions, integrators, CT scheme,
// positivity limiter, and fluid/field boundary conditions are registered via
// namespace-scope static initializers in the MHD module TUs; those TUs are pulled
// into the link by referencing an MhdSolver2D symbol below (the same force-link
// hack the PIC linkage test uses). If the registrations ever regress, the
// solver's Registry::create(...) would throw at runtime; this CPU-only test fails
// loudly at the registry level instead of needing a device run.
//
// Mirrors tests/unit/physics/pic/test_scheme_registry_linkage.cpp, retargeted to
// the MHD bases/names. Pure host: only .contains()/.create() are exercised, which
// touch no device memory.

#include "quasar/physics/mhd/mhd_solver.hpp"
#include "quasar/numerics/riemann_solver.hpp"
#include "quasar/numerics/flux_reconstruction.hpp"
#include "quasar/numerics/ssprk_integrator.hpp"
#include "quasar/numerics/ct_scheme.hpp"
#include "quasar/numerics/positivity_limiter.hpp"
#include "quasar/boundary/mhd_boundary.hpp"
#include "quasar/core/registry.hpp"

#include <gtest/gtest.h>

// Reference an MhdSolver2D symbol so the linker keeps the MHD module TUs (which
// own the QUASAR_REGISTER_* static initializers) in this executable. A pointer to
// the out-of-line step() member is defined in that TU, so naming it forces the
// link — exactly how the PIC linkage test pins EmPic2D3V::step.
namespace {
auto kForceLink = &quasar::mhd::MhdSolver2D::step;
}

TEST(MhdSchemeRegistryLinkage, ForceLinkSymbolResolved) {
  ASSERT_NE(kForceLink, nullptr);
}

TEST(MhdSchemeRegistryLinkage, RiemannSolversAreRegistered) {
  const auto& reg = ::quasar::Registry<::quasar::numerics::IRiemannSolver>::instance();
  ASSERT_TRUE(reg.contains("hlld"));
  EXPECT_NE(reg.create("hlld"), nullptr);
}

TEST(MhdSchemeRegistryLinkage, ReconstructionsAreRegistered) {
  const auto& reg = ::quasar::Registry<::quasar::numerics::IFluxReconstruction>::instance();
  ASSERT_TRUE(reg.contains("muscl_minmod"));
  ASSERT_TRUE(reg.contains("mp5"));
  ASSERT_TRUE(reg.contains("mp7"));
  EXPECT_NE(reg.create("muscl_minmod"), nullptr);
  EXPECT_NE(reg.create("mp5"), nullptr);
  EXPECT_NE(reg.create("mp7"), nullptr);
}

TEST(MhdSchemeRegistryLinkage, IntegratorsAreRegistered) {
  const auto& reg = ::quasar::Registry<::quasar::numerics::ISsprkIntegrator>::instance();
  ASSERT_TRUE(reg.contains("ssprk3"));
  EXPECT_NE(reg.create("ssprk3"), nullptr);
}

TEST(MhdSchemeRegistryLinkage, CtSchemesAreRegistered) {
  const auto& reg = ::quasar::Registry<::quasar::numerics::ICtScheme>::instance();
  ASSERT_TRUE(reg.contains("fd_ct_christlieb"));
  EXPECT_NE(reg.create("fd_ct_christlieb"), nullptr);
}

TEST(MhdSchemeRegistryLinkage, PositivityLimitersAreRegistered) {
  const auto& reg = ::quasar::Registry<::quasar::numerics::IPositivityLimiter>::instance();
  ASSERT_TRUE(reg.contains("troubled_cell"));
  EXPECT_NE(reg.create("troubled_cell"), nullptr);
}

TEST(MhdSchemeRegistryLinkage, FluidBoundariesAreRegistered) {
  const auto& reg = ::quasar::Registry<::quasar::boundary::IMhdFluidBoundary>::instance();
  ASSERT_TRUE(reg.contains("periodic"));
  ASSERT_TRUE(reg.contains("outflow"));
  ASSERT_TRUE(reg.contains("wall"));
  EXPECT_FALSE(reg.contains("reflecting"));
  EXPECT_NE(reg.create("periodic"), nullptr);
  EXPECT_NE(reg.create("outflow"), nullptr);
  EXPECT_NE(reg.create("wall"), nullptr);
}

TEST(MhdSchemeRegistryLinkage, FieldBoundariesAreRegistered) {
  const auto& reg = ::quasar::Registry<::quasar::boundary::IMhdFieldBoundary>::instance();
  ASSERT_TRUE(reg.contains("periodic"));
  ASSERT_TRUE(reg.contains("outflow"));
  ASSERT_TRUE(reg.contains("wall"));
  EXPECT_FALSE(reg.contains("reflecting"));
  EXPECT_NE(reg.create("periodic"), nullptr);
  EXPECT_NE(reg.create("outflow"), nullptr);
  EXPECT_NE(reg.create("wall"), nullptr);
}
