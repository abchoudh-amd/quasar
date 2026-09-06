// Port gate: the device solver against the recorded reference equilibrium.
//
// The per-kernel tests prove each piece matches the host implementation. This
// proves the assembled thing still computes the equilibrium it computed before
// the port, measured against numbers recorded BEFORE any of this work started:
// a host-only header build of the reference deck at commit a7e80e7.
//
// That distinction is what makes this gate meaningful now that the host solver
// is deleted. Every other test compares device code against host code; if both
// were wrong in the same way they would agree. These numbers predate the device
// code entirely.
//
// The constants below ARE the record. They were transcribed from a run under
// runs/, which is scratch space excluded from the repository by its own
// .gitignore -- so that directory cannot be relied on to exist, and this file
// is the only durable copy. Treat a change to these numbers the way you would
// treat a change to a golden file: it is either a deliberate physics change
// that needs justifying, or a regression.
//
// -- Why tolerances rather than equality --------------------------------------
// Three independent reasons, none of which is slack:
//
//   1. summary.txt records 12 significant figures (the driver uses
//      setprecision(12)), so the reference itself does not carry more
//      information than that.
//   2. The plasma-current integral is compensated on device and naive in the
//      host reference, so profile_scale and everything downstream differ in the
//      last bits by construction.
//   3. Ray-marched q and shape inherit the host/device libm difference in
//      cos/sin.
//
// 5e-9 relative sits comfortably above the recorded precision and the measured
// shift from retaining a fully coherent Picard state on every exit. It remains
// more than six orders below a percent-level defect such as a wrong sign, a
// dropped term, or a mis-indexed stencil.

#include "quasar/backend/memory.hpp"
#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/equilibrium_profile.hpp"
#include "quasar/physics/equilibrium/gs_solver.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::equilibrium::CoilFilament;
using quasar::equilibrium::GsConfig;
using quasar::equilibrium::GsResult;
using quasar::equilibrium::GsSolver;
using quasar::equilibrium::GsStatus;
using quasar::equilibrium::PolynomialProfile;
using quasar::numerics::EllipticGrid;

// Recorded reference, transcribed from
// runs/grad_shafranov_tokamak/latest/summary.txt. Do NOT regenerate these from
// the device solver: the whole point is that they predate it.
namespace reference {
constexpr int  kIterations       = 222;
constexpr Real kAchievedCurrent  = Real{1000000};
constexpr Real kProfileScale     = Real{2.23658376659};
constexpr Real kAxisR            = Real{1.22284133631};
constexpr Real kAxisZ            = Real{-8.89461180282e-15};
constexpr Real kPsiAxis          = Real{0.243551266298};
constexpr Real kPsiBoundary      = Real{0.0708419575965};
constexpr Real kQAxis            = Real{3.82503344908};
constexpr Real kQ95              = Real{11.739314832};
constexpr Real kOutermostPsiN    = Real{0.969696969697};
constexpr int  kOpenSurfaces     = 0;
constexpr Real kPlasmaVolume     = Real{10.5133295265};
constexpr Real kMajorRadius      = Real{1.17370583228};
constexpr Real kMinorRadius      = Real{0.602787517767};
constexpr Real kElongation       = Real{1.28723988654};
constexpr Real kTriangularity    = Real{0.045268258133};
}  // namespace reference

constexpr Real kRelTol = Real{5e-9};
constexpr int  kFSamples = 257;
constexpr int  kNSurfaces = 32;
constexpr int  kNTheta = 128;
constexpr Real kFVacuum = Real{5};

// The reference deck, runs/grad_shafranov_tokamak/input.yaml.
GsConfig reference_config() {
  GsConfig cfg;
  cfg.grid = EllipticGrid{65, 65, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  cfg.coils = {
      CoilFilament{Real{2.4}, Real{0.9}, Real{-3.0e5}},
      CoilFilament{Real{2.4}, Real{-0.9}, Real{-3.0e5}},
      CoilFilament{Real{0.28}, Real{0.0}, Real{1.0e5}},
  };
  cfg.plasma_current = Real{1e6};
  cfg.max_iterations = 400;
  cfg.tolerance = Real{1e-9};
  cfg.picard_relaxation = Real{1};
  cfg.enable_newton = false;
  return cfg;
}

PolynomialProfile reference_profile() {
  return PolynomialProfile{std::vector<Real>{Real{1}, Real{-1}},
                           std::vector<Real>{Real{1}, Real{-1}}};
}

void expect_relative(Real actual, Real expected, const char* what,
                     Real rel_tol = kRelTol) {
  EXPECT_NEAR(actual, expected, std::abs(expected) * rel_tol)
      << what << ": got " << actual << ", reference " << expected;
}

TEST(GsPortGate, DeviceSolverReproducesRecordedEquilibrium) {
  const GsConfig cfg = reference_config();
  const auto profile = reference_profile();

  const GsResult res =
      GsSolver{cfg, std::make_shared<PolynomialProfile>(profile)}.solve();

  ASSERT_EQ(res.status, GsStatus::converged)
      << "device solve did not converge: "
      << quasar::equilibrium::to_string(res.status);

  // The iteration count is allowed to move: compensated current integration
  // and retaining the exact evaluated Picard state perturb the trajectory and
  // the convergence test is a threshold. Keep a 10% gate so a qualitatively
  // different convergence path still fails while the corrected 211-iteration
  // device trajectory remains comparable to the 222-iteration host record.
  EXPECT_NEAR(res.iterations, reference::kIterations,
              Real{0.1} * reference::kIterations)
      << "iteration count moved materially from the recorded run";

  expect_relative(res.achieved_current, reference::kAchievedCurrent,
                  "achieved_current_A");
  expect_relative(res.profile_scale, reference::kProfileScale, "profile_scale");
  expect_relative(res.critical.axis.r, reference::kAxisR, "axis_r_m");
  expect_relative(res.critical.psi_axis, reference::kPsiAxis, "psi_axis_Wb");
  expect_relative(res.critical.psi_boundary, reference::kPsiBoundary,
                  "psi_boundary_Wb");

  // axis_z is zero up to up-down symmetry; the recorded -8.9e-15 is noise, so a
  // relative bound on it would be meaningless. The scale that matters is the
  // grid spacing.
  EXPECT_LT(std::abs(res.critical.axis.z), Real{1e-9} * cfg.grid.dz())
      << "axis drifted off the midplane";
}

TEST(GsPortGate, DeviceDiagnosticsReproduceRecordedProfile) {
  const GsConfig cfg = reference_config();
  const auto profile = reference_profile();
  const EllipticGrid& g = cfg.grid;

  const GsResult res =
      GsSolver{cfg, std::make_shared<PolynomialProfile>(profile)}.solve();
  ASSERT_EQ(res.status, GsStatus::converged);

  DeviceBuffer<Real> d_psi{res.psi.size()};
  d_psi.copy_from_host(res.psi.data(), res.psi.size());
  DeviceBuffer<Real> d_f{static_cast<std::size_t>(kFSamples)};
  quasar::equilibrium::launch_gs_integrate_f_profile(
      quasar::equilibrium::to_coefficients(profile), kFVacuum,
      res.critical.psi_axis, res.critical.psi_boundary, res.profile_scale,
      kFSamples, d_f.device_ptr(), nullptr);

  quasar::equilibrium::GsOperatorScratch op{g};
  quasar::equilibrium::GsDerivativeFields deriv{g};
  quasar::equilibrium::launch_gs_compute_derivatives(g, d_psi.device_ptr(),
                                                     deriv, op, nullptr);
  quasar::equilibrium::GsMagneticField field{g};
  quasar::equilibrium::launch_gs_compute_field(
      g, d_psi.device_ptr(), deriv, d_f.device_ptr(), kFSamples,
      res.critical.psi_axis, res.critical.psi_boundary, field, nullptr);

  quasar::equilibrium::GsFluxSurfaces surfaces{kNSurfaces, kNTheta};
  quasar::equilibrium::launch_gs_trace_surfaces(
      g, d_psi.device_ptr(), field, res.critical.axis.r, res.critical.axis.z,
      res.critical.psi_axis, res.critical.psi_boundary, surfaces, nullptr);

  DeviceBuffer<quasar::equilibrium::GsDiagnostics> d_diag{1};
  quasar::equilibrium::launch_gs_reduce_diagnostics(surfaces,
                                                    d_diag.device_ptr(),
                                                    nullptr);
  const auto diag =
      quasar::equilibrium::copy_diagnostics_to_host(d_diag, nullptr);

  ASSERT_TRUE(diag.have_closed);
  EXPECT_EQ(diag.n_open_surfaces, reference::kOpenSurfaces);
  expect_relative(diag.psi_n_boundary, reference::kOutermostPsiN,
                  "outermost_closed_psi_n");
  expect_relative(diag.q_axis, reference::kQAxis, "q_axis");
  expect_relative(diag.q_95, reference::kQ95, "q_95");
  expect_relative(diag.total_volume, reference::kPlasmaVolume,
                  "plasma_volume_m3");
  expect_relative(diag.r_major, reference::kMajorRadius, "major_radius_m");
  expect_relative(diag.r_minor, reference::kMinorRadius, "minor_radius_m");
  expect_relative(diag.elongation, reference::kElongation, "elongation");
  expect_relative(diag.triangularity, reference::kTriangularity,
                  "triangularity");
}

}  // namespace
