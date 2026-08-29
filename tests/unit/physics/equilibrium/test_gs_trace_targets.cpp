// Exact-target device flux-surface tracing.
//
// The stability discretization samples equilibrium geometry on nonuniform
// radial grids, including mapped Chebyshev nodes. Reconstructing those targets
// from a uniform surface index changes the collocation problem, so this test
// pins the device contract: caller-supplied psi_N values are traced directly
// and recorded without numerical modification.

#include "quasar/backend/memory.hpp"
#include "quasar/physics/equilibrium/critical_points.hpp"
#include "quasar/physics/equilibrium/equilibrium_profile.hpp"
#include "quasar/physics/equilibrium/gs_solver.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

using quasar::Real;
using quasar::backend::DeviceBuffer;
using quasar::equilibrium::CoilFilament;
using quasar::equilibrium::GsConfig;
using quasar::equilibrium::GsDeviceResult;
using quasar::equilibrium::GsSolver;
using quasar::equilibrium::GsStatus;
using quasar::equilibrium::PolynomialProfile;
using quasar::numerics::EllipticGrid;

constexpr int kFSamples = 257;
constexpr int kNTheta = 128;
constexpr Real kFVacuum = Real{5};
constexpr Real kPi = Real{3.14159265358979323846};

struct KnownEquilibrium {
  EllipticGrid grid{33, 33, Real{0.3}, Real{1.9}, Real{-0.8}, Real{0.8}};
  PolynomialProfile profile{std::vector<Real>{Real{1}, Real{-1}},
                            std::vector<Real>{Real{1}, Real{-1}}};
  GsDeviceResult result{};
  DeviceBuffer<Real> f_table{};
  quasar::equilibrium::GsOperatorScratch op{};
  quasar::equilibrium::GsDerivativeFields derivatives{};
  quasar::equilibrium::GsMagneticField field{};

  KnownEquilibrium() {
    GsConfig cfg;
    cfg.grid = grid;
    cfg.coils = {
        CoilFilament{Real{2.4}, Real{0.9}, Real{-3.0e5}},
        CoilFilament{Real{2.4}, Real{-0.9}, Real{-3.0e5}},
        CoilFilament{Real{0.28}, Real{0.0}, Real{1.0e5}},
    };
    cfg.plasma_current = Real{1e6};
    cfg.max_iterations = 400;
    cfg.tolerance = Real{1e-9};
    result = GsSolver{cfg, std::make_shared<PolynomialProfile>(profile)}
                 .solve_device();
    if (!result.ok()) return;

    f_table = DeviceBuffer<Real>{static_cast<std::size_t>(kFSamples)};
    quasar::equilibrium::launch_gs_integrate_f_profile(
        quasar::equilibrium::to_coefficients(profile), kFVacuum,
        result.critical.psi_axis, result.critical.psi_boundary,
        result.profile_scale, kFSamples, f_table.device_ptr(), nullptr);

    op.resize(grid);
    derivatives.resize(grid);
    quasar::equilibrium::launch_gs_compute_derivatives(
        grid, result.psi.device_ptr(), derivatives, op, nullptr);
    field.resize(grid);
    quasar::equilibrium::launch_gs_compute_field(
        grid, result.psi.device_ptr(), derivatives, f_table.device_ptr(),
        kFSamples, result.critical.psi_axis, result.critical.psi_boundary,
        field, nullptr);
    quasar::backend::device_synchronize(nullptr);
  }
};

const KnownEquilibrium& known_equilibrium() {
  static const KnownEquilibrium equilibrium;
  return equilibrium;
}

template <class T>
std::vector<T> download(const DeviceBuffer<T>& buffer) {
  std::vector<T> host(buffer.size());
  buffer.copy_to_host(host.data(), host.size());
  return host;
}

std::vector<Real> bulk_chebyshev_targets() {
  constexpr int count = 5;
  constexpr Real lo = Real{0.18};
  constexpr Real hi = Real{0.78};
  const Real midpoint = Real{0.5} * (lo + hi);
  const Real half_width = Real{0.5} * (hi - lo);

  std::vector<Real> targets(count);
  for (int s = 0; s < count; ++s) {
    // Chebyshev-Gauss-Lobatto nodes mapped to a bulk interval, ordered from
    // magnetic axis toward the separatrix. The truncated interval avoids both
    // singular limits while retaining the deliberately nonuniform spacing.
    const Real x = std::cos(kPi * static_cast<Real>(count - 1 - s)
                            / static_cast<Real>(count - 1));
    targets[static_cast<std::size_t>(s)] = midpoint + half_width * x;
  }
  return targets;
}

TEST(GsTraceTargets, TracesAndRecordsNonuniformNormalizedFluxExactly) {
  const KnownEquilibrium& fx = known_equilibrium();
  ASSERT_EQ(fx.result.status, GsStatus::converged);
  ASSERT_TRUE(fx.result.critical.axis.valid);

  const std::vector<Real> targets = bulk_chebyshev_targets();
  DeviceBuffer<Real> d_targets{targets.size()};
  d_targets.copy_from_host(targets.data(), targets.size());

  quasar::equilibrium::GsFluxSurfaces surfaces{
      static_cast<int>(targets.size()), kNTheta};
  quasar::equilibrium::launch_gs_trace_surfaces_at(
      fx.grid, fx.result.psi.device_ptr(), fx.field,
      fx.result.critical.axis.r, fx.result.critical.axis.z,
      fx.result.critical.psi_axis, fx.result.critical.psi_boundary, d_targets,
      surfaces, nullptr);
  quasar::backend::device_synchronize(nullptr);

  const std::vector<Real> traced_targets = download(surfaces.psi_n);
  const std::vector<Real> hit_r = download(surfaces.r);
  const std::vector<Real> hit_z = download(surfaces.z);
  const std::vector<int> counts = download(surfaces.count);
  const std::vector<int> closed = download(surfaces.closed);
  const std::vector<Real> psi = download(fx.result.psi);

  const Real flux_span =
      fx.result.critical.psi_boundary - fx.result.critical.psi_axis;
  ASSERT_NE(flux_span, Real{0});

  // The hit is a secant interpolation across one ray-march interval. Inside a
  // grid cell the bilinear psi interpolant is quadratic along a ray, so the
  // normalized-flux residual is O((ds/h)^2), where h is the smaller grid
  // spacing. Here ds = 1.6/4000 = 4e-4, h = 0.05, and that scale is 6.4e-5.
  // The measured worst residual is 6.15585e-6 (outer target, rays 57 and 71),
  // with every other target/ray below 5e-6. Using the method scale therefore
  // gives 10.4x measured margin while remaining over 200x smaller than the
  // nearest difference between this target set and the legacy uniform grid.
  constexpr Real kRayMarchSteps = Real{4000};
  const Real ray_step =
      std::max(fx.grid.r_max - fx.grid.r_min,
               fx.grid.z_max - fx.grid.z_min)
      / kRayMarchSteps;
  const Real cell_scale = std::min(fx.grid.dr(), fx.grid.dz());
  const Real kTracingTolerance =
      (ray_step / cell_scale) * (ray_step / cell_scale);

  Real worst_residual = Real{0};
  std::size_t worst_surface = 0;
  int worst_ray = 0;

  for (std::size_t s = 0; s < targets.size(); ++s) {
    EXPECT_EQ(traced_targets[s], targets[s]) << "surface " << s;
    ASSERT_EQ(closed[s], 1) << "surface " << s;
    ASSERT_EQ(counts[s], kNTheta) << "surface " << s;

    const std::size_t base = s * static_cast<std::size_t>(kNTheta);
    for (int t = 0; t < counts[s]; ++t) {
      const std::size_t k = base + static_cast<std::size_t>(t);
      const Real sampled = quasar::equilibrium::sample_bilinear(
          fx.grid, psi, hit_r[k], hit_z[k]);
      const Real sampled_psi_n =
          (sampled - fx.result.critical.psi_axis) / flux_span;
      ASSERT_TRUE(std::isfinite(sampled_psi_n))
          << "surface " << s << ", ray " << t;
      const Real residual = std::abs(sampled_psi_n - targets[s]);
      if (residual > worst_residual) {
        worst_residual = residual;
        worst_surface = s;
        worst_ray = t;
      }
    }
  }

  EXPECT_LE(worst_residual, kTracingTolerance)
      << "worst sampled psi_N residual at surface " << worst_surface
      << ", ray " << worst_ray;
}

TEST(GsTraceTargets, LegacyEntryPointRetainsUniformTargetGrid) {
  const KnownEquilibrium& fx = known_equilibrium();
  ASSERT_EQ(fx.result.status, GsStatus::converged);

  constexpr int n_surfaces = 7;
  quasar::equilibrium::GsFluxSurfaces surfaces{n_surfaces, kNTheta};
  quasar::equilibrium::launch_gs_trace_surfaces(
      fx.grid, fx.result.psi.device_ptr(), fx.field,
      fx.result.critical.axis.r, fx.result.critical.axis.z,
      fx.result.critical.psi_axis, fx.result.critical.psi_boundary, surfaces,
      nullptr);
  quasar::backend::device_synchronize(nullptr);

  const std::vector<Real> psi_n = download(surfaces.psi_n);
  for (int s = 0; s < n_surfaces; ++s) {
    const Real expected =
        static_cast<Real>(s + 1) / static_cast<Real>(n_surfaces + 1);
    EXPECT_EQ(psi_n[static_cast<std::size_t>(s)], expected)
        << "surface " << s;
  }
}

TEST(GsTraceTargets, RejectsTargetAndOutputSizeMismatches) {
  const KnownEquilibrium& fx = known_equilibrium();
  ASSERT_EQ(fx.result.status, GsStatus::converged);

  constexpr int n_surfaces = 5;
  DeviceBuffer<Real> short_targets{n_surfaces - 1};
  quasar::equilibrium::GsFluxSurfaces surfaces{n_surfaces, kNTheta};
  EXPECT_THROW(
      quasar::equilibrium::launch_gs_trace_surfaces_at(
          fx.grid, fx.result.psi.device_ptr(), fx.field,
          fx.result.critical.axis.r, fx.result.critical.axis.z,
          fx.result.critical.psi_axis, fx.result.critical.psi_boundary,
          short_targets, surfaces, nullptr),
      std::invalid_argument);

  DeviceBuffer<Real> targets{n_surfaces};
  surfaces.psi_n = DeviceBuffer<Real>{n_surfaces - 1};
  EXPECT_THROW(
      quasar::equilibrium::launch_gs_trace_surfaces_at(
          fx.grid, fx.result.psi.device_ptr(), fx.field,
          fx.result.critical.axis.r, fx.result.critical.axis.z,
          fx.result.critical.psi_axis, fx.result.critical.psi_boundary, targets,
          surfaces, nullptr),
      std::invalid_argument);
}

}  // namespace
