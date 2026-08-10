// Axisymmetric (r,z) ideal-MHD geometric-source contract.
//
// In cylindrical (r,z), m=0 geometry the conservation laws pick up geometric
// source terms from the divergence in curvilinear coordinates (e.g. the
// p_total/r and rho*v_phi^2/r radial source, and the B-tension source). With an
// active split, the stored energy is
// E'=rho*e+|m|^2/(2*rho)+|b|^2/2; it never contains the prescribed B0 baseline.
// The
// solver must (a) construct without throwing for a grid that includes the axis
// (origin_x == 0, i.e. r_min = 0), (b) hold a balanced equilibrium stationary to
// truncation order, (c) keep the CT divergence at machine epsilon in the
// cylindrical discretization, and (d) report a positive, finite cylindrical CFL
// built from the (dr, dz) spacing.
//
// EQUILIBRIUM SEEDED (documented so the implementer's geometric-source math
// targets the same state):
//   A uniform, static, force-free trivial equilibrium:
//     rho   = rho0           (uniform)
//     v     = 0              (no momentum)
//     p     = p0             (uniform ⇒ zero pressure gradient)
//     B_r   = 0, B_phi = 0, B_z = B0  (uniform axial field)
//   With v = 0, B_phi = 0, and dp/dr = 0, the fused tensor residual vanishes:
//   pressure is absent from T_phiphi-T_rr, its ordinary radial derivative is
//   zero, and a uniform axial B_z has neither derivative nor curvature force.
//   Hence the state is a discrete equilibrium and must remain stationary to
//   truncation order after several steps. This is the minimal seed that still
//   exercises the cylindrical flux/tensor path (non-zero B_z, non-zero
//   pressure, axis column included) without requiring a non-trivial profile.
//
// Staggering/conventions assumed (same as the Cartesian tests, with x≡r, y≡z):
//   * grid.origin_x == 0 places the i=0 left edge on the axis (r_at_edge(0)==0).
//   * seed_state writes full-storage host vectors indexed by Grid2D::index(i,j);
//     bx≡B_r on radial faces, by≡B_z on axial faces, bz_cell≡B_phi at centers.
//   * state_component_to_host reads components back over the same layout.
//   * divergence_b_max() uses the cylindrical discrete divergence; for the seeded
//     uniform field it is zero to round-off.
//   * cfl_limit() in cylindrical mode uses dr=grid.dx() and dz=grid.dy().

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_geometric_source.hpp"
#include "quasar/physics/mhd/mhd_solver.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using quasar::Real;
using quasar::Grid2D;

constexpr Real kRho0 = 1.0;
constexpr Real kP0 = 0.5;
constexpr Real kB0 = 0.2;  // uniform axial B_z
constexpr Real kGamma = 5.0 / 3.0;

// (r,z) grid with the axis on the i=0 left edge: origin_x = 0 ⇒ r_at_edge(0)=0.
quasar::mhd::MhdConfig make_cyl_config(
    const std::string& reconstruction = "muscl_minmod") {
  const int nghost = reconstruction == "mp7" ? 4
      : reconstruction == "mp5" ? 3 : 2;
  quasar::Grid2D g{32, 32, /*lx=lr=*/1.0, /*ly=lz=*/1.0,
                   /*origin_x=r_min=*/0.0, /*origin_y=*/0.0, nghost};
  quasar::mhd::MhdConfig cfg;
  cfg.grid = g;
  cfg.geometry = "cylindrical";
  cfg.reconstruction = reconstruction;
  cfg.riemann = "hlld";
  cfg.integrator = "ssprk3";
  cfg.ct = "fd_ct_christlieb";
  cfg.positivity = "troubled_cell";
  for (int s = 0; s < 4; ++s) {
    cfg.boundary.fluid[s] = "outflow";
    cfg.boundary.field[s] = "outflow";
  }
  cfg.boundary.fluid[0] = "axis";
  cfg.boundary.field[0] = "axis";
  return cfg;
}

void seed_uniform_equilibrium(quasar::mhd::MhdSolver2D& solver,
                              const quasar::Grid2D& g) {
  const std::size_t n = g.storage_size();
  std::vector<Real> rho(n, kRho0), mx(n, 0.0), my(n, 0.0), mz(n, 0.0), en(n, 0.0);
  std::vector<Real> bx(n, 0.0), by(n, kB0), bz(n, 0.0);  // B_r=0, B_z=B0, B_phi=0

  const Real magnetic = 0.5 * kB0 * kB0;
  const Real e_cell = kP0 / (kGamma - 1.0) + magnetic;  // v=0 ⇒ no kinetic part
  for (std::size_t k = 0; k < n; ++k) en[k] = e_cell;

  solver.seed_state("rho", rho);
  solver.seed_state("mx", mx);
  solver.seed_state("my", my);
  solver.seed_state("mz", mz);
  solver.seed_state("energy", en);
  solver.seed_state("bx", bx);
  solver.seed_state("by", by);
  solver.seed_state("bz", bz);
}

// Same physical equilibrium, but carry the uniform axial field as static B0
// and seed a zero perturbation. This isolates whether the fused cylindrical
// tensor residual includes the total-field static Maxwell stress while avoiding
// a spurious axial-field curvature force.
void seed_background_equilibrium(quasar::mhd::MhdSolver2D& solver,
                                 const quasar::Grid2D& g) {
  const std::size_t n = g.storage_size();
  std::vector<Real> rho(n, kRho0), zero(n, 0.0);
  std::vector<Real> en(n, kP0 / (kGamma - 1.0));
  solver.seed_state("rho", rho);
  solver.seed_state("mx", zero);
  solver.seed_state("my", zero);
  solver.seed_state("mz", zero);
  solver.seed_state("energy", en);
  solver.seed_state("bx", zero);
  solver.seed_state("by", zero);
  solver.seed_state("bz", zero);
  solver.seed_background("b0x", zero);
  solver.seed_background("b0y", std::vector<Real>(n, kB0));
  solver.seed_background("b0z", zero);
}

Real max_abs_diff(const std::vector<Real>& a, const std::vector<Real>& b,
                  const quasar::Grid2D& g) {
  Real m = 0.0;
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      m = std::max(m, std::abs(a[g.index(i, j)] - b[g.index(i, j)]));
    }
  }
  return m;
}

// Rigidly-rotating, centrifugally-balanced column (pure hydro, uniform rho):
//   v_phi(r) = Omega * r,  v_r = v_z = 0,  B = 0,
//   dp/dr = rho * v_phi^2 / r = rho * Omega^2 * r
//   => p(r) = p0 + 0.5 * rho * Omega^2 * r^2.
// This is an exact equilibrium of the axisymmetric Euler equations: the radial
// pressure gradient is balanced by the centrifugal geometric source
// rho*v_phi^2/r. Unlike the uniform seed, the geometric source is NON-ZERO here
// (it carries the rho*v_phi^2/r term and is balanced against a real dp/dr), so
// this seed exercises the source magnitude/sign rather than its vanishing.
void seed_rotating_equilibrium(quasar::mhd::MhdSolver2D& solver,
                               const quasar::Grid2D& g, Real omega) {
  const std::size_t n = g.storage_size();
  std::vector<Real> rho(n, kRho0), mx(n, 0.0), my(n, 0.0), mz(n, 0.0), en(n, 0.0);
  std::vector<Real> bx(n, 0.0), by(n, 0.0), bz(n, 0.0);  // B = 0 (pure hydro)

  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const std::size_t k = g.index(i, j);
      const Real r = g.r_at_cell_center(i);
      const Real vphi = omega * r;             // rigid rotation
      const Real p = kP0 + 0.5 * kRho0 * omega * omega * r * r;
      mz[k] = kRho0 * vphi;                     // m_phi (azimuthal momentum slot)
      const Real kinetic = 0.5 * kRho0 * vphi * vphi;
      en[k] = p / (kGamma - 1.0) + kinetic;     // B=0 => no magnetic energy
    }
  }
  solver.seed_state("rho", rho);
  solver.seed_state("mx", mx);
  solver.seed_state("my", my);
  solver.seed_state("mz", mz);
  solver.seed_state("energy", en);
  solver.seed_state("bx", bx);
  solver.seed_state("by", by);
  solver.seed_state("bz", bz);
}

// Uniform radial outflow: rho = rho0, v_r = vr0 (> 0), v_z = v_phi = 0, B = 0,
// p = p0. Although the face mass flux is spatially uniform, the annular operator
// -(r_hi F_hi-r_lo F_lo)/int(r dr) gives the physical -rho*v_r/r dilution. This
// exercises the regular ring-volume update in the axis cell, whose center lies
// at r=0.5*dr. It is a flux-divergence check, not a geometric-source check:
// continuity has no separate pointwise cylindrical source in this formulation.
void seed_uniform_radial_outflow(quasar::mhd::MhdSolver2D& solver,
                                 const quasar::Grid2D& g, Real vr0) {
  const std::size_t n = g.storage_size();
  std::vector<Real> rho(n, kRho0), mx(n, kRho0 * vr0), my(n, 0.0), mz(n, 0.0), en(n, 0.0);
  std::vector<Real> bx(n, 0.0), by(n, 0.0), bz(n, 0.0);  // B = 0 (pure hydro)
  const Real kinetic = 0.5 * kRho0 * vr0 * vr0;
  const Real e_cell = kP0 / (kGamma - 1.0) + kinetic;  // B=0 => no magnetic energy
  for (std::size_t k = 0; k < n; ++k) en[k] = e_cell;
  solver.seed_state("rho", rho);
  solver.seed_state("mx", mx);
  solver.seed_state("my", my);
  solver.seed_state("mz", mz);
  solver.seed_state("energy", en);
  solver.seed_state("bx", bx);
  solver.seed_state("by", by);
  solver.seed_state("bz", bz);
}

}  // namespace

// Constructing a cylindrical solver with the axis included (origin_x == 0) must
// not throw — this is a pure-host construction probe.
TEST(MhdCylindricalSource, ConstructsOnAxisWithoutThrowing) {
  const auto cfg = make_cyl_config();
  EXPECT_NO_THROW({ quasar::mhd::MhdSolver2D solver{cfg}; });
}

TEST(MhdCylindricalSource, ConstructsOnAxisWithMp7WithoutThrowing) {
  const auto cfg = make_cyl_config("mp7");
  EXPECT_EQ(cfg.grid.nghost, 4);
  EXPECT_NO_THROW({ quasar::mhd::MhdSolver2D solver{cfg}; });
}

TEST(MhdCylindricalSource, ConstructsFiniteRadiusAnnulusWithBackground) {
  auto cfg = make_cyl_config();
  cfg.grid.origin_x = 0.75;
  cfg.background.enabled = true;
  // An annular inner edge is a physical boundary, not the r=0 parity closure.
  cfg.boundary.fluid[0] = "wall";
  cfg.boundary.field[0] = "wall";
  EXPECT_NO_THROW({ quasar::mhd::MhdSolver2D solver{cfg}; });
}

TEST(MhdCylindricalSource, RejectsNegativeRadiusAndAxisOnAnnulus) {
  auto negative = make_cyl_config();
  negative.grid.origin_x = -0.1;
  EXPECT_THROW({ quasar::mhd::MhdSolver2D solver{negative}; },
               std::invalid_argument);

  auto annulus_axis = make_cyl_config();
  annulus_axis.grid.origin_x = 0.75;
  annulus_axis.boundary.fluid[0] = "axis";
  annulus_axis.boundary.field[0] = "axis";
  EXPECT_THROW({ quasar::mhd::MhdSolver2D solver{annulus_axis}; },
               std::invalid_argument);
}

// The cylindrical CFL limit must be positive and finite and derive from the
// (dr, dz) grid spacing.
TEST(MhdCylindricalSource, CylindricalCflIsPositiveFinite) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const auto cfg = make_cyl_config();
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_uniform_equilibrium(solver, cfg.grid);

  const Real dt_max = solver.cfl_limit();
  EXPECT_GT(dt_max, 0.0);
  EXPECT_TRUE(std::isfinite(dt_max));

  quasar::numerics::MhdPrim w{};
  w.rho = kRho0;
  w.p = kP0;
  w.by = kB0;
  const auto u = quasar::numerics::to_conserved(w, kGamma);
  const Real alpha_r = quasar::numerics::fast_magnetosonic_speed(
      u, /*dir=*/0, kGamma);
  const Real alpha_z = quasar::numerics::fast_magnetosonic_speed(
      u, /*dir=*/1, kGamma);
  // The exact int(r^2 dr) angular-momentum operator is limiting in the axis
  // cell: its high radial face contributes 1.5*alpha_r/dr. The ordinary axial
  // pair contributes alpha_z/dz.
  const Real expected = cfg.cfl /
      (Real{1.5} * alpha_r / cfg.grid.dx() + alpha_z / cfg.grid.dy());
  EXPECT_NEAR(dt_max, expected, Real{3e-13} * expected);
}

TEST(MhdCylindricalSource,
     AngularMomentumFluxTelescopesWithExactCellMoment) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D g{7, 3, Real{1.4}, Real{0.9},
                 Real{0}, Real{0}, /*nghost=*/2};
  quasar::mhd::MhdField2D<Real> flux{g};
  quasar::mhd::MhdField2D<Real> residual{g};
  const std::size_t n = g.storage_size();
  const std::vector<Real> zero(n, Real{0});
  std::vector<Real> angular_flux(n, Real{0});
  for (int j = 0; j < g.ny; ++j) {
    for (int face = 1; face < g.nx; ++face) {
      angular_flux[g.index(face, j)] =
          Real{0.17} * static_cast<Real>((face + 2) * (j + 1));
    }
    // Closed radial boundaries: face 0 and face nx remain exactly zero.
  }

  flux.rho.copy_from_host(zero.data(), n);
  flux.mx.copy_from_host(zero.data(), n);
  flux.my.copy_from_host(zero.data(), n);
  flux.mz.copy_from_host(angular_flux.data(), n);
  flux.energy.copy_from_host(zero.data(), n);
  flux.bx_face.copy_from_host(zero.data(), n);
  flux.by_face.copy_from_host(zero.data(), n);
  flux.bz_cell.copy_from_host(zero.data(), n);
  residual.rho.copy_from_host(zero.data(), n);
  residual.mx.copy_from_host(zero.data(), n);
  residual.my.copy_from_host(zero.data(), n);
  residual.mz.copy_from_host(zero.data(), n);
  residual.energy.copy_from_host(zero.data(), n);
  residual.bx_face.copy_from_host(zero.data(), n);
  residual.by_face.copy_from_host(zero.data(), n);
  residual.bz_cell.copy_from_host(zero.data(), n);

  quasar::mhd::launch_mhd_flux_difference(
      flux, /*dir=*/0, residual, nullptr, /*cylindrical=*/true);
  quasar::backend::device_synchronize(nullptr);
  std::vector<Real> dmphi(n);
  residual.mz.copy_to_host(dmphi.data(), n);

  const Real dr = g.dx();
  for (int j = 0; j < g.ny; ++j) {
    long double global_rate = 0.0L;
    for (int i = 0; i < g.nx; ++i) {
      const long double rc = static_cast<long double>(
          g.r_at_cell_center(i));
      const long double dr_ld = static_cast<long double>(dr);
      const long double moment =
          dr_ld * (rc * rc + dr_ld * dr_ld / 12.0L);
      global_rate += moment * static_cast<long double>(
          dmphi[g.index(i, j)]);
    }
    EXPECT_NEAR(static_cast<Real>(global_rate), Real{0}, Real{3e-15});

    // r_lo=0 gives W=dr^3/3, hence dm_phi/dt=-3 F_hi/dr in the
    // first cell. This pins the regular axis coefficient explicitly.
    const Real expected_axis =
        -Real{3} * angular_flux[g.index(1, j)] / dr;
    EXPECT_NEAR(dmphi[g.index(0, j)], expected_axis,
                Real{3e-15} * std::abs(expected_axis));
  }
}

TEST(MhdCylindricalSource, ThinLargeRadiusDivergenceAvoidsCoordinateProducts) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  auto cfg = make_cyl_config();
  cfg.grid = quasar::Grid2D{
      8, 4, Real{8e136}, Real{1}, Real{1e150}, Real{0}, /*nghost=*/2};
  cfg.boundary.fluid[0] = "outflow";
  cfg.boundary.field[0] = "outflow";
  const auto& g = cfg.grid;
  const std::size_t n = g.storage_size();
  const Real br = Real{1e170};
  ASSERT_FALSE(std::isfinite(g.origin_x * br));

  quasar::mhd::MhdSolver2D solver{cfg};
  const std::vector<Real> one(n, Real{1});
  const std::vector<Real> zero(n, Real{0});
  const std::vector<Real> radial_field(n, br);
  solver.seed_state("rho", one);
  solver.seed_state("mx", zero);
  solver.seed_state("my", zero);
  solver.seed_state("mz", zero);
  solver.seed_state("energy", one);
  solver.seed_state("bx_face", radial_field);
  solver.seed_state("by_face", zero);
  solver.seed_state("bz", zero);

  const Real got = solver.divergence_b_max();
  const Real expected = br / g.r_at_cell_center(0);
  ASSERT_TRUE(std::isfinite(got));
  ASSERT_TRUE(std::isfinite(expected));
  EXPECT_NEAR(got / expected, Real{1}, Real{2e-14});
}

TEST(MhdCylindricalSource, StrongToroidalCurvatureAvoidsSquareOverflow) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const quasar::Grid2D g{4, 3, Real{1}, Real{1}, Real{1}, Real{0}, 2};
  quasar::mhd::MhdField2D<Real> state{g};
  quasar::mhd::MhdField2D<Real> residual{g};
  quasar::mhd::MhdBackgroundField<Real> inactive_background{};
  const std::size_t n = g.storage_size();
  const Real bphi = Real{1.5e154};
  ASSERT_FALSE(std::isfinite(bphi * bphi));
  const Real magnetic = quasar::numerics::half_squared_norm3(
      Real{0}, Real{0}, bphi);
  ASSERT_TRUE(std::isfinite(magnetic));

  const std::vector<Real> rho(n, Real{1});
  const std::vector<Real> zero(n, Real{0});
  const std::vector<Real> energy(n, magnetic);
  const std::vector<Real> toroidal(n, bphi);
  state.rho.copy_from_host(rho.data(), n);
  state.mx.copy_from_host(zero.data(), n);
  state.my.copy_from_host(zero.data(), n);
  state.mz.copy_from_host(zero.data(), n);
  state.energy.copy_from_host(energy.data(), n);
  state.bx_face.copy_from_host(zero.data(), n);
  state.by_face.copy_from_host(zero.data(), n);
  state.bz_cell.copy_from_host(toroidal.data(), n);
  residual.rho.copy_from_host(zero.data(), n);
  residual.mx.copy_from_host(zero.data(), n);
  residual.my.copy_from_host(zero.data(), n);
  residual.mz.copy_from_host(zero.data(), n);
  residual.energy.copy_from_host(zero.data(), n);
  residual.bx_face.copy_from_host(zero.data(), n);
  residual.by_face.copy_from_host(zero.data(), n);
  residual.bz_cell.copy_from_host(zero.data(), n);

  quasar::mhd::MhdGeometricSource::add(
      state, residual, inactive_background, g, kGamma);
  quasar::backend::device_synchronize(nullptr);
  std::vector<Real> radial(n);
  residual.mx.copy_to_host(radial.data(), n);

  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const Real r = g.r_at_cell_center(i);
      const long double expected_ld =
          -(0.5L * static_cast<long double>(bphi) *
            static_cast<long double>(bphi)) / static_cast<long double>(r);
      const Real expected = static_cast<Real>(expected_ld);
      ASSERT_TRUE(std::isfinite(expected));
      ASSERT_TRUE(std::isfinite(radial[g.index(i, j)]));
      EXPECT_NEAR(radial[g.index(i, j)] / expected, Real{1}, Real{3e-15});
    }
  }
}

TEST(MhdCylindricalSource,
     DominantRotationToroidalBalanceRetainsRadialFieldSurvivor) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const Grid2D g{4, 3, Real{1}, Real{1}, Real{1}, Real{0}, 2};
  quasar::mhd::MhdField2D<Real> state{g};
  quasar::mhd::MhdField2D<Real> radial_flux{g};
  quasar::mhd::MhdField2D<Real> axial_flux{g};
  quasar::mhd::MhdField2D<Real> residual{g};
  quasar::mhd::MhdBackgroundField<Real> inactive_background{};
  const std::size_t n = g.storage_size();
  const Real dominant = Real{1.5e154};
  ASSERT_FALSE(std::isfinite(dominant * dominant));
  const Real constant_stress = quasar::numerics::half_squared_norm3(
      Real{0}, Real{0}, dominant);
  ASSERT_TRUE(std::isfinite(constant_stress));

  const std::vector<Real> one(n, Real{1});
  const std::vector<Real> zero(n, Real{0});
  const std::vector<Real> large(n, dominant);
  const std::vector<Real> stress(n, constant_stress);
  state.rho.copy_from_host(one.data(), n);
  state.mx.copy_from_host(zero.data(), n);
  state.mz.copy_from_host(large.data(), n);
  state.bx_face.copy_from_host(one.data(), n);
  state.bz_cell.copy_from_host(large.data(), n);
  radial_flux.mx.copy_from_host(stress.data(), n);
  axial_flux.mx.copy_from_host(zero.data(), n);
  residual.mx.copy_from_host(zero.data(), n);

  quasar::mhd::BoundaryFlags4 flags{{1, 1, 1, 1}};
  quasar::mhd::launch_mhd_cylindrical_radial_momentum_residual(
      state, inactive_background, radial_flux, axial_flux, residual,
      flags, nullptr, kGamma, /*collocation_order=*/1);
  quasar::backend::device_synchronize(nullptr);

  std::vector<Real> radial_rate(n);
  residual.mx.copy_to_host(radial_rate.data(), n);
  for (int j = 0; j < g.ny; ++j) {
    for (int i = 0; i < g.nx; ++i) {
      const Real r = g.r_at_cell_center(i);
      // T_phiphi-T_rr = mphi^2/rho + Br^2-Bphi^2 = 1. The two
      // individually overflowing O(dominant^2) terms must cancel before the
      // finite Br^2 survivor is rounded. An annular aggregate flux plus a
      // separately rounded curvature source loses this survivor.
      ASSERT_TRUE(std::isfinite(radial_rate[g.index(i, j)]));
      EXPECT_NEAR(radial_rate[g.index(i, j)] * r, Real{1}, Real{4e-15});
    }
  }
}

// The uniform, radially-balanced equilibrium must stay stationary to truncation
// order: with v=0, B_phi=0 and dp/dr=0 the geometric source vanishes, so the
// state should barely move after several steps.
TEST(MhdCylindricalSource, BalancedEquilibriumStaysStationary) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const auto cfg = make_cyl_config();
  const auto& g = cfg.grid;
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_uniform_equilibrium(solver, g);

  const std::vector<Real> rho0 = solver.state_component_to_host("rho");
  const std::vector<Real> en0 = solver.state_component_to_host("energy");

  const Real dt = 0.4 * solver.cfl_limit();
  for (int s = 0; s < 6; ++s) solver.step(dt);

  const std::vector<Real> rho1 = solver.state_component_to_host("rho");
  const std::vector<Real> en1 = solver.state_component_to_host("energy");

  // Equilibrium ⇒ the state holds to truncation order. The seed is uniform, so a
  // correctly-cancelling geometric source leaves it essentially unchanged.
  EXPECT_LT(max_abs_diff(rho1, rho0, g), 1e-8);
  EXPECT_LT(max_abs_diff(en1, en0, g), 1e-8);
}

TEST(MhdCylindricalSource, BackgroundMagneticPressureBalancesOnAnnulus) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  auto cfg = make_cyl_config();
  cfg.grid.origin_x = 0.75;
  cfg.background.enabled = true;
  cfg.boundary.fluid[0] = "wall";
  cfg.boundary.field[0] = "wall";
  const auto& g = cfg.grid;
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_background_equilibrium(solver, g);

  const auto mr0 = solver.state_component_to_host("mx");
  const Real dt = 0.2 * solver.cfl_limit();
  solver.step(dt);
  const auto mr1 = solver.state_component_to_host("mx");

  // Exclude the wall closure; the uniform total-pressure state is an exact
  // annular flux/source balance in the deep interior.
  Real drift = 0.0;
  for (int j = 4; j < g.ny - 4; ++j) {
    for (int i = 4; i < g.nx - 4; ++i) {
      drift = std::max(drift,
                       std::abs(mr1[g.index(i, j)] - mr0[g.index(i, j)]));
    }
  }
  EXPECT_LT(drift, 1e-9);
}

// A rigidly-rotating, centrifugally-balanced column must hold near-stationary in
// the deep interior to truncation order. The radial pressure gradient is exactly
// balanced by the rho*v_phi^2/r geometric source, so a CORRECT source keeps the
// central momentum from drifting. With the centrifugal term missing (or wrong
// sign/coefficient) the unbalanced dp/dr would drive an O(1) radial momentum in
// a single step; we assert the interior radial momentum stays tiny relative to
// that pressure-gradient scale. Restricting to the interior columns avoids the
// outflow-boundary mismatch (the seed's r-dependent profile is not zero-gradient
// at the wall, so the boundary ring is expected to move).
TEST(MhdCylindricalSource, RotatingEquilibriumHoldsInInterior) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const auto cfg = make_cyl_config();
  const auto& g = cfg.grid;
  quasar::mhd::MhdSolver2D solver{cfg};
  const Real omega = 0.3;
  seed_rotating_equilibrium(solver, g, omega);

  const std::vector<Real> mr0 = solver.state_component_to_host("mx");

  const Real dt = 0.2 * solver.cfl_limit();
  solver.step(dt);

  const std::vector<Real> mr1 = solver.state_component_to_host("mx");

  // Scale of the radial momentum an UNBALANCED dp/dr would inject in one step:
  // |dp/dr| * dt ~ rho*omega^2*r * dt at mid-radius. The balanced source must
  // keep the actual interior drift well below this.
  const Real r_mid = g.r_at_cell_center(g.nx / 2);
  const Real unbalanced_scale = kRho0 * omega * omega * r_mid * dt;
  EXPECT_GT(unbalanced_scale, 0.0);

  // Deep-interior window (exclude the outer/inner few columns near r=0 and the
  // wall, where the boundary closure and on-axis guard legitimately move it).
  Real max_interior_drift = 0.0;
  const int lo = g.nx / 4, hi = 3 * g.nx / 4;
  for (int j = g.ny / 4; j < 3 * g.ny / 4; ++j) {
    for (int i = lo; i < hi; ++i) {
      max_interior_drift = std::max(
          max_interior_drift, std::abs(mr1[g.index(i, j)] - mr0[g.index(i, j)]));
    }
  }
  // A correct centrifugal source keeps the drift to a small fraction of the
  // unbalanced scale (finite-volume truncation error), not O(1) of it.
  EXPECT_LT(max_interior_drift, 0.1 * unbalanced_scale);
}

// The on-axis annular control volume (i=0, r_center=0.5*dr) must receive the
// ring-weighted radial flux divergence. For uniform F_rho=rho*v_r, the exact
// finite-volume rate is -rho*v_r/r_center. A Cartesian difference or an axis
// guard would leave this column frozen. Check its sign and magnitude.
TEST(MhdCylindricalSource, AxisColumnReceivesRingFluxDivergence) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const auto cfg = make_cyl_config();
  const auto& g = cfg.grid;
  quasar::mhd::MhdSolver2D solver{cfg};
  const Real vr0 = 0.1;
  seed_uniform_radial_outflow(solver, g, vr0);

  const std::vector<Real> rho0 = solver.state_component_to_host("rho");
  const Real dt = 0.2 * solver.cfl_limit();
  solver.step(dt);
  const std::vector<Real> rho1 = solver.state_component_to_host("rho");

  // Sample a mid-z row to avoid any z-boundary influence; column i=0 is the axis.
  const int j = g.ny / 2;
  const Real r_axis = g.r_at_cell_center(0);  // 0.5 * dr
  const Real expected_axis = -kRho0 * vr0 / r_axis * dt;
  const Real actual_axis = rho1[g.index(0, j)] - rho0[g.index(0, j)];

  // The axis cell must not be frozen: a Cartesian radial difference gives zero.
  EXPECT_GT(std::abs(actual_axis), 0.2 * std::abs(expected_axis));
  // Mass drains radially outward, so density drops at the axis: sign must match.
  EXPECT_LT(actual_axis, 0.0);
  // Order-of-magnitude band around the analytic ring divergence: the axis BC
  // stencil makes the flux difference at i=0 not exactly zero, so this is a
  // factor-of-a-few check, not an exact equality.
  EXPECT_LT(std::abs(actual_axis), 3.0 * std::abs(expected_axis));
}

// CT divergence stays at machine epsilon in the cylindrical discretization for
// the uniform (divergence-free) field.
TEST(MhdCylindricalSource, DivergenceStaysAtMachineEpsilon) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";
  const auto cfg = make_cyl_config();
  quasar::mhd::MhdSolver2D solver{cfg};
  seed_uniform_equilibrium(solver, cfg.grid);

  EXPECT_LT(solver.divergence_b_max(), 1e-10);

  const Real dt = 0.4 * solver.cfl_limit();
  for (int s = 0; s < 6; ++s) solver.step(dt);

  EXPECT_LT(solver.divergence_b_max(), 1e-9);
}
