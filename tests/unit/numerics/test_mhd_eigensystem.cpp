// Tests for the ideal-MHD eigensystem.
//
// Targets the blind contract in include/quasar/numerics/mhd_eigensystem.hpp:
//   class MhdEigensystem {
//    public:
//     void build(const MhdState&, int dir, Real gamma);
//     // implementation-defined accessors for the left/right eigenvectors and
//     // wave speeds.
//   };
//
// The exact accessor names are implementation-defined. This test ASSUMES the
// concrete accessor shape spelled out in the contract:
//   const Real* left_row(int k);   // length-7 row of L, k in [0,7)
//   const Real* right_col(int k);  // length-7 column of R, k in [0,7)
//   Real        wave_speed(int k); // k ordered fast-, alfven-, slow-,
//                                  // entropy/contact, slow+, alfven+, fast+
// The implementer must match these accessor names so these property tests bind.
//
// We test OBSERVABLE mathematical invariants:
//   (1) L * R == Identity (biorthonormality of left/right eigenvectors).
//   (2) R * diag(wave_speeds) * L applied to a small perturbation reproduces a
//       finite-difference of the analytic conserved-variable MHD flux
//       (the decomposition diagonalizes the flux Jacobian).
//   (3) Degenerate states (B->0 and B_perp->0) give FINITE eigenvectors that
//       still satisfy L*R == I within a loosened tolerance.

#include "quasar/numerics/mhd_eigensystem.hpp"
#include "quasar/numerics/mhd_state.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace {

using quasar::Real;
using quasar::numerics::MhdEigensystem;
using quasar::numerics::MhdPrim;
using quasar::numerics::MhdState;

// Number of MHD waves / size of the eigen-basis used by these property checks.
constexpr int kN = 7;

// Tolerances. The biorthonormality and diagonalization tolerances are loose
// enough to absorb finite-difference truncation but tight enough to fail a
// wrong decomposition.
constexpr Real kIdentityTol = 1e-9;
constexpr Real kIdentityTolDegenerate = 1e-6;
constexpr Real kJacobianTol = 1e-5;

// The conserved MHD state laid out as a length-8 vector. Only the 7
// non-divergence-constrained components participate in the dir-normal 7-wave
// eigen-basis: for dir=0 the normal field component Bx is constant across the
// wave fan, so the active variables are
//   [rho, mx, my, mz, energy, by, bz].
// We expose helpers to pack/unpack those 7 active variables.
void pack7_x(const MhdState& s, Real v[kN]) {
  v[0] = s.rho;
  v[1] = s.mx;
  v[2] = s.my;
  v[3] = s.mz;
  v[4] = s.energy;
  v[5] = s.by;
  v[6] = s.bz;
}

MhdState unpack7_x(const Real v[kN], Real bx) {
  MhdState s{};
  s.rho = v[0];
  s.mx = v[1];
  s.my = v[2];
  s.mz = v[3];
  s.energy = v[4];
  s.bx = bx;
  s.by = v[5];
  s.bz = v[6];
  return s;
}

// y-normal (dir=1) packing: the normal field component By is constant across the
// wave fan, so the active variables are [rho, mx, my, mz, energy, bx, bz].
void pack7_y(const MhdState& s, Real v[kN]) {
  v[0] = s.rho;
  v[1] = s.mx;
  v[2] = s.my;
  v[3] = s.mz;
  v[4] = s.energy;
  v[5] = s.bx;
  v[6] = s.bz;
}

MhdState unpack7_y(const Real v[kN], Real by) {
  MhdState s{};
  s.rho = v[0];
  s.mx = v[1];
  s.my = v[2];
  s.mz = v[3];
  s.energy = v[4];
  s.bx = v[5];
  s.by = by;
  s.bz = v[6];
  return s;
}

// Analytic conserved-variable ideal-MHD flux in the x-direction. Local helper
// for the finite-difference Jacobian; does not call any library internals.
//   F = [ rho*vx,
//         rho*vx^2 + p_total - Bx^2,
//         rho*vx*vy - Bx*By,
//         rho*vx*vz - Bx*Bz,
//         (E + p_total)*vx - Bx*(v.B),
//         By*vx - Bx*vy,
//         Bz*vx - Bx*vz ]
// with p_total = p_gas + 0.5*|B|^2, p_gas = (gamma-1)*(E - 0.5*rho*|v|^2 - 0.5*|B|^2).
void flux_x(const MhdState& s, Real gamma, Real f[kN]) {
  const Real rho = s.rho;
  const Real vx = s.mx / rho;
  const Real vy = s.my / rho;
  const Real vz = s.mz / rho;
  const Real bx = s.bx;
  const Real by = s.by;
  const Real bz = s.bz;
  const Real b2 = bx * bx + by * by + bz * bz;
  const Real v2 = vx * vx + vy * vy + vz * vz;
  const Real vdotb = vx * bx + vy * by + vz * bz;
  const Real p_gas = (gamma - 1.0) * (s.energy - 0.5 * rho * v2 - 0.5 * b2);
  const Real p_tot = p_gas + 0.5 * b2;

  f[0] = rho * vx;
  f[1] = rho * vx * vx + p_tot - bx * bx;
  f[2] = rho * vx * vy - bx * by;
  f[3] = rho * vx * vz - bx * bz;
  f[4] = (s.energy + p_tot) * vx - bx * vdotb;
  f[5] = by * vx - bx * vy;
  f[6] = bz * vx - bx * vz;
}

// Analytic conserved-variable ideal-MHD flux in the y-direction, packed in the
// dir=1 active-variable order [rho, mx, my, mz, energy, bx, bz] (By constant
// across the y-normal wave fan).
//   F_y = [ rho*vy,
//           rho*vy*vx - By*Bx,
//           rho*vy^2 + p_total - By^2,
//           rho*vy*vz - By*Bz,
//           (E + p_total)*vy - By*(v.B),
//           Bx*vy - By*vx,
//           Bz*vy - By*vz ]
void flux_y(const MhdState& s, Real gamma, Real f[kN]) {
  const Real rho = s.rho;
  const Real vx = s.mx / rho;
  const Real vy = s.my / rho;
  const Real vz = s.mz / rho;
  const Real bx = s.bx;
  const Real by = s.by;
  const Real bz = s.bz;
  const Real b2 = bx * bx + by * by + bz * bz;
  const Real v2 = vx * vx + vy * vy + vz * vz;
  const Real vdotb = vx * bx + vy * by + vz * bz;
  const Real p_gas = (gamma - 1.0) * (s.energy - 0.5 * rho * v2 - 0.5 * b2);
  const Real p_tot = p_gas + 0.5 * b2;

  f[0] = rho * vy;
  f[1] = rho * vy * vx - by * bx;
  f[2] = rho * vy * vy + p_tot - by * by;
  f[3] = rho * vy * vz - by * bz;
  f[4] = (s.energy + p_tot) * vy - by * vdotb;
  f[5] = bx * vy - by * vx;
  f[6] = bz * vy - by * vz;
}

// A representative, well-separated magnetized state (no degeneracies).
MhdState make_reference_state(Real gamma) {
  const Real rho = 1.1;
  const Real vx = 0.2, vy = -0.15, vz = 0.3;
  const Real p = 1.4;
  const Real bx = 0.55, by = 0.8, bz = -0.4;
  const Real e_int = p / ((gamma - 1.0) * rho);
  MhdState s{};
  s.rho = rho;
  s.mx = rho * vx;
  s.my = rho * vy;
  s.mz = rho * vz;
  s.energy = rho * e_int + 0.5 * rho * (vx * vx + vy * vy + vz * vz) +
             0.5 * (bx * bx + by * by + bz * bz);
  s.bx = bx;
  s.by = by;
  s.bz = bz;
  return s;
}

bool any_nonfinite(const Real* p, int n) {
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(p[i])) return true;
  }
  return false;
}

}  // namespace

// (1) L * R == Identity for a representative magnetized state.
TEST(MhdEigensystem, LeftRightVectorsAreBiorthonormal) {
  const Real gamma = 5.0 / 3.0;
  const MhdState s = make_reference_state(gamma);

  MhdEigensystem eig;
  eig.build(s, /*dir=*/0, gamma);

  for (int i = 0; i < kN; ++i) {
    const Real* li = eig.left_row(i);
    ASSERT_NE(li, nullptr);
    ASSERT_FALSE(any_nonfinite(li, kN));
    for (int j = 0; j < kN; ++j) {
      const Real* rj = eig.right_col(j);
      ASSERT_NE(rj, nullptr);
      Real dot = 0.0;
      for (int k = 0; k < kN; ++k) dot += li[k] * rj[k];
      const Real expected = (i == j) ? 1.0 : 0.0;
      EXPECT_NEAR(dot, expected, kIdentityTol)
          << "L row " << i << " . R col " << j;
    }
  }
}

// (2) Diagonalization: R * diag(lambda) * L * du reproduces the finite-difference
// of the analytic flux, A*du, where A = dF/dU is the flux Jacobian. Equivalent
// to checking the eigen-decomposition reconstructs the Jacobian.
TEST(MhdEigensystem, DiagonalizesFluxJacobian) {
  const Real gamma = 5.0 / 3.0;
  const MhdState s = make_reference_state(gamma);
  const Real bx = s.bx;

  MhdEigensystem eig;
  eig.build(s, /*dir=*/0, gamma);

  Real lambda[kN];
  for (int k = 0; k < kN; ++k) lambda[k] = eig.wave_speed(k);

  // Base active-variable vector and base flux.
  Real u0[kN];
  pack7_x(s, u0);
  Real f0[kN];
  flux_x(s, gamma, f0);

  // A few independent perturbation directions.
  const Real eps = 1e-6;
  for (int comp = 0; comp < kN; ++comp) {
    Real du[kN] = {0, 0, 0, 0, 0, 0, 0};
    // Scale the perturbation to the magnitude of the component for stability.
    const Real scale = (std::abs(u0[comp]) > 1.0 ? std::abs(u0[comp]) : 1.0);
    du[comp] = eps * scale;

    // Finite-difference of the analytic flux: A*du ~= F(U+du) - F(U).
    Real up[kN];
    for (int k = 0; k < kN; ++k) up[k] = u0[k] + du[k];
    const MhdState sp = unpack7_x(up, bx);
    Real fp[kN];
    flux_x(sp, gamma, fp);
    Real fd[kN];
    for (int k = 0; k < kN; ++k) fd[k] = fp[k] - f0[k];

    // Reconstructed: R * diag(lambda) * (L * du).
    Real ldu[kN];
    for (int k = 0; k < kN; ++k) {
      const Real* lk = eig.left_row(k);
      Real acc = 0.0;
      for (int m = 0; m < kN; ++m) acc += lk[m] * du[m];
      ldu[k] = lambda[k] * acc;
    }
    Real adu[kN] = {0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < kN; ++k) {
      const Real* rk = eig.right_col(k);
      for (int m = 0; m < kN; ++m) adu[m] += rk[m] * ldu[k];
    }

    for (int m = 0; m < kN; ++m) {
      EXPECT_NEAR(adu[m], fd[m], kJacobianTol * scale)
          << "perturb comp " << comp << " response row " << m;
    }
  }
}

// (3a) Degeneracy at B -> 0: the eigensystem must regularize and produce finite
// vectors that still satisfy L*R == I within a loosened tolerance.
TEST(MhdEigensystem, FiniteAndOrthonormalWhenFieldVanishes) {
  const Real gamma = 5.0 / 3.0;
  MhdState s{};
  const Real rho = 1.0, p = 1.0;
  s.rho = rho;
  s.mx = 0.0;
  s.my = 0.0;
  s.mz = 0.0;
  s.energy = p / (gamma - 1.0);  // zero velocity, zero field
  s.bx = 0.0;
  s.by = 0.0;
  s.bz = 0.0;

  MhdEigensystem eig;
  eig.build(s, /*dir=*/0, gamma);

  for (int i = 0; i < kN; ++i) {
    const Real* li = eig.left_row(i);
    const Real* ri = eig.right_col(i);
    ASSERT_NE(li, nullptr);
    ASSERT_NE(ri, nullptr);
    EXPECT_FALSE(any_nonfinite(li, kN)) << "left row " << i << " non-finite";
    EXPECT_FALSE(any_nonfinite(ri, kN)) << "right col " << i << " non-finite";
    EXPECT_TRUE(std::isfinite(eig.wave_speed(i)));
  }

  for (int i = 0; i < kN; ++i) {
    const Real* li = eig.left_row(i);
    for (int j = 0; j < kN; ++j) {
      const Real* rj = eig.right_col(j);
      Real dot = 0.0;
      for (int k = 0; k < kN; ++k) dot += li[k] * rj[k];
      EXPECT_NEAR(dot, (i == j) ? 1.0 : 0.0, kIdentityTolDegenerate);
    }
  }
}

// (1y) L * R == Identity for the y-normal (dir=1) eigensystem about the same
// representative state. The 7-wave algebra must hold in both directions.
TEST(MhdEigensystem, LeftRightVectorsAreBiorthonormalDirY) {
  const Real gamma = 5.0 / 3.0;
  const MhdState s = make_reference_state(gamma);

  MhdEigensystem eig;
  eig.build(s, /*dir=*/1, gamma);
  EXPECT_EQ(eig.dir(), 1);

  for (int i = 0; i < kN; ++i) {
    const Real* li = eig.left_row(i);
    ASSERT_NE(li, nullptr);
    ASSERT_FALSE(any_nonfinite(li, kN));
    for (int j = 0; j < kN; ++j) {
      const Real* rj = eig.right_col(j);
      ASSERT_NE(rj, nullptr);
      Real dot = 0.0;
      for (int k = 0; k < kN; ++k) dot += li[k] * rj[k];
      const Real expected = (i == j) ? 1.0 : 0.0;
      EXPECT_NEAR(dot, expected, kIdentityTol)
          << "L row " << i << " . R col " << j;
    }
  }
}

TEST(MhdEigensystem, SplitBuildStaysFiniteForDominantBackground) {
  constexpr Real gamma = Real{5} / Real{3};
  const MhdPrim w{Real{1}, Real{0.2}, Real{-0.1}, Real{0.3}, Real{1},
                  Real{0.25}, Real{-0.5}, Real{0.75}};
  const MhdState s = quasar::numerics::to_conserved(w, gamma);
  const quasar::numerics::MhdBackground b0{
      Real{1e100}, Real{-2e100}, Real{0.5e100}};

  ASSERT_NEAR(quasar::numerics::pressure(s, b0, gamma), Real{1}, Real{2e-15});
  MhdEigensystem eig;
  eig.build(s, b0, /*dir=*/0, gamma);
  for (int k = 0; k < kN; ++k) {
    EXPECT_TRUE(std::isfinite(eig.wave_speed(k)));
    EXPECT_FALSE(any_nonfinite(eig.left_row(k), kN));
    EXPECT_FALSE(any_nonfinite(eig.right_col(k), kN));
  }
}

TEST(MhdEigensystem, ZeroBackgroundOverloadIsBitExact) {
  constexpr Real gamma = Real{5} / Real{3};
  const MhdState s = make_reference_state(gamma);
  MhdEigensystem ordinary;
  MhdEigensystem split;
  ordinary.build(s, /*dir=*/1, gamma);
  split.build(s, quasar::numerics::MhdBackground{}, /*dir=*/1, gamma);
  for (int wave = 0; wave < kN; ++wave) {
    EXPECT_EQ(split.wave_speed(wave), ordinary.wave_speed(wave));
    const Real* lo = ordinary.left_row(wave);
    const Real* ls = split.left_row(wave);
    const Real* ro = ordinary.right_col(wave);
    const Real* rs = split.right_col(wave);
    for (int var = 0; var < kN; ++var) {
      EXPECT_EQ(ls[var], lo[var]);
      EXPECT_EQ(rs[var], ro[var]);
    }
  }
}

// (2y) Diagonalization of the y-direction flux Jacobian: R*diag(lambda)*L*du
// reproduces the finite-difference of the analytic y-flux. The active-variable
// ordering for dir=1 is [rho, mx, my, mz, energy, bx, bz] (By held constant).
TEST(MhdEigensystem, DiagonalizesFluxJacobianDirY) {
  const Real gamma = 5.0 / 3.0;
  const MhdState s = make_reference_state(gamma);
  const Real by = s.by;

  MhdEigensystem eig;
  eig.build(s, /*dir=*/1, gamma);

  Real lambda[kN];
  for (int k = 0; k < kN; ++k) lambda[k] = eig.wave_speed(k);

  Real u0[kN];
  pack7_y(s, u0);
  Real f0[kN];
  flux_y(s, gamma, f0);

  const Real eps = 1e-6;
  for (int comp = 0; comp < kN; ++comp) {
    Real du[kN] = {0, 0, 0, 0, 0, 0, 0};
    const Real scale = (std::abs(u0[comp]) > 1.0 ? std::abs(u0[comp]) : 1.0);
    du[comp] = eps * scale;

    Real up[kN];
    for (int k = 0; k < kN; ++k) up[k] = u0[k] + du[k];
    const MhdState sp = unpack7_y(up, by);
    Real fp[kN];
    flux_y(sp, gamma, fp);
    Real fd[kN];
    for (int k = 0; k < kN; ++k) fd[k] = fp[k] - f0[k];

    Real ldu[kN];
    for (int k = 0; k < kN; ++k) {
      const Real* lk = eig.left_row(k);
      Real acc = 0.0;
      for (int m = 0; m < kN; ++m) acc += lk[m] * du[m];
      ldu[k] = lambda[k] * acc;
    }
    Real adu[kN] = {0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < kN; ++k) {
      const Real* rk = eig.right_col(k);
      for (int m = 0; m < kN; ++m) adu[m] += rk[m] * ldu[k];
    }

    for (int m = 0; m < kN; ++m) {
      EXPECT_NEAR(adu[m], fd[m], kJacobianTol * scale)
          << "perturb comp " << comp << " response row " << m;
    }
  }
}

// (3b) Degeneracy at B_perp -> 0 (only normal field present, coincident wave
// speeds): still finite and biorthonormal within the loosened tolerance.
TEST(MhdEigensystem, FiniteAndOrthonormalWhenTransverseFieldVanishes) {
  const Real gamma = 5.0 / 3.0;
  const Real rho = 1.0, p = 1.0, bx = 1.0;
  MhdState s{};
  s.rho = rho;
  s.mx = 0.0;
  s.my = 0.0;
  s.mz = 0.0;
  s.energy = p / (gamma - 1.0) + 0.5 * bx * bx;
  s.bx = bx;  // normal (x) field only; By = Bz = 0
  s.by = 0.0;
  s.bz = 0.0;

  MhdEigensystem eig;
  eig.build(s, /*dir=*/0, gamma);

  for (int i = 0; i < kN; ++i) {
    const Real* li = eig.left_row(i);
    const Real* ri = eig.right_col(i);
    ASSERT_NE(li, nullptr);
    ASSERT_NE(ri, nullptr);
    EXPECT_FALSE(any_nonfinite(li, kN)) << "left row " << i << " non-finite";
    EXPECT_FALSE(any_nonfinite(ri, kN)) << "right col " << i << " non-finite";
    EXPECT_TRUE(std::isfinite(eig.wave_speed(i)));
  }

  for (int i = 0; i < kN; ++i) {
    const Real* li = eig.left_row(i);
    for (int j = 0; j < kN; ++j) {
      const Real* rj = eig.right_col(j);
      Real dot = 0.0;
      for (int k = 0; k < kN; ++k) dot += li[k] * rj[k];
      EXPECT_NEAR(dot, (i == j) ? 1.0 : 0.0, kIdentityTolDegenerate);
    }
  }
}

TEST(MhdEigensystem, ExtremeFiniteStateHasFiniteRootsAndBasis) {
  const Real gamma = Real{5} / Real{3};
  MhdPrim w{};
  w.rho = Real{1};
  w.vx = Real{0.2};
  w.vy = Real{-0.1};
  w.vz = Real{0.05};
  w.p = Real{2e299};
  w.bx = Real{8e149};
  w.by = Real{-6e149};
  w.bz = Real{4e149};
  const MhdState s = quasar::numerics::to_conserved(w, gamma);
  ASSERT_TRUE(std::isfinite(s.energy));

  MhdEigensystem eig;
  eig.build(s, /*dir=*/0, gamma);
  for (int k = 0; k < kN; ++k) {
    EXPECT_TRUE(std::isfinite(eig.wave_speed(k))) << "wave " << k;
    EXPECT_FALSE(any_nonfinite(eig.left_row(k), kN)) << "left row " << k;
    EXPECT_FALSE(any_nonfinite(eig.right_col(k), kN)) << "right col " << k;
  }

  for (int i = 0; i < kN; ++i) {
    const Real* li = eig.left_row(i);
    for (int j = 0; j < kN; ++j) {
      const Real* rj = eig.right_col(j);
      Real dot = Real{0};
      for (int k = 0; k < kN; ++k) dot += li[k] * rj[k];
      EXPECT_NEAR(dot, (i == j) ? Real{1} : Real{0}, Real{2e-6})
          << "L row " << i << " . R col " << j;
    }
  }
}
