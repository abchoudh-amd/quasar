#pragma once

// Full 7-wave ideal-MHD eigenstructure for characteristic-variable
// reconstruction, following Stone, Gardiner, Teuben, Hawley & Simon (2008),
// "Athena: A New Code for Astrophysical MHD", ApJS 178, 137 -- specifically the
// adiabatic eigenvectors of Appendix A (their Roe-type primitive eigensystem).
//
// We work in PRIMITIVE variables w = (rho, vn, vt1, vt2, p, bt1, bt2), where for
// dir=0 (x-normal): vn=vx, vt1=vy, vt2=vz, bt1=by, bt2=bz; for dir=1 (y-normal):
// vn=vy, vt1=vx (sign-consistent rotation below), bt1=bx, bt2=bz. The normal
// magnetic component B_n is held continuous (constrained transport) and is NOT
// an independent wave in the 7-wave set.
//
// The class exposes 7x7 L and R built in CONSERVED variables via L = Lp * dW/dU
// and R = dU/dW * Rp, so that L*R = I and R*diag(speeds)*L = dF/dU (the flux
// Jacobian along `dir`). The 7 components map to the 7 conserved variables that
// vary across the waves: (rho, mn, mt1, mt2, energy, bt1, bt2). The normal
// B-field (component dir of B) is the constrained 8th variable, handled outside
// this 7-wave system (see characteristic_projection.hpp).
//
// Wave ordering (index k): 0 fast-, 1 alfven-, 2 slow-, 3 entropy/contact,
// 4 slow+, 5 alfven+, 6 fast+.

#include "quasar/core/types.hpp"
#include "quasar/numerics/mhd_state.hpp"

#include <cmath>

namespace quasar::numerics {

class MhdEigensystem {
 public:
  // Compute L (7x7, row-major), R (7x7, row-major), and the 7 wave speeds about
  // conserved state `s` along normal `dir` (0=x, 1=y) for ratio-of-specific-heats
  // `gamma`. Degeneracies (B_perp -> 0, coincident speeds) are regularized with
  // the Roe/Balsara renormalization so L,R stay finite and L*R = I (loosened tol).
  QUASAR_HOST_DEVICE void build(const MhdState& s, int dir, Real gamma) {
    dir_ = dir;

    const MhdPrim w = to_primitive(s, gamma);
    const Real rho = w.rho;
    const Real p   = (w.p > Real{0}) ? w.p : Real{0};
    const Real sqrt_rho = sqrt(rho);
    const Real inv_sqrt_rho = Real{1} / sqrt_rho;

    // Rotate to (normal, transverse1, transverse2). For dir=0: n=x, t1=y, t2=z.
    // For dir=1: n=y, t1=z, t2=x  -- a right-handed cyclic rotation keeps the
    // eigenvectors' cross-coupling signs consistent. We track the index map so
    // L,R are returned in the canonical (rho,mn,mt1,mt2,E,bt1,bt2) ordering, and
    // the projector applies the same rotation on the conserved delta.
    Real bn, bt1, bt2;
    if (dir == 0) {
      bn = w.bx; bt1 = w.by; bt2 = w.bz;
    } else {
      bn = w.by; bt1 = w.bz; bt2 = w.bx;
    }

    // Sound speed, Alfven speeds.
    const Real a2 = gamma * p / rho;             // a^2
    const Real a  = sqrt(a2);
    const Real bn2 = bn * bn;
    const Real bt_sq = bt1 * bt1 + bt2 * bt2;    // |B_perp|^2
    const Real cax2 = bn2 / rho;                 // normal Alfven speed^2
    const Real cax  = sqrt(cax2);
    const Real ca2  = (bn2 + bt_sq) / rho;       // total Alfven speed^2 (b^2/rho)

    // Fast/slow magnetosonic speeds.
    const Real sum = a2 + ca2;
    Real disc = sum * sum - Real{4} * a2 * cax2;
    if (disc < Real{0}) disc = Real{0};
    const Real sqrt_disc = sqrt(disc);
    Real cf2 = Real{0.5} * (sum + sqrt_disc);
    Real cs2 = Real{0.5} * (sum - sqrt_disc);
    if (cf2 < Real{0}) cf2 = Real{0};
    if (cs2 < Real{0}) cs2 = Real{0};
    const Real cf = sqrt(cf2);
    const Real cs = sqrt(cs2);

    // Roe/Balsara normalization coefficients alpha_f, alpha_s. Guard the
    // cf^2-cs^2 denominator: when fast and slow coincide (e.g. B_perp=0 and
    // a==cax) split the weight evenly.
    Real alpha_f, alpha_s;
    {
      const Real denom = cf2 - cs2;
      const Real eps = static_cast<Real>(1e-12);
      if (denom <= eps * (cf2 + cs2 + eps)) {
        alpha_f = Real{1} / sqrt(Real{2});
        alpha_s = Real{1} / sqrt(Real{2});
      } else {
        Real af2 = (a2 - cs2) / denom;
        Real as2 = (cf2 - a2) / denom;
        if (af2 < Real{0}) af2 = Real{0};
        if (as2 < Real{0}) as2 = Real{0};
        alpha_f = sqrt(af2);
        alpha_s = sqrt(as2);
      }
    }

    // beta_y, beta_z: direction cosines of the transverse field. The well-known
    // fallback beta = 1/sqrt(2) when |B_perp| is negligible keeps the Alfven and
    // magnetosonic eigenvectors finite at B_perp -> 0.
    Real beta1, beta2;
    {
      const Real bt = sqrt(bt_sq);
      const Real eps = static_cast<Real>(1e-12) * (a + sqrt(ca2) + static_cast<Real>(1e-30));
      if (bt < eps) {
        beta1 = Real{1} / sqrt(Real{2});
        beta2 = Real{1} / sqrt(Real{2});
      } else {
        beta1 = bt1 / bt;
        beta2 = bt2 / bt;
      }
    }

    // Sign of B_n (sbn): enters the Alfven/magnetosonic eigenvectors. Treat
    // bn==0 as +1 to keep a definite branch.
    const Real sbn = (bn >= Real{0}) ? Real{1} : Real{-1};

    // Wave speeds (vn is the bulk normal velocity).
    Real vn, vt1, vt2;
    if (dir == 0) {
      vn = w.vx; vt1 = w.vy; vt2 = w.vz;
    } else {
      vn = w.vy; vt1 = w.vz; vt2 = w.vx;
    }
    speeds_[0] = vn - cf;
    speeds_[1] = vn - cax;
    speeds_[2] = vn - cs;
    speeds_[3] = vn;
    speeds_[4] = vn + cs;
    speeds_[5] = vn + cax;
    speeds_[6] = vn + cf;

    // ---------------------------------------------------------------------
    // Primitive-variable eigenvectors (Stone et al. 2008, Eqs. A1-A20), in the
    // rotated order p = (rho, vn, vt1, vt2, pgas, bt1, bt2). qf = cf*alpha_f*sbn,
    // qs = cs*alpha_s*sbn, Af = a*alpha_f*sqrt(rho), As = a*alpha_s*sqrt(rho).
    // ---------------------------------------------------------------------
    const Real qf = cf * alpha_f * sbn;
    const Real qs = cs * alpha_s * sbn;
    const Real Af = a * alpha_f * sqrt_rho;
    const Real As = a * alpha_s * sqrt_rho;
    const Real Nf = Real{0.5} / (a2 > Real{0} ? a2 : static_cast<Real>(1e-300));  // 1/(2 a^2) normalization

    // Right eigenvectors in primitive space, columns Rp[wave][var].
    // var index: 0=rho,1=vn,2=vt1,3=vt2,4=p,5=bt1,6=bt2.
    Real Rp[7][7] = {{0}};

    // Fast - (k=0) and Fast + (k=6).
    // R_fast(+/-): rho: rho*alpha_f ; vn: -/+ alpha_f*cf ;
    //   vt1: +/- qs*beta1 ; vt2: +/- qs*beta2 ; p: rho*alpha_f*a2 ;
    //   bt1: As*beta1 ; bt2: As*beta2 .
    {
      // k=0 fast-minus (use upper sign of -/+ -> vn coefficient = +alpha_f*cf? )
      // Convention: column for vn-cf has vn-component = -alpha_f*cf, and the
      // transverse velocity uses +qs*beta; column for vn+cf flips vn and vt.
      Rp[0][0] = rho * alpha_f;
      Rp[0][1] = -alpha_f * cf;
      Rp[0][2] = qs * beta1;
      Rp[0][3] = qs * beta2;
      Rp[0][4] = rho * alpha_f * a2;
      Rp[0][5] = As * beta1;
      Rp[0][6] = As * beta2;

      Rp[6][0] = rho * alpha_f;
      Rp[6][1] = alpha_f * cf;
      Rp[6][2] = -qs * beta1;
      Rp[6][3] = -qs * beta2;
      Rp[6][4] = rho * alpha_f * a2;
      Rp[6][5] = As * beta1;
      Rp[6][6] = As * beta2;
    }

    // Alfven - (k=1) and Alfven + (k=5). These carry only transverse v and B,
    // rotated 90 deg in the transverse plane: (beta2, -beta1).
    {
      const Real s2 = Real{1} / sqrt(Real{2});
      Rp[1][0] = Real{0};
      Rp[1][1] = Real{0};
      Rp[1][2] = -beta2 * s2;
      Rp[1][3] = beta1 * s2;
      Rp[1][4] = Real{0};
      Rp[1][5] = -sbn * beta2 * s2 * sqrt_rho;
      Rp[1][6] = sbn * beta1 * s2 * sqrt_rho;

      Rp[5][0] = Real{0};
      Rp[5][1] = Real{0};
      Rp[5][2] = -beta2 * s2;
      Rp[5][3] = beta1 * s2;
      Rp[5][4] = Real{0};
      Rp[5][5] = sbn * beta2 * s2 * sqrt_rho;
      Rp[5][6] = -sbn * beta1 * s2 * sqrt_rho;
    }

    // Slow - (k=2) and Slow + (k=4).
    // R_slow(+/-): rho: rho*alpha_s ; vn: -/+ alpha_s*cs ;
    //   vt1: -/+ qf*beta1 ; vt2: -/+ qf*beta2 ; p: rho*alpha_s*a2 ;
    //   bt1: -Af*beta1 ; bt2: -Af*beta2 .
    {
      Rp[2][0] = rho * alpha_s;
      Rp[2][1] = -alpha_s * cs;
      Rp[2][2] = -qf * beta1;
      Rp[2][3] = -qf * beta2;
      Rp[2][4] = rho * alpha_s * a2;
      Rp[2][5] = -Af * beta1;
      Rp[2][6] = -Af * beta2;

      Rp[4][0] = rho * alpha_s;
      Rp[4][1] = alpha_s * cs;
      Rp[4][2] = qf * beta1;
      Rp[4][3] = qf * beta2;
      Rp[4][4] = rho * alpha_s * a2;
      Rp[4][5] = -Af * beta1;
      Rp[4][6] = -Af * beta2;
    }

    // Entropy / contact (k=3): density jump only.
    {
      Rp[3][0] = Real{1};
      Rp[3][1] = Real{0};
      Rp[3][2] = Real{0};
      Rp[3][3] = Real{0};
      Rp[3][4] = Real{0};
      Rp[3][5] = Real{0};
      Rp[3][6] = Real{0};
    }

    // Left eigenvectors in primitive space, Lp[wave][var], biorthonormal to Rp.
    // Stone et al. Eqs. A21-A40. With the normalizations above, Lp*Rp = I.
    Real Lp[7][7] = {{0}};
    {
      const Real s2 = Real{1} / sqrt(Real{2});
      const Real inv_rho = Real{1} / rho;

      // Fast - (k=0) and Fast + (k=6).
      Lp[0][0] = Real{0};
      Lp[0][1] = -Nf * alpha_f * cf;
      Lp[0][2] = Nf * qs * beta1;
      Lp[0][3] = Nf * qs * beta2;
      Lp[0][4] = Nf * alpha_f / rho;
      Lp[0][5] = Nf * As * beta1 / rho;
      Lp[0][6] = Nf * As * beta2 / rho;

      Lp[6][0] = Real{0};
      Lp[6][1] = Nf * alpha_f * cf;
      Lp[6][2] = -Nf * qs * beta1;
      Lp[6][3] = -Nf * qs * beta2;
      Lp[6][4] = Nf * alpha_f / rho;
      Lp[6][5] = Nf * As * beta1 / rho;
      Lp[6][6] = Nf * As * beta2 / rho;

      // Alfven - (k=1) and Alfven + (k=5).
      Lp[1][0] = Real{0};
      Lp[1][1] = Real{0};
      Lp[1][2] = -beta2 * s2;
      Lp[1][3] = beta1 * s2;
      Lp[1][4] = Real{0};
      Lp[1][5] = -sbn * beta2 * s2 / sqrt_rho;
      Lp[1][6] = sbn * beta1 * s2 / sqrt_rho;

      Lp[5][0] = Real{0};
      Lp[5][1] = Real{0};
      Lp[5][2] = -beta2 * s2;
      Lp[5][3] = beta1 * s2;
      Lp[5][4] = Real{0};
      Lp[5][5] = sbn * beta2 * s2 / sqrt_rho;
      Lp[5][6] = -sbn * beta1 * s2 / sqrt_rho;

      // Slow - (k=2) and Slow + (k=4).
      Lp[2][0] = Real{0};
      Lp[2][1] = -Nf * alpha_s * cs;
      Lp[2][2] = -Nf * qf * beta1;
      Lp[2][3] = -Nf * qf * beta2;
      Lp[2][4] = Nf * alpha_s / rho;
      Lp[2][5] = -Nf * Af * beta1 / rho;
      Lp[2][6] = -Nf * Af * beta2 / rho;

      Lp[4][0] = Real{0};
      Lp[4][1] = Nf * alpha_s * cs;
      Lp[4][2] = Nf * qf * beta1;
      Lp[4][3] = Nf * qf * beta2;
      Lp[4][4] = Nf * alpha_s / rho;
      Lp[4][5] = -Nf * Af * beta1 / rho;
      Lp[4][6] = -Nf * Af * beta2 / rho;

      // Entropy / contact (k=3): rho - p/a^2.
      Lp[3][0] = Real{1};
      Lp[3][1] = Real{0};
      Lp[3][2] = Real{0};
      Lp[3][3] = Real{0};
      Lp[3][4] = -Real{1} / a2;
      Lp[3][5] = Real{0};
      Lp[3][6] = Real{0};

      (void)inv_rho;
    }

    // ---------------------------------------------------------------------
    // Transform primitive eigenvectors to conserved variables.
    //   U = (rho, mn, mt1, mt2, E, bt1, bt2),  W = (rho, vn, vt1, vt2, p, bt1, bt2).
    //   R_U = (dU/dW) Rp ,  L_U = Lp (dW/dU) ,  with dW/dU = (dU/dW)^{-1}.
    // dU/dW (M) and dW/dU (Minv) for the rotated 7-var set:
    //   m_i = rho*v_i  ->  d(m_i) = v_i drho + rho dv_i
    //   E = p/(g-1) + 0.5 rho|v|^2 + 0.5(bn^2 + bt1^2 + bt2^2)
    //       (bn constant across these waves) ->
    //   dE = 0.5|v|^2 drho + rho vn dvn + rho vt1 dvt1 + rho vt2 dvt2
    //        + dp/(g-1) + bt1 dbt1 + bt2 dbt2
    // ---------------------------------------------------------------------
    const Real gm1 = gamma - Real{1};
    const Real v2 = vn * vn + vt1 * vt1 + vt2 * vt2;

    // M = dU/dW (7x7).
    Real M[7][7] = {{0}};
    M[0][0] = Real{1};                                   // rho
    M[1][0] = vn;   M[1][1] = rho;                       // mn
    M[2][0] = vt1;  M[2][2] = rho;                       // mt1
    M[3][0] = vt2;  M[3][3] = rho;                       // mt2
    M[4][0] = Real{0.5} * v2;
    M[4][1] = rho * vn; M[4][2] = rho * vt1; M[4][3] = rho * vt2;
    M[4][4] = Real{1} / gm1;
    M[4][5] = bt1; M[4][6] = bt2;                        // E
    M[5][5] = Real{1};                                   // bt1
    M[6][6] = Real{1};                                   // bt2

    // Minv = dW/dU (7x7), the analytic inverse of M.
    Real Minv[7][7] = {{0}};
    const Real inv_rho = Real{1} / rho;
    Minv[0][0] = Real{1};                                // rho
    Minv[1][0] = -vn * inv_rho;  Minv[1][1] = inv_rho;   // vn
    Minv[2][0] = -vt1 * inv_rho; Minv[2][2] = inv_rho;   // vt1
    Minv[3][0] = -vt2 * inv_rho; Minv[3][3] = inv_rho;   // vt2
    // p = (g-1)*(E - 0.5 rho|v|^2 - 0.5 B^2). In differentials about state:
    //   dp = (g-1)[ dE - 0.5 v^2 drho - rho(vn dvn+...) - bt1 dbt1 - bt2 dbt2 ]
    // expressed in U: dv_i = (dm_i - v_i drho)/rho, so rho v.dv = v.dm - v^2 drho.
    //   dp = (g-1)[ dE - 0.5 v^2 drho - (v.dm - v^2 drho) - bt.dbt ]
    //      = (g-1)[ dE + 0.5 v^2 drho - v.dm - bt.dbt ]
    Minv[4][0] = gm1 * Real{0.5} * v2;
    Minv[4][1] = -gm1 * vn;
    Minv[4][2] = -gm1 * vt1;
    Minv[4][3] = -gm1 * vt2;
    Minv[4][4] = gm1;
    Minv[4][5] = -gm1 * bt1;
    Minv[4][6] = -gm1 * bt2;
    Minv[5][5] = Real{1};                                // bt1
    Minv[6][6] = Real{1};                                // bt2

    // R_U[k][var] = sum_j M[var][j] * Rp[k][j]   (apply M to each Rp column).
    // L_U[k][var] = sum_j Lp[k][j] * Minv[j][var].
    for (int k = 0; k < 7; ++k) {
      for (int var = 0; var < 7; ++var) {
        Real racc = Real{0};
        Real lacc = Real{0};
        for (int j = 0; j < 7; ++j) {
          racc += M[var][j] * Rp[k][j];
          lacc += Lp[k][j] * Minv[j][var];
        }
        // Store R as columns (right_col(k) is length-7), L as rows.
        R_[var * 7 + k] = racc;   // R_[var,k]
        L_[k * 7 + var] = lacc;   // L_[k,var]
      }
    }
  }

  // Pointer to length-7 left eigenvector k (row k of L). w_k = L_row(k) . delta.
  QUASAR_HOST_DEVICE const Real* left_row(int k) const { return &L_[k * 7]; }

  // Pointer to length-7 right eigenvector k (column k of R). Stored contiguously
  // in a per-column scratch refreshed on each call.
  QUASAR_HOST_DEVICE const Real* right_col(int k) const {
    for (int var = 0; var < 7; ++var) {
      col_scratch_[var] = R_[var * 7 + k];
    }
    return col_scratch_;
  }

  // Wave speed for wave k (0..6), ordering: fast-,alfven-,slow-,entropy,slow+,
  // alfven+,fast+.
  QUASAR_HOST_DEVICE Real wave_speed(int k) const { return speeds_[k]; }

  QUASAR_HOST_DEVICE int dir() const { return dir_; }

 private:
  Real L_[49]{};        // row-major 7x7 left-eigenvector matrix (rows = waves)
  Real R_[49]{};        // row-major 7x7 right-eigenvector matrix (cols = waves)
  Real speeds_[7]{};
  int  dir_{0};
  mutable Real col_scratch_[7]{};
};

}  // namespace quasar::numerics
