#pragma once

// Projection of a conserved-variable delta into characteristic (wave) space and
// back, using a built MhdEigensystem. The 7-wave ideal-MHD set does NOT include
// the normal magnetic-field component as an independent wave: under constrained
// transport B_n is continuous across the interface, so a delta in B_n is not a
// characteristic degree of freedom.
//
// Normal-B handling decision: the normal B component (B_x for dir=0, B_y for
// dir=1) is carried through to_char/from_char UNCHANGED -- it never enters the
// L*delta projection and is restored verbatim by from_char. Consequently
// from_char(to_char(d)) == d exactly for any delta whose normal-B perturbation
// is zero; for a nonzero normal-B delta the reconstruction is the orthogonal
// projection onto the 7-wave subspace (normal-B passed through), which is the
// physically correct treatment for CT.
//
// The eigensystem is phrased in the rotated 7-vector
//   U7 = (rho, m_n, m_t1, m_t2, energy, b_t1, b_t2)
// where for dir=0: (n,t1,t2) = (x,y,z); for dir=1: (n,t1,t2) = (y,z,x). The same
// rotation is applied here so to_char/from_char are consistent with build().

#include "quasar/core/types.hpp"
#include "quasar/numerics/mhd_eigensystem.hpp"
#include "quasar/numerics/mhd_state.hpp"

namespace quasar::numerics {

// POD carrier for the 7 characteristic-wave amplitudes. Replaces
// std::array<Real,7> as the to_char/from_char wire type so the projection is
// trivially passable into HIP device code (no <array>, no host-only machinery).
struct CharVec7 {
  Real w[7];
};

class CharacteristicProjector {
 public:
  // w_k = sum_var L[k][var] * delta_rotated[var]   (k = 0..6)
  QUASAR_HOST_DEVICE static CharVec7 to_char(const MhdState& delta, const MhdEigensystem& eig) {
    const Real u7[7] = {
      delta.rho,
      (eig.dir() == 0) ? delta.mx : delta.my,                       // m_n
      (eig.dir() == 0) ? delta.my : delta.mz,                       // m_t1
      (eig.dir() == 0) ? delta.mz : delta.mx,                       // m_t2
      delta.energy,
      (eig.dir() == 0) ? delta.by : delta.bz,                       // b_t1
      (eig.dir() == 0) ? delta.bz : delta.bx,                       // b_t2
    };
    CharVec7 w{};
    for (int k = 0; k < 7; ++k) {
      const Real* lrow = eig.left_row(k);
      Real acc = Real{0};
      for (int var = 0; var < 7; ++var) {
        acc += lrow[var] * u7[var];
      }
      w.w[k] = acc;
    }
    return w;
  }

  // delta_rotated[var] = sum_k R[var][k] * w_k. The normal-B component is set to
  // zero here (it is not a wave); callers that need to preserve a nonzero normal-B
  // delta restore it separately. With a zero normal-B input this is the exact
  // inverse of to_char.
  QUASAR_HOST_DEVICE static MhdState from_char(const CharVec7& w, const MhdEigensystem& eig) {
    Real u7[7] = {0, 0, 0, 0, 0, 0, 0};
    for (int k = 0; k < 7; ++k) {
      const Real* rcol = eig.right_col(k);   // length-7, indexed by var
      for (int var = 0; var < 7; ++var) {
        u7[var] += rcol[var] * w.w[k];
      }
    }
    MhdState d{};
    d.rho    = u7[0];
    d.energy = u7[4];
    if (eig.dir() == 0) {
      d.mx = u7[1]; d.my = u7[2]; d.mz = u7[3];
      d.by = u7[5]; d.bz = u7[6];
      d.bx = Real{0};   // normal B: not a characteristic wave (carried separately)
    } else {
      d.my = u7[1]; d.mz = u7[2]; d.mx = u7[3];
      d.bz = u7[5]; d.bx = u7[6];
      d.by = Real{0};   // normal B: not a characteristic wave (carried separately)
    }
    return d;
  }
};

}  // namespace quasar::numerics
