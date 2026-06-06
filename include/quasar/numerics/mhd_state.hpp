#pragma once

// Ideal-MHD state, primitive, and flux types plus the host/device-callable
// equation-of-state helpers shared by the MHD numerics axis (reconstruction,
// Riemann solver, eigensystem). All math here is `inline QUASAR_HOST_DEVICE`
// so kernels can call it directly; src/numerics/mhd_state.cpp provides the
// translation-unit anchor for the HIP-tagged module object.

#include "quasar/core/types.hpp"

#include <cmath>

namespace quasar::numerics {

// Conserved variables (total energy). Component order is the canonical 8-tuple
// (rho, mx, my, mz, energy, bx, by, bz) used everywhere downstream.
struct MhdState {
  Real rho{};
  Real mx{};
  Real my{};
  Real mz{};
  Real energy{};
  Real bx{};
  Real by{};
  Real bz{};
};

// Primitive variables (velocity + gas pressure).
struct MhdPrim {
  Real rho{};
  Real vx{};
  Real vy{};
  Real vz{};
  Real p{};
  Real bx{};
  Real by{};
  Real bz{};
};

// Physical flux in the canonical conserved-variable order.
struct MhdFlux {
  Real rho{};
  Real mx{};
  Real my{};
  Real mz{};
  Real energy{};
  Real bx{};
  Real by{};
  Real bz{};
};

// Static background magnetic field B0 for the field-split formulation
// B = B0 + b. The stored conserved state carries the perturbation b in
// u.bx/u.by/u.bz and the perturbation-only magnetic energy 0.5|b|^2; this POD
// carries the (spatially sampled) constant background components passed to the
// background-aware EOS overloads below. It is a trivial value type so it can be
// passed by value into HIP device code.
struct MhdBackground {
  Real b0x{};
  Real b0y{};
  Real b0z{};
};

// p = (gamma-1)*(E - 0.5*rho|v|^2 - 0.5|B|^2). Total energy convention.
QUASAR_HOST_DEVICE inline Real pressure(const MhdState& u, Real gamma) {
  const Real inv_rho = Real{1} / u.rho;
  const Real kinetic = Real{0.5} * (u.mx * u.mx + u.my * u.my + u.mz * u.mz) * inv_rho;
  const Real magnetic = Real{0.5} * (u.bx * u.bx + u.by * u.by + u.bz * u.bz);
  return (gamma - Real{1}) * (u.energy - kinetic - magnetic);
}

// Field-split gas pressure. By construction the stored energy carries only the
// perturbation magnetic energy 0.5|b|^2 (b = u.bx/by/bz), so the gas pressure
// has NO B0 cross term and does not depend on the background at all:
//   p = (gamma-1)*(E - 0.5*rho|v|^2 - 0.5|b|^2).
// The MhdBackground argument exists only to give the EOS a uniform call surface
// shared with fast_magnetosonic_speed(); it is intentionally unread here so that
// pressure(u, {0,0,0}, gamma) is bit-for-bit identical to pressure(u, gamma).
QUASAR_HOST_DEVICE inline Real pressure(const MhdState& u, const MhdBackground& /*b0*/,
                                        Real gamma) {
  return pressure(u, gamma);
}

// Conserved -> primitive. v = m/rho; p from the gamma law above. Magnetic
// components pass through unchanged. Exact inverse of to_conserved().
QUASAR_HOST_DEVICE inline MhdPrim to_primitive(const MhdState& u, Real gamma) {
  const Real inv_rho = Real{1} / u.rho;
  MhdPrim w;
  w.rho = u.rho;
  w.vx  = u.mx * inv_rho;
  w.vy  = u.my * inv_rho;
  w.vz  = u.mz * inv_rho;
  w.bx  = u.bx;
  w.by  = u.by;
  w.bz  = u.bz;
  const Real kinetic  = Real{0.5} * u.rho * (w.vx * w.vx + w.vy * w.vy + w.vz * w.vz);
  const Real magnetic = Real{0.5} * (u.bx * u.bx + u.by * u.by + u.bz * u.bz);
  w.p = (gamma - Real{1}) * (u.energy - kinetic - magnetic);
  return w;
}

// Primitive -> conserved. m = rho*v; E = p/(gamma-1) + 0.5*rho|v|^2 + 0.5|B|^2.
// Exact inverse of to_primitive().
QUASAR_HOST_DEVICE inline MhdState to_conserved(const MhdPrim& w, Real gamma) {
  MhdState u;
  u.rho = w.rho;
  u.mx  = w.rho * w.vx;
  u.my  = w.rho * w.vy;
  u.mz  = w.rho * w.vz;
  u.bx  = w.bx;
  u.by  = w.by;
  u.bz  = w.bz;
  const Real kinetic  = Real{0.5} * w.rho * (w.vx * w.vx + w.vy * w.vy + w.vz * w.vz);
  const Real magnetic = Real{0.5} * (w.bx * w.bx + w.by * w.by + w.bz * w.bz);
  u.energy = w.p / (gamma - Real{1}) + kinetic + magnetic;
  return u;
}

// Fast magnetosonic speed along the normal `dir` (0=x, 1=y).
//   a^2  = gamma*p/rho                       (sound speed squared)
//   ca^2 = |B|^2/rho                         (total Alfven speed squared)
//   cax  = B_dir/sqrt(rho)                   (directional Alfven speed)
//   c_f^2 = 0.5*[ (a^2+ca^2) + sqrt((a^2+ca^2)^2 - 4 a^2 cax^2) ]
// c_f >= a, c_f >= |cax|, and reduces to a when B=0.
//
// Field-split overload: the wave speeds see the TOTAL magnetic field
// B = b + B0, where b = u.bx/by/bz is the stored perturbation and B0 is the
// background. The gas pressure p still comes from the perturbation-only EOS
// above (no B0 cross term). With b0 == {0,0,0} the total field equals b and this
// is bit-for-bit identical to the zero-background overload below.
QUASAR_HOST_DEVICE inline Real fast_magnetosonic_speed(const MhdState& u,
                                                       const MhdBackground& b0,
                                                       int dir, Real gamma) {
  const Real inv_rho = Real{1} / u.rho;
  const Real p  = pressure(u, b0, gamma);
  const Real Bx = u.bx + b0.b0x;
  const Real By = u.by + b0.b0y;
  const Real Bz = u.bz + b0.b0z;
  const Real a2 = gamma * p * inv_rho;
  const Real ca2 = (Bx * Bx + By * By + Bz * Bz) * inv_rho;
  const Real bn = (dir == 0) ? Bx : By;
  const Real cax2 = bn * bn * inv_rho;
  const Real sum = a2 + ca2;
  // Guard the discriminant against negative round-off near degeneracy.
  Real disc = sum * sum - Real{4} * a2 * cax2;
  if (disc < Real{0}) {
    disc = Real{0};
  }
  const Real cf2 = Real{0.5} * (sum + std::sqrt(disc));
  return std::sqrt(cf2 < Real{0} ? Real{0} : cf2);
}

// Zero-background overload. Delegates to the field-split overload with
// B0 == {0,0,0}: the total field reduces to b = u.bx/by/bz and p reduces to
// pressure(u, gamma), so the result is unchanged from the original body.
QUASAR_HOST_DEVICE inline Real fast_magnetosonic_speed(const MhdState& u, int dir, Real gamma) {
  return fast_magnetosonic_speed(u, MhdBackground{}, dir, gamma);
}

// Translation-unit anchor (defined in src/numerics/mhd_state.cpp) so the
// HIP-tagged module object is non-trivial even though all math is header-inline.
const char* mhd_state_version() noexcept;

}  // namespace quasar::numerics
