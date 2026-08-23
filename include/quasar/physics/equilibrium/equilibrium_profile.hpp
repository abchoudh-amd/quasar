#pragma once

// Free-function profiles for the Grad-Shafranov equation
//
//   Delta* psi = -mu0 r^2 p'(psi) - F F'(psi)
//
// p(psi) and F(psi) = R B_phi are the two functions a GS solve does not
// determine -- they are physics input. This header defines how they are
// specified, evaluated, and normalized.
//
// -- Normalized flux ----------------------------------------------------------
// Profiles are functions of psi_N = (psi - psi_axis) / (psi_bdry - psi_axis),
// which is 0 on the magnetic axis and 1 on the plasma boundary. This is the
// EFIT/FreeGS convention, so a polynomial profile here is directly comparable
// with published equilibria.
//
// psi_N depends on psi_axis and psi_bdry, which are themselves outputs of the
// solve. That circularity is why the outer loop must re-locate the critical
// points every iteration (physics/equilibrium/critical_points.hpp).
//
// -- The current normalization ------------------------------------------------
// The scale factors A and B multiplying p' and FF' are NOT free: they are set
// each outer iteration so the resulting toroidal current integrates to the
// requested total plasma current I_p. Without this the Picard iteration is
// ill-posed -- profile shape and current amplitude would be independently
// specifiable, and the solution could drift to any current.
//
// -- Registry ------------------------------------------------------------------
// Concrete profiles self-register by name (core/registry.hpp) so a deck selects
// them by string, mirroring IMhdBackgroundProfile. Spline and tabulated profiles
// can be added later without touching the solver.

#include "quasar/core/types.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace quasar::equilibrium {

class IEquilibriumProfile {
 public:
  virtual ~IEquilibriumProfile() = default;

  // dp/dpsi at normalized flux psi_N, in the profile's own (unscaled) units.
  // The solver applies the I_p normalization on top of this.
  virtual Real dp_dpsi(Real psi_n) const = 0;

  // d(F^2/2)/dpsi = F F' at normalized flux psi_N, unscaled.
  virtual Real ff_prime(Real psi_n) const = 0;

  // Second derivatives, needed for the Newton Jacobian's diagonal term. A
  // profile that cannot supply them analytically may return a finite difference,
  // but the Newton phase then converges at a reduced rate.
  virtual Real d2p_dpsi2(Real psi_n) const = 0;
  virtual Real ff_prime_prime(Real psi_n) const = 0;

  virtual bool set_parameter(std::string_view /*name*/, Real /*value*/) {
    return false;
  }
};

// Polynomial profile in psi_N (the EFIT/FreeGS family).
//
//   p'(psi_N)  = sum_k a_k psi_N^k
//   FF'(psi_N) = sum_k b_k psi_N^k
//
// The default is the standard one-term form p' = 1 - psi_N, FF' = 1 - psi_N,
// which vanishes at the plasma boundary so the current density goes to zero
// there -- a physically sensible default and the usual textbook starting point.
class PolynomialProfile final : public IEquilibriumProfile {
 public:
  PolynomialProfile()
    : p_coeffs_{Real{1}, Real{-1}}, f_coeffs_{Real{1}, Real{-1}} {}

  PolynomialProfile(std::vector<Real> p_coeffs, std::vector<Real> f_coeffs)
    : p_coeffs_{std::move(p_coeffs)}, f_coeffs_{std::move(f_coeffs)} {
    if (p_coeffs_.empty() || f_coeffs_.empty()) {
      throw std::invalid_argument{
          "PolynomialProfile: coefficient lists must be non-empty"};
    }
  }

  Real dp_dpsi(Real psi_n) const override { return horner(p_coeffs_, psi_n); }
  Real ff_prime(Real psi_n) const override { return horner(f_coeffs_, psi_n); }
  Real d2p_dpsi2(Real psi_n) const override {
    return horner_derivative(p_coeffs_, psi_n);
  }
  Real ff_prime_prime(Real psi_n) const override {
    return horner_derivative(f_coeffs_, psi_n);
  }

  const std::vector<Real>& p_coefficients() const noexcept { return p_coeffs_; }
  const std::vector<Real>& f_coefficients() const noexcept { return f_coeffs_; }

 private:
  static Real horner(const std::vector<Real>& c, Real x) {
    Real acc = Real{0};
    for (std::size_t k = c.size(); k-- > 0;) acc = acc * x + c[k];
    return acc;
  }
  static Real horner_derivative(const std::vector<Real>& c, Real x) {
    Real acc = Real{0};
    for (std::size_t k = c.size(); k-- > 1;) {
      acc = acc * x + c[k] * static_cast<Real>(k);
    }
    return acc;
  }

  std::vector<Real> p_coeffs_;
  std::vector<Real> f_coeffs_;
};

// -- Solov'ev exact solution --------------------------------------------------
//
// The one closed-form GS equilibrium, obtained by taking both free functions
// CONSTANT in psi:
//
//   mu0 p'(psi) = -C_p     (constant)
//   F F'(psi)   = -C_f     (constant)
//
// so the GS equation becomes linear:
//
//   Delta* psi = C_p r^2 + C_f
//
// with the particular solution
//
//   psi_p = (C_p/8) r^4 + (C_f/2) z^2
//
// plus any homogeneous solution (Delta* psi_h = 0). The homogeneous space
// includes 1, r^2, z, r^2 z, and z^2 - r^2 ln r, which is what lets a Solov'ev
// equilibrium be shaped.
//
// IMPORTANT CAVEAT for verification: this solution is a low-degree polynomial in
// r and z (plus one logarithm). A sixth-order scheme reproduces the polynomial
// part EXACTLY, so a Solov'ev order study can show machine-precision error at
// every resolution and reveal nothing about the scheme's order. That is a pass,
// not a proof. The manufactured-solution tests are what actually establish
// order; Solov'ev establishes that the PHYSICS -- the r^2 source structure and
// the sign conventions -- is right.
struct SolovevSolution {
  Real c_p{Real{1}};   // coefficient of the r^2 source term
  Real c_f{Real{1}};   // constant source term
  // Homogeneous admixture, used to place the boundary where we want it.
  Real h_const{Real{0}};
  Real h_r2{Real{0}};

  Real psi(Real r, Real z) const {
    return c_p / Real{8} * r * r * r * r
         + c_f / Real{2} * z * z
         + h_const
         + h_r2 * r * r;
  }

  // Delta* psi for the above, evaluated analytically. This is the source term
  // the solver must reproduce.
  Real delta_star(Real r, Real /*z*/) const {
    // Delta*(r^4/8)   = (1/8)(12 r^2 - 4 r^2) = r^2
    // Delta*(z^2/2)   = 1
    // Delta*(1)       = 0
    // Delta*(r^2)     = 2 - 2 = 0
    return c_p * r * r + c_f;
  }

  Real dpsi_dr(Real r, Real /*z*/) const {
    return c_p / Real{2} * r * r * r + Real{2} * h_r2 * r;
  }
  Real dpsi_dz(Real /*r*/, Real z) const { return c_f * z; }
};

}  // namespace quasar::equilibrium
