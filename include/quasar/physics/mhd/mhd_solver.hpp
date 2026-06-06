#pragma once

// Ideal-MHD solver driver for the 2D conservative finite-difference + CT vertical
// slice. This is the MHD analogue of pic::EmPic2D3V: a thin driver that resolves
// every pluggable scheme (flux reconstruction, Riemann solver, CT scheme, SSP-RK
// integrator, positivity limiter, per-side fluid/field boundaries) by deck-facing
// string name through the plugin registry, owns ALL device scratch (the live
// state, the SSP-RK stage registers, the residual register, the per-direction
// interface states, a flux scratch field, and the corner EMF), and exposes the
// residual/stage seam the SSP-RK integrator drives.
//
// The integrator (numerics::ISsprkIntegrator) is intentionally state-free: it
// only sequences calls to compute_residual / combine_stage / rk_register /
// residual_register on this solver, so the register routing and all buffer
// ownership live here. See docs/dev-guide and the kernels.hpp launch ABI for the
// device seam this driver calls.

#include "quasar/boundary/mhd_boundary.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/ct_scheme.hpp"
#include "quasar/numerics/flux_reconstruction.hpp"
#include "quasar/numerics/interface_states.hpp"
#include "quasar/numerics/positivity_limiter.hpp"
#include "quasar/numerics/riemann_solver.hpp"
#include "quasar/numerics/ssprk_integrator.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace quasar::mhd {

// Deck-facing configuration for the MHD solver. Every scheme axis is a registry
// name so a new scheme is selectable without touching this struct or the driver.
struct MhdConfig {
  Grid2D grid{};
  Real gamma{Real{5} / Real{3}};
  std::string geometry{"cartesian"};
  std::string reconstruction{"mp7"};
  std::string riemann{"hlld"};
  std::string integrator{"ssprk3"};
  std::string ct{"fd_ct_christlieb"};
  std::string positivity{"troubled_cell"};
  Real rho_floor{Real{1e-8}};
  Real p_floor{Real{1e-9}};
  // CFL safety factor (deck: numerics.cfl). The C++ contract carried no cfl
  // field, but the Python deck passes one; added here so cfl_limit() scales the
  // stable step by it. Default 0.4 is a conservative MHD multidimensional value.
  Real cfl{Real{0.4}};
  boundary::MhdBoundarySpec boundary{};
};

class MhdSolver2D {
 public:
  explicit MhdSolver2D(MhdConfig cfg);

  Grid2D grid() const noexcept { return grid_; }
  MhdField2D<Real>& state() noexcept { return rk_[0]; }
  const MhdConfig& config() const noexcept { return cfg_; }

  // Stage a host buffer (sized grid.storage_size()) into the named live-state
  // component. Magnetic spellings accept both the staggered name and the short
  // name: "bx"/"bx_face", "by"/"by_face", "bz"/"bz_cell".
  void seed_state(std::string_view component, const std::vector<Real>& host_buf);

  // One SSP-RK3 step; rejects a dt above cfl_limit() (an over-CFL step diverges).
  void step(Real dt);
  // Loop step() to t_end, CFL-checked each step.
  void advance(Real t_end, Real dt);

  // cfl * min(dx,dy) / max_cells(|v_dir| + c_fast,dir). Cylindrical uses (dr,dz).
  Real cfl_limit() const;

  // Max |div B| over the interior; delegates to the CT scheme diagnostic.
  Real divergence_b_max() const;

  // Read a state component back to host (sized grid.storage_size()). The "bx"/
  // "by"/"bz" spellings sample the face/cell B to cell centers for the .npz.
  std::vector<Real> state_component_to_host(std::string_view component) const;

  // -- SSP-RK integrator seam (the integrator only calls these) ---------------
  // dudt := L(u): the conservative residual (-div F + geometric source).
  void compute_residual(const MhdField2D<Real>& u, MhdField2D<Real>& dudt);
  // Apply the Shu-Osher combine + CT face-B update + positivity for `stage`.
  void combine_stage(int stage, Real dt);
  MhdField2D<Real>& rk_register(int k);
  MhdField2D<Real>& residual_register() { return residual_; }
  int n_rk_registers() const noexcept { return kNumRkRegisters; }

 private:
  // Map a component spelling to its DeviceBuffer in a field (throws on unknown).
  static backend::DeviceBuffer<Real>& component_buffer(MhdField2D<Real>& f,
                                                       std::string_view component);
  void check_cfl(Real dt) const;
  bool is_cylindrical() const noexcept { return cfg_.geometry == "cylindrical"; }
  void fill_ghosts(MhdField2D<Real>& u) const;
  int reconstruction_order() const;

  static constexpr int kNumRkRegisters = 3;  // live state + 2 SSP-RK3 stage regs

  MhdConfig cfg_{};
  // The reconstruction scheme is declared (and built) FIRST: its required_nghost()
  // fixes the working grid, which in turn sizes every field/register below. C++
  // initializes members in declaration order, so this must precede grid_.
  std::unique_ptr<numerics::IFluxReconstruction> reconstruction_{};
  Grid2D grid_{};
  // rk_[0] is the live state U; rk_[1], rk_[2] are the SSP-RK3 stage registers.
  std::array<MhdField2D<Real>, kNumRkRegisters> rk_{};
  MhdField2D<Real> residual_{};  // dudt accumulator (the L(u) register)
  MhdField2D<Real> flux_{};      // per-direction interface flux scratch
  numerics::MhdInterfaceStates<Real> ifx_;  // dir=0 reconstructed interface states
  numerics::MhdInterfaceStates<Real> ify_;  // dir=1 reconstructed interface states
  EmfField2D<Real> emf_{};       // corner-staggered CT EMF

  std::unique_ptr<numerics::IRiemannSolver> riemann_{};
  std::unique_ptr<numerics::ICtScheme> ct_{};
  std::unique_ptr<numerics::ISsprkIntegrator> integrator_{};
  std::unique_ptr<numerics::IPositivityLimiter> positivity_{};
  std::array<std::unique_ptr<boundary::IMhdFluidBoundary>, 4> fluid_bcs_{};
  std::array<std::unique_ptr<boundary::IMhdFieldBoundary>, 4> field_bcs_{};
};

}  // namespace quasar::mhd
