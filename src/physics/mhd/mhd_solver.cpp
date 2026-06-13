#include "quasar/physics/mhd/mhd_solver.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_geometric_source.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace quasar::mhd {

namespace {

// Resolve the working-grid ghost halo authoritatively from the reconstruction
// scheme. A deck may leave grid.nghost at 0 (let the solver size it); if it set
// a positive value it must be at least the scheme's required halo, else the
// reconstruction would read past the allocated ghosts.
Grid2D resolve_working_grid(const Grid2D& deck_grid, int required) {
  Grid2D g = deck_grid;
  if (deck_grid.nghost > 0 && deck_grid.nghost < required) {
    throw std::invalid_argument{
        "MhdSolver2D: grid nghost (" + std::to_string(deck_grid.nghost) +
        ") is smaller than the reconstruction scheme's required halo (" +
        std::to_string(required) + ")"};
  }
  g.nghost = std::max(deck_grid.nghost, required);
  g.validate();
  return g;
}

// Resolve a pluggable scheme by registry name. Throws std::invalid_argument on an
// unknown deck name (Registry::create throws std::out_of_range, which we rephrase
// for the deck). Used for every MHD scheme axis, including the reconstruction
// scheme that is built first because its required_nghost() fixes the working grid
// (so MhdSolver2D's member initializer list can size every field/register at the
// resolved grid).
template <class Base>
std::unique_ptr<Base> make_scheme(const std::string& name, const char* what) {
  try {
    auto p = Registry<Base>::instance().create(name);
    if (!p) {
      throw std::invalid_argument{std::string{"MhdSolver2D: unknown "} + what + " '" + name + "'"};
    }
    return p;
  } catch (const std::out_of_range&) {
    throw std::invalid_argument{std::string{"MhdSolver2D: unknown "} + what + " '" + name + "'"};
  }
}

}  // namespace

MhdSolver2D::MhdSolver2D(MhdConfig cfg)
  : cfg_{cfg},
    reconstruction_{make_scheme<numerics::IFluxReconstruction>(cfg.reconstruction,
                                                              "reconstruction scheme")},
    grid_{resolve_working_grid(cfg.grid, reconstruction_->required_nghost())},
    rk_{MhdField2D<Real>{grid_}, MhdField2D<Real>{grid_}, MhdField2D<Real>{grid_}},
    residual_{grid_},
    flux_{grid_},
    ifx_{grid_, 0},
    ify_{grid_, 1},
    emf_{grid_} {
  // Field-split background B0: allocate at the working grid and mark active iff
  // the deck enabled it. Otherwise leave b0_ default-constructed (inactive =>
  // zero-B0 fast path that is bit-identical to the no-background solver). The
  // actual B0 values are seeded from Python via seed_background().
  if (cfg_.background.enabled) {
    // The cylindrical geometric source (MhdGeometricSource / geometric_source_kernel)
    // builds its 1/r curvature terms from the stored perturbation b only, while the
    // radial flux uses the total field B = B0 + b. Combining a static background with
    // cylindrical geometry would therefore apply an inconsistent (flux-from-B,
    // source-from-b) update, so reject it until the geometric source folds in B0.
    if (is_cylindrical()) {
      throw std::invalid_argument{
          "MhdSolver2D: a static background field (background.enabled) is not yet "
          "supported with cylindrical geometry (the geometric source does not fold "
          "in B0)"};
    }
    b0_ = MhdBackgroundField<Real>{grid_};
    b0_.active = true;
  }

  // Resolve the remaining schemes by registry string (no if/else over types).
  riemann_ = make_scheme<numerics::IRiemannSolver>(cfg_.riemann, "Riemann solver");
  ct_ = make_scheme<numerics::ICtScheme>(cfg_.ct, "CT scheme");
  integrator_ = make_scheme<numerics::ISsprkIntegrator>(cfg_.integrator, "integrator");
  positivity_ = make_scheme<numerics::IPositivityLimiter>(cfg_.positivity, "positivity limiter");

  // Per-side fluid + field boundaries, built through the registry (the documented
  // pluggable path); concrete BCs self-register in src/physics/mhd/mhd_boundary.cpp.
  for (int side = 0; side < 4; ++side) {
    fluid_bcs_[side] =
        make_scheme<boundary::IMhdFluidBoundary>(cfg_.boundary.fluid[side], "fluid boundary");
    field_bcs_[side] =
        make_scheme<boundary::IMhdFieldBoundary>(cfg_.boundary.field[side], "field boundary");
  }
}

backend::DeviceBuffer<Real>& MhdSolver2D::component_buffer(MhdField2D<Real>& f,
                                                          std::string_view component) {
  if (component == "rho") return f.rho;
  if (component == "mx") return f.mx;
  if (component == "my") return f.my;
  if (component == "mz") return f.mz;
  if (component == "energy") return f.energy;
  if (component == "bx" || component == "bx_face") return f.bx_face;
  if (component == "by" || component == "by_face") return f.by_face;
  if (component == "bz" || component == "bz_cell") return f.bz_cell;
  throw std::invalid_argument{"MhdSolver2D: unknown state component '" +
                              std::string{component} + "'"};
}

void MhdSolver2D::seed_state(std::string_view component, const std::vector<Real>& host_buf) {
  auto& buf = component_buffer(state(), component);
  if (host_buf.size() != buf.size()) {
    throw std::invalid_argument{
        "MhdSolver2D::seed_state: host buffer size (" + std::to_string(host_buf.size()) +
        ") does not match the component storage size (" + std::to_string(buf.size()) + ")"};
  }
  buf.copy_from_host(host_buf.data(), host_buf.size());
}

void MhdSolver2D::seed_background(std::string_view component,
                                  const std::vector<Real>& host_buf) {
  if (!b0_.active) {
    throw std::logic_error{
        "MhdSolver2D::seed_background: the field-split background is not enabled "
        "(set background.enabled in the deck before seeding B0)"};
  }
  // Map the deck-facing spelling to its background buffer, mirroring the
  // magnetic-component aliases accepted by seed_state.
  backend::DeviceBuffer<Real>* buf = nullptr;
  if (component == "b0x" || component == "b0x_face") {
    buf = &b0_.b0x_face;
  } else if (component == "b0y" || component == "b0y_face") {
    buf = &b0_.b0y_face;
  } else if (component == "b0z" || component == "b0z_cell") {
    buf = &b0_.b0z_cell;
  } else {
    throw std::invalid_argument{"MhdSolver2D::seed_background: unknown background "
                                "component '" + std::string{component} + "'"};
  }
  if (host_buf.size() != buf->size()) {
    throw std::invalid_argument{
        "MhdSolver2D::seed_background: host buffer size (" +
        std::to_string(host_buf.size()) +
        ") does not match the component storage size (" +
        std::to_string(buf->size()) + ")"};
  }
  buf->copy_from_host(host_buf.data(), host_buf.size());
}

bool MhdSolver2D::has_background() const noexcept { return cfg_.background.enabled; }

std::vector<Real> MhdSolver2D::state_component_to_host(std::string_view component) const {
  // const-correct read: the buffers are only read, but component_buffer returns a
  // mutable ref, so cast away constness on the live state for the staging read.
  auto& self = const_cast<MhdSolver2D&>(*this);
  const auto& buf = component_buffer(self.state(), component);
  std::vector<Real> out(buf.size());
  buf.copy_to_host(out.data(), out.size());
  return out;
}

int MhdSolver2D::reconstruction_order() const {
  // Map the scheme's required halo to a spatial order for the device path:
  //   nghost 2 -> order 2 (muscl_minmod), 3 -> 5 (mp5), 4 -> 7 (mp7).
  switch (reconstruction_->required_nghost()) {
    case 3: return 5;
    case 4: return 7;
    default: return 2;
  }
}

void MhdSolver2D::fill_ghosts(MhdField2D<Real>& u) const {
  for (int side = 0; side < 4; ++side) {
    fluid_bcs_[side]->fill_ghosts(u, static_cast<Side>(side));
    field_bcs_[side]->fill_ghosts(u, static_cast<Side>(side));
  }
}

BoundaryFlags4 MhdSolver2D::boundary_flags() const {
  // Per-side one-sided flag from the field-boundary name: non-periodic => 1
  // (the device path drops the ghost-gradient dependence at that side),
  // periodic => 0 (two-sided wrap). cfg_.boundary.field is ordered
  // [x_lo, x_hi, y_lo, y_hi], matching BoundaryFlags4::side. An all-periodic
  // deck yields all-zero flags (the periodic fast path).
  BoundaryFlags4 flags{};
  for (int side = 0; side < 4; ++side) {
    flags.side[side] =
        boundary::mhd_boundary_is_periodic(cfg_.boundary.field[side]) ? 0 : 1;
  }
  return flags;
}

void MhdSolver2D::compute_residual(const MhdField2D<Real>& u, MhdField2D<Real>& dudt) {
  // The residual is built into dudt = L(u). Zero it, fill ghosts of u (the
  // boundary fill mutates ghost layers only, so a const-ref input is fine to
  // refresh), then accumulate -div F per direction and add the geometric source.
  backend::device_memset_async(dudt.rho.device_ptr(), 0, dudt.rho.bytes(), nullptr);
  backend::device_memset_async(dudt.mx.device_ptr(), 0, dudt.mx.bytes(), nullptr);
  backend::device_memset_async(dudt.my.device_ptr(), 0, dudt.my.bytes(), nullptr);
  backend::device_memset_async(dudt.mz.device_ptr(), 0, dudt.mz.bytes(), nullptr);
  backend::device_memset_async(dudt.energy.device_ptr(), 0, dudt.energy.bytes(), nullptr);
  backend::device_memset_async(dudt.bx_face.device_ptr(), 0, dudt.bx_face.bytes(), nullptr);
  backend::device_memset_async(dudt.by_face.device_ptr(), 0, dudt.by_face.bytes(), nullptr);
  backend::device_memset_async(dudt.bz_cell.device_ptr(), 0, dudt.bz_cell.bytes(), nullptr);

  auto& mutable_u = const_cast<MhdField2D<Real>&>(u);
  fill_ghosts(mutable_u);

  const int order = reconstruction_order();
  const Real gamma = cfg_.gamma;
  const BoundaryFlags4 flags = boundary_flags();

  // dir = 0 (x faces) then dir = 1 (y faces). Each direction: reconstruct L/R
  // interface states, form the HLLD flux, then accumulate the conservative flux
  // difference (-dF/dx) into dudt. This writes ALL 8 slots, including the face-B
  // slots bx_face/by_face -- but the CT invariant forbids advancing the staggered
  // poloidal field by the (non-div-free) Godunov flux divergence, so those two
  // slots are OVERWRITTEN below by the pure EMF-curl rate. Only the 5 fluid vars
  // and the cell-centered toroidal bz_cell keep the flux-difference contribution.
  launch_mhd_reconstruct(u, b0_, 0, ifx_, order, flags, gamma, nullptr);
  launch_mhd_hlld_flux(ifx_, b0_, 0, flux_, gamma, nullptr);
  launch_mhd_flux_difference(flux_, 0, dudt, nullptr);

  launch_mhd_reconstruct(u, b0_, 1, ify_, order, flags, gamma, nullptr);
  launch_mhd_hlld_flux(ify_, b0_, 1, flux_, gamma, nullptr);
  launch_mhd_flux_difference(flux_, 1, dudt, nullptr);

  // Build the corner EMF from the two-direction interface states (kinematic
  // E = -(v x B)). Then write the discrete curl of that EMF as a RATE into the
  // face-B slots of dudt (overwrite, no dt), discarding the flux-difference
  // contamination in bx_face/by_face. Because face B is now carried as an ordinary
  // residual component, combine_stage advances it through the SAME SSP-RK3 convex
  // combination as the other 7 components -- and since the curl stencil telescopes
  // discretely (div(curl) = 0) and a convex combination of div-free fields stays
  // div-free, div(B) is preserved at round-off through every stage. There is
  // therefore NO separate launch_mhd_face_b_update step (that was the double-count
  // bug: face B advanced by both the flux divergence and the CT curl).
  launch_mhd_ct_emf(u, b0_, ifx_, ify_, flags, emf_, gamma, nullptr);
  launch_mhd_emf_curl_rate(emf_, dudt, grid_, nullptr);

  // Cylindrical (r,z): add the axisymmetric geometric source S(u) into dudt.
  if (is_cylindrical()) {
    MhdGeometricSource::add(u, dudt, grid_, gamma);
  }

  // The interface-state host accessors and the next stage's reads are correct
  // only after the queued kernels finish; block once here so the seam is
  // sequentially consistent for the integrator.
  backend::device_synchronize(nullptr);
}

void MhdSolver2D::combine_stage(int stage, Real dt) {
  // Standard SSP-RK3 (Shu-Osher), with rk_[0]=U^n (live), rk_[1]=U1, rk_[2]=U2:
  //   stage 0: U1   = 1*U^n + 0*U1 + dt*L(U^n)         -> out rk_[1]
  //   stage 1: U2   = 3/4 U^n + 1/4 U1 + 1/4 dt*L(U1)  -> out rk_[2]
  //   stage 2: U^n+1= 1/3 U^n + 2/3 U2 + 2/3 dt*L(U2)  -> out rk_[0] (live)
  // launch_mhd_rk_stage(out, un, ustage, dudt, a, b, c_dt): out = a*un + b*ustage
  // + c_dt*dudt. The integrator calls compute_residual(stage input, residual_)
  // before each combine, so residual_ holds L of this stage's input field.
  MhdField2D<Real>* out = nullptr;
  Real a = Real{0}, b = Real{0}, c = Real{0};
  switch (stage) {
    case 0:
      out = &rk_[1];
      a = Real{1};
      b = Real{0};
      c = dt;
      break;
    case 1:
      out = &rk_[2];
      a = Real{3} / Real{4};
      b = Real{1} / Real{4};
      c = (Real{1} / Real{4}) * dt;
      break;
    case 2:
      out = &rk_[0];
      a = Real{1} / Real{3};
      b = Real{2} / Real{3};
      c = (Real{2} / Real{3}) * dt;
      break;
    default:
      throw std::invalid_argument{"MhdSolver2D::combine_stage: stage must be 0, 1, or 2"};
  }
  // For stage 0 the "ustage" operand is unused (b == 0); reuse rk_[1] as a safe
  // alias so we never dereference an uninitialized field. un is always U^n.
  const MhdField2D<Real>& un = rk_[0];
  const MhdField2D<Real>& ustage = (stage == 0) ? rk_[1] : (stage == 1) ? rk_[1] : rk_[2];

  // ONE uniform combine over all 8 components, INCLUDING the face-B slots. Because
  // compute_residual already wrote the pure CT EMF-curl rate into residual_'s
  // bx_face/by_face slots (overwriting the flux-difference contamination), this
  // single convex combination advances face B by constrained transport alone --
  // there is deliberately NO separate launch_mhd_face_b_update call. The Shu-Osher
  // weights stay consistent across every component, and div(B) is preserved at
  // round-off (convex combination of div-free fields + c*curl rate is div-free,
  // since div(curl) telescopes to zero discretely).
  launch_mhd_rk_stage(*out, un, ustage, residual_, a, b, c, nullptr);

  // Positivity: floor density / pressure on the stage output. Prefer the
  // pluggable limiter; it delegates to the same device floors kernel.
  positivity_->apply(*out, cfg_.rho_floor, cfg_.p_floor, cfg_.gamma);

  // Re-derive the ghost layers of the stage output from the configured BCs. This
  // is load-bearing for TWO reasons:
  //   1. The rk_stage combine runs over the WHOLE storage (including ghosts), so
  //      the ghost FACE-B slots were just advanced by the leftover (contaminated)
  //      flux-difference residual that launch_mhd_emf_curl_rate only overwrote on
  //      the interior. Refilling ghosts here re-derives those ghost faces from the
  //      BC (periodic wrap etc.), discarding the bad residual so they match the
  //      updated interior -- which keeps div(B) at round-off in the boundary ring.
  //   2. It leaves every stage output ghost-consistent, so the NEXT stage's
  //      reconstruction reads correct periodic-wrapped ghost CELLS and the
  //      boundary-face fluxes telescope (conserving mass/momentum/energy on a
  //      periodic grid). compute_residual still refills the input ghosts at its
  //      top; this makes the field consistent at stage completion too, so the
  //      final state() handed back after step() is ghost-consistent for any
  //      downstream reader (e.g. divergence_b_max).
  fill_ghosts(*out);

  backend::device_synchronize(nullptr);
}

MhdField2D<Real>& MhdSolver2D::rk_register(int k) {
  if (k < 0 || k >= kNumRkRegisters) {
    throw std::out_of_range{"MhdSolver2D::rk_register: index out of range"};
  }
  return rk_[k];
}

Real MhdSolver2D::cfl_limit() const {
  // Reduce the maximum interior ADDITIVE signal rate
  //   (|v_x| + c_fast_x)/dx + (|v_y| + c_fast_y)/dy
  // on device, then return cfl / max_rate. The additive (rather than per-direction
  // max over min(dx,dy)) bound is the correct stability limit for the UNSPLIT
  // residual, which sums both directional flux differences into one dudt per
  // SSP-RK3 stage; a per-direction max would accept up to ~2x the stable step on
  // an isotropic grid. The device reduction is b0-aware (it folds the total field
  // B = B0 + b) and skips cells with non-positive / non-finite rho. Cylindrical
  // uses (dr, dz) = (dx, dy) on the (r,z) grid, which Grid2D dx()/dy() report.
  //
  // launch_mhd_cfl_max_rate takes the field by const ref; state() is a mutable
  // accessor, so reuse the established const_cast pattern to obtain the live
  // field reference without mutating it.
  auto& self = const_cast<MhdSolver2D&>(*this);
  Real max_rate = Real{0};
  launch_mhd_cfl_max_rate(self.state(), b0_, cfg_.gamma, cfl_scratch_, &max_rate, nullptr);

  if (!(max_rate > Real{0}) || !std::isfinite(max_rate)) {
    // Reached only when every interior cell has a non-positive / non-finite
    // signal rate -- e.g. a pressureless, field-free, motionless state (the
    // positivity floor keeps rho>0 and p>0, so a normal run always yields a
    // finite positive rate). Any dt is stable for a truly signal-free field;
    // return a finite, dimensionally-consistent dt using the additive rate with
    // a unit reference speed (cfl / (1/dx + 1/dy)) so the fallback is a TIME like
    // the main path, not a length.
    const Real ref_rate = Real{1} / grid_.dx() + Real{1} / grid_.dy();
    return cfg_.cfl / ref_rate;
  }
  return cfg_.cfl / max_rate;
}

Real MhdSolver2D::divergence_b_max() const {
  // The cell-centered divB stencil at the last interior column/row reads a GHOST
  // face (bx_face(i+1,j) / by_face(i,j+1)). Refill the field ghosts from the
  // configured BC first so those ghost faces match the current interior under the
  // periodic wrap (or other closure); measuring a ghost-stale field would
  // otherwise report a spurious nonzero div concentrated in the boundary ring,
  // even when the interior is exactly div-free. Logically const (only ghost
  // layers are touched, and they are derived state), hence the const_cast.
  auto& self = const_cast<MhdSolver2D&>(*this);
  self.fill_ghosts(self.state());
  backend::device_synchronize(nullptr);
  return ct_->divergence_b_linf(self.state());
}

void MhdSolver2D::check_cfl(Real dt) const {
  if (dt <= Real{0}) {
    throw std::invalid_argument{"MhdSolver2D: dt must be positive"};
  }
  if (dt > cfl_limit()) {
    throw std::invalid_argument{
        "MhdSolver2D: dt exceeds the CFL stability limit for this grid and scheme"};
  }
}

void MhdSolver2D::step(Real dt) {
  check_cfl(dt);
  integrator_->advance(*this, dt);
}

void MhdSolver2D::step_unchecked(Real dt) {
  // CFL pre-validated by the caller (the auto-dt loop just called cfl_limit());
  // skip the redundant full-grid reduction inside check_cfl(). Still guard the
  // sign so a bad caller cannot advance with a non-positive dt.
  if (dt <= Real{0}) {
    throw std::invalid_argument{"MhdSolver2D: dt must be positive"};
  }
  integrator_->advance(*this, dt);
}

void MhdSolver2D::advance(Real t_end, Real dt) {
  if (dt <= Real{0}) {
    throw std::invalid_argument{"MhdSolver2D::advance: dt must be positive"};
  }
  // Drive the loop by an integer step count so repeated += dt cannot drift the
  // termination by up to a full step from floating-point accumulation.
  const long n_steps = (t_end > Real{0})
                           ? static_cast<long>(std::ceil(t_end / dt))
                           : 0L;
  for (long s = 0; s < n_steps; ++s) {
    step(dt);
  }
}

}  // namespace quasar::mhd
