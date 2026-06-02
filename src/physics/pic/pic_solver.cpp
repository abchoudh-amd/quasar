#include "quasar/physics/pic/pic_solver.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/core/registry.hpp"

#include "quasar/physics/pic/kernels.hpp"

#include <stdexcept>
#include <string>

namespace quasar::numerics {

template <>
void YeeFdtd2D<2>::advance_b(YeeField2D<Real>& f, Real dt) const {
  ::launch_pic_fdtd_b_order2(f.grid, f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                             f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(), dt, nullptr);
}

template <>
void YeeFdtd2D<4>::advance_b(YeeField2D<Real>& f, Real dt) const {
  ::launch_pic_fdtd_b_order4(f.grid, f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                             f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(), dt, nullptr);
}

template <>
void YeeFdtd2D<2>::advance_e(YeeField2D<Real>& f, const JField2D<Real>& j, Real dt) const {
  ::launch_pic_fdtd_e_order2(f.grid, f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(),
                             f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                             j.jx.device_ptr(), j.jy.device_ptr(), j.jz.device_ptr(), dt, nullptr);
}

template <>
void YeeFdtd2D<4>::advance_e(YeeField2D<Real>& f, const JField2D<Real>& j, Real dt) const {
  ::launch_pic_fdtd_e_order4(f.grid, f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(),
                             f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                             j.jx.device_ptr(), j.jy.device_ptr(), j.jz.device_ptr(), dt, nullptr);
}

template <>
void BorisPusher<1>::push(pic::ParticleSpecies& s, const YeeField2D<Real>& f,
                          const YeeField2D<Real>& ext, Real dt) const {
  ::launch_pic_gather_push_shape1(f.grid, s, f, ext, periodic_x_ ? 1 : 0,
                                  periodic_y_ ? 1 : 0, dt, nullptr);
}

template <>
void BorisPusher<2>::push(pic::ParticleSpecies& s, const YeeField2D<Real>& f,
                          const YeeField2D<Real>& ext, Real dt) const {
  ::launch_pic_gather_push_shape2(f.grid, s, f, ext, periodic_x_ ? 1 : 0,
                                  periodic_y_ ? 1 : 0, dt, nullptr);
}

template <>
void Esirkepov2D<1>::deposit(const pic::ParticleSpecies& s, JField2D<Real>& j, Real dt) const {
  ::launch_pic_deposit_shape1(j.grid, s, j, dt, periodic_x_ ? 1 : 0,
                              periodic_y_ ? 1 : 0, nullptr);
}

template <>
void Esirkepov2D<2>::deposit(const pic::ParticleSpecies& s, JField2D<Real>& j, Real dt) const {
  ::launch_pic_deposit_shape2(j.grid, s, j, dt, periodic_x_ ? 1 : 0,
                              periodic_y_ ? 1 : 0, nullptr);
}

}  // namespace quasar::numerics

// Register the concrete field-solver / pusher / deposit schemes under deck-facing
// names so EmPic2D3V builds them through the registry instead of an if/else ladder
// over the integer order/shape. These registrations live in this TU (whose
// EmPic2D3V symbols are externally referenced), so the static initializers are
// never dropped by the linker — no WHOLE_ARCHIVE needed.
QUASAR_REGISTER_FIELD_SOLVER("yee_o2", ::quasar::numerics::YeeFdtd2D<2>)
QUASAR_REGISTER_FIELD_SOLVER("yee_o4", ::quasar::numerics::YeeFdtd2D<4>)
QUASAR_REGISTER_PUSHER("boris_cic", ::quasar::numerics::BorisPusher<1>)
QUASAR_REGISTER_PUSHER("boris_tsc", ::quasar::numerics::BorisPusher<2>)
QUASAR_REGISTER_DEPOSIT("esirkepov_cic", ::quasar::numerics::Esirkepov2D<1>)
QUASAR_REGISTER_DEPOSIT("esirkepov_tsc", ::quasar::numerics::Esirkepov2D<2>)

namespace quasar::pic {

namespace {

// Translates the deck's order/shape vocabulary to the registry names registered
// just above. Kept adjacent to the QUASAR_REGISTER_* lines so a new scheme's
// name appears in exactly one place: register it, then add its vocabulary entry
// here. The driver constructor never builds these strings itself.
std::string field_solver_name(int fdtd_order) {
  return "yee_o" + std::to_string(fdtd_order);
}
std::string pusher_name(const std::string& shape) { return "boris_" + shape; }
std::string deposit_name(const std::string& shape) { return "esirkepov_" + shape; }

}  // namespace

EmPic2D3V::EmPic2D3V(EmPicConfig cfg)
  : cfg_{cfg},
    grid_{cfg.grid},
    fields_{grid_},
    external_fields_{grid_},
    current_{grid_} {
  // Build per-side boundary conditions through the registry (the documented
  // pluggable construction path); concrete BCs self-register in src/boundary.
  // The FDTD stencil is ghost-aware, so these field BCs run every step (see step).
  for (int side = 0; side < 4; ++side) {
    particle_bcs_[side] = Registry<boundary::IParticleBoundary>::instance().create(
        cfg_.boundary.particle[side]);
    field_bcs_[side] = Registry<boundary::IFieldBoundary>::instance().create(
        cfg_.boundary.field[side]);
    // The FDTD order fixes how many boundary layers a one-sided/characteristic
    // closure must rewrite; let each field BC pick its order-dependent kernel.
    field_bcs_[side]->configure(cfg_.fdtd_order);
  }
  // Corner ownership: the field-BC corrections run x-faces before y-faces, so the
  // x-face owns the doubly-tangential ez at a corner. Tell each y-face which of
  // its end columns abut a non-periodic x-face so it can skip ez there (the
  // x-face's pin / Mur already handled it). PEC y-faces ignore this (idempotent).
  const auto& fb = cfg_.boundary.field;
  const bool x_lo_np = fb[0] != "periodic";
  const bool x_hi_np = fb[1] != "periodic";
  field_bcs_[2]->set_corner_skip(x_lo_np, x_hi_np);  // y_lo
  field_bcs_[3]->set_corner_skip(x_lo_np, x_hi_np);  // y_hi
  if (cfg_.fdtd_order == 4 && grid_.nghost < 2) {
    // The 4th-order staggered curl reads two cells past each boundary, so the
    // ghost-aware stencil needs at least two ghost layers.
    throw std::invalid_argument{"EmPic2D3V: fdtd_order 4 requires grid nghost >= 2"};
  }
  // Build the order/shape-templated schemes through the registry (same pluggable
  // path as the BCs/filters). The vocabulary->name mapping lives next to the
  // registrations above, so the driver passes a resolved name through verbatim;
  // Registry::create throws on an unregistered name.
  field_solver_ = Registry<numerics::IFieldSolver>::instance().create(
      field_solver_name(cfg_.fdtd_order));
  pusher_ = Registry<numerics::IParticlePusher>::instance().create(
      pusher_name(cfg_.shape));
  deposit_ = Registry<numerics::IDepositScheme>::instance().create(
      deposit_name(cfg_.shape));
  // The deposit/gather wrap an axis only when both of its sides are periodic; a
  // wall on either side switches that axis to ghost-cell deposition + specular
  // fold-back (deposit) and ghost-clamped interpolation (gather), so neither
  // wraps across a non-periodic boundary.
  const auto& pb = cfg_.boundary.particle;
  const bool periodic_x = pb[0] == "periodic" && pb[1] == "periodic";
  const bool periodic_y = pb[2] == "periodic" && pb[3] == "periodic";
  deposit_->set_periodic_axes(periodic_x, periodic_y);
  pusher_->set_periodic_axes(periodic_x, periodic_y);

  // Build the current-smoothing pipeline from the deck via the registry (same
  // pluggable path as the BCs); concrete filters self-register in src/numerics.
  for (const auto& spec : cfg_.filters) {
    auto filter = Registry<numerics::ICurrentFilter>::instance().create(spec.name);
    filter->set_passes(spec.passes);
    filters_.add(std::move(filter));
  }
}

void EmPic2D3V::add_species(ParticleSpecies s) {
  s.set_grid(grid_);
  species_.push_back(std::move(s));
}

bool EmPic2D3V::has_absorbing_boundary() const noexcept {
  for (int side = 0; side < 4; ++side) {
    if (cfg_.boundary.particle[side] == "absorbing") {
      return true;
    }
  }
  return false;
}

void EmPic2D3V::fill_field_ghosts() {
  for (int side = 0; side < 4; ++side) {
    field_bcs_[side]->fill_ghosts(fields_, static_cast<Side>(side));
  }
}

void EmPic2D3V::correct_field_boundaries_b(Real dt) {
  for (int side = 0; side < 4; ++side) {
    field_bcs_[side]->correct_after_b(fields_, static_cast<Side>(side), dt);
  }
}

void EmPic2D3V::correct_field_boundaries_e(Real dt) {
  for (int side = 0; side < 4; ++side) {
    field_bcs_[side]->correct_after_e(fields_, static_cast<Side>(side), dt);
  }
}

void EmPic2D3V::apply_particle_bcs(ParticleSpecies& s) {
  // Dispatch through the registry-built IParticleBoundary objects. The former
  // "heisenbug" that made this path fault was a registry factory collision
  // (identical-code folding collapsed the stateless make_unique lambdas, so
  // create() returned the wrong concrete type); it is fixed in
  // core/registry.hpp via type-keyed registration.
  for (int side = 0; side < 4; ++side) {
    particle_bcs_[side]->apply(s, static_cast<Side>(side));
  }
}

void EmPic2D3V::step(Real dt) {
  // Clear the current accumulators asynchronously on the default stream. The
  // deposit and every downstream kernel also run on the default stream, so the
  // clear is correctly ordered before them without a host-side block (the old
  // synchronous hipMemset stalled the host every step).
  backend::device_memset_async(current_.jx.device_ptr(), 0, current_.jx.bytes(), nullptr);
  backend::device_memset_async(current_.jy.device_ptr(), 0, current_.jy.bytes(), nullptr);
  backend::device_memset_async(current_.jz.device_ptr(), 0, current_.jz.bytes(), nullptr);

  // The FDTD stencil reads neighbours through ghost cells, so the per-side ghost
  // fill must run before each curl. Only a periodic side fills ghosts (copying the
  // opposite interior edge, reproducing the implicit wrap bit-for-bit); pec and
  // outflow leave fill_ghosts a no-op and instead correct the boundary nodes after
  // each curl (see correct_field_boundaries_b/e below).
  fill_field_ghosts();
  field_solver_->advance_b(fields_, dt);
  // One-sided / characteristic field BCs (pec, outflow) overwrite the boundary
  // row the interior B-curl just computed from stale ghosts. Periodic is a no-op.
  correct_field_boundaries_b(dt);
  for (auto& s : species_) {
    pusher_->push(s, fields_, external_fields_, dt);
    apply_particle_bcs(s);
    deposit_->deposit(s, current_, dt);
  }
  // On reflecting (specular) sides the deposit left boundary-crossing current in
  // the ghost cells; the BC's fold_current hook folds it back into the interior
  // as image current before the filter / E-update read J. Periodic/absorbing
  // BCs leave fold_current a no-op.
  for (int side = 0; side < 4; ++side) {
    particle_bcs_[side]->fold_current(current_, static_cast<Side>(side));
  }
  filters_.apply(current_, cfg_.boundary);
  fill_field_ghosts();
  field_solver_->advance_e(fields_, current_, dt);
  // Outflow Mur reads the just-updated adjacent interior E node, so the E-side
  // correction runs after advance_e. PEC pins tangential E here; periodic no-op.
  correct_field_boundaries_e(dt);

  // Reclaim slots vacated by absorbing boundaries on a fixed cadence so the
  // push/deposit/gather kernels stop iterating over dead particles. Only
  // meaningful when a boundary can actually kill particles; periodic/specular
  // runs never shrink, so skip the work entirely.
  ++step_count_;
  constexpr std::size_t kCompactEvery = 64;
  const bool cadence_hit = step_count_ % kCompactEvery == 0;
  if (has_absorbing_boundary() && cadence_hit) {
    for (auto& s : species_) {
      ::launch_pic_particle_compact(s, nullptr);
    }
  }
  // The deposit no longer synchronizes per step; instead it accumulates a
  // persistent overflow flag that we drain on the same cadence (and at end-of-run
  // via finalize()). This throws "reduce dt" at most kCompactEvery steps late,
  // which is acceptable for a fatal stability error.
  if (cadence_hit) {
    check_deposit_overflow();
  }
}

void EmPic2D3V::check_deposit_overflow() {
  for (auto& s : species_) {
    ::launch_pic_deposit_overflow_check(s, nullptr);
  }
}

void EmPic2D3V::finalize() {
  check_deposit_overflow();
}

void EmPic2D3V::advance(Real t_end, Real dt) {
  if (dt <= Real{0}) {
    throw std::invalid_argument{"EmPic2D3V::advance: dt must be positive"};
  }
  for (Real t = 0; t < t_end; t += dt) {
    step(dt);
  }
  finalize();
}

}  // namespace quasar::pic
