#include "quasar/physics/pic/pic_solver.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/core/registry.hpp"

#include "backend/hip/pic/launch.hpp"

#include <stdexcept>
#include <string_view>

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
  ::launch_pic_gather_push_shape1(f.grid, s, f, ext, dt, nullptr);
}

template <>
void BorisPusher<2>::push(pic::ParticleSpecies& s, const YeeField2D<Real>& f,
                          const YeeField2D<Real>& ext, Real dt) const {
  ::launch_pic_gather_push_shape2(f.grid, s, f, ext, dt, nullptr);
}

template <>
void Esirkepov2D<1>::deposit(const pic::ParticleSpecies& s, JField2D<Real>& j, Real dt) const {
  ::launch_pic_deposit_shape1(j.grid, s, j, dt, nullptr);
}

template <>
void Esirkepov2D<2>::deposit(const pic::ParticleSpecies& s, JField2D<Real>& j, Real dt) const {
  ::launch_pic_deposit_shape2(j.grid, s, j, dt, nullptr);
}

}  // namespace quasar::numerics

namespace quasar::pic {

namespace {

std::string_view particle_bc_name(boundary::ParticleBoundaryKind k) {
  switch (k) {
    case boundary::ParticleBoundaryKind::periodic:  return "periodic";
    case boundary::ParticleBoundaryKind::specular:  return "specular";
    case boundary::ParticleBoundaryKind::absorbing: return "absorbing";
  }
  throw std::invalid_argument{"EmPic2D3V: unknown ParticleBoundaryKind"};
}

std::string_view field_bc_name(boundary::FieldBoundaryKind k) {
  switch (k) {
    case boundary::FieldBoundaryKind::periodic: return "periodic";
    case boundary::FieldBoundaryKind::pec:      return "pec";
  }
  throw std::invalid_argument{"EmPic2D3V: unknown FieldBoundaryKind"};
}

}  // namespace

EmPic2D3V::EmPic2D3V(EmPicConfig cfg)
  : cfg_{cfg},
    grid_{cfg.grid},
    fields_{grid_},
    external_fields_{grid_},
    current_{grid_} {
  // NOTE: once the boundary-aware stencil (QUASAR_PIC_FIELD_GHOSTS) is enabled,
  // fdtd_order 4 will require grid nghost >= 2 (it reads two cells past the
  // boundary). While field ghosts are wrap-based that constraint does not apply.
  // Build per-side boundary conditions through the registry (the documented
  // pluggable construction path); concrete BCs self-register in src/boundary.
  for (int side = 0; side < 4; ++side) {
    particle_bcs_[side] = Registry<boundary::IParticleBoundary>::instance().create(
        particle_bc_name(cfg_.boundary.particle[side]));
    field_bcs_[side] = Registry<boundary::IFieldBoundary>::instance().create(
        field_bc_name(cfg_.boundary.field[side]));
  }
  if (cfg_.fdtd_order == 4) {
    field_solver_ = std::make_unique<numerics::YeeFdtd2D<4>>();
  } else if (cfg_.fdtd_order == 2) {
    field_solver_ = std::make_unique<numerics::YeeFdtd2D<2>>();
  } else {
    throw std::invalid_argument{"EmPic2D3V: fdtd_order must be 2 or 4"};
  }
  if (cfg_.shape_order == 2) {
    pusher_ = std::make_unique<numerics::BorisPusher<2>>();
    deposit_ = std::make_unique<numerics::Esirkepov2D<2>>();
  } else if (cfg_.shape_order == 1) {
    pusher_ = std::make_unique<numerics::BorisPusher<1>>();
    deposit_ = std::make_unique<numerics::Esirkepov2D<1>>();
  } else {
    throw std::invalid_argument{"EmPic2D3V: shape_order must be 1 or 2"};
  }
}

void EmPic2D3V::add_species(ParticleSpecies s) {
  s.set_grid(grid_);
  species_.push_back(std::move(s));
}

bool EmPic2D3V::has_absorbing_boundary() const noexcept {
  for (int side = 0; side < 4; ++side) {
    if (cfg_.boundary.particle[side] == boundary::ParticleBoundaryKind::absorbing) {
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
  backend::device_memset(current_.jx.device_ptr(), 0, current_.jx.bytes());
  backend::device_memset(current_.jy.device_ptr(), 0, current_.jy.bytes());
  backend::device_memset(current_.jz.device_ptr(), 0, current_.jz.bytes());

  // The field-ghost fill (fill_field_ghosts + the periodic/PEC field kernels) is
  // wired but gated behind QUASAR_PIC_FIELD_GHOSTS. The crash that originally
  // blocked it was the registry factory collision (now fixed in
  // core/registry.hpp), and the gated path no longer faults. It stays OFF by
  // default only because the FDTD stencil still reads via periodic_index (the
  // implicit periodic wrap), so PEC field walls are not yet physically correct
  // — completing them requires switching the stencil to ghost-aware index reads
  // (nghost>=2 for order 4) and a PEC reflection test. See
  // plans/field_bc_heisenbug.md for the remaining checklist.
#ifdef QUASAR_PIC_FIELD_GHOSTS
  fill_field_ghosts();
#endif
  field_solver_->advance_b(fields_, dt);
  for (auto& s : species_) {
    pusher_->push(s, fields_, external_fields_, dt);
    apply_particle_bcs(s);
    deposit_->deposit(s, current_, dt);
  }
  filters_.apply(current_, cfg_.boundary);
#ifdef QUASAR_PIC_FIELD_GHOSTS
  fill_field_ghosts();
#endif
  field_solver_->advance_e(fields_, current_, dt);

  // Reclaim slots vacated by absorbing boundaries on a fixed cadence so the
  // push/deposit/gather kernels stop iterating over dead particles. Only
  // meaningful when a boundary can actually kill particles; periodic/specular
  // runs never shrink, so skip the work entirely.
  ++step_count_;
  constexpr std::size_t kCompactEvery = 64;
  if (has_absorbing_boundary() && step_count_ % kCompactEvery == 0) {
    for (auto& s : species_) {
      ::launch_pic_particle_compact(s, nullptr);
    }
  }
}

void EmPic2D3V::advance(Real t_end, Real dt) {
  if (dt <= Real{0}) {
    throw std::invalid_argument{"EmPic2D3V::advance: dt must be positive"};
  }
  for (Real t = 0; t < t_end; t += dt) {
    step(dt);
  }
}

}  // namespace quasar::pic
