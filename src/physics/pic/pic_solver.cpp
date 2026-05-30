#include "quasar/physics/pic/pic_solver.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/core/registry.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string_view>

namespace quasar::numerics {

template <>
void YeeFdtd2D<2>::advance_b(YeeField2D<Real>& f, Real dt) const {
  ::launch_pic_fdtd_b_order2(f.grid, f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                             f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(), dt, nullptr);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

template <>
void YeeFdtd2D<4>::advance_b(YeeField2D<Real>& f, Real dt) const {
  ::launch_pic_fdtd_b_order4(f.grid, f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                             f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(), dt, nullptr);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

template <>
void YeeFdtd2D<2>::advance_e(YeeField2D<Real>& f, const JField2D<Real>& j, Real dt) const {
  ::launch_pic_fdtd_e_order2(f.grid, f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(),
                             f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                             j.jx.device_ptr(), j.jy.device_ptr(), j.jz.device_ptr(), dt, nullptr);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

template <>
void YeeFdtd2D<4>::advance_e(YeeField2D<Real>& f, const JField2D<Real>& j, Real dt) const {
  ::launch_pic_fdtd_e_order4(f.grid, f.ex.device_ptr(), f.ey.device_ptr(), f.ez.device_ptr(),
                             f.bx.device_ptr(), f.by.device_ptr(), f.bz.device_ptr(),
                             j.jx.device_ptr(), j.jy.device_ptr(), j.jz.device_ptr(), dt, nullptr);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

template <>
void BorisPusher<1>::push(pic::ParticleSpecies& s, const YeeField2D<Real>& f,
                          const YeeField2D<Real>& ext, Real dt) const {
  ::launch_pic_gather_push_shape1(f.grid, s, f, ext, dt, nullptr);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

template <>
void BorisPusher<2>::push(pic::ParticleSpecies& s, const YeeField2D<Real>& f,
                          const YeeField2D<Real>& ext, Real dt) const {
  ::launch_pic_gather_push_shape2(f.grid, s, f, ext, dt, nullptr);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

template <>
void Esirkepov2D<1>::deposit(const pic::ParticleSpecies& s, JField2D<Real>& j, Real dt) const {
  ::launch_pic_deposit_shape1(j.grid, s, j, dt, nullptr);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

template <>
void Esirkepov2D<2>::deposit(const pic::ParticleSpecies& s, JField2D<Real>& j, Real dt) const {
  ::launch_pic_deposit_shape2(j.grid, s, j, dt, nullptr);
  QUASAR_HIP_CHECK(::hipGetLastError());
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
  // The per-side boundary kind comes from the registry-built BC objects (see the
  // ctor); we read the configured kind back and invoke the corresponding kernel
  // with the solver's own grid_.
  //
  // TODO(particle-bc-heisenbug): The intended dispatch is
  //   particle_bcs_[side]->apply(s, static_cast<Side>(side));
  // i.e. straight through the IParticleBoundary interface. That path triggers an
  // intermittent (~30%) HIP illegal-memory-access (it bottoms out in the wrap
  // kernel reading what the BC class passes as species.grid()), whereas calling
  // the same kernels here with grid_ is stable (verified 10/10 vs ~4/10). The
  // grid value itself is NOT corrupt (an equality guard on species.grid() vs
  // grid_ never fired), so the fault is in the interface-object call path, not
  // the data. Until it is root-caused, dispatch on the configured kind with
  // grid_. See plans/field_bc_heisenbug.md.
  bool any_periodic = false;
  for (int side = 0; side < 4; ++side) {
    switch (cfg_.boundary.particle[side]) {
      case boundary::ParticleBoundaryKind::periodic:
        any_periodic = true;  // wrap kernel ignores side; apply once below
        break;
      case boundary::ParticleBoundaryKind::specular:
        ::launch_pic_boundary_specular_particles(grid_, s, side, nullptr);
        QUASAR_HIP_CHECK(::hipGetLastError());
        break;
      case boundary::ParticleBoundaryKind::absorbing:
        ::launch_pic_boundary_absorb_particles(grid_, s, side, nullptr);
        QUASAR_HIP_CHECK(::hipGetLastError());
        break;
    }
  }
  if (any_periodic) {
    ::launch_pic_boundary_periodic_particles(grid_, s, nullptr);
    QUASAR_HIP_CHECK(::hipGetLastError());
  }
}

void EmPic2D3V::step(Real dt) {
  QUASAR_HIP_CHECK(::hipMemset(current_.jx.device_ptr(), 0, current_.jx.bytes()));
  QUASAR_HIP_CHECK(::hipMemset(current_.jy.device_ptr(), 0, current_.jy.bytes()));
  QUASAR_HIP_CHECK(::hipMemset(current_.jz.device_ptr(), 0, current_.jz.bytes()));

  // TODO(field-bc-heisenbug): The boundary-aware stencil + field-ghost fill is
  // wired (fill_field_ghosts + the periodic/PEC field kernels) but currently
  // DISABLED behind QUASAR_PIC_FIELD_GHOSTS because enabling it triggers an
  // intermittent (~30%) HIP "illegal memory access" surfaced in
  // periodic_fields_kernel: the kernel intermittently receives a garbage
  // Grid2D-by-value even though the host fields_.grid stays valid right up to
  // the launch (verified by instrumentation). Ruled out: the stencil math
  // (fault persists with the old periodic_index stencil), ODR/stale build
  // (persists after a from-scratch rebuild), and a corrupt fields_.grid host
  // member (a guard at every sub-stage never fired). Likely a kernarg
  // marshaling / async-ordering issue specific to that kernel's argument list.
  // Until it is root-caused, fields use the implicit periodic wrap baked into
  // Grid2D::periodic_index (the original behavior), so periodic runs are
  // correct and PEC field walls are inert. See plans/ for the deferred fix.
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
