#include "quasar/physics/pic/pic_solver.hpp"

#include "quasar/backend/device.hpp"

#include <hip/hip_runtime.h>

#include <stdexcept>

extern "C" void launch_pic_fdtd_b_order2(const quasar::Grid2D&, double*, double*, double*,
                                         const double*, const double*, const double*, double,
                                         hipStream_t);
extern "C" void launch_pic_fdtd_b_order4(const quasar::Grid2D&, double*, double*, double*,
                                         const double*, const double*, const double*, double,
                                         hipStream_t);
extern "C" void launch_pic_fdtd_e_order2(const quasar::Grid2D&, double*, double*, double*,
                                         const double*, const double*, const double*,
                                         const double*, const double*, const double*, double,
                                         hipStream_t);
extern "C" void launch_pic_fdtd_e_order4(const quasar::Grid2D&, double*, double*, double*,
                                         const double*, const double*, const double*,
                                         const double*, const double*, const double*, double,
                                         hipStream_t);
extern "C" void launch_pic_gather_push_shape1(const quasar::Grid2D&, quasar::pic::ParticleSpecies&,
                                              const quasar::YeeField2D<double>&,
                                              const quasar::YeeField2D<double>&, double,
                                              hipStream_t);
extern "C" void launch_pic_gather_push_shape2(const quasar::Grid2D&, quasar::pic::ParticleSpecies&,
                                              const quasar::YeeField2D<double>&,
                                              const quasar::YeeField2D<double>&, double,
                                              hipStream_t);
extern "C" void launch_pic_deposit_shape1(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                                          quasar::JField2D<double>&, double, hipStream_t);
extern "C" void launch_pic_deposit_shape2(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                                          quasar::JField2D<double>&, double, hipStream_t);

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

EmPic2D3V::EmPic2D3V(EmPicConfig cfg)
  : cfg_{cfg},
    grid_{cfg.grid},
    fields_{grid_},
    external_fields_{grid_},
    current_{grid_} {
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

void EmPic2D3V::step(Real dt) {
  QUASAR_HIP_CHECK(::hipMemset(current_.jx.device_ptr(), 0, current_.jx.bytes()));
  QUASAR_HIP_CHECK(::hipMemset(current_.jy.device_ptr(), 0, current_.jy.bytes()));
  QUASAR_HIP_CHECK(::hipMemset(current_.jz.device_ptr(), 0, current_.jz.bytes()));

  field_solver_->advance_b(fields_, dt);
  for (auto& s : species_) {
    pusher_->push(s, fields_, external_fields_, dt);
    deposit_->deposit(s, current_, dt);
  }
  filters_.apply(current_, cfg_.boundary);
  field_solver_->advance_e(fields_, current_, dt);
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
