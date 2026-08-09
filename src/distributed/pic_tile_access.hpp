#pragma once

// The sole private seam between the distributed scheduler and the serial PIC
// solver. It exposes phase operations and narrowly grouped state views without
// making solver layout part of the public distributed API.

#include "quasar/physics/pic/kernels.hpp"
#include "quasar/physics/pic/pic_solver.hpp"

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

namespace quasar::distributed {

class PicTileAccess {
 public:
  using Solver = pic::EmPic2D3V;
  using Species = pic::ParticleSpecies;

  static const Grid2D& grid(const Solver& solver) noexcept {
    return solver.grid_;
  }
  static YeeField2D<Real>& fields(Solver& solver) noexcept {
    return solver.fields_;
  }
  static const YeeField2D<Real>& fields(const Solver& solver) noexcept {
    return solver.fields_;
  }
  static YeeField2D<Real>& external_fields(Solver& solver) noexcept {
    return solver.external_fields_;
  }
  static const YeeField2D<Real>& external_fields(
      const Solver& solver) noexcept {
    return solver.external_fields_;
  }
  static BField2D<Real>& previous_b(Solver& solver) noexcept {
    return solver.previous_b_;
  }
  static const BField2D<Real>& previous_b(const Solver& solver) noexcept {
    return solver.previous_b_;
  }
  static JField2D<Real>& current(Solver& solver) noexcept {
    return solver.current_;
  }
  static const JField2D<Real>& current(const Solver& solver) noexcept {
    return solver.current_;
  }
  static ScalarGrid2D<Real>& charge(Solver& solver, bool next) noexcept {
    return next ? solver.next_charge_ : solver.charge_;
  }
  static const ScalarGrid2D<Real>& charge(
      const Solver& solver, bool next) noexcept {
    return next ? solver.next_charge_ : solver.charge_;
  }
  static backend::DeviceBuffer<unsigned int>& source_finite_error(
      Solver& solver) noexcept {
    return solver.source_finite_error_;
  }
  static std::vector<Species>& species(Solver& solver) noexcept {
    return solver.species_;
  }
  static const std::vector<Species>& species(
      const Solver& solver) noexcept {
    return solver.species_;
  }
  static const pic::EmPicConfig& config(const Solver& solver) noexcept {
    return solver.cfg_;
  }

  static void swap_sources(
      Solver& solver, bool next_charge,
      std::array<backend::DeviceBuffer<Real>, 4>& work) noexcept {
    std::swap(solver.current_.jx, work[0]);
    std::swap(solver.current_.jy, work[1]);
    std::swap(solver.current_.jz, work[2]);
    std::swap(next_charge ? solver.next_charge_.values : solver.charge_.values,
              work[3]);
  }
  static void swap_filtered_current(
      Solver& solver, backend::DeviceBuffer<Real>& work) noexcept {
    std::swap(solver.current_.jz, work);
  }
  static void swap_inplane_current(
      Solver& solver,
      std::array<backend::DeviceBuffer<Real>, 2>& work) noexcept {
    std::swap(solver.current_.jx, work[0]);
    std::swap(solver.current_.jy, work[1]);
  }

  static void replace_species_host(
      Solver& solver, std::size_t kind,
      const Species::HostSnapshot& particles) {
    solver.species_.at(kind).replace_host_particles(particles);
  }
  static void append_migrated_species(
      Solver& solver, std::size_t kind,
      const Species::HostSnapshot& particles, backend::stream_t stream) {
    solver.species_.at(kind).append_migrated_particles(particles, stream);
  }

  static void set_background(Solver& solver, bool initialized,
                             Real density) noexcept {
    solver.background_initialized_ = initialized;
    solver.background_charge_density_ = density;
  }
  static Real background_density(const Solver& solver) noexcept {
    return solver.background_charge_density_;
  }
  static bool background_initialized(const Solver& solver) noexcept {
    return solver.background_initialized_;
  }
  static void mark_charge_valid(Solver& solver, bool valid = true) noexcept {
    solver.charge_valid_ = valid;
  }
  static void restore_runtime_state(
      Solver& solver, std::size_t step_count, Real previous_dt,
      bool has_previous_dt, bool background_initialized,
      Real background_density) noexcept {
    solver.step_count_ = step_count;
    solver.previous_dt_ = previous_dt;
    solver.has_previous_dt_ = has_previous_dt;
    solver.evolution_started_ = step_count != 0 || has_previous_dt;
    solver.background_initialized_ = background_initialized;
    solver.background_charge_density_ = background_density;
    solver.charge_valid_ = true;
  }

  static void fill_field_ghosts(Solver& solver) {
    solver.fill_field_ghosts();
  }

  static void deposit_initial_charge(
      Solver& solver, backend::stream_t compute_stream,
      backend::DeviceEvent& compute_ready) {
    backend::device_memset_async(solver.current_.jx.device_ptr(), 0,
                                 solver.current_.jx.bytes(), compute_stream);
    backend::device_memset_async(solver.current_.jy.device_ptr(), 0,
                                 solver.current_.jy.bytes(), compute_stream);
    backend::device_memset_async(solver.current_.jz.device_ptr(), 0,
                                 solver.current_.jz.bytes(), compute_stream);
    backend::device_memset_async(solver.charge_.values.device_ptr(), 0,
                                 solver.charge_.values.bytes(), compute_stream);
    compute_ready.record(compute_stream);
    backend::stream_wait_event(nullptr, compute_ready.get());
    for (const auto& species : solver.species_) {
      solver.deposit_->deposit_charge(species, solver.charge_);
    }
    for (int side = 0; side < 4; ++side) {
      solver.particle_bcs_[side]->fold_charge(
          solver.charge_, static_cast<Side>(side));
    }
    solver.check_deposit_overflow();
  }

  static void faraday_phase(
      Solver& solver, Real magnetic_dt, backend::stream_t compute_stream,
      backend::DeviceEvent& compute_ready,
      backend::DeviceEvent& communication_ready) {
    solver.evolution_started_ = true;
    backend::device_memset_async(solver.current_.jx.device_ptr(), 0,
                                 solver.current_.jx.bytes(), compute_stream);
    backend::device_memset_async(solver.current_.jy.device_ptr(), 0,
                                 solver.current_.jy.bytes(), compute_stream);
    backend::device_memset_async(solver.current_.jz.device_ptr(), 0,
                                 solver.current_.jz.bytes(), compute_stream);
    compute_ready.record(compute_stream);
    backend::stream_wait_event(nullptr, compute_ready.get());
    solver.fill_field_ghosts();
    communication_ready.record(nullptr);
    backend::stream_wait_event(compute_stream, communication_ready.get());
    ::launch_pic_copy_b(solver.grid_, solver.fields_, solver.previous_b_,
                        compute_stream);
    compute_ready.record(compute_stream);
    backend::stream_wait_event(nullptr, compute_ready.get());
    solver.field_solver_->advance_b(solver.fields_, magnetic_dt);
    solver.correct_field_boundaries_b(magnetic_dt);
    backend::device_synchronize(nullptr);
  }

  static void push_deposit_phase(
      Solver& solver, Real force_dt, Real position_dt,
      Real previous_b_weight, Real current_b_weight,
      backend::stream_t compute_stream,
      backend::DeviceEvent& compute_ready) {
    solver.prime_outflow_corners();
    backend::device_memset_async(solver.next_charge_.values.device_ptr(), 0,
                                 solver.next_charge_.values.bytes(),
                                 compute_stream);
    compute_ready.record(compute_stream);
    backend::stream_wait_event(nullptr, compute_ready.get());
    for (auto& species : solver.species_) {
      solver.pusher_->push(species, solver.fields_, solver.external_fields_,
                           solver.previous_b_, force_dt, position_dt,
                           previous_b_weight, current_b_weight);
      solver.apply_particle_bcs_before_deposit(species);
      solver.prepare_absorbing_bcs_for_deposit(species);
      solver.deposit_->deposit(species, solver.current_, position_dt);
      solver.apply_absorbing_bcs_after_deposit(species);
    }
    for (const auto& species : solver.species_) {
      solver.deposit_->deposit_charge(species, solver.next_charge_);
    }
    for (int side = 0; side < 4; ++side) {
      solver.particle_bcs_[side]->fold_charge(
          solver.next_charge_, static_cast<Side>(side));
    }
    solver.check_deposit_overflow();
    for (int side = 0; side < 4; ++side) {
      solver.particle_bcs_[side]->fold_current(
          solver.current_, static_cast<Side>(side));
    }
    backend::device_synchronize(nullptr);
  }

  static void ampere_phase(
      Solver& solver, Real dt, backend::stream_t compute_stream,
      backend::DeviceEvent& compute_ready,
      backend::DeviceEvent& communication_ready) {
    ::launch_pic_validate_finite_sources(
        solver.grid_, &solver.current_, &solver.next_charge_,
        solver.source_finite_error_.device_ptr(), compute_stream);
    solver.fill_field_ghosts();
    solver.field_solver_->advance_e(solver.fields_, solver.current_, dt);
    solver.correct_field_boundaries_e(dt);
    solver.correct_outflow_corners(dt);
    std::swap(solver.charge_.values, solver.next_charge_.values);
    solver.charge_valid_ = true;
    solver.previous_dt_ = dt;
    solver.has_previous_dt_ = true;
    ++solver.step_count_;
    constexpr std::size_t compact_every = 64;
    if (solver.has_absorbing_boundary() &&
        solver.step_count_ % compact_every == 0) {
      communication_ready.record(nullptr);
      backend::stream_wait_event(compute_stream, communication_ready.get());
      for (auto& species : solver.species_) {
        ::launch_pic_particle_compact(species, compute_stream);
      }
      compute_ready.record(compute_stream);
      compute_ready.synchronize();
    }
    backend::device_synchronize(nullptr);
  }

  static bool side_uses_outflow(const Solver& solver, int side) {
    return solver.cfg_.boundary.field.at(static_cast<std::size_t>(side)) ==
        "outflow";
  }
  static bool side_history_primed(const Solver& solver, int side) {
    return solver.field_bcs_.at(static_cast<std::size_t>(side))
        ->checkpoint_history_primed();
  }
  static std::vector<Real> side_history(const Solver& solver, int side) {
    return solver.field_bcs_.at(static_cast<std::size_t>(side))
        ->checkpoint_history();
  }
  static void restore_side_history(Solver& solver, int side,
                                   std::vector<Real> history,
                                   bool primed) {
    solver.field_bcs_.at(static_cast<std::size_t>(side))
        ->restore_checkpoint_history(std::move(history), primed);
  }
  static unsigned int corner_mask(const Solver& solver) noexcept {
    return solver.outflow_corner_mask_;
  }
  static bool corners_primed(const Solver& solver) noexcept {
    return solver.outflow_corners_primed_;
  }
  static backend::DeviceBuffer<Real>& corner_history(
      Solver& solver) noexcept {
    return solver.outflow_corner_history_;
  }
  static const backend::DeviceBuffer<Real>& corner_history(
      const Solver& solver) noexcept {
    return solver.outflow_corner_history_;
  }
  static void set_corners_primed(Solver& solver, bool primed) noexcept {
    solver.outflow_corners_primed_ = primed;
  }
};

}  // namespace quasar::distributed
