#pragma once

// Private distributed/solver seam. MhdSolver2D befriends only this facade;
// orchestration code receives phase operations and narrowly-scoped storage
// views instead of becoming coupled to every solver data member.

#include "quasar/backend/device.hpp"
#include "quasar/physics/mhd/mhd_solver.hpp"

#include <stdexcept>

namespace quasar::distributed {

class MhdTileAccess final {
 public:
  struct FaceRecordView {
    mhd::MhdField2D<Real>& flux;
    mhd::MhdMomentumFluxParts2D<Real>& momentum;
  };

  struct ConstFaceRecordView {
    const mhd::MhdField2D<Real>& flux;
    const mhd::MhdMomentumFluxParts2D<Real>& momentum;
  };

  static mhd::MhdField2D<Real>& state_register(
      mhd::MhdSolver2D& solver, int index) {
    require_register(index);
    return solver.rk_[static_cast<std::size_t>(index)];
  }

  static const mhd::MhdField2D<Real>& state_register(
      const mhd::MhdSolver2D& solver, int index) {
    require_register(index);
    return solver.rk_[static_cast<std::size_t>(index)];
  }

  static mhd::MhdField2D<Real>& residual(mhd::MhdSolver2D& solver) noexcept {
    return solver.residual_;
  }

  static const mhd::MhdField2D<Real>& residual(
      const mhd::MhdSolver2D& solver) noexcept {
    return solver.residual_;
  }

  static mhd::MhdBackgroundField<Real>& background(
      mhd::MhdSolver2D& solver) noexcept {
    return solver.b0_;
  }

  static const mhd::MhdBackgroundField<Real>& background(
      const mhd::MhdSolver2D& solver) noexcept {
    return solver.b0_;
  }

  static FaceRecordView face_records(
      mhd::MhdSolver2D& solver, bool x_direction) noexcept {
    return x_direction
        ? FaceRecordView{solver.flux_x_, solver.momentum_flux_x_}
        : FaceRecordView{solver.flux_y_, solver.momentum_flux_y_};
  }

  static ConstFaceRecordView face_records(
      const mhd::MhdSolver2D& solver, bool x_direction) noexcept {
    return x_direction
        ? ConstFaceRecordView{solver.flux_x_, solver.momentum_flux_x_}
        : ConstFaceRecordView{solver.flux_y_, solver.momentum_flux_y_};
  }

  static mhd::EmfField2D<Real>& emf(mhd::MhdSolver2D& solver) noexcept {
    return solver.emf_;
  }

  static void note_state_reconciled(mhd::MhdSolver2D& solver) noexcept {
    solver.invalidate_interface_cache();
  }

  static void note_background_reconciled(mhd::MhdSolver2D& solver) noexcept {
    solver.background_validated_ = false;
    solver.invalidate_interface_cache();
  }

  static Real cfl_limit(mhd::MhdSolver2D& solver,
                        int collocation_order) {
    return solver.cfl_limit_for_collocation(collocation_order);
  }

  static void snapshot_request(mhd::MhdSolver2D& solver) {
    solver.copy_state(solver.rk_[0], solver.request_backup_);
    solver.last_positivity_substeps_ = 0;
  }

  static void restore_request(mhd::MhdSolver2D& solver) {
    reset_step_modes(solver);
    solver.copy_state(solver.request_backup_, solver.rk_[0]);
  }

  static Real low_order_anchor(mhd::MhdSolver2D& solver) {
    return solver.positivity_->admissible_fraction(
        solver.rk_[0], solver.rk_[0], Real{0}, Real{0}, solver.cfg_.gamma,
        /*collocation_order=*/1);
  }

  static void snapshot_substep(mhd::MhdSolver2D& solver,
                               bool low_order_interval,
                               bool low_order_anchor_available,
                               backend::stream_t copy_stream) {
    // copy_state updates owned cells/faces. Preserve the reconciled complete
    // face arrays as well because the positivity collocation reads guards.
    solver.fill_ghosts(solver.rk_[0]);
    solver.copy_state(solver.rk_[0], solver.step_backup_);
    copy_full_buffer(solver.rk_[0].bx_face, solver.step_backup_.bx_face,
                     copy_stream);
    copy_full_buffer(solver.rk_[0].by_face, solver.step_backup_.by_face,
                     copy_stream);
    backend::device_synchronize(copy_stream);
    solver.positivity_control_active_ = true;
    solver.positivity_reconstruction_order_ =
        low_order_interval ? 1 : 0;
    solver.positivity_low_order_anchor_available_ =
        low_order_anchor_available;
    solver.internal_integrator_access_ = true;
  }

  static void prepare_face_records(mhd::MhdSolver2D& solver, int stage,
                                   mhd::FaceOwnershipFlags4 ownership) {
    require_register(stage);
    solver.prepare_residual_face_records(
        solver.rk_[static_cast<std::size_t>(stage)], solver.residual_,
        ownership);
  }

  static void consume_face_records(mhd::MhdSolver2D& solver, int stage) {
    require_register(stage);
    solver.consume_residual_face_records(
        solver.rk_[static_cast<std::size_t>(stage)], solver.residual_);
  }

  static void finish_emf(mhd::MhdSolver2D& solver) {
    solver.finish_ct_emf();
  }

  static void compute_ct_rate(mhd::MhdSolver2D& solver) {
    solver.compute_ct_rate_from_emf(solver.residual_);
  }

  static void finish_split_energy(mhd::MhdSolver2D& solver) {
    solver.finish_split_energy(solver.residual_);
  }

  static void apply_stage(mhd::MhdSolver2D& solver, int stage, Real dt) {
    (void)solver.apply_stage_update(stage, dt);
  }

  static Real assess_stage(mhd::MhdSolver2D& solver, int stage) {
    require_register(stage);
    const int output = stage == 2 ? 0 : stage + 1;
    solver.fill_ghosts(solver.rk_[static_cast<std::size_t>(output)]);
    return solver.stage_admissible_fraction(stage);
  }

  static void rollback_substep(mhd::MhdSolver2D& solver) {
    solver.positivity_control_active_ = false;
    solver.positivity_reconstruction_order_ = 0;
    solver.copy_state(solver.step_backup_, solver.rk_[0]);
  }

  static void accept_substep(mhd::MhdSolver2D& solver,
                             bool low_order_interval) noexcept {
    solver.positivity_control_active_ = false;
    solver.positivity_reconstruction_order_ =
        low_order_interval ? 1 : 0;
    solver.internal_integrator_access_ = false;
    solver.live_state_solver_owned_ =
        !solver.external_mutable_state_exposed_;
    ++solver.last_positivity_substeps_;
  }

  static void reset_step_modes(mhd::MhdSolver2D& solver) noexcept {
    solver.positivity_control_active_ = false;
    solver.positivity_low_order_anchor_available_ = false;
    solver.positivity_reconstruction_order_ = 0;
    solver.internal_integrator_access_ = false;
  }

  static void mark_restart_state_trusted(mhd::MhdSolver2D& solver) noexcept {
    solver.live_state_solver_owned_ = true;
  }

 private:
  static void require_register(int index) {
    if (index < 0 || index >= mhd::MhdSolver2D::kNumRkRegisters) {
      throw std::out_of_range{"distributed MHD register is out of range"};
    }
  }

  static void copy_full_buffer(const backend::DeviceBuffer<Real>& source,
                               backend::DeviceBuffer<Real>& destination,
                               backend::stream_t stream) {
    if (source.size() != destination.size() ||
        source.owner_device() != destination.owner_device()) {
      throw std::logic_error{
          "distributed MHD snapshot face buffers are incompatible"};
    }
    backend::device_memcpy_peer_async(
        destination.device_ptr(), destination.owner_device(),
        source.device_ptr(), source.owner_device(), source.bytes(), stream);
  }
};

}  // namespace quasar::distributed
