#pragma once

#include "quasar/distributed/checkpoint.hpp"
#include "quasar/distributed/device_mapping.hpp"
#include "quasar/distributed/mpi_runtime.hpp"
#include "quasar/distributed/runtime_lifecycle.hpp"
#include "quasar/distributed/topology.hpp"
#include "quasar/distributed/transport.hpp"
#include "quasar/distributed/worker_pool.hpp"
#include "quasar/physics/mhd/mhd_solver.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace quasar::distributed {

// Canonical, global, unpadded MHD state.  Cell quantities are row-major
// [global_ny, global_nx].  bx_face is [global_ny, global_nx+1] and by_face is
// [global_ny+1, global_nx].  Periodic high-face duplicates are retained in this
// interchange form and are checked/rebuilt by the runtime.
struct MhdGlobalState {
  std::size_t global_nx{0};
  std::size_t global_ny{0};
  std::vector<Real> rho{};
  std::vector<Real> mx{};
  std::vector<Real> my{};
  std::vector<Real> mz{};
  std::vector<Real> energy{};
  std::vector<Real> bx_face{};
  std::vector<Real> by_face{};
  std::vector<Real> bz_cell{};
};

// Rank-local, unpadded diagnostic tile.  Magnetic components are collocated
// through the active reconstruction operator, matching the serial NPZ cell
// fields rather than exposing staggered storage or numerical guards.
struct MhdOwnedShard {
  std::size_t endpoint{0};
  std::size_t tile_x{0};
  std::size_t tile_y{0};
  std::size_t offset_x{0};
  std::size_t offset_y{0};
  std::size_t owned_nx{0};
  std::size_t owned_ny{0};
  std::vector<Real> rho{};
  std::vector<Real> mx{};
  std::vector<Real> my{};
  std::vector<Real> mz{};
  std::vector<Real> energy{};
  std::vector<Real> bx{};
  std::vector<Real> by{};
  std::vector<Real> bz{};
};

struct MhdGlobalCellSums {
  // Cartesian values are the legacy raw cell sums used by existing invariant
  // diagnostics. Cylindrical values are physical axisymmetric volume
  // integrals, including the 2*pi*r*dr*dz measure.
  Real rho{0};
  Real energy{0};
};

struct MhdRuntimeTelemetry {
  std::uint64_t accepted_steps{0};
  std::uint64_t accepted_substeps{0};
  std::uint64_t rejected_attempts{0};
  std::uint64_t stage_evaluations{0};
  std::uint64_t state_reconciliations{0};
  std::uint64_t register_halo_epochs{0};
  std::uint64_t canonical_face_record_passes{0};
  std::uint64_t dense_residual_face_reconciliations{0};
  std::uint64_t dense_emf_input_reconciliations{0};
  std::uint64_t emf_reconciliations{0};
  std::uint64_t dense_ct_collective_bytes{0};
  std::uint64_t collective_bytes{0};
  std::uint64_t local_shard_extractions{0};
  TransportTelemetry transport{};
};

struct MhdGlobalBackground {
  std::size_t global_nx{0};
  std::size_t global_ny{0};
  std::vector<Real> b0x_face{};
  std::vector<Real> b0y_face{};
  std::vector<Real> b0z_cell{};
};

void validate_mhd_global_state(const MhdGlobalState& state);
void validate_mhd_global_background(const MhdGlobalBackground& background);

// A genuine tile-decomposed MHD runtime.  Each rank constructs only the tile
// solvers assigned to its local endpoints; one persistent worker owns each GPU.
// RK-register guards, canonical complete HLLD face records, derived CT tables,
// corner EMFs, and shared CT rates use packed nearest-neighbour device-buffer
// transport. Diagnostics/checkpoints retain topology-independent global file
// layouts by design.
class MhdTileRuntime {
 public:
  MhdTileRuntime(MpiRuntime& runtime,
                 EndpointMapping mapping,
                 VirtualTopology topology,
                 mhd::MhdConfig global_config,
                 TransportPolicy transport_policy = TransportPolicy::automatic);
  ~MhdTileRuntime() noexcept;

  MhdTileRuntime(const MhdTileRuntime&) = delete;
  MhdTileRuntime& operator=(const MhdTileRuntime&) = delete;
  MhdTileRuntime(MhdTileRuntime&&) = delete;
  MhdTileRuntime& operator=(MhdTileRuntime&&) = delete;

  [[nodiscard]] const VirtualTopology& topology() const noexcept;
  [[nodiscard]] const EndpointMapping& mapping() const noexcept;
  [[nodiscard]] bool closed() const noexcept;
  [[nodiscard]] bool seeded() const noexcept;
  [[nodiscard]] bool poisoned() const noexcept;
  [[nodiscard]] const MhdRuntimeTelemetry& telemetry() const noexcept;
  [[nodiscard]] const TransportResolution& transport_resolution() const noexcept;

  // All ranks supply the same canonical global seed.  This is intentionally a
  // host interchange boundary: Python deck construction and checkpoint restart
  // can both feed it without exposing padded tile storage.
  void seed(const MhdGlobalState& state,
            const MhdGlobalBackground* background = nullptr);

  [[nodiscard]] Real cfl_limit();
  void step(Real dt, bool check_cfl = true);
  [[nodiscard]] Real divergence_b_max();

  // Returns the canonical global state on every rank.  Callers that only write
  // on rank zero simply discard it elsewhere; checkpoint code can use it on all
  // ranks until direct hyperslab staging is selected.
  [[nodiscard]] MhdGlobalState gather_state();
  // Gather one cell-collocated live component on the canonical [ny,nx]
  // lattice.  In particular, "bx"/"by" use each solver's reconstruction-order
  // collocation after internal halos have been reconciled, matching the serial
  // diagnostics contract rather than a lower-order host average.
  [[nodiscard]] std::vector<Real> gather_cell_component(
      std::string_view component);
  [[nodiscard]] std::vector<MhdOwnedShard> local_owned_shards();
  [[nodiscard]] MhdGlobalCellSums global_cell_sums();

  // Construct the topology-independent compatibility record used by MHD
  // checkpoints.  Rank/GPU placement, decomposition, and output policy are
  // deliberately absent; the physical mesh and every numerical/physics
  // setting that must remain fixed across restart are encoded in the derived
  // signatures.
  [[nodiscard]] CheckpointMetadata checkpoint_metadata(
      std::uint64_t step, double time,
      std::string_view unit_system,
      const MhdGlobalBackground* expected_background = nullptr) const;

  // Write one committed-step image through collective parallel HDF5.  Each
  // rank contributes the hyperslabs owned by its local endpoints, and rank
  // zero atomically replaces `path` only after every dataset is complete.
  void write_checkpoint(const std::filesystem::path& path,
                        std::uint64_t step, double time,
                        std::string_view unit_system,
                        std::span<const std::uint8_t> diagnostic_state = {});

  // Restore into this runtime's (possibly different) rank/GPU/tile mapping.
  // The runtime must not already be seeded.  Compatibility and all file data
  // are validated collectively before solver state is mutated.  The returned
  // record carries the stored absolute step and time.
  [[nodiscard]] CheckpointMetadata restart_from_checkpoint(
      const std::filesystem::path& path,
      std::string_view unit_system,
      const MhdGlobalBackground* expected_background = nullptr,
      std::vector<std::vector<std::uint8_t>>* diagnostic_state = nullptr);

  // One-shot rank-local hooks used to verify allocation consensus, mutation
  // poisoning, and collective checkpoint-reader cleanup. The API is present
  // in every configuration; setters are inert unless test hooks were compiled.
  void inject_next_worker_task_allocation_failure_for_testing(
      bool enabled) noexcept;
  void inject_seed_post_mutation_failure_for_testing(bool enabled) noexcept;
  void inject_checkpoint_metadata_copy_failure_for_testing(
      bool enabled) noexcept;
  // Inject one rank-local failure after restart has published and reconciled
  // its restored tile state. Tests use this to prove post-mutation failures
  // poison every replica while leaving collective close operational.
  void inject_restart_post_reconcile_failure_for_testing(
      bool enabled) noexcept;

  // Explicit and collective. Once a fatal operation poisons the runtime, this
  // is the only legal state-changing or diagnostic operation. The destructor
  // deliberately performs no MPI or parallel-HDF5 operation.
  void close();

 private:
  struct DenseField;
  struct RegisterHaloBuffers;
  struct RestartPayload;
  struct StepController;

  [[nodiscard]] mhd::MhdConfig tile_config(std::size_t endpoint) const;
  [[nodiscard]] DenseField collect_register(int register_index,
                                            std::string_view phase);
  [[nodiscard]] DenseField collect_field(int register_index, bool residual,
                                         std::string_view phase);
  std::uint64_t transfer_halo_axis(Direction positive_direction,
                                   std::string_view phase);
  void exchange_register_axis(int register_index,
                              Direction positive_direction,
                              std::string_view phase);
  void exchange_background_axis(Direction positive_direction,
                                std::string_view phase);
  void reconcile_background(std::string_view phase);
  void reconcile_register(int register_index, std::string_view phase);
  void exchange_face_records_axis(Direction positive_direction,
                                  std::string_view phase);
  void reconcile_face_records(std::string_view phase);
  void exchange_residual_faces_axis(Direction positive_direction,
                                    std::string_view phase);
  void reconcile_residual_faces(std::string_view phase);
  void exchange_emf_inputs_axis(Direction positive_direction, bool masks,
                                std::string_view phase);
  void reconcile_emf_inputs(std::string_view phase);
  void exchange_corner_emf_axis(Direction positive_direction,
                                std::string_view phase);
  void reconcile_emf(std::string_view phase);
  void prepare_step_request(Real dt, bool check_cfl);
  [[nodiscard]] Real low_order_anchor();
  void snapshot_substep(bool low_order_interval, Real global_anchor);
  void prepare_step_stage(int stage);
  [[nodiscard]] Real apply_and_assess_step_stage(int stage, Real trial);
  [[nodiscard]] Real attempt_substep(Real trial);
  [[nodiscard]] Real substep_cfl_limit(int order,
                                       std::string_view phase);
  void reject_substep(StepController& controller, Real rejected_theta);
  [[nodiscard]] bool accept_and_advance_substep(
      StepController& controller);
  void advance_step(Real dt);
  void restore_step_after_failure() noexcept;
  void finish_step();
  void seed_local_tiles(const MhdGlobalState& state,
                        const MhdGlobalBackground* background,
                        bool& mutation_started);
  [[nodiscard]] CheckpointMetadata copy_restart_metadata(
      ParallelCheckpointReader& reader);
  void require_compatible_restart(
      const CheckpointMetadata& stored, std::string_view unit_system,
      const MhdGlobalBackground* expected_background);
  [[nodiscard]] RestartPayload stage_restart_payload();
  void read_restart_payload(ParallelCheckpointReader& reader,
                            RestartPayload& payload);
  void validate_restart_payload(const CheckpointMetadata& stored,
                                RestartPayload& payload);
  void commit_restart_payload(RestartPayload& payload,
                              bool& mutation_started);
  void publish_restart_payload(
      RestartPayload& payload,
      std::vector<std::vector<std::uint8_t>>* diagnostic_state);
  [[nodiscard]] MhdGlobalBackground collect_background(
      std::string_view phase);
  void require_open_seeded() const;
  void restore_request_backups();
  [[noreturn]] void poison_collectively(std::string_view phase,
                                        std::string_view local_message);

  MpiRuntime* runtime_{nullptr};
  EndpointMapping mapping_{};
  VirtualTopology topology_;
  mhd::MhdConfig global_config_{};
  std::vector<int> local_devices_{};
  std::unique_ptr<EndpointWorkerPool> workers_{};
  std::vector<std::unique_ptr<mhd::MhdSolver2D>> solvers_{};
  std::vector<std::unique_ptr<RegisterHaloBuffers>> register_halos_{};
  std::unique_ptr<Transport> transport_{};
  MhdRuntimeTelemetry telemetry_{};
  // Non-empty only when the active B0 was supplied as canonical explicit
  // arrays rather than sampled solely from the analytic config profile.
  std::string background_content_signature_{};
  std::uint64_t worker_epoch_{0};
  // Shared seeded/poisoned/closed lifecycle. A checkpoint write or
  // post-publication restart failure is fatal:
  // worker/device state may still be intact, but only collective close is
  // legal. The previously committed file remains owned by the writer's atomic
  // replace protocol.
  detail::RuntimeLifecycleState lifecycle_{};
  // Inert unless the test-hook compile definition enables the corresponding
  // setter and post-reconcile gate. Keeping storage unconditional preserves
  // the class layout for consumers of a test-enabled library.
  bool inject_restart_post_reconcile_failure_{false};
};

}  // namespace quasar::distributed
