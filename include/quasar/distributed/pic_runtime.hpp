#pragma once

#include "quasar/distributed/checkpoint.hpp"
#include "quasar/distributed/device_mapping.hpp"
#include "quasar/distributed/mpi_runtime.hpp"
#include "quasar/distributed/runtime_lifecycle.hpp"
#include "quasar/distributed/topology.hpp"
#include "quasar/distributed/transport.hpp"
#include "quasar/distributed/worker_pool.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/physics/pic/kernels.hpp"
#include "quasar/physics/pic/pic_solver.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace quasar::distributed {

struct PicFixedExchangePlan;
struct PicHaloBuffers;
struct PicCheckpointTile;
struct PicCheckpointSnapshot;
struct PicRestartPayload;

// Canonical global, unpadded PIC field state.  Each component retains its true
// Yee lattice: a face/node dimension has one more sample than the cell mesh.
// Periodic high-face/node duplicates are retained at this interchange boundary
// and are verified/rebuilt by PicTileRuntime.
struct PicGlobalFields {
  std::size_t global_nx{0};
  std::size_t global_ny{0};
  std::vector<Real> ex{}, ey{}, ez{}, bx{}, by{}, bz{};
};

struct PicGlobalSources {
  std::size_t global_nx{0};
  std::size_t global_ny{0};
  std::vector<Real> jx{}, jy{}, jz{}, charge{};
};

struct PicSpeciesState {
  pic::SpeciesConfig config{};
  pic::ParticleSpecies::HostSnapshot particles{};
};

// Rank-local diagnostics payload.  Each array contains one canonical owned
// rectangle of its Yee lattice in row-major order.  Offsets are expressed in
// the corresponding global component lattice, not in cell-mesh coordinates.
struct PicOwnedArray {
  std::size_t offset_x{0};
  std::size_t offset_y{0};
  std::size_t nx{0};
  std::size_t ny{0};
  std::vector<Real> values{};
};

struct PicOwnedFields {
  PicOwnedArray ex{}, ey{}, ez{}, bx{}, by{}, bz{};
};

struct PicOwnedShard {
  std::size_t endpoint{0};
  std::size_t tile_x{0};
  std::size_t tile_y{0};
  std::size_t offset_x{0};
  std::size_t offset_y{0};
  std::size_t owned_nx{0};
  std::size_t owned_ny{0};
  PicOwnedFields fields{};
  PicOwnedFields external_fields{};
  std::vector<PicSpeciesState> species{};
};

// Canonical topology-independent characteristic-boundary state. Each active
// Mur side is [4, transverse_cells+1] in strip-major order. The eight corner
// entries are [old_wall(4), old_adjacent(4)] indexed by global corner.
struct PicBoundaryState {
  std::array<std::vector<Real>, 4> mur_history{};
  std::array<std::uint8_t, 4> mur_primed{};
  std::vector<Real> outflow_corner_history{};
  bool outflow_corners_primed{false};
};

// A committed-step restart image.  Scratch, guards, filter work arrays, and
// temporary migration buffers are deliberately absent and are rebuilt after
// repartitioning.
struct PicGlobalState {
  PicGlobalFields fields{};
  PicGlobalFields external_fields{};
  PicGlobalSources sources{};
  std::vector<Real> previous_bx{}, previous_by{}, previous_bz{};
  std::vector<PicSpeciesState> species{};
  std::uint64_t step_count{0};
  Real previous_dt{Real{0}};
  bool has_previous_dt{false};
  bool background_initialized{false};
  Real background_charge_density{Real{0}};
  PicBoundaryState boundary{};
};

struct PicRuntimeTelemetry {
  std::uint64_t accepted_steps{0};
  std::uint64_t state_reconciliations{0};
  std::uint64_t source_reconciliations{0};
  std::uint64_t particle_migrations{0};
  std::uint64_t migrated_particles{0};
  std::uint64_t collective_bytes{0};
  std::uint64_t global_state_gathers{0};
  std::uint64_t local_shard_extractions{0};
  std::uint64_t transport_epochs{0};
  std::uint64_t transport_messages{0};
  std::uint64_t transport_bytes{0};
  std::uint64_t transport_peer_bytes{0};
  std::uint64_t transport_local_staged_bytes{0};
  std::uint64_t transport_staged_mpi_bytes{0};
  std::uint64_t transport_direct_mpi_bytes{0};
  std::uint64_t checkpoint_local_lattice_writes{0};
  std::uint64_t checkpoint_local_lattice_reads{0};
  std::uint64_t checkpoint_global_lattice_materializations{0};
};

void validate_pic_global_fields(const PicGlobalFields& fields,
                                std::string_view geometry);
void validate_pic_species_state(const PicSpeciesState& species);

// Genuine tile-decomposed PIC evolution.  Every local endpoint owns one solver
// and persistent GPU worker.  The orchestration thread alone performs MPI;
// workers execute solver kernels and stage host data.  The current transport is
// correctness-first dense reconciliation, but ownership, additive source
// reduction, and half-open particle migration are decomposition independent.
class PicTileRuntime {
 public:
  PicTileRuntime(MpiRuntime& runtime,
                 EndpointMapping mapping,
                 VirtualTopology topology,
                 pic::EmPicConfig global_config,
                 TransportPolicy transport_policy = TransportPolicy::staged);
  ~PicTileRuntime() noexcept;

  PicTileRuntime(const PicTileRuntime&) = delete;
  PicTileRuntime& operator=(const PicTileRuntime&) = delete;
  PicTileRuntime(PicTileRuntime&&) = delete;
  PicTileRuntime& operator=(PicTileRuntime&&) = delete;

  [[nodiscard]] const VirtualTopology& topology() const noexcept;
  [[nodiscard]] const EndpointMapping& mapping() const noexcept;
  [[nodiscard]] bool closed() const noexcept;
  [[nodiscard]] bool seeded() const noexcept;
  [[nodiscard]] bool poisoned() const noexcept;
  [[nodiscard]] const PicRuntimeTelemetry& telemetry() const noexcept;
  [[nodiscard]] const TransportResolution& transport_resolution() const noexcept;

  // All ranks provide the same canonical seed.  Particles are routed by their
  // position to a half-open tile owner and ordered by stable ID.
  void seed(const PicGlobalFields& fields,
            const PicGlobalFields* external_fields,
            std::span<const PicSpeciesState> species);
  void restore(const PicGlobalState& state);

  // Sample the prescribed field independently on every tile, including its
  // true physical ghost coordinates.  Canonical global field arrays cannot
  // represent those nonperiodic ghost samples, so callers using a nonuniform
  // evaluator invoke this after seed/restart. A failure occurs after mutation
  // may have begun and therefore poisons the distributed runtime.
  void sample_external_fields(numerics::IFieldEvaluator& evaluator,
                              const core::IFieldSource& source,
                              Real length_scale = Real{1},
                              Real e_field_scale = Real{1},
                              Real b_field_scale = Real{1});

  [[nodiscard]] Real cfl_limit() const;
  void step(Real dt);

  [[nodiscard]] PicGlobalState gather_state();
  // Extract only this rank's endpoint-owned diagnostics data.  Field and
  // particle payloads never cross MPI ranks; the only collective is worker
  // failure consensus.  Passing false avoids particle snapshots for
  // field-only cadence diagnostics.
  [[nodiscard]] std::vector<PicOwnedShard> local_owned_shards(
      bool include_particles = true);
  [[nodiscard]] std::vector<std::uint64_t> alive_counts();
  [[nodiscard]] std::vector<Real> kinetic_energies();
  [[nodiscard]] Real total_em_energy();
  [[nodiscard]] Real gauss_residual();

  [[nodiscard]] CheckpointMetadata checkpoint_metadata(
      std::uint64_t step, double time, std::string_view unit_system,
      std::span<const pic::SpeciesConfig> species) const;
  void write_checkpoint(const std::filesystem::path& path,
                        std::uint64_t step, double time,
                        std::string_view unit_system,
                        std::span<const std::uint8_t> diagnostic_state = {});
  // Each expected species capacity is the trusted global particle limit from
  // the input deck. Restart rejects a larger stored count before allocating
  // per-particle checkpoint arrays.
  [[nodiscard]] CheckpointMetadata restart_from_checkpoint(
      const std::filesystem::path& path, std::string_view unit_system,
      std::span<const pic::SpeciesConfig> expected_species,
      std::vector<std::vector<std::uint8_t>>* diagnostic_state = nullptr);

  // One-shot rank-local hooks used to verify allocation consensus, mutation
  // poisoning, and collective checkpoint-reader cleanup. The API is present
  // in every configuration; setters are inert unless test hooks were compiled.
  void inject_next_worker_task_allocation_failure_for_testing(
      bool enabled) noexcept;
  void inject_seed_post_mutation_failure_for_testing(bool enabled) noexcept;
  void inject_checkpoint_metadata_copy_failure_for_testing(
      bool enabled) noexcept;
  void inject_restart_post_reconcile_failure_for_testing(
      bool enabled) noexcept;

  // Explicit and collective.  Once a post-mutation failure poisons the
  // runtime, this is the only legal operation.
  void close();

 private:
  enum class FixedExchangeKind : std::size_t {
    field_copy = 0,
    source_copy,
    source_additive,
    cylindrical_axis,
  };

  struct DenseFields;
  struct DenseSources;
  struct LocalSources;
  struct MigrationBatch;

  [[nodiscard]] pic::EmPicConfig tile_config(std::size_t endpoint) const;
  void require_usable(bool require_seeded = true) const;
  void seed_local_fields(const PicGlobalFields& fields,
                         const PicGlobalFields* external_fields,
                         bool& mutation_started);
  void seed_local_species(std::span<const PicSpeciesState> species);
  void replace_local_species_particles(
      std::span<const PicSpeciesState> species);
  [[nodiscard]] std::vector<PicSpeciesState> snapshot_local_particles();
  [[nodiscard]] PicSpeciesState route_checkpoint_species(
      PicSpeciesState local_chunk, std::uint64_t global_count);
  void restore_impl(const PicGlobalState& state, bool particles_partitioned);
  void initialize_global_background();
  void materialize_charge();

  [[nodiscard]] DenseFields collect_fields(bool external,
                                            bool previous_b,
                                            std::string_view phase);
  void apply_fields(const DenseFields& fields, bool external,
                    bool previous_b, std::string_view phase,
                    bool* mutation_started = nullptr);
  void exchange_field_halos(std::string_view phase,
                            bool external = false,
                            bool previous_b = false);
  void reconcile_fields(std::string_view phase);
  void reconcile_magnetic(std::string_view phase);

  void reconcile_sources_device(bool use_next_charge,
                                std::string_view phase);
  void add_source_background_device(bool use_next_charge, Real density,
                                    std::string_view phase);
  void exchange_source_halos_device(
      bool use_next_charge, std::span<const std::size_t> components,
      std::string_view phase);
  void filter_sources_device();
  void correct_order_four_sources_device();
  void exchange_source_halos(LocalSources& sources,
                             std::span<const std::size_t> components,
                             std::string_view phase);
  void transfer_fixed_halos(FixedExchangeKind kind,
                            std::string_view phase);
  void apply_sources(const LocalSources& sources, bool use_next_charge,
                     std::string_view phase);
  [[nodiscard]] DenseSources collect_sources_owned(
      bool use_next_charge, std::string_view phase);
  void apply_sources(const DenseSources& sources, bool use_next_charge,
                     std::string_view phase);
  void rebuild_periodic_source_duplicates(DenseSources& sources) const;

  void migrate_particles();
  // Global mesh + decomposition in the form the migration routing kernel
  // takes, and the endpoint-to-rank table it reads. The table is built per
  // call because a DeviceBuffer belongs to whichever device was current when
  // it was allocated, and the workers each run on their own.
  [[nodiscard]] pic::PicMigrationTopology migration_topology() const;
  [[nodiscard]] backend::DeviceBuffer<std::uint64_t> endpoint_rank_table()
      const;
  [[nodiscard]] std::unique_ptr<MigrationBatch>
  extract_departing_particles();
  void route_departing_particles(MigrationBatch& batch);
  void commit_migrated_particles(MigrationBatch& batch);
  [[nodiscard]] std::vector<std::vector<std::byte>>
  exchange_variable_payloads(
      std::span<const std::vector<std::byte>> outgoing,
      std::string_view phase);
  void validate_distributed_particle_ids(
      std::span<const PicSpeciesState> species,
      std::string_view phase);
  void update_transport_telemetry() noexcept;
  [[nodiscard]] std::vector<PicSpeciesState> gather_particles();
  [[nodiscard]] PicBoundaryState gather_boundary_state();
  void apply_boundary_state(const PicBoundaryState& state);
  void write_checkpoint_impl(const std::filesystem::path& path,
                             std::uint64_t step, double time,
                             std::string_view unit_system,
                             std::span<const std::uint8_t> diagnostic_state);
  [[nodiscard]] CheckpointMetadata prepare_checkpoint_metadata(
      std::uint64_t step, double time, std::string_view unit_system,
      std::span<const std::uint8_t> diagnostic_state,
      std::vector<std::uint8_t>& encoded_diagnostic_state);
  [[nodiscard]] PicCheckpointSnapshot capture_checkpoint_snapshot(
      std::uint64_t step);
  void write_checkpoint_lattices(
      ParallelCheckpointWriter& writer,
      const PicCheckpointSnapshot& snapshot);
  void write_checkpoint_runtime(
      ParallelCheckpointWriter& writer,
      const PicCheckpointSnapshot& snapshot);
  void write_checkpoint_particles(
      ParallelCheckpointWriter& writer,
      const PicCheckpointSnapshot& snapshot);
  void read_restart_lattices(ParallelCheckpointReader& reader,
                             PicRestartPayload& payload);
  void read_restart_runtime(ParallelCheckpointReader& reader,
                            PicRestartPayload& payload);
  void read_restart_particles(
      ParallelCheckpointReader& reader,
      std::span<const pic::SpeciesConfig> expected_species,
      PicRestartPayload& payload);
  void validate_restart_payload(
      const CheckpointMetadata& stored, PicRestartPayload& payload);
  void commit_restart_payload(PicRestartPayload& payload,
                              bool& mutation_started);
  [[nodiscard]] std::size_t owner_for_position(Real x, Real y) const;
  [[noreturn]] void poison_collectively(std::string_view phase,
                                        std::string_view local_message);

  MpiRuntime* runtime_{nullptr};
  EndpointMapping mapping_{};
  VirtualTopology topology_;
  pic::EmPicConfig global_config_{};
  std::vector<int> local_devices_{};
  std::unique_ptr<EndpointWorkerPool> workers_{};
  std::vector<std::unique_ptr<pic::EmPic2D3V>> solvers_{};
  std::unique_ptr<PicFixedExchangePlan> fixed_exchange_plan_{};
  std::vector<std::unique_ptr<PicHaloBuffers>> halo_buffers_{};
  std::unique_ptr<Transport> transport_{};
  std::vector<numerics::DistributedFilterStencil> filter_stencils_{};
  PicRuntimeTelemetry telemetry_{};
  std::uint64_t worker_epoch_{0};
  std::uint64_t step_count_{0};
  Real previous_dt_{Real{0}};
  bool has_previous_dt_{false};
  detail::RuntimeLifecycleState lifecycle_{};
  // Inert unless test hooks are enabled; used to prove that a failure after
  // checkpoint lattices, halos, particles, and metadata are published poisons
  // every replica while retaining collective close.
  bool inject_restart_post_reconcile_failure_{false};
};

}  // namespace quasar::distributed
