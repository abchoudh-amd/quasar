#pragma once

#include "quasar/core/field_source.hpp"
#include "quasar/distributed/device_mapping.hpp"
#include "quasar/distributed/mhd_runtime.hpp"
#include "quasar/distributed/pic_runtime.hpp"
#include "quasar/distributed/topology.hpp"
#include "quasar/numerics/field_evaluator.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace quasar::distributed {

struct SessionTelemetry {
  std::uint64_t barriers{0};
  std::uint64_t endpoint_configurations{0};
  std::uint64_t topology_selections{0};
};

struct SessionTelemetrySnapshot {
  SessionTelemetry counters{};
  std::optional<MhdRuntimeTelemetry> mhd{};
  std::optional<TransportResolution> mhd_transport{};
  std::optional<PicRuntimeTelemetry> pic{};
  std::optional<TransportResolution> pic_transport{};
  std::size_t endpoint_count{0};
  std::size_t devices_per_rank{0};
};

// Process-global owner of one distributed runtime and, at most, one active
// physics runtime. Construction, all collective operations, and close must be
// performed on the orchestration thread. Destruction never performs MPI work:
// destroying an open session instead permanently poisons session creation in
// that process so abandoned communicators cannot be reused accidentally.
class RuntimeSession {
 public:
  RuntimeSession();
  ~RuntimeSession() noexcept;

  RuntimeSession(const RuntimeSession&) = delete;
  RuntimeSession& operator=(const RuntimeSession&) = delete;
  RuntimeSession(RuntimeSession&&) = delete;
  RuntimeSession& operator=(RuntimeSession&&) = delete;

  [[nodiscard]] int rank() const noexcept;
  [[nodiscard]] int size() const noexcept;
  [[nodiscard]] int node_rank() const noexcept;
  [[nodiscard]] int node_size() const noexcept;
  [[nodiscard]] int thread_level() const noexcept;
  [[nodiscard]] bool owns_mpi() const;
  [[nodiscard]] bool closed() const;

  [[nodiscard]] EndpointMapping endpoint_mapping() const;
  [[nodiscard]] std::optional<VirtualTopology> topology() const;
  [[nodiscard]] SessionTelemetrySnapshot telemetry() const;

  void barrier();

  // Configuration-stable test seam. The setter is inert unless the library
  // was built with QUASAR_DISTRIBUTED_TEST_HOOKS enabled.
  void inject_candidate_cleanup_failure_for_testing(bool enabled);
  void inject_mhd_close_failure_for_testing(bool enabled);

  void consensus(bool success, std::string_view phase,
                 std::string_view message);
  void require_same_string(std::string_view value, std::string_view phase,
                           std::string_view message);

  void configure_owned_devices(std::vector<DeviceIdentity> local_devices,
                               std::string_view parse_error = {});
  void configure_devices(std::vector<int> eligible_ordinals,
                         std::string_view parse_error = {});

  // parse_error is bounded by the caller before any rank enters this
  // collective. It lets language bindings keep language-specific parsing out
  // of the library while still turning a local parse failure into a common
  // collective failure.
  void select_topology(
      std::size_t global_nx, std::size_t global_ny,
      std::optional<DecompositionShape> shape,
      std::size_t minimum_tile_width, std::string_view parse_error = {});

  void start_mhd(mhd::MhdConfig config, MhdGlobalState state,
                 std::optional<MhdGlobalBackground> background,
                 TransportPolicy transport_policy,
                 std::string_view parse_error = {});
  [[nodiscard]] CheckpointMetadata restart_mhd(
      mhd::MhdConfig config, const std::string& path,
      const std::string& unit_system,
      const std::optional<MhdGlobalBackground>& expected_background,
      TransportPolicy transport_policy, std::string_view parse_error,
      std::vector<std::vector<std::uint8_t>>& diagnostic_state);
  [[nodiscard]] Real mhd_cfl_limit();
  void mhd_step(Real dt, bool check_cfl);
  [[nodiscard]] Real mhd_divergence_b_max();
  [[nodiscard]] MhdGlobalState mhd_gather_state();
  [[nodiscard]] std::vector<Real> mhd_gather_cell_component(
      std::string_view component);
  [[nodiscard]] std::vector<MhdOwnedShard> mhd_local_owned_shards();
  [[nodiscard]] MhdGlobalCellSums mhd_global_cell_sums();
  void mhd_write_checkpoint(
      const std::string& path, std::uint64_t step, double time,
      const std::string& unit_system,
      std::span<const std::uint8_t> diagnostic_state);
  void close_mhd();

  void start_pic(pic::EmPicConfig config, PicGlobalFields fields,
                 std::optional<PicGlobalFields> external_fields,
                 std::vector<PicSpeciesState> species,
                 TransportPolicy transport_policy,
                 std::string_view parse_error = {});
  [[nodiscard]] CheckpointMetadata restart_pic(
      pic::EmPicConfig config, const std::string& path,
      const std::string& unit_system,
      const std::vector<pic::SpeciesConfig>& expected_species,
      TransportPolicy transport_policy, std::string_view parse_error,
      std::vector<std::vector<std::uint8_t>>& diagnostic_state);
  [[nodiscard]] Real pic_cfl_limit();
  void pic_sample_external_fields(
      numerics::IFieldEvaluator& evaluator,
      const core::IFieldSource& source, Real length_scale,
      Real e_field_scale, Real b_field_scale);
  void pic_step(Real dt);
  [[nodiscard]] PicGlobalState pic_gather_state();
  [[nodiscard]] std::vector<PicOwnedShard> pic_local_owned_shards(
      bool include_particles);
  [[nodiscard]] std::vector<std::uint64_t> pic_alive_counts();
  [[nodiscard]] std::vector<Real> pic_kinetic_energies();
  [[nodiscard]] Real pic_total_em_energy();
  [[nodiscard]] Real pic_gauss_residual();
  void pic_write_checkpoint(
      const std::string& path, std::uint64_t step, double time,
      const std::string& unit_system,
      std::span<const std::uint8_t> diagnostic_state);
  void close_pic();

  // Completes every teardown obligation even if an earlier one fails. The
  // first failure is rethrown after the MPI runtime has had its close attempt.
  void close();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_{};
};

}  // namespace quasar::distributed
