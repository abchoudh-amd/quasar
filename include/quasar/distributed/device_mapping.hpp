#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace quasar::distributed {

// A node-local HIP device as observed through one rank's visibility mask.
// UUID is preferred as the physical identity; PCI bus ID is the fallback.
// At least one of them must be populated before devices are compared across
// ranks.  Ordinals are deliberately not treated as physical identities.
struct DeviceIdentity {
  int         ordinal{-1};
  std::string uuid{};
  std::string pci_bus_id{};

  [[nodiscard]] std::string physical_key() const;
};

struct RankDeviceVisibility {
  int                         world_rank{-1};
  int                         node_local_rank{-1};
  std::string                 node_id{};
  std::vector<DeviceIdentity> visible_devices{};
};

struct RankDeviceAssignment {
  int                         world_rank{-1};
  int                         node_local_rank{-1};
  std::string                 node_id{};
  std::vector<DeviceIdentity> owned_devices{};
};

struct Endpoint {
  std::size_t    index{0};
  int            world_rank{-1};
  int            node_local_rank{-1};
  std::size_t    rank_local_index{0};
  std::string    node_id{};
  DeviceIdentity device{};
};

// Validated rank-major endpoint ordering.  All ranks own the same positive
// number of devices, so an endpoint can be addressed as rank*G + local_index.
class EndpointMapping {
 public:
  EndpointMapping() = default;

  [[nodiscard]] std::size_t size() const noexcept { return endpoints_.size(); }
  [[nodiscard]] bool empty() const noexcept { return endpoints_.empty(); }
  [[nodiscard]] std::size_t rank_count() const noexcept { return rank_count_; }
  [[nodiscard]] std::size_t devices_per_rank() const noexcept {
    return devices_per_rank_;
  }
  [[nodiscard]] std::span<const Endpoint> endpoints() const noexcept {
    return endpoints_;
  }
  [[nodiscard]] const Endpoint& endpoint(std::size_t index) const;
  [[nodiscard]] const Endpoint& endpoint_for(int world_rank,
                                             std::size_t local_index) const;
  [[nodiscard]] std::span<const Endpoint> endpoints_for_rank(
      int world_rank) const;

 private:
  friend EndpointMapping make_endpoint_mapping(
      std::span<const RankDeviceAssignment> assignments);

  std::size_t           rank_count_{0};
  std::size_t           devices_per_rank_{0};
  std::vector<Endpoint> endpoints_{};
};

// Apply a user-supplied eligible ordinal list to every rank's visibility view.
// The returned views preserve the ordinal-list order.  Passing an empty list is
// rejected; the CLI's `auto` path should call auto_assign_devices directly.
[[nodiscard]] std::vector<RankDeviceVisibility> select_device_ordinals(
    std::span<const RankDeviceVisibility> visibility,
    std::span<const int> eligible_ordinals);

// Resolve launcher visibility on each node:
//   * pairwise-disjoint masks are already assigned and are kept as-is;
//   * an identical shared pool is divided evenly by node-local rank;
//   * partially overlapping masks are ambiguous and are rejected.
// The resulting assignments are returned in MPI world-rank order and must have
// a uniform positive GPU count across the whole world.
[[nodiscard]] std::vector<RankDeviceAssignment> auto_assign_devices(
    std::span<const RankDeviceVisibility> visibility);

// Validate explicit ownership reports, reject duplicate physical ownership on
// a node, and build rank-major virtual endpoints.
[[nodiscard]] EndpointMapping make_endpoint_mapping(
    std::span<const RankDeviceAssignment> assignments);

[[nodiscard]] EndpointMapping make_auto_endpoint_mapping(
    std::span<const RankDeviceVisibility> visibility);

}  // namespace quasar::distributed
