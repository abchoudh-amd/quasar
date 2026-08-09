#include "quasar/distributed/device_mapping.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace quasar::distributed {
namespace {

std::string normalized_identity(std::string_view value) {
  std::string normalized;
  normalized.reserve(value.size());
  for (const unsigned char character : value) {
    if (!std::isspace(character)) {
      normalized.push_back(static_cast<char>(std::tolower(character)));
    }
  }
  return normalized;
}

void validate_device(const DeviceIdentity& device, int rank) {
  if (device.ordinal < 0) {
    std::ostringstream message;
    message << "rank " << rank << " reported a negative device ordinal";
    throw std::invalid_argument{message.str()};
  }
  (void)device.physical_key();
}

template <class RankRecord, class DeviceAccessor>
void validate_rank_records(std::span<const RankRecord> records,
                           DeviceAccessor devices) {
  if (records.empty()) {
    throw std::invalid_argument{
        "distributed device mapping requires at least one MPI rank"};
  }

  std::set<int> world_ranks;
  std::map<std::string, std::set<int>> local_ranks;
  for (const auto& record : records) {
    if (record.world_rank < 0 || record.node_local_rank < 0) {
      throw std::invalid_argument{
          "device reports require non-negative world and node-local ranks"};
    }
    if (record.node_id.empty()) {
      throw std::invalid_argument{"device reports require a non-empty node ID"};
    }
    if (!world_ranks.insert(record.world_rank).second) {
      throw std::invalid_argument{"duplicate MPI world rank in device reports"};
    }
    if (!local_ranks[record.node_id].insert(record.node_local_rank).second) {
      throw std::invalid_argument{
          "duplicate node-local rank in device reports"};
    }

    std::unordered_set<std::string> identities;
    std::set<int> ordinals;
    for (const DeviceIdentity& device : devices(record)) {
      validate_device(device, record.world_rank);
      if (!ordinals.insert(device.ordinal).second) {
        throw std::invalid_argument{
            "a rank reported the same visible device ordinal more than once"};
      }
      if (!identities.insert(device.physical_key()).second) {
        throw std::invalid_argument{
            "a rank reported the same physical GPU more than once"};
      }
    }
  }

  int expected_world_rank = 0;
  for (const int rank : world_ranks) {
    if (rank != expected_world_rank++) {
      throw std::invalid_argument{
          "MPI world ranks in device reports must be contiguous from zero"};
    }
  }
  for (const auto& [node, ranks] : local_ranks) {
    (void)node;
    int expected_local_rank = 0;
    for (const int rank : ranks) {
      if (rank != expected_local_rank++) {
        throw std::invalid_argument{
            "node-local ranks in device reports must be contiguous from zero"};
      }
    }
  }
}

std::unordered_set<std::string> identity_set(
    const std::vector<DeviceIdentity>& devices) {
  std::unordered_set<std::string> result;
  for (const DeviceIdentity& device : devices) {
    result.insert(device.physical_key());
  }
  return result;
}

bool sets_equal(const std::unordered_set<std::string>& left,
                const std::unordered_set<std::string>& right) {
  if (left.size() != right.size()) return false;
  for (const std::string& identity : left) {
    if (!right.contains(identity)) return false;
  }
  return true;
}

bool sets_disjoint(const std::unordered_set<std::string>& left,
                   const std::unordered_set<std::string>& right) {
  const auto& smaller = left.size() <= right.size() ? left : right;
  const auto& larger  = left.size() <= right.size() ? right : left;
  for (const std::string& identity : smaller) {
    if (larger.contains(identity)) return false;
  }
  return true;
}

void require_uniform_assignment(
    const std::vector<RankDeviceAssignment>& assignments) {
  if (assignments.empty()) return;
  const std::size_t expected = assignments.front().owned_devices.size();
  if (expected == 0) {
    throw std::invalid_argument{
        "every distributed MPI rank must own at least one GPU"};
  }
  for (const auto& assignment : assignments) {
    if (assignment.owned_devices.size() != expected) {
      throw std::invalid_argument{
          "distributed v1 requires the same GPU count on every MPI rank"};
    }
  }
}

}  // namespace

std::string DeviceIdentity::physical_key() const {
  if (const std::string normalized = normalized_identity(uuid);
      !normalized.empty()) {
    return "uuid:" + normalized;
  }
  if (const std::string normalized = normalized_identity(pci_bus_id);
      !normalized.empty()) {
    return "pci:" + normalized;
  }
  throw std::invalid_argument{
      "physical GPU identity requires a UUID or PCI bus ID"};
}

const Endpoint& EndpointMapping::endpoint(std::size_t index) const {
  if (index >= endpoints_.size()) {
    throw std::out_of_range{"virtual GPU endpoint is out of range"};
  }
  return endpoints_[index];
}

const Endpoint& EndpointMapping::endpoint_for(
    int world_rank, std::size_t local_index) const {
  if (world_rank < 0 || static_cast<std::size_t>(world_rank) >= rank_count_) {
    throw std::out_of_range{"MPI rank is outside the endpoint mapping"};
  }
  if (local_index >= devices_per_rank_) {
    throw std::out_of_range{"rank-local GPU index is outside the endpoint mapping"};
  }
  return endpoint(static_cast<std::size_t>(world_rank) * devices_per_rank_
                  + local_index);
}

std::span<const Endpoint> EndpointMapping::endpoints_for_rank(
    int world_rank) const {
  if (world_rank < 0 || static_cast<std::size_t>(world_rank) >= rank_count_) {
    throw std::out_of_range{"MPI rank is outside the endpoint mapping"};
  }
  return std::span<const Endpoint>{endpoints_}.subspan(
      static_cast<std::size_t>(world_rank) * devices_per_rank_,
      devices_per_rank_);
}

std::vector<RankDeviceVisibility> select_device_ordinals(
    std::span<const RankDeviceVisibility> visibility,
    std::span<const int> eligible_ordinals) {
  validate_rank_records(visibility, [](const auto& record) -> const auto& {
    return record.visible_devices;
  });
  if (eligible_ordinals.empty()) {
    throw std::invalid_argument{
        "an explicit device selection requires at least one ordinal"};
  }
  std::set<int> requested;
  for (const int ordinal : eligible_ordinals) {
    if (ordinal < 0 || !requested.insert(ordinal).second) {
      throw std::invalid_argument{
          "explicit device ordinals must be unique non-negative integers"};
    }
  }

  std::vector<RankDeviceVisibility> selected;
  selected.reserve(visibility.size());
  for (const auto& rank : visibility) {
    RankDeviceVisibility result{
        rank.world_rank, rank.node_local_rank, rank.node_id, {}};
    result.visible_devices.reserve(eligible_ordinals.size());
    for (const int ordinal : eligible_ordinals) {
      const auto match = std::find_if(
          rank.visible_devices.begin(), rank.visible_devices.end(),
          [ordinal](const DeviceIdentity& device) {
            return device.ordinal == ordinal;
          });
      if (match == rank.visible_devices.end()) {
        std::ostringstream message;
        message << "device ordinal " << ordinal << " is not visible on rank "
                << rank.world_rank;
        throw std::invalid_argument{message.str()};
      }
      result.visible_devices.push_back(*match);
    }
    selected.push_back(std::move(result));
  }
  return selected;
}

std::vector<RankDeviceAssignment> auto_assign_devices(
    std::span<const RankDeviceVisibility> visibility) {
  validate_rank_records(visibility, [](const auto& record) -> const auto& {
    return record.visible_devices;
  });
  for (const auto& rank : visibility) {
    if (rank.visible_devices.empty()) {
      std::ostringstream message;
      message << "rank " << rank.world_rank << " has no visible GPUs";
      throw std::invalid_argument{message.str()};
    }
  }

  std::map<std::string, std::vector<const RankDeviceVisibility*>> nodes;
  for (const auto& rank : visibility) {
    nodes[rank.node_id].push_back(&rank);
  }

  std::vector<RankDeviceAssignment> assignments;
  assignments.reserve(visibility.size());
  for (auto& [node_id, ranks] : nodes) {
    std::sort(ranks.begin(), ranks.end(), [](const auto* left, const auto* right) {
      return left->node_local_rank < right->node_local_rank;
    });

    std::vector<std::unordered_set<std::string>> sets;
    sets.reserve(ranks.size());
    for (const auto* rank : ranks) sets.push_back(identity_set(rank->visible_devices));

    bool identical = true;
    bool disjoint  = true;
    for (std::size_t i = 0; i < sets.size(); ++i) {
      for (std::size_t j = i + 1; j < sets.size(); ++j) {
        identical = identical && sets_equal(sets[i], sets[j]);
        disjoint  = disjoint && sets_disjoint(sets[i], sets[j]);
      }
    }

    if (identical) {
      const auto& pool = ranks.front()->visible_devices;
      if (pool.size() % ranks.size() != 0) {
        std::ostringstream message;
        message << "node " << node_id << " exposes " << pool.size()
                << " GPUs to " << ranks.size()
                << " ranks, which cannot be partitioned uniformly";
        throw std::invalid_argument{message.str()};
      }
      const std::size_t count = pool.size() / ranks.size();
      if (count == 0) {
        throw std::invalid_argument{
            "shared GPU visibility would leave a node-local rank with no GPU"};
      }
      for (std::size_t i = 0; i < ranks.size(); ++i) {
        std::vector<DeviceIdentity> owned;
        owned.reserve(count);
        for (std::size_t offset = 0; offset < count; ++offset) {
          const std::string key = pool[i * count + offset].physical_key();
          const auto local = std::find_if(
              ranks[i]->visible_devices.begin(), ranks[i]->visible_devices.end(),
              [&key](const DeviceIdentity& device) {
                return device.physical_key() == key;
              });
          // Set equality above guarantees this lookup. Keep the check so a
          // future identity-normalization change cannot silently assign an
          // ordinal reported by a different rank.
          if (local == ranks[i]->visible_devices.end()) {
            throw std::logic_error{
                "identical GPU visibility sets lost a physical identity"};
          }
          owned.push_back(*local);
        }
        assignments.push_back(RankDeviceAssignment{
            ranks[i]->world_rank,
            ranks[i]->node_local_rank,
            node_id,
            std::move(owned)});
      }
    } else if (disjoint) {
      for (const auto* rank : ranks) {
        assignments.push_back(RankDeviceAssignment{
            rank->world_rank,
            rank->node_local_rank,
            node_id,
            rank->visible_devices});
      }
    } else {
      std::ostringstream message;
      message << "GPU visibility masks on node " << node_id
              << " partially overlap; ownership is ambiguous";
      throw std::invalid_argument{message.str()};
    }
  }

  std::sort(assignments.begin(), assignments.end(),
            [](const auto& left, const auto& right) {
              return left.world_rank < right.world_rank;
            });
  require_uniform_assignment(assignments);
  return assignments;
}

EndpointMapping make_endpoint_mapping(
    std::span<const RankDeviceAssignment> assignments_view) {
  validate_rank_records(assignments_view, [](const auto& record) -> const auto& {
    return record.owned_devices;
  });

  std::vector<RankDeviceAssignment> assignments(
      assignments_view.begin(), assignments_view.end());
  std::sort(assignments.begin(), assignments.end(),
            [](const auto& left, const auto& right) {
              return left.world_rank < right.world_rank;
            });
  require_uniform_assignment(assignments);

  std::map<std::string, std::unordered_map<std::string, int>> node_owners;
  for (const auto& assignment : assignments) {
    for (const auto& device : assignment.owned_devices) {
      const std::string key = device.physical_key();
      const auto [owner, inserted] =
          node_owners[assignment.node_id].emplace(key, assignment.world_rank);
      if (!inserted) {
        std::ostringstream message;
        message << "physical GPU " << key << " on node "
                << assignment.node_id << " is owned by both rank "
                << owner->second << " and rank " << assignment.world_rank;
        throw std::invalid_argument{message.str()};
      }
    }
  }

  EndpointMapping mapping;
  mapping.rank_count_       = assignments.size();
  mapping.devices_per_rank_ = assignments.front().owned_devices.size();
  if (mapping.devices_per_rank_
      > std::numeric_limits<std::size_t>::max() / mapping.rank_count_) {
    throw std::length_error{"virtual GPU endpoint count overflows host size"};
  }
  mapping.endpoints_.reserve(
      mapping.rank_count_ * mapping.devices_per_rank_);
  for (const auto& assignment : assignments) {
    for (std::size_t local = 0; local < assignment.owned_devices.size(); ++local) {
      mapping.endpoints_.push_back(Endpoint{
          mapping.endpoints_.size(),
          assignment.world_rank,
          assignment.node_local_rank,
          local,
          assignment.node_id,
          assignment.owned_devices[local]});
    }
  }
  return mapping;
}

EndpointMapping make_auto_endpoint_mapping(
    std::span<const RankDeviceVisibility> visibility) {
  const auto assignments = auto_assign_devices(visibility);
  return make_endpoint_mapping(assignments);
}

}  // namespace quasar::distributed
