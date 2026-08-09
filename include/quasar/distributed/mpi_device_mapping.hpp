#pragma once

#include "quasar/distributed/device_mapping.hpp"
#include "quasar/distributed/mpi_runtime.hpp"

#include <span>

namespace quasar::distributed {

// Discover every rank's launcher-visible HIP devices, allgather their physical
// identities, and resolve a rank-major ownership map.  An empty ordinal span
// selects the `auto` policy; a non-empty span is an eligible node-local pool,
// which is still partitioned among ranks sharing the same visibility mask.
[[nodiscard]] EndpointMapping discover_endpoint_mapping(
    MpiRuntime& runtime, std::span<const int> eligible_ordinals = {});

}  // namespace quasar::distributed
