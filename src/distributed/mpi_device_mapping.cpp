#include "quasar/distributed/mpi_device_mapping.hpp"

#include "mpi_runtime_native.hpp"
#include "runtime_helpers.hpp"

#include "quasar/backend/device.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace quasar::distributed {
namespace {

void require_local_preflight(MpiRuntime& runtime, bool success,
                             std::string_view phase,
                             std::string_view message) {
  runtime.require_collective_success(
      success
          ? CollectiveErrorRecord::success(0, runtime.rank(), phase)
          : CollectiveErrorRecord::failure(
                0, runtime.rank(), -1, -1, phase, message));
}

void append_string(std::ostringstream& output, std::string_view value) {
  output << value.size() << ':' << value;
}

std::string read_string(std::string_view input, std::size_t& cursor) {
  const std::size_t separator = input.find(':', cursor);
  if (separator == std::string_view::npos) {
    throw std::runtime_error{"invalid gathered GPU visibility record"};
  }
  const std::string size_text{input.substr(cursor, separator - cursor)};
  std::size_t parsed = 0;
  const unsigned long long size = std::stoull(size_text, &parsed);
  if (parsed != size_text.size() || size > input.size() - separator - 1) {
    throw std::runtime_error{"invalid gathered GPU visibility string length"};
  }
  cursor = separator + 1;
  std::string result{input.substr(cursor, static_cast<std::size_t>(size))};
  cursor += static_cast<std::size_t>(size);
  return result;
}

std::string serialize_visibility(const RankDeviceVisibility& visibility) {
  std::ostringstream output;
  output << visibility.world_rank << ' ' << visibility.node_local_rank << ' ';
  append_string(output, visibility.node_id);
  output << visibility.visible_devices.size() << ';';
  for (const auto& device : visibility.visible_devices) {
    output << device.ordinal << ' ';
    append_string(output, device.uuid);
    append_string(output, device.pci_bus_id);
  }
  return output.str();
}

RankDeviceVisibility parse_visibility(std::string_view input) {
  RankDeviceVisibility result;
  std::istringstream header{std::string{input}};
  if (!(header >> result.world_rank >> result.node_local_rank)) {
    throw std::runtime_error{"invalid gathered GPU visibility header"};
  }
  const auto header_end = header.tellg();
  if (header_end < 0) {
    throw std::runtime_error{"invalid gathered GPU visibility header"};
  }
  std::size_t cursor = static_cast<std::size_t>(header_end);
  while (cursor < input.size() && input[cursor] == ' ') ++cursor;
  result.node_id = read_string(input, cursor);

  const std::size_t count_end = input.find(';', cursor);
  if (count_end == std::string_view::npos) {
    throw std::runtime_error{"invalid gathered GPU visibility device count"};
  }
  const std::string count_text{input.substr(cursor, count_end - cursor)};
  std::size_t count_parsed = 0;
  const std::size_t count = static_cast<std::size_t>(
      std::stoull(count_text, &count_parsed));
  if (count_parsed != count_text.size()) {
    throw std::runtime_error{"invalid gathered GPU visibility device count"};
  }
  cursor = count_end + 1;
  // Every encoded device consumes at least an ordinal byte, a separator, and
  // two length prefixes.  Reject an impossible peer-supplied count before it
  // can drive an unbounded reserve().
  if (count > input.size() - cursor) {
    throw std::runtime_error{"gathered GPU visibility device count exceeds payload"};
  }
  result.visible_devices.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t ordinal_end = input.find(' ', cursor);
    if (ordinal_end == std::string_view::npos) {
      throw std::runtime_error{"invalid gathered GPU ordinal"};
    }
    const std::string ordinal_text{input.substr(cursor, ordinal_end - cursor)};
    std::size_t ordinal_parsed = 0;
    const int ordinal = std::stoi(ordinal_text, &ordinal_parsed);
    if (ordinal_parsed != ordinal_text.size()) {
      throw std::runtime_error{"invalid gathered GPU ordinal"};
    }
    cursor = ordinal_end + 1;
    DeviceIdentity identity;
    identity.ordinal = ordinal;
    identity.uuid = read_string(input, cursor);
    identity.pci_bus_id = read_string(input, cursor);
    result.visible_devices.push_back(std::move(identity));
  }
  if (cursor != input.size()) {
    throw std::runtime_error{"gathered GPU visibility has trailing data"};
  }
  return result;
}

std::vector<std::string> allgather_strings(MpiRuntime& runtime,
                                           const std::string& local) {
  require_local_preflight(
      runtime,
      local.size()
          <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
      "device-visibility-size",
      "GPU visibility record exceeds MPI count range");
  const int local_size = static_cast<int>(local.size());
  std::vector<int> sizes;
  bool allocation_ok = true;
  try {
    sizes.resize(static_cast<std::size_t>(runtime.size()));
  } catch (...) {
    allocation_ok = false;
  }
  require_local_preflight(runtime, allocation_ok,
                          "device-visibility-size-storage",
                          "GPU visibility size storage allocation failed");
  check_mpi(MPI_Allgather(
                &local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT,
                detail::MpiRuntimeNativeAccess::world(runtime)),
            "MPI_Allgather(GPU visibility sizes)");
  std::vector<int> offsets;
  std::vector<char> bytes;
  std::int64_t total = 0;
  bool payload_ok = true;
  try {
    offsets.resize(sizes.size());
    for (std::size_t index = 0; index < sizes.size(); ++index) {
      if (sizes[index] < 0
          || total > std::numeric_limits<int>::max() - sizes[index]) {
        payload_ok = false;
        break;
      }
      offsets[index] = static_cast<int>(total);
      total += sizes[index];
    }
    if (payload_ok) bytes.resize(static_cast<std::size_t>(total));
  } catch (...) {
    payload_ok = false;
  }
  require_local_preflight(
      runtime, payload_ok, "device-visibility-payload-storage",
      "gathered GPU visibility is too large or cannot be allocated");
  char dummy{};
  check_mpi(MPI_Allgatherv(local.data(), local_size, MPI_CHAR,
                           bytes.empty() ? &dummy : bytes.data(),
                           sizes.data(), offsets.data(), MPI_CHAR,
                           detail::MpiRuntimeNativeAccess::world(runtime)),
            "MPI_Allgatherv(GPU visibility)");
  std::vector<std::string> result;
  result.reserve(sizes.size());
  for (std::size_t index = 0; index < sizes.size(); ++index) {
    result.emplace_back(bytes.data() + offsets[index],
                        static_cast<std::size_t>(sizes[index]));
  }
  return result;
}

std::string processor_name() {
  std::array<char, MPI_MAX_PROCESSOR_NAME + 1> storage{};
  int length = 0;
  check_mpi(MPI_Get_processor_name(storage.data(), &length),
            "MPI_Get_processor_name");
  return std::string{storage.data(), static_cast<std::size_t>(length)};
}

}  // namespace

EndpointMapping discover_endpoint_mapping(
    MpiRuntime& runtime, std::span<const int> eligible_ordinals) {
  runtime.require_orchestration_thread();

  bool selection_agrees = true;
  std::string selection_error;
  require_local_preflight(
      runtime,
      eligible_ordinals.size()
          <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
      "device-selection-size",
      "explicit GPU ordinal pool exceeds the MPI count range");
  const int local_count = static_cast<int>(eligible_ordinals.size());
  std::vector<int> counts;
  bool selection_storage_ok = true;
  try {
    counts.resize(static_cast<std::size_t>(runtime.size()));
  } catch (...) {
    selection_storage_ok = false;
  }
  require_local_preflight(runtime, selection_storage_ok,
                          "device-selection-size-storage",
                          "explicit GPU ordinal size storage allocation failed");
  check_mpi(MPI_Allgather(
                &local_count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                detail::MpiRuntimeNativeAccess::world(runtime)),
            "MPI_Allgather(explicit GPU ordinal counts)");
  std::vector<int> offsets;
  std::vector<int> gathered;
  try {
    offsets.resize(counts.size());
    std::int64_t total = 0;
    for (std::size_t index = 0; index < counts.size(); ++index) {
      if (counts[index] < 0 ||
          total > std::numeric_limits<int>::max() - counts[index]) {
        selection_storage_ok = false;
        break;
      }
      offsets[index] = static_cast<int>(total);
      total += counts[index];
    }
    if (selection_storage_ok) {
      gathered.resize(static_cast<std::size_t>(total));
    }
  } catch (...) {
    selection_storage_ok = false;
  }
  require_local_preflight(
      runtime, selection_storage_ok, "device-selection-payload-storage",
      "gathered explicit GPU ordinal pools are too large or cannot be allocated");
  int dummy{};
  check_mpi(MPI_Allgatherv(
                eligible_ordinals.data(), local_count, MPI_INT,
                gathered.empty() ? &dummy : gathered.data(), counts.data(),
                offsets.data(), MPI_INT,
                detail::MpiRuntimeNativeAccess::world(runtime)),
            "MPI_Allgatherv(explicit GPU ordinals)");
  try {
    for (std::size_t rank = 0; rank < counts.size(); ++rank) {
      if (counts[rank] != local_count ||
          !std::equal(eligible_ordinals.begin(), eligible_ordinals.end(),
                      gathered.begin() + offsets[rank])) {
        selection_agrees = false;
        selection_error =
            "MPI ranks supplied different explicit GPU ordinal pools";
        break;
      }
    }
  } catch (const std::exception& error) {
    selection_agrees = false;
    capture_error_text(selection_error, error.what());
  } catch (...) {
    selection_agrees = false;
  }
  runtime.require_collective_success(
      selection_agrees
          ? CollectiveErrorRecord::success(
                0, runtime.rank(), "device-selection-agreement")
          : CollectiveErrorRecord::failure(
                0, runtime.rank(), -1, -1,
                "device-selection-agreement",
                error_text_or(selection_error,
                              "explicit GPU ordinal agreement failed")));

  RankDeviceVisibility local;
  local.world_rank = runtime.rank();
  local.node_local_rank = runtime.node_rank();

  bool discovery_success = true;
  std::string discovery_error;
  try {
    local.node_id = processor_name();
    const int count = backend::device_count();
    if (count <= 0) {
      throw std::runtime_error{"MPI rank has no visible HIP devices"};
    }
    local.visible_devices.reserve(static_cast<std::size_t>(count));
    for (int ordinal = 0; ordinal < count; ++ordinal) {
      local.visible_devices.push_back(DeviceIdentity{
          ordinal,
          backend::device_uuid(ordinal),
          backend::device_pci_bus_id(ordinal)});
    }
  } catch (const std::exception& error) {
    discovery_success = false;
    capture_error_text(discovery_error, error.what());
  } catch (...) {
    discovery_success = false;
  }
  runtime.require_collective_success(
      discovery_success
          ? CollectiveErrorRecord::success(
                1, runtime.rank(), "device-discovery")
            : CollectiveErrorRecord::failure(
                1, runtime.rank(), -1, -1,
                "device-discovery",
                error_text_or(discovery_error, "GPU discovery failed")));

  std::vector<RankDeviceVisibility> visibility;
  bool parse_success = true;
  std::string parse_error;
  std::string local_visibility;
  try {
    local_visibility = serialize_visibility(local);
  } catch (const std::exception& error) {
    parse_success = false;
    capture_error_text(parse_error, error.what());
  } catch (...) {
    parse_success = false;
  }
  runtime.require_collective_success(
      parse_success
          ? CollectiveErrorRecord::success(
                2, runtime.rank(), "device-visibility-serialize")
            : CollectiveErrorRecord::failure(
                2, runtime.rank(), -1, -1,
                "device-visibility-serialize",
                error_text_or(parse_error,
                              "GPU visibility serialization failed")));
  try {
    const auto records = allgather_strings(runtime, local_visibility);
    visibility.reserve(records.size());
    for (const auto& record : records) {
      visibility.push_back(parse_visibility(record));
    }
  } catch (const std::exception& error) {
    parse_success = false;
    capture_error_text(parse_error, error.what());
  } catch (...) {
    parse_success = false;
  }
  runtime.require_collective_success(
      parse_success
          ? CollectiveErrorRecord::success(
                2, runtime.rank(), "device-visibility-gather")
            : CollectiveErrorRecord::failure(
                2, runtime.rank(), -1, -1,
                "device-visibility-gather",
                error_text_or(parse_error,
                              "GPU visibility gather or parse failed")));

  bool mapping_success = true;
  std::string mapping_error;
  EndpointMapping mapping;
  try {
    if (eligible_ordinals.empty()) {
      mapping = make_auto_endpoint_mapping(visibility);
    } else {
      auto selected = select_device_ordinals(visibility, eligible_ordinals);
      mapping = make_auto_endpoint_mapping(selected);
    }
  } catch (const std::exception& error) {
    mapping_success = false;
    capture_error_text(mapping_error, error.what());
  } catch (...) {
    mapping_success = false;
  }
  runtime.require_collective_success(
      mapping_success
          ? CollectiveErrorRecord::success(
                3, runtime.rank(), "device-ownership")
            : CollectiveErrorRecord::failure(
                3, runtime.rank(), -1, -1,
                "device-ownership",
                error_text_or(mapping_error,
                              "GPU endpoint ownership mapping failed")));
  return mapping;
}

}  // namespace quasar::distributed
