// Pybind11 bindings for Quasar's distributed MPI/multi-GPU runtime.
//
// Exposes collective session ownership, device/topology selection, PIC/MHD
// tile runtimes, diagnostics, and checkpoint/restart orchestration.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "quasar/backend/device.hpp"
#include "quasar/core/field_source.hpp"
#include "quasar/distributed/device_mapping.hpp"
#include "quasar/distributed/mhd_runtime.hpp"
#include "quasar/distributed/pic_runtime.hpp"
#include "quasar/distributed/runtime_session.hpp"
#include "quasar/distributed/topology.hpp"
#include "quasar/numerics/field_evaluator.hpp"

#include "fixed_message.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef QUASAR_DISTRIBUTED_PIC_RUNNER_AVAILABLE
#  define QUASAR_DISTRIBUTED_PIC_RUNNER_AVAILABLE 0
#endif

#ifndef QUASAR_DISTRIBUTED_MHD_RUNNER_AVAILABLE
#  define QUASAR_DISTRIBUTED_MHD_RUNNER_AVAILABLE 0
#endif

namespace py = pybind11;

namespace quasar::distributed {
namespace {

constexpr bool pic_runner_available =
    QUASAR_DISTRIBUTED_PIC_RUNNER_AVAILABLE != 0;
constexpr bool mhd_runner_available =
    QUASAR_DISTRIBUTED_MHD_RUNNER_AVAILABLE != 0;
constexpr bool complete_runner_available =
    pic_runner_available && mhd_runner_available;

constexpr std::string_view incomplete_runner_reason =
    "the native distributed foundation is built, but PIC and MHD distributed "
    "runners are not implemented";

struct ParsedOwnedDevices {
  std::vector<DeviceIdentity> devices{};
  CollectiveLocalError error{};
};

std::size_t parse_size(const py::handle value, std::string_view label,
                       bool require_positive);

std::vector<std::uint8_t> parse_checkpoint_diagnostic_fragment(
    const py::handle value) {
  if (value.is_none()) return {};
  if (!py::isinstance<py::bytes>(value)) {
    throw std::invalid_argument{
        "checkpoint diagnostic state must be bytes or None"};
  }
  char* data = nullptr;
  Py_ssize_t size = 0;
  if (PyBytes_AsStringAndSize(value.ptr(), &data, &size) != 0 || size < 0) {
    throw py::error_already_set{};
  }
  if (static_cast<std::uint64_t>(size)
      > max_checkpoint_diagnostic_state_bytes) {
    throw std::length_error{
        "checkpoint diagnostic state exceeds the bounded size"};
  }
  const auto* begin = reinterpret_cast<const std::uint8_t*>(data);
  return {begin, begin + static_cast<std::size_t>(size)};
}

py::list diagnostic_state_parts_to_python(
    const std::vector<std::vector<std::uint8_t>>& parts) {
  py::list result;
  for (const auto& part : parts) {
    result.append(py::bytes{
        reinterpret_cast<const char*>(part.data()), part.size()});
  }
  return result;
}

std::vector<int> parse_device_ordinals(const py::handle value) {
  if (py::isinstance<py::str>(value)) {
    std::string text = py::cast<std::string>(value);
    std::transform(text.begin(), text.end(), text.begin(), [](char character) {
      return static_cast<char>(
          std::tolower(static_cast<unsigned char>(character)));
    });
    text.erase(std::remove_if(text.begin(), text.end(), [](char character) {
      return std::isspace(static_cast<unsigned char>(character)) != 0;
    }), text.end());
    if (text == "auto") return {};
    throw std::invalid_argument{
        "devices must be 'auto' or a sequence of integer ordinals"};
  }
  if (!py::isinstance<py::sequence>(value) || py::isinstance<py::bytes>(value)) {
    throw std::invalid_argument{
        "devices must be 'auto' or a sequence of integer ordinals"};
  }
  const py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
  std::vector<int> result;
  result.reserve(static_cast<std::size_t>(py::len(sequence)));
  for (const py::handle item : sequence) {
    if (PyBool_Check(item.ptr()) || !PyLong_Check(item.ptr())) {
      throw std::invalid_argument{"device ordinals must be integers"};
    }
    const long long ordinal = py::cast<long long>(item);
    if (ordinal < 0 || ordinal > std::numeric_limits<int>::max()) {
      throw std::invalid_argument{
          "device ordinals must be representable non-negative integers"};
    }
    result.push_back(static_cast<int>(ordinal));
  }
  if (result.empty()) {
    throw std::invalid_argument{
        "an explicit devices sequence must not be empty"};
  }
  return result;
}

std::vector<Real> numpy_real_vector(const py::handle value,
                                    std::string_view label) {
  using Array = py::array_t<Real, py::array::c_style | py::array::forcecast>;
  const Array array = Array::ensure(value);
  if (!array) {
    throw std::invalid_argument{std::string{label} +
                                " must be a real NumPy-compatible array"};
  }
  const auto count = static_cast<std::size_t>(array.size());
  return std::vector<Real>{array.data(), array.data() + count};
}

template <class T>
std::vector<T> numpy_integral_vector(const py::handle value,
                                     std::string_view label) {
  using Array = py::array_t<T, py::array::c_style>;
  const Array array = Array::ensure(value);
  if (!array) {
    throw std::invalid_argument{std::string{label} +
                                " must be a NumPy-compatible array"};
  }
  const auto count = static_cast<std::size_t>(array.size());
  return std::vector<T>{array.data(), array.data() + count};
}

PicGlobalFields parse_pic_fields(const py::dict& values,
                                 std::string_view label) {
  PicGlobalFields fields;
  if (!values.contains("global_nx") || !values.contains("global_ny")) {
    throw std::invalid_argument{
        std::string{label} + " requires global_nx and global_ny"};
  }
  fields.global_nx = parse_size(values["global_nx"], "global_nx", true);
  fields.global_ny = parse_size(values["global_ny"], "global_ny", true);
  const auto component = [&values, label](const char* name) {
    if (!values.contains(name)) {
      throw std::invalid_argument{std::string{label} +
                                  " is missing component " + name};
    }
    return numpy_real_vector(values[name], name);
  };
  fields.ex = component("ex");
  fields.ey = component("ey");
  fields.ez = component("ez");
  fields.bx = component("bx");
  fields.by = component("by");
  fields.bz = component("bz");
  return fields;
}

pic::SpeciesConfig parse_pic_species_config(const py::dict& values) {
  for (const char* key : {"name", "charge", "mass", "capacity"}) {
    if (!values.contains(key)) {
      throw std::invalid_argument{
          std::string{"distributed PIC species config is missing "} + key};
    }
  }
  if (!py::isinstance<py::str>(values["name"])) {
    throw std::invalid_argument{"distributed PIC species name must be a string"};
  }
  pic::SpeciesConfig config;
  config.name = py::cast<std::string>(values["name"]);
  config.charge = py::cast<Real>(values["charge"]);
  config.mass = py::cast<Real>(values["mass"]);
  config.capacity = parse_size(values["capacity"], "species capacity", false);
  return config;
}

pic::ParticleSpecies::HostSnapshot parse_pic_particles(
    const py::dict& values) {
  for (const char* key : {"x", "y", "x_prev", "y_prev", "vx", "vy",
                          "vz", "vphi_deposit", "weight", "alive", "id"}) {
    if (!values.contains(key)) {
      throw std::invalid_argument{
          std::string{"distributed PIC particle state is missing "} + key};
    }
  }
  pic::ParticleSpecies::HostSnapshot particles;
  particles.x = numpy_real_vector(values["x"], "x");
  particles.y = numpy_real_vector(values["y"], "y");
  particles.x_prev = numpy_real_vector(values["x_prev"], "x_prev");
  particles.y_prev = numpy_real_vector(values["y_prev"], "y_prev");
  particles.vx = numpy_real_vector(values["vx"], "vx");
  particles.vy = numpy_real_vector(values["vy"], "vy");
  particles.vz = numpy_real_vector(values["vz"], "vz");
  particles.vphi_deposit =
      numpy_real_vector(values["vphi_deposit"], "vphi_deposit");
  particles.weight = numpy_real_vector(values["weight"], "weight");
  particles.alive =
      numpy_integral_vector<std::uint8_t>(values["alive"], "alive");
  particles.id =
      numpy_integral_vector<std::uint64_t>(values["id"], "id");
  return particles;
}

std::vector<PicSpeciesState> parse_pic_species(const py::handle values) {
  if (!py::isinstance<py::sequence>(values)
      || py::isinstance<py::str>(values)
      || py::isinstance<py::bytes>(values)) {
    throw std::invalid_argument{
        "distributed PIC species must be a sequence of dictionaries"};
  }
  const py::sequence sequence = py::reinterpret_borrow<py::sequence>(values);
  std::vector<PicSpeciesState> result;
  result.reserve(static_cast<std::size_t>(py::len(sequence)));
  std::unordered_set<std::string> names;
  std::unordered_set<std::uint64_t> identifiers;
  for (const py::handle item : sequence) {
    if (!py::isinstance<py::dict>(item)) {
      throw std::invalid_argument{
          "each distributed PIC species must be a dictionary"};
    }
    const py::dict dictionary = py::reinterpret_borrow<py::dict>(item);
    if (!dictionary.contains("config") || !dictionary.contains("particles")
        || !py::isinstance<py::dict>(dictionary["config"])
        || !py::isinstance<py::dict>(dictionary["particles"])) {
      throw std::invalid_argument{
          "distributed PIC species requires config and particles dictionaries"};
    }
    PicSpeciesState state;
    state.config = parse_pic_species_config(
        py::reinterpret_borrow<py::dict>(dictionary["config"]));
    state.particles = parse_pic_particles(
        py::reinterpret_borrow<py::dict>(dictionary["particles"]));
    validate_pic_species_state(state);
    if (!names.insert(state.config.name).second) {
      throw std::invalid_argument{
          "distributed PIC species names must be unique"};
    }
    for (const std::uint64_t identifier : state.particles.id) {
      if (!identifiers.insert(identifier).second) {
        throw std::invalid_argument{
            "distributed PIC particle IDs must be globally unique"};
      }
    }
    result.push_back(std::move(state));
  }
  return result;
}

std::vector<pic::SpeciesConfig> parse_pic_species_configs(
    const py::handle values) {
  if (!py::isinstance<py::sequence>(values)
      || py::isinstance<py::str>(values)
      || py::isinstance<py::bytes>(values)) {
    throw std::invalid_argument{
        "expected PIC species configs must be a sequence"};
  }
  const py::sequence sequence = py::reinterpret_borrow<py::sequence>(values);
  std::vector<pic::SpeciesConfig> result;
  result.reserve(static_cast<std::size_t>(py::len(sequence)));
  for (const py::handle item : sequence) {
    if (!py::isinstance<py::dict>(item)) {
      throw std::invalid_argument{
          "each expected PIC species config must be a dictionary"};
    }
    result.push_back(parse_pic_species_config(
        py::reinterpret_borrow<py::dict>(item)));
  }
  return result;
}

MhdGlobalState parse_mhd_state(const py::dict& values) {
  MhdGlobalState state;
  if (!values.contains("global_nx") || !values.contains("global_ny")) {
    throw std::invalid_argument{
        "distributed MHD state requires global_nx and global_ny"};
  }
  state.global_nx = parse_size(values["global_nx"], "global_nx", true);
  state.global_ny = parse_size(values["global_ny"], "global_ny", true);
  const auto component = [&values](const char* name) {
    if (!values.contains(name)) {
      throw std::invalid_argument{
          std::string{"distributed MHD state is missing component "} + name};
    }
    return numpy_real_vector(values[name], name);
  };
  state.rho = component("rho");
  state.mx = component("mx");
  state.my = component("my");
  state.mz = component("mz");
  state.energy = component("energy");
  state.bx_face = component("bx_face");
  state.by_face = component("by_face");
  state.bz_cell = component("bz_cell");
  validate_mhd_global_state(state);
  return state;
}

MhdGlobalBackground parse_mhd_background(const py::dict& values) {
  MhdGlobalBackground background;
  background.global_nx = parse_size(values["global_nx"], "global_nx", true);
  background.global_ny = parse_size(values["global_ny"], "global_ny", true);
  background.b0x_face = numpy_real_vector(values["b0x_face"], "b0x_face");
  background.b0y_face = numpy_real_vector(values["b0y_face"], "b0y_face");
  background.b0z_cell = numpy_real_vector(values["b0z_cell"], "b0z_cell");
  validate_mhd_global_background(background);
  return background;
}

struct ParsedTopologyRequest {
  std::size_t global_nx{0};
  std::size_t global_ny{0};
  std::size_t minimum_tile_width{0};
  std::optional<DecompositionShape> shape{};
  CollectiveLocalError error{};
};

std::string dictionary_string(const py::dict& dictionary, const char* key) {
  if (!dictionary.contains(key)) return {};
  const py::handle value = dictionary[key];
  if (!py::isinstance<py::str>(value)) {
    throw std::invalid_argument{std::string{key} + " must be a string"};
  }
  return py::cast<std::string>(value);
}

ParsedOwnedDevices parse_owned_devices(const py::handle value) {
  ParsedOwnedDevices result;
  result.error = capture_collective_local_error([&] {
    if (!py::isinstance<py::sequence>(value)
        || py::isinstance<py::str>(value)
        || py::isinstance<py::bytes>(value)) {
      throw std::invalid_argument{
          "owned devices must be a sequence of device dictionaries"};
    }
    const py::sequence sequence = py::reinterpret_borrow<py::sequence>(value);
    result.devices.reserve(static_cast<std::size_t>(py::len(sequence)));
    for (const py::handle& item : sequence) {
      if (!py::isinstance<py::dict>(item)) {
        throw std::invalid_argument{
            "each owned device must be a dictionary"};
      }
      const py::dict dictionary = py::reinterpret_borrow<py::dict>(item);
      if (!dictionary.contains("ordinal")) {
        throw std::invalid_argument{"owned device ordinal is required"};
      }
      const py::handle ordinal_value = dictionary["ordinal"];
      if (PyBool_Check(ordinal_value.ptr()) || !PyLong_Check(ordinal_value.ptr())) {
        throw std::invalid_argument{"owned device ordinal must be an integer"};
      }
      const long long ordinal = py::cast<long long>(ordinal_value);
      if (ordinal < 0
          || ordinal > static_cast<long long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument{
            "owned device ordinal is outside the supported range"};
      }
      result.devices.push_back(DeviceIdentity{
          static_cast<int>(ordinal),
          dictionary_string(dictionary, "uuid"),
          dictionary_string(dictionary, "pci_bus_id")});
    }
  });
  if (!result.error.ok()) {
    result.devices.clear();
  }
  return result;
}

std::size_t parse_size(const py::handle value, std::string_view label,
                       bool require_positive) {
  if (PyBool_Check(value.ptr()) || !PyLong_Check(value.ptr())) {
    throw std::invalid_argument{std::string{label} + " must be an integer"};
  }
  const unsigned long long parsed = py::cast<unsigned long long>(value);
  if ((require_positive && parsed == 0)
      || parsed > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument{
        std::string{label} + (require_positive
            ? " must be a representable positive integer"
            : " must be a representable non-negative integer")};
  }
  return static_cast<std::size_t>(parsed);
}

std::string parse_string(const py::handle value, std::string_view label) {
  if (!py::isinstance<py::str>(value)) {
    throw std::invalid_argument{std::string{label} + " must be a string"};
  }
  return py::cast<std::string>(value);
}

bool parse_bool(const py::handle value, std::string_view label) {
  if (!PyBool_Check(value.ptr())) {
    throw std::invalid_argument{std::string{label} + " must be a boolean"};
  }
  return py::cast<bool>(value);
}

Real parse_real(const py::handle value, std::string_view label) {
  if (PyBool_Check(value.ptr()) ||
      !(PyFloat_Check(value.ptr()) || PyLong_Check(value.ptr()))) {
    throw std::invalid_argument{
        std::string{label} + " must be a real number"};
  }
  return py::cast<Real>(value);
}

std::uint64_t parse_uint64(const py::handle value, std::string_view label) {
  if (PyBool_Check(value.ptr()) || !PyLong_Check(value.ptr())) {
    throw std::invalid_argument{
        std::string{label} + " must be a non-negative integer"};
  }
  return py::cast<std::uint64_t>(value);
}

TransportPolicy parse_transport(const py::handle value) {
  return parse_transport_policy(parse_string(value, "transport"));
}

std::string transport_policy_name(TransportPolicy policy) {
  switch (policy) {
    case TransportPolicy::automatic: return "auto";
    case TransportPolicy::staged: return "staged";
    case TransportPolicy::direct: return "direct";
  }
  throw std::invalid_argument{"unknown transport policy"};
}

void append_argument_signature(std::ostringstream& output,
                               std::string_view value) {
  output << value.size() << ':' << value << ';';
}

template <class T>
void append_argument_bits(std::ostringstream& output, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>);
  const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
  output << sizeof(T) << ':';
  output << std::hex;
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    output << static_cast<unsigned>(bytes[index]) << ',';
  }
  output << std::dec << ';';
}

DecompositionShape parse_shape_string(std::string value) {
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](char c) {
    return !std::isspace(static_cast<unsigned char>(c));
  }));
  value.erase(std::find_if(value.rbegin(), value.rend(), [](char c) {
    return !std::isspace(static_cast<unsigned char>(c));
  }).base(), value.end());
  const std::size_t separator = value.find_first_of("xX");
  if (separator == std::string::npos
      || value.find_first_of("xX", separator + 1) != std::string::npos) {
    throw std::invalid_argument{
        "decomposition must be 'auto' or PXxPY"};
  }
  const auto parse_dimension = [](std::string_view text) {
    if (text.empty()
        || !std::all_of(text.begin(), text.end(), [](char character) {
             return std::isdigit(static_cast<unsigned char>(character)) != 0;
           })) {
      throw std::invalid_argument{
          "decomposition dimensions must be positive integers"};
    }
    std::size_t consumed = 0;
    const unsigned long long dimension =
        std::stoull(std::string{text}, &consumed, 10);
    if (consumed != text.size() || dimension == 0
        || dimension > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument{
          "decomposition dimensions must be representable positive integers"};
    }
    return static_cast<std::size_t>(dimension);
  };
  return {parse_dimension(std::string_view{value}.substr(0, separator)),
          parse_dimension(std::string_view{value}.substr(separator + 1))};
}

ParsedTopologyRequest parse_topology_request(
    const py::handle global_nx, const py::handle global_ny,
    const py::handle decomposition, const py::handle minimum_tile_width) {
  ParsedTopologyRequest result;
  result.error = capture_collective_local_error([&] {
    result.global_nx = parse_size(global_nx, "global_nx", true);
    result.global_ny = parse_size(global_ny, "global_ny", true);
    result.minimum_tile_width =
        parse_size(minimum_tile_width, "minimum_tile_width", false);
    if (py::isinstance<py::str>(decomposition)) {
      std::string text = py::cast<std::string>(decomposition);
      std::string normalized = text;
      std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                     [](char character) {
                       return static_cast<char>(
                           std::tolower(static_cast<unsigned char>(character)));
                     });
      normalized.erase(
          std::remove_if(normalized.begin(), normalized.end(), [](char character) {
            return std::isspace(static_cast<unsigned char>(character)) != 0;
          }),
          normalized.end());
      if (normalized != "auto") result.shape = parse_shape_string(text);
    } else {
      if (!py::isinstance<py::sequence>(decomposition)
          || py::isinstance<py::bytes>(decomposition)) {
        throw std::invalid_argument{
            "decomposition must be 'auto', PXxPY, or a two-integer sequence"};
      }
      const py::sequence values =
          py::reinterpret_borrow<py::sequence>(decomposition);
      if (py::len(values) != 2) {
        throw std::invalid_argument{
            "decomposition must contain exactly two dimensions"};
      }
      result.shape = DecompositionShape{
          parse_size(values[0], "decomposition px", true),
          parse_size(values[1], "decomposition py", true)};
    }
  });
  return result;
}

py::list endpoint_mapping_to_python(const EndpointMapping& mapping) {
  py::list result;
  for (const auto& endpoint : mapping.endpoints()) {
    py::dict item;
    item["index"] = endpoint.index;
    item["rank"] = endpoint.world_rank;
    item["node_rank"] = endpoint.node_local_rank;
    item["rank_local_index"] = endpoint.rank_local_index;
    item["node_id"] = endpoint.node_id;
    item["ordinal"] = endpoint.device.ordinal;
    item["uuid"] = endpoint.device.uuid;
    item["pci_bus_id"] = endpoint.device.pci_bus_id;
    item["device_identity"] = endpoint.device.physical_key();
    result.append(std::move(item));
  }
  return result;
}

py::dict topology_to_python(const VirtualTopology& topology) {
  py::dict result;
  result["global_shape"] = py::make_tuple(
      topology.global_ny(), topology.global_nx());
  result["decomposition"] = py::make_tuple(
      topology.shape().px, topology.shape().py);
  result["minimum_tile_width"] = topology.minimum_tile_width();
  py::list tiles;
  for (const auto& tile : topology.tiles()) {
    py::dict item;
    item["endpoint"] = tile.endpoint;
    item["tile"] = py::make_tuple(tile.coordinate.x, tile.coordinate.y);
    item["offset"] = py::make_tuple(tile.y.begin, tile.x.begin);
    item["owned_shape"] = py::make_tuple(tile.y.size(), tile.x.size());
    tiles.append(std::move(item));
  }
  result["tiles"] = std::move(tiles);
  return result;
}

py::array_t<Real> real_vector_to_numpy(const std::vector<Real>& values) {
  py::array_t<Real> result(values.size());
  if (!values.empty()) {
    std::memcpy(result.mutable_data(), values.data(),
                values.size() * sizeof(Real));
  }
  return result;
}

py::array_t<Real> real_matrix_to_numpy(const std::vector<Real>& values,
                                       std::size_t rows,
                                       std::size_t columns) {
  if (rows != 0 && columns > std::numeric_limits<std::size_t>::max() / rows) {
    throw std::length_error{"owned MHD shard shape overflows"};
  }
  if (values.size() != rows * columns) {
    throw std::logic_error{"owned MHD shard component has the wrong size"};
  }
  py::array_t<Real> result({static_cast<py::ssize_t>(rows),
                            static_cast<py::ssize_t>(columns)});
  if (!values.empty()) {
    std::memcpy(result.mutable_data(), values.data(),
                values.size() * sizeof(Real));
  }
  return result;
}

template <class T>
py::array_t<T> integral_vector_to_numpy(const std::vector<T>& values) {
  py::array_t<T> result(values.size());
  if (!values.empty()) {
    std::memcpy(result.mutable_data(), values.data(),
                values.size() * sizeof(T));
  }
  return result;
}

py::dict pic_fields_to_python(const PicGlobalFields& fields) {
  py::dict result;
  result["global_nx"] = fields.global_nx;
  result["global_ny"] = fields.global_ny;
  result["ex"] = real_vector_to_numpy(fields.ex);
  result["ey"] = real_vector_to_numpy(fields.ey);
  result["ez"] = real_vector_to_numpy(fields.ez);
  result["bx"] = real_vector_to_numpy(fields.bx);
  result["by"] = real_vector_to_numpy(fields.by);
  result["bz"] = real_vector_to_numpy(fields.bz);
  return result;
}

py::dict pic_sources_to_python(const PicGlobalSources& sources) {
  py::dict result;
  result["global_nx"] = sources.global_nx;
  result["global_ny"] = sources.global_ny;
  result["jx"] = real_vector_to_numpy(sources.jx);
  result["jy"] = real_vector_to_numpy(sources.jy);
  result["jz"] = real_vector_to_numpy(sources.jz);
  result["charge"] = real_vector_to_numpy(sources.charge);
  return result;
}

py::dict pic_species_config_to_python(const pic::SpeciesConfig& config) {
  py::dict result;
  result["name"] = config.name;
  result["charge"] = config.charge;
  result["mass"] = config.mass;
  result["capacity"] = config.capacity;
  return result;
}

py::dict pic_particles_to_python(
    const pic::ParticleSpecies::HostSnapshot& particles) {
  py::dict result;
  result["x"] = real_vector_to_numpy(particles.x);
  result["y"] = real_vector_to_numpy(particles.y);
  result["x_prev"] = real_vector_to_numpy(particles.x_prev);
  result["y_prev"] = real_vector_to_numpy(particles.y_prev);
  result["vx"] = real_vector_to_numpy(particles.vx);
  result["vy"] = real_vector_to_numpy(particles.vy);
  result["vz"] = real_vector_to_numpy(particles.vz);
  result["vphi_deposit"] = real_vector_to_numpy(particles.vphi_deposit);
  result["weight"] = real_vector_to_numpy(particles.weight);
  result["alive"] = integral_vector_to_numpy(particles.alive);
  result["id"] = integral_vector_to_numpy(particles.id);
  return result;
}

py::dict pic_owned_array_to_python(const PicOwnedArray& array) {
  py::dict result;
  result["offset"] = py::make_tuple(array.offset_y, array.offset_x);
  result["shape"] = py::make_tuple(array.ny, array.nx);
  result["values"] = real_vector_to_numpy(array.values);
  return result;
}

py::dict pic_owned_fields_to_python(const PicOwnedFields& fields) {
  py::dict result;
  result["ex"] = pic_owned_array_to_python(fields.ex);
  result["ey"] = pic_owned_array_to_python(fields.ey);
  result["ez"] = pic_owned_array_to_python(fields.ez);
  result["bx"] = pic_owned_array_to_python(fields.bx);
  result["by"] = pic_owned_array_to_python(fields.by);
  result["bz"] = pic_owned_array_to_python(fields.bz);
  return result;
}

py::list pic_owned_shards_to_python(
    const std::vector<PicOwnedShard>& shards) {
  py::list result;
  for (const auto& shard : shards) {
    py::dict item;
    item["endpoint"] = shard.endpoint;
    item["tile"] = py::make_tuple(shard.tile_x, shard.tile_y);
    item["offset"] = py::make_tuple(shard.offset_y, shard.offset_x);
    item["owned_shape"] =
        py::make_tuple(shard.owned_ny, shard.owned_nx);
    item["fields"] = pic_owned_fields_to_python(shard.fields);
    item["external_fields"] =
        pic_owned_fields_to_python(shard.external_fields);
    py::list species;
    for (const auto& state : shard.species) {
      py::dict record;
      record["config"] = pic_species_config_to_python(state.config);
      record["particles"] = pic_particles_to_python(state.particles);
      species.append(std::move(record));
    }
    item["species"] = std::move(species);
    result.append(std::move(item));
  }
  return result;
}

py::dict pic_boundary_to_python(const PicBoundaryState& boundary) {
  py::dict result;
  py::list history;
  for (const auto& side : boundary.mur_history) {
    history.append(real_vector_to_numpy(side));
  }
  result["mur_history"] = std::move(history);
  result["mur_primed"] = integral_vector_to_numpy(
      std::vector<std::uint8_t>{boundary.mur_primed.begin(),
                                boundary.mur_primed.end()});
  result["outflow_corner_history"] =
      real_vector_to_numpy(boundary.outflow_corner_history);
  result["outflow_corners_primed"] = boundary.outflow_corners_primed;
  return result;
}

py::dict pic_state_to_python(const PicGlobalState& state) {
  py::dict result;
  result["fields"] = pic_fields_to_python(state.fields);
  result["external_fields"] = pic_fields_to_python(state.external_fields);
  result["sources"] = pic_sources_to_python(state.sources);
  result["previous_bx"] = real_vector_to_numpy(state.previous_bx);
  result["previous_by"] = real_vector_to_numpy(state.previous_by);
  result["previous_bz"] = real_vector_to_numpy(state.previous_bz);
  py::list species;
  for (const auto& item : state.species) {
    py::dict record;
    record["config"] = pic_species_config_to_python(item.config);
    record["particles"] = pic_particles_to_python(item.particles);
    species.append(std::move(record));
  }
  result["species"] = std::move(species);
  result["step_count"] = state.step_count;
  result["previous_dt"] = state.previous_dt;
  result["has_previous_dt"] = state.has_previous_dt;
  result["background_initialized"] = state.background_initialized;
  result["background_charge_density"] = state.background_charge_density;
  result["boundary"] = pic_boundary_to_python(state.boundary);
  return result;
}

py::dict mhd_state_to_python(const MhdGlobalState& state) {
  py::dict result;
  result["global_nx"] = state.global_nx;
  result["global_ny"] = state.global_ny;
  result["rho"] = real_vector_to_numpy(state.rho);
  result["mx"] = real_vector_to_numpy(state.mx);
  result["my"] = real_vector_to_numpy(state.my);
  result["mz"] = real_vector_to_numpy(state.mz);
  result["energy"] = real_vector_to_numpy(state.energy);
  result["bx_face"] = real_vector_to_numpy(state.bx_face);
  result["by_face"] = real_vector_to_numpy(state.by_face);
  result["bz_cell"] = real_vector_to_numpy(state.bz_cell);
  return result;
}

py::list mhd_owned_shards_to_python(
    const std::vector<MhdOwnedShard>& shards) {
  py::list result;
  for (const auto& shard : shards) {
    py::dict item;
    item["endpoint"] = shard.endpoint;
    item["tile"] = py::make_tuple(shard.tile_x, shard.tile_y);
    item["offset"] = py::make_tuple(shard.offset_y, shard.offset_x);
    item["owned_shape"] =
        py::make_tuple(shard.owned_ny, shard.owned_nx);
    py::dict state;
    state["rho"] = real_matrix_to_numpy(
        shard.rho, shard.owned_ny, shard.owned_nx);
    state["mx"] = real_matrix_to_numpy(
        shard.mx, shard.owned_ny, shard.owned_nx);
    state["my"] = real_matrix_to_numpy(
        shard.my, shard.owned_ny, shard.owned_nx);
    state["mz"] = real_matrix_to_numpy(
        shard.mz, shard.owned_ny, shard.owned_nx);
    state["energy"] = real_matrix_to_numpy(
        shard.energy, shard.owned_ny, shard.owned_nx);
    state["bx"] = real_matrix_to_numpy(
        shard.bx, shard.owned_ny, shard.owned_nx);
    state["by"] = real_matrix_to_numpy(
        shard.by, shard.owned_ny, shard.owned_nx);
    state["bz"] = real_matrix_to_numpy(
        shard.bz, shard.owned_ny, shard.owned_nx);
    item["state"] = std::move(state);
    result.append(std::move(item));
  }
  return result;
}

py::dict checkpoint_metadata_to_python(const CheckpointMetadata& metadata) {
  py::dict result;
  result["schema"] = metadata.schema;
  result["physics"] = metadata.physics;
  result["precision"] = metadata.precision;
  result["geometry"] = metadata.geometry;
  result["unit_system"] = metadata.unit_system;
  result["global_nx"] = metadata.global_nx;
  result["global_ny"] = metadata.global_ny;
  result["boundary_signature"] = metadata.boundary_signature;
  result["species_signature"] = metadata.species_signature;
  result["background_signature"] = metadata.background_signature;
  result["numerics_signature"] = metadata.numerics_signature;
  result["step"] = metadata.step;
  result["time"] = metadata.time;
  result["diagnostic_state_bytes"] = metadata.diagnostic_state_bytes;
  return result;
}

py::dict telemetry_to_python(const RuntimeSession& session) {
  const SessionTelemetrySnapshot snapshot = session.telemetry();
  const SessionTelemetry& telemetry = snapshot.counters;
  py::dict result;
  result["barriers"] = telemetry.barriers;
  result["endpoint_configurations"] = telemetry.endpoint_configurations;
  result["topology_selections"] = telemetry.topology_selections;
  result["endpoint_count"] = snapshot.endpoint_count;
  result["devices_per_rank"] = snapshot.devices_per_rank;
  if (snapshot.mhd) {
    py::dict mhd;
    mhd["accepted_steps"] = snapshot.mhd->accepted_steps;
    mhd["accepted_substeps"] = snapshot.mhd->accepted_substeps;
    mhd["rejected_attempts"] = snapshot.mhd->rejected_attempts;
    mhd["stage_evaluations"] = snapshot.mhd->stage_evaluations;
    mhd["state_reconciliations"] = snapshot.mhd->state_reconciliations;
    mhd["register_halo_epochs"] = snapshot.mhd->register_halo_epochs;
    mhd["canonical_face_record_passes"] =
        snapshot.mhd->canonical_face_record_passes;
    mhd["dense_residual_face_reconciliations"] =
        snapshot.mhd->dense_residual_face_reconciliations;
    mhd["dense_emf_input_reconciliations"] =
        snapshot.mhd->dense_emf_input_reconciliations;
    mhd["emf_reconciliations"] = snapshot.mhd->emf_reconciliations;
    mhd["dense_ct_collective_bytes"] =
        snapshot.mhd->dense_ct_collective_bytes;
    mhd["collective_bytes"] = snapshot.mhd->collective_bytes;
    mhd["local_shard_extractions"] =
        snapshot.mhd->local_shard_extractions;
    py::dict transport;
    transport["epochs"] = snapshot.mhd->transport.epochs;
    transport["messages"] = snapshot.mhd->transport.messages;
    transport["bytes"] = snapshot.mhd->transport.bytes;
    transport["peer_bytes"] = snapshot.mhd->transport.peer_bytes;
    transport["local_staged_bytes"] =
        snapshot.mhd->transport.local_staged_bytes;
    transport["staged_mpi_bytes"] =
        snapshot.mhd->transport.staged_mpi_bytes;
    transport["direct_mpi_bytes"] =
        snapshot.mhd->transport.direct_mpi_bytes;
    if (snapshot.mhd_transport) {
      transport["requested"] =
          transport_policy_name(snapshot.mhd_transport->requested);
      transport["interprocess"] =
          transport_policy_name(snapshot.mhd_transport->interprocess);
      transport["direct_query_recognized"] =
          snapshot.mhd_transport->direct.recognized_query;
      transport["direct_startup_probe"] =
          snapshot.mhd_transport->direct.startup_probe;
    }
    mhd["transport"] = std::move(transport);
    result["mhd"] = std::move(mhd);
  }
  if (snapshot.pic) {
    py::dict pic;
    pic["accepted_steps"] = snapshot.pic->accepted_steps;
    pic["state_reconciliations"] = snapshot.pic->state_reconciliations;
    pic["source_reconciliations"] = snapshot.pic->source_reconciliations;
    pic["particle_migrations"] = snapshot.pic->particle_migrations;
    pic["migrated_particles"] = snapshot.pic->migrated_particles;
    pic["collective_bytes"] = snapshot.pic->collective_bytes;
    pic["global_state_gathers"] = snapshot.pic->global_state_gathers;
    pic["local_shard_extractions"] =
        snapshot.pic->local_shard_extractions;
    pic["transport_epochs"] = snapshot.pic->transport_epochs;
    pic["transport_messages"] = snapshot.pic->transport_messages;
    pic["transport_bytes"] = snapshot.pic->transport_bytes;
    pic["transport_peer_bytes"] = snapshot.pic->transport_peer_bytes;
    pic["transport_local_staged_bytes"] =
        snapshot.pic->transport_local_staged_bytes;
    pic["transport_staged_mpi_bytes"] =
        snapshot.pic->transport_staged_mpi_bytes;
    pic["transport_direct_mpi_bytes"] =
        snapshot.pic->transport_direct_mpi_bytes;
    pic["checkpoint_local_lattice_writes"] =
        snapshot.pic->checkpoint_local_lattice_writes;
    pic["checkpoint_local_lattice_reads"] =
        snapshot.pic->checkpoint_local_lattice_reads;
    pic["checkpoint_global_lattice_materializations"] =
        snapshot.pic->checkpoint_global_lattice_materializations;
    py::dict transport;
    transport["epochs"] = snapshot.pic->transport_epochs;
    transport["messages"] = snapshot.pic->transport_messages;
    transport["bytes"] = snapshot.pic->transport_bytes;
    transport["peer_bytes"] = snapshot.pic->transport_peer_bytes;
    transport["local_staged_bytes"] =
        snapshot.pic->transport_local_staged_bytes;
    transport["staged_mpi_bytes"] =
        snapshot.pic->transport_staged_mpi_bytes;
    transport["direct_mpi_bytes"] =
        snapshot.pic->transport_direct_mpi_bytes;
    if (snapshot.pic_transport) {
      transport["requested"] =
          transport_policy_name(snapshot.pic_transport->requested);
      transport["interprocess"] =
          transport_policy_name(snapshot.pic_transport->interprocess);
      transport["direct_query_recognized"] =
          snapshot.pic_transport->direct.recognized_query;
      transport["direct_startup_probe"] =
          snapshot.pic_transport->direct.startup_probe;
    }
    pic["transport"] = std::move(transport);
    result["pic"] = std::move(pic);
  }
  return result;
}

}  // namespace
}  // namespace quasar::distributed

PYBIND11_MODULE(_distributed, module) {
  using quasar::distributed::RuntimeSession;
  namespace qd = quasar::distributed;

  module.doc() = "Native MPI/multi-GPU runtime foundation for Quasar";
  module.def("foundation_available", [] { return true; });
  module.def("pic_runtime_available", [] { return true; });
  module.def("mhd_runtime_available", [] { return true; });
  module.def("visible_device_count", &quasar::backend::device_count);
  module.def("is_available", [] { return qd::complete_runner_available; });
  module.def("unavailable_reason", []() -> py::object {
    if constexpr (qd::complete_runner_available) return py::none();
    return py::str{qd::incomplete_runner_reason};
  });

  py::class_<RuntimeSession>(module, "RuntimeSession")
      .def(py::init([] {
        py::gil_scoped_release release;
        return std::make_unique<RuntimeSession>();
      }))
      .def_property_readonly("rank", &RuntimeSession::rank)
      .def_property_readonly("size", &RuntimeSession::size)
      .def_property_readonly("node_rank", &RuntimeSession::node_rank)
      .def_property_readonly("node_size", &RuntimeSession::node_size)
      .def_property_readonly("thread_level", &RuntimeSession::thread_level)
      .def_property_readonly("owns_mpi", &RuntimeSession::owns_mpi)
      .def_property_readonly("closed", &RuntimeSession::closed)
      .def_property_readonly(
          "endpoint_mapping", [](const RuntimeSession& session) {
            const qd::EndpointMapping mapping = session.endpoint_mapping();
            return qd::endpoint_mapping_to_python(mapping);
          })
      .def_property_readonly("telemetry", &qd::telemetry_to_python)
      .def("_inject_candidate_cleanup_failure_for_testing",
           &RuntimeSession::inject_candidate_cleanup_failure_for_testing,
           py::arg("enabled"))
      .def("barrier", [](RuntimeSession& session) {
        py::gil_scoped_release release;
        session.barrier();
      })
      .def("collective_require",
           [](RuntimeSession& session, const py::handle success_value,
              const py::handle phase_value, const py::handle message_value) {
             bool success = false;
             std::string phase;
             std::string message;
             const qd::CollectiveLocalError parse_error =
                 qd::capture_collective_local_error([&] {
               success = qd::parse_bool(success_value, "success");
               phase = qd::parse_string(phase_value, "phase");
               message = qd::parse_string(message_value, "message");
             });
             py::gil_scoped_release release;
             session.consensus(parse_error.ok(),
                               "python-consensus-input",
                               parse_error.message());
             session.require_same_string(
                 phase, "python-consensus-phase",
                 "ranks supplied different collective phases");
             session.consensus(success, phase, message);
           },
           py::arg("success"), py::arg("phase"), py::arg("message") = "")
      .def("collective_agree",
           [](RuntimeSession& session, const py::handle value_handle,
              const py::handle phase_handle, const py::handle message_handle) {
             std::string value;
             std::string phase;
             std::string message;
             const qd::CollectiveLocalError parse_error =
                 qd::capture_collective_local_error([&] {
               value = qd::parse_string(value_handle, "collective value");
               phase = qd::parse_string(phase_handle, "phase");
               message = qd::parse_string(message_handle, "message");
             });
             py::gil_scoped_release release;
             session.consensus(parse_error.ok(),
                               "python-agreement-input",
                               parse_error.message());
             session.require_same_string(
                 phase, "python-agreement-phase",
                 "ranks supplied different collective agreement phases");
             session.require_same_string(value, phase, message);
           },
           py::arg("value"), py::arg("phase"),
           py::arg("message") =
               "ranks supplied different collective agreement values")
      .def("configure_owned_devices",
           [](RuntimeSession& session, const py::handle devices) {
             qd::ParsedOwnedDevices parsed = qd::parse_owned_devices(devices);
             {
               py::gil_scoped_release release;
               session.configure_owned_devices(
                   std::move(parsed.devices), parsed.error.message());
             }
             const qd::EndpointMapping mapping = session.endpoint_mapping();
             return qd::endpoint_mapping_to_python(mapping);
           },
           py::arg("devices"))
      .def("configure_devices",
           [](RuntimeSession& session, const py::handle devices) {
             std::vector<int> ordinals;
             const qd::CollectiveLocalError parse_error =
                 qd::capture_collective_local_error([&] {
               ordinals = qd::parse_device_ordinals(devices);
             });
             {
               py::gil_scoped_release release;
               session.configure_devices(
                   std::move(ordinals), parse_error.message());
             }
             return qd::endpoint_mapping_to_python(
                 session.endpoint_mapping());
           },
           py::arg("devices") = "auto")
      .def("select_topology",
           [](RuntimeSession& session, const py::handle global_nx,
              const py::handle global_ny, const py::handle decomposition,
              const py::handle minimum_tile_width) {
             const qd::ParsedTopologyRequest request =
                 qd::parse_topology_request(
                     global_nx, global_ny, decomposition,
                     minimum_tile_width);
             {
               py::gil_scoped_release release;
               session.select_topology(
                   request.global_nx, request.global_ny, request.shape,
                   request.minimum_tile_width, request.error.message());
             }
             const std::optional<qd::VirtualTopology> topology =
                 session.topology();
             if (!topology) {
               throw std::logic_error{
                   "distributed topology selection produced no topology"};
             }
             return qd::topology_to_python(*topology);
           },
           py::arg("global_nx"), py::arg("global_ny"),
           py::arg("decomposition"), py::arg("minimum_tile_width"))
      .def("start_mhd",
           [](RuntimeSession& session, const py::handle config_value,
              const py::handle state, const py::handle background,
              const py::handle transport_value) {
             quasar::mhd::MhdConfig config;
             qd::MhdGlobalState parsed_state;
             std::optional<qd::MhdGlobalBackground> parsed_background;
             qd::TransportPolicy transport = qd::TransportPolicy::automatic;
             const qd::CollectiveLocalError parse_error =
                 qd::capture_collective_local_error([&] {
               config = py::cast<quasar::mhd::MhdConfig>(config_value);
               if (!py::isinstance<py::dict>(state)) {
                 throw std::invalid_argument{
                     "distributed MHD state must be a dictionary"};
               }
               parsed_state = qd::parse_mhd_state(
                   py::reinterpret_borrow<py::dict>(state));
               if (!background.is_none()) {
                 if (!py::isinstance<py::dict>(background)) {
                   throw std::invalid_argument{
                       "distributed MHD background must be a dictionary or None"};
                 }
                 parsed_background = qd::parse_mhd_background(
                     py::reinterpret_borrow<py::dict>(background));
               }
               transport = qd::parse_transport(transport_value);
             });
             py::gil_scoped_release release;
             session.start_mhd(std::move(config), std::move(parsed_state),
                               std::move(parsed_background),
                               transport,
                               parse_error.message());
           },
           py::arg("config"), py::arg("state"),
           py::arg("background") = py::none(),
           py::arg("transport") = "auto")
      .def("restart_mhd",
           [](RuntimeSession& session, const py::handle config_value,
              const py::handle path_value, const py::handle unit_system_value,
              const py::handle expected_background,
              const py::handle transport_value) {
             quasar::mhd::MhdConfig config;
             std::string path;
             std::string unit_system;
             std::optional<qd::MhdGlobalBackground> parsed_background;
             qd::TransportPolicy transport = qd::TransportPolicy::automatic;
             std::string signature;
             const qd::CollectiveLocalError parse_error =
                 qd::capture_collective_local_error([&] {
               config = py::cast<quasar::mhd::MhdConfig>(config_value);
               path = qd::parse_string(path_value, "checkpoint path");
               unit_system = qd::parse_string(
                   unit_system_value, "unit system");
               if (!expected_background.is_none()) {
                 if (!py::isinstance<py::dict>(expected_background)) {
                   throw std::invalid_argument{
                       "distributed MHD restart background must be a dictionary or None"};
                 }
                 parsed_background = qd::parse_mhd_background(
                     py::reinterpret_borrow<py::dict>(expected_background));
               }
               transport = qd::parse_transport(transport_value);
               std::ostringstream output;
               qd::append_argument_signature(output, path);
               qd::append_argument_signature(output, unit_system);
               signature = output.str();
             });
             qd::CheckpointMetadata metadata;
             std::vector<std::vector<std::uint8_t>> diagnostic_state;
             {
               py::gil_scoped_release release;
               session.consensus(parse_error.ok(),
                                 "python-mhd-restart-input",
                                 parse_error.message());
               session.require_same_string(
                   signature, "python-mhd-restart-agreement",
                   "ranks supplied different MHD restart paths or units");
               metadata = session.restart_mhd(
                   std::move(config), path, unit_system, parsed_background,
                   transport, std::string_view{}, diagnostic_state);
             }
             std::optional<py::dict> result;
             const qd::CollectiveLocalError conversion_error =
                 qd::capture_collective_local_error([&] {
               result.emplace(qd::checkpoint_metadata_to_python(metadata));
               (*result)["diagnostic_state"] =
                   qd::diagnostic_state_parts_to_python(diagnostic_state);
             });
             {
               py::gil_scoped_release release;
               session.consensus(
                   conversion_error.ok(),
                   "python-mhd-restart-result", conversion_error.message());
             }
             return std::move(*result);
           },
           py::arg("config"), py::arg("path"), py::arg("unit_system"),
           py::arg("expected_background") = py::none(),
           py::arg("transport") = "auto")
      .def("mhd_cfl_limit", [](RuntimeSession& session) {
        py::gil_scoped_release release;
        return session.mhd_cfl_limit();
      })
      .def("mhd_step",
           [](RuntimeSession& session, const py::handle dt_value,
              const py::handle check_cfl_value) {
             quasar::Real dt = 0;
             bool check_cfl = true;
             std::string signature;
             const qd::CollectiveLocalError parse_error =
                 qd::capture_collective_local_error([&] {
               dt = qd::parse_real(dt_value, "MHD timestep");
               check_cfl = qd::parse_bool(
                   check_cfl_value, "MHD CFL-check flag");
               std::ostringstream output;
               qd::append_argument_bits(output, dt);
               qd::append_argument_bits(output, check_cfl);
               signature = output.str();
             });
             py::gil_scoped_release release;
             session.consensus(parse_error.ok(),
                               "python-mhd-step-input",
                               parse_error.message());
             session.require_same_string(
                 signature, "python-mhd-step-agreement",
                 "ranks supplied different MHD step arguments");
             session.mhd_step(dt, check_cfl);
           },
           py::arg("dt"), py::arg("check_cfl") = true)
      .def("mhd_divergence_b_max", [](RuntimeSession& session) {
        py::gil_scoped_release release;
        return session.mhd_divergence_b_max();
      })
      .def("mhd_gather_state", [](RuntimeSession& session) {
        qd::MhdGlobalState state;
        {
          py::gil_scoped_release release;
          state = session.mhd_gather_state();
        }
        return qd::mhd_state_to_python(state);
      })
      .def("mhd_gather_cell_component",
           [](RuntimeSession& session, const py::handle component_value) {
             std::string component;
             const qd::CollectiveLocalError parse_error =
                 qd::capture_collective_local_error([&] {
               component = qd::parse_string(
                   component_value, "MHD component");
             });
             std::vector<quasar::Real> values;
             {
               py::gil_scoped_release release;
               session.consensus(parse_error.ok(),
                                 "python-mhd-component-input",
                                 parse_error.message());
               session.require_same_string(
                   component, "python-mhd-component-agreement",
                   "ranks requested different MHD components");
               values = session.mhd_gather_cell_component(component);
             }
             return qd::real_vector_to_numpy(values);
           },
           py::arg("component"))
      .def("mhd_local_owned_shards", [](RuntimeSession& session) {
        std::vector<qd::MhdOwnedShard> shards;
        {
          py::gil_scoped_release release;
          shards = session.mhd_local_owned_shards();
        }
        return qd::mhd_owned_shards_to_python(shards);
      })
      .def("mhd_global_cell_sums", [](RuntimeSession& session) {
        qd::MhdGlobalCellSums sums;
        {
          py::gil_scoped_release release;
          sums = session.mhd_global_cell_sums();
        }
        py::dict result;
        result["rho"] = sums.rho;
        result["energy"] = sums.energy;
        return result;
      })
      .def("mhd_write_checkpoint",
           [](RuntimeSession& session, const py::handle path_value,
              const py::handle step_value, const py::handle time_value,
              const py::handle unit_system_value,
              const py::handle diagnostic_state_value) {
             std::string path;
             std::uint64_t step = 0;
             double time = 0;
             std::string unit_system;
             std::vector<std::uint8_t> diagnostic_state;
             std::string signature;
             const qd::CollectiveLocalError parse_error =
                 qd::capture_collective_local_error([&] {
               path = qd::parse_string(path_value, "checkpoint path");
               step = qd::parse_uint64(step_value, "checkpoint step");
               time = qd::parse_real(time_value, "checkpoint time");
               unit_system = qd::parse_string(
                   unit_system_value, "unit system");
               diagnostic_state =
                   qd::parse_checkpoint_diagnostic_fragment(
                       diagnostic_state_value);
               std::ostringstream output;
               qd::append_argument_signature(output, path);
               qd::append_argument_bits(output, step);
               qd::append_argument_bits(output, time);
               qd::append_argument_signature(output, unit_system);
               signature = output.str();
             });
             py::gil_scoped_release release;
             session.consensus(parse_error.ok(),
                               "python-mhd-checkpoint-input",
                               parse_error.message());
             session.require_same_string(
                 signature, "python-mhd-checkpoint-agreement",
                 "ranks supplied different MHD checkpoint arguments");
             session.mhd_write_checkpoint(
                 path, step, time, unit_system, diagnostic_state);
           },
           py::arg("path"), py::arg("step"), py::arg("time"),
           py::arg("unit_system"), py::arg("diagnostic_state") = py::bytes{})
      .def("close_mhd", [](RuntimeSession& session) {
        py::gil_scoped_release release;
        session.close_mhd();
      })
      .def("start_pic",
           [](RuntimeSession& session, const py::handle config_value,
              const py::handle fields, const py::handle external_fields,
              const py::handle species, const py::handle transport_value) {
             quasar::pic::EmPicConfig config;
             qd::PicGlobalFields parsed_fields;
             std::optional<qd::PicGlobalFields> parsed_external;
             std::vector<qd::PicSpeciesState> parsed_species;
             qd::TransportPolicy transport = qd::TransportPolicy::automatic;
             const qd::CollectiveLocalError parse_error =
                 qd::capture_collective_local_error([&] {
               config = py::cast<quasar::pic::EmPicConfig>(config_value);
               if (!py::isinstance<py::dict>(fields)) {
                 throw std::invalid_argument{
                     "distributed PIC fields must be a dictionary"};
               }
               parsed_fields = qd::parse_pic_fields(
                   py::reinterpret_borrow<py::dict>(fields),
                   "distributed PIC fields");
               if (!external_fields.is_none()) {
                 if (!py::isinstance<py::dict>(external_fields)) {
                   throw std::invalid_argument{
                       "distributed PIC external fields must be a dictionary or None"};
                 }
                 parsed_external = qd::parse_pic_fields(
                     py::reinterpret_borrow<py::dict>(external_fields),
                     "distributed PIC external fields");
               }
               parsed_species = qd::parse_pic_species(species);
               transport = qd::parse_transport(transport_value);
             });
             py::gil_scoped_release release;
             session.start_pic(std::move(config), std::move(parsed_fields),
                               std::move(parsed_external),
                               std::move(parsed_species),
                               transport,
                               parse_error.message());
           },
           py::arg("config"), py::arg("fields"),
           py::arg("external_fields") = py::none(),
           py::arg("species") = py::tuple(),
           py::arg("transport") = "auto")
      .def("restart_pic",
           [](RuntimeSession& session, const py::handle config_value,
              const py::handle path_value, const py::handle unit_system_value,
              const py::handle expected_species,
              const py::handle transport_value) {
             quasar::pic::EmPicConfig config;
             std::string path;
             std::string unit_system;
             std::vector<quasar::pic::SpeciesConfig> parsed_species;
             qd::TransportPolicy transport = qd::TransportPolicy::automatic;
             std::string signature;
             const qd::CollectiveLocalError parse_error =
                 qd::capture_collective_local_error([&] {
               config = py::cast<quasar::pic::EmPicConfig>(config_value);
               path = qd::parse_string(path_value, "checkpoint path");
               unit_system = qd::parse_string(
                   unit_system_value, "unit system");
               parsed_species =
                   qd::parse_pic_species_configs(expected_species);
               transport = qd::parse_transport(transport_value);
               std::ostringstream output;
               qd::append_argument_signature(output, path);
               qd::append_argument_signature(output, unit_system);
               signature = output.str();
             });
             qd::CheckpointMetadata metadata;
             std::vector<std::vector<std::uint8_t>> diagnostic_state;
             {
               py::gil_scoped_release release;
               session.consensus(parse_error.ok(),
                                 "python-pic-restart-input",
                                 parse_error.message());
               session.require_same_string(
                   signature, "python-pic-restart-agreement",
                   "ranks supplied different PIC restart paths or units");
               metadata = session.restart_pic(
                   std::move(config), path, unit_system, parsed_species,
                   transport, std::string_view{}, diagnostic_state);
             }
             std::optional<py::dict> result;
             const qd::CollectiveLocalError conversion_error =
                 qd::capture_collective_local_error([&] {
               result.emplace(qd::checkpoint_metadata_to_python(metadata));
               (*result)["diagnostic_state"] =
                   qd::diagnostic_state_parts_to_python(diagnostic_state);
             });
             {
               py::gil_scoped_release release;
               session.consensus(
                   conversion_error.ok(),
                   "python-pic-restart-result", conversion_error.message());
             }
             return std::move(*result);
           },
           py::arg("config"), py::arg("path"), py::arg("unit_system"),
           py::arg("expected_species"), py::arg("transport") = "auto")
      .def("pic_cfl_limit", [](RuntimeSession& session) {
        py::gil_scoped_release release;
        return session.pic_cfl_limit();
      })
      .def("pic_sample_external_fields",
           [](RuntimeSession& session, const py::handle evaluator_value,
              const py::handle source_value,
              const py::handle length_scale_value,
              const py::handle e_field_scale_value,
              const py::handle b_field_scale_value) {
             quasar::numerics::IFieldEvaluator* evaluator = nullptr;
             const quasar::core::IFieldSource* source = nullptr;
             quasar::Real length_scale = 0;
             quasar::Real e_field_scale = 0;
             quasar::Real b_field_scale = 0;
             std::string signature;
             const qd::CollectiveLocalError parse_error =
                 qd::capture_collective_local_error([&] {
               evaluator = &py::cast<quasar::numerics::IFieldEvaluator&>(
                   evaluator_value);
               source = &py::cast<const quasar::core::IFieldSource&>(
                   source_value);
               length_scale = qd::parse_real(
                   length_scale_value, "external-field length scale");
               e_field_scale = qd::parse_real(
                   e_field_scale_value, "external electric-field scale");
               b_field_scale = qd::parse_real(
                   b_field_scale_value, "external magnetic-field scale");
               std::ostringstream output;
               qd::append_argument_bits(output, length_scale);
               qd::append_argument_bits(output, e_field_scale);
               qd::append_argument_bits(output, b_field_scale);
               signature = output.str();
             });
             py::gil_scoped_release release;
             session.consensus(parse_error.ok(),
                               "python-pic-external-input",
                               parse_error.message());
             session.require_same_string(
                 signature, "python-pic-external-agreement",
                 "ranks supplied different external-field scales");
             session.pic_sample_external_fields(
                 *evaluator, *source, length_scale,
                 e_field_scale, b_field_scale);
           },
           py::arg("evaluator"), py::arg("source"),
           py::arg("length_scale") = 1.0,
           py::arg("e_field_scale") = 1.0,
           py::arg("b_field_scale") = 1.0)
      .def("pic_step", [](RuntimeSession& session,
                           const py::handle dt_value) {
        quasar::Real dt = 0;
        std::string signature;
        const qd::CollectiveLocalError parse_error =
            qd::capture_collective_local_error([&] {
          dt = qd::parse_real(dt_value, "PIC timestep");
          std::ostringstream output;
          qd::append_argument_bits(output, dt);
          signature = output.str();
        });
        py::gil_scoped_release release;
        session.consensus(parse_error.ok(),
                          "python-pic-step-input", parse_error.message());
        session.require_same_string(
            signature, "python-pic-step-agreement",
            "ranks supplied different PIC timesteps");
        session.pic_step(dt);
      }, py::arg("dt"))
      .def("pic_gather_state", [](RuntimeSession& session) {
        qd::PicGlobalState state;
        {
          py::gil_scoped_release release;
          state = session.pic_gather_state();
        }
        return qd::pic_state_to_python(state);
      })
      .def("pic_local_owned_shards",
           [](RuntimeSession& session,
              const py::handle include_particles_value) {
             bool include_particles = true;
             const qd::CollectiveLocalError parse_error =
                 qd::capture_collective_local_error([&] {
               include_particles = qd::parse_bool(
                   include_particles_value, "include_particles");
             });
             std::vector<qd::PicOwnedShard> shards;
             {
               py::gil_scoped_release release;
               session.consensus(parse_error.ok(),
                                 "python-pic-shards-input",
                                 parse_error.message());
               session.require_same_string(
                   include_particles ? "1" : "0",
                   "python-pic-shards-agreement",
                   "ranks supplied different PIC shard options");
               shards = session.pic_local_owned_shards(include_particles);
             }
             return qd::pic_owned_shards_to_python(shards);
           },
           py::arg("include_particles") = true)
      .def("pic_alive_counts", [](RuntimeSession& session) {
        std::vector<std::uint64_t> counts;
        {
          py::gil_scoped_release release;
          counts = session.pic_alive_counts();
        }
        return qd::integral_vector_to_numpy(counts);
      })
      .def("pic_kinetic_energies", [](RuntimeSession& session) {
        std::vector<quasar::Real> energies;
        {
          py::gil_scoped_release release;
          energies = session.pic_kinetic_energies();
        }
        return qd::real_vector_to_numpy(energies);
      })
      .def("pic_total_em_energy", [](RuntimeSession& session) {
        py::gil_scoped_release release;
        return session.pic_total_em_energy();
      })
      .def("pic_gauss_residual", [](RuntimeSession& session) {
        py::gil_scoped_release release;
        return session.pic_gauss_residual();
      })
      .def("pic_write_checkpoint",
           [](RuntimeSession& session, const py::handle path_value,
              const py::handle step_value, const py::handle time_value,
              const py::handle unit_system_value,
              const py::handle diagnostic_state_value) {
             std::string path;
             std::uint64_t step = 0;
             double time = 0;
             std::string unit_system;
             std::vector<std::uint8_t> diagnostic_state;
             std::string signature;
             const qd::CollectiveLocalError parse_error =
                 qd::capture_collective_local_error([&] {
               path = qd::parse_string(path_value, "checkpoint path");
               step = qd::parse_uint64(step_value, "checkpoint step");
               time = qd::parse_real(time_value, "checkpoint time");
               unit_system = qd::parse_string(
                   unit_system_value, "unit system");
               diagnostic_state =
                   qd::parse_checkpoint_diagnostic_fragment(
                       diagnostic_state_value);
               std::ostringstream output;
               qd::append_argument_signature(output, path);
               qd::append_argument_bits(output, step);
               qd::append_argument_bits(output, time);
               qd::append_argument_signature(output, unit_system);
               signature = output.str();
             });
             py::gil_scoped_release release;
             session.consensus(parse_error.ok(),
                               "python-pic-checkpoint-input",
                               parse_error.message());
             session.require_same_string(
                 signature, "python-pic-checkpoint-agreement",
                 "ranks supplied different PIC checkpoint arguments");
             session.pic_write_checkpoint(
                 path, step, time, unit_system, diagnostic_state);
           },
           py::arg("path"), py::arg("step"), py::arg("time"),
           py::arg("unit_system"), py::arg("diagnostic_state") = py::bytes{})
      .def("close_pic", [](RuntimeSession& session) {
        py::gil_scoped_release release;
        session.close_pic();
      })
      .def("close", [](RuntimeSession& session) {
        py::gil_scoped_release release;
        session.close();
      })
      .def("__enter__", [](RuntimeSession& session) -> RuntimeSession& {
        return session;
      }, py::return_value_policy::reference_internal)
      .def("__exit__", [](RuntimeSession& session, const py::object&,
                           const py::object&, const py::object&) {
        py::gil_scoped_release release;
        session.close();
        return false;
      });
}
