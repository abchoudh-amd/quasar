#include "quasar/distributed/diagnostics.hpp"

#include "collective_helpers.hpp"
#include "test_hooks.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <new>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <unistd.h>

namespace quasar::distributed {
namespace {

constexpr CollectiveStringBroadcastMessages diagnostics_string_broadcast{
    .preparation_failure = "diagnostics string preparation failed",
    .range_failure = "diagnostics string exceeds MPI count range",
    .allocation_failure = "diagnostics string allocation failed",
    .length_operation = "MPI_Bcast(diagnostics string length)",
    .bytes_operation = "MPI_Bcast(diagnostics string bytes)",
};

bool is_valid_utf8(std::string_view value) {
  const auto continuation = [](unsigned char byte) {
    return byte >= 0x80 && byte <= 0xBF;
  };
  for (std::size_t index = 0; index < value.size();) {
    const auto first = static_cast<unsigned char>(value[index]);
    if (first <= 0x7F) {
      ++index;
      continue;
    }
    if (first >= 0xC2 && first <= 0xDF) {
      if (index + 1 >= value.size()
          || !continuation(static_cast<unsigned char>(value[index + 1]))) {
        return false;
      }
      index += 2;
      continue;
    }
    if (first >= 0xE0 && first <= 0xEF) {
      if (index + 2 >= value.size()) return false;
      const auto second = static_cast<unsigned char>(value[index + 1]);
      const auto third = static_cast<unsigned char>(value[index + 2]);
      if (!continuation(third)
          || (first == 0xE0 && (second < 0xA0 || second > 0xBF))
          || (first == 0xED && (second < 0x80 || second > 0x9F))
          || ((first != 0xE0 && first != 0xED) && !continuation(second))) {
        return false;
      }
      index += 3;
      continue;
    }
    if (first >= 0xF0 && first <= 0xF4) {
      if (index + 3 >= value.size()) return false;
      const auto second = static_cast<unsigned char>(value[index + 1]);
      const auto third = static_cast<unsigned char>(value[index + 2]);
      const auto fourth = static_cast<unsigned char>(value[index + 3]);
      if (!continuation(third) || !continuation(fourth)
          || (first == 0xF0 && (second < 0x90 || second > 0xBF))
          || (first == 0xF4 && (second < 0x80 || second > 0x8F))
          || ((first != 0xF0 && first != 0xF4) && !continuation(second))) {
        return false;
      }
      index += 4;
      continue;
    }
    return false;
  }
  return true;
}

std::string escape_json(std::string_view value) {
  std::ostringstream output;
  for (const unsigned char character : value) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(character) << std::dec;
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  return output.str();
}

void append_field(std::ostringstream& output, std::string_view value) {
  output << value.size() << ':' << value;
}

std::string serialize_shard(const DiagnosticShard& shard) {
  std::ostringstream output;
  output << shard.rank << ' ' << shard.node_rank << ' '
         << shard.local_device << ' ' << shard.endpoint << ' '
         << shard.tile_x << ' ' << shard.tile_y << ' '
         << shard.offset_x << ' ' << shard.offset_y << ' '
         << shard.owned_nx << ' ' << shard.owned_ny << ' ';
  append_field(output, shard.device_identity);
  append_field(output, shard.path.generic_string());
  return output.str();
}

std::string serialize_shards(std::span<const DiagnosticShard> shards) {
  std::ostringstream output;
  output << shards.size() << ';';
  for (const auto& shard : shards) {
    const std::string encoded = serialize_shard(shard);
    output << encoded.size() << ':' << encoded;
  }
  return output.str();
}

std::string read_length_prefixed(std::string_view text, std::size_t& cursor) {
  const std::size_t separator = text.find(':', cursor);
  if (separator == std::string_view::npos) {
    throw std::runtime_error{"invalid diagnostics shard record"};
  }
  const std::string length_text{text.substr(cursor, separator - cursor)};
  std::size_t consumed = 0;
  const unsigned long long length = std::stoull(length_text, &consumed);
  if (consumed != length_text.size()
      || length > text.size() - separator - 1) {
    throw std::runtime_error{"invalid diagnostics shard record length"};
  }
  cursor = separator + 1;
  std::string result{text.substr(cursor, static_cast<std::size_t>(length))};
  cursor += static_cast<std::size_t>(length);
  return result;
}

DiagnosticShard parse_shard(std::string_view text) {
  DiagnosticShard shard;
  std::istringstream input{std::string{text}};
  if (!(input >> shard.rank >> shard.node_rank
        >> shard.local_device >> shard.endpoint
        >> shard.tile_x >> shard.tile_y
        >> shard.offset_x >> shard.offset_y
        >> shard.owned_nx >> shard.owned_ny)) {
    throw std::runtime_error{"invalid diagnostics shard numeric fields"};
  }
  const auto position = input.tellg();
  if (position < 0) {
    throw std::runtime_error{"invalid diagnostics shard record"};
  }
  std::size_t cursor = static_cast<std::size_t>(position);
  while (cursor < text.size() && text[cursor] == ' ') ++cursor;
  shard.device_identity = read_length_prefixed(text, cursor);
  shard.path = read_length_prefixed(text, cursor);
  if (cursor != text.size()) {
    throw std::runtime_error{"diagnostics shard record has trailing data"};
  }
  return shard;
}

std::vector<DiagnosticShard> parse_shards(std::string_view text) {
  const std::size_t separator = text.find(';');
  if (separator == std::string_view::npos) {
    throw std::runtime_error{"invalid diagnostics shard list"};
  }
  const std::size_t expected = static_cast<std::size_t>(
      std::stoull(std::string{text.substr(0, separator)}));
  std::size_t cursor = separator + 1;
  // Each entry needs at least a one-byte length and ':'.  Bound the decoded
  // count by the payload before reserve() so malformed peer metadata cannot
  // request an arbitrary allocation.
  if (expected > (text.size() - cursor) / 2) {
    throw std::runtime_error{"diagnostics shard list count exceeds payload"};
  }
  std::vector<DiagnosticShard> result;
  result.reserve(expected);
  while (cursor < text.size()) {
    result.push_back(parse_shard(read_length_prefixed(text, cursor)));
  }
  if (result.size() != expected) {
    throw std::runtime_error{"diagnostics shard list count mismatch"};
  }
  return result;
}

bool rectangles_overlap(const DiagnosticShard& left,
                        const DiagnosticShard& right) {
  return left.offset_x < right.offset_x + right.owned_nx
      && right.offset_x < left.offset_x + left.owned_nx
      && left.offset_y < right.offset_y + right.owned_ny
      && right.offset_y < left.offset_y + left.owned_ny;
}

std::string common_descriptor(const ShardedDiagnosticsManifest& manifest) {
  std::ostringstream output;
  output << manifest.schema << '\n' << manifest.physics << '\n'
         << manifest.geometry << '\n' << manifest.global_nx << '\n'
         << manifest.global_ny << '\n' << manifest.step << '\n'
         << std::hexfloat << manifest.time << '\n'
         << manifest.px << '\n' << manifest.py;
  return output.str();
}

class ExclusiveTemporaryFile {
 public:
  ExclusiveTemporaryFile() = default;
  ~ExclusiveTemporaryFile() noexcept { discard(); }
  ExclusiveTemporaryFile(const ExclusiveTemporaryFile&) = delete;
  ExclusiveTemporaryFile& operator=(const ExclusiveTemporaryFile&) = delete;

  void create(const std::filesystem::path& final_path) {
    if (fd_ >= 0 || !path_.empty()) {
      throw std::logic_error{"diagnostics temporary file already exists"};
    }
    std::string pattern = final_path.string() + ".tmp.XXXXXX";
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    fd_ = ::mkstemp(writable.data());
    if (fd_ < 0) {
      throw std::system_error{
          errno, std::generic_category(),
          "failed to create diagnostics manifest temporary file"};
    }
    path_ = writable.data();
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

  void write_document(std::string_view document) {
    std::size_t offset = 0;
    while (offset < document.size()) {
      const std::size_t remaining = document.size() - offset;
      const std::size_t chunk = std::min(
          remaining,
          static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
      const ssize_t written = ::write(fd_, document.data() + offset, chunk);
      if (written < 0) {
        if (errno == EINTR) continue;
        throw std::system_error{
            errno, std::generic_category(),
            "failed to write diagnostics manifest temporary file"};
      }
      if (written == 0) {
        throw std::runtime_error{
            "diagnostics manifest temporary write made no progress"};
      }
      offset += static_cast<std::size_t>(written);
    }
    if (::fsync(fd_) != 0) {
      throw std::system_error{
          errno, std::generic_category(),
          "failed to synchronize diagnostics manifest temporary file"};
    }
    if (::close(fd_) != 0) {
      fd_ = -1;
      throw std::system_error{
          errno, std::generic_category(),
          "failed to close diagnostics manifest temporary file"};
    }
    fd_ = -1;
  }

  [[nodiscard]] int publish(const std::filesystem::path& final_path) noexcept {
    errno = 0;
    if (std::rename(path_.c_str(), final_path.c_str()) != 0) {
      return errno == 0 ? EIO : errno;
    }
    path_.clear();
    return 0;
  }

  void discard() noexcept {
    if (fd_ >= 0) {
      (void)::close(fd_);
      fd_ = -1;
    }
    if (!path_.empty()) {
      (void)std::remove(path_.c_str());
      path_.clear();
    }
  }

 private:
  std::filesystem::path path_{};
  int fd_{-1};
};

std::filesystem::path normalized_on_disk_path(
    const std::filesystem::path& path) {
  std::error_code error;
  std::filesystem::path absolute = std::filesystem::absolute(path, error);
  if (error || absolute.empty()) {
    throw std::invalid_argument{
        "diagnostics path cannot be normalized to an absolute path"};
  }
  return absolute.lexically_normal();
}

std::filesystem::path resolved_shard_path(
    const std::filesystem::path& manifest_path,
    const std::filesystem::path& shard_path) {
  return normalized_on_disk_path(
      shard_path.is_absolute()
          ? shard_path
          : manifest_path.parent_path() / shard_path);
}

void validate_published_paths(
    const std::filesystem::path& manifest_path,
    const std::filesystem::path& temporary_path,
    std::span<const DiagnosticShard> shards) {
  const auto normalized_manifest = normalized_on_disk_path(manifest_path);
  const auto normalized_temporary = normalized_on_disk_path(temporary_path);
  if (normalized_manifest == normalized_temporary) {
    throw std::invalid_argument{
        "diagnostics manifest and temporary paths collide"};
  }
  std::set<std::filesystem::path> normalized_shards;
  for (const auto& shard : shards) {
    const auto normalized = resolved_shard_path(manifest_path, shard.path);
    if (normalized == normalized_manifest || normalized == normalized_temporary) {
      throw std::invalid_argument{
          "diagnostics shard path collides with manifest publication paths"};
    }
    if (!normalized_shards.insert(normalized).second) {
      throw std::invalid_argument{
          "diagnostics shard paths are not unique after normalization"};
    }
  }
}

void prepare_manifest_path(MpiRuntime& runtime, std::uint64_t& epoch,
                           const std::filesystem::path& manifest_path) {
  std::string native_path;
  collective_try(
      runtime, epoch, "diagnostics-manifest-path",
      "diagnostics manifest path preparation failed", -1, [&] {
        const std::string generic_path = manifest_path.generic_string();
        native_path = manifest_path.string();
        if (manifest_path.empty()
            || generic_path.find('\0') != std::string::npos
            || !is_valid_utf8(generic_path)) {
          throw std::invalid_argument{
              "diagnostics manifest path must be non-empty UTF-8"};
        }
      });
  collective_require_common_string(
      runtime, epoch, native_path, "diagnostics-manifest-path",
      diagnostics_string_broadcast,
      "ranks supplied different diagnostics manifest paths");
}

std::filesystem::path prepare_temporary_manifest(
    MpiRuntime& runtime, std::uint64_t& epoch,
    const std::filesystem::path& manifest_path,
    ExclusiveTemporaryFile& temporary_file) {
  std::string local_temporary;
  collective_try_on_root(
      runtime, epoch, "diagnostics-manifest-temporary-path",
      "diagnostics temporary-path preparation failed", -1, [&] {
        temporary_file.create(manifest_path);
        local_temporary = temporary_file.path().string();
      });
  const std::string root_temporary = collective_broadcast_string(
      runtime, epoch, local_temporary,
      "diagnostics-manifest-temporary-path", diagnostics_string_broadcast);
  std::filesystem::path temporary;
  collective_try(
      runtime, epoch, "diagnostics-manifest-temporary-path",
      "diagnostics temporary path allocation failed", -1,
      [&] { temporary = std::filesystem::path{root_temporary}; });
  return temporary;
}

void validate_local_shards_complete(
    int rank, const std::filesystem::path& manifest_path,
    const std::filesystem::path& temporary_path,
    std::span<const DiagnosticShard> local_shards) {
  validate_published_paths(manifest_path, temporary_path, local_shards);
  for (const auto& shard : local_shards) {
    if (shard.rank != rank) {
      throw std::invalid_argument{
          "local diagnostics shard carries a different MPI rank"};
    }
    const std::filesystem::path on_disk_path =
        resolved_shard_path(manifest_path, shard.path);
    std::error_code error;
    if (!std::filesystem::is_regular_file(on_disk_path, error) || error) {
      throw std::runtime_error{
          "a local diagnostics shard is missing or incomplete"};
    }
  }
}

void require_local_shards_complete(
    MpiRuntime& runtime, std::uint64_t& epoch,
    const std::filesystem::path& manifest_path,
    const std::filesystem::path& temporary_path,
    std::span<const DiagnosticShard> local_shards) {
  collective_try(
      runtime, epoch, "diagnostics-shards-complete",
      "diagnostics shard path validation failed", -1, [&] {
        validate_local_shards_complete(runtime.rank(), manifest_path,
                                       temporary_path, local_shards);
      });
}

void require_common_manifest_descriptor(
    MpiRuntime& runtime, std::uint64_t& epoch,
    const ShardedDiagnosticsManifest& manifest) {
  std::string descriptor;
  collective_try(
      runtime, epoch, "diagnostics-manifest-metadata",
      "diagnostics manifest descriptor preparation failed", -1,
      [&] { descriptor = common_descriptor(manifest); });
  collective_require_common_string(
      runtime, epoch, descriptor, "diagnostics-manifest-metadata",
      diagnostics_string_broadcast,
      "ranks supplied different diagnostics manifest metadata");
}

struct GatheredShardMetadata {
  std::vector<int> sizes;
  std::vector<int> offsets;
  std::vector<char> bytes;
};

void prepare_root_gather_layout(GatheredShardMetadata& gathered) {
  gathered.offsets.resize(gathered.sizes.size());
  std::int64_t total = 0;
  for (std::size_t index = 0; index < gathered.sizes.size(); ++index) {
    if (gathered.sizes[index] < 0
        || total > std::numeric_limits<int>::max()
                       - gathered.sizes[index]) {
      throw std::length_error{"global diagnostics metadata is too large"};
    }
    gathered.offsets[index] = static_cast<int>(total);
    total += gathered.sizes[index];
  }
  gathered.bytes.resize(static_cast<std::size_t>(total));
}

GatheredShardMetadata gather_shard_metadata(
    MpiRuntime& runtime, std::uint64_t& epoch,
    std::span<const DiagnosticShard> local_shards) {
  std::string local;
  collective_try(
      runtime, epoch, "diagnostics-manifest-gather",
      "diagnostics shard serialization failed", -1, [&] {
        local = serialize_shards(local_shards);
        if (local.size()
            > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
          throw std::length_error{
              "one rank's diagnostics metadata exceeds MPI count range"};
        }
      });
  const int local_size = static_cast<int>(local.size());
  GatheredShardMetadata gathered;
  collective_try_with_fallback(
      runtime, epoch, "diagnostics-manifest-gather",
      "diagnostics size storage allocation failed", -1, [&] {
        if (runtime.rank() == 0) {
          gathered.sizes.resize(static_cast<std::size_t>(runtime.size()));
        }
      });
  int size_dummy{};
  check_mpi(MPI_Gather(
                &local_size, 1, MPI_INT,
                gathered.sizes.empty() ? &size_dummy : gathered.sizes.data(),
                1, MPI_INT, 0,
                detail::MpiRuntimeNativeAccess::world(runtime)),
            "MPI_Gather(diagnostics metadata sizes)");
  collective_try_with_fallback(
      runtime, epoch, "diagnostics-manifest-gather",
      "global diagnostics metadata is too large or cannot be allocated", -1,
      [&] {
        if (runtime.rank() == 0) prepare_root_gather_layout(gathered);
      });
  char byte_dummy{};
  int offset_dummy{};
  check_mpi(MPI_Gatherv(
                local.data(), local_size, MPI_CHAR,
                gathered.bytes.empty() ? &byte_dummy : gathered.bytes.data(),
                gathered.sizes.empty() ? &size_dummy : gathered.sizes.data(),
                gathered.offsets.empty() ? &offset_dummy
                                         : gathered.offsets.data(),
                MPI_CHAR, 0,
                detail::MpiRuntimeNativeAccess::world(runtime)),
            "MPI_Gatherv(diagnostics metadata)");
  return gathered;
}

std::string assemble_manifest_document(
    ShardedDiagnosticsManifest manifest,
    const std::filesystem::path& manifest_path,
    const std::filesystem::path& temporary_path,
    const GatheredShardMetadata& gathered) {
  manifest.shards.clear();
  for (std::size_t rank = 0; rank < gathered.sizes.size(); ++rank) {
    const std::string_view encoded{
        gathered.bytes.data() + gathered.offsets[rank],
        static_cast<std::size_t>(gathered.sizes[rank])};
    auto rank_shards = parse_shards(encoded);
    manifest.shards.insert(manifest.shards.end(),
                           std::make_move_iterator(rank_shards.begin()),
                           std::make_move_iterator(rank_shards.end()));
  }
  std::sort(manifest.shards.begin(), manifest.shards.end(),
            [](const auto& left, const auto& right) {
              return left.endpoint < right.endpoint;
            });
  validate_published_paths(manifest_path, temporary_path, manifest.shards);
  return diagnostics_manifest_json(manifest);
}

std::string prepare_manifest_document(
    MpiRuntime& runtime, std::uint64_t& epoch,
    ShardedDiagnosticsManifest manifest,
    const std::filesystem::path& manifest_path,
    const std::filesystem::path& temporary_path,
    const GatheredShardMetadata& gathered) {
  std::string document;
  collective_try_on_root(
      runtime, epoch, "diagnostics-manifest-validate",
      "diagnostics manifest preparation failed", -1, [&] {
        if (distributed_test_failure_enabled(
                "QUASAR_TEST_DIAGNOSTICS_MANIFEST_PREPARATION_FAILURE")) {
          throw std::bad_alloc{};
        }
        document = assemble_manifest_document(
            std::move(manifest), manifest_path, temporary_path, gathered);
      });
  return document;
}

void write_temporary_manifest(MpiRuntime& runtime, std::uint64_t& epoch,
                              ExclusiveTemporaryFile& temporary_file,
                              std::string_view document) {
  collective_try_on_root_with_failure(
      runtime, epoch, "diagnostics-manifest-write",
      "diagnostics manifest temporary write failed", -1, [&] {
        if (distributed_test_failure_enabled(
                "QUASAR_TEST_DIAGNOSTICS_PUBLICATION_FAILURE")) {
          throw std::bad_alloc{};
        }
        temporary_file.write_document(document);
      },
      [&temporary_file]() noexcept { temporary_file.discard(); });
}

void publish_manifest(MpiRuntime& runtime,
                      ExclusiveTemporaryFile& temporary_file,
                      const std::filesystem::path& manifest_path) {
  const RootPublicationNotification notification = publish_from_root(
      runtime, [&temporary_file, &manifest_path]() noexcept {
        const int error = temporary_file.publish(manifest_path);
        if (error != 0) temporary_file.discard();
        return error;
      });
  check_mpi(notification.mpi_status,
            "MPI_Bcast(diagnostics publication status)");
  if (notification.publication_error != 0) {
    throw std::system_error{
        notification.publication_error, std::generic_category(),
        "failed to atomically publish diagnostics manifest"};
  }
}

}  // namespace

void validate_diagnostics_manifest(
    const ShardedDiagnosticsManifest& manifest) {
  if (manifest.schema != diagnostics_shard_schema) {
    throw std::invalid_argument{
        "diagnostics manifest schema must be quasar-diagnostics-shards/v1"};
  }
  if (manifest.physics.empty()) {
    throw std::invalid_argument{
        "diagnostics manifest physics must be a non-empty string"};
  }
  if (manifest.geometry.empty() || manifest.global_nx == 0
      || manifest.global_ny == 0 || manifest.px == 0 || manifest.py == 0
      || !(std::isfinite(manifest.time) && manifest.time >= 0.0)) {
    throw std::invalid_argument{"diagnostics manifest metadata is incomplete"};
  }
  if (!is_valid_utf8(manifest.schema) || !is_valid_utf8(manifest.physics)
      || !is_valid_utf8(manifest.geometry)) {
    throw std::invalid_argument{
        "diagnostics manifest text must be valid UTF-8"};
  }
  if (manifest.px > std::numeric_limits<std::uint64_t>::max() / manifest.py
      || manifest.shards.size() != manifest.px * manifest.py) {
    throw std::invalid_argument{
        "diagnostics shard count does not match the decomposition"};
  }
  std::set<std::uint64_t> endpoints;
  std::set<std::pair<std::uint64_t, std::uint64_t>> tiles;
  std::set<std::string> paths;
  std::uint64_t cells = 0;
  for (std::size_t index = 0; index < manifest.shards.size(); ++index) {
    const auto& shard = manifest.shards[index];
    if (shard.rank < 0 || shard.node_rank < 0
        || shard.device_identity.empty() || shard.path.empty()
        || shard.owned_nx == 0 || shard.owned_ny == 0
        || shard.tile_x >= manifest.px || shard.tile_y >= manifest.py
        || shard.offset_x > manifest.global_nx
        || shard.owned_nx > manifest.global_nx - shard.offset_x
        || shard.offset_y > manifest.global_ny
        || shard.owned_ny > manifest.global_ny - shard.offset_y) {
      throw std::invalid_argument{"diagnostics shard metadata is invalid"};
    }
    if (!endpoints.insert(shard.endpoint).second
        || !tiles.emplace(shard.tile_x, shard.tile_y).second) {
      throw std::invalid_argument{
          "diagnostics manifest has duplicate endpoints or tile coordinates"};
    }
    const std::string raw_path = shard.path.generic_string();
    const std::string normalized_path =
        shard.path.lexically_normal().generic_string();
    if (normalized_path.empty() || normalized_path == "."
        || normalized_path.find('\0') != std::string::npos
        || !is_valid_utf8(raw_path) || !is_valid_utf8(shard.device_identity)
        || !paths.insert(normalized_path).second) {
      throw std::invalid_argument{
          "diagnostics shard paths must be unique normalized UTF-8 paths"};
    }
    if (shard.owned_nx > std::numeric_limits<std::uint64_t>::max()
                              / shard.owned_ny
        || cells > std::numeric_limits<std::uint64_t>::max()
                       - shard.owned_nx * shard.owned_ny) {
      throw std::overflow_error{"diagnostics owned-cell count overflows"};
    }
    cells += shard.owned_nx * shard.owned_ny;
    for (std::size_t prior = 0; prior < index; ++prior) {
      if (rectangles_overlap(manifest.shards[prior], shard)) {
        throw std::invalid_argument{"diagnostics owned extents overlap"};
      }
    }
  }
  if (manifest.global_nx > std::numeric_limits<std::uint64_t>::max()
                               / manifest.global_ny
      || cells != manifest.global_nx * manifest.global_ny) {
    throw std::invalid_argument{
        "diagnostics owned extents do not cover the global mesh"};
  }
  for (std::uint64_t endpoint = 0; endpoint < manifest.shards.size();
       ++endpoint) {
    if (!endpoints.contains(endpoint)) {
      throw std::invalid_argument{
          "diagnostics endpoints must be contiguous from zero"};
    }
  }
}

std::string diagnostics_manifest_json(
    const ShardedDiagnosticsManifest& manifest) {
  validate_diagnostics_manifest(manifest);
  std::ostringstream output;
  output << "{\n"
         << "  \"schema\": \"" << escape_json(manifest.schema) << "\",\n"
         << "  \"physics\": \"" << escape_json(manifest.physics) << "\",\n"
         << "  \"geometry\": \"" << escape_json(manifest.geometry) << "\",\n"
         << "  \"global_shape\": [" << manifest.global_ny << ", "
         << manifest.global_nx << "],\n"
         << "  \"step\": " << manifest.step << ",\n"
         << "  \"time\": "
         << std::setprecision(std::numeric_limits<double>::max_digits10)
         << manifest.time << ",\n"
         << "  \"decomposition\": {\"px\": " << manifest.px
         << ", \"py\": " << manifest.py << "},\n"
         << "  \"shards\": [\n";
  for (std::size_t index = 0; index < manifest.shards.size(); ++index) {
    const auto& shard = manifest.shards[index];
    output << "    {\"rank\": " << shard.rank
           << ", \"node_rank\": " << shard.node_rank
           << ", \"local_device\": " << shard.local_device
           << ", \"endpoint\": " << shard.endpoint
           << ", \"device_identity\": \""
           << escape_json(shard.device_identity) << "\""
           << ", \"tile\": [" << shard.tile_x << ", " << shard.tile_y << ']'
           << ", \"offset\": [" << shard.offset_y << ", "
           << shard.offset_x << ']'
           << ", \"owned_shape\": [" << shard.owned_ny << ", "
           << shard.owned_nx << ']'
           << ", \"path\": \"" << escape_json(shard.path.generic_string())
           << "\"}" << (index + 1 == manifest.shards.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  return output.str();
}

void publish_diagnostics_manifest(
    MpiRuntime& runtime, std::filesystem::path manifest_path,
    ShardedDiagnosticsManifest common,
    std::span<const DiagnosticShard> local_shards) {
  runtime.require_orchestration_thread();
  std::uint64_t epoch = 0;
  prepare_manifest_path(runtime, epoch, manifest_path);
  ExclusiveTemporaryFile temporary_file;
  const std::filesystem::path temporary = prepare_temporary_manifest(
      runtime, epoch, manifest_path, temporary_file);
  require_local_shards_complete(runtime, epoch, manifest_path, temporary,
                                local_shards);
  require_common_manifest_descriptor(runtime, epoch, common);
  const GatheredShardMetadata gathered =
      gather_shard_metadata(runtime, epoch, local_shards);
  const std::string document = prepare_manifest_document(
      runtime, epoch, std::move(common), manifest_path, temporary, gathered);
  write_temporary_manifest(runtime, epoch, temporary_file, document);

  // Every rank has agreed that the complete temporary document may become
  // the completion marker before this irreversible rename.
  publish_manifest(runtime, temporary_file, manifest_path);
}

}  // namespace quasar::distributed
