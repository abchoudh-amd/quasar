#include "quasar/distributed/topology.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace quasar::distributed {
namespace {

std::size_t checked_product(std::size_t a, std::size_t b,
                            const char* description) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::invalid_argument{description};
  }
  return a * b;
}

IndexRange partition_range(std::size_t global_size,
                           std::size_t partitions,
                           std::size_t coordinate) {
  const std::size_t base      = global_size / partitions;
  const std::size_t remainder = global_size % partitions;
  const std::size_t begin = coordinate * base
                          + std::min(coordinate, remainder);
  const std::size_t size = base + (coordinate < remainder ? 1u : 0u);
  return {begin, begin + size};
}

std::size_t owner_coordinate(std::size_t index,
                             std::size_t global_size,
                             std::size_t partitions) {
  const std::size_t base      = global_size / partitions;
  const std::size_t remainder = global_size % partitions;
  const std::size_t large_region = (base + 1) * remainder;
  if (index < large_region) {
    return index / (base + 1);
  }
  return remainder + (index - large_region) / base;
}

void validate_dimensions(std::size_t global_nx,
                         std::size_t global_ny,
                         std::size_t endpoint_count) {
  if (global_nx == 0 || global_ny == 0) {
    throw std::invalid_argument{
        "distributed topology requires a non-empty global mesh"};
  }
  if (endpoint_count == 0) {
    throw std::invalid_argument{
        "distributed topology requires at least one GPU endpoint"};
  }
}

bool valid_shape(std::size_t global_nx,
                 std::size_t global_ny,
                 DecompositionShape shape,
                 std::size_t minimum_tile_width) {
  if (shape.px == 0 || shape.py == 0
      || shape.px > global_nx || shape.py > global_ny) {
    return false;
  }
  const std::size_t required = std::max<std::size_t>(1, minimum_tile_width);
  return global_nx / shape.px >= required
      && global_ny / shape.py >= required;
}

std::string invalid_shape_message(std::size_t global_nx,
                                  std::size_t global_ny,
                                  DecompositionShape shape,
                                  std::size_t minimum_tile_width) {
  std::ostringstream message;
  message << "decomposition " << shape.px << 'x' << shape.py
          << " creates a tile thinner than the required stencil reach "
          << minimum_tile_width << " on global mesh "
          << global_nx << 'x' << global_ny;
  return message.str();
}

}  // namespace

VirtualTopology VirtualTopology::create_auto(std::size_t global_nx,
                                             std::size_t global_ny,
                                             std::size_t endpoint_count,
                                             std::size_t minimum_tile_width) {
  validate_dimensions(global_nx, global_ny, endpoint_count);

  struct Candidate {
    DecompositionShape shape{};
    long double perimeter{0};
    long double aspect{0};
  };
  std::optional<Candidate> best;

  for (std::size_t px = 1; px <= endpoint_count / px; ++px) {
    if (endpoint_count % px != 0) continue;
    const std::size_t paired = endpoint_count / px;
    for (const DecompositionShape shape :
         {DecompositionShape{px, paired},
          DecompositionShape{paired, px}}) {
      if (!valid_shape(global_nx, global_ny, shape, minimum_tile_width)) {
        continue;
      }

      const long double tile_x = static_cast<long double>(global_nx) / shape.px;
      const long double tile_y = static_cast<long double>(global_ny) / shape.py;
      const Candidate candidate{
          shape,
          2.0L * (tile_x + tile_y),
          std::max(tile_x, tile_y) / std::min(tile_x, tile_y)};

      const auto key = [](const Candidate& value) {
        return std::tuple{value.perimeter, value.aspect,
                          value.shape.px, value.shape.py};
      };
      if (!best || key(candidate) < key(*best)) {
        best = candidate;
      }
    }
  }

  if (!best) {
    std::ostringstream message;
    message << "no factorization of " << endpoint_count
            << " endpoints can cover global mesh " << global_nx << 'x'
            << global_ny << " with minimum tile width "
            << minimum_tile_width;
    throw std::invalid_argument{message.str()};
  }
  return VirtualTopology{global_nx, global_ny, best->shape,
                         minimum_tile_width};
}

VirtualTopology VirtualTopology::create(std::size_t global_nx,
                                        std::size_t global_ny,
                                        std::size_t endpoint_count,
                                        DecompositionShape shape,
                                        std::size_t minimum_tile_width) {
  validate_dimensions(global_nx, global_ny, endpoint_count);
  if (shape.px == 0 || shape.py == 0
      || checked_product(shape.px, shape.py,
                         "decomposition dimensions overflow")
             != endpoint_count) {
    std::ostringstream message;
    message << "decomposition " << shape.px << 'x' << shape.py
            << " does not match " << endpoint_count << " GPU endpoints";
    throw std::invalid_argument{message.str()};
  }
  if (!valid_shape(global_nx, global_ny, shape, minimum_tile_width)) {
    throw std::invalid_argument{invalid_shape_message(
        global_nx, global_ny, shape, minimum_tile_width)};
  }
  return VirtualTopology{global_nx, global_ny, shape, minimum_tile_width};
}

VirtualTopology::VirtualTopology(std::size_t global_nx,
                                 std::size_t global_ny,
                                 DecompositionShape shape,
                                 std::size_t minimum_tile_width)
  : global_nx_{global_nx},
    global_ny_{global_ny},
    shape_{shape},
    minimum_tile_width_{minimum_tile_width} {
  tiles_.reserve(shape_.size());
  for (std::size_t y = 0; y < shape_.py; ++y) {
    for (std::size_t x = 0; x < shape_.px; ++x) {
      const std::size_t endpoint = y * shape_.px + x;
      tiles_.push_back(TileExtent{
          endpoint,
          {x, y},
          partition_range(global_nx_, shape_.px, x),
          partition_range(global_ny_, shape_.py, y)});
    }
  }
}

const TileExtent& VirtualTopology::tile(std::size_t endpoint) const {
  if (endpoint >= tiles_.size()) {
    throw std::out_of_range{"GPU endpoint is outside the virtual topology"};
  }
  return tiles_[endpoint];
}

std::size_t VirtualTopology::endpoint_at(TileCoordinate coordinate) const {
  if (coordinate.x >= shape_.px || coordinate.y >= shape_.py) {
    throw std::out_of_range{"tile coordinate is outside the virtual topology"};
  }
  return coordinate.y * shape_.px + coordinate.x;
}

std::size_t VirtualTopology::owner_of_cell(std::size_t global_x,
                                           std::size_t global_y) const {
  if (global_x >= global_nx_ || global_y >= global_ny_) {
    throw std::out_of_range{"global cell is outside the simulation mesh"};
  }
  const TileCoordinate coordinate{
      owner_coordinate(global_x, global_nx_, shape_.px),
      owner_coordinate(global_y, global_ny_, shape_.py)};
  return endpoint_at(coordinate);
}

std::optional<std::size_t> VirtualTopology::neighbor(
    std::size_t endpoint, Direction direction,
    bool periodic_x, bool periodic_y) const {
  TileCoordinate coordinate = tile(endpoint).coordinate;
  switch (direction) {
    case Direction::x_low:
      if (coordinate.x != 0) {
        --coordinate.x;
      } else if (periodic_x) {
        coordinate.x = shape_.px - 1;
      } else {
        return std::nullopt;
      }
      break;
    case Direction::x_high:
      if (coordinate.x + 1 < shape_.px) {
        ++coordinate.x;
      } else if (periodic_x) {
        coordinate.x = 0;
      } else {
        return std::nullopt;
      }
      break;
    case Direction::y_low:
      if (coordinate.y != 0) {
        --coordinate.y;
      } else if (periodic_y) {
        coordinate.y = shape_.py - 1;
      } else {
        return std::nullopt;
      }
      break;
    case Direction::y_high:
      if (coordinate.y + 1 < shape_.py) {
        ++coordinate.y;
      } else if (periodic_y) {
        coordinate.y = 0;
      } else {
        return std::nullopt;
      }
      break;
  }
  return endpoint_at(coordinate);
}

bool VirtualTopology::is_physical_boundary(std::size_t endpoint,
                                            Direction direction,
                                            bool periodic_x,
                                            bool periodic_y) const {
  return !neighbor(endpoint, direction, periodic_x, periodic_y).has_value();
}

std::optional<std::size_t> VirtualTopology::canonical_face_owner(
    std::size_t endpoint, Direction direction,
    bool periodic_x, bool periodic_y) const {
  const auto adjacent = neighbor(endpoint, direction, periodic_x, periodic_y);
  if (!adjacent) return std::nullopt;

  const TileCoordinate here  = tile(endpoint).coordinate;
  const TileCoordinate there = tile(*adjacent).coordinate;
  switch (direction) {
    case Direction::x_low:
    case Direction::x_high:
      return here.x <= there.x ? endpoint : *adjacent;
    case Direction::y_low:
    case Direction::y_high:
      return here.y <= there.y ? endpoint : *adjacent;
  }
  return std::nullopt;
}

bool VirtualTopology::owns_shared_face(std::size_t endpoint,
                                       Direction direction,
                                       bool periodic_x,
                                       bool periodic_y) const {
  const auto owner = canonical_face_owner(
      endpoint, direction, periodic_x, periodic_y);
  return owner && *owner == endpoint;
}

std::size_t VirtualTopology::canonical_corner_owner(
    std::span<const std::size_t> participants) const {
  if (participants.empty()) {
    throw std::invalid_argument{
        "a shared corner requires at least one participating endpoint"};
  }
  std::size_t owner = participants.front();
  TileCoordinate best = tile(owner).coordinate;
  for (const std::size_t endpoint : participants.subspan(1)) {
    const TileCoordinate candidate = tile(endpoint).coordinate;
    if (candidate.x < best.x
        || (candidate.x == best.x && candidate.y < best.y)) {
      owner = endpoint;
      best  = candidate;
    }
  }
  return owner;
}

}  // namespace quasar::distributed
