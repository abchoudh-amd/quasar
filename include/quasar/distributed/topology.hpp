#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace quasar::distributed {

// Coordinates in the virtual GPU topology.  X is the fastest-varying
// coordinate, so endpoint e maps to {e % px, e / px}.
struct TileCoordinate {
  std::size_t x{0};
  std::size_t y{0};

  friend bool operator==(const TileCoordinate&, const TileCoordinate&) = default;
};

struct IndexRange {
  std::size_t begin{0};
  std::size_t end{0};

  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return end - begin;
  }

  [[nodiscard]] constexpr bool contains(std::size_t index) const noexcept {
    return begin <= index && index < end;
  }

  friend bool operator==(const IndexRange&, const IndexRange&) = default;
};

struct DecompositionShape {
  std::size_t px{0};
  std::size_t py{0};

  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return px * py;
  }

  friend bool operator==(const DecompositionShape&,
                         const DecompositionShape&) = default;
};

struct TileExtent {
  std::size_t    endpoint{0};
  TileCoordinate coordinate{};
  IndexRange     x{};
  IndexRange     y{};

  [[nodiscard]] constexpr std::size_t cell_count() const noexcept {
    return x.size() * y.size();
  }

  friend bool operator==(const TileExtent&, const TileExtent&) = default;
};

enum class Direction {
  x_low,
  x_high,
  y_low,
  y_high,
};

[[nodiscard]] constexpr Direction opposite(Direction direction) noexcept {
  switch (direction) {
    case Direction::x_low:  return Direction::x_high;
    case Direction::x_high: return Direction::x_low;
    case Direction::y_low:  return Direction::y_high;
    case Direction::y_high: return Direction::y_low;
  }
  return Direction::x_low;
}

// A two-dimensional topology over virtual GPU endpoints.  It deliberately has
// no MPI dependency: one rank may own several consecutive endpoints and the
// orchestration layer supplies that rank/endpoint relationship separately.
//
// Cell ownership is half open.  When a dimension does not divide evenly, its
// remainder cells are assigned one each to the lowest tile coordinates.
class VirtualTopology {
 public:
  // Select a valid factorization of endpoint_count.  The selector minimizes
  // average tile perimeter (with aspect ratio and px as deterministic
  // tie-breakers), subject to every tile dimension being at least
  // minimum_tile_width.  A width of zero means only non-empty tiles are
  // required.
  [[nodiscard]] static VirtualTopology create_auto(
      std::size_t global_nx,
      std::size_t global_ny,
      std::size_t endpoint_count,
      std::size_t minimum_tile_width = 1);

  // Construct an explicitly requested Px-by-Py topology.  px*py must equal
  // endpoint_count and every owned dimension must satisfy the same stencil
  // reach check used by create_auto.
  [[nodiscard]] static VirtualTopology create(
      std::size_t global_nx,
      std::size_t global_ny,
      std::size_t endpoint_count,
      DecompositionShape shape,
      std::size_t minimum_tile_width = 1);

  [[nodiscard]] std::size_t global_nx() const noexcept { return global_nx_; }
  [[nodiscard]] std::size_t global_ny() const noexcept { return global_ny_; }
  [[nodiscard]] std::size_t endpoint_count() const noexcept {
    return tiles_.size();
  }
  [[nodiscard]] std::size_t minimum_tile_width() const noexcept {
    return minimum_tile_width_;
  }
  [[nodiscard]] DecompositionShape shape() const noexcept { return shape_; }
  [[nodiscard]] std::span<const TileExtent> tiles() const noexcept {
    return tiles_;
  }

  [[nodiscard]] const TileExtent& tile(std::size_t endpoint) const;
  [[nodiscard]] std::size_t endpoint_at(TileCoordinate coordinate) const;
  [[nodiscard]] std::size_t owner_of_cell(std::size_t global_x,
                                          std::size_t global_y) const;

  // Return the adjacent endpoint.  A periodic global seam wraps to the tile at
  // the opposite side; a non-periodic exterior has no neighbor.  A periodic
  // dimension containing one tile returns the endpoint itself.
  [[nodiscard]] std::optional<std::size_t> neighbor(
      std::size_t endpoint,
      Direction direction,
      bool periodic_x = false,
      bool periodic_y = false) const;

  [[nodiscard]] bool is_physical_boundary(
      std::size_t endpoint,
      Direction direction,
      bool periodic_x = false,
      bool periodic_y = false) const;

  // Select the unique producer for a shared face.  X faces are owned by the
  // lower-x tile and Y faces by the lower-y tile, including global periodic
  // seams.  A physical exterior has no shared-face owner.
  [[nodiscard]] std::optional<std::size_t> canonical_face_owner(
      std::size_t endpoint,
      Direction direction,
      bool periodic_x = false,
      bool periodic_y = false) const;

  [[nodiscard]] bool owns_shared_face(
      std::size_t endpoint,
      Direction direction,
      bool periodic_x = false,
      bool periodic_y = false) const;

  // Select the lower-x/lower-y participant for a shared corner.  Duplicate
  // endpoint IDs are accepted (useful in one-tile periodic dimensions), but an
  // empty set or an out-of-range ID is rejected.
  [[nodiscard]] std::size_t canonical_corner_owner(
      std::span<const std::size_t> participants) const;

 private:
  VirtualTopology(std::size_t global_nx,
                  std::size_t global_ny,
                  DecompositionShape shape,
                  std::size_t minimum_tile_width);

  std::size_t              global_nx_{0};
  std::size_t              global_ny_{0};
  DecompositionShape       shape_{};
  std::size_t              minimum_tile_width_{1};
  std::vector<TileExtent>  tiles_{};
};

}  // namespace quasar::distributed
