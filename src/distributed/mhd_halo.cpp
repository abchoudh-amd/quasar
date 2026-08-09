#include "quasar/distributed/mhd_halo.hpp"

#include <limits>
#include <stdexcept>
#include <string>

namespace quasar::distributed {
namespace {

bool is_x(Direction direction) noexcept {
  return direction == Direction::x_low || direction == Direction::x_high;
}

bool is_low(Direction direction) noexcept {
  return direction == Direction::x_low || direction == Direction::y_low;
}

std::size_t checked_product(std::size_t left, std::size_t right,
                            const char* description) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error{description};
  }
  return left * right;
}

void validate_grid(const Grid2D& grid) {
  if (grid.nghost <= 0 || grid.nx < grid.nghost || grid.ny < grid.nghost) {
    throw std::invalid_argument{
        "MHD halo exchange requires non-empty tiles at least as wide as the halo"};
  }
}

template <class Components>
void validate_components(const Grid2D& grid, const Components& components) {
  for (const auto component : components) {
    if (component.size() != grid.storage_size()) {
      throw std::invalid_argument{
          "MHD halo component does not match the padded tile grid"};
    }
  }
}

void validate_payload(const Grid2D& grid, Direction direction,
                      std::size_t size) {
  if (size != mhd_register_halo_scalar_count(grid, direction)) {
    throw std::invalid_argument{
        "MHD halo payload does not match the tile and direction"};
  }
}

bool is_x_face(MhdHaloStagger layout) noexcept {
  return layout == MhdHaloStagger::x_face
      || layout == MhdHaloStagger::x_face_extended_y
      || layout == MhdHaloStagger::node;
}

bool is_y_face(MhdHaloStagger layout) noexcept {
  return layout == MhdHaloStagger::y_face
      || layout == MhdHaloStagger::node;
}

bool includes_high_y(MhdHaloStagger layout) noexcept {
  return layout == MhdHaloStagger::cell_extended_y
      || layout == MhdHaloStagger::x_face_extended_y
      || layout == MhdHaloStagger::y_face
      || layout == MhdHaloStagger::node;
}

}  // namespace

std::vector<MhdHaloEdge> make_mhd_halo_edges(
    const VirtualTopology& topology, Direction positive_direction,
    bool periodic_x, bool periodic_y) {
  if (positive_direction != Direction::x_high
      && positive_direction != Direction::y_high) {
    throw std::invalid_argument{
        "MHD halo edges must be enumerated in a positive direction"};
  }
  std::vector<MhdHaloEdge> result;
  result.reserve(topology.endpoint_count());
  for (std::size_t endpoint = 0;
       endpoint < topology.endpoint_count(); ++endpoint) {
    const auto neighbor = topology.neighbor(
        endpoint, positive_direction, periodic_x, periodic_y);
    if (!neighbor || *neighbor == endpoint) continue;
    result.push_back(MhdHaloEdge{
        endpoint, positive_direction, *neighbor,
        opposite(positive_direction)});
  }
  return result;
}

std::size_t mhd_register_halo_scalar_count(const Grid2D& grid,
                                           Direction direction) {
  validate_grid(grid);
  const std::size_t depth = static_cast<std::size_t>(grid.nghost) + 1;
  const std::size_t transverse = is_x(direction)
      ? static_cast<std::size_t>(grid.ny) + 1
      : static_cast<std::size_t>(grid.pitch());
  return checked_product(
      mhd_register_component_count,
      checked_product(depth, transverse, "MHD halo payload size overflows"),
      "MHD halo payload size overflows");
}

void pack_mhd_register_halo(
    const Grid2D& grid, Direction direction,
    const ConstMhdRegisterComponents& components,
    std::span<Real> payload) {
  validate_grid(grid);
  validate_components(grid, components);
  validate_payload(grid, direction, payload.size());

  const int halo = grid.nghost;
  std::size_t output = 0;
  for (std::size_t component = 0;
       component < mhd_register_component_count; ++component) {
    const auto values = components[component];
    if (is_x(direction)) {
      const int first_i = is_low(direction) ? 0 : grid.nx - halo;
      for (int j = 0; j <= grid.ny; ++j) {
        for (int offset = 0; offset <= halo; ++offset) {
          payload[output++] = values[grid.index(first_i + offset, j)];
        }
      }
    } else {
      const int first_j = is_low(direction) ? 0 : grid.ny - halo;
      for (int offset = 0; offset <= halo; ++offset) {
        const int j = first_j + offset;
        for (int i = -halo; i < grid.nx + halo; ++i) {
          payload[output++] = values[grid.index(i, j)];
        }
      }
    }
  }
}

void unpack_mhd_register_halo(
    const Grid2D& grid, Direction direction,
    std::span<const Real> payload,
    const MhdRegisterComponents& components,
    bool receive_shared_face) {
  unpack_mhd_register_halo(
      grid, direction, payload, components, receive_shared_face,
      mhd_register_halo_layouts());
}

void unpack_mhd_register_halo(
    const Grid2D& grid, Direction direction,
    std::span<const Real> payload,
    const MhdRegisterComponents& components,
    bool receive_shared_face,
    const MhdHaloLayouts& layouts) {
  validate_grid(grid);
  validate_components(grid, components);
  validate_payload(grid, direction, payload.size());

  const int halo = grid.nghost;
  std::size_t input = 0;
  for (std::size_t component = 0;
       component < mhd_register_component_count; ++component) {
    auto values = components[component];
    const MhdHaloStagger layout = layouts[component];
    if (is_x(direction)) {
      for (int j = 0; j <= grid.ny; ++j) {
        for (int offset = 0; offset <= halo; ++offset) {
          const Real value = payload[input++];
          const int i = is_low(direction)
              ? -halo + offset
              : grid.nx + offset;
          const bool transverse_owned = includes_high_y(layout)
              ? j <= grid.ny : j < grid.ny;
          const bool normal_target = !is_x_face(layout)
              ? offset < halo
              : is_low(direction)
                  ? (offset < halo
                     || (offset == halo && receive_shared_face))
                  : (offset > 0 && offset < halo)
                     || (offset == 0 && receive_shared_face);
          if (transverse_owned && normal_target) {
            values[grid.index(i, j)] = value;
          }
        }
      }
    } else {
      for (int offset = 0; offset <= halo; ++offset) {
        const int j = is_low(direction)
            ? -halo + offset
            : grid.ny + offset;
        for (int i = -halo; i < grid.nx + halo; ++i) {
          const Real value = payload[input++];
          const bool normal_target = !is_y_face(layout)
              ? offset < halo
              : is_low(direction)
                  ? (offset < halo
                     || (offset == halo && receive_shared_face))
                  : (offset > 0 && offset < halo)
                     || (offset == 0 && receive_shared_face);
          if (normal_target) values[grid.index(i, j)] = value;
        }
      }
    }
  }
}

}  // namespace quasar::distributed
