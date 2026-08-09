#pragma once

#include "quasar/core/grid.hpp"
#include "quasar/distributed/topology.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace quasar::distributed {

// Wire order for one packed MHD RK-register halo.  Keeping all components in
// one payload gives every topology edge one symmetric ByteTransfer and avoids
// consuming a separate MPI tag for each field.
enum class MhdRegisterComponent : std::size_t {
  rho = 0,
  mx,
  my,
  mz,
  energy,
  bx_face,
  by_face,
  bz_cell,
  count,
};

inline constexpr std::size_t mhd_register_component_count =
    static_cast<std::size_t>(MhdRegisterComponent::count);

using ConstMhdRegisterComponents =
    std::array<std::span<const Real>, mhd_register_component_count>;
using MhdRegisterComponents =
    std::array<std::span<Real>, mhd_register_component_count>;

enum class MhdHaloStagger {
  cell,
  // Cell-centred CT input tables also carry the physical high-y closure row.
  // Unlike a face layout, this does not change normal shared-face ownership.
  cell_extended_y,
  x_face,
  // Derived x-face records carry the true physical high-y closure used by CT.
  x_face_extended_y,
  y_face,
  node,
};

using MhdHaloLayouts =
    std::array<MhdHaloStagger, mhd_register_component_count>;

[[nodiscard]] constexpr MhdHaloLayouts mhd_register_halo_layouts() {
  return {MhdHaloStagger::cell, MhdHaloStagger::cell,
          MhdHaloStagger::cell, MhdHaloStagger::cell,
          MhdHaloStagger::cell, MhdHaloStagger::x_face,
          MhdHaloStagger::y_face, MhdHaloStagger::cell};
}

struct MhdHaloEdge {
  std::size_t first_endpoint{0};
  Direction first_direction{Direction::x_high};
  std::size_t second_endpoint{0};
  Direction second_direction{Direction::x_low};

  friend bool operator==(const MhdHaloEdge&, const MhdHaloEdge&) = default;
};

// Enumerate each physical interface once by walking the positive direction.
// With two tiles on a periodic axis the same endpoint pair correctly appears
// twice: once for the interior cut and once for the global periodic seam.
// A one-tile periodic axis is omitted because the serial boundary kernel wraps
// that tile locally.
[[nodiscard]] std::vector<MhdHaloEdge> make_mhd_halo_edges(
    const VirtualTopology& topology, Direction positive_direction,
    bool periodic_x, bool periodic_y);

// Return the symmetric payload size for a transfer crossing `direction`.
// X payloads contain (nghost+1) columns and ny+1 rows.  Y payloads contain
// (nghost+1) rows across the complete x-extended pitch, so an x exchange
// followed by a y exchange also propagates diagonal corner guards.
[[nodiscard]] std::size_t mhd_register_halo_scalar_count(
    const Grid2D& grid, Direction direction);

// Pack values sent out through `direction`.  The extra row/column makes the
// payload symmetric despite canonical staggered-face ownership: the receiver
// ignores the padding for cell-centred components and retains its canonical
// high-side face where appropriate.
void pack_mhd_register_halo(
    const Grid2D& grid, Direction direction,
    const ConstMhdRegisterComponents& components,
    std::span<Real> payload);

// Apply a payload received through `direction` to guards and to the one shared
// staggered face canonically owned by the lower-x/lower-y tile.  Physical
// exteriors are never passed to this function.
void unpack_mhd_register_halo(
    const Grid2D& grid, Direction direction,
    std::span<const Real> payload,
    const MhdRegisterComponents& components,
    bool receive_shared_face);

void unpack_mhd_register_halo(
    const Grid2D& grid, Direction direction,
    std::span<const Real> payload,
    const MhdRegisterComponents& components,
    bool receive_shared_face,
    const MhdHaloLayouts& layouts);

}  // namespace quasar::distributed
