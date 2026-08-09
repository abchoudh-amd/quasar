#include "quasar/distributed/mhd_halo.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <vector>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::distributed::ConstMhdRegisterComponents;
using quasar::distributed::Direction;
using quasar::distributed::MhdRegisterComponent;
using quasar::distributed::MhdRegisterComponents;
using quasar::distributed::MhdHaloLayouts;
using quasar::distributed::MhdHaloStagger;
using quasar::distributed::mhd_register_component_count;

struct HostRegister {
  explicit HostRegister(const Grid2D& grid, Real initial = Real{-1}) {
    for (auto& component : storage) {
      component.assign(grid.storage_size(), initial);
    }
  }

  [[nodiscard]] ConstMhdRegisterComponents const_views() const {
    ConstMhdRegisterComponents result;
    for (std::size_t component = 0;
         component < mhd_register_component_count; ++component) {
      result[component] = storage[component];
    }
    return result;
  }

  [[nodiscard]] MhdRegisterComponents views() {
    MhdRegisterComponents result;
    for (std::size_t component = 0;
         component < mhd_register_component_count; ++component) {
      result[component] = storage[component];
    }
    return result;
  }

  std::array<std::vector<Real>, mhd_register_component_count> storage;
};

constexpr std::size_t component_index(MhdRegisterComponent component) {
  return static_cast<std::size_t>(component);
}

Real encoded(std::size_t component, std::size_t position) {
  return static_cast<Real>(100000 * component + position + 1);
}

std::vector<Real> encoded_payload(std::size_t size,
                                  std::size_t block_size) {
  std::vector<Real> result(size);
  for (std::size_t component = 0;
       component < mhd_register_component_count; ++component) {
    for (std::size_t position = 0; position < block_size; ++position) {
      result[component * block_size + position] =
          encoded(component, position);
    }
  }
  return result;
}

}  // namespace

TEST(MhdRegisterHalo, UnevenNeighborsHaveSymmetricPayloads) {
  const Grid2D left{5, 3, 5.0, 3.0, 0.0, 0.0, 2};
  const Grid2D right{4, 3, 4.0, 3.0, 5.0, 0.0, 2};
  EXPECT_EQ(quasar::distributed::mhd_register_halo_scalar_count(
                left, Direction::x_high),
            quasar::distributed::mhd_register_halo_scalar_count(
                right, Direction::x_low));
  EXPECT_EQ(quasar::distributed::mhd_register_halo_scalar_count(
                left, Direction::x_high),
            mhd_register_component_count * 3u * 4u);

  const Grid2D lower{4, 5, 4.0, 5.0, 0.0, 0.0, 2};
  const Grid2D upper{4, 3, 4.0, 3.0, 0.0, 5.0, 2};
  EXPECT_EQ(quasar::distributed::mhd_register_halo_scalar_count(
                lower, Direction::y_high),
            quasar::distributed::mhd_register_halo_scalar_count(
                upper, Direction::y_low));
  EXPECT_EQ(quasar::distributed::mhd_register_halo_scalar_count(
                lower, Direction::y_high),
            mhd_register_component_count * 3u * 8u);
}

TEST(MhdRegisterHalo, EdgePlanKeepsDistinctPeriodicInterfaces) {
  const auto topology = quasar::distributed::VirtualTopology::create(
      9, 7, 4, quasar::distributed::DecompositionShape{2, 2}, 2);
  const auto nonperiodic_x = quasar::distributed::make_mhd_halo_edges(
      topology, Direction::x_high, false, false);
  ASSERT_EQ(nonperiodic_x.size(), 2u);

  const auto periodic_x = quasar::distributed::make_mhd_halo_edges(
      topology, Direction::x_high, true, false);
  ASSERT_EQ(periodic_x.size(), 4u);
  // For a two-tile ring each row has an interior cut and a distinct periodic
  // seam between the same endpoint IDs but with reversed directed sides.
  EXPECT_EQ(periodic_x[0].first_endpoint, 0u);
  EXPECT_EQ(periodic_x[0].second_endpoint, 1u);
  EXPECT_EQ(periodic_x[1].first_endpoint, 1u);
  EXPECT_EQ(periodic_x[1].second_endpoint, 0u);

  const auto periodic_y = quasar::distributed::make_mhd_halo_edges(
      topology, Direction::y_high, false, true);
  EXPECT_EQ(periodic_y.size(), 4u);
  EXPECT_THROW(
      (void)quasar::distributed::make_mhd_halo_edges(
          topology, Direction::x_low, false, false),
      std::invalid_argument);
}

TEST(MhdRegisterHalo, PacksXBoundaryIncludingSharedFacePadding) {
  const Grid2D grid{5, 3, 5.0, 3.0, 0.0, 0.0, 2};
  HostRegister source{grid};
  for (std::size_t component = 0;
       component < mhd_register_component_count; ++component) {
    for (std::size_t index = 0; index < grid.storage_size(); ++index) {
      source.storage[component][index] = encoded(component, index);
    }
  }

  std::vector<Real> payload(
      quasar::distributed::mhd_register_halo_scalar_count(
          grid, Direction::x_high));
  quasar::distributed::pack_mhd_register_halo(
      grid, Direction::x_high, source.const_views(), payload);

  const std::size_t block = 3u * 4u;
  for (std::size_t component = 0;
       component < mhd_register_component_count; ++component) {
    for (int j = 0; j <= grid.ny; ++j) {
      for (int offset = 0; offset <= grid.nghost; ++offset) {
        const std::size_t position =
            static_cast<std::size_t>(j * (grid.nghost + 1) + offset);
        EXPECT_EQ(payload[component * block + position],
                  encoded(component,
                          grid.index(grid.nx - grid.nghost + offset, j)));
      }
    }
  }
}

TEST(MhdRegisterHalo, PacksYBoundaryAcrossExistingXGuards) {
  const Grid2D grid{4, 5, 4.0, 5.0, 0.0, 0.0, 2};
  HostRegister source{grid};
  for (std::size_t component = 0;
       component < mhd_register_component_count; ++component) {
    for (std::size_t index = 0; index < grid.storage_size(); ++index) {
      source.storage[component][index] = encoded(component, index);
    }
  }

  std::vector<Real> payload(
      quasar::distributed::mhd_register_halo_scalar_count(
          grid, Direction::y_high));
  quasar::distributed::pack_mhd_register_halo(
      grid, Direction::y_high, source.const_views(), payload);

  const std::size_t row = static_cast<std::size_t>(grid.pitch());
  const std::size_t block = 3u * row;
  for (std::size_t component = 0;
       component < mhd_register_component_count; ++component) {
    for (int offset = 0; offset <= grid.nghost; ++offset) {
      for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
        const std::size_t position =
            static_cast<std::size_t>(offset) * row
            + static_cast<std::size_t>(i + grid.nghost);
        EXPECT_EQ(payload[component * block + position],
                  encoded(component,
                          grid.index(i, grid.ny - grid.nghost + offset)));
      }
    }
  }
}

TEST(MhdRegisterHalo, XUnpackPreservesCanonicalHighBxFace) {
  const Grid2D grid{5, 3, 5.0, 3.0, 0.0, 0.0, 2};
  const std::size_t block = 3u * 4u;
  const auto payload = encoded_payload(
      quasar::distributed::mhd_register_halo_scalar_count(
          grid, Direction::x_high),
      block);

  HostRegister high_target{grid};
  quasar::distributed::unpack_mhd_register_halo(
      grid, Direction::x_high, payload, high_target.views(), false);
  const std::size_t bx = component_index(MhdRegisterComponent::bx_face);
  const std::size_t rho = component_index(MhdRegisterComponent::rho);
  for (int j = 0; j < grid.ny; ++j) {
    const std::size_t base =
        static_cast<std::size_t>(j * (grid.nghost + 1));
    EXPECT_EQ(high_target.storage[rho][grid.index(grid.nx, j)],
              encoded(rho, base));
    EXPECT_EQ(high_target.storage[rho][grid.index(grid.nx + 1, j)],
              encoded(rho, base + 1));
    EXPECT_EQ(high_target.storage[bx][grid.index(grid.nx, j)], Real{-1});
    EXPECT_EQ(high_target.storage[bx][grid.index(grid.nx + 1, j)],
              encoded(bx, base + 1));
  }

  HostRegister low_target{grid};
  quasar::distributed::unpack_mhd_register_halo(
      grid, Direction::x_low, payload, low_target.views(), true);
  for (int j = 0; j < grid.ny; ++j) {
    const std::size_t base =
        static_cast<std::size_t>(j * (grid.nghost + 1));
    EXPECT_EQ(low_target.storage[rho][grid.index(-2, j)],
              encoded(rho, base));
    EXPECT_EQ(low_target.storage[rho][grid.index(-1, j)],
              encoded(rho, base + 1));
    EXPECT_EQ(low_target.storage[rho][grid.index(0, j)], Real{-1});
    EXPECT_EQ(low_target.storage[bx][grid.index(0, j)],
              encoded(bx, base + 2));
  }
}

TEST(MhdRegisterHalo, PeriodicSeamUsesLowerCoordinateFaceOwner) {
  const Grid2D grid{5, 3, 5.0, 3.0, 0.0, 0.0, 2};
  const std::size_t block = 3u * 4u;
  const auto payload = encoded_payload(
      quasar::distributed::mhd_register_halo_scalar_count(
          grid, Direction::x_high),
      block);
  const std::size_t bx = component_index(MhdRegisterComponent::bx_face);

  // Across the global seam, coordinate zero owns the face.  The high-x tile
  // therefore receives offset zero, while the low-x tile retains its i=0.
  HostRegister high_coordinate_target{grid};
  quasar::distributed::unpack_mhd_register_halo(
      grid, Direction::x_high, payload,
      high_coordinate_target.views(), true);
  EXPECT_EQ(high_coordinate_target.storage[bx][grid.index(grid.nx, 0)],
            encoded(bx, 0));

  HostRegister zero_coordinate_target{grid};
  quasar::distributed::unpack_mhd_register_halo(
      grid, Direction::x_low, payload,
      zero_coordinate_target.views(), false);
  EXPECT_EQ(zero_coordinate_target.storage[bx][grid.index(0, 0)], Real{-1});
}

TEST(MhdRegisterHalo, YUnpackPreservesCanonicalHighByFaceAndCorners) {
  const Grid2D grid{4, 5, 4.0, 5.0, 0.0, 0.0, 2};
  const std::size_t row = static_cast<std::size_t>(grid.pitch());
  const std::size_t block = 3u * row;
  const auto payload = encoded_payload(
      quasar::distributed::mhd_register_halo_scalar_count(
          grid, Direction::y_high),
      block);
  const std::size_t by = component_index(MhdRegisterComponent::by_face);
  const std::size_t rho = component_index(MhdRegisterComponent::rho);

  HostRegister high_target{grid};
  quasar::distributed::unpack_mhd_register_halo(
      grid, Direction::y_high, payload, high_target.views(), false);
  for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
    const std::size_t position = static_cast<std::size_t>(i + grid.nghost);
    EXPECT_EQ(high_target.storage[rho][grid.index(i, grid.ny)],
              encoded(rho, position));
    EXPECT_EQ(high_target.storage[rho][grid.index(i, grid.ny + 1)],
              encoded(rho, row + position));
    EXPECT_EQ(high_target.storage[by][grid.index(i, grid.ny)], Real{-1});
    EXPECT_EQ(high_target.storage[by][grid.index(i, grid.ny + 1)],
              encoded(by, row + position));
  }

  HostRegister low_target{grid};
  quasar::distributed::unpack_mhd_register_halo(
      grid, Direction::y_low, payload, low_target.views(), true);
  for (int i = -grid.nghost; i < grid.nx + grid.nghost; ++i) {
    const std::size_t position = static_cast<std::size_t>(i + grid.nghost);
    EXPECT_EQ(low_target.storage[rho][grid.index(i, -2)],
              encoded(rho, position));
    EXPECT_EQ(low_target.storage[rho][grid.index(i, -1)],
              encoded(rho, row + position));
    EXPECT_EQ(low_target.storage[rho][grid.index(i, 0)], Real{-1});
    EXPECT_EQ(low_target.storage[by][grid.index(i, 0)],
              encoded(by, 2u * row + position));
  }
}

TEST(MhdRegisterHalo, ExtendedCellLayoutRetainsPhysicalHighYClosureRow) {
  const Grid2D grid{5, 3, 5.0, 3.0, 0.0, 0.0, 2};
  const std::size_t block = 3u * 4u;
  const auto payload = encoded_payload(
      quasar::distributed::mhd_register_halo_scalar_count(
          grid, Direction::x_low),
      block);
  HostRegister target{grid};
  MhdHaloLayouts layouts{};
  layouts.fill(MhdHaloStagger::cell);
  layouts[0] = MhdHaloStagger::cell_extended_y;

  quasar::distributed::unpack_mhd_register_halo(
      grid, Direction::x_low, payload, target.views(), false, layouts);

  const std::size_t high_y_base =
      static_cast<std::size_t>(grid.ny * (grid.nghost + 1));
  EXPECT_EQ(target.storage[0][grid.index(-2, grid.ny)],
            encoded(0, high_y_base));
  EXPECT_EQ(target.storage[0][grid.index(-1, grid.ny)],
            encoded(0, high_y_base + 1));
  // Cell staggering still rejects the shared normal face itself.
  EXPECT_EQ(target.storage[0][grid.index(0, grid.ny)], Real{-1});
  // Ordinary cell-centred tables continue to omit the high-y padding row.
  EXPECT_EQ(target.storage[1][grid.index(-2, grid.ny)], Real{-1});
}

TEST(MhdRegisterHalo, RejectsThinTilesAndMismatchedBuffers) {
  const Grid2D thin{2, 2, 2.0, 2.0, 0.0, 0.0, 3};
  EXPECT_THROW(
      (void)quasar::distributed::mhd_register_halo_scalar_count(
          thin, Direction::x_low),
      std::invalid_argument);

  const Grid2D grid{4, 4, 4.0, 4.0, 0.0, 0.0, 2};
  HostRegister source{grid};
  std::vector<Real> short_payload(1);
  EXPECT_THROW(
      quasar::distributed::pack_mhd_register_halo(
          grid, Direction::x_low, source.const_views(), short_payload),
      std::invalid_argument);
}
