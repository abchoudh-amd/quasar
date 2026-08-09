#include "quasar/distributed/topology.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <stdexcept>

using quasar::distributed::DecompositionShape;
using quasar::distributed::Direction;
using quasar::distributed::IndexRange;
using quasar::distributed::TileCoordinate;
using quasar::distributed::VirtualTopology;

TEST(VirtualTopology, PartitionsUnevenMeshIntoHalfOpenOwnedExtents) {
  const auto topology = VirtualTopology::create(10, 7, 6, {3, 2}, 3);

  EXPECT_EQ(topology.shape(), (DecompositionShape{3, 2}));
  EXPECT_EQ(topology.endpoint_count(), 6u);
  EXPECT_EQ(topology.tile(0).coordinate, (TileCoordinate{0, 0}));
  EXPECT_EQ(topology.tile(0).x, (IndexRange{0, 4}));
  EXPECT_EQ(topology.tile(1).x, (IndexRange{4, 7}));
  EXPECT_EQ(topology.tile(2).x, (IndexRange{7, 10}));
  EXPECT_EQ(topology.tile(0).y, (IndexRange{0, 4}));
  EXPECT_EQ(topology.tile(3).y, (IndexRange{4, 7}));

  std::size_t covered_cells = 0;
  for (const auto& tile : topology.tiles()) covered_cells += tile.cell_count();
  EXPECT_EQ(covered_cells, 70u);

  for (std::size_t y = 0; y < 7; ++y) {
    for (std::size_t x = 0; x < 10; ++x) {
      const auto& owner = topology.tile(topology.owner_of_cell(x, y));
      EXPECT_TRUE(owner.x.contains(x));
      EXPECT_TRUE(owner.y.contains(y));
    }
  }
  EXPECT_EQ(topology.owner_of_cell(3, 3), 0u);
  EXPECT_EQ(topology.owner_of_cell(4, 3), 1u);
  EXPECT_EQ(topology.owner_of_cell(9, 6), 5u);
}

TEST(VirtualTopology, AutoFactorizationMinimizesTileSurface) {
  EXPECT_EQ(VirtualTopology::create_auto(120, 20, 6, 1).shape(),
            (DecompositionShape{6, 1}));
  EXPECT_EQ(VirtualTopology::create_auto(64, 64, 6, 1).shape(),
            (DecompositionShape{2, 3}));

  // Both one-dimensional splits are too thin for this stencil.  Auto must
  // choose the valid 2x2 factorization instead of constructing multi-hop halos.
  EXPECT_EQ(VirtualTopology::create_auto(12, 8, 4, 4).shape(),
            (DecompositionShape{2, 2}));
}

TEST(VirtualTopology, RejectsMismatchedOrStencilThinDecompositions) {
  EXPECT_THROW((void)VirtualTopology::create(16, 16, 4, {3, 1}, 1),
               std::invalid_argument);
  EXPECT_THROW((void)VirtualTopology::create(8, 16, 4, {4, 1}, 3),
               std::invalid_argument);
  EXPECT_THROW((void)VirtualTopology::create_auto(4, 4, 8, 2),
               std::invalid_argument);
  EXPECT_THROW((void)VirtualTopology::create_auto(0, 8, 1),
               std::invalid_argument);
}

TEST(VirtualTopology, FindsInternalAndPeriodicSeamNeighbors) {
  const auto topology = VirtualTopology::create(12, 8, 6, {3, 2});

  EXPECT_EQ(topology.neighbor(0, Direction::x_high), 1u);
  EXPECT_EQ(topology.neighbor(1, Direction::x_low), 0u);
  EXPECT_EQ(topology.neighbor(0, Direction::y_high), 3u);
  EXPECT_EQ(topology.neighbor(0, Direction::x_low), std::nullopt);
  EXPECT_EQ(topology.neighbor(0, Direction::y_low), std::nullopt);

  EXPECT_EQ(topology.neighbor(0, Direction::x_low, true, false), 2u);
  EXPECT_EQ(topology.neighbor(2, Direction::x_high, true, false), 0u);
  EXPECT_EQ(topology.neighbor(0, Direction::y_low, false, true), 3u);
  EXPECT_FALSE(topology.is_physical_boundary(
      0, Direction::x_low, true, false));
  EXPECT_TRUE(topology.is_physical_boundary(
      0, Direction::y_low, true, false));
}

TEST(VirtualTopology, AssignsOneCanonicalOwnerPerSharedFace) {
  const auto topology = VirtualTopology::create(12, 8, 6, {3, 2});

  EXPECT_EQ(topology.canonical_face_owner(0, Direction::x_high), 0u);
  EXPECT_EQ(topology.canonical_face_owner(1, Direction::x_low), 0u);
  EXPECT_TRUE(topology.owns_shared_face(0, Direction::x_high));
  EXPECT_FALSE(topology.owns_shared_face(1, Direction::x_low));

  EXPECT_EQ(topology.canonical_face_owner(0, Direction::y_high), 0u);
  EXPECT_EQ(topology.canonical_face_owner(3, Direction::y_low), 0u);

  // At a periodic high/low seam the numeric lower coordinate is canonical.
  EXPECT_EQ(topology.canonical_face_owner(
                2, Direction::x_high, true, false),
            0u);
  EXPECT_EQ(topology.canonical_face_owner(
                0, Direction::x_low, true, false),
            0u);
  EXPECT_EQ(topology.canonical_face_owner(0, Direction::x_low), std::nullopt);
}

TEST(VirtualTopology, AssignsLowerXLowerYCornerOwner) {
  const auto topology = VirtualTopology::create(8, 8, 4, {2, 2});
  const std::array<std::size_t, 4> participants{3, 2, 1, 0};
  EXPECT_EQ(topology.canonical_corner_owner(participants), 0u);

  const std::array<std::size_t, 3> duplicate_participants{2, 0, 0};
  EXPECT_EQ(topology.canonical_corner_owner(duplicate_participants), 0u);
  EXPECT_THROW((void)topology.canonical_corner_owner({}),
               std::invalid_argument);
}

TEST(VirtualTopology, OneTilePeriodicDimensionNeighborsItself) {
  const auto topology = VirtualTopology::create(8, 8, 1, {1, 1});
  EXPECT_EQ(topology.neighbor(0, Direction::x_low, true, true), 0u);
  EXPECT_EQ(topology.canonical_face_owner(
                0, Direction::x_high, true, true),
            0u);
}

TEST(VirtualTopology, RejectsOutOfRangeQueries) {
  const auto topology = VirtualTopology::create(8, 8, 4, {2, 2});
  EXPECT_THROW((void)topology.tile(4), std::out_of_range);
  EXPECT_THROW((void)topology.endpoint_at({2, 0}), std::out_of_range);
  EXPECT_THROW((void)topology.owner_of_cell(8, 0), std::out_of_range);
  const std::array<std::size_t, 2> participants{0, 4};
  EXPECT_THROW((void)topology.canonical_corner_owner(participants),
               std::out_of_range);
}
