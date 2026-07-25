#include "quasar/core/field.hpp"

#include <gtest/gtest.h>

#include <utility>

TEST(Field, EmptyIterationDoesNotPerformNullPointerArithmetic) {
  quasar::Field<int> empty;
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.begin(), empty.end());
  EXPECT_EQ(empty.data(), nullptr);
}

TEST(Field, MoveConstructionLeavesSourceInValidEmptyState) {
  quasar::Field<int> source{3};
  source[0] = 4;
  source[1] = 5;
  source[2] = 6;

  quasar::Field<int> destination{std::move(source)};
  EXPECT_TRUE(source.empty());
  EXPECT_EQ(source.size(), 0u);
  EXPECT_EQ(source.data(), nullptr);
  EXPECT_EQ(source.begin(), source.end());
  ASSERT_EQ(destination.size(), 3u);
  EXPECT_EQ(destination[0], 4);
  EXPECT_EQ(destination[1], 5);
  EXPECT_EQ(destination[2], 6);
}

TEST(Field, MoveAssignmentAndSelfMovePreserveValidStates) {
  quasar::Field<int> source{2};
  source[0] = 7;
  source[1] = 8;
  quasar::Field<int> destination{1};
  destination = std::move(source);

  EXPECT_TRUE(source.empty());
  ASSERT_EQ(destination.size(), 2u);
  EXPECT_EQ(destination[0], 7);
  EXPECT_EQ(destination[1], 8);

  destination = std::move(destination);
  ASSERT_EQ(destination.size(), 2u);
  EXPECT_EQ(destination[0], 7);
  EXPECT_EQ(destination[1], 8);
}
