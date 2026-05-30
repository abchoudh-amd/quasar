#include "quasar/physics/analytic_fields/file_grid.hpp"

#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <gtest/gtest.h>

// NOTE: FileGridEvaluator is a placeholder that currently returns a zero field
// regardless of its path (no file loader is implemented yet). These tests pin
// that contract — registration, path round-trip, and per-point sizing returning
// zeros — so a future real loader has a baseline to diff against.

TEST(FileGridEvaluator, RegisteredAndStoresPath) {
  quasar::analytic_fields::FileGridEvaluator eval{"some/grid.npz"};
  EXPECT_EQ(eval.path(), "some/grid.npz");
  EXPECT_TRUE(quasar::Registry<quasar::numerics::IFieldEvaluator>::instance()
                  .contains("file_grid"));
}

TEST(FileGridEvaluator, ReturnsZeroFieldSizedToObservations) {
  quasar::analytic_fields::FileGridEvaluator eval{"unused"};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{1, 2, 3});
  pts.add(quasar::Vec3{4, 5, 6});
  pts.add(quasar::Vec3{7, 8, 9});

  auto b = eval.evaluate_B(cs, pts);
  ASSERT_EQ(b.size(), 3u);
  for (const auto& v : b) {
    EXPECT_DOUBLE_EQ(v.x, 0.0);
    EXPECT_DOUBLE_EQ(v.y, 0.0);
    EXPECT_DOUBLE_EQ(v.z, 0.0);
  }

  auto gb = eval.evaluate_grad_B(cs, pts);
  ASSERT_EQ(gb.size(), 3u);
  EXPECT_DOUBLE_EQ(gb[0].r0.x, 0.0);
}
