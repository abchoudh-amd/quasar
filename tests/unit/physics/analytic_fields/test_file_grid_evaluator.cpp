#include "quasar/physics/analytic_fields/file_grid.hpp"

#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <stdexcept>

#include <gtest/gtest.h>

// NOTE: FileGridEvaluator is registered (so the deck name "file_grid" is
// reserved) but not yet implemented. Until a real loader exists it must fail
// loudly rather than return a silently-zero field that looks like a valid
// result. These tests pin registration + the throw contract.

TEST(FileGridEvaluator, RegisteredAndStoresPath) {
  quasar::analytic_fields::FileGridEvaluator eval{"some/grid.npz"};
  EXPECT_EQ(eval.path(), "some/grid.npz");
  EXPECT_TRUE(quasar::Registry<quasar::numerics::IFieldEvaluator>::instance()
                  .contains("file_grid"));
}

TEST(FileGridEvaluator, EvaluateThrowsUntilImplemented) {
  quasar::analytic_fields::FileGridEvaluator eval{"unused"};
  quasar::magnetostatics::ConductorSystem cs;
  quasar::magnetostatics::PointCloud pts;
  pts.add(quasar::Vec3{1, 2, 3});

  EXPECT_THROW((void)eval.evaluate_B(cs, pts), std::logic_error);
  EXPECT_THROW((void)eval.evaluate_grad_B(cs, pts), std::logic_error);
}
