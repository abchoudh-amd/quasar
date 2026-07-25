#include "quasar/physics/analytic_fields/file_grid.hpp"

#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include <filesystem>
#include <fstream>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <gtest/gtest.h>

namespace {

using quasar::Mat3x3;
using quasar::Vec3;
using quasar::analytic_fields::FileGridEvaluator;
using quasar::magnetostatics::ConductorSystem;
using quasar::magnetostatics::PointCloud;

std::filesystem::path unique_path(const char* stem) {
  const auto token = std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count());
  return std::filesystem::temp_directory_path() / (std::string{stem} + token + ".qgrid");
}

struct TempFile {
  std::filesystem::path path;
  explicit TempFile(std::filesystem::path value) : path{std::move(value)} {}
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;
  TempFile(TempFile&& other) noexcept : path{std::exchange(other.path, {})} {}
  TempFile& operator=(TempFile&& other) noexcept {
    if (this != &other) path = std::exchange(other.path, {});
    return *this;
  }
  ~TempFile() { std::error_code ignored; std::filesystem::remove(path, ignored); }
};

Vec3 linear_field(double x, double y, double z) {
  return Vec3{1.0 + 2.0 * x + 3.0 * y - z,
              -2.0 - 4.0 * x + 0.5 * y + 5.0 * z,
              0.25 + x - 2.0 * y - 2.5 * z};
}

TempFile write_linear_grid() {
  TempFile file{unique_path("quasar_file_grid_")};
  std::ofstream out{file.path};
  out << "QUASAR_FILE_GRID 1\n"
      << "dims 2 2 2\n"
      << "origin 10 -4 2\n"
      << "spacing 0.5 2 4\n"
      << "data\n";
  for (int k = 0; k < 2; ++k) {
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 2; ++i) {
        const Vec3 value = linear_field(10.0 + 0.5 * i, -4.0 + 2.0 * j,
                                        2.0 + 4.0 * k);
        out << value.x << ' ' << value.y << ' ' << value.z << '\n';
      }
    }
  }
  out.close();
  return file;
}

void expect_vec_near(Vec3 actual, Vec3 expected, double tolerance = 1e-12) {
  EXPECT_NEAR(actual.x, expected.x, tolerance);
  EXPECT_NEAR(actual.y, expected.y, tolerance);
  EXPECT_NEAR(actual.z, expected.z, tolerance);
}

}  // namespace

TEST(FileGridEvaluator, RegisteredAndDefaultRequiresConfiguration) {
  FileGridEvaluator eval;
  EXPECT_TRUE(quasar::Registry<quasar::numerics::IFieldEvaluator>::instance()
                  .contains("file_grid"));
  EXPECT_FALSE(eval.configured());
  ConductorSystem source;
  PointCloud points;
  points.add(Vec3{0, 0, 0});
  EXPECT_THROW((void)eval.evaluate_B(source, points), std::invalid_argument);
  EXPECT_THROW((void)eval.evaluate_grad_B(source, points), std::invalid_argument);
}

TEST(FileGridEvaluator, TextGridTrilinearInterpolationIsExactForLinearField) {
  const TempFile file = write_linear_grid();
  FileGridEvaluator eval{file.path.string()};
  EXPECT_TRUE(eval.configured());
  EXPECT_EQ(eval.path(), file.path.string());

  ConductorSystem source;
  PointCloud points;
  points.add(Vec3{10.125, -2.5, 5.0});
  points.add(Vec3{10.5, -2.0, 6.0});  // upper corner
  const auto field = eval.evaluate_B(source, points);
  ASSERT_EQ(field.size(), 2u);
  expect_vec_near(field[0], linear_field(10.125, -2.5, 5.0));
  expect_vec_near(field[1], linear_field(10.5, -2.0, 6.0));
}

TEST(FileGridEvaluator, GradientMatchesPiecewiseTrilinearJacobian) {
  const TempFile file = write_linear_grid();
  FileGridEvaluator eval{file.path.string()};
  ConductorSystem source;
  PointCloud points;
  points.add(Vec3{10.25, -3.0, 4.0});
  const auto gradient = eval.evaluate_grad_B(source, points);
  ASSERT_EQ(gradient.size(), 1u);
  const Mat3x3 expected{Vec3{2.0, 3.0, -1.0},
                        Vec3{-4.0, 0.5, 5.0},
                        Vec3{1.0, -2.0, -2.5}};
  expect_vec_near(gradient[0].r0, expected.r0);
  expect_vec_near(gradient[0].r1, expected.r1);
  expect_vec_near(gradient[0].r2, expected.r2);
}

TEST(FileGridEvaluator, ExtremeScaleInterpolationAndGradientRemainFinite) {
  const double max = std::numeric_limits<double>::max();
  FileGridEvaluator wide;
  wide.configure({
      {"origin", {0, 0, 0}}, {"spacing", {max, 0, 0}},
      {"dims", {2, 1, 1}},
      // A transverse field may reverse across x without contributing to
      // div(B).  The midpoint cancels, while dBy/dx=(max-(-max))/max=2.
      {"values", {0, -max, 0, 0, max, 0}},
  });
  ConductorSystem source;
  PointCloud midpoint;
  midpoint.add(Vec3{max / 2, 0, 0});
  expect_vec_near(wide.evaluate_B(source, midpoint)[0], Vec3{0, 0, 0});
  expect_vec_near(wide.evaluate_grad_B(source, midpoint)[0].r1,
                  Vec3{2, 0, 0});

  const double tiny = std::numeric_limits<double>::denorm_min();
  FileGridEvaluator narrow;
  narrow.configure({
      {"origin", {0, 0, 0}}, {"spacing", {tiny, 0, 0}},
      {"dims", {2, 1, 1}},
      // dBy/dx=tiny/tiny=1 even though 1/tiny is not representable.
      {"values", {0, 0, 0, 0, tiny, 0}},
  });
  PointCloud lower_face;
  lower_face.add(Vec3{0, 0, 0});
  expect_vec_near(narrow.evaluate_grad_B(source, lower_face)[0].r1,
                  Vec3{1, 0, 0});

  FileGridEvaluator constant;
  constant.configure({
      {"origin", {0, 0, 0}}, {"spacing", {tiny, 0, 0}},
      {"dims", {2, 1, 1}},
      // Differentiating a huge constant must cancel before the reciprocal of
      // the subnormal spacing is materialized.
      {"values", {0, max, 0, 0, max, 0}},
  });
  expect_vec_near(constant.evaluate_grad_B(source, lower_face)[0].r1,
                  Vec3{0, 0, 0});
}

TEST(FileGridEvaluator, EndpointDistinctnessUsesIndependentlyRoundedNodes) {
  FileGridEvaluator valid;
  EXPECT_NO_THROW(valid.configure({
      {"origin", {0x1.fffffffffffffp-1, 0, 0}},
      {"spacing", {0x1p-54, 0, 0}}, {"dims", {2, 1, 1}},
      {"values", {0, 0, 0, 0, 0, 0}},
  }));

  FileGridEvaluator collapsed;
  EXPECT_THROW(collapsed.configure({
      {"origin", {0x1p-1000, 0, 0}},
      {"spacing", {0x0.0000000280000p-1022, 0, 0}},
      {"dims", {3, 1, 1}},
      {"values", {0, 0, 0, 0, 0, 0, 0, 0, 0}},
  }), std::overflow_error);
}

TEST(FileGridEvaluator, RegistryConfigurationAcceptsSingletonAxes) {
  FileGridEvaluator eval;
  quasar::numerics::EvaluatorParams params{
      {"origin", {1.0, 2.0, 3.0}},
      {"spacing", {0.5, 1.0, 2.0}},
      {"dims", {2.0, 1.0, 1.0}},
      // Bx is constant, while transverse components may vary along x; hence
      // the one-dimensional map remains divergence-free.
      {"values", {1.0, 2.0, 3.0, 1.0, 4.0, 5.0}},
  };
  eval.configure(params);
  ConductorSystem source;
  PointCloud points;
  points.add(Vec3{1.25, 2.0, 3.0});
  const auto field = eval.evaluate_B(source, points);
  expect_vec_near(field[0], Vec3{1.0, 3.0, 4.0});
  const auto gradient = eval.evaluate_grad_B(source, points);
  expect_vec_near(gradient[0].r0, Vec3{0.0, 0.0, 0.0});
  expect_vec_near(gradient[0].r1, Vec3{4.0, 0.0, 0.0});
  expect_vec_near(gradient[0].r2, Vec3{4.0, 0.0, 0.0});
}

TEST(FileGridEvaluator, InternalKnotJacobianUsesRightHandCell) {
  FileGridEvaluator eval;
  eval.configure({
      {"origin", {0, 0, 0}},
      {"spacing", {1, 0, 0}},
      {"dims", {3, 1, 1}},
      // B=(0, f(x), 0), f=[0,1,3], is solenoidal.  Its slope is 1 in
      // [0,1] and 2 in [1,2].
      {"values", {0, 0, 0, 0, 1, 0, 0, 3, 0}},
  });
  ConductorSystem source;
  PointCloud points;
  points.add(Vec3{1.0, 0.0, 0.0});
  points.add(Vec3{2.0, 0.0, 0.0});
  const auto gradient = eval.evaluate_grad_B(source, points);
  ASSERT_EQ(gradient.size(), 2u);
  // Interior knots select the positive-coordinate cell; the upper endpoint
  // selects the only adjacent cell on its left.  Both therefore see slope 2.
  expect_vec_near(gradient[0].r1, Vec3{2.0, 0.0, 0.0});
  expect_vec_near(gradient[1].r1, Vec3{2.0, 0.0, 0.0});
}

TEST(FileGridEvaluator, SingletonAxisDoesNotUseAnAbsolutePositionFloor) {
  FileGridEvaluator eval;
  eval.configure({
      {"origin", {0, 0, 0}}, {"spacing", {1, 1, 1}}, {"dims", {1, 1, 1}},
      {"values", {1, 2, 3}},
  });
  ConductorSystem source;
  PointCloud points;
  points.add(Vec3{1.0e-15, 0, 0});
  EXPECT_THROW((void)eval.evaluate_B(source, points), std::out_of_range);

  FileGridEvaluator translated;
  translated.configure({
      {"origin", {1.0e16, 0, 0}}, {"spacing", {0, 0, 0}},
      {"dims", {1, 1, 1}}, {"values", {1, 2, 3}},
  });
  PointCloud adjacent;
  adjacent.add(Vec3{std::nextafter(1.0e16,
                                  std::numeric_limits<double>::infinity()),
                    0, 0});
  EXPECT_THROW((void)translated.evaluate_B(source, adjacent), std::out_of_range);
}

TEST(FileGridEvaluator, ExtremeTranslatedCoordinatesUseAScaledIndexRatio) {
  const double max = std::numeric_limits<double>::max();
  FileGridEvaluator eval;
  eval.configure({
      {"origin", {-max, 0, 0}}, {"spacing", {max, 0, 0}},
      {"dims", {3, 1, 1}},
      {"values", {0, 0, 0, 0, 1, 0, 0, 2, 0}},
  });
  ConductorSystem source;
  PointCloud points;
  points.add(Vec3{max, 0, 0});
  expect_vec_near(eval.evaluate_B(source, points)[0], Vec3{0, 2, 0});
}

TEST(FileGridEvaluator, RejectsOutsideAndNonFiniteObservations) {
  const TempFile file = write_linear_grid();
  FileGridEvaluator eval{file.path.string()};
  ConductorSystem source;

  PointCloud outside;
  outside.add(Vec3{9.9, -3.0, 4.0});
  EXPECT_THROW((void)eval.evaluate_B(source, outside), std::out_of_range);

  PointCloud nonfinite;
  EXPECT_THROW(nonfinite.add(Vec3{std::numeric_limits<double>::quiet_NaN(), 0, 0}),
               std::invalid_argument);
}

TEST(FileGridEvaluator, RejectsMalformedFilesAndConfiguration) {
  TempFile malformed{unique_path("quasar_bad_grid_")};
  {
    std::ofstream out{malformed.path};
    out << "QUASAR_FILE_GRID 1\n"
        << "dims 2 2 2\n"
        << "origin 0 0 0\n"
        << "spacing 1 1 1\n"
        << "data\n0 0 0\n";
  }
  EXPECT_THROW((FileGridEvaluator{malformed.path.string()}), std::invalid_argument);
  EXPECT_THROW((FileGridEvaluator{"/definitely/not/a/grid.qgrid"}),
               std::runtime_error);

  FileGridEvaluator eval;
  EXPECT_THROW(eval.configure({{"origin", {0, 0, 0}}}), std::invalid_argument);
  EXPECT_THROW(eval.configure({{"origin", {0, 0, 0}},
                               {"spacing", {1, 1, 1}},
                               {"dims", {1, 1, 1}},
                               {"values", {0, 0, 0}},
                               {"typo", {1}}}),
               std::invalid_argument);
  EXPECT_THROW(eval.configure({{"origin", {0, 0, 0}},
                               {"spacing", {1, 1, 1}},
                               {"dims", {1.5, 1, 1}},
                               {"values", {0, 0, 0}}}),
               std::invalid_argument);
  EXPECT_THROW(eval.configure({{"origin", {0, 0, 0}},
                               {"spacing", {1, 0, 1}},
                               {"dims", {1, 2, 1}},
                               {"values", {0, 0, 0, 0, 0, 0}}}),
               std::invalid_argument);

  if constexpr (std::numeric_limits<std::size_t>::digits
                < std::numeric_limits<double>::max_exponent) {
    const double first_too_large = std::ldexp(
        1.0, std::numeric_limits<std::size_t>::digits);
    EXPECT_THROW(eval.configure({{"origin", {0, 0, 0}},
                                 {"spacing", {1, 0, 0}},
                                 {"dims", {first_too_large, 1, 1}},
                                 {"values", {0, 0, 0}}}),
                 std::invalid_argument);
  }
}

TEST(FileGridEvaluator, RejectsNonSolenoidalAndCollapsedMapsTransactionally) {
  const TempFile file = write_linear_grid();
  FileGridEvaluator eval{file.path.string()};
  const std::string original_path = eval.path();

  auto invalid = quasar::numerics::EvaluatorParams{
      {"origin", {0, 0, 0}}, {"spacing", {1, 1, 1}}, {"dims", {2, 2, 2}},
      {"values", std::vector<double>(24, 0.0)}};
  // B=(x,y,z) has divergence 3.
  for (int k = 0; k < 2; ++k) {
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 2; ++i) {
        const std::size_t n = static_cast<std::size_t>(i + 2 * (j + 2 * k));
        invalid["values"][3 * n] = i;
        invalid["values"][3 * n + 1] = j;
        invalid["values"][3 * n + 2] = k;
      }
    }
  }
  EXPECT_THROW(eval.configure(invalid), std::invalid_argument);
  EXPECT_TRUE(eval.configured());
  EXPECT_EQ(eval.path(), original_path);

  invalid["origin"] = {1.0e308, 0, 0};
  invalid["values"].assign(24, 0.0);
  EXPECT_THROW(eval.configure(invalid), std::overflow_error);
  EXPECT_EQ(eval.path(), original_path);
}
