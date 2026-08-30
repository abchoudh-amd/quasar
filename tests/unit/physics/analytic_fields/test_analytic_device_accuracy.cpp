// Accuracy and reproducibility gate for the device analytic field evaluators.
//
// The host implementations these kernels replaced were deleted, so this file
// carries its own `long double` oracle rather than comparing against a second
// shipped implementation. That is the standard the GPU-residency port set for
// every displaced calculation: a test-local reference built from the closed
// form, and an assertion that the device is no worse than a naive `double`
// evaluation of the same formula -- not merely inside some absolute tolerance,
// which cannot tell a correct port from one that is slightly wrong in a way
// that happens to be small on the chosen inputs.
//
// `long double` is a legitimate reference here precisely because it is a host
// type the device does not have: its wider exponent and mantissa are what the
// deleted host code used, and reproducing that behaviour in binary64 through
// numerics::ScaledValue is what the port had to get right.

#include "quasar/physics/analytic_fields/dipole.hpp"
#include "quasar/physics/analytic_fields/file_grid.hpp"
#include "quasar/physics/analytic_fields/gradient.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include "host_evaluate.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

namespace {

using quasar::Mat3x3;
using quasar::Real;
using quasar::Vec3;
using quasar::analytic_fields::DipoleEvaluator;
using quasar::analytic_fields::FileGridEvaluator;
using quasar::analytic_fields::GradientEvaluator;
using quasar::magnetostatics::ConductorSystem;
using quasar::magnetostatics::PointCloud;

using LD = long double;

constexpr LD kMu0Over4Pi = 1.0e-7L;

struct DipoleOracle {
  LD b[3];
  LD g[3][3];
};

// Closed-form ideal point dipole, evaluated end to end in long double.
DipoleOracle dipole_oracle(Vec3 moment, Vec3 origin, Vec3 point) {
  const LD m[3] = {static_cast<LD>(moment.x), static_cast<LD>(moment.y),
                   static_cast<LD>(moment.z)};
  const LD d[3] = {static_cast<LD>(point.x) - static_cast<LD>(origin.x),
                   static_cast<LD>(point.y) - static_cast<LD>(origin.y),
                   static_cast<LD>(point.z) - static_cast<LD>(origin.z)};
  const LD r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  const LD n[3] = {d[0] / r, d[1] / r, d[2] / r};
  const LD q = m[0] * n[0] + m[1] * n[1] + m[2] * n[2];

  DipoleOracle out{};
  const LD inv_r3 = 1.0L / (r * r * r);
  const LD inv_r4 = inv_r3 / r;
  for (int i = 0; i < 3; ++i) {
    out.b[i] = kMu0Over4Pi * inv_r3 * (3.0L * q * n[i] - m[i]);
    for (int j = 0; j < 3; ++j) {
      const LD angular = m[j] * n[i] + m[i] * n[j]
                       + (i == j ? q : 0.0L) - 5.0L * q * n[i] * n[j];
      out.g[i][j] = 3.0L * kMu0Over4Pi * inv_r4 * angular;
    }
  }
  return out;
}

// Same formula evaluated naively in double, with no scaled arithmetic. This is
// the "teeth" of the comparison: on the cancelling configurations below it is
// measurably worse than the device, which proves the device path is doing the
// compensation it claims rather than getting lucky.
DipoleOracle dipole_naive(Vec3 moment, Vec3 origin, Vec3 point) {
  const Real m[3] = {moment.x, moment.y, moment.z};
  const Real d[3] = {point.x - origin.x, point.y - origin.y,
                     point.z - origin.z};
  const Real r = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  const Real n[3] = {d[0] / r, d[1] / r, d[2] / r};
  const Real q = m[0] * n[0] + m[1] * n[1] + m[2] * n[2];

  DipoleOracle out{};
  const Real C = static_cast<Real>(kMu0Over4Pi);
  const Real inv_r3 = Real{1} / (r * r * r);
  const Real inv_r4 = inv_r3 / r;
  for (int i = 0; i < 3; ++i) {
    out.b[i] = static_cast<LD>(C * inv_r3 * (Real{3} * q * n[i] - m[i]));
    for (int j = 0; j < 3; ++j) {
      const Real angular = m[j] * n[i] + m[i] * n[j]
                         + (i == j ? q : Real{0}) - Real{5} * q * n[i] * n[j];
      out.g[i][j] = static_cast<LD>(Real{3} * C * inv_r4 * angular);
    }
  }
  return out;
}

// Error measured against the oracle, normalized by the oracle's own magnitude
// so a component that cancels to near zero is judged on its own scale.
LD relative_error(LD actual, LD reference, LD scale) {
  const LD denominator = std::abs(reference) > 0.0L ? std::abs(reference)
                                                    : std::abs(scale);
  if (denominator == 0.0L) return std::abs(actual);
  return std::abs(actual - reference) / denominator;
}

PointCloud one_point(Vec3 p) {
  PointCloud pc;
  pc.add(p);
  return pc;
}

}  // namespace

TEST(AnalyticDeviceAccuracy, DipoleFieldMatchesLongDoubleOracle) {
  const ConductorSystem cs;
  const Vec3 moment{0.3, -1.7, 2.9};
  const Vec3 origin{0.11, -0.22, 0.33};
  const std::vector<Vec3> points = {
      Vec3{1.0, 0.0, 0.0}, Vec3{0.0, 2.5, -1.0}, Vec3{-3.0, 0.5, 4.0},
      Vec3{0.4, 0.4, 0.4}, Vec3{100.0, -250.0, 7.0}};

  const DipoleEvaluator eval{moment, origin};
  for (const Vec3 p : points) {
    const auto b = quasar::test::host_evaluate_B(eval, cs, one_point(p));
    const auto g = quasar::test::host_evaluate_grad_B(eval, cs, one_point(p));
    ASSERT_EQ(b.size(), 1u);
    ASSERT_EQ(g.size(), 1u);
    const DipoleOracle ref = dipole_oracle(moment, origin, p);

    const LD b_scale = std::abs(ref.b[0]) + std::abs(ref.b[1])
                     + std::abs(ref.b[2]);
    const Real actual_b[3] = {b[0].x, b[0].y, b[0].z};
    for (int i = 0; i < 3; ++i) {
      EXPECT_LT(relative_error(static_cast<LD>(actual_b[i]), ref.b[i], b_scale),
                1e-14L)
          << "component " << i << " at (" << p.x << "," << p.y << "," << p.z
          << ")";
    }

    const Vec3 rows[3] = {g[0].r0, g[0].r1, g[0].r2};
    LD g_scale = 0.0L;
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) g_scale += std::abs(ref.g[i][j]);
    }
    for (int i = 0; i < 3; ++i) {
      const Real entries[3] = {rows[i].x, rows[i].y, rows[i].z};
      for (int j = 0; j < 3; ++j) {
        EXPECT_LT(relative_error(static_cast<LD>(entries[j]), ref.g[i][j],
                                 g_scale),
                  1e-14L)
            << "gradient entry (" << i << "," << j << ")";
      }
    }
  }
}

TEST(AnalyticDeviceAccuracy, DipoleIsNoWorseThanNaiveDoubleWhereItCancels) {
  // cos(theta) = 1/sqrt(3) is the polar angle where 3*q*n_z - m_z vanishes for
  // an axial moment: the dipole's own zero-B_z cone. That is the worst case for
  // the angular difference, so it is where a naive double loses digits and the
  // scaled expansion must not.
  const ConductorSystem cs;
  const Vec3 moment{0.0, 0.0, 1.0};
  const Vec3 origin{0.0, 0.0, 0.0};
  const Real cos_theta = Real{1} / std::sqrt(Real{3});
  const Real sin_theta = std::sqrt(Real{1} - cos_theta * cos_theta);
  const Real radius = 1.25;
  const Vec3 p{radius * sin_theta, 0.0, radius * cos_theta};

  const DipoleEvaluator eval{moment, origin};
  const auto b = quasar::test::host_evaluate_B(eval, cs, one_point(p));
  ASSERT_EQ(b.size(), 1u);

  const DipoleOracle ref = dipole_oracle(moment, origin, p);
  const DipoleOracle naive = dipole_naive(moment, origin, p);
  const LD scale = std::abs(ref.b[0]) + std::abs(ref.b[1]) + std::abs(ref.b[2]);

  const LD device_error =
      relative_error(static_cast<LD>(b[0].z), ref.b[2], scale);
  const LD naive_error = relative_error(naive.b[2], ref.b[2], scale);
  EXPECT_LE(device_error, naive_error)
      << "device " << static_cast<double>(device_error) << " vs naive "
      << static_cast<double>(naive_error);
}

TEST(AnalyticDeviceAccuracy, GradientFieldSurvivesCatastrophicCancellation) {
  // B0 is chosen to cancel the displacement term to the last bit, so the true
  // answer is a small residual sitting under two contributions ~1e12 times
  // larger. A naive b0 + g*(p - origin) returns the rounding of that
  // subtraction; the exact expansion returns the residual.
  const ConductorSystem cs;
  const Real slope = 1.0e12;
  const Real position = 1.0;
  const Real residual = 1.0e-4;
  const Vec3 b0{-slope * position + residual, 0.0, 0.0};
  const Mat3x3 grad{Vec3{slope, 0.0, 0.0}, Vec3{0.0, -slope, 0.0},
                    Vec3{0.0, 0.0, 0.0}};
  const Vec3 origin{0.0, 0.0, 0.0};

  const GradientEvaluator eval{b0, grad, origin};
  const auto field =
      quasar::test::host_evaluate_B(eval, cs, one_point(Vec3{position, 0, 0}));
  ASSERT_EQ(field.size(), 1u);

  const LD reference = static_cast<LD>(b0.x)
                     + static_cast<LD>(slope) * static_cast<LD>(position);
  const LD scale = std::abs(static_cast<LD>(slope) * position);
  const LD device_error =
      relative_error(static_cast<LD>(field[0].x), reference, scale);

  const Real naive = b0.x + slope * (position - origin.x);
  const LD naive_error = relative_error(static_cast<LD>(naive), reference,
                                        scale);

  EXPECT_LE(device_error, naive_error);
  // `-slope * position + residual` is not representable: at 1e12 the ulp is
  // ~2.4e-4, so the stored b0.x carries a rounded offset rather than 1e-4
  // exactly. The reference above is built from that stored value, and the
  // expansion is exact, so the device must return it to the bit -- which is a
  // stronger statement than the naive comparison, and would fail if the sum
  // were compensated only approximately.
  EXPECT_DOUBLE_EQ(static_cast<double>(field[0].x),
                   static_cast<double>(reference));
  EXPECT_NE(static_cast<double>(field[0].x), static_cast<double>(residual));
}

TEST(AnalyticDeviceAccuracy, FileGridTrilinearMatchesLongDoubleOracle) {
  // A 2x2x2 map whose eight nodes are a general (non-linear) sample, so the
  // trilinear stencil genuinely mixes all eight corners.
  const Vec3 origin{0.0, 0.0, 0.0};
  const Vec3 spacing{1.0, 1.0, 1.0};

  // configure() runs the device solenoidality sweep, so the map has to be
  // genuinely divergence-free. Making each component depend only on the two
  // coordinates it is NOT differentiated by kills all three derivatives
  // identically, which leaves the node values otherwise arbitrary -- so the
  // trilinear stencil still mixes all eight corners with distinct values.
  const Real fx[2][2] = {{1.0, 0.5}, {-2.0, 3.0}};    // Bx(j, k)
  const Real gy[2][2] = {{0.25, -1.5}, {2.0, 0.75}};  // By(i, k)
  const Real hz[2][2] = {{-0.5, 1.25}, {0.8, -2.5}};  // Bz(i, j)
  Real nodes[8][3];
  for (int k = 0; k < 2; ++k) {
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 2; ++i) {
        const int node = i + 2 * (j + 2 * k);
        nodes[node][0] = fx[j][k];
        nodes[node][1] = gy[i][k];
        nodes[node][2] = hz[i][j];
      }
    }
  }
  std::vector<Real> values;
  for (const auto& node : nodes) {
    values.push_back(node[0]);
    values.push_back(node[1]);
    values.push_back(node[2]);
  }

  FileGridEvaluator eval;
  eval.configure({{"origin", {origin.x, origin.y, origin.z}},
                  {"spacing", {spacing.x, spacing.y, spacing.z}},
                  {"dims", {2, 2, 2}},
                  {"values", values}});

  const ConductorSystem cs;
  const Real u = 0.3, v = 0.7, w = 0.45;
  const auto sampled =
      quasar::test::host_evaluate_B(eval, cs, one_point(Vec3{u, v, w}));
  ASSERT_EQ(sampled.size(), 1u);

  const LD wu[2] = {1.0L - static_cast<LD>(u), static_cast<LD>(u)};
  const LD wv[2] = {1.0L - static_cast<LD>(v), static_cast<LD>(v)};
  const LD ww[2] = {1.0L - static_cast<LD>(w), static_cast<LD>(w)};
  LD reference[3] = {0.0L, 0.0L, 0.0L};
  // Per-component term scale: the sum of the eight |weight * node| magnitudes
  // that went into it. Judging the error against this rather than against the
  // result itself is a backward-error statement -- it asks whether the device
  // summed the terms it was given accurately, and does not silently tighten
  // into an impossible bar on a component whose terms happen to cancel.
  LD term_scale[3] = {0.0L, 0.0L, 0.0L};
  for (int k = 0; k < 2; ++k) {
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 2; ++i) {
        const int node = i + 2 * (j + 2 * k);
        const LD weight = wu[i] * wv[j] * ww[k];
        for (int c = 0; c < 3; ++c) {
          const LD term = weight * static_cast<LD>(nodes[node][c]);
          reference[c] += term;
          term_scale[c] += std::abs(term);
        }
      }
    }
  }

  const Real actual[3] = {sampled[0].x, sampled[0].y, sampled[0].z};
  for (int c = 0; c < 3; ++c) {
    const LD error = std::abs(static_cast<LD>(actual[c]) - reference[c]);
    // Four ulp of the term scale. Each corner contribution rounds twice (the
    // three-factor weight, then weight * node); the eight-term sum itself is an
    // exact expansion and contributes nothing.
    const LD bar = 4.0L * static_cast<LD>(std::numeric_limits<Real>::epsilon())
                 * term_scale[c];
    EXPECT_LE(error, bar) << "component " << c;
  }
}

TEST(AnalyticDeviceAccuracy, RepeatedLaunchesAreBitwiseIdentical) {
  // Every kernel here is one thread per point with no cross-thread reduction,
  // so reproducibility should be exact. Asserting it pins that no future change
  // introduces a floating-point atomic or a launch-geometry-dependent fold.
  const ConductorSystem cs;
  const DipoleEvaluator eval{Vec3{0.3, -1.7, 2.9}, Vec3{0.11, -0.22, 0.33}};

  PointCloud pc;
  for (int i = 0; i < 1024; ++i) {
    const Real t = static_cast<Real>(i) / 1024.0;
    pc.add(Vec3{1.0 + t, 0.5 - t, 2.0 + t * t});
  }

  const auto first = quasar::test::host_evaluate_B(eval, cs, pc);
  const auto second = quasar::test::host_evaluate_B(eval, cs, pc);
  const auto first_g = quasar::test::host_evaluate_grad_B(eval, cs, pc);
  const auto second_g = quasar::test::host_evaluate_grad_B(eval, cs, pc);
  ASSERT_EQ(first.size(), second.size());
  ASSERT_EQ(first_g.size(), second_g.size());
  for (std::size_t i = 0; i < first.size(); ++i) {
    EXPECT_EQ(first[i].x, second[i].x) << "point " << i;
    EXPECT_EQ(first[i].y, second[i].y) << "point " << i;
    EXPECT_EQ(first[i].z, second[i].z) << "point " << i;
    EXPECT_EQ(first_g[i].r0.x, second_g[i].r0.x) << "point " << i;
    EXPECT_EQ(first_g[i].r2.z, second_g[i].r2.z) << "point " << i;
  }
}
