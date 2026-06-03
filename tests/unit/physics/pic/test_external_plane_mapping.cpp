// Pins the slice-plane component mapping in sample_external_field. A uniform
// external field is constant everywhere, so after sampling every stored node (and
// its edge-replicated ghost) must equal the mapped constant. The "xz" plane maps
// the lab vector through a right-handed 90-degree rotation about lab x
// (pic frame = (x, z, -y)): bx<-B.x, by<-B.z, bz<--B.y. The sign on bz is what
// keeps the Boris cross products right-handed, so this test exists to catch a
// regression that silently flips it (a bare y<->z swap).

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/physics/analytic_fields/uniform.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/pic/pic_solver.hpp"

#include <gtest/gtest.h>

#include <vector>

namespace {

quasar::Real max_abs_diff(quasar::backend::DeviceBuffer<quasar::Real>& buf,
                          quasar::Real expected) {
  std::vector<quasar::Real> host(buf.size());
  buf.copy_to_host(host.data(), host.size());
  quasar::Real d = 0;
  for (auto v : host) d = std::max(d, std::abs(v - expected));
  return d;
}

}  // namespace

TEST(PicExternalPlaneMapping, XyIsIdentityXzIsRightHandedRotation) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // A distinct value per lab axis so a wrong axis/sign is detectable.
  const quasar::Vec3 b_lab{1.0, 2.0, 3.0};
  const quasar::Vec3 e_lab{4.0, 5.0, 6.0};
  quasar::analytic_fields::UniformEvaluator eval{b_lab, e_lab};
  quasar::magnetostatics::ConductorSystem cs;  // ignored by the analytic evaluator

  quasar::Grid2D g{8, 8, 0.10, 0.10, 0.0, 0.0, 1};
  constexpr quasar::Real kTol = 1e-12;

  // "xy" (default): identity map, external components equal the lab components.
  {
    quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
    quasar::pic::sample_external_field(eval, cs, solver.external_fields(),
                                       1.0, 1.0, 1.0, "xy");
    auto& f = solver.external_fields();
    EXPECT_LT(max_abs_diff(f.bx, b_lab.x), kTol);
    EXPECT_LT(max_abs_diff(f.by, b_lab.y), kTol);
    EXPECT_LT(max_abs_diff(f.bz, b_lab.z), kTol);
    EXPECT_LT(max_abs_diff(f.ex, e_lab.x), kTol);
    EXPECT_LT(max_abs_diff(f.ey, e_lab.y), kTol);
    EXPECT_LT(max_abs_diff(f.ez, e_lab.z), kTol);
  }

  // "xz": pic-x<-lab x, pic-y<-lab z, pic-z<--lab y.
  {
    quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
    quasar::pic::sample_external_field(eval, cs, solver.external_fields(),
                                       1.0, 1.0, 1.0, "xz");
    auto& f = solver.external_fields();
    EXPECT_LT(max_abs_diff(f.bx, b_lab.x), kTol);
    EXPECT_LT(max_abs_diff(f.by, b_lab.z), kTol);
    EXPECT_LT(max_abs_diff(f.bz, -b_lab.y), kTol);
    EXPECT_LT(max_abs_diff(f.ex, e_lab.x), kTol);
    EXPECT_LT(max_abs_diff(f.ey, e_lab.z), kTol);
    EXPECT_LT(max_abs_diff(f.ez, -e_lab.y), kTol);
  }
}
