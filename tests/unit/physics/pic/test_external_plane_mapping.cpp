// Pins the slice-plane component mapping in sample_external_field. A uniform
// external field is constant everywhere, so after sampling every stored node (and
// its analytically sampled ghosts) must equal the mapped constant. The "xz" plane maps
// the lab vector through a right-handed 90-degree rotation about lab x
// (pic frame = (x, z, -y)): bx<-B.x, by<-B.z, bz<--B.y. The sign on bz is what
// keeps the Boris cross products right-handed, so this test exists to catch a
// regression that silently flips it (a bare y<->z swap).

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/physics/analytic_fields/uniform.hpp"
#include "quasar/physics/analytic_fields/gradient.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/pic/pic_solver.hpp"

#include "host_field_evaluator.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

quasar::Real max_abs_diff(quasar::backend::DeviceBuffer<quasar::Real>& buf,
                          quasar::Real expected) {
  std::vector<quasar::Real> host(buf.size());
  buf.copy_to_host(host.data(), host.size());
  quasar::Real d = 0;
  for (const auto v : host) {
    if (!std::isfinite(v)) return std::numeric_limits<quasar::Real>::infinity();
    d = std::max(d, std::abs(v - expected));
  }
  return d;
}

quasar::Real integer_power(quasar::Real value, int exponent) {
  quasar::Real result = 1.0;
  for (int power = 0; power < exponent; ++power) result *= value;
  return result;
}

// Rotationally covariant about lab z, but deliberately violates Maxwell's
// magnetic solenoidality: B=(x,y,0) has div(B)=2.  A cylindrical sampler must
// reject it even though it has perfect axis parity and quarter-turn covariance.
class AxisymmetricMonopoleEvaluator final
    : public quasar::test::HostFieldEvaluator {
 public:
  bool provides_grad_B() const noexcept override { return true; }

  quasar::Field<quasar::Vec3> host_B(
      const quasar::core::IFieldSource&,
      const quasar::core::PointCloud& observations) const override {
    quasar::Field<quasar::Vec3> out(observations.size());
    for (std::size_t i = 0; i < observations.size(); ++i) {
      const auto p = observations.points()[i];
      out[i] = quasar::Vec3{p.x, p.y, 0.0};
    }
    return out;
  }

  quasar::Field<quasar::Mat3x3> host_grad_B(
      const quasar::core::IFieldSource&,
      const quasar::core::PointCloud& observations) const override {
    quasar::Field<quasar::Mat3x3> out(observations.size());
    for (std::size_t i = 0; i < observations.size(); ++i) {
      out[i] = quasar::Mat3x3{
          quasar::Vec3{1.0, 0.0, 0.0},
          quasar::Vec3{0.0, 1.0, 0.0},
          quasar::Vec3{0.0, 0.0, 0.0}};
    }
    return out;
  }
};

// The flux function psi=r^2*z^n gives the smooth, regular, axisymmetric field
//   Br=-n*r*z^(n-1), Bz=2*z^n.
// Its continuous divergence vanishes identically.  For n=3 the sampled O2 Yee
// divergence has an O(dz^2) truncation residual; for n=5 the O4 residual is
// O(dz^4).  These fields pin that prescribed force-only fields are checked by
// continuous Maxwell divergence, not rejected for ordinary stencil truncation.
class NonlinearAxisymmetricEvaluator final
    : public quasar::test::HostFieldEvaluator {
 public:
  explicit NonlinearAxisymmetricEvaluator(int axial_power)
      : axial_power_{axial_power} {}

  bool provides_grad_B() const noexcept override { return true; }

  quasar::Field<quasar::Vec3> host_B(
      const quasar::core::IFieldSource&,
      const quasar::core::PointCloud& observations) const override {
    quasar::Field<quasar::Vec3> out(observations.size());
    for (std::size_t i = 0; i < observations.size(); ++i) {
      const auto p = observations.points()[i];
      const double axial = integer_power(p.z, axial_power_ - 1);
      out[i] = quasar::Vec3{
          -axial_power_ * p.x * axial,
          -axial_power_ * p.y * axial,
          2.0 * integer_power(p.z, axial_power_)};
    }
    return out;
  }

  quasar::Field<quasar::Mat3x3> host_grad_B(
      const quasar::core::IFieldSource&,
      const quasar::core::PointCloud& observations) const override {
    quasar::Field<quasar::Mat3x3> out(observations.size());
    for (std::size_t i = 0; i < observations.size(); ++i) {
      const auto p = observations.points()[i];
      const double z_nm1 = integer_power(p.z, axial_power_ - 1);
      const double z_nm2 = integer_power(p.z, axial_power_ - 2);
      const double radial_diagonal = -axial_power_ * z_nm1;
      const double radial_axial =
          -axial_power_ * (axial_power_ - 1) * z_nm2;
      out[i] = quasar::Mat3x3{
          quasar::Vec3{radial_diagonal, 0.0, radial_axial * p.x},
          quasar::Vec3{0.0, radial_diagonal, radial_axial * p.y},
          quasar::Vec3{0.0, 0.0, 2.0 * axial_power_ * z_nm1}};
    }
    return out;
  }

 private:
  int axial_power_{};
};

// A divergence-free m=4 axial field on an annulus.  It is covariant under a
// quarter-turn but not under general rotations, so a C4-only check accepts it.
class FourfoldAxialEvaluator final
    : public quasar::test::HostFieldEvaluator {
 public:
  bool provides_grad_B() const noexcept override { return true; }

  quasar::Field<quasar::Vec3> host_B(
      const quasar::core::IFieldSource&,
      const quasar::core::PointCloud& observations) const override {
    quasar::Field<quasar::Vec3> out(observations.size());
    for (std::size_t i = 0; i < observations.size(); ++i) {
      const auto p = observations.points()[i];
      const double phi = std::atan2(p.y, p.x);
      out[i] = quasar::Vec3{0.0, 0.0, std::cos(4.0 * phi)};
    }
    return out;
  }

  quasar::Field<quasar::Mat3x3> host_grad_B(
      const quasar::core::IFieldSource&,
      const quasar::core::PointCloud& observations) const override {
    quasar::Field<quasar::Mat3x3> out(observations.size());
    for (std::size_t i = 0; i < observations.size(); ++i) {
      const auto p = observations.points()[i];
      const double radius_squared = p.x * p.x + p.y * p.y;
      const double sin_four_phi = std::sin(4.0 * std::atan2(p.y, p.x));
      out[i] = quasar::Mat3x3{
          quasar::Vec3{0.0, 0.0, 0.0},
          quasar::Vec3{0.0, 0.0, 0.0},
          quasar::Vec3{4.0 * p.y * sin_four_phi / radius_squared,
                       -4.0 * p.x * sin_four_phi / radius_squared, 0.0}};
    }
    return out;
  }
};

class ElectricOnlyEvaluator final
    : public quasar::test::HostFieldEvaluator {
 public:
  quasar::Field<quasar::Vec3> host_B(
      const quasar::core::IFieldSource&,
      const quasar::core::PointCloud& observations) const override {
    return quasar::Field<quasar::Vec3>{observations.size()};
  }

  quasar::Field<quasar::Vec3> host_E(
      const quasar::core::IFieldSource&,
      const quasar::core::PointCloud& observations) const override {
    quasar::Field<quasar::Vec3> out(observations.size());
    for (std::size_t i = 0; i < observations.size(); ++i) {
      out[i] = quasar::Vec3{0.0, 0.0, 1.0};
    }
    return out;
  }
};

class NonzeroMagneticWithoutGradientEvaluator final
    : public quasar::test::HostFieldEvaluator {
 public:
  quasar::Field<quasar::Vec3> host_B(
      const quasar::core::IFieldSource&,
      const quasar::core::PointCloud& observations) const override {
    quasar::Field<quasar::Vec3> out(observations.size());
    for (std::size_t i = 0; i < observations.size(); ++i) {
      out[i] = quasar::Vec3{0.0, 0.0, 1.0};
    }
    return out;
  }
};

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

TEST(PicExternalPlaneMapping, NonuniformFieldIsSampledOnEachYeeSublattice) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  const quasar::Mat3x3 grad{
      quasar::Vec3{1.0, 2.0, 3.0},
      quasar::Vec3{4.0, 2.0, 6.0},
      quasar::Vec3{7.0, 8.0, -3.0}};  // trace = 0
  quasar::analytic_fields::GradientEvaluator eval{
      quasar::Vec3{0.0, 0.0, 0.0}, grad};
  quasar::magnetostatics::ConductorSystem source;
  quasar::Grid2D g{8, 8, 0.8, 0.8, 0.0, 0.0, 2};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  quasar::pic::sample_external_field(eval, source, solver.external_fields());

  std::vector<quasar::Real> bx(g.storage_size()), by(g.storage_size());
  std::vector<quasar::Real> bz(g.storage_size());
  solver.external_fields().bx.copy_to_host(bx.data(), bx.size());
  solver.external_fields().by.copy_to_host(by.data(), by.size());
  solver.external_fields().bz.copy_to_host(bz.data(), bz.size());
  const int i = 2, j = 3;
  const double dx = g.dx(), dy = g.dy();
  // Cartesian Bx=(1/2,0), By=(0,1/2), Bz=(0,0).
  EXPECT_NEAR(bx[g.index(i, j)],
              1.0 * ((i + 0.5) * dx) + 2.0 * (j * dy), 1.0e-12);
  EXPECT_NEAR(by[g.index(i, j)],
              4.0 * (i * dx) + 2.0 * ((j + 0.5) * dy), 1.0e-12);
  EXPECT_NEAR(bz[g.index(i, j)],
              7.0 * (i * dx) + 8.0 * (j * dy), 1.0e-12);
  // Physical high faces/corner are part of the Yee lattice, not generic halo
  // cells to be copied from nx-1/ny-1.
  EXPECT_NEAR(by[g.index(g.nx, j)],
              4.0 * g.lx + 2.0 * ((j + 0.5) * dy), 1.0e-12);
  EXPECT_NEAR(bx[g.index(i, g.ny)],
              1.0 * ((i + 0.5) * dx) + 2.0 * g.ly, 1.0e-12);
  EXPECT_NEAR(bz[g.index(g.nx, g.ny)],
              7.0 * g.lx + 8.0 * g.ly, 1.0e-12);
  // True ghosts are evaluated at their own coordinates as well. Replicating an
  // edge value would fail each of these checks for the linear field.
  EXPECT_NEAR(bx[g.index(-1, j)],
              1.0 * (-0.5 * dx) + 2.0 * (j * dy), 1.0e-12);
  EXPECT_NEAR(by[g.index(g.nx + 1, j)],
              4.0 * ((g.nx + 1) * dx)
                  + 2.0 * ((j + 0.5) * dy), 1.0e-12);
  EXPECT_NEAR(bz[g.index(g.nx + 1, g.ny + 1)],
              7.0 * ((g.nx + 1) * dx)
                  + 8.0 * ((g.ny + 1) * dy), 1.0e-12);
}

TEST(PicExternalPlaneMapping, RejectsCoordinateAndFieldScalingUnderflow) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{8, 8, 1.0, 1.0, 0.0, 0.0, 1};
  quasar::pic::EmPic2D3V solver{quasar::pic::EmPicConfig{g, 2, "cic"}};
  quasar::magnetostatics::ConductorSystem source;
  const double tiny = std::numeric_limits<double>::denorm_min();
  const double huge = std::numeric_limits<double>::max();

  quasar::analytic_fields::UniformEvaluator zero;
  EXPECT_THROW(
      quasar::pic::sample_external_field(
          zero, source, solver.external_fields(), tiny, 1.0, 1.0),
      std::underflow_error);

  quasar::analytic_fields::UniformEvaluator tiny_b{
      quasar::Vec3{tiny, 0.0, 0.0}, quasar::Vec3{0.0, 0.0, 0.0}};
  EXPECT_THROW(
      quasar::pic::sample_external_field(
          tiny_b, source, solver.external_fields(), 1.0, 1.0, huge),
      std::underflow_error);
}

TEST(PicExternalPlaneMapping, ExtremeTranslatedYeeCoordinatesStayFinite) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // The high ghost of a half-offset component is
  //   -0.75*max + 4.5*(max/4) = 0.375*max.
  // Materialising 4.5*(max/4) first overflows; an affine FMA evaluates the
  // finite coordinate directly.  The low face ghost is exactly -max, so every
  // padded Yee point remains representable.  Uniform evaluation is enough to
  // exercise every sub-lattice without making the field itself range-sensitive.
  const double max = std::numeric_limits<double>::max();
  quasar::Grid2D g{4, 2, max, 2.0, -0.75 * max, 0.0, 1};
  quasar::YeeField2D<quasar::Real> external{g};
  quasar::analytic_fields::UniformEvaluator eval{
      quasar::Vec3{1.0, 2.0, 3.0}, quasar::Vec3{4.0, 5.0, 6.0}};
  quasar::magnetostatics::ConductorSystem source;

  EXPECT_NO_THROW(quasar::pic::sample_external_field(
      eval, source, external, 1.0, 1.0, 1.0, "xy", "cartesian"));
  EXPECT_DOUBLE_EQ(max_abs_diff(external.bx, 1.0), 0.0);
  EXPECT_DOUBLE_EQ(max_abs_diff(external.by, 2.0), 0.0);
  EXPECT_DOUBLE_EQ(max_abs_diff(external.bz, 3.0), 0.0);
}

TEST(PicExternalPlaneMapping, CylindricalAxisRejectsUniformRadialAndToroidalField) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{8, 8, 1.0, 1.0, 0.0, -0.5, 2};
  quasar::YeeField2D<quasar::Real> external{g};
  quasar::magnetostatics::ConductorSystem source;

  // In the cylindrical xz map, lab (x,y,z) becomes physical (r,phi,z).
  quasar::analytic_fields::UniformEvaluator radial{
      quasar::Vec3{1.0, 0.0, 0.0}, quasar::Vec3{0.0, 0.0, 0.0}};
  EXPECT_THROW(quasar::pic::sample_external_field(
                   radial, source, external, 1.0, 1.0, 1.0,
                   "xz", "cylindrical"),
               std::invalid_argument);

  quasar::analytic_fields::UniformEvaluator toroidal{
      quasar::Vec3{0.0, 1.0, 0.0}, quasar::Vec3{0.0, 0.0, 0.0}};
  EXPECT_THROW(quasar::pic::sample_external_field(
                   toroidal, source, external, 1.0, 1.0, 1.0,
                   "xz", "cylindrical"),
               std::invalid_argument);

  quasar::analytic_fields::UniformEvaluator axial{
      quasar::Vec3{0.0, 0.0, 1.0}, quasar::Vec3{0.0, 0.0, 0.0}};
  EXPECT_NO_THROW(quasar::pic::sample_external_field(
      axial, source, external, 1.0, 1.0, 1.0, "xz", "cylindrical"));
}

TEST(PicExternalPlaneMapping, CylindricalAxisAcceptsAndClosesRegularParity) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{8, 8, 1.0, 2.0, 0.0, -1.0, 2};
  quasar::YeeField2D<quasar::Real> external{g};
  quasar::magnetostatics::ConductorSystem source;
  // B=(x,y,-2z) is axisymmetric about lab z and exactly divergence-free.
  const quasar::Mat3x3 grad{
      quasar::Vec3{1.0, 0.0, 0.0},
      quasar::Vec3{0.0, 1.0, 0.0},
      quasar::Vec3{0.0, 0.0, -2.0}};
  quasar::analytic_fields::GradientEvaluator eval{
      quasar::Vec3{0.0, 0.0, 0.0}, grad};

  ASSERT_NO_THROW(quasar::pic::sample_external_field(
      eval, source, external, 1.0, 1.0, 1.0, "xz", "cylindrical"));

  std::vector<double> br(g.storage_size()), bz(g.storage_size());
  external.bx.copy_to_host(br.data(), br.size());
  external.by.copy_to_host(bz.data(), bz.size());
  for (int j = -g.nghost; j < g.ny + g.nghost; ++j) {
    EXPECT_DOUBLE_EQ(br[g.index(0, j)], 0.0);
    for (int gh = 1; gh <= g.nghost; ++gh) {
      EXPECT_DOUBLE_EQ(br[g.index(-gh, j)], -br[g.index(gh, j)]);
      EXPECT_DOUBLE_EQ(bz[g.index(-gh, j)], bz[g.index(gh - 1, j)]);
    }
  }
}

TEST(PicExternalPlaneMapping,
     CylindricalAnnulusRejectsNonAxisymmetricDivergenceFreeSlice) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // B=(x,-y,0) is divergence-free in Cartesian coordinates and its y=0 slice
  // even has regular-looking Br=r.  It is not covariant under rotations about
  // lab z, however, so treating that one slice as an axisymmetric field would
  // create the invalid cylindrical divergence (1/r)d(r^2)/dr=2.
  const quasar::Mat3x3 grad{
      quasar::Vec3{1.0, 0.0, 0.0},
      quasar::Vec3{0.0, -1.0, 0.0},
      quasar::Vec3{0.0, 0.0, 0.0}};
  quasar::analytic_fields::GradientEvaluator eval{
      quasar::Vec3{0.0, 0.0, 0.0}, grad};
  quasar::magnetostatics::ConductorSystem source;
  // Start away from the axis so this specifically exercises rotational
  // covariance rather than the separate r=0 parity closure.
  quasar::Grid2D g{8, 8, 1.0, 1.0, 0.25, -0.5, 2};
  quasar::YeeField2D<quasar::Real> external{g};
  std::vector<quasar::Real> sentinel(g.storage_size(), 7.0);
  external.ex.copy_from_host(sentinel.data(), sentinel.size());
  external.bx.copy_from_host(sentinel.data(), sentinel.size());

  EXPECT_THROW(quasar::pic::sample_external_field(
                   eval, source, external, 1.0, 1.0, 1.0,
                   "xz", "cylindrical"),
               std::invalid_argument);
  // Validation is transactional: a rejected field cannot replace E before a
  // later B check fails or otherwise leave a mixed old/new prescribed state.
  EXPECT_DOUBLE_EQ(max_abs_diff(external.ex, 7.0), 0.0);
  EXPECT_DOUBLE_EQ(max_abs_diff(external.bx, 7.0), 0.0);
}

TEST(PicExternalPlaneMapping,
     CylindricalRejectsAxisymmetricMagneticMonopoleOnYeeGrid) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  AxisymmetricMonopoleEvaluator eval;
  quasar::magnetostatics::ConductorSystem source;
  quasar::Grid2D g{8, 8, 1.0, 1.0, 0.0, -0.5, 2};
  quasar::YeeField2D<quasar::Real> external{g};

  EXPECT_THROW(quasar::pic::sample_external_field(
                   eval, source, external, 1.0, 1.0, 1.0,
                   "xz", "cylindrical"),
               std::invalid_argument);
}

TEST(PicExternalPlaneMapping,
     CylindricalXyPlaneAcceptsOrderFourAxisymmetricSolenoidalField) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // In plane="xy" the physical symmetry axis is lab y.  The field
  // B=(x,-2y,z) is covariant about that axis and divergence-free.  Its mapped
  // cylindrical components on lab z=0 are Br=r and Bz=-2z, whose fourth-order
  // staggered divergence cancels exactly.
  const quasar::Mat3x3 grad{
      quasar::Vec3{1.0, 0.0, 0.0},
      quasar::Vec3{0.0, -2.0, 0.0},
      quasar::Vec3{0.0, 0.0, 1.0}};
  quasar::analytic_fields::GradientEvaluator eval{
      quasar::Vec3{0.0, 0.0, 0.0}, grad};
  quasar::magnetostatics::ConductorSystem source;
  quasar::Grid2D g{8, 8, 1.0, 2.0, 0.0, -1.0, 2};
  quasar::YeeField2D<quasar::Real> external{g};

  EXPECT_NO_THROW(quasar::pic::sample_external_field(
      eval, source, external, 1.0, 1.0, 1.0,
      "xy", "cylindrical", 4));
}

TEST(PicExternalPlaneMapping,
     CylindricalAcceptsSmoothNonlinearContinuouslySolenoidalFields) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  struct Case {
    int fdtd_order;
    int axial_power;
    int nghost;
  };
  const Case cases[] = {{2, 3, 1}, {4, 5, 2}};
  quasar::magnetostatics::ConductorSystem source;
  for (const Case test_case : cases) {
    SCOPED_TRACE("fdtd_order=" + std::to_string(test_case.fdtd_order));
    quasar::Grid2D g{
        8, 8, 1.0, 1.0, 0.0, 0.0, test_case.nghost};
    quasar::YeeField2D<quasar::Real> external{g};
    NonlinearAxisymmetricEvaluator eval{test_case.axial_power};

    EXPECT_NO_THROW(quasar::pic::sample_external_field(
        eval, source, external, 1.0, 1.0, 1.0,
        "xz", "cylindrical", test_case.fdtd_order));
  }
}

TEST(PicExternalPlaneMapping,
     CylindricalRejectsFourfoldFieldThatIsNotAxisymmetric) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // Keep the complete padded lattice away from r=0: cos(4*phi) is smooth on
  // this annulus and passes an exact quarter-turn covariance test.
  quasar::Grid2D g{8, 8, 1.0, 1.0, 1.0, -0.5, 2};
  quasar::YeeField2D<quasar::Real> external{g};
  FourfoldAxialEvaluator eval;
  quasar::magnetostatics::ConductorSystem source;

  EXPECT_THROW(quasar::pic::sample_external_field(
                   eval, source, external, 1.0, 1.0, 1.0,
                   "xz", "cylindrical", 4),
               std::invalid_argument);
}

TEST(PicExternalPlaneMapping,
     CylindricalCovarianceDoesNotLetHugeAxialFieldHideTransverseError) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  // A uniform lab-y component is not covariant about lab z.  The old vector-wide
  // normalization divided its unit mismatch by the unrelated 1e100 axial field.
  quasar::Grid2D g{8, 8, 1.0, 1.0, 1.0, -0.5, 2};
  quasar::YeeField2D<quasar::Real> external{g};
  quasar::analytic_fields::UniformEvaluator eval{
      quasar::Vec3{0.0, 1.0, 1.0e100}};
  quasar::magnetostatics::ConductorSystem source;

  EXPECT_THROW(quasar::pic::sample_external_field(
                   eval, source, external, 1.0, 1.0, 1.0,
                   "xz", "cylindrical", 4),
               std::invalid_argument);
}

TEST(PicExternalPlaneMapping,
     CylindricalAllowsElectricOnlyEvaluatorWithoutMagneticGradient) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{8, 8, 1.0, 1.0, 0.0, -0.5, 1};
  quasar::YeeField2D<quasar::Real> external{g};
  ElectricOnlyEvaluator eval;
  quasar::magnetostatics::ConductorSystem source;

  ASSERT_NO_THROW(quasar::pic::sample_external_field(
      eval, source, external, 1.0, 1.0, 1.0,
      "xz", "cylindrical", 2));
  EXPECT_DOUBLE_EQ(max_abs_diff(external.ey, 1.0), 0.0);
  EXPECT_DOUBLE_EQ(max_abs_diff(external.bx, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(max_abs_diff(external.by, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(max_abs_diff(external.bz, 0.0), 0.0);
}

TEST(PicExternalPlaneMapping,
     CylindricalRejectsNonzeroMagneticEvaluatorWithoutGradientCapability) {
  if (!quasar::backend::has_hip_runtime()) GTEST_SKIP() << "no HIP runtime";

  quasar::Grid2D g{8, 8, 1.0, 1.0, 0.0, -0.5, 1};
  quasar::YeeField2D<quasar::Real> external{g};
  NonzeroMagneticWithoutGradientEvaluator eval;
  quasar::magnetostatics::ConductorSystem source;

  EXPECT_THROW(quasar::pic::sample_external_field(
                   eval, source, external, 1.0, 1.0, 1.0,
                   "xz", "cylindrical", 2),
               std::invalid_argument);
}
