#include "quasar/backend/device.hpp"
#include "quasar/core/yee_field.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

TEST(YeeField2D, DefaultObjectReportsNoAllocatedValues) {
  const quasar::YeeField2D<double> fields;
  const quasar::JField2D<double> current;
  EXPECT_EQ(fields.component_size(), 0u);
  EXPECT_EQ(fields.total_values(), 0u);
  EXPECT_EQ(current.component_size(), 0u);
  EXPECT_EQ(current.total_values(), 0u);
}

TEST(YeeField2D, AllocatesSixComponents) {
  if (!quasar::backend::has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime";
  }
  const quasar::Grid2D g{4, 3, 1.0, 2.0, 0.0, 0.0, 2};
  quasar::YeeField2D<double> fields{g};
  EXPECT_EQ(fields.component_size(), g.storage_size());
  EXPECT_EQ(fields.total_values(), 6 * g.storage_size());
  quasar::JField2D<double> current{g};
  EXPECT_EQ(current.total_values(), 3 * g.storage_size());
}

TEST(YeeField2D, CartesianLogicalExtentsIncludePhysicalHighFaces) {
  const quasar::Grid2D g{4, 3, 1.0, 1.0, 0.0, 0.0, 2};
  EXPECT_EQ(quasar::CartesianYeeLayout2D::ex(g).nx, 5);
  EXPECT_EQ(quasar::CartesianYeeLayout2D::ex(g).ny, 3);
  EXPECT_EQ(quasar::CartesianYeeLayout2D::ey(g).nx, 4);
  EXPECT_EQ(quasar::CartesianYeeLayout2D::ey(g).ny, 4);
  EXPECT_EQ(quasar::CartesianYeeLayout2D::ez(g).nx, 4);
  EXPECT_EQ(quasar::CartesianYeeLayout2D::ez(g).ny, 3);
  EXPECT_EQ(quasar::CartesianYeeLayout2D::bx(g).ny, 4);
  EXPECT_EQ(quasar::CartesianYeeLayout2D::by(g).nx, 5);
  EXPECT_EQ(quasar::CartesianYeeLayout2D::bz(g).nx, 5);
  EXPECT_EQ(quasar::CartesianYeeLayout2D::bz(g).ny, 4);
}

TEST(YeeField2D, CylindricalLogicalExtentsFollowRadialParity) {
  const quasar::Grid2D g{4, 3, 1.0, 1.0, 0.0, 0.0, 2};
  EXPECT_EQ(quasar::CylindricalYeeLayout2D::ex(g).nx, 5);  // Er
  EXPECT_EQ(quasar::CylindricalYeeLayout2D::ey(g).ny, 4);  // Ez
  EXPECT_EQ(quasar::CylindricalYeeLayout2D::ez(g).nx, 5);  // Ephi
  EXPECT_EQ(quasar::CylindricalYeeLayout2D::bx(g).nx, 5);  // Br
  EXPECT_EQ(quasar::CylindricalYeeLayout2D::bx(g).ny, 4);
  EXPECT_EQ(quasar::CylindricalYeeLayout2D::by(g).nx, 4);  // Bz
  EXPECT_EQ(quasar::CylindricalYeeLayout2D::by(g).ny, 3);
  EXPECT_EQ(quasar::CylindricalYeeLayout2D::bz(g).nx, 5);  // Bphi
  EXPECT_EQ(quasar::CylindricalYeeLayout2D::bz(g).ny, 4);
}

TEST(YeeField2D, RejectsAZeroHaloBeforeAnyDeviceAllocation) {
  const quasar::Grid2D no_halo{4, 3, 1.0, 1.0, 0.0, 0.0, 0};
  EXPECT_THROW((quasar::YeeField2D<double>{no_halo}), std::invalid_argument);
  EXPECT_THROW((quasar::JField2D<double>{no_halo}), std::invalid_argument);
  EXPECT_THROW((quasar::BField2D<double>{no_halo}), std::invalid_argument);
  EXPECT_THROW((quasar::HostYeeField2D<double>{no_halo}), std::invalid_argument);
}

TEST(YeeField2D, RejectsAggregateSizeOverflowBeforeAnyDeviceAllocation) {
  // Each char component is individually representable, but six components are
  // not. The first checked initializer must reject this before touching HIP.
  const quasar::Grid2D six_component_overflow{
      1'800'000'000, 1'800'000'000, 1'800'000'000.0, 1'800'000'000.0,
      0.0, 0.0, 1};
  EXPECT_THROW((quasar::YeeField2D<unsigned char>{six_component_overflow}),
               std::length_error);
  EXPECT_THROW((quasar::HostYeeField2D<unsigned char>{six_component_overflow}),
               std::length_error);

  // Three component counts fit size_t on this grid, but their aggregate double
  // payload does not. J/B construction must reject that aggregate as well.
  EXPECT_THROW((quasar::JField2D<double>{six_component_overflow}),
               std::length_error);
  EXPECT_THROW((quasar::BField2D<double>{six_component_overflow}),
               std::length_error);
}
