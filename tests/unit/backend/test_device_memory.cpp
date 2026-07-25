#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"

#include <gtest/gtest.h>

#include <limits>

#include <cstddef>
#include <utility>
#include <vector>

using ::quasar::Real;
using ::quasar::backend::DeviceBuffer;
using ::quasar::backend::has_hip_runtime;
using ::quasar::backend::mirror_view;

namespace {

constexpr std::size_t kRoundTripN = 1024;
constexpr std::size_t kMirrorN    = 256;

}  // namespace

TEST(DeviceBuffer, RoundTripPattern) {
  if (!has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime visible";
  }

  std::vector<Real> host_src(kRoundTripN);
  for (std::size_t i = 0; i < kRoundTripN; ++i) {
    host_src[i] = static_cast<Real>(i) * Real{0.5};
  }

  DeviceBuffer<Real> buf(kRoundTripN);
  ASSERT_EQ(buf.size(), kRoundTripN);
  ASSERT_EQ(buf.bytes(), kRoundTripN * sizeof(Real));
  ASSERT_NE(buf.device_ptr(), nullptr);

  buf.copy_from_host(host_src.data(), kRoundTripN);

  std::vector<Real> host_dst(kRoundTripN, Real{-1});
  buf.copy_to_host(host_dst.data(), kRoundTripN);

  for (std::size_t i = 0; i < kRoundTripN; ++i) {
    EXPECT_EQ(host_dst[i], host_src[i]) << "mismatch at i=" << i;
  }
}

TEST(DeviceBuffer, RejectsByteSizeOverflowBeforeAllocation) {
  const std::size_t too_many =
      std::numeric_limits<std::size_t>::max() / sizeof(double) + 1;
  EXPECT_THROW((quasar::backend::DeviceBuffer<double>{too_many}),
               std::length_error);
  EXPECT_THROW((quasar::backend::DeviceBuffer<double>{
                   too_many, quasar::backend::uninitialized}),
               std::length_error);
}

TEST(DeviceMemoryPrimitives, ZeroByteAllocationIsPortableAndDoesNotNeedADevice) {
  EXPECT_EQ(quasar::backend::device_alloc(0), nullptr);
  EXPECT_EQ(quasar::backend::device_alloc_uninit(0), nullptr);
}

TEST(DeviceMemoryPrimitives, ZeroByteOperationsArePortableAndDoNotNeedADevice) {
  using namespace quasar::backend;
  EXPECT_NO_THROW(device_memset(nullptr, 0, 0));
  EXPECT_NO_THROW(device_memset_async(nullptr, 0, 0, nullptr));
  EXPECT_NO_THROW(device_memcpy_h2d(nullptr, nullptr, 0));
  EXPECT_NO_THROW(device_memcpy_d2h(nullptr, nullptr, 0));
  EXPECT_NO_THROW(device_memcpy_h2d_async(nullptr, nullptr, 0, nullptr));
  EXPECT_NO_THROW(device_memcpy_d2h_async(nullptr, nullptr, 0, nullptr));
}

TEST(DeviceMemoryPrimitives, CheckedSizeProductRejectsOverflow) {
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(quasar::backend::detail::checked_size_product(7, 3, "unused"), 21u);
  EXPECT_THROW(
      (void)quasar::backend::detail::checked_size_product(
          maximum / 3 + 1, 3, "scratch element count is not representable"),
      std::length_error);
}

TEST(DeviceBuffer, RejectsOutOfBoundsAndNullHostCopies) {
  if (!has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime visible";
  }
  quasar::backend::DeviceBuffer<double> buffer{4};
  double host[5]{};
  EXPECT_THROW(buffer.copy_from_host(host, 5), std::out_of_range);
  EXPECT_THROW(buffer.copy_to_host(host, 5), std::out_of_range);
  EXPECT_THROW(buffer.copy_from_host(nullptr, 1), std::invalid_argument);
  EXPECT_THROW(buffer.copy_to_host(nullptr, 1), std::invalid_argument);
  EXPECT_NO_THROW(buffer.copy_from_host(nullptr, 0));
  EXPECT_NO_THROW(buffer.copy_to_host(nullptr, 0));
}

TEST(DeviceBuffer, EmptyConstructionDoesNotAllocate) {
  DeviceBuffer<int> buf;
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_EQ(buf.bytes(), 0u);
  EXPECT_TRUE(buf.empty());
  EXPECT_EQ(buf.device_ptr(), nullptr);
}

TEST(DeviceBuffer, MoveOnlySemantics) {
  if (!has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime visible";
  }

  DeviceBuffer<int> a(128);
  int* raw_ptr_a = a.device_ptr();
  ASSERT_NE(raw_ptr_a, nullptr);

  DeviceBuffer<int> b{std::move(a)};
  EXPECT_EQ(a.size(), 0u);
  EXPECT_EQ(a.device_ptr(), nullptr);
  EXPECT_EQ(b.size(), 128u);
  EXPECT_EQ(b.device_ptr(), raw_ptr_a);

  DeviceBuffer<int> c;
  c = std::move(b);
  EXPECT_EQ(b.size(), 0u);
  EXPECT_EQ(b.device_ptr(), nullptr);
  EXPECT_EQ(c.size(), 128u);
  EXPECT_EQ(c.device_ptr(), raw_ptr_a);
}

TEST(MirrorView, SyncsHostAndDeviceBidirectionally) {
  if (!has_hip_runtime()) {
    GTEST_SKIP() << "no HIP runtime visible";
  }

  mirror_view<Real> mv(kMirrorN);
  ASSERT_EQ(mv.size(), kMirrorN);
  ASSERT_NE(mv.device_ptr(), nullptr);

  for (std::size_t i = 0; i < kMirrorN; ++i) {
    mv.host_data()[i] = Real{1} + static_cast<Real>(i);
  }

  mv.sync_to_device();

  // Wipe the host side so any post-sync host read must have come from the device.
  for (std::size_t i = 0; i < kMirrorN; ++i) {
    mv.host_data()[i] = Real{0};
  }

  mv.sync_to_host();

  for (std::size_t i = 0; i < kMirrorN; ++i) {
    EXPECT_EQ(mv.host_data()[i], Real{1} + static_cast<Real>(i))
        << "mismatch at i=" << i;
  }
}
