#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/field_evaluator.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/physics/magnetostatics/kernels.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/core/device_observations.hpp"
#include "quasar/core/field.hpp"
#include "quasar/core/types.hpp"

#include <cstddef>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// The biot_savart launch ABI (declared once in
// include/quasar/physics/magnetostatics/kernels.hpp) speaks the backend-neutral
// stream handle, so this orchestrator needs no HIP header.
using stream_t = ::quasar::backend::stream_t;

namespace quasar::magnetostatics {

namespace {

int checked_kernel_count(std::size_t n, const char* what) {
  constexpr auto max_count = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (n > max_count) {
    throw std::length_error{
        std::string{"BiotSavartEvaluator: "} + what
        + " exceeds the signed kernel-index limit"};
  }
  return static_cast<int>(n);
}

std::size_t checked_staging_size(int count, std::size_t components,
                                 const char* what) {
  const std::size_t n = static_cast<std::size_t>(count);
  if (n != 0 && components > std::numeric_limits<std::size_t>::max() / n) {
    throw std::length_error{
        std::string{"BiotSavartEvaluator: "} + what
        + " exceeds the host size limit"};
  }
  return n * components;
}

template <class T>
void dispatch_launch_B(const T* ax, const T* ay, const T* az,
                       const T* bx, const T* by, const T* bz,
                       const T* I_, int N,
                       const T* px, const T* py, const T* pz, int M,
                       T* Bx, T* By, T* Bz, int* status,
                       stream_t stream) {
  if constexpr (std::is_same_v<T, double>) {
    ::launch_biot_savart_B_f64(ax, ay, az, bx, by, bz, I_, N,
                                px, py, pz, M, Bx, By, Bz, status, stream);
  } else {
    ::launch_biot_savart_B_f32(ax, ay, az, bx, by, bz, I_, N,
                                px, py, pz, M, Bx, By, Bz, status, stream);
  }
}

template <class T>
void dispatch_launch_A(const T* ax, const T* ay, const T* az,
                       const T* bx, const T* by, const T* bz,
                       const T* I_, int N,
                       const T* px, const T* py, const T* pz, int M,
                       T* Ax, T* Ay, T* Az, int* status,
                       stream_t stream) {
  if constexpr (std::is_same_v<T, double>) {
    ::launch_biot_savart_A_f64(ax, ay, az, bx, by, bz, I_, N,
                                px, py, pz, M, Ax, Ay, Az, status, stream);
  } else {
    ::launch_biot_savart_A_f32(ax, ay, az, bx, by, bz, I_, N,
                                px, py, pz, M, Ax, Ay, Az, status, stream);
  }
}

template <class T>
void dispatch_launch_gradB(const T* ax, const T* ay, const T* az,
                           const T* bx, const T* by, const T* bz,
                           const T* I_, int N,
                           const T* px, const T* py, const T* pz, int M,
                           T* G, int* status,
                           stream_t stream) {
  if constexpr (std::is_same_v<T, double>) {
    ::launch_biot_savart_gradB_f64(ax, ay, az, bx, by, bz, I_, N,
                                    px, py, pz, M, G, status, stream);
  } else {
    ::launch_biot_savart_gradB_f32(ax, ay, az, bx, by, bz, I_, N,
                                    px, py, pz, M, G, status, stream);
  }
}

// Downcast the axis-neutral source to the conductor system this evaluator needs.
const ConductorSystem& as_conductors(const core::IFieldSource& source) {
  const auto* cs = dynamic_cast<const ConductorSystem*>(&source);
  if (cs == nullptr) {
    throw std::invalid_argument{
        "BiotSavartEvaluator: field source is not a ConductorSystem"};
  }
  return *cs;
}

// -- fp64 device path -------------------------------------------------------
//
// Every one of these is: validate, upload the coil, launch, read back one
// status int. The outputs stay where the kernel wrote them.

int check_status(const ::quasar::backend::DeviceBuffer<int>& d_status,
                 stream_t stream, const char* singular_message,
                 const char* overflow_message) {
  int status = 0;
  d_status.copy_to_host_async(&status, 1, stream);
  ::quasar::backend::device_synchronize(stream);
  if ((status & 1) != 0) throw std::domain_error{singular_message};
  if ((status & 2) != 0) throw std::overflow_error{overflow_message};
  return status;
}

}  // namespace

// -- Double-precision evaluator --------------------------------------------

BiotSavartEvaluator::BiotSavartEvaluator() = default;
BiotSavartEvaluator::BiotSavartEvaluator(BiotSavartConfig cfg) : cfg_{cfg} {}

core::DeviceVectorField BiotSavartEvaluator::evaluate_B(
    const core::IFieldSource& source,
    const core::DevicePointCloud& obs) const {
  using ::quasar::backend::DeviceBuffer;
  // The conductor system flattens and validates its segments on the device and
  // caches the result, so there is nothing to upload here: both operands are
  // already resident.
  const DeviceSegmentSoA& seg = as_conductors(source).device_segments();
  const int N = checked_kernel_count(seg.n_segments(), "segment count");
  const int M = checked_kernel_count(obs.size(), "observation count");

  // No segments or no points means no field; the kernel would not write, so the
  // zero-filled default constructor is the answer.
  if (N == 0 || M == 0) return core::DeviceVectorField(obs.size());

  // The kernel writes every observation-point entry, so skip the zero-fill.
  core::DeviceVectorField out(obs.size(), ::quasar::backend::uninitialized);
  DeviceBuffer<int> d_status(1);  // zero-initialized bit field

  dispatch_launch_B<Real>(
      seg.ax.device_ptr(), seg.ay.device_ptr(), seg.az.device_ptr(),
      seg.bx.device_ptr(), seg.by.device_ptr(), seg.bz.device_ptr(),
      seg.I.device_ptr(), N,
      obs.x(), obs.y(), obs.z(), M,
      out.x(), out.y(), out.z(), d_status.device_ptr(), cfg_.stream);

  check_status(d_status, cfg_.stream,
               "BiotSavartEvaluator: ideal-filament field is singular at an "
               "observation point",
               "BiotSavartEvaluator: magnetic field is not representable in "
               "the working precision");
  return out;
}

core::DeviceVectorField BiotSavartEvaluator::evaluate_A(
    const core::IFieldSource& source,
    const core::DevicePointCloud& obs) const {
  using ::quasar::backend::DeviceBuffer;
  // The conductor system flattens and validates its segments on the device and
  // caches the result, so there is nothing to upload here: both operands are
  // already resident.
  const DeviceSegmentSoA& seg = as_conductors(source).device_segments();
  const int N = checked_kernel_count(seg.n_segments(), "segment count");
  const int M = checked_kernel_count(obs.size(), "observation count");

  if (N == 0 || M == 0) return core::DeviceVectorField(obs.size());

  core::DeviceVectorField out(obs.size(), ::quasar::backend::uninitialized);
  DeviceBuffer<int> d_status(1);

  dispatch_launch_A<Real>(
      seg.ax.device_ptr(), seg.ay.device_ptr(), seg.az.device_ptr(),
      seg.bx.device_ptr(), seg.by.device_ptr(), seg.bz.device_ptr(),
      seg.I.device_ptr(), N,
      obs.x(), obs.y(), obs.z(), M,
      out.x(), out.y(), out.z(), d_status.device_ptr(), cfg_.stream);

  check_status(d_status, cfg_.stream,
               "BiotSavartEvaluator: ideal-filament vector potential is "
               "singular at an observation point",
               "BiotSavartEvaluator: vector potential is not representable in "
               "the working precision");
  return out;
}

core::DeviceTensorField BiotSavartEvaluator::evaluate_grad_B(
    const core::IFieldSource& source,
    const core::DevicePointCloud& obs) const {
  using ::quasar::backend::DeviceBuffer;
  // The conductor system flattens and validates its segments on the device and
  // caches the result, so there is nothing to upload here: both operands are
  // already resident.
  const DeviceSegmentSoA& seg = as_conductors(source).device_segments();
  const int N = checked_kernel_count(seg.n_segments(), "segment count");
  const int M = checked_kernel_count(obs.size(), "observation count");

  if (N == 0 || M == 0) return core::DeviceTensorField(obs.size());

  // The gradient kernel already writes the component-major 9*M layout that
  // DeviceTensorField documents, so it fills the container directly.
  core::DeviceTensorField out(obs.size(), ::quasar::backend::uninitialized);
  DeviceBuffer<int> d_status(1);

  dispatch_launch_gradB<Real>(
      seg.ax.device_ptr(), seg.ay.device_ptr(), seg.az.device_ptr(),
      seg.bx.device_ptr(), seg.by.device_ptr(), seg.bz.device_ptr(),
      seg.I.device_ptr(), N,
      obs.x(), obs.y(), obs.z(), M,
      out.data(), d_status.device_ptr(), cfg_.stream);

  check_status(d_status, cfg_.stream,
               "BiotSavartEvaluator: ideal-filament field gradient is singular "
               "at an observation point",
               "BiotSavartEvaluator: magnetic-field gradient is not "
               "representable in the working precision");
  return out;
}

// -- Single-precision evaluator --------------------------------------------
//
// This one is deliberately NOT on IFieldEvaluator: it exists so a test can put
// the fp32 and fp64 answers side by side, and its results are fp32 host values
// at that comparison boundary. It therefore keeps the host staging the fp64
// path shed -- there is no DeviceVectorField in float, and inventing one for a
// single test consumer would be abstraction without a second user.
//
// The narrowing itself is on the device. Both operands arrive as fp64 device
// planes, and a common double-precision origin is subtracted before the cast so
// a rigid translation (x=0 to x=1e8 m) is invisible to the narrowing rather
// than collapsing a short segment into one float coordinate. That origin is the
// first segment endpoint, which is three scalars read back from the device.

namespace {

struct NarrowedF {
  ::quasar::backend::DeviceBuffer<float> ax, ay, az, bx, by, bz, I;
  ::quasar::backend::DeviceBuffer<float> px, py, pz;
};

// Reads one element from a device plane. Three of these per evaluation, to
// establish the fp32 origin; not a sweep.
Real first_element(const ::quasar::backend::DeviceBuffer<Real>& plane) {
  Real value = Real{0};
  plane.copy_to_host(&value, 1);
  return value;
}

NarrowedF narrow_inputs(const DeviceSegmentSoA& seg,
                        const core::DevicePointCloud& obs, int N, int M,
                        stream_t stream) {
  using ::quasar::backend::DeviceBuffer;
  using ::quasar::backend::uninitialized;

  const Real ox = first_element(seg.ax);
  const Real oy = first_element(seg.ay);
  const Real oz = first_element(seg.az);

  NarrowedF out{
      DeviceBuffer<float>(N, uninitialized), DeviceBuffer<float>(N, uninitialized),
      DeviceBuffer<float>(N, uninitialized), DeviceBuffer<float>(N, uninitialized),
      DeviceBuffer<float>(N, uninitialized), DeviceBuffer<float>(N, uninitialized),
      DeviceBuffer<float>(N, uninitialized), DeviceBuffer<float>(M, uninitialized),
      DeviceBuffer<float>(M, uninitialized), DeviceBuffer<float>(M, uninitialized)};

  DeviceBuffer<int> segment_status(1);
  ::launch_ms_narrow_segments(
      seg.ax.device_ptr(), seg.ay.device_ptr(), seg.az.device_ptr(),
      seg.bx.device_ptr(), seg.by.device_ptr(), seg.bz.device_ptr(),
      seg.I.device_ptr(), N, ox, oy, oz,
      out.ax.device_ptr(), out.ay.device_ptr(), out.az.device_ptr(),
      out.bx.device_ptr(), out.by.device_ptr(), out.bz.device_ptr(),
      out.I.device_ptr(), segment_status.device_ptr(), stream);

  DeviceBuffer<int> point_status(1);
  ::launch_ms_narrow_points(
      obs.x(), obs.y(), obs.z(), M, ox, oy, oz,
      out.px.device_ptr(), out.py.device_ptr(), out.pz.device_ptr(),
      point_status.device_ptr(), stream);

  int segment_flags = 0;
  int point_flags = 0;
  segment_status.copy_to_host_async(&segment_flags, 1, stream);
  point_status.copy_to_host_async(&point_flags, 1, stream);
  ::quasar::backend::device_synchronize(stream);

  if ((segment_flags & 1) != 0) {
    throw std::invalid_argument{
        "BiotSavartEvaluatorF: segment coordinate is not representable after "
        "fp32 origin shifting"};
  }
  if ((point_flags & 1) != 0) {
    throw std::invalid_argument{
        "BiotSavartEvaluatorF: observation coordinate is not representable "
        "after fp32 origin shifting"};
  }
  if ((segment_flags & 2) != 0) {
    throw std::invalid_argument{
        "BiotSavartEvaluatorF: a segment collapses after fp32 narrowing"};
  }
  return out;
}

}  // namespace

BiotSavartEvaluatorF::BiotSavartEvaluatorF() = default;
BiotSavartEvaluatorF::BiotSavartEvaluatorF(BiotSavartConfig cfg) : cfg_{cfg} {}

Field<Vec3f> BiotSavartEvaluatorF::evaluate_B(
    const ConductorSystem& cs, const core::DevicePointCloud& obs) const {
  using ::quasar::backend::DeviceBuffer;
  const DeviceSegmentSoA& seg = cs.device_segments();
  const int N = checked_kernel_count(seg.n_segments(), "segment count");
  const int M = checked_kernel_count(obs.size(), "observation count");

  Field<Vec3f> result(static_cast<std::size_t>(M));
  if (N == 0 || M == 0) {
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = Vec3f{0.0F, 0.0F, 0.0F};
    }
    return result;
  }

  const NarrowedF in = narrow_inputs(seg, obs, N, M, cfg_.stream);
  DeviceBuffer<float> d_Bx(M, ::quasar::backend::uninitialized);
  DeviceBuffer<float> d_By(M, ::quasar::backend::uninitialized);
  DeviceBuffer<float> d_Bz(M, ::quasar::backend::uninitialized);
  DeviceBuffer<int> d_status(1);

  dispatch_launch_B<float>(
      in.ax.device_ptr(), in.ay.device_ptr(), in.az.device_ptr(),
      in.bx.device_ptr(), in.by.device_ptr(), in.bz.device_ptr(),
      in.I.device_ptr(), N,
      in.px.device_ptr(), in.py.device_ptr(), in.pz.device_ptr(), M,
      d_Bx.device_ptr(), d_By.device_ptr(), d_Bz.device_ptr(),
      d_status.device_ptr(), cfg_.stream);

  const std::size_t MM = static_cast<std::size_t>(M);
  std::vector<float> hB(checked_staging_size(M, 3, "vector staging size"));
  d_Bx.copy_to_host_async(hB.data(),          M, cfg_.stream);
  d_By.copy_to_host_async(hB.data() + MM,     M, cfg_.stream);
  d_Bz.copy_to_host_async(hB.data() + 2 * MM, M, cfg_.stream);
  check_status(d_status, cfg_.stream,
               "BiotSavartEvaluator: ideal-filament field is singular at an "
               "observation point",
               "BiotSavartEvaluator: magnetic field is not representable in "
               "the working precision");

  for (std::size_t i = 0; i < MM; ++i) {
    result[i] = Vec3f{hB[i], hB[MM + i], hB[2 * MM + i]};
  }
  return result;
}

Field<Vec3f> BiotSavartEvaluatorF::evaluate_A(
    const ConductorSystem& cs, const core::DevicePointCloud& obs) const {
  using ::quasar::backend::DeviceBuffer;
  const DeviceSegmentSoA& seg = cs.device_segments();
  const int N = checked_kernel_count(seg.n_segments(), "segment count");
  const int M = checked_kernel_count(obs.size(), "observation count");

  Field<Vec3f> result(static_cast<std::size_t>(M));
  if (N == 0 || M == 0) {
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = Vec3f{0.0F, 0.0F, 0.0F};
    }
    return result;
  }

  const NarrowedF in = narrow_inputs(seg, obs, N, M, cfg_.stream);
  DeviceBuffer<float> d_Ax(M, ::quasar::backend::uninitialized);
  DeviceBuffer<float> d_Ay(M, ::quasar::backend::uninitialized);
  DeviceBuffer<float> d_Az(M, ::quasar::backend::uninitialized);
  DeviceBuffer<int> d_status(1);

  dispatch_launch_A<float>(
      in.ax.device_ptr(), in.ay.device_ptr(), in.az.device_ptr(),
      in.bx.device_ptr(), in.by.device_ptr(), in.bz.device_ptr(),
      in.I.device_ptr(), N,
      in.px.device_ptr(), in.py.device_ptr(), in.pz.device_ptr(), M,
      d_Ax.device_ptr(), d_Ay.device_ptr(), d_Az.device_ptr(),
      d_status.device_ptr(), cfg_.stream);

  const std::size_t MM = static_cast<std::size_t>(M);
  std::vector<float> hA(checked_staging_size(M, 3, "vector staging size"));
  d_Ax.copy_to_host_async(hA.data(),          M, cfg_.stream);
  d_Ay.copy_to_host_async(hA.data() + MM,     M, cfg_.stream);
  d_Az.copy_to_host_async(hA.data() + 2 * MM, M, cfg_.stream);
  check_status(d_status, cfg_.stream,
               "BiotSavartEvaluator: ideal-filament vector potential is "
               "singular at an observation point",
               "BiotSavartEvaluator: vector potential is not representable in "
               "the working precision");

  for (std::size_t i = 0; i < MM; ++i) {
    result[i] = Vec3f{hA[i], hA[MM + i], hA[2 * MM + i]};
  }
  return result;
}

Field<Mat3x3f> BiotSavartEvaluatorF::evaluate_grad_B(
    const ConductorSystem& cs, const core::DevicePointCloud& obs) const {
  using ::quasar::backend::DeviceBuffer;
  const DeviceSegmentSoA& seg = cs.device_segments();
  const int N = checked_kernel_count(seg.n_segments(), "segment count");
  const int M = checked_kernel_count(obs.size(), "observation count");

  Field<Mat3x3f> result(static_cast<std::size_t>(M));
  if (N == 0 || M == 0) {
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = Mat3x3f{};
    return result;
  }

  const NarrowedF in = narrow_inputs(seg, obs, N, M, cfg_.stream);
  const std::size_t gradient_values =
      checked_staging_size(M, 9, "gradient staging size");
  DeviceBuffer<float> d_G(gradient_values, ::quasar::backend::uninitialized);
  DeviceBuffer<int> d_status(1);

  dispatch_launch_gradB<float>(
      in.ax.device_ptr(), in.ay.device_ptr(), in.az.device_ptr(),
      in.bx.device_ptr(), in.by.device_ptr(), in.bz.device_ptr(),
      in.I.device_ptr(), N,
      in.px.device_ptr(), in.py.device_ptr(), in.pz.device_ptr(), M,
      d_G.device_ptr(), d_status.device_ptr(), cfg_.stream);

  std::vector<float> hG(gradient_values);
  d_G.copy_to_host_async(hG.data(), hG.size(), cfg_.stream);
  check_status(d_status, cfg_.stream,
               "BiotSavartEvaluator: ideal-filament field gradient is singular "
               "at an observation point",
               "BiotSavartEvaluator: magnetic-field gradient is not "
               "representable in the working precision");

  const std::size_t MM = static_cast<std::size_t>(M);
  for (std::size_t mi = 0; mi < MM; ++mi) {
    Mat3x3f g;
    g.r0 = Vec3f{hG[0 * MM + mi], hG[1 * MM + mi], hG[2 * MM + mi]};
    g.r1 = Vec3f{hG[3 * MM + mi], hG[4 * MM + mi], hG[5 * MM + mi]};
    g.r2 = Vec3f{hG[6 * MM + mi], hG[7 * MM + mi], hG[8 * MM + mi]};
    result[mi] = g;
  }
  return result;
}

}  // namespace quasar::magnetostatics

QUASAR_REGISTER_FIELD_EVALUATOR("biot_savart",
                                 ::quasar::magnetostatics::BiotSavartEvaluator)
