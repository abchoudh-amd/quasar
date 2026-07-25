#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/field_evaluator.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/physics/magnetostatics/kernels.hpp"
#include "quasar/backend/memory.hpp"
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

// Provides an upload-ready const T* for a host-side std::vector<Real>. When T is
// Real (double) it aliases the source directly (no copy); for float it owns a
// narrowed copy. Stored at function scope so the pointer stays valid across the
// async H2D copy until the stream sync.
template <class T>
class UploadSrc {
 public:
  explicit UploadSrc(const std::vector<Real>& src, Real origin = Real{0},
                     const char* label = "input") {
    if constexpr (std::is_same_v<T, Real>) {
      view_ = &src;
    } else {
      owned_.reserve(src.size());
      for (Real v : src) {
        const T narrowed = static_cast<T>(v - origin);
        if (!std::isfinite(narrowed)) {
          throw std::invalid_argument{
              std::string{"BiotSavartEvaluatorF: "} + label
              + " is not representable after fp32 origin shifting"};
        }
        owned_.push_back(narrowed);
      }
      view_ = &owned_;
    }
  }
  const T* data() const noexcept { return view_->data(); }
  const T& operator[](std::size_t i) const noexcept { return (*view_)[i]; }

 private:
  std::vector<T> owned_{};
  const std::vector<T>* view_{nullptr};
};

int checked_kernel_count(std::size_t n, const char* what) {
  constexpr auto max_count = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (n > max_count) {
    throw std::length_error{
        std::string{"BiotSavartEvaluator: "} + what
        + " exceeds the signed kernel-index limit"};
  }
  return static_cast<int>(n);
}

struct KernelCounts {
  int segments{0};
  int points{0};
};

KernelCounts checked_input_counts(const SegmentSoA& seg, const PointSoA& pts) {
  // Validate every public SoA plane before sizing a device buffer from only the
  // leading component. Otherwise a shorter plane would be read past its host
  // allocation by the asynchronous upload.
  seg.validate();
  pts.validate();
  return {checked_kernel_count(seg.n_segments(), "segment count"),
          checked_kernel_count(pts.n_points(), "observation count")};
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

// Device-resident conductor + observation inputs for the Biot-Savart kernels.
// Owns the segment SoA (a/b endpoints + current) and the observation points on
// the device, uploaded asynchronously on cfg.stream. Both evaluate_*_impl build
// one of these, then allocate only their own output buffer.
template <class T>
struct UploadedInputs {
  using DeviceBuffer = ::quasar::backend::DeviceBuffer<T>;
  DeviceBuffer ax, ay, az, bx, by, bz, I, px, py, pz;
  int N{0};
  int M{0};

  UploadedInputs(const SegmentSoA& seg, const PointSoA& pts,
                 int n_segments, int n_points, stream_t stream)
      : ax(seg.n_segments()), ay(seg.n_segments()), az(seg.n_segments()),
        bx(seg.n_segments()), by(seg.n_segments()), bz(seg.n_segments()),
        I(seg.n_segments()),
        px(pts.n_points()), py(pts.n_points()), pz(pts.n_points()),
        N(n_segments), M(n_points) {
    // UploadSrc keeps each narrowed/aliased host buffer alive until the sync that
    // the caller performs after the kernel launch.
    // fp32 inputs are translated by one common origin while still in double
    // precision. This makes a rigid translation (for example, from x=0 to
    // x=1e8 m) invisible to the narrowing conversion instead of collapsing a
    // short segment into one float coordinate.
    Real ox = Real{0}, oy = Real{0}, oz = Real{0};
    if constexpr (!std::is_same_v<T, Real>) {
      ox = seg.ax.front();
      oy = seg.ay.front();
      oz = seg.az.front();
    }
    const UploadSrc<T> s_ax{seg.ax, ox, "segment x-coordinate"};
    const UploadSrc<T> s_ay{seg.ay, oy, "segment y-coordinate"};
    const UploadSrc<T> s_az{seg.az, oz, "segment z-coordinate"};
    const UploadSrc<T> s_bx{seg.bx, ox, "segment x-coordinate"};
    const UploadSrc<T> s_by{seg.by, oy, "segment y-coordinate"};
    const UploadSrc<T> s_bz{seg.bz, oz, "segment z-coordinate"};
    const UploadSrc<T> s_I{seg.I, Real{0}, "current"};
    const UploadSrc<T> s_px{pts.px, ox, "observation x-coordinate"};
    const UploadSrc<T> s_py{pts.py, oy, "observation y-coordinate"};
    const UploadSrc<T> s_pz{pts.pz, oz, "observation z-coordinate"};

    if constexpr (!std::is_same_v<T, Real>) {
      for (std::size_t i = 0; i < seg.n_segments(); ++i) {
        if (s_ax[i] == s_bx[i] && s_ay[i] == s_by[i] && s_az[i] == s_bz[i]) {
          throw std::invalid_argument{
              "BiotSavartEvaluatorF: a segment collapses after fp32 narrowing"};
        }
      }
    }
    ax.copy_from_host_async(s_ax.data(), N, stream);
    ay.copy_from_host_async(s_ay.data(), N, stream);
    az.copy_from_host_async(s_az.data(), N, stream);
    bx.copy_from_host_async(s_bx.data(), N, stream);
    by.copy_from_host_async(s_by.data(), N, stream);
    bz.copy_from_host_async(s_bz.data(), N, stream);
    I.copy_from_host_async(s_I.data(), N, stream);
    px.copy_from_host_async(s_px.data(), M, stream);
    py.copy_from_host_async(s_py.data(), M, stream);
    pz.copy_from_host_async(s_pz.data(), M, stream);
    // The host UploadSrc buffers must outlive the async copies (the copies are on
    // `stream`). For the narrowed-float path UploadSrc::owned_ is a ctor-local that
    // dies at scope exit, so we must sync the upload here before it is freed. For
    // the identity Real path UploadSrc aliases the caller's SoA, which outlives
    // this ctor, so the upload can stay queued — the subsequent kernel + readback
    // already serialize on the same stream — and we skip the extra sync.
    if constexpr (!std::is_same_v<T, Real>) {
      ::quasar::backend::device_synchronize(stream);
    }
  }
};

template <class T>
Field<Vec3T<T>> evaluate_B_impl(const BiotSavartConfig&  cfg,
                                const ConductorSystem&    cs,
                                const PointCloud&         obs) {
  using ::quasar::backend::DeviceBuffer;

  const SegmentSoA& seg = cs.segments_soa();
  const PointSoA    pts = obs.to_point_soa();
  const KernelCounts counts = checked_input_counts(seg, pts);
  const int N = counts.segments;
  const int M = counts.points;

  Field<Vec3T<T>> result(static_cast<std::size_t>(M));
  if (N == 0 || M == 0) {
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = Vec3T<T>{T{0}, T{0}, T{0}};
    }
    return result;
  }

  const UploadedInputs<T> in{seg, pts, N, M, cfg.stream};
  // The kernel writes every observation-point entry, so skip the zero-fill.
  DeviceBuffer<T> d_Bx(M, ::quasar::backend::uninitialized);
  DeviceBuffer<T> d_By(M, ::quasar::backend::uninitialized);
  DeviceBuffer<T> d_Bz(M, ::quasar::backend::uninitialized);
  DeviceBuffer<int> d_status(1);  // zero-initialized bit field

  dispatch_launch_B<T>(
      in.ax.device_ptr(), in.ay.device_ptr(), in.az.device_ptr(),
      in.bx.device_ptr(), in.by.device_ptr(), in.bz.device_ptr(),
      in.I.device_ptr(), N,
      in.px.device_ptr(), in.py.device_ptr(), in.pz.device_ptr(), M,
      d_Bx.device_ptr(), d_By.device_ptr(), d_Bz.device_ptr(),
      d_status.device_ptr(),
      cfg.stream);

  // One host staging buffer (SoA, three M-length component planes) instead of
  // three separate allocations; the device outputs are SoA and Field is AoS, so
  // a single transpose pass into the result is still required.
  const std::size_t MM = static_cast<std::size_t>(M);
  std::vector<T> hB(checked_staging_size(M, 3, "vector staging size"));
  d_Bx.copy_to_host_async(hB.data(),           M, cfg.stream);
  d_By.copy_to_host_async(hB.data() + MM,      M, cfg.stream);
  d_Bz.copy_to_host_async(hB.data() + 2 * MM,  M, cfg.stream);
  int status = 0;
  d_status.copy_to_host_async(&status, 1, cfg.stream);
  ::quasar::backend::device_synchronize(cfg.stream);

  if ((status & 1) != 0) {
    throw std::domain_error{
        "BiotSavartEvaluator: ideal-filament field is singular at an "
        "observation point"};
  }
  if ((status & 2) != 0) {
    throw std::overflow_error{
        "BiotSavartEvaluator: magnetic field is not representable in the "
        "working precision"};
  }

  for (std::size_t i = 0; i < MM; ++i) {
    result[i] = Vec3T<T>{hB[i], hB[MM + i], hB[2 * MM + i]};
  }
  return result;
}

// Vector potential A: structurally identical to evaluate_B_impl (Vec3 SoA
// output), differing only in the launched kernel.
template <class T>
Field<Vec3T<T>> evaluate_A_impl(const BiotSavartConfig&  cfg,
                                const ConductorSystem&    cs,
                                const PointCloud&         obs) {
  using ::quasar::backend::DeviceBuffer;

  const SegmentSoA& seg = cs.segments_soa();
  const PointSoA    pts = obs.to_point_soa();
  const KernelCounts counts = checked_input_counts(seg, pts);
  const int N = counts.segments;
  const int M = counts.points;

  Field<Vec3T<T>> result(static_cast<std::size_t>(M));
  if (N == 0 || M == 0) {
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = Vec3T<T>{T{0}, T{0}, T{0}};
    }
    return result;
  }

  const UploadedInputs<T> in{seg, pts, N, M, cfg.stream};
  // The kernel writes every observation-point entry, so skip the zero-fill.
  DeviceBuffer<T> d_Ax(M, ::quasar::backend::uninitialized);
  DeviceBuffer<T> d_Ay(M, ::quasar::backend::uninitialized);
  DeviceBuffer<T> d_Az(M, ::quasar::backend::uninitialized);
  DeviceBuffer<int> d_status(1);

  dispatch_launch_A<T>(
      in.ax.device_ptr(), in.ay.device_ptr(), in.az.device_ptr(),
      in.bx.device_ptr(), in.by.device_ptr(), in.bz.device_ptr(),
      in.I.device_ptr(), N,
      in.px.device_ptr(), in.py.device_ptr(), in.pz.device_ptr(), M,
      d_Ax.device_ptr(), d_Ay.device_ptr(), d_Az.device_ptr(),
      d_status.device_ptr(),
      cfg.stream);

  const std::size_t MM = static_cast<std::size_t>(M);
  std::vector<T> hA(checked_staging_size(M, 3, "vector staging size"));
  d_Ax.copy_to_host_async(hA.data(),           M, cfg.stream);
  d_Ay.copy_to_host_async(hA.data() + MM,      M, cfg.stream);
  d_Az.copy_to_host_async(hA.data() + 2 * MM,  M, cfg.stream);
  int status = 0;
  d_status.copy_to_host_async(&status, 1, cfg.stream);
  ::quasar::backend::device_synchronize(cfg.stream);

  if ((status & 1) != 0) {
    throw std::domain_error{
        "BiotSavartEvaluator: ideal-filament vector potential is singular at "
        "an observation point"};
  }
  if ((status & 2) != 0) {
    throw std::overflow_error{
        "BiotSavartEvaluator: vector potential is not representable in the "
        "working precision"};
  }

  for (std::size_t i = 0; i < MM; ++i) {
    result[i] = Vec3T<T>{hA[i], hA[MM + i], hA[2 * MM + i]};
  }
  return result;
}

template <class T>
Field<Mat3x3T<T>> evaluate_grad_B_impl(const BiotSavartConfig& cfg,
                                       const ConductorSystem&   cs,
                                       const PointCloud&        obs) {
  using ::quasar::backend::DeviceBuffer;

  const SegmentSoA& seg = cs.segments_soa();
  const PointSoA    pts = obs.to_point_soa();
  const KernelCounts counts = checked_input_counts(seg, pts);
  const int N = counts.segments;
  const int M = counts.points;

  Field<Mat3x3T<T>> result(static_cast<std::size_t>(M));
  if (N == 0 || M == 0) {
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = Mat3x3T<T>{};
    }
    return result;
  }

  const UploadedInputs<T> in{seg, pts, N, M, cfg.stream};
  // The kernel writes every (component, point) entry, so skip the zero-fill.
  const std::size_t gradient_values =
      checked_staging_size(M, 9, "gradient staging size");
  DeviceBuffer<T> d_G(gradient_values, ::quasar::backend::uninitialized);
  DeviceBuffer<int> d_status(1);

  dispatch_launch_gradB<T>(
      in.ax.device_ptr(), in.ay.device_ptr(), in.az.device_ptr(),
      in.bx.device_ptr(), in.by.device_ptr(), in.bz.device_ptr(),
      in.I.device_ptr(), N,
      in.px.device_ptr(), in.py.device_ptr(), in.pz.device_ptr(), M,
      d_G.device_ptr(), d_status.device_ptr(),
      cfg.stream);

  std::vector<T> hG(gradient_values);
  d_G.copy_to_host_async(hG.data(), hG.size(), cfg.stream);
  int status = 0;
  d_status.copy_to_host_async(&status, 1, cfg.stream);
  ::quasar::backend::device_synchronize(cfg.stream);

  if ((status & 1) != 0) {
    throw std::domain_error{
        "BiotSavartEvaluator: ideal-filament field gradient is singular at "
        "an observation point"};
  }
  if ((status & 2) != 0) {
    throw std::overflow_error{
        "BiotSavartEvaluator: magnetic-field gradient is not representable in "
        "the working precision"};
  }

  for (int i = 0; i < M; ++i) {
    const std::size_t mi = static_cast<std::size_t>(i);
    const std::size_t MM = static_cast<std::size_t>(M);
    Mat3x3T<T> g;
    g.r0 = Vec3T<T>{hG[0 * MM + mi], hG[1 * MM + mi], hG[2 * MM + mi]};
    g.r1 = Vec3T<T>{hG[3 * MM + mi], hG[4 * MM + mi], hG[5 * MM + mi]};
    g.r2 = Vec3T<T>{hG[6 * MM + mi], hG[7 * MM + mi], hG[8 * MM + mi]};
    result[mi] = g;
  }
  return result;
}

}  // namespace

// -- Double-precision evaluator --------------------------------------------

BiotSavartEvaluator::BiotSavartEvaluator() = default;
BiotSavartEvaluator::BiotSavartEvaluator(BiotSavartConfig cfg) : cfg_{cfg} {}

namespace {

// Downcast the axis-neutral source to the conductor system this evaluator needs.
const ConductorSystem& as_conductors(const core::IFieldSource& source) {
  const auto* cs = dynamic_cast<const ConductorSystem*>(&source);
  if (cs == nullptr) {
    throw std::invalid_argument{
        "BiotSavartEvaluator: field source is not a ConductorSystem"};
  }
  return *cs;
}

}  // namespace

Field<Vec3> BiotSavartEvaluator::evaluate_B(const core::IFieldSource& source,
                                             const PointCloud&      obs) const {
  return evaluate_B_impl<double>(cfg_, as_conductors(source), obs);
}

Field<Mat3x3> BiotSavartEvaluator::evaluate_grad_B(const core::IFieldSource& source,
                                                    const PointCloud&      obs) const {
  return evaluate_grad_B_impl<double>(cfg_, as_conductors(source), obs);
}

Field<Vec3> BiotSavartEvaluator::evaluate_A(const core::IFieldSource& source,
                                             const PointCloud&      obs) const {
  return evaluate_A_impl<double>(cfg_, as_conductors(source), obs);
}

// -- Single-precision evaluator --------------------------------------------

BiotSavartEvaluatorF::BiotSavartEvaluatorF() = default;
BiotSavartEvaluatorF::BiotSavartEvaluatorF(BiotSavartConfig cfg) : cfg_{cfg} {}

Field<Vec3f> BiotSavartEvaluatorF::evaluate_B(const ConductorSystem& cs,
                                               const PointCloud&      obs) const {
  return evaluate_B_impl<float>(cfg_, cs, obs);
}

Field<Mat3x3f> BiotSavartEvaluatorF::evaluate_grad_B(const ConductorSystem& cs,
                                                      const PointCloud&      obs) const {
  return evaluate_grad_B_impl<float>(cfg_, cs, obs);
}

Field<Vec3f> BiotSavartEvaluatorF::evaluate_A(const ConductorSystem& cs,
                                               const PointCloud&      obs) const {
  return evaluate_A_impl<float>(cfg_, cs, obs);
}

}  // namespace quasar::magnetostatics

QUASAR_REGISTER_FIELD_EVALUATOR("biot_savart",
                                 ::quasar::magnetostatics::BiotSavartEvaluator)
