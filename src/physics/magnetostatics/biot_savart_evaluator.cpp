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
#include <stdexcept>
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
  explicit UploadSrc(const std::vector<Real>& src) {
    if constexpr (std::is_same_v<T, Real>) {
      view_ = &src;
    } else {
      owned_.reserve(src.size());
      for (Real v : src) owned_.push_back(static_cast<T>(v));
      view_ = &owned_;
    }
  }
  const T* data() const noexcept { return view_->data(); }

 private:
  std::vector<T> owned_{};
  const std::vector<T>* view_{nullptr};
};

template <class T>
void dispatch_launch_B(const T* ax, const T* ay, const T* az,
                       const T* bx, const T* by, const T* bz,
                       const T* I_, int N,
                       const T* px, const T* py, const T* pz, int M,
                       T* Bx, T* By, T* Bz,
                       stream_t stream) {
  if constexpr (std::is_same_v<T, double>) {
    ::launch_biot_savart_B_f64(ax, ay, az, bx, by, bz, I_, N,
                                px, py, pz, M, Bx, By, Bz, stream);
  } else {
    ::launch_biot_savart_B_f32(ax, ay, az, bx, by, bz, I_, N,
                                px, py, pz, M, Bx, By, Bz, stream);
  }
}

template <class T>
void dispatch_launch_gradB(const T* ax, const T* ay, const T* az,
                           const T* bx, const T* by, const T* bz,
                           const T* I_, int N,
                           const T* px, const T* py, const T* pz, int M,
                           T* G,
                           stream_t stream) {
  if constexpr (std::is_same_v<T, double>) {
    ::launch_biot_savart_gradB_f64(ax, ay, az, bx, by, bz, I_, N,
                                    px, py, pz, M, G, stream);
  } else {
    ::launch_biot_savart_gradB_f32(ax, ay, az, bx, by, bz, I_, N,
                                    px, py, pz, M, G, stream);
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

  UploadedInputs(const SegmentSoA& seg, const PointSoA& pts, stream_t stream)
      : ax(seg.n_segments()), ay(seg.n_segments()), az(seg.n_segments()),
        bx(seg.n_segments()), by(seg.n_segments()), bz(seg.n_segments()),
        I(seg.n_segments()),
        px(pts.n_points()), py(pts.n_points()), pz(pts.n_points()),
        N(static_cast<int>(seg.n_segments())),
        M(static_cast<int>(pts.n_points())) {
    // UploadSrc keeps each narrowed/aliased host buffer alive until the sync that
    // the caller performs after the kernel launch.
    const UploadSrc<T> s_ax{seg.ax}, s_ay{seg.ay}, s_az{seg.az};
    const UploadSrc<T> s_bx{seg.bx}, s_by{seg.by}, s_bz{seg.bz};
    const UploadSrc<T> s_I{seg.I};
    const UploadSrc<T> s_px{pts.px}, s_py{pts.py}, s_pz{pts.pz};
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
  const int N = static_cast<int>(seg.n_segments());
  const int M = static_cast<int>(pts.n_points());

  Field<Vec3T<T>> result(static_cast<std::size_t>(M));
  if (N == 0 || M == 0) {
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = Vec3T<T>{T{0}, T{0}, T{0}};
    }
    return result;
  }

  const UploadedInputs<T> in{seg, pts, cfg.stream};
  // The kernel writes every observation-point entry, so skip the zero-fill.
  DeviceBuffer<T> d_Bx(M, ::quasar::backend::uninitialized);
  DeviceBuffer<T> d_By(M, ::quasar::backend::uninitialized);
  DeviceBuffer<T> d_Bz(M, ::quasar::backend::uninitialized);

  dispatch_launch_B<T>(
      in.ax.device_ptr(), in.ay.device_ptr(), in.az.device_ptr(),
      in.bx.device_ptr(), in.by.device_ptr(), in.bz.device_ptr(),
      in.I.device_ptr(), N,
      in.px.device_ptr(), in.py.device_ptr(), in.pz.device_ptr(), M,
      d_Bx.device_ptr(), d_By.device_ptr(), d_Bz.device_ptr(),
      cfg.stream);

  std::vector<T> hBx(static_cast<std::size_t>(M));
  std::vector<T> hBy(static_cast<std::size_t>(M));
  std::vector<T> hBz(static_cast<std::size_t>(M));
  d_Bx.copy_to_host_async(hBx.data(), M, cfg.stream);
  d_By.copy_to_host_async(hBy.data(), M, cfg.stream);
  d_Bz.copy_to_host_async(hBz.data(), M, cfg.stream);
  ::quasar::backend::device_synchronize(cfg.stream);

  for (int i = 0; i < M; ++i) {
    result[static_cast<std::size_t>(i)] =
        Vec3T<T>{hBx[static_cast<std::size_t>(i)],
                 hBy[static_cast<std::size_t>(i)],
                 hBz[static_cast<std::size_t>(i)]};
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
  const int N = static_cast<int>(seg.n_segments());
  const int M = static_cast<int>(pts.n_points());

  Field<Mat3x3T<T>> result(static_cast<std::size_t>(M));
  if (N == 0 || M == 0) {
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = Mat3x3T<T>{};
    }
    return result;
  }

  const UploadedInputs<T> in{seg, pts, cfg.stream};
  // The kernel writes every (component, point) entry, so skip the zero-fill.
  DeviceBuffer<T> d_G(static_cast<std::size_t>(9) * static_cast<std::size_t>(M),
                      ::quasar::backend::uninitialized);

  dispatch_launch_gradB<T>(
      in.ax.device_ptr(), in.ay.device_ptr(), in.az.device_ptr(),
      in.bx.device_ptr(), in.by.device_ptr(), in.bz.device_ptr(),
      in.I.device_ptr(), N,
      in.px.device_ptr(), in.py.device_ptr(), in.pz.device_ptr(), M,
      d_G.device_ptr(),
      cfg.stream);

  std::vector<T> hG(static_cast<std::size_t>(9)
                    * static_cast<std::size_t>(M));
  d_G.copy_to_host_async(hG.data(), hG.size(), cfg.stream);
  ::quasar::backend::device_synchronize(cfg.stream);

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

}  // namespace quasar::magnetostatics

QUASAR_REGISTER_FIELD_EVALUATOR("biot_savart",
                                 ::quasar::magnetostatics::BiotSavartEvaluator)
