#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/field_evaluator.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/core/field.hpp"
#include "quasar/core/types.hpp"

#include <cstddef>
#include <type_traits>
#include <vector>

// The biot_savart launch ABI speaks the backend-neutral stream handle, so this
// orchestrator (compiled with the HIP toolchain for device-memory ownership)
// needs no HIP header. The .hip definitions cast the handle back internally.
using stream_t = ::quasar::backend::stream_t;

// Defined in src/backend/hip/magnetostatics/biot_savart_hip.hip.
extern "C" void launch_biot_savart_B_f64(
    const double* ax, const double* ay, const double* az,
    const double* bx, const double* by, const double* bz,
    const double* I_, int N,
    const double* px, const double* py, const double* pz, int M,
    double* Bx, double* By, double* Bz,
    stream_t stream);

extern "C" void launch_biot_savart_B_f32(
    const float* ax, const float* ay, const float* az,
    const float* bx, const float* by, const float* bz,
    const float* I_, int N,
    const float* px, const float* py, const float* pz, int M,
    float* Bx, float* By, float* Bz,
    stream_t stream);

// Defined in src/backend/hip/magnetostatics/biot_savart_grad_hip.hip.
extern "C" void launch_biot_savart_gradB_f64(
    const double* ax, const double* ay, const double* az,
    const double* bx, const double* by, const double* bz,
    const double* I_, int N,
    const double* px, const double* py, const double* pz, int M,
    double* G,
    stream_t stream);

extern "C" void launch_biot_savart_gradB_f32(
    const float* ax, const float* ay, const float* az,
    const float* bx, const float* by, const float* bz,
    const float* I_, int N,
    const float* px, const float* py, const float* pz, int M,
    float* G,
    stream_t stream);

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

template <class T>
Field<Vec3T<T>> evaluate_B_impl(const BiotSavartConfig&  cfg,
                                const ConductorSystem&    cs,
                                const PointCloud&         obs) {
  using ::quasar::backend::DeviceBuffer;

  const SegmentSoA seg = cs.to_segments_soa();
  const PointSoA   pts = obs.to_point_soa();
  const int N = static_cast<int>(seg.n_segments());
  const int M = static_cast<int>(pts.n_points());

  Field<Vec3T<T>> result(static_cast<std::size_t>(M));
  if (N == 0 || M == 0) {
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = Vec3T<T>{T{0}, T{0}, T{0}};
    }
    return result;
  }

  const UploadSrc<T> seg_ax{seg.ax};
  const UploadSrc<T> seg_ay{seg.ay};
  const UploadSrc<T> seg_az{seg.az};
  const UploadSrc<T> seg_bx{seg.bx};
  const UploadSrc<T> seg_by{seg.by};
  const UploadSrc<T> seg_bz{seg.bz};
  const UploadSrc<T> seg_I {seg.I};
  const UploadSrc<T> pts_px{pts.px};
  const UploadSrc<T> pts_py{pts.py};
  const UploadSrc<T> pts_pz{pts.pz};

  DeviceBuffer<T> d_ax(N), d_ay(N), d_az(N);
  DeviceBuffer<T> d_bx(N), d_by(N), d_bz(N);
  DeviceBuffer<T> d_I (N);
  DeviceBuffer<T> d_px(M), d_py(M), d_pz(M);
  DeviceBuffer<T> d_Bx(M), d_By(M), d_Bz(M);

  d_ax.copy_from_host_async(seg_ax.data(), N, cfg.stream);
  d_ay.copy_from_host_async(seg_ay.data(), N, cfg.stream);
  d_az.copy_from_host_async(seg_az.data(), N, cfg.stream);
  d_bx.copy_from_host_async(seg_bx.data(), N, cfg.stream);
  d_by.copy_from_host_async(seg_by.data(), N, cfg.stream);
  d_bz.copy_from_host_async(seg_bz.data(), N, cfg.stream);
  d_I .copy_from_host_async(seg_I .data(), N, cfg.stream);
  d_px.copy_from_host_async(pts_px.data(), M, cfg.stream);
  d_py.copy_from_host_async(pts_py.data(), M, cfg.stream);
  d_pz.copy_from_host_async(pts_pz.data(), M, cfg.stream);

  dispatch_launch_B<T>(
      d_ax.device_ptr(), d_ay.device_ptr(), d_az.device_ptr(),
      d_bx.device_ptr(), d_by.device_ptr(), d_bz.device_ptr(),
      d_I.device_ptr(), N,
      d_px.device_ptr(), d_py.device_ptr(), d_pz.device_ptr(), M,
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

  const SegmentSoA seg = cs.to_segments_soa();
  const PointSoA   pts = obs.to_point_soa();
  const int N = static_cast<int>(seg.n_segments());
  const int M = static_cast<int>(pts.n_points());

  Field<Mat3x3T<T>> result(static_cast<std::size_t>(M));
  if (N == 0 || M == 0) {
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = Mat3x3T<T>{};
    }
    return result;
  }

  const UploadSrc<T> seg_ax{seg.ax};
  const UploadSrc<T> seg_ay{seg.ay};
  const UploadSrc<T> seg_az{seg.az};
  const UploadSrc<T> seg_bx{seg.bx};
  const UploadSrc<T> seg_by{seg.by};
  const UploadSrc<T> seg_bz{seg.bz};
  const UploadSrc<T> seg_I {seg.I};
  const UploadSrc<T> pts_px{pts.px};
  const UploadSrc<T> pts_py{pts.py};
  const UploadSrc<T> pts_pz{pts.pz};

  DeviceBuffer<T> d_ax(N), d_ay(N), d_az(N);
  DeviceBuffer<T> d_bx(N), d_by(N), d_bz(N);
  DeviceBuffer<T> d_I (N);
  DeviceBuffer<T> d_px(M), d_py(M), d_pz(M);
  DeviceBuffer<T> d_G(static_cast<std::size_t>(9)
                      * static_cast<std::size_t>(M));

  d_ax.copy_from_host_async(seg_ax.data(), N, cfg.stream);
  d_ay.copy_from_host_async(seg_ay.data(), N, cfg.stream);
  d_az.copy_from_host_async(seg_az.data(), N, cfg.stream);
  d_bx.copy_from_host_async(seg_bx.data(), N, cfg.stream);
  d_by.copy_from_host_async(seg_by.data(), N, cfg.stream);
  d_bz.copy_from_host_async(seg_bz.data(), N, cfg.stream);
  d_I .copy_from_host_async(seg_I .data(), N, cfg.stream);
  d_px.copy_from_host_async(pts_px.data(), M, cfg.stream);
  d_py.copy_from_host_async(pts_py.data(), M, cfg.stream);
  d_pz.copy_from_host_async(pts_pz.data(), M, cfg.stream);

  dispatch_launch_gradB<T>(
      d_ax.device_ptr(), d_ay.device_ptr(), d_az.device_ptr(),
      d_bx.device_ptr(), d_by.device_ptr(), d_bz.device_ptr(),
      d_I.device_ptr(), N,
      d_px.device_ptr(), d_py.device_ptr(), d_pz.device_ptr(), M,
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

Field<Vec3> BiotSavartEvaluator::evaluate_B(const ConductorSystem& cs,
                                             const PointCloud&      obs) const {
  return evaluate_B_impl<double>(cfg_, cs, obs);
}

Field<Mat3x3> BiotSavartEvaluator::evaluate_grad_B(const ConductorSystem& cs,
                                                    const PointCloud&      obs) const {
  return evaluate_grad_B_impl<double>(cfg_, cs, obs);
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
