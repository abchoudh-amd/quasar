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

// Device-resident conductor segments (a/b endpoints plus current).
//
// The observation points are NOT part of this any more. In the fp64 path they
// arrive already on the device as a core::DevicePointCloud -- that is the point
// of the SoA interface, and it is what lets a caller evaluate the same coil at
// the same points repeatedly without re-uploading either. Only the fp32 sibling
// still uploads points, because it narrows them against a shared origin.
template <class T>
struct UploadedSegments {
  using DeviceBuffer = ::quasar::backend::DeviceBuffer<T>;
  DeviceBuffer ax, ay, az, bx, by, bz, I;
  int N{0};

  UploadedSegments(const SegmentSoA& seg, int n_segments,
                   Real ox, Real oy, Real oz, stream_t stream)
      : ax(seg.n_segments()), ay(seg.n_segments()), az(seg.n_segments()),
        bx(seg.n_segments()), by(seg.n_segments()), bz(seg.n_segments()),
        I(seg.n_segments()), N(n_segments) {
    const UploadSrc<T> s_ax{seg.ax, ox, "segment x-coordinate"};
    const UploadSrc<T> s_ay{seg.ay, oy, "segment y-coordinate"};
    const UploadSrc<T> s_az{seg.az, oz, "segment z-coordinate"};
    const UploadSrc<T> s_bx{seg.bx, ox, "segment x-coordinate"};
    const UploadSrc<T> s_by{seg.by, oy, "segment y-coordinate"};
    const UploadSrc<T> s_bz{seg.bz, oz, "segment z-coordinate"};
    const UploadSrc<T> s_I{seg.I, Real{0}, "current"};

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
    // The host UploadSrc buffers must outlive the async copies (the copies are on
    // `stream`). For the narrowed-float path UploadSrc::owned_ is a ctor-local that
    // dies at scope exit, so we must sync the upload here before it is freed. For
    // the identity Real path UploadSrc aliases the caller's SoA, which outlives
    // this ctor, so the upload can stay queued -- the subsequent kernel + readback
    // already serialize on the same stream -- and we skip the extra sync.
    if constexpr (!std::is_same_v<T, Real>) {
      ::quasar::backend::device_synchronize(stream);
    }
  }
};

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

int prepare_segments(const SegmentSoA& seg) {
  // Validate every public SoA plane before sizing a device buffer from only the
  // leading component. Otherwise a shorter plane would be read past its host
  // allocation by the asynchronous upload.
  seg.validate();
  return checked_kernel_count(seg.n_segments(), "segment count");
}

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
  const SegmentSoA& seg = as_conductors(source).segments_soa();
  const int N = prepare_segments(seg);
  const int M = checked_kernel_count(obs.size(), "observation count");

  // No segments or no points means no field; the kernel would not write, so the
  // zero-filled default constructor is the answer.
  if (N == 0 || M == 0) return core::DeviceVectorField(obs.size());

  const UploadedSegments<Real> in{seg, N, Real{0}, Real{0}, Real{0}, cfg_.stream};
  // The kernel writes every observation-point entry, so skip the zero-fill.
  core::DeviceVectorField out(obs.size(), ::quasar::backend::uninitialized);
  DeviceBuffer<int> d_status(1);  // zero-initialized bit field

  dispatch_launch_B<Real>(
      in.ax.device_ptr(), in.ay.device_ptr(), in.az.device_ptr(),
      in.bx.device_ptr(), in.by.device_ptr(), in.bz.device_ptr(),
      in.I.device_ptr(), N,
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
  const SegmentSoA& seg = as_conductors(source).segments_soa();
  const int N = prepare_segments(seg);
  const int M = checked_kernel_count(obs.size(), "observation count");

  if (N == 0 || M == 0) return core::DeviceVectorField(obs.size());

  const UploadedSegments<Real> in{seg, N, Real{0}, Real{0}, Real{0}, cfg_.stream};
  core::DeviceVectorField out(obs.size(), ::quasar::backend::uninitialized);
  DeviceBuffer<int> d_status(1);

  dispatch_launch_A<Real>(
      in.ax.device_ptr(), in.ay.device_ptr(), in.az.device_ptr(),
      in.bx.device_ptr(), in.by.device_ptr(), in.bz.device_ptr(),
      in.I.device_ptr(), N,
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
  const SegmentSoA& seg = as_conductors(source).segments_soa();
  const int N = prepare_segments(seg);
  const int M = checked_kernel_count(obs.size(), "observation count");

  if (N == 0 || M == 0) return core::DeviceTensorField(obs.size());

  const UploadedSegments<Real> in{seg, N, Real{0}, Real{0}, Real{0}, cfg_.stream};
  // The gradient kernel already writes the component-major 9*M layout that
  // DeviceTensorField documents, so it fills the container directly.
  core::DeviceTensorField out(obs.size(), ::quasar::backend::uninitialized);
  DeviceBuffer<int> d_status(1);

  dispatch_launch_gradB<Real>(
      in.ax.device_ptr(), in.ay.device_ptr(), in.az.device_ptr(),
      in.bx.device_ptr(), in.by.device_ptr(), in.bz.device_ptr(),
      in.I.device_ptr(), N,
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
// path just shed -- there is no DeviceVectorField in float, and inventing one
// for a single test consumer would be abstraction without a second user.
//
// It also still uploads its own observation points, because it narrows them
// against an origin shared with the segments: a rigid translation (x=0 to
// x=1e8 m) must be invisible to the narrowing rather than collapsing a short
// segment into one float coordinate. A DevicePointCloud is fp64 and carries no
// such origin, so it cannot serve this path.

namespace {

struct UploadedPointsF {
  ::quasar::backend::DeviceBuffer<float> px, py, pz;
  int M{0};

  UploadedPointsF(const PointSoA& pts, int n_points, Real ox, Real oy, Real oz,
                  stream_t stream)
      : px(pts.n_points()), py(pts.n_points()), pz(pts.n_points()),
        M(n_points) {
    const UploadSrc<float> s_px{pts.px, ox, "observation x-coordinate"};
    const UploadSrc<float> s_py{pts.py, oy, "observation y-coordinate"};
    const UploadSrc<float> s_pz{pts.pz, oz, "observation z-coordinate"};
    px.copy_from_host_async(s_px.data(), M, stream);
    py.copy_from_host_async(s_py.data(), M, stream);
    pz.copy_from_host_async(s_pz.data(), M, stream);
    ::quasar::backend::device_synchronize(stream);
  }
};

struct InputsF {
  UploadedSegments<float> segments;
  UploadedPointsF points;

  InputsF(const SegmentSoA& seg, const PointSoA& pts, int N, int M,
          stream_t stream)
      : segments(seg, N, seg.ax.front(), seg.ay.front(), seg.az.front(), stream),
        points(pts, M, seg.ax.front(), seg.ay.front(), seg.az.front(), stream) {}
};

struct CountsF {
  int N;
  int M;
};

CountsF checked_counts_f(const SegmentSoA& seg, const PointSoA& pts) {
  seg.validate();
  pts.validate();
  return {checked_kernel_count(seg.n_segments(), "segment count"),
          checked_kernel_count(pts.n_points(), "observation count")};
}

}  // namespace

BiotSavartEvaluatorF::BiotSavartEvaluatorF() = default;
BiotSavartEvaluatorF::BiotSavartEvaluatorF(BiotSavartConfig cfg) : cfg_{cfg} {}

Field<Vec3f> BiotSavartEvaluatorF::evaluate_B(const ConductorSystem& cs,
                                               const PointCloud&      obs) const {
  using ::quasar::backend::DeviceBuffer;
  const SegmentSoA& seg = cs.segments_soa();
  const PointSoA    pts = obs.to_point_soa();
  const CountsF counts = checked_counts_f(seg, pts);
  const int N = counts.N;
  const int M = counts.M;

  Field<Vec3f> result(static_cast<std::size_t>(M));
  if (N == 0 || M == 0) {
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = Vec3f{0.0F, 0.0F, 0.0F};
    }
    return result;
  }

  const InputsF in{seg, pts, N, M, cfg_.stream};
  DeviceBuffer<float> d_Bx(M, ::quasar::backend::uninitialized);
  DeviceBuffer<float> d_By(M, ::quasar::backend::uninitialized);
  DeviceBuffer<float> d_Bz(M, ::quasar::backend::uninitialized);
  DeviceBuffer<int> d_status(1);

  dispatch_launch_B<float>(
      in.segments.ax.device_ptr(), in.segments.ay.device_ptr(),
      in.segments.az.device_ptr(), in.segments.bx.device_ptr(),
      in.segments.by.device_ptr(), in.segments.bz.device_ptr(),
      in.segments.I.device_ptr(), N,
      in.points.px.device_ptr(), in.points.py.device_ptr(),
      in.points.pz.device_ptr(), M,
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

Field<Vec3f> BiotSavartEvaluatorF::evaluate_A(const ConductorSystem& cs,
                                               const PointCloud&      obs) const {
  using ::quasar::backend::DeviceBuffer;
  const SegmentSoA& seg = cs.segments_soa();
  const PointSoA    pts = obs.to_point_soa();
  const CountsF counts = checked_counts_f(seg, pts);
  const int N = counts.N;
  const int M = counts.M;

  Field<Vec3f> result(static_cast<std::size_t>(M));
  if (N == 0 || M == 0) {
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = Vec3f{0.0F, 0.0F, 0.0F};
    }
    return result;
  }

  const InputsF in{seg, pts, N, M, cfg_.stream};
  DeviceBuffer<float> d_Ax(M, ::quasar::backend::uninitialized);
  DeviceBuffer<float> d_Ay(M, ::quasar::backend::uninitialized);
  DeviceBuffer<float> d_Az(M, ::quasar::backend::uninitialized);
  DeviceBuffer<int> d_status(1);

  dispatch_launch_A<float>(
      in.segments.ax.device_ptr(), in.segments.ay.device_ptr(),
      in.segments.az.device_ptr(), in.segments.bx.device_ptr(),
      in.segments.by.device_ptr(), in.segments.bz.device_ptr(),
      in.segments.I.device_ptr(), N,
      in.points.px.device_ptr(), in.points.py.device_ptr(),
      in.points.pz.device_ptr(), M,
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
    const ConductorSystem& cs, const PointCloud& obs) const {
  using ::quasar::backend::DeviceBuffer;
  const SegmentSoA& seg = cs.segments_soa();
  const PointSoA    pts = obs.to_point_soa();
  const CountsF counts = checked_counts_f(seg, pts);
  const int N = counts.N;
  const int M = counts.M;

  Field<Mat3x3f> result(static_cast<std::size_t>(M));
  if (N == 0 || M == 0) {
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = Mat3x3f{};
    return result;
  }

  const InputsF in{seg, pts, N, M, cfg_.stream};
  const std::size_t gradient_values =
      checked_staging_size(M, 9, "gradient staging size");
  DeviceBuffer<float> d_G(gradient_values, ::quasar::backend::uninitialized);
  DeviceBuffer<int> d_status(1);

  dispatch_launch_gradB<float>(
      in.segments.ax.device_ptr(), in.segments.ay.device_ptr(),
      in.segments.az.device_ptr(), in.segments.bx.device_ptr(),
      in.segments.by.device_ptr(), in.segments.bz.device_ptr(),
      in.segments.I.device_ptr(), N,
      in.points.px.device_ptr(), in.points.py.device_ptr(),
      in.points.pz.device_ptr(), M,
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
