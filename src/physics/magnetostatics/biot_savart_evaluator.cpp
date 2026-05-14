#include "quasar/physics/magnetostatics/biot_savart.hpp"
#include "quasar/physics/magnetostatics/conductor.hpp"
#include "quasar/physics/magnetostatics/field_evaluator.hpp"
#include "quasar/physics/magnetostatics/observation.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/core/field.hpp"
#include "quasar/core/types.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <vector>

// Defined in src/backend/hip/magnetostatics/biot_savart_hip.hip.
extern "C" void launch_biot_savart_B(
    const ::quasar::Real* ax, const ::quasar::Real* ay, const ::quasar::Real* az,
    const ::quasar::Real* bx, const ::quasar::Real* by, const ::quasar::Real* bz,
    const ::quasar::Real* I_, int N,
    const ::quasar::Real* px, const ::quasar::Real* py, const ::quasar::Real* pz, int M,
    ::quasar::Real* Bx, ::quasar::Real* By, ::quasar::Real* Bz,
    ::hipStream_t stream);

namespace quasar::magnetostatics {

BiotSavartEvaluator::BiotSavartEvaluator() = default;

BiotSavartEvaluator::BiotSavartEvaluator(BiotSavartConfig cfg) : cfg_{cfg} {}

Field<Vec3> BiotSavartEvaluator::evaluate_B(const ConductorSystem& cs,
                                            const PointCloud&      obs) const {
  using ::quasar::backend::DeviceBuffer;

  const SegmentSoA seg = cs.to_segments_soa();
  const PointSoA   pts = obs.to_point_soa();
  const int N = static_cast<int>(seg.n_segments());
  const int M = static_cast<int>(pts.n_points());

  Field<Vec3> result(static_cast<std::size_t>(M));
  if (N == 0 || M == 0) {
    for (std::size_t i = 0; i < result.size(); ++i) {
      result[i] = Vec3{Real{0}, Real{0}, Real{0}};
    }
    return result;
  }

  DeviceBuffer<Real> d_ax(N), d_ay(N), d_az(N);
  DeviceBuffer<Real> d_bx(N), d_by(N), d_bz(N);
  DeviceBuffer<Real> d_I (N);
  DeviceBuffer<Real> d_px(M), d_py(M), d_pz(M);
  DeviceBuffer<Real> d_Bx(M), d_By(M), d_Bz(M);

  d_ax.copy_from_host_async(seg.ax.data(), N, cfg_.stream);
  d_ay.copy_from_host_async(seg.ay.data(), N, cfg_.stream);
  d_az.copy_from_host_async(seg.az.data(), N, cfg_.stream);
  d_bx.copy_from_host_async(seg.bx.data(), N, cfg_.stream);
  d_by.copy_from_host_async(seg.by.data(), N, cfg_.stream);
  d_bz.copy_from_host_async(seg.bz.data(), N, cfg_.stream);
  d_I .copy_from_host_async(seg.I .data(), N, cfg_.stream);
  d_px.copy_from_host_async(pts.px.data(), M, cfg_.stream);
  d_py.copy_from_host_async(pts.py.data(), M, cfg_.stream);
  d_pz.copy_from_host_async(pts.pz.data(), M, cfg_.stream);

  ::launch_biot_savart_B(
      d_ax.device_ptr(), d_ay.device_ptr(), d_az.device_ptr(),
      d_bx.device_ptr(), d_by.device_ptr(), d_bz.device_ptr(),
      d_I.device_ptr(), N,
      d_px.device_ptr(), d_py.device_ptr(), d_pz.device_ptr(), M,
      d_Bx.device_ptr(), d_By.device_ptr(), d_Bz.device_ptr(),
      cfg_.stream);
  QUASAR_HIP_CHECK(::hipGetLastError());

  std::vector<Real> hBx(static_cast<std::size_t>(M));
  std::vector<Real> hBy(static_cast<std::size_t>(M));
  std::vector<Real> hBz(static_cast<std::size_t>(M));
  d_Bx.copy_to_host_async(hBx.data(), M, cfg_.stream);
  d_By.copy_to_host_async(hBy.data(), M, cfg_.stream);
  d_Bz.copy_to_host_async(hBz.data(), M, cfg_.stream);
  QUASAR_HIP_CHECK(::hipStreamSynchronize(cfg_.stream));

  for (int i = 0; i < M; ++i) {
    result[static_cast<std::size_t>(i)] =
        Vec3{hBx[static_cast<std::size_t>(i)],
             hBy[static_cast<std::size_t>(i)],
             hBz[static_cast<std::size_t>(i)]};
  }
  return result;
}

}  // namespace quasar::magnetostatics

QUASAR_REGISTER_FIELD_EVALUATOR("biot_savart",
                                 ::quasar::magnetostatics::BiotSavartEvaluator)
