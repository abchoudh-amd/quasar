// Device evaluation of the seeded initial electromagnetic field.
//
// See include/quasar/physics/pic/initial_fields.hpp for why this is a kernel and
// why the generators dispatch on an enumerator rather than through a registry.
//
// Two properties of the displaced NumPy are preserved deliberately:
//
//   * Every physical degree of freedom on a component's own Yee lattice is
//     written, including the independent high face at index nx/ny for a
//     face-located component. Periodic fills later make that face duplicate; a
//     nonperiodic run needs it.
//   * The padded slots outside a component's lattice are ZEROED, matching the
//     host path, which allocated a fresh zero buffer per component and wrote
//     only the interior block into it. The ghost values are meaningless until
//     the boundary fill runs, but leaving them indeterminate would make a
//     pre-fill diagnostic read uninitialized memory.
//
// One property is deliberately NOT preserved: lattice coordinates are formed
// with fused multiply-adds where the Python used `(i + offset) / n`. The results
// differ in the last bit. This is the same trade initial_conditions.hpp records
// for the MHD seed, and for the same reason -- it is the coordinate form the
// solver's own mesh helpers use.

#include "quasar/core/yee_field.hpp"
#include "quasar/physics/pic/initial_fields.hpp"
#include "quasar/physics/pic/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::pic::PicInitialFieldKind;
using quasar::pic::PicInitialFieldSpec;
using quasar::pic::PicRadialHalfStepSpec;

inline constexpr unsigned kBlock2D = 16;
inline constexpr int kComponentCount = 6;

// Half-cell offsets of each storage component on its Yee lattice, in
// (ex, ey, ez, bx, by, bz) order. The cylindrical table is a different
// staggering, not a relabelling of the Cartesian one, which is why both are
// carried explicitly.
__device__ inline void component_offsets(int cylindrical, int component,
                                         Real& ox, Real& oy) {
  // Encoded as halves to keep the table exact and branch-free.
  const int cartesian_x[kComponentCount] = {0, 1, 1, 1, 0, 0};
  const int cartesian_y[kComponentCount] = {1, 0, 1, 0, 1, 0};
  const int cylindrical_x[kComponentCount] = {0, 1, 0, 0, 1, 0};
  const int cylindrical_y[kComponentCount] = {1, 0, 1, 0, 1, 0};
  const int hx = cylindrical != 0 ? cylindrical_x[component]
                                  : cartesian_x[component];
  const int hy = cylindrical != 0 ? cylindrical_y[component]
                                  : cartesian_y[component];
  ox = Real{0.5} * static_cast<Real>(hx);
  oy = Real{0.5} * static_cast<Real>(hy);
}

// A component located ON a face along an axis carries one extra independent
// sample: the high face at index n.
__device__ inline void lattice_extent(const Grid2D& g, Real ox, Real oy,
                                      int& x_count, int& y_count) {
  x_count = g.nx + (ox == Real{0} ? 1 : 0);
  y_count = g.ny + (oy == Real{0} ? 1 : 0);
}

// A*sin(2*pi*m*(i+offset)/n + phase), the shared profile behind the Cartesian
// perturbation and both polarizations of the travelling wave.
__device__ inline Real sinusoid(Real amplitude, int mode, int n, int i, Real ox,
                                Real phase) {
  const Real index = static_cast<Real>(i) + ox;
  const Real turns = index / static_cast<Real>(n);
  const Real angle = ::fma(Real{2} * Real{M_PI} * static_cast<Real>(mode),
                           turns, phase);
  return amplitude * ::sin(angle);
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 seed_initial_fields_kernel(
    PicInitialFieldSpec spec, double* __restrict__ ex,
    double* __restrict__ ey, double* __restrict__ ez, double* __restrict__ bx,
    double* __restrict__ by, double* __restrict__ bz,
    int* __restrict__ status) {
  const Grid2D g = spec.grid;
  const int i =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - g.nghost;
  const int j =
      static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - g.nghost;
  if (i >= g.nx + g.nghost || j >= g.ny + g.nghost) return;

  double* planes[kComponentCount] = {ex, ey, ez, bx, by, bz};
  const std::size_t k = g.index(i, j);

  for (int c = 0; c < kComponentCount; ++c) {
    Real ox = Real{0};
    Real oy = Real{0};
    component_offsets(spec.cylindrical, c, ox, oy);
    int x_count = 0;
    int y_count = 0;
    lattice_extent(g, ox, oy, x_count, y_count);

    Real value = Real{0};
    const bool inside = i >= 0 && i < x_count && j >= 0 && j < y_count;
    if (inside) {
      if (spec.kind == PicInitialFieldKind::seed_tm_cavity) {
        // TM_mn on the exact Cartesian Yee lattices. The odd continuation of
        // the cell-centred Ez sine at all four walls makes its wall
        // interpolation exactly zero, and Bx/By carry the matching
        // sine/cosine parity, hence discrete div(B)=0 at every Yee node.
        const Real sx = ::sin(Real{M_PI} * static_cast<Real>(spec.mode_x)
                              * (static_cast<Real>(i) + Real{0.5})
                              / static_cast<Real>(g.nx));
        const Real sy = ::sin(Real{M_PI} * static_cast<Real>(spec.mode_y)
                              * (static_cast<Real>(j) + Real{0.5})
                              / static_cast<Real>(g.ny));
        const Real cx = ::cos(Real{M_PI} * static_cast<Real>(spec.mode_x)
                              * static_cast<Real>(i)
                              / static_cast<Real>(g.nx));
        const Real cy = ::cos(Real{M_PI} * static_cast<Real>(spec.mode_y)
                              * static_cast<Real>(j)
                              / static_cast<Real>(g.ny));
        if (c == static_cast<int>(quasar::pic::PicFieldComponent::ez)) {
          value = spec.amplitude * (sy * sx);
        } else if (c == static_cast<int>(quasar::pic::PicFieldComponent::bx)) {
          value = spec.amplitude * spec.direction_y * spec.half_time * (cy * sx);
        } else if (c == static_cast<int>(quasar::pic::PicFieldComponent::by)) {
          value = -spec.amplitude * spec.direction_x * spec.half_time
                * (sy * cx);
        }
      } else if (spec.kind == PicInitialFieldKind::seed_perturbation
                 && spec.cylindrical != 0) {
        // The axisymmetric radial eigenmode J0(j_{0,m} r/R). A plain
        // sin(2 pi r/R) is dominated by higher radial Bessel modes on an (r,z)
        // grid and would ring the cavity at the wrong line.
        if (c == spec.component) {
          const Real r_over_R = (static_cast<Real>(i) + ox)
                              / static_cast<Real>(g.nx);
          value = spec.amplitude * ::j0(spec.bessel_root * r_over_R);
        }
      } else {
        // Cartesian perturbation and the travelling wave share one profile;
        // the wave additionally seeds a magnetic partner a half step ahead.
        if (c == spec.component) {
          value = sinusoid(spec.amplitude, spec.mode_x, g.nx, i, ox, Real{0});
        } else if (spec.magnetic_component >= 0
                   && c == spec.magnetic_component) {
          value = spec.magnetic_sign
                * sinusoid(spec.amplitude, spec.mode_x, g.nx, i, ox,
                           spec.magnetic_phase);
        }
      }
    }

    if (!::isfinite(value)) {
      ::atomicOr(status, quasar::pic::kPicSeedFieldNotFinite);
      value = Real{0};
    }
    planes[c][k] = value;
  }
}

// Bphi = -(dt/2) * D_r Ez, read off the ghost-filled electric component.
//
// The stencil indexes the source plane directly, including negative and
// past-nx columns. Those slots hold whatever the configured boundary closure
// put there, which is the entire point: the axis-even / wall-odd continuation
// is not restated here.
__global__ void QUASAR_LAUNCH_BOUNDS_256 seed_radial_half_step_kernel(
    PicRadialHalfStepSpec spec, const double* __restrict__ source,
    double* __restrict__ target, int* __restrict__ status) {
  const Grid2D g = spec.grid;
  const int i =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - g.nghost;
  const int j =
      static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - g.nghost;
  if (i >= g.nx + g.nghost || j >= g.ny + g.nghost) return;

  // The magnetic target sits on faces in r, so it owns one more column than
  // there are cells.
  const int x_count = g.nx + 1;
  const int y_count = g.ny + 1;
  const std::size_t k = g.index(i, j);
  if (!(i >= 0 && i < x_count && j >= 0 && j < y_count)) {
    target[k] = Real{0};
    return;
  }

  const auto ez_at = [&](int cell) {
    return source[g.index(cell, j)];
  };
  Real derivative = Real{0};
  if (spec.fdtd_order == 4) {
    derivative = ((Real{9} / Real{8}) * (ez_at(i) - ez_at(i - 1))
                  - (Real{1} / Real{24}) * (ez_at(i + 1) - ez_at(i - 2)))
               / spec.dr;
  } else {
    derivative = (ez_at(i) - ez_at(i - 1)) / spec.dr;
  }
  const Real value = spec.half_dt * derivative;
  if (!::isfinite(value)) {
    ::atomicOr(status, quasar::pic::kPicSeedFieldNotFinite);
    target[k] = Real{0};
    return;
  }
  target[k] = value;
}

double* plane_for(quasar::YeeField2D<Real>& fields, int component) {
  switch (component) {
    case 0: return fields.ex.device_ptr();
    case 1: return fields.ey.device_ptr();
    case 2: return fields.ez.device_ptr();
    case 3: return fields.bx.device_ptr();
    case 4: return fields.by.device_ptr();
    default: return fields.bz.device_ptr();
  }
}

dim3 padded_blocks(const Grid2D& g, dim3 block) {
  const unsigned width = static_cast<unsigned>(g.nx + 2 * g.nghost);
  const unsigned height = static_cast<unsigned>(g.ny + 2 * g.nghost);
  return dim3((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
}

}  // namespace

namespace quasar::pic {

void launch_pic_seed_initial_fields(const PicInitialFieldSpec& spec,
                                    YeeField2D<Real>& fields, int* status,
                                    backend::stream_t stream) {
  const dim3 block(kBlock2D, kBlock2D);
  hipLaunchKernelGGL(seed_initial_fields_kernel, padded_blocks(spec.grid, block),
                     block, 0, static_cast<hipStream_t>(stream), spec,
                     fields.ex.device_ptr(), fields.ey.device_ptr(),
                     fields.ez.device_ptr(), fields.bx.device_ptr(),
                     fields.by.device_ptr(), fields.bz.device_ptr(), status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_pic_seed_radial_half_step(const PicRadialHalfStepSpec& spec,
                                      YeeField2D<Real>& fields, int* status,
                                      backend::stream_t stream) {
  const dim3 block(kBlock2D, kBlock2D);
  hipLaunchKernelGGL(seed_radial_half_step_kernel,
                     padded_blocks(spec.grid, block), block, 0,
                     static_cast<hipStream_t>(stream), spec,
                     plane_for(fields, spec.source_component),
                     plane_for(fields, spec.target_component), status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::pic
