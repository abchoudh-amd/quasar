// Device projection of a converged Grad-Shafranov equilibrium onto the
// staggered MHD mesh.
//
// This was the last grid-scale host computation in a module CLAUDE.md describes
// as "fully GPU-resident ... GsSolver retains only control flow". Every other
// equilibrium stage had a launch_gs_* twin; these three did not, and they are
// nested loops over every MHD cell with up to four bilinear psi samples each.
//
// -- Why the profiles arrive as a POD -----------------------------------------
// The host forms took the F and pressure profiles as TEMPLATE CALLABLES. A
// callable cannot cross to the device, so both are lowered to
// ProfileCoefficients -- the same POD, and the same reason, as
// launch_gs_build_current. That POD already carries both the p and the F
// coefficient arrays, so one object serves the field projection and the fluid
// projection. GsDeviceResult retains it for exactly this purpose.
//
// -- Why the plasma mask is not rebuilt here ----------------------------------
// The host path called axis_connected_plasma_mask, whose flood fill needs the
// Pade derivative fields and a de Casteljau edge test. That work already has a
// device twin in launch_gs_build_plasma_mask, with the same edge-aware
// connectivity and the same private-flux-lobe exclusion. These kernels take the
// resulting device mask as an input rather than recomputing it, so there is no
// second flood fill and no new definition of "inside the separatrix".
//
// -- Sampling ------------------------------------------------------------------
// sample_bilinear_device and nearest_mask_device reproduce their host
// counterparts operation for operation, including the CLAMP rather than an
// extrapolation. Clamping is deliberate in the original and stays: an MHD grid
// slightly larger than the GS domain would otherwise receive extrapolated flux,
// which can create a spurious current sheet at the overhang.
//
// The connectivity result lives on the SOURCE nodes and is sampled by nearest
// node, not bilinearly. Interpolating a binary mask would smear it across the
// separatrix and destroy the topological exclusion that motivates having it.

#include "quasar/physics/equilibrium/kernels.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/reduction_detail.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>

namespace {

namespace hd = quasar::backend::hip_detail;

using quasar::Grid2D;
using quasar::Real;
using quasar::equilibrium::EllipticGrid;
using quasar::equilibrium::GsProjectionSpec;
using quasar::equilibrium::ProfileCoefficients;

inline constexpr unsigned kBlock2D = 16;

__device__ inline Real horner(const Real* c, int n, Real x) {
  Real acc = Real{0};
  for (int k = n; k-- > 0;) acc = acc * x + c[k];
  return acc;
}

__device__ inline Real grid_dr(const EllipticGrid& g) {
  return (g.r_max - g.r_min) / static_cast<Real>(g.nr - 1);
}

__device__ inline Real grid_dz(const EllipticGrid& g) {
  return (g.z_max - g.z_min) / static_cast<Real>(g.nz - 1);
}

__device__ inline std::size_t grid_index(const EllipticGrid& g, int i, int j) {
  return static_cast<std::size_t>(j) * static_cast<std::size_t>(g.nr)
       + static_cast<std::size_t>(i);
}

// Mirrors equilibrium::sample_bilinear.
__device__ inline Real sample_bilinear_device(const EllipticGrid& g,
                                              const Real* __restrict__ f,
                                              Real r, Real z) {
  if (!::isfinite(r) || !::isfinite(z)) return nan("");
  Real fi = (r - g.r_min) / grid_dr(g);
  Real fj = (z - g.z_min) / grid_dz(g);
  if (!::isfinite(fi) || !::isfinite(fj)) return nan("");
  fi = ::fmin(::fmax(fi, Real{0}), static_cast<Real>(g.nr - 1));
  fj = ::fmin(::fmax(fj, Real{0}), static_cast<Real>(g.nz - 1));
  int i = static_cast<int>(fi);
  int j = static_cast<int>(fj);
  if (i > g.nr - 2) i = g.nr - 2;
  if (j > g.nz - 2) j = g.nz - 2;
  const Real tr = fi - static_cast<Real>(i);
  const Real tz = fj - static_cast<Real>(j);
  return (Real{1} - tr) * (Real{1} - tz) * f[grid_index(g, i, j)]
       + tr * (Real{1} - tz) * f[grid_index(g, i + 1, j)]
       + (Real{1} - tr) * tz * f[grid_index(g, i, j + 1)]
       + tr * tz * f[grid_index(g, i + 1, j + 1)];
}

// Mirrors equilibrium::sample_plasma_mask: nearest clamped source node.
__device__ inline bool nearest_mask_device(const EllipticGrid& g,
                                           const int* __restrict__ mask,
                                           Real r, Real z) {
  Real fi = (r - g.r_min) / grid_dr(g);
  Real fj = (z - g.z_min) / grid_dz(g);
  fi = ::fmin(::fmax(fi, Real{0}), static_cast<Real>(g.nr - 1));
  fj = ::fmin(::fmax(fj, Real{0}), static_cast<Real>(g.nz - 1));
  int i = static_cast<int>(::floor(fi + Real{0.5}));
  int j = static_cast<int>(::floor(fj + Real{0.5}));
  if (i > g.nr - 1) i = g.nr - 1;
  if (j > g.nz - 1) j = g.nz - 1;
  return mask[grid_index(g, i, j)] != 0;
}

// Mirrors equilibrium::normalized_flux.
__device__ inline Real normalized_flux_device(Real psi, Real psi_axis,
                                              Real psi_bdry) {
  if (!::isfinite(psi) || !::isfinite(psi_axis) || !::isfinite(psi_bdry)) {
    return Real{1};
  }
  const Real denom = psi_bdry - psi_axis;
  if (denom == Real{0} || !::isfinite(denom)) return Real{1};
  const Real numerator = psi - psi_axis;
  if (!::isfinite(numerator)) return Real{1};
  const Real n = numerator / denom;
  if (!::isfinite(n)) return Real{1};
  return ::fmin(::fmax(n, Real{0}), Real{1});
}

__device__ inline Real profile_coordinate_device(
    const GsProjectionSpec& spec, const Real* __restrict__ psi,
    const int* __restrict__ mask, Real r, Real z) {
  if (!nearest_mask_device(spec.source, mask, r, z)) return Real{1};
  return normalized_flux_device(
      sample_bilinear_device(spec.source, psi, r, z), spec.psi_axis,
      spec.psi_boundary);
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 project_to_mhd_kernel(
    GsProjectionSpec spec, const Real* __restrict__ psi,
    const int* __restrict__ mask, double* __restrict__ b0x,
    double* __restrict__ b0y, double* __restrict__ b0z,
    int* __restrict__ status) {
  const Grid2D t = spec.target;
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)
              - t.nghost;
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y)
              - t.nghost;
  if (i >= t.nx + t.nghost || j >= t.ny + t.nghost) return;

  const std::size_t k = t.index(i, j);
  const Real dz = t.dy();

  // Corner coordinates of cell (i,j): the low corner is (r_edge, z_edge).
  const Real r_lo = t.r_at_edge(i);
  const Real r_hi = t.r_at_edge(i + 1);
  const Real z_lo = ::fma(static_cast<Real>(j), dz, t.origin_y);
  const Real z_hi = z_lo + dz;

  const Real p00 = sample_bilinear_device(spec.source, psi, r_lo, z_lo);
  const Real p01 = sample_bilinear_device(spec.source, psi, r_lo, z_hi);
  const Real p10 = sample_bilinear_device(spec.source, psi, r_hi, z_lo);
  if (!::isfinite(p00) || !::isfinite(p01) || !::isfinite(p10)) {
    ::atomicOr(status, quasar::equilibrium::kGsProjectionCornerFluxNotFinite);
    return;
  }

  // B_r on the x-low face: -(1/r) dpsi/dz, as a corner difference along the
  // face.
  const Real br_denominator = dz * r_lo;
  if (!(::isfinite(br_denominator) && br_denominator > Real{0})) {
    ::atomicOr(status, quasar::equilibrium::kGsProjectionRadialMeasureInvalid);
    return;
  }
  const Real br = -(p01 - p00) / br_denominator;

  // B_z on the y-low face: (1/r) dpsi/dr in annular form.
  const Real ring = t.dx() * t.r_at_cell_center(i);
  if (!(::isfinite(ring) && ring > Real{0})) {
    ::atomicOr(status, quasar::equilibrium::kGsProjectionAnnularMeasureInvalid);
    return;
  }
  const Real bz = (p10 - p00) / ring;
  if (!::isfinite(br) || !::isfinite(bz)) {
    ::atomicOr(status, quasar::equilibrium::kGsProjectionPoloidalNotFinite);
    return;
  }

  // B_phi = F(psi_N)/r at the cell center.
  const Real rc = t.r_at_cell_center(i);
  const Real zc = t.y_at_cell_center(j);
  const Real pn = profile_coordinate_device(spec, psi, mask, rc, zc);
  const Real f_value =
      horner(spec.profile.f_coeffs, spec.profile.n_f, pn);
  if (!::isfinite(f_value)) {
    ::atomicOr(status, quasar::equilibrium::kGsProjectionProfileNotFinite);
    return;
  }
  const Real bphi = f_value / rc;
  if (!::isfinite(bphi)) {
    ::atomicOr(status, quasar::equilibrium::kGsProjectionToroidalNotFinite);
    return;
  }

  b0x[k] = br;
  b0y[k] = bz;
  b0z[k] = bphi;
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 project_fluid_kernel(
    GsProjectionSpec spec, const Real* __restrict__ psi,
    const int* __restrict__ mask, Real rho_axis, Real rho_edge,
    double* __restrict__ rho, double* __restrict__ pressure,
    int* __restrict__ status) {
  const Grid2D t = spec.target;
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)
              - t.nghost;
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y)
              - t.nghost;
  if (i >= t.nx + t.nghost || j >= t.ny + t.nghost) return;

  const std::size_t k = t.index(i, j);
  const Real rc = t.r_at_cell_center(i);
  const Real zc = t.y_at_cell_center(j);
  const Real pn = profile_coordinate_device(spec, psi, mask, rc, zc);

  const Real p_value =
      horner(spec.profile.p_coeffs, spec.profile.n_p, pn);
  if (!(::isfinite(p_value) && p_value >= Real{0})) {
    ::atomicOr(status, quasar::equilibrium::kGsProjectionProfileNotFinite);
    return;
  }
  // Linear in psi_N: peaked on axis, falling to the edge value.
  const Real density = (Real{1} - pn) * rho_axis + pn * rho_edge;
  if (!(::isfinite(density) && density > Real{0})) {
    ::atomicOr(status, quasar::equilibrium::kGsProjectionDensityInvalid);
    return;
  }
  pressure[k] = p_value;
  rho[k] = density;
}

// Discrete divergence in the same annular form the MHD CT scheme uses, reduced
// to one partial max per block.
__global__ void QUASAR_LAUNCH_BOUNDS_256 projection_divergence_kernel(
    Grid2D t, const double* __restrict__ b0x, const double* __restrict__ b0y,
    double* __restrict__ partial) {
  __shared__ Real sdata[hd::kReduceThreads];
  const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

  Real value = Real{0};
  if (i < t.nx && j < t.ny) {
    const Real r_lo = t.r_at_edge(i);
    const Real r_hi = t.r_at_edge(i + 1);
    const Real ring = t.dx() * t.r_at_cell_center(i);
    if (!(::isfinite(ring) && ring > Real{0})) {
      value = HUGE_VAL;
    } else {
      const Real bx_hi = b0x[t.index(i + 1, j)];
      const Real bx_lo = b0x[t.index(i, j)];
      const Real by_hi = b0y[t.index(i, j + 1)];
      const Real by_lo = b0y[t.index(i, j)];
      const Real radial = (r_hi * bx_hi - r_lo * bx_lo) / ring;
      const Real axial = (by_hi - by_lo) / t.dy();
      const Real divergence = radial + axial;
      value = ::isfinite(divergence) ? ::fabs(divergence) : HUGE_VAL;
    }
  }
  sdata[tid] = value;
  __syncthreads();
  hd::block_max_reduce<hd::kReduceThreads>(sdata, tid);
  if (tid == 0) partial[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 divergence_finish_kernel(
    unsigned blocks, const double* __restrict__ partial,
    double* __restrict__ out) {
  __shared__ Real sdata[hd::kReduceThreads];
  const unsigned tid = threadIdx.x;
  Real best = Real{0};
  for (unsigned k = tid; k < blocks; k += hd::kReduceThreads) {
    best = hd::max_propagating_nan(best, partial[k]);
  }
  sdata[tid] = best;
  __syncthreads();
  hd::block_max_reduce<hd::kReduceThreads>(sdata, tid);
  if (tid == 0) out[0] = sdata[0];
}

dim3 padded_blocks(const Grid2D& g, dim3 block) {
  const unsigned width = static_cast<unsigned>(g.nx + 2 * g.nghost);
  const unsigned height = static_cast<unsigned>(g.ny + 2 * g.nghost);
  return dim3((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
}

}  // namespace

namespace quasar::equilibrium {

void launch_gs_project_to_mhd(const GsProjectionSpec& spec, const Real* d_psi,
                              const int* d_plasma_mask,
                              backend::DeviceBuffer<Real>& b0x_face,
                              backend::DeviceBuffer<Real>& b0y_face,
                              backend::DeviceBuffer<Real>& b0z_cell,
                              int* status, stream_t stream) {
  const dim3 block(kBlock2D, kBlock2D);
  hipLaunchKernelGGL(project_to_mhd_kernel, padded_blocks(spec.target, block),
                     block, 0, static_cast<hipStream_t>(stream), spec, d_psi,
                     d_plasma_mask, b0x_face.device_ptr(),
                     b0y_face.device_ptr(), b0z_cell.device_ptr(), status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_gs_project_fluid(const GsProjectionSpec& spec, const Real* d_psi,
                             const int* d_plasma_mask, Real rho_axis,
                             Real rho_edge,
                             backend::DeviceBuffer<Real>& rho,
                             backend::DeviceBuffer<Real>& pressure,
                             int* status, stream_t stream) {
  const dim3 block(kBlock2D, kBlock2D);
  hipLaunchKernelGGL(project_fluid_kernel, padded_blocks(spec.target, block),
                     block, 0, static_cast<hipStream_t>(stream), spec, d_psi,
                     d_plasma_mask, rho_axis, rho_edge, rho.device_ptr(),
                     pressure.device_ptr(), status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

Real launch_gs_projection_max_divergence(
    Grid2D target, const backend::DeviceBuffer<Real>& b0x_face,
    const backend::DeviceBuffer<Real>& b0y_face, stream_t stream) {
  const auto hip_stream = static_cast<hipStream_t>(stream);
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid((static_cast<unsigned>(target.nx) + block.x - 1) / block.x,
                  (static_cast<unsigned>(target.ny) + block.y - 1) / block.y);
  const unsigned blocks = grid.x * grid.y;

  backend::DeviceBuffer<Real> partial{blocks};
  backend::DeviceBuffer<Real> result{1};
  hipLaunchKernelGGL(projection_divergence_kernel, grid, block, 0, hip_stream,
                     target, b0x_face.device_ptr(), b0y_face.device_ptr(),
                     partial.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  hipLaunchKernelGGL(divergence_finish_kernel, dim3{1},
                     dim3{hd::kReduceThreads}, 0, hip_stream, blocks,
                     partial.device_ptr(), result.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
  Real host = Real{0};
  result.copy_to_host(&host, 1);
  QUASAR_HIP_CHECK(::hipStreamSynchronize(hip_stream));
  return host;
}

}  // namespace quasar::equilibrium
