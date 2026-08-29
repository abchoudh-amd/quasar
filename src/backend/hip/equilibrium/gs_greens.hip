// Device Green's-function coupling for the free-boundary Grad-Shafranov solve.
//
// Three kernels, all sharing one device-side elliptic-integral evaluation:
// coil flux on the boundary, coil flux everywhere (the vacuum seed), and the
// plasma's own boundary contribution -- the last being the dominant cost of the
// whole solver.
//
// -- Bit-exactness, and what it costs -----------------------------------------
// Every kernel here assigns one thread per OUTPUT node and loops over sources
// serially in the host's order. Nothing is reduced across threads, so each
// accumulation is the same sequence of roundings the host performs and the
// results are bit-identical.
//
// For the plasma boundary integral that means the launch is only as wide as the
// boundary node count -- 256 on the 65x65 reference case, a handful of
// wavefronts. The alternative (a block per boundary node, cooperatively
// reducing over the ~4000 interior sources) would be far wider but would sum in
// a different order, which would downgrade the equivalence test from an
// equality to a tolerance. During a hard-replace port the oracle is worth more
// than the occupancy: even a few wavefronts comfortably beat one CPU core on
// ~1e6 elliptic-integral evaluations per Picard step.
//
// -- Determinism ---------------------------------------------------------------
// No atomics, no cross-thread reduction, no dependence on launch
// configuration. Each output node is written by exactly one thread.

#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace quasar::equilibrium {
namespace {

inline constexpr unsigned kBlock1D = 128;
inline constexpr unsigned kBlock2D = 16;
inline constexpr Real kPi = Real{3.14159265358979323846};

struct EllipticKEDev {
  Real k;
  Real e;
};

// Complete elliptic integrals K(m) and E(m) with PARAMETER m = k^2, by the AGM.
// Mirrors complete_elliptic_ke in free_boundary.hpp operation for operation --
// including the endpoint guards, which matter: m -> 1 is the coincident-source
// limit, and returning a large finite value there keeps a NaN from propagating
// silently into the residual.
__device__ inline EllipticKEDev complete_elliptic_ke(Real m) {
  if (m < Real{0}) m = Real{0};
  if (m >= Real{1}) m = Real{1} - Real{1e-16};

  Real a = Real{1};
  Real b = sqrt(Real{1} - m);
  Real c = sqrt(m);
  Real sum = Real{0};
  Real power = Real{0.5};

  for (int it = 0; it < 60; ++it) {
    sum += power * c * c;
    if (fabs(c) < Real{1e-17} * fabs(a)) break;
    const Real a_next = Real{0.5} * (a + b);
    const Real b_next = sqrt(a * b);
    c = Real{0.5} * (a - b);
    a = a_next;
    b = b_next;
    power *= Real{2};
  }

  EllipticKEDev out;
  out.k = kPi / (Real{2} * a);
  out.e = out.k * (Real{1} - sum);
  return out;
}

// Poloidal flux at (r, z) from a unit toroidal current filament at (rp, zp).
__device__ inline Real greens_function_dev(Real r, Real z, Real rp, Real zp) {
  if (r <= Real{0} || rp <= Real{0}) return Real{0};
  const Real dz = z - zp;
  const Real sum_r = r + rp;
  const Real denom = sum_r * sum_r + dz * dz;
  if (denom <= Real{0}) return Real{0};

  const Real m = Real{4} * r * rp / denom;
  if (m >= Real{1} - Real{1e-14}) return Real{0};

  const EllipticKEDev ke = complete_elliptic_ke(m);
  const Real k = sqrt(m);
  return kMu0 / (Real{2} * kPi) * sqrt(r * rp) / k
       * ((Real{2} - m) * ke.k - Real{2} * ke.e);
}

// Grid geometry, passed by value so the device never dereferences a host
// EllipticGrid (whose members are fine as data but whose accessors are host
// functions).
struct GridView {
  int nr;
  int nz;
  Real r_min;
  Real z_min;
  Real dr;
  Real dz;

  __device__ Real r(int i) const { return r_min + static_cast<Real>(i) * dr; }
  __device__ Real z(int j) const { return z_min + static_cast<Real>(j) * dz; }
  __device__ bool on_boundary(int i, int j) const {
    return i == 0 || j == 0 || i == nr - 1 || j == nz - 1;
  }
  __device__ std::size_t index(int i, int j) const {
    return static_cast<std::size_t>(i) + static_cast<std::size_t>(nr) * j;
  }
};

GridView view_of(const EllipticGrid& g) {
  return GridView{g.nr, g.nz, g.r_min, g.z_min, g.dr(), g.dz()};
}

// `boundary_only` selects between apply_coil_boundary (writes boundary nodes,
// leaves the interior untouched) and evaluate_coil_field (writes everything).
template <bool BoundaryOnly>
__global__ void coil_flux_kernel(GridView g,
                                 const CoilFilament* __restrict__ coils,
                                 int n_coils, Real* __restrict__ psi) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= g.nr || j >= g.nz) return;
  if constexpr (BoundaryOnly) {
    if (!g.on_boundary(i, j)) return;
  }

  const Real r = g.r(i);
  const Real z = g.z(j);
  Real acc = Real{0};
  for (int c = 0; c < n_coils; ++c) {
    acc += coils[c].current * greens_function_dev(r, z, coils[c].r, coils[c].z);
  }
  psi[g.index(i, j)] = acc;
}

// One thread per boundary node. Boundary nodes are enumerated by a flat index
// so the launch is dense rather than a 2D grid that discards its interior.
//
// Ordering: the perimeter is walked as (j=0 row), (j=nz-1 row), then the two
// side columns excluding the corners already covered. The enumeration order is
// irrelevant to the result -- each thread writes a distinct node -- but a dense
// mapping keeps every lane busy.
__device__ inline void boundary_node(const GridView& g, int t, int& i, int& j) {
  if (t < g.nr) {
    i = t;
    j = 0;
    return;
  }
  t -= g.nr;
  if (t < g.nr) {
    i = t;
    j = g.nz - 1;
    return;
  }
  t -= g.nr;
  const int inner = g.nz - 2;
  if (t < inner) {
    i = 0;
    j = 1 + t;
    return;
  }
  t -= inner;
  i = g.nr - 1;
  j = 1 + t;
}

__global__ void plasma_boundary_kernel(GridView g,
                                       const Real* __restrict__ j_phi,
                                       Real* psi, int n_boundary,
                                       Real cell_area) {
  const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= n_boundary) return;

  int ib, jb;
  boundary_node(g, t, ib, jb);

  const Real rb = g.r(ib);
  const Real zb = g.z(jb);

  // Source loop in the host's row-major order. The zero-source skip is kept
  // because it is a large win on a plasma that occupies part of the domain --
  // and it changes nothing numerically, since adding an exact zero leaves the
  // accumulator untouched.
  Real acc = Real{0};
  for (int js = 1; js < g.nz - 1; ++js) {
    for (int is = 1; is < g.nr - 1; ++is) {
      const Real src = j_phi[g.index(is, js)];
      if (src == Real{0}) continue;
      acc += src * cell_area * greens_function_dev(rb, zb, g.r(is), g.z(js));
    }
  }
  psi[g.index(ib, jb)] += acc;
}

dim3 grid_2d(int nx, int ny, dim3 block) {
  return dim3((static_cast<unsigned>(nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(ny) + block.y - 1) / block.y);
}

}  // namespace

void GsCoilSet::upload(const std::vector<CoilFilament>& coils) {
  count_ = static_cast<int>(coils.size());
  if (count_ == 0) {
    buffer_ = backend::DeviceBuffer<CoilFilament>{};
    return;
  }
  buffer_ = backend::DeviceBuffer<CoilFilament>{coils.size(),
                                                backend::uninitialized};
  buffer_.copy_from_host(coils.data(), coils.size());
}

void launch_gs_apply_coil_boundary(const EllipticGrid& g,
                                   const GsCoilSet& coils, Real* d_psi,
                                   stream_t stream) {
  auto* s = static_cast<hipStream_t>(stream);
  const dim3 block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL((coil_flux_kernel<true>), grid_2d(g.nr, g.nz, block), block,
                     0, s, view_of(g), coils.device_ptr(), coils.count(), d_psi);
  QUASAR_HIP_CHECK(hipGetLastError());
}

void launch_gs_evaluate_coil_field(const EllipticGrid& g,
                                   const GsCoilSet& coils, Real* d_psi,
                                   stream_t stream) {
  auto* s = static_cast<hipStream_t>(stream);
  const dim3 block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL((coil_flux_kernel<false>), grid_2d(g.nr, g.nz, block),
                     block, 0, s, view_of(g), coils.device_ptr(), coils.count(),
                     d_psi);
  QUASAR_HIP_CHECK(hipGetLastError());
}

void launch_gs_add_plasma_boundary(const EllipticGrid& g, const Real* d_j_phi,
                                   Real* d_psi, stream_t stream) {
  auto* s = static_cast<hipStream_t>(stream);
  // Perimeter node count: both full rows plus the two side columns with the
  // corners removed.
  const int n_boundary = 2 * g.nr + 2 * (g.nz - 2);
  const dim3 block{kBlock1D};
  const dim3 grid{(static_cast<unsigned>(n_boundary) + kBlock1D - 1) / kBlock1D};

  hipLaunchKernelGGL(plasma_boundary_kernel, grid, block, 0, s, view_of(g),
                     d_j_phi, d_psi, n_boundary, g.dr() * g.dz());
  QUASAR_HIP_CHECK(hipGetLastError());
}

}  // namespace quasar::equilibrium
