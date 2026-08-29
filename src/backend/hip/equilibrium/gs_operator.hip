// Device sixth-order Grad-Shafranov operator: the compact (Pade) line
// derivatives and the Delta* combination built from them.
//
// -- Why one thread per line, and not cyclic reduction -------------------------
// The obvious GPU tridiagonal solve is parallel cyclic reduction, and it is NOT
// valid here. The Pade closures are not diagonally dominant in the boundary and
// near-boundary rows (alpha = 5, 126/11, and -131/22), and plain elimination
// without pivoting drives the row-2 diagonal to EXACTLY zero on the
// second-derivative system -- see the extended note in
// numerics/pade_line_solve.hpp. Pivoting is mandatory, and pivoting is what PCR
// cannot express.
//
// The parallelism is therefore taken one level up: the lines are mutually
// independent (nz radial rows, nr axial columns), so each thread owns a whole
// line and runs the same serial pivoting solve the host reference runs. This
// has two consequences worth stating plainly:
//
//   1. It is bit-identical to the host, because the operation order within a
//      line is unchanged. That is what lets the equivalence test assert
//      equality rather than a tolerance.
//   2. Occupancy is bounded by the line COUNT, not the node count. On a 65x65
//      grid that is 65 threads -- poor. This is accepted: the operator is not
//      the hot spot (the Green's-function boundary integral is), and a
//      correctness-preserving structure is worth more here than occupancy on
//      the smallest grids.
//
// Band scratch is element-major -- band[elem * n_lines + line] -- so adjacent
// threads touch adjacent addresses at every step of the solve. The natural
// [line][elem] layout would stride every access by the line length.

#include "quasar/numerics/elliptic_grid.hpp"
#include "quasar/numerics/pade_derivative.hpp"
#include "quasar/physics/equilibrium/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <type_traits>

namespace quasar::equilibrium {
namespace {

using ::quasar::numerics::pade::FirstDerivativeBoundary;
using ::quasar::numerics::pade::FirstDerivativeInterior;
using ::quasar::numerics::pade::FirstDerivativeNearBoundary;
using ::quasar::numerics::pade::SecondDerivativeBoundary;
using ::quasar::numerics::pade::SecondDerivativeInterior;
using ::quasar::numerics::pade::SecondDerivativeNearBoundary;

inline constexpr unsigned kLineBlock = 64;
inline constexpr unsigned kBlock2D   = 16;

// Which derivative a line sweep is assembling.
enum class Deriv { first, second };

// Partial-pivoting tridiagonal solve over a strided band, mirroring
// pade::tridiagonal_solve_pivoting operation for operation.
//
// Element i of each band array lives at [i * stride]. Pivoting produces fill-in
// one column beyond the original band, held in `upper2`.
//
// The host version throws on a zero pivot. A kernel cannot throw, so a singular
// system writes a quiet NaN into the solution instead: the caller's residual
// norm then becomes NaN and the outer Picard loop reports failure through
// GsStatus rather than silently accepting a garbage step.
__device__ inline void tridiag_solve_pivoting(Real* lower, Real* diag,
                                              Real* upper, Real* upper2,
                                              Real* rhs, int n,
                                              std::size_t stride) {
  for (int i = 0; i + 1 < n; ++i) {
    const std::size_t a = static_cast<std::size_t>(i) * stride;
    const std::size_t b = static_cast<std::size_t>(i + 1) * stride;

    if (fabs(lower[b]) > fabs(diag[a])) {
      Real t;
      t = diag[a];   diag[a]   = lower[b];  lower[b]  = t;
      t = upper[a];  upper[a]  = diag[b];   diag[b]   = t;
      t = upper2[a]; upper2[a] = upper[b];  upper[b]  = t;
      t = rhs[a];    rhs[a]    = rhs[b];    rhs[b]    = t;
    }
    if (diag[a] == Real{0}) {
      for (int k = 0; k < n; ++k) {
        rhs[static_cast<std::size_t>(k) * stride] = __longlong_as_double(0x7ff8000000000000LL);
      }
      return;
    }
    const Real factor = lower[b] / diag[a];
    diag[b]  -= factor * upper[a];
    upper[b] -= factor * upper2[a];
    rhs[b]   -= factor * rhs[a];
  }

  const std::size_t last = static_cast<std::size_t>(n - 1) * stride;
  if (diag[last] == Real{0}) {
    for (int k = 0; k < n; ++k) {
      rhs[static_cast<std::size_t>(k) * stride] = __longlong_as_double(0x7ff8000000000000LL);
    }
    return;
  }
  rhs[last] /= diag[last];
  if (n == 1) return;

  const std::size_t prev = static_cast<std::size_t>(n - 2) * stride;
  rhs[prev] = (rhs[prev] - upper[prev] * rhs[last]) / diag[prev];
  for (int i = n - 2; i-- > 0;) {
    const std::size_t k  = static_cast<std::size_t>(i) * stride;
    const std::size_t k1 = static_cast<std::size_t>(i + 1) * stride;
    const std::size_t k2 = static_cast<std::size_t>(i + 2) * stride;
    rhs[k] = (rhs[k] - upper[k] * rhs[k1] - upper2[k] * rhs[k2]) / diag[k];
  }
}

// One thread per line: gather the line, assemble the compact rows, solve, and
// scatter the derivative back into the output field.
//
// Element e of line L sits at field index `L * line_stride + e * elem_stride`,
// which expresses both sweeps: radial rows use (line_stride = nr, elem_stride =
// 1) and axial columns use (line_stride = 1, elem_stride = nr).
template <Deriv D>
__global__ void pade_line_kernel(const Real* __restrict__ f,
                                 Real* __restrict__ out, int n_lines,
                                 int n_elem, std::size_t line_stride,
                                 std::size_t elem_stride, Real h,
                                 Real* __restrict__ lower,
                                 Real* __restrict__ diag,
                                 Real* __restrict__ upper,
                                 Real* __restrict__ upper2,
                                 Real* __restrict__ rhs) {
  const int line = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (line >= n_lines) return;

  const std::size_t band = static_cast<std::size_t>(line);
  const std::size_t bstride = static_cast<std::size_t>(n_lines);
  const std::size_t fbase = static_cast<std::size_t>(line) * line_stride;

  // Field value at line element e.
  auto fv = [&](int e) -> Real {
    return f[fbase + static_cast<std::size_t>(e) * elem_stride];
  };
  // Band slot for line element e.
  auto slot = [&](int e) -> std::size_t {
    return band + static_cast<std::size_t>(e) * bstride;
  };

  const Real hp = (D == Deriv::first) ? h : h * h;

  for (int e = 0; e < n_elem; ++e) {
    const std::size_t s = slot(e);
    lower[s]  = Real{0};
    diag[s]   = Real{1};
    upper[s]  = Real{0};
    upper2[s] = Real{0};
    rhs[s]    = Real{0};
  }

  // -- End closures ------------------------------------------------------------
  // The high end mirrors the low-end stencil. First-derivative weights are
  // NEGATED there (odd parity under reflection); second-derivative weights are
  // not (even parity).
  if constexpr (D == Deriv::first) {
    using B = FirstDerivativeBoundary;
    constexpr Real w[B::stencil_width] = {B::w[0], B::w[1], B::w[2],
                                          B::w[3], B::w[4], B::w[5]};
    upper[slot(0)] = B::alpha;
    Real acc = Real{0};
    for (int s = 0; s < B::stencil_width; ++s) acc += w[s] * fv(s);
    rhs[slot(0)] = acc / hp;

    lower[slot(n_elem - 1)] = B::alpha;
    acc = Real{0};
    for (int s = 0; s < B::stencil_width; ++s) acc += -w[s] * fv(n_elem - 1 - s);
    rhs[slot(n_elem - 1)] = acc / hp;
  } else {
    using B = SecondDerivativeBoundary;
    constexpr Real w[B::stencil_width] = {B::w[0], B::w[1], B::w[2], B::w[3],
                                          B::w[4], B::w[5], B::w[6]};
    upper[slot(0)] = B::alpha;
    Real acc = Real{0};
    for (int s = 0; s < B::stencil_width; ++s) acc += w[s] * fv(s);
    rhs[slot(0)] = acc / hp;

    lower[slot(n_elem - 1)] = B::alpha;
    acc = Real{0};
    for (int s = 0; s < B::stencil_width; ++s) acc += w[s] * fv(n_elem - 1 - s);
    rhs[slot(n_elem - 1)] = acc / hp;
  }

  // -- Near-boundary rows (node 1 and its mirror) ------------------------------
  // These cannot be dropped to a lower-order compact pair: doing so degrades
  // node 0 below even the order of the row that caused it (see the note on
  // FirstDerivativeNearBoundary).
  if constexpr (D == Deriv::first) {
    using N = FirstDerivativeNearBoundary;
    constexpr Real w[N::stencil_width] = {N::w[0], N::w[1], N::w[2], N::w[3],
                                          N::w[4]};
    lower[slot(1)] = N::alpha_lo;
    upper[slot(1)] = N::alpha_hi;
    Real s = Real{0};
    for (int k = 0; k < N::stencil_width; ++k) s += w[k] * fv(1 + N::stencil_lo + k);
    rhs[slot(1)] = s / hp;

    const int m = n_elem - 2;
    lower[slot(m)] = N::alpha_hi;
    upper[slot(m)] = N::alpha_lo;
    s = Real{0};
    for (int k = 0; k < N::stencil_width; ++k) s += -w[k] * fv(m - (N::stencil_lo + k));
    rhs[slot(m)] = s / hp;
  } else {
    using N = SecondDerivativeNearBoundary;
    constexpr Real w[N::stencil_width] = {N::w[0], N::w[1], N::w[2],
                                          N::w[3], N::w[4], N::w[5]};
    lower[slot(1)] = N::alpha_lo;
    upper[slot(1)] = N::alpha_hi;
    Real s = Real{0};
    for (int k = 0; k < N::stencil_width; ++k) s += w[k] * fv(1 + N::stencil_lo + k);
    rhs[slot(1)] = s / hp;

    const int m = n_elem - 2;
    lower[slot(m)] = N::alpha_hi;
    upper[slot(m)] = N::alpha_lo;
    s = Real{0};
    for (int k = 0; k < N::stencil_width; ++k) s += w[k] * fv(m - (N::stencil_lo + k));
    rhs[slot(m)] = s / hp;
  }

  // -- Interior rows -----------------------------------------------------------
  for (int e = 2; e + 2 < n_elem; ++e) {
    const std::size_t s = slot(e);
    if constexpr (D == Deriv::first) {
      using I = FirstDerivativeInterior;
      lower[s] = I::alpha;
      upper[s] = I::alpha;
      rhs[s]   = I::a * (fv(e + 1) - fv(e - 1)) / (Real{2} * hp)
               + I::b * (fv(e + 2) - fv(e - 2)) / (Real{4} * hp);
    } else {
      using I = SecondDerivativeInterior;
      lower[s] = I::alpha;
      upper[s] = I::alpha;
      rhs[s]   = I::a * (fv(e + 1) - Real{2} * fv(e) + fv(e - 1)) / hp
               + I::b * (fv(e + 2) - Real{2} * fv(e) + fv(e - 2)) / (Real{4} * hp);
    }
  }

  tridiag_solve_pivoting(lower + band, diag + band, upper + band,
                         upper2 + band, rhs + band, n_elem, bstride);

  for (int e = 0; e < n_elem; ++e) {
    out[fbase + static_cast<std::size_t>(e) * elem_stride] = rhs[slot(e)];
  }
}

// Delta* psi = psi_rr - psi_r / r + psi_zz, interior only.
__global__ void gs_combine_kernel(Real* __restrict__ y,
                                  const Real* __restrict__ d_r,
                                  const Real* __restrict__ d_rr,
                                  const Real* __restrict__ d_zz, int nr, int nz,
                                  Real r_min, Real dr) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= nr || j >= nz) return;

  const std::size_t k =
      static_cast<std::size_t>(i) + static_cast<std::size_t>(nr) * j;
  if (i == 0 || j == 0 || i == nr - 1 || j == nz - 1) {
    y[k] = Real{0};
    return;
  }
  const Real r = r_min + static_cast<Real>(i) * dr;
  y[k] = d_rr[k] - d_r[k] / r + d_zz[k];
}

// r := b - y on the interior; zero on the boundary, where the Dirichlet data is
// exact and a residual entry would be meaningless.
//
// `r` and `y` deliberately alias (the caller passes one buffer for both, so
// Delta* x is overwritten in place), hence no __restrict__ on either. The
// operation is elementwise, so each thread reads and writes only its own index.
__global__ void gs_residual_kernel(Real* r, const Real* b, const Real* y,
                                   int nr, int nz) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (i >= nr || j >= nz) return;

  const std::size_t k =
      static_cast<std::size_t>(i) + static_cast<std::size_t>(nr) * j;
  r[k] = (i == 0 || j == 0 || i == nr - 1 || j == nz - 1) ? Real{0}
                                                          : b[k] - y[k];
}

dim3 grid_2d(int nx, int ny, dim3 block) {
  return dim3((static_cast<unsigned>(nx) + block.x - 1) / block.x,
              (static_cast<unsigned>(ny) + block.y - 1) / block.y);
}

// Both line sweeps, leaving d_r / d_rr / d_zz populated. Shared by the operator
// and residual entry points.
void run_derivative_sweeps(const EllipticGrid& g, const Real* d_x,
                           GsOperatorScratch& s, hipStream_t stream) {
  const int nr = g.nr;
  const int nz = g.nz;
  const Real dr = g.dr();
  const Real dz = g.dz();

  // Radial sweeps: nz lines of length nr, contiguous in memory.
  {
    const dim3 block{kLineBlock};
    const dim3 grid{(static_cast<unsigned>(nz) + kLineBlock - 1) / kLineBlock};
    hipLaunchKernelGGL((pade_line_kernel<Deriv::first>), grid, block, 0, stream,
                       d_x, s.d_r.device_ptr(), nz, nr,
                       static_cast<std::size_t>(nr), std::size_t{1}, dr,
                       s.lower.device_ptr(), s.diag.device_ptr(),
                       s.upper.device_ptr(), s.upper2.device_ptr(),
                       s.rhs.device_ptr());
    QUASAR_HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL((pade_line_kernel<Deriv::second>), grid, block, 0, stream,
                       d_x, s.d_rr.device_ptr(), nz, nr,
                       static_cast<std::size_t>(nr), std::size_t{1}, dr,
                       s.lower.device_ptr(), s.diag.device_ptr(),
                       s.upper.device_ptr(), s.upper2.device_ptr(),
                       s.rhs.device_ptr());
    QUASAR_HIP_CHECK(hipGetLastError());
  }

  // Axial sweeps: nr lines of length nz, strided by nr.
  {
    const dim3 block{kLineBlock};
    const dim3 grid{(static_cast<unsigned>(nr) + kLineBlock - 1) / kLineBlock};
    hipLaunchKernelGGL((pade_line_kernel<Deriv::second>), grid, block, 0, stream,
                       d_x, s.d_zz.device_ptr(), nr, nz, std::size_t{1},
                       static_cast<std::size_t>(nr), dz, s.lower.device_ptr(),
                       s.diag.device_ptr(), s.upper.device_ptr(),
                       s.upper2.device_ptr(), s.rhs.device_ptr());
    QUASAR_HIP_CHECK(hipGetLastError());
  }
}

}  // namespace

void GsOperatorScratch::resize(const EllipticGrid& g) {
  const std::size_t n = g.size();
  // Band scratch is sized for the sweep with the most slots. Both sweeps cover
  // every node exactly once (n_lines * n_elem == nr * nz), so one grid-sized
  // allocation serves both and the two directions can reuse it sequentially.
  d_r  = backend::DeviceBuffer<Real>{n};
  d_rr = backend::DeviceBuffer<Real>{n};
  d_zz = backend::DeviceBuffer<Real>{n};

  lower  = backend::DeviceBuffer<Real>{n, backend::uninitialized};
  diag   = backend::DeviceBuffer<Real>{n, backend::uninitialized};
  upper  = backend::DeviceBuffer<Real>{n, backend::uninitialized};
  upper2 = backend::DeviceBuffer<Real>{n, backend::uninitialized};
  rhs    = backend::DeviceBuffer<Real>{n, backend::uninitialized};

  band_stride = n;
}

void launch_gs_apply_l6(const EllipticGrid& g, const Real* d_x, Real* d_y,
                        GsOperatorScratch& scratch, stream_t stream) {
  auto* s = static_cast<hipStream_t>(stream);
  run_derivative_sweeps(g, d_x, scratch, s);

  const dim3 block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(gs_combine_kernel, grid_2d(g.nr, g.nz, block), block, 0, s,
                     d_y, scratch.d_r.device_ptr(), scratch.d_rr.device_ptr(),
                     scratch.d_zz.device_ptr(), g.nr, g.nz, g.r_min, g.dr());
  QUASAR_HIP_CHECK(hipGetLastError());
}

void GsDerivativeFields::resize(const EllipticGrid& g) {
  const std::size_t n = g.size();
  d_r  = backend::DeviceBuffer<Real>{n};
  d_z  = backend::DeviceBuffer<Real>{n};
  d_rr = backend::DeviceBuffer<Real>{n};
  d_zz = backend::DeviceBuffer<Real>{n};
  d_rz = backend::DeviceBuffer<Real>{n};
}

void launch_gs_compute_derivatives(const EllipticGrid& g, const Real* d_psi,
                                   GsDerivativeFields& out,
                                   GsOperatorScratch& scratch,
                                   stream_t stream) {
  auto* s = static_cast<hipStream_t>(stream);
  const int nr = g.nr;
  const int nz = g.nz;
  const Real dr = g.dr();
  const Real dz = g.dz();

  const dim3 rblock{kLineBlock};
  const dim3 rgrid{(static_cast<unsigned>(nz) + kLineBlock - 1) / kLineBlock};
  const dim3 zblock{kLineBlock};
  const dim3 zgrid{(static_cast<unsigned>(nr) + kLineBlock - 1) / kLineBlock};

  auto sweep = [&](auto deriv_tag, const Real* src, Real* dst, bool radial) {
    constexpr Deriv D = decltype(deriv_tag)::value;
    if (radial) {
      hipLaunchKernelGGL((pade_line_kernel<D>), rgrid, rblock, 0, s, src, dst,
                         nz, nr, static_cast<std::size_t>(nr), std::size_t{1},
                         dr, scratch.lower.device_ptr(),
                         scratch.diag.device_ptr(), scratch.upper.device_ptr(),
                         scratch.upper2.device_ptr(), scratch.rhs.device_ptr());
    } else {
      hipLaunchKernelGGL((pade_line_kernel<D>), zgrid, zblock, 0, s, src, dst,
                         nr, nz, std::size_t{1}, static_cast<std::size_t>(nr),
                         dz, scratch.lower.device_ptr(),
                         scratch.diag.device_ptr(), scratch.upper.device_ptr(),
                         scratch.upper2.device_ptr(), scratch.rhs.device_ptr());
    }
    QUASAR_HIP_CHECK(hipGetLastError());
  };

  using First  = std::integral_constant<Deriv, Deriv::first>;
  using Second = std::integral_constant<Deriv, Deriv::second>;

  sweep(First{},  d_psi, out.d_r.device_ptr(),  true);
  sweep(Second{}, d_psi, out.d_rr.device_ptr(), true);
  sweep(First{},  d_psi, out.d_z.device_ptr(),  false);
  sweep(Second{}, d_psi, out.d_zz.device_ptr(), false);

  // Mixed derivative: differentiate d_r along z, in THAT order. Differentiating
  // d_z along r instead is a different composition and is not numerically
  // identical; the host reference picks this one.
  sweep(First{}, out.d_r.device_ptr(), out.d_rz.device_ptr(), false);
}

void launch_gs_residual_l6(const EllipticGrid& g, const Real* d_x,
                           const Real* d_b, Real* d_r,
                           GsOperatorScratch& scratch, stream_t stream) {
  auto* s = static_cast<hipStream_t>(stream);
  // Delta* x lands directly in the caller's output buffer, then is turned into
  // b - Delta* x in place. This avoids a second grid-sized scratch field.
  launch_gs_apply_l6(g, d_x, d_r, scratch, stream);

  const dim3 block{kBlock2D, kBlock2D};
  hipLaunchKernelGGL(gs_residual_kernel, grid_2d(g.nr, g.nz, block), block, 0, s,
                     d_r, d_b, d_r, g.nr, g.nz);
  QUASAR_HIP_CHECK(hipGetLastError());
}

}  // namespace quasar::equilibrium
