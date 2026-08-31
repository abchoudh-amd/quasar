// Device evaluation of the six benchmark ideal-MHD initial conditions.
//
// One thread per PADDED cell, including ghosts: every generator here is an
// analytic profile of the cell-centre coordinate, so a ghost cell carries the
// same analytic value the interior would, and the halo is consistent before the
// first boundary fill rather than after it.
//
// The seed is assembled in two passes because the magnetic energy has to be
// re-formed from the face-to-cell COLLOCATED field, which needs every face in
// the stencil written first:
//
//   pass 1  primitives -> conserved, in deck magnetic units, with the raw face
//           slots in the energy; the deck-unit magnetic half-norm is kept in
//           scratch, and the magnetic components are written already scaled to
//           the solver's B/sqrt(mu0) variable.
//   pass 2  collocate bx_face/by_face to the cell, swap the raw magnetic energy
//           for the collocated one, and run the positivity preflight on the
//           exact conserved arrays that pass 1 produced.
//
// The `(E - old) + new` sequence in pass 2 is the arithmetic the displaced
// NumPy performed, kept literally rather than algebraically simplified to
// `p/(gamma-1) + kinetic + new`. The simplified form is more accurate, but it
// would move every seeded energy by an ulp for a reason unrelated to moving the
// work to the GPU, and the MHD example references are not being re-baselined in
// this package.
//
// Compiled -ffp-contract=off: the preflight decides admissibility by comparing
// a difference of same-magnitude quantities against zero, and the collocation
// shares the scaled-arithmetic helpers whose compensation contraction defeats.

#include "quasar/physics/mhd/initial_conditions.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_staggering.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

namespace {

using quasar::Grid2D;
using quasar::Real;
using quasar::mhd::MhdInitialConditionKind;
using quasar::mhd::MhdInitialConditionSpec;

inline constexpr unsigned kBlock2D = 16;

// Primitive state at one cell centre, before the unit conversion.
struct Primitive {
  Real rho{};
  Real p{};
  Real vx{}, vy{}, vz{};
  Real bx{}, by{}, bz{};
  // alfven_wave's exact sub-cell energy variance; zero for every other
  // generator. Added after the conserved energy is formed, exactly where the
  // displaced Python added it.
  Real energy_offset{};
};

// numpy.sinc(q) = sin(pi q)/(pi q), with sinc(0) = 1.
__device__ inline Real sinc(Real q) {
  if (q == Real{0}) return Real{1};
  const Real pq = quasar::pi_v<Real> * q;
  return ::sin(pq) / pq;
}

__device__ inline Primitive brio_wu(const MhdInitialConditionSpec& s, Real xc) {
  // Piecewise constant: the cell-centre value IS the exact cell average except
  // in the single cut cell containing the interface.
  const Real* side = (xc < s.interface) ? s.left : s.right;
  Primitive w;
  w.rho = side[0];
  w.p = side[1];
  w.vx = side[2];
  w.vy = side[3];
  w.vz = side[4];
  w.bx = side[5];
  w.by = side[6];
  w.bz = side[7];
  return w;
}

__device__ inline Primitive alfven_wave(const MhdInitialConditionSpec& s,
                                        const Grid2D& g, Real xc) {
  // Circularly polarised eigenmode travelling along +x. Every transverse
  // primitive is an EXACT cell average: an x-only sine over a cell of width dx
  // averages to sinc(k dx/2) = sinc(n/nx) times its centre value, and a y-normal
  // face spans exactly one cell in x, so by_face carries the same factor.
  const Real n = static_cast<Real>(s.wavenumber);
  const Real k = Real{2} * quasar::pi_v<Real> * n / g.lx;
  const Real phase = k * (xc - g.origin_x);
  const Real fv = sinc(n / static_cast<Real>(g.nx));
  // The eigen-relation is dv = -dB/sqrt(mu0 rho); a normalized deck already
  // carries B in mu0 = 1 units. Only the SIGN of the total axial field matters,
  // and it orients the mode along +x.
  const Real inv_sqrt_rho =
      ::copysign(s.magnetic_velocity_scale, s.total_b0) / ::sqrt(s.rho);

  Primitive w;
  w.rho = s.rho;
  w.p = s.pressure;
  w.vx = Real{0};
  w.vy = -s.amplitude * inv_sqrt_rho * fv * ::sin(phase);
  w.vz = -s.amplitude * inv_sqrt_rho * fv * ::cos(phase);
  w.bx = s.b0;  // uniform, so the face value equals the cell value
  w.by = s.amplitude * fv * ::sin(phase);
  w.bz = s.amplitude * fv * ::cos(phase);
  // Circular polarisation makes the sum of the sine/cosine kinetic and magnetic
  // energies spatially constant, so the exact energy does NOT carry the sinc
  // factor the averaged primitives do. Restore the missing variance.
  const Real scaled_amplitude = s.amplitude * s.magnetic_velocity_scale;
  w.energy_offset = scaled_amplitude * scaled_amplitude * (Real{1} - fv * fv);
  return w;
}

__device__ inline Primitive orszag_tang(const MhdInitialConditionSpec& s,
                                        const Grid2D& g, Real xc, Real yc) {
  // Midpoint projection, not an exact finite-volume one: the profile is smooth
  // and nonlinear, so this is O(h^2) accurate as an element moment regardless
  // of the reconstruction order. Deliberate -- it is the standard OT benchmark.
  const Real xi = (xc - g.origin_x) / g.lx;
  const Real eta = (yc - g.origin_y) / g.ly;
  const Real two_pi = Real{2} * quasar::pi_v<Real>;
  const Real b0 = s.b_uniform[0];

  Primitive w;
  w.rho = s.gamma * s.gamma;
  w.p = s.gamma;
  w.vx = -::sin(two_pi * eta);
  w.vy = ::sin(two_pi * xi);
  w.vz = Real{0};
  // Bx depends only on y and By only on x, so sampling each on its own face
  // leaves the staggered div(B) at round-off.
  w.bx = -b0 * ::sin(two_pi * eta);
  w.by = b0 * ::sin(Real{2} * two_pi * xi);
  w.bz = Real{0};
  return w;
}

__device__ inline Primitive blast(const MhdInitialConditionSpec& s, Real xc,
                                  Real yc) {
  const Real r = ::hypot(xc - s.center[0], yc - s.center[1]);
  Primitive w;
  w.rho = s.rho_ambient;
  w.p = (r < s.r_in) ? s.p_core : s.p_ambient;
  w.vx = Real{0};
  w.vy = Real{0};
  w.vz = Real{0};
  w.bx = s.b_uniform[0];
  w.by = s.b_uniform[1];
  w.bz = s.b_uniform[2];
  return w;
}

__device__ inline Primitive rotor(const MhdInitialConditionSpec& s, Real xc,
                                  Real yc) {
  const Real rx = xc - s.center[0];
  const Real ry = yc - s.center[1];
  const Real r = ::hypot(rx, ry);

  // Linear taper weight: 1 inside r <= r0, 0 outside r >= r1.
  Real f = (s.r1 - r) / (s.r1 - s.r0);
  f = ::fmin(::fmax(f, Real{0}), Real{1});
  const bool inside = r <= s.r0;
  if (inside) f = Real{1};
  if (r >= s.r1) f = Real{0};

  // v_phi/r. Rigid body inside; in the taper the standard rotor profile
  // v_phi = f*u0 is continuous with the rim speed at r0 and rests at r1. The
  // taper never contains r = 0, so the division is safe.
  Real speed_factor = Real{0};
  if (inside) {
    speed_factor = s.u0 / s.r0;
  } else if (r < s.r1) {
    speed_factor = f * s.u0 / r;
  }

  Primitive w;
  w.rho = s.rho_out + f * (s.rho_in - s.rho_out);
  w.p = s.p_ambient;
  w.vx = -speed_factor * ry;
  w.vy = speed_factor * rx;
  w.vz = Real{0};
  w.bx = s.b_uniform[0];
  w.by = s.b_uniform[1];
  w.bz = s.b_uniform[2];
  return w;
}

__device__ inline Primitive confined_blob(const MhdInitialConditionSpec& s,
                                          Real xc, Real yc) {
  const bool inside = ::fabs(xc - s.center[0]) <= s.blob_half &&
                      ::fabs(yc - s.center[1]) <= s.blob_half;
  Primitive w;
  w.rho = inside ? s.rho_in : s.rho_out;
  w.p = inside ? s.p_in : s.p_out;
  w.vx = Real{0};
  w.vy = Real{0};
  w.vz = Real{0};
  // The in-plane perturbation is exactly zero so constrained transport starts
  // and stays at zero divergence under any closure; the confining poloidal
  // field is supplied separately as a static background B0.
  w.bx = Real{0};
  w.by = Real{0};
  w.bz = s.b_uniform[2];
  return w;
}

__device__ inline Primitive evaluate(const MhdInitialConditionSpec& s,
                                     const Grid2D& g, Real xc, Real yc) {
  switch (s.kind) {
    case MhdInitialConditionKind::brio_wu:       return brio_wu(s, xc);
    case MhdInitialConditionKind::alfven_wave:   return alfven_wave(s, g, xc);
    case MhdInitialConditionKind::orszag_tang:   return orszag_tang(s, g, xc, yc);
    case MhdInitialConditionKind::blast:         return blast(s, xc, yc);
    case MhdInitialConditionKind::rotor:         return rotor(s, xc, yc);
    case MhdInitialConditionKind::confined_blob: return confined_blob(s, xc, yc);
  }
  return Primitive{};
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 seed_kernel(
    MhdInitialConditionSpec s,
    double* __restrict__ rho, double* __restrict__ mx,
    double* __restrict__ my, double* __restrict__ mz,
    double* __restrict__ energy, double* __restrict__ bx_face,
    double* __restrict__ by_face, double* __restrict__ bz_cell,
    double* __restrict__ raw_magnetic, int* __restrict__ status) {
  const Grid2D g = s.grid;
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - g.nghost;
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - g.nghost;
  if (i >= g.nx + g.nghost || j >= g.ny + g.nghost) return;

  const Real xc = g.x_at_cell_center(i);
  const Real yc = g.y_at_cell_center(j);
  if (!(::isfinite(xc) && ::isfinite(yc))) {
    ::atomicOr(status, quasar::mhd::kMhdSeedCoordinateNotRepresentable);
    return;
  }

  const Primitive w = evaluate(s, g, xc, yc);
  if (!(::isfinite(w.rho) && w.rho > Real{0})) {
    ::atomicOr(status, quasar::mhd::kMhdSeedDensityNotPositive);
    return;
  }
  if (!(::isfinite(w.p) && w.p > Real{0})) {
    ::atomicOr(status, quasar::mhd::kMhdSeedPressureNotPositive);
    return;
  }

  const Real kinetic =
      quasar::numerics::kinetic_from_velocity(w.rho, w.vx, w.vy, w.vz);
  const Real magnetic = quasar::numerics::half_squared_norm3(w.bx, w.by, w.bz);
  const Real total_energy =
      w.p / (s.gamma - Real{1}) + kinetic + magnetic + w.energy_offset;
  const Real momentum_x = w.rho * w.vx;
  const Real momentum_y = w.rho * w.vy;
  const Real momentum_z = w.rho * w.vz;
  if (!(::isfinite(momentum_x) && ::isfinite(momentum_y) &&
        ::isfinite(momentum_z))) {
    ::atomicOr(status, quasar::mhd::kMhdSeedMomentumNotRepresentable);
    return;
  }
  if (!::isfinite(total_energy)) {
    ::atomicOr(status, quasar::mhd::kMhdSeedEnergyNotRepresentable);
    return;
  }

  const std::size_t k = g.index(i, j);
  rho[k] = w.rho;
  mx[k] = momentum_x;
  my[k] = momentum_y;
  mz[k] = momentum_z;
  energy[k] = total_energy;
  // Deck tesla -> the solver's B/sqrt(mu0) variable. The raw half-norm above is
  // in deck units on purpose: it is exactly the term the energy carries and
  // therefore exactly the term pass 2 removes.
  bx_face[k] = s.magnetic_scale * w.bx;
  by_face[k] = s.magnetic_scale * w.by;
  bz_cell[k] = s.magnetic_scale * w.bz;
  raw_magnetic[k] = magnetic;
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 recollocate_kernel(
    Grid2D g, Real gamma, quasar::numerics::RadialTablesView radial_tables,
    const double* __restrict__ rho, const double* __restrict__ mx,
    const double* __restrict__ my, const double* __restrict__ mz,
    const double* __restrict__ bx_face, const double* __restrict__ by_face,
    const double* __restrict__ bz_cell, const double* __restrict__ raw_magnetic,
    double* __restrict__ energy, int* __restrict__ status) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) - g.nghost;
  const int j = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y) - g.nghost;
  if (i >= g.nx + g.nghost || j >= g.ny + g.nghost) return;
  const std::size_t k = g.index(i, j);

  // The solver's own face-to-cell quadrature, including its order-matched
  // one-sided ghost closure and (when the view is active) the cylindrical
  // radius-weighted moments. Using it here is the point: the seeded energy and
  // the EOS the solver evaluates now read the same B by construction.
  const Real bx_cell = quasar::mhd::cell_bx(g, bx_face, i, j, radial_tables);
  const Real by_cell = quasar::mhd::cell_by(g, by_face, i, j, radial_tables);
  if (!(::isfinite(bx_cell) && ::isfinite(by_cell))) {
    ::atomicOr(status, quasar::mhd::kMhdSeedCollocationNotRepresentable);
    return;
  }
  const Real collocated_magnetic =
      quasar::numerics::half_squared_norm3(bx_cell, by_cell, bz_cell[k]);
  const Real adjusted = (energy[k] - raw_magnetic[k]) + collocated_magnetic;
  if (!::isfinite(adjusted)) {
    ::atomicOr(status, quasar::mhd::kMhdSeedEnergyNotRepresentable);
    return;
  }
  energy[k] = adjusted;

  // Preflight on the exact conserved arrays handed to the solver, not on the
  // primitives they came from: the collocation and the float64 energy assembly
  // can expose a loss of internal energy that the primitive check cannot see.
  const Real density = rho[k];
  if (!(::isfinite(density) && density > Real{0})) {
    ::atomicOr(status, quasar::mhd::kMhdSeedDensityNotPositive);
    return;
  }
  quasar::numerics::MhdState u;
  u.rho = density;
  u.mx = mx[k];
  u.my = my[k];
  u.mz = mz[k];
  u.energy = adjusted;
  u.bx = bx_cell;
  u.by = by_cell;
  u.bz = bz_cell[k];
  const Real p = quasar::numerics::pressure(u, gamma);
  if (!::isfinite(p)) {
    ::atomicOr(status, quasar::mhd::kMhdSeedEnergyNotRepresentable);
    return;
  }
  if (!(p > Real{0})) {
    ::atomicOr(status, quasar::mhd::kMhdSeedPressureNotPositive);
  }
}

dim3 padded_grid(const Grid2D& g, dim3 block) {
  return dim3((static_cast<unsigned>(g.pitch()) + block.x - 1) / block.x,
              (static_cast<unsigned>(g.height()) + block.y - 1) / block.y);
}

}  // namespace

namespace quasar::mhd {

void launch_mhd_seed_initial_state(
    const MhdInitialConditionSpec& spec, MhdField2D<Real>& out,
    backend::DeviceBuffer<Real>& raw_magnetic,
    quasar::numerics::RadialTablesView radial_tables, int* status,
    stream_t stream) {
  const Grid2D g = spec.grid;
  const dim3 block(kBlock2D, kBlock2D);
  const dim3 grid = padded_grid(g, block);
  hipLaunchKernelGGL(seed_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream), spec,
                     out.rho.device_ptr(), out.mx.device_ptr(),
                     out.my.device_ptr(), out.mz.device_ptr(),
                     out.energy.device_ptr(), out.bx_face.device_ptr(),
                     out.by_face.device_ptr(), out.bz_cell.device_ptr(),
                     raw_magnetic.device_ptr(), status);
  QUASAR_HIP_CHECK(::hipGetLastError());
  hipLaunchKernelGGL(recollocate_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream), g, spec.gamma,
                     radial_tables, out.rho.device_ptr(), out.mx.device_ptr(),
                     out.my.device_ptr(), out.mz.device_ptr(),
                     out.bx_face.device_ptr(), out.by_face.device_ptr(),
                     out.bz_cell.device_ptr(), raw_magnetic.device_ptr(),
                     out.energy.device_ptr(), status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::mhd
