#pragma once

// Backend-neutral declaration of the Grad-Shafranov kernel-launch ABI.
//
// The equilibrium analogue of physics/mhd/kernels.hpp: a per-physics seam that
// speaks equilibrium types (EllipticGrid and flat device fields over it), so it
// lives under physics/equilibrium/ rather than backend/ -- the backend axis
// stays physics-neutral (device.hpp / memory.hpp only). Every launch_gs_* entry
// point defined under src/backend/hip/equilibrium/ is declared here exactly
// once and included both by its .hip definition (so signature drift is a
// compile error) and by the sibling host driver. Callers reach the backend only
// through this header.
//
// Pure declarations: no __global__, no <hip/hip_runtime.h>, safe to include
// from non-HIP host translation units.
//
// -- Ownership contract --------------------------------------------------------
// The kernels allocate NOTHING. Every buffer is owned by the caller and passed
// in; scratch is owned by an explicit *Scratch struct that the caller keeps
// alive across launches. This mirrors the MHD backend convention and keeps
// allocation out of the Picard inner loop.
//
// -- Determinism ---------------------------------------------------------------
// Every kernel here is deterministic by construction and carries no
// floating-point atomics. Reductions use a fixed-order tree that finishes on
// device; the line solves are per-thread serial with the same operation order
// as the host reference. The equilibrium HIP module is compiled with
// -ffp-contract=off, so device results are bit-identical to the host reference
// rather than merely close. That exactness is what makes the host code
// deletable at the end of the port: the equivalence tests are an equality, not
// a tolerance.
//
// One documented exception: the plasma-current SUM cannot be bit-exact against
// a sequential host sum, because summation is not associative. It is instead
// held to a stronger standard -- see launch_gs_total_plasma_current.

#include "quasar/backend/device.hpp"
#include "quasar/backend/memory.hpp"
#include "quasar/core/types.hpp"
#include "quasar/numerics/elliptic_grid.hpp"

namespace quasar::equilibrium {

using stream_t = ::quasar::backend::stream_t;
using ::quasar::numerics::EllipticGrid;

// Scratch for one sixth-order operator application.
//
// The Pade line solve pivots, which produces fill-in one column beyond the
// original band (see pade_line_solve.hpp), so the band carries five arrays
// rather than four. The two derivative directions reuse the same band scratch
// sequentially.
//
// Band arrays are laid out ELEMENT-MAJOR -- band[elem * n_lines + line] -- so
// that adjacent threads, which own adjacent lines, touch adjacent addresses at
// every step of the solve. The natural [line][elem] layout would stride every
// access by the line length and lose coalescing entirely.
struct GsOperatorScratch {
  GsOperatorScratch() = default;
  explicit GsOperatorScratch(const EllipticGrid& g) { resize(g); }

  void resize(const EllipticGrid& g);

  // Per-node derivative fields, in grid index order.
  backend::DeviceBuffer<Real> d_r{};   // dpsi/dr
  backend::DeviceBuffer<Real> d_rr{};  // d2psi/dr2
  backend::DeviceBuffer<Real> d_zz{};  // d2psi/dz2

  // Band scratch, element-major. Sized for the longer of the two directions.
  backend::DeviceBuffer<Real> lower{};
  backend::DeviceBuffer<Real> diag{};
  backend::DeviceBuffer<Real> upper{};
  backend::DeviceBuffer<Real> upper2{};
  backend::DeviceBuffer<Real> rhs{};

  std::size_t band_stride{0};  // n_lines for the currently sized grid
};

// y := Delta* x, sixth order, on interior nodes; zero on the boundary.
//
// The Pade line solves run across the FULL line including boundary nodes,
// because the one-sided closures need those values; only the returned operator
// is restricted to the interior. Matches gs_apply_l6 in
// numerics/gs_operator_l6.hpp bit-for-bit.
void launch_gs_apply_l6(const EllipticGrid& g, const Real* d_x, Real* d_y,
                        GsOperatorScratch& scratch, stream_t stream);

// r := b - Delta* x on interior nodes; zero on the boundary, where the
// Dirichlet data is exact and a residual entry would be meaningless.
void launch_gs_residual_l6(const EllipticGrid& g, const Real* d_x,
                           const Real* d_b, Real* d_r,
                           GsOperatorScratch& scratch, stream_t stream);

// -- Reductions ----------------------------------------------------------------
//
// Both reductions are two-pass and finish ON DEVICE: a first kernel reduces to
// one partial per block, a second single-block kernel folds the partials. The
// host tail used elsewhere in the codebase is deliberately avoided here so the
// value can stay resident for device-side control flow; copy_scalar_to_host()
// is the explicit opt-out when the host genuinely needs the number.
//
// Scratch holds the block partials and the final scalar, both device-side.
struct GsReduceScratch {
  GsReduceScratch() = default;
  explicit GsReduceScratch(const EllipticGrid& g) { resize(g); }

  void resize(const EllipticGrid& g);

  backend::DeviceBuffer<Real> partial_hi{};
  backend::DeviceBuffer<Real> partial_lo{};
  backend::DeviceBuffer<Real> result{};  // single element
};

// Copy the reduction result out. Synchronizes the stream.
Real copy_scalar_to_host(const GsReduceScratch& scratch, stream_t stream);

// Max |f| over INTERIOR nodes. Boundary nodes carry Dirichlet data and are
// excluded: including them would report zero there by construction and dilute
// the measurement.
//
// Bit-exact against numerics::interior_max_norm. Max is associative and
// incurs no rounding, so reduction order is irrelevant and the tree result
// equals the sequential result exactly.
void launch_gs_interior_max_norm(const EllipticGrid& g, const Real* d_f,
                                 GsReduceScratch& scratch, stream_t stream);

// Total toroidal plasma current: the integral of j_phi over the poloidal
// cross-section, which is what the profile normalization targets.
//
// NOT bit-exact against the host, and cannot be. The host reference is a naive
// sequential sum, whose value depends on the summation order; any parallel tree
// necessarily sums in a different order. Rather than pretend, this kernel is
// held to a STRONGER standard than the host: double-double (two-sum)
// compensated accumulation in a fixed tree order, which is both reproducible
// run-to-run and closer to the exactly-rounded sum than the host result is.
// The equivalence test asserts exactly that -- device is at least as accurate
// as host, measured against a long-double reference -- rather than asserting an
// equality that would be false.
void launch_gs_total_plasma_current(const EllipticGrid& g, const Real* d_j_phi,
                                    GsReduceScratch& scratch, stream_t stream);

}  // namespace quasar::equilibrium
