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
#include "quasar/physics/equilibrium/critical_points.hpp"
#include "quasar/physics/equilibrium/equilibrium_profile.hpp"
#include "quasar/physics/equilibrium/free_boundary.hpp"

#include <vector>

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

// -- Green's-function boundary coupling ----------------------------------------
//
// This is the dominant cost of a free-boundary solve: the plasma boundary
// integral is O(N_boundary * N_interior) elliptic-integral evaluations per
// Picard step, and on the 65x65 reference case it is most of the 30 s the
// host-side test spends.

// Device-resident coil set. Uploaded once and reused across every Picard step;
// the coils do not change during a solve.
class GsCoilSet {
 public:
  GsCoilSet() = default;
  explicit GsCoilSet(const std::vector<CoilFilament>& coils) { upload(coils); }

  void upload(const std::vector<CoilFilament>& coils);

  const CoilFilament* device_ptr() const noexcept {
    return buffer_.device_ptr();
  }
  int count() const noexcept { return count_; }

 private:
  backend::DeviceBuffer<CoilFilament> buffer_{};
  int count_{0};
};

// psi on the computational boundary from external coils alone. Interior nodes
// are left untouched, matching apply_coil_boundary.
void launch_gs_apply_coil_boundary(const EllipticGrid& g,
                                   const GsCoilSet& coils, Real* d_psi,
                                   stream_t stream);

// psi everywhere from external coils alone -- the vacuum field used to seed the
// nonlinear iteration.
void launch_gs_evaluate_coil_field(const EllipticGrid& g,
                                   const GsCoilSet& coils, Real* d_psi,
                                   stream_t stream);

// -- Derivative fields ---------------------------------------------------------
//
// First and second derivatives of psi over the whole grid, computed once with
// the sixth-order compact operators and reused by the critical-point search and
// the field/flux-surface diagnostics. Matches
// critical_points.hpp::compute_derivatives bit-for-bit.
//
// d_rz is obtained by differentiating d_r along z, in that order -- not by
// differentiating d_z along r. The two are not numerically identical, and the
// host reference picks the former.
struct GsDerivativeFields {
  GsDerivativeFields() = default;
  explicit GsDerivativeFields(const EllipticGrid& g) { resize(g); }

  void resize(const EllipticGrid& g);

  backend::DeviceBuffer<Real> d_r{};
  backend::DeviceBuffer<Real> d_z{};
  backend::DeviceBuffer<Real> d_rr{};
  backend::DeviceBuffer<Real> d_zz{};
  backend::DeviceBuffer<Real> d_rz{};
};

void launch_gs_compute_derivatives(const EllipticGrid& g, const Real* d_psi,
                                   GsDerivativeFields& out,
                                   GsOperatorScratch& scratch,
                                   stream_t stream);

// -- Derived diagnostics: field, flux surfaces, q profile ----------------------
//
// These are computed inside the equilibrium module rather than left to each
// consumer, because B and q are derivatives of psi and the sixth-order Pade
// operators that make them accurate are only available here. A consumer handed
// psi on a grid would re-differentiate it with whatever scheme it has, and the
// sixth-order claim would quietly stop being true at the module boundary.

struct GsMagneticField {
  GsMagneticField() = default;
  explicit GsMagneticField(const EllipticGrid& g) { resize(g); }

  void resize(const EllipticGrid& g);

  backend::DeviceBuffer<Real> b_r{};
  backend::DeviceBuffer<Real> b_z{};
  backend::DeviceBuffer<Real> b_phi{};
  backend::DeviceBuffer<Real> b_poloidal{};  // the denominator in q
};

// Recover F(psi_N) = R*B_phi by integrating FF' inward from a prescribed signed
// vacuum value at the boundary.  The Grad--Shafranov equation determines F^2;
// the sign of `f_vacuum` selects the branch used at every realizable sample.
// Serial by nature -- each sample depends on the one outside it -- so this runs
// single-threaded on device over `samples` points. `profile_scale` must be the
// amplitude GsResult reports; using the raw profile reconstructs a field
// inconsistent with the solved source.
void launch_gs_integrate_f_profile(const ProfileCoefficients& profile,
                                   Real f_vacuum, Real psi_axis, Real psi_bdry,
                                   Real profile_scale, int samples,
                                   Real* d_f_table, stream_t stream);

// B from psi. `d_f_table` is the sampled F(psi_N) above, looked up by nearest
// neighbour exactly as the host driver does.
void launch_gs_compute_field(const EllipticGrid& g, const Real* d_psi,
                             const GsDerivativeFields& derivatives,
                             const Real* d_f_table, int f_samples,
                             Real psi_axis, Real psi_boundary,
                             GsMagneticField& out, stream_t stream);

// Traced flux surfaces. Storage is dense (n_surfaces * n_theta) with a per
// surface point count, because a ray that finds no crossing contributes no
// point and marks the surface open.
struct GsFluxSurfaces {
  GsFluxSurfaces() = default;
  GsFluxSurfaces(int n_surfaces, int n_theta) { resize(n_surfaces, n_theta); }

  void resize(int n_surfaces, int n_theta);

  backend::DeviceBuffer<Real> r{};
  backend::DeviceBuffer<Real> z{};
  backend::DeviceBuffer<int>  count{};   // points actually found, per surface
  backend::DeviceBuffer<int>  closed{};  // 1 if every ray hit
  backend::DeviceBuffer<Real> q{};
  backend::DeviceBuffer<Real> area{};
  backend::DeviceBuffer<Real> volume{};
  backend::DeviceBuffer<Real> psi_n{};

  int n_surfaces{0};
  int n_theta{0};
};

// Aggregates, small enough to copy back as one struct.
struct GsDiagnostics {
  Real q_axis{0};
  Real q_95{0};  // q at the traced surface nearest psi_N = 0.95
  Real total_volume{0};
  Real psi_n_boundary{0};
  Real r_major{0};
  Real r_minor{0};
  Real elongation{0};
  Real triangularity{0};
  int  n_open_surfaces{0};
  bool have_closed{false};
};

// Trace `n_surfaces` surfaces, then compute each one's area, volume and q.
//
// Two phases, split by what can be parallelized. Ray marching is one thread per
// (surface, ray) -- rays are independent. Compaction and the per-surface
// integrals are one thread per SURFACE, walking rays in index order, because
// the host appends hit points in ray order and both the shoelace area and the q
// contour integral accumulate sequentially around the resulting polygon.
void launch_gs_trace_surfaces(const EllipticGrid& g, const Real* d_psi,
                              const GsMagneticField& field, Real axis_r,
                              Real axis_z, Real psi_axis, Real psi_boundary,
                              GsFluxSurfaces& out, stream_t stream);

// Trace the caller-selected normalized-flux surfaces. `psi_n_targets` remains
// device-resident throughout the launch and must contain exactly one target per
// output surface. Each value is used directly for ray tracing and copied
// bit-for-bit to `out.psi_n`; callers may therefore use nonuniform grids such as
// mapped Chebyshev nodes without an intervening host resampling step.
void launch_gs_trace_surfaces_at(
    const EllipticGrid& g, const Real* d_psi,
    const GsMagneticField& field, Real axis_r, Real axis_z, Real psi_axis,
    Real psi_boundary, const backend::DeviceBuffer<Real>& psi_n_targets,
    GsFluxSurfaces& out, stream_t stream);

// Fold the per-surface results into the aggregate bundle. Single-threaded: the
// selections (first closed surface, nearest to psi_N = 0.95, last closed) are
// order-dependent first-wins choices, exactly as in the critical-point merge.
void launch_gs_reduce_diagnostics(GsFluxSurfaces& surfaces,
                                  GsDiagnostics* d_out, stream_t stream);

GsDiagnostics copy_diagnostics_to_host(const backend::DeviceBuffer<GsDiagnostics>& buffer,
                                       stream_t stream);

// -- Multigrid and defect correction -------------------------------------------
//
// The elliptic solve: second-order multigrid as a preconditioner, driven to
// sixth-order accuracy by defect correction against the L6 residual.
//
// Every stage of this ports bit-exactly, which was not a given. The host
// smoother is red-black rather than lexicographic Gauss-Seidel, so the two
// colours are independent within a sweep and one thread per node reproduces the
// sequential result exactly. The bilinear prolongation looks like it
// accumulates -- four passes each doing `+=` into the fine field -- but the
// passes are disjoint by the parity of (i, j), so every fine node receives
// exactly one contribution and there is no accumulation order to preserve.
// Restriction keeps the host's (dj, di) stencil traversal order for the same
// reason.
//
// The level recursion itself stays on the host: it is control flow that
// launches kernels and performs no arithmetic, so there is nothing to move.

struct GsMultigridConfig {
  int pre_smooth{2};
  int post_smooth{2};
  int coarse_sweeps{60};
  int max_levels{32};
};

// Deliberately mirrors numerics::DefectCorrectionConfig rather than reusing it.
// That header drags in the host L6 operator, which this port exists to delete;
// duplicating four fields is cheaper than an include that has to be unpicked
// later.
struct GsDefectCorrectionConfig {
  int  max_iterations{60};
  Real tolerance{Real{1e-10}};  // relative, on the L6 residual
  Real relaxation{Real{0.8}};   // near-optimal; above 1 diverges
  int  inner_cycles{2};         // V-cycles per outer step
};

struct GsDefectCorrectionReport {
  bool converged{false};
  int  iterations{0};
  Real initial_residual{0};
  Real final_residual{0};
};

// Device-resident multigrid hierarchy. Built once per grid and reused across
// every Picard step, so the per-level allocations stay out of the inner loop.
class GsDeviceMultigrid {
 public:
  GsDeviceMultigrid() = default;
  explicit GsDeviceMultigrid(const EllipticGrid& finest,
                             GsMultigridConfig cfg = {});

  // One V-cycle toward Delta* x = b. `d_x` must already carry the Dirichlet
  // boundary values; they are preserved.
  void v_cycle(Real* d_x, const Real* d_b, stream_t stream);

  int n_levels() const noexcept { return static_cast<int>(levels_.size()); }
  const EllipticGrid& level(int l) const {
    return levels_[static_cast<std::size_t>(l)];
  }

 private:
  void vcycle_level(int l, Real* d_x, const Real* d_b, stream_t stream);

  GsMultigridConfig cfg_{};
  std::vector<EllipticGrid> levels_{};
  std::vector<backend::DeviceBuffer<Real>> x_{};
  std::vector<backend::DeviceBuffer<Real>> b_{};
  std::vector<backend::DeviceBuffer<Real>> r_{};
};

// Solve L6 x = b to the configured tolerance, using L2 multigrid as the
// preconditioner. `d_x` must carry the Dirichlet boundary values on entry;
// they are preserved, because the correction starts from zero.
//
// The convergence test reads the residual norm back each outer iteration. That
// is a synchronization per iteration, and it is kept deliberately: dropping it
// in favour of always running max_iterations would change the iteration count
// and therefore the answer.
GsDefectCorrectionReport launch_gs_solve_defect_corrected(
    const EllipticGrid& g, Real* d_x, const Real* d_b, GsDeviceMultigrid& mg,
    GsOperatorScratch& op_scratch, GsReduceScratch& reduce_scratch,
    GsDefectCorrectionConfig cfg, stream_t stream);

// -- Critical points -----------------------------------------------------------
//
// Locating the magnetic axis and the X-points is the step that decides psi_N,
// which decides the current, which creates the axis -- so it runs every Picard
// iteration and its tie-breaking has to match the host exactly.
//
// The algorithm splits into a wide part and a narrow part:
//
//   * candidate scan and Newton refinement -- one thread per interior node,
//     embarrassingly parallel, and bit-exact because each Newton iterate is
//     serial within its own thread;
//   * duplicate merge and axis/boundary selection -- inherently SEQUENTIAL and
//     order-dependent. The host keeps the first candidate in row-major scan
//     order and breaks psi ties by first-wins, so any reordering silently picks
//     a different axis on a symmetric equilibrium.
//
// The narrow part therefore runs single-threaded ON DEVICE rather than being
// copied back to the host. That is a deliberate reading of the
// device-everything decision: the candidate count is tens, so a single-threaded
// device pass costs nothing measurable, and it keeps the result resident. The
// alternative -- parallelizing the merge -- would change the tie-breaking.

// Device-side result of the critical-point search. Fixed capacity, because a
// kernel cannot grow a vector.
struct GsCriticalResult {
  // Far above any physical configuration; a diverted tokamak has one or two
  // X-points and the merge collapses near-duplicates before they are stored.
  static constexpr int kMaxXPoints = 64;

  CriticalPoint axis{};
  CriticalPoint x_points[kMaxXPoints]{};
  int  n_x{0};
  Real psi_axis{0};
  Real psi_boundary{0};
  bool has_closed_surface{false};
  // Set when the merge ran out of X-point capacity. The host has no such limit,
  // so this is the one place the device can diverge; it is reported rather than
  // silently truncated.
  bool x_point_overflow{false};
};

struct GsCriticalScratch {
  GsCriticalScratch() = default;
  explicit GsCriticalScratch(const EllipticGrid& g) { resize(g); }

  void resize(const EllipticGrid& g);

  backend::DeviceBuffer<Real> grad_scale{};          // single element
  backend::DeviceBuffer<CriticalPoint> candidates{}; // grid-sized, sparse
  backend::DeviceBuffer<GsCriticalResult> result{};  // single element
};

// Locate the magnetic axis and every interior X-point. `derivatives` must
// already hold the sixth-order derivative fields of `d_psi`.
void launch_gs_find_critical_points(const EllipticGrid& g, const Real* d_psi,
                                    const GsDerivativeFields& derivatives,
                                    GsCriticalScratch& scratch,
                                    stream_t stream);

// Copy the search result out. Synchronizes the stream.
GsCriticalResult copy_critical_to_host(const GsCriticalScratch& scratch,
                                       stream_t stream);

// -- Source terms --------------------------------------------------------------

// j_phi = r p'(psi_N) + FF'(psi_N) / (mu0 r), zero where psi_N >= 1.
//
// psi_N == 1 is outside the last closed surface, so no current is driven there.
// The profile arrives as a flat POD rather than through IEquilibriumProfile:
// see ProfileCoefficients for why the virtual interface cannot cross to device.
void launch_gs_build_current(const EllipticGrid& g, const Real* d_psi,
                             const ProfileCoefficients& profile, Real psi_axis,
                             Real psi_boundary, Real* d_j_phi,
                             stream_t stream);

// In-place scalar multiply over the whole field. Used to apply the I_p
// normalization to j_phi once the raw current is known.
void launch_gs_scale_field(const EllipticGrid& g, Real scale, Real* d_field,
                           stream_t stream);

// rhs = -mu0 * r * j_phi on interior nodes; untouched on the boundary, which
// carries Dirichlet data rather than a source.
void launch_gs_build_rhs(const EllipticGrid& g, const Real* d_j_phi,
                         Real* d_rhs, stream_t stream);

// dS/dpsi, the diagonal of the source's derivative with respect to psi, zero
// outside the plasma. This is the term whose omission leaves a Picard
// iteration linear; it is the Newton path's Jacobian.
void launch_gs_build_jacobian_diagonal(const EllipticGrid& g,
                                       const Real* d_psi,
                                       const ProfileCoefficients& profile,
                                       Real psi_axis, Real psi_boundary,
                                       Real profile_scale, Real* d_jac,
                                       stream_t stream);

// Superimpose the seeded plasma column: psi += sign * depth * (1 - s^2)^2 for
// s < 1, where s is the normalized distance from the seed centre.
//
// All parameters arrive already resolved -- the host applies the "0 means use
// the domain default" rules and scales the depth by mu0*|I_p| before calling,
// so the kernel has no policy in it. Boundary nodes are skipped, since those
// carry Green's-function data.
void launch_gs_add_plasma_seed(const EllipticGrid& g, Real r_center,
                               Real z_center, Real minor_radius, Real depth,
                               Real sign, Real* d_psi, stream_t stream);

// x += alpha * delta on interior nodes only, leaving Dirichlet data intact.
void launch_gs_axpy_interior(const EllipticGrid& g, Real* d_x,
                             const Real* d_delta, Real alpha, stream_t stream);

// -- Newton path ---------------------------------------------------------------
//
// Off by default. The diagonal profile Jacobian is incomplete for a
// free-boundary solve, where psi on the boundary is itself a dense functional
// of the interior psi, and the measured result is that Newton stalls where
// damped Picard converges. Ported anyway so the two paths stay comparable and
// the switch remains meaningful.

// Solve [Delta* - diag(jac)] delta = -residual, applying the diagonal shift as
// a defect-corrected outer iteration around the existing V-cycle rather than
// rebuilding the multigrid hierarchy for the shifted operator.
void launch_gs_newton_correction(const EllipticGrid& g, const Real* d_jac,
                                 const Real* d_residual, Real* d_delta,
                                 GsDeviceMultigrid& mg, stream_t stream);

// Backtracking line search on the L6 residual norm. Returns the accepted step
// length, or 0 if no tested length improved on the current residual.
Real launch_gs_newton_line_search(const EllipticGrid& g, const Real* d_current,
                                  const Real* d_delta, const Real* d_rhs,
                                  Real* d_trial, GsOperatorScratch& op_scratch,
                                  GsReduceScratch& reduce_scratch,
                                  stream_t stream);

// target += weight * (candidate - target), over the whole field.
void launch_gs_blend(const EllipticGrid& g, Real* d_target,
                     const Real* d_candidate, Real weight, stream_t stream);

// Copy boundary nodes from `source` into `target`, leaving the interior alone.
// The Picard blend interpolates the boundary too, so the exact Dirichlet data
// has to be restored afterwards.
void launch_gs_restore_boundary(const EllipticGrid& g, const Real* d_source,
                                Real* d_target, stream_t stream);

// ADD the plasma's own contribution to the boundary flux, from the toroidal
// current density on interior nodes.
//
// One thread per boundary node, each looping over interior sources in the same
// row-major order the host uses. That ordering is deliberate: it keeps the
// accumulation bit-identical to the host reference, so this kernel is covered
// by an equality test like the operator rather than a tolerance.
//
// The cost is occupancy -- a 65x65 grid has only 256 boundary nodes, so the
// launch is a few wavefronts wide. That is accepted for the port: even at this
// width it is far ahead of the single-threaded host loop, and a cooperative
// block-per-boundary-node variant would reduce over sources in a different
// order and forfeit the equality test. Revisit only if profiling says this is
// the bottleneck after the port is complete and the oracle is no longer needed.
void launch_gs_add_plasma_boundary(const EllipticGrid& g, const Real* d_j_phi,
                                   Real* d_psi, stream_t stream);

}  // namespace quasar::equilibrium
