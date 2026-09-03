// Backend-neutral declaration of the PIC kernel-launch ABI.
//
// This is a per-physics seam (it speaks pic types like ParticleSpecies / JField2D),
// so it lives under physics/pic/ rather than backend/ — the backend axis stays
// physics-neutral (device.hpp / memory.hpp only). Every launch_pic_* entry point
// defined under src/backend/hip/pic/ is declared here exactly once and included
// both by its .hip definition (so a signature drift is a compile error) and by
// every caller in the physics/boundary/numerics layers. Callers reach the backend
// only through this header; do not hand-redeclare these extern "C" prototypes.
#pragma once

#include "quasar/backend/device.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/core/yee_field.hpp"
#include "quasar/physics/pic/initial_fields.hpp"
#include "quasar/physics/pic/species.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

// The kernel-launch ABI speaks the backend-neutral stream handle so callers in
// the physics/boundary/numerics layers never include a HIP header. The .hip
// definitions cast it back to quasar_stream_t internally.
using quasar_stream_t = ::quasar::backend::stream_t;

namespace quasar::pic {

// Device-resident execution form of one fixed distributed PIC halo-plan
// entry. The runtime uploads these immutable tables once when it constructs a
// tile; every exchange thereafter moves only the referenced Real payloads.
struct PicDeviceHaloEntry {
  std::uint32_t component{0};
  std::int32_t source_x{0};
  std::int32_t source_y{0};
  std::int32_t destination_x{0};
  std::int32_t destination_y{0};
};

inline constexpr std::size_t pic_device_halo_component_count = 6;

struct PicDeviceHaloConstComponents {
  const Real* component[pic_device_halo_component_count]{};
};

struct PicDeviceHaloComponents {
  Real* component[pic_device_halo_component_count]{};
};

// Topology facts needed by device-resident distributed source operations.
// Global offsets/counts stay 64-bit even though one tile's Grid2D dimensions
// are int-sized, so decomposition never narrows the canonical coordinates.
struct PicDeviceTileExtent {
  std::uint64_t global_x_begin{0};
  std::uint64_t global_y_begin{0};
  std::uint64_t global_nx{0};
  std::uint64_t global_ny{0};
  std::int32_t tile_x{0};
  std::int32_t tile_y{0};
  std::int32_t tiles_x{1};
  std::int32_t tiles_y{1};
  std::int32_t periodic_x{0};
  std::int32_t periodic_y{0};
};

enum class PicDeviceHaloUpdate : int { assign = 0, add = 1 };

void launch_pic_device_halo_pack(
    Grid2D grid, const PicDeviceHaloConstComponents& components,
    const backend::DeviceBuffer<PicDeviceHaloEntry>& entries,
    std::size_t payload_count, std::uint32_t component_mask,
    backend::DeviceBuffer<Real>& payload, backend::stream_t stream);
void launch_pic_device_halo_unpack(
    Grid2D grid, const backend::DeviceBuffer<Real>& payload,
    std::size_t payload_count,
    const backend::DeviceBuffer<PicDeviceHaloEntry>& entries,
    const PicDeviceHaloComponents& components,
    std::uint32_t component_mask, PicDeviceHaloUpdate update,
    backend::stream_t stream);
void launch_pic_device_halo_accumulate(
    Grid2D grid, const PicDeviceHaloConstComponents& sources,
    const backend::DeviceBuffer<PicDeviceHaloEntry>& entries,
    const PicDeviceHaloComponents& destinations,
    std::uint32_t component_mask, backend::stream_t stream);
void launch_pic_device_components_copy(
    Grid2D grid, const PicDeviceHaloConstComponents& sources,
    const PicDeviceHaloComponents& destinations,
    std::uint32_t component_mask, backend::stream_t stream);
void launch_pic_distributed_source_background(
    Grid2D grid, Real* charge, Real density, backend::stream_t stream);
void launch_pic_distributed_filter_axis(
    Grid2D grid, PicDeviceTileExtent tile, const Real* input, Real* output,
    int axis, Real neighbor_weight, Real center_weight, int cylindrical,
    backend::stream_t stream);
void launch_pic_distributed_axis_values(
    Grid2D grid, const Real* radial_current, Real* axis_values,
    int owns_axis, backend::stream_t stream);
void launch_pic_distributed_current_correct_order4(
    Grid2D grid, PicDeviceTileExtent tile, const Real* radial_current,
    const Real* axial_current, const Real* rhs_radial,
    const Real* rhs_axial, const Real* axis_values, Real* output_radial,
    Real* output_axial, int x_lo_mode, int x_hi_mode, int y_lo_mode,
    int y_hi_mode, int cylindrical, int on_axis,
    backend::stream_t stream);

// Removes live particles that no longer belong to this tile, compacts every
// resident record in device memory, and downloads only the departing records.
// The returned records retain their complete trajectory/deposition state and
// stable IDs for routing by the distributed runtime.
ParticleSpecies::HostSnapshot extract_pic_departing_particles(
    ParticleSpecies& species, int include_x_high, int include_y_high,
    backend::stream_t stream);

// -- Diagnostic reductions ---------------------------------------------------
//
// The energy and Gauss-law diagnostics sum strictly positive terms whose
// individual factors span an enormous dynamic range (a mass of 1e-31 times a
// weight of 1e10 times a cell volume). Forming those products in double
// underflows, so each term is carried as a mantissa and a binary exponent and
// the sum is accumulated in that frame. This POD is that accumulator crossing
// back from the device.
//
// The reduction is O(cells) or O(particles) and runs entirely on device. Only
// the final normalization to a Real stays on the host, because it must decide
// between throwing std::domain_error (a field sample was not finite) and
// std::overflow_error (the total is not representable) -- see
// src/physics/pic/diagnostics.cpp. That epilogue reads three scalars, not data.
struct PicScaledSum {
  // Running sum and its Kahan compensation, both in units of 2^exponent.
  Real sum{Real{0}};
  Real correction{Real{0}};
  int exponent{0};
  bool initialized{false};
  // A factor was negative or non-finite: the total is meaningless.
  bool invalid{false};
  // A sampled field value was not finite. Reported separately from `invalid`
  // because the caller raises a different, more specific exception.
  bool nonfinite_input{false};
};

// Sum of 0.5*m*|v|^2*w over live particles. Skips dead slots.
PicScaledSum launch_pic_kinetic_energy(const ParticleSpecies& species,
                                       backend::stream_t stream);

// Sum of |field|^2 times each staggered sample's own control volume, i.e.
// twice the Yee electromagnetic energy. Components are NOT collocated before
// squaring: averaging first would erase a checkerboard mode, which is not the
// Yee energy norm. `periodic_x`/`periodic_y` select the half-width edge
// weights; `cylindrical` selects annular control volumes.
PicScaledSum launch_pic_field_energy(const YeeField2D<Real>& fields,
                                     Grid2D grid, int periodic_x,
                                     int periodic_y, int cylindrical,
                                     backend::stream_t stream);

// Volume-weighted sum of (div(E)-rho)^2 and the matching total volume; the
// caller forms the RMS from the pair. Kept as two sums rather than one ratio
// so the host can distinguish an unrepresentable numerator from a degenerate
// domain volume.
struct PicGaussResidualSums {
  PicScaledSum weighted_square;
  PicScaledSum volume;
};

PicGaussResidualSums launch_pic_gauss_residual(
    const YeeField2D<Real>& fields, const ScalarGrid2D<Real>& charge,
    int fdtd_order, int periodic_x, int periodic_y, int cylindrical,
    backend::stream_t stream);

// Net and absolute charge carried by one species' live particles.
//
// Both are needed together: the net charge sets the neutralizing background,
// and the doubly-periodic neutrality check is only meaningful as the ratio of
// the two. Computing them in one pass also guarantees they see identical
// particle state.
struct PicChargeTotals {
  // Signed sum of charge*weight. Unlike the energy sums this one is expected
  // to cancel to near zero, which is exactly why it is accumulated with
  // compensation in a shared exponent frame rather than naively.
  PicScaledSum net;
  // Sum of |charge*weight|, the scale against which a residual net charge is
  // judged negligible.
  PicScaledSum absolute;
};

PicChargeTotals launch_pic_total_charge(const ParticleSpecies& species,
                                        backend::stream_t stream);

// -- Distributed particle migration ------------------------------------------
//
// A particle that leaves its tile has to be told which tile it belongs to next,
// and that decision is arithmetic: a periodic coordinate is wrapped back into
// the global domain, then divided by the cell width and floored to a global
// cell index, which an integer partition maps to an owning endpoint. All of it
// used to run on the host over a downloaded snapshot. It runs here now, so the
// records that cross to the host are already routed and already ordered, and
// the host does nothing to them but memcpy.
//
// These entry points speak pic and backend types, so they are ordinary C++ in
// this namespace rather than part of the extern "C" field-data ABI below.

// The departing set left where the partition kernel wrote it. Same records
// `extract_pic_departing_particles` downloads, minus the download.
struct PicDepartingParticles {
  backend::DeviceBuffer<Real> x, y, x_prev, y_prev;
  backend::DeviceBuffer<Real> vx, vy, vz, vphi_deposit, weight;
  backend::DeviceBuffer<std::uint8_t> alive;
  backend::DeviceBuffer<std::uint64_t> id;
  std::size_t count{0};
};

PicDepartingParticles extract_pic_departing_particles_device(
    ParticleSpecies& species, int include_x_high, int include_y_high,
    backend::stream_t stream);

// One migrating particle on the wire. Trivially copyable and transmitted
// verbatim, so this layout *is* the migration wire format; both the packing
// kernel and the receiving host path go through this one definition rather
// than two that have to be kept in agreement.
struct PicParticleRecord {
  Real x{}, y{}, x_prev{}, y_prev{}, vx{}, vy{}, vz{}, vphi_deposit{},
      weight{};
  std::uint64_t id{0};
  std::uint64_t source_endpoint{0};
  std::uint8_t alive{0};
};

struct PicParticleMigrationRecord {
  PicParticleRecord particle{};
  std::uint64_t destination_endpoint{0};
  std::uint64_t species{0};
};

// The global mesh and its tile decomposition, reduced to the plain integers and
// grid scalars a kernel can take. The distributed topology's own types stay on
// its side of this seam; `quasar::distributed::VirtualTopology` fills this in.
//
// `px`/`py` are the tile counts per axis and the endpoint of tile (tx, ty) is
// ty * px + tx, matching VirtualTopology::endpoint_at. The cell-to-tile split
// is the same balanced partition as VirtualTopology's: the first
// `global_n % p` tiles get one extra cell.
struct PicMigrationTopology {
  Grid2D grid{};
  std::uint64_t global_nx{0}, global_ny{0};
  std::uint64_t px{1}, py{1};
  std::uint64_t endpoint_count{1};
  int periodic_x{0};
  int periodic_y{0};
};

// Status bits ORed into the routing status word. A kernel cannot throw, so it
// reports and the host raises; see src/distributed/pic_runtime.cpp for the
// messages. Bits are independent and may all be set by one batch.
inline constexpr int kPicMigrationCoordinateOutsideMesh = 1;
inline constexpr int kPicMigrationOwnerOutOfRange = 2;
inline constexpr int kPicMigrationDuplicateId = 4;

// Route one species' departing particles and emit their wire records.
//
// Records come back grouped by destination rank -- `rank_offsets` has
// rank_count + 1 entries and group r occupies [rank_offsets[r],
// rank_offsets[r + 1]) -- and within a group ordered by ascending stable id.
// Both orderings are produced on the device, and the id order is what makes the
// result independent of the order the partition kernel happened to emit.
//
// `endpoint_rank` is a device array of `topology.endpoint_count` entries giving
// the world rank that owns each endpoint.
struct PicMigrationRouting {
  std::vector<PicParticleMigrationRecord> records;
  std::vector<std::uint64_t> rank_offsets;
  // Particles whose owner changed. Counted on the device for the same reason
  // the routing is: it is a property of the routing decision.
  std::uint64_t migrated{0};
};

PicMigrationRouting launch_pic_route_departing_particles(
    const PicDepartingParticles& departing, PicMigrationTopology topology,
    std::uint64_t source_endpoint, std::uint64_t species_index,
    const backend::DeviceBuffer<std::uint64_t>& endpoint_rank,
    std::size_t rank_count, int* status, backend::stream_t stream);

// Sort a merged arrival set by stable id and append it to the species.
//
// The merge itself is a host memcpy of records that arrived from several ranks
// in no particular order; this restores a deterministic order and expands the
// records into the species' SoA planes without either step touching the host.
void launch_pic_append_migrated_records(
    ParticleSpecies& species,
    const std::vector<PicParticleMigrationRecord>& records,
    backend::stream_t stream);

// Report whether `ids` contains a repeat. Sorts on the device rather than
// building an O(N) host hash set.
bool launch_pic_ids_have_duplicate(const std::vector<std::uint64_t>& ids,
                                   backend::stream_t stream);

// -- Initial particle sampling -----------------------------------------------
//
// Quiet-start positions and Maxwellian velocities for a species at t = 0. This
// used to be NumPy in `quasar.pic.cli`: an O(N) lattice, an O(N) normal sample
// from `np.random.default_rng`, an O(N) perturbation, and an O(N) speed check,
// all on the host and then uploaded. It is kernels now, writing straight into
// the species' device planes.
//
// The RNG changed with the move and the sampled velocities are therefore
// different draws. `np.random.default_rng` is a stateful PCG64 stream, which a
// kernel cannot reproduce without serializing; the replacement is Philox4x32-10
// keyed on the seed and the species and *counted* by the particle's own index,
// so every particle's draw is a pure function of where it sits. That makes the
// sample independent of thread order, of block size, and of how the population
// is partitioned across devices -- properties the stream RNG never had. The
// antithetic pairing is unchanged: particle 2j and particle 2j+1 receive equal
// and opposite thermal velocities, so the thermal sample carries exactly zero
// bulk momentum and an odd population gives its last particle a zero draw.

// A rank-1 lattice over the block. `stride` must be coprime with `count`; the
// caller picks it (a golden-ratio choice adjusted upward to coprimality) since
// that is O(1) integer work.
struct PicQuietStartSpec {
  std::uint64_t count{0};
  std::uint64_t stride{1};
  Real x_min{0}, x_max{1}, y_min{0}, y_max{1};
  // Cylindrical blocks sample uniformly in r^2 rather than r, so an equal
  // weight is an equal ring volume.
  int cylindrical{0};
  // The physical domain. A block inside the domain puts every lattice point
  // inside it too, up to the rounding of one multiply-add per coordinate, so
  // the check is still made per particle rather than on the four block bounds.
  Real domain_x_min{0}, domain_x_max{1}, domain_y_min{0}, domain_y_max{1};
};

struct PicMaxwellianSpec {
  Real thermal_speed{0};
  Real drift_x{0}, drift_y{0}, drift_z{0};
  std::uint64_t seed{0};
  std::uint64_t species_key{0};
};

// v += sin(2*pi*(mx*(x-ox)/lx + my*(y-oy)/ly) + phase) * amplitude.
struct PicVelocityPerturbationSpec {
  int active{0};
  Real mode_x{0}, mode_y{0}, phase{0};
  Real amplitude_x{0}, amplitude_y{0}, amplitude_z{0};
  Real origin_x{0}, origin_y{0}, lx{1}, ly{1};
};

// -- Seeded initial field -----------------------------------------------------
//
// Status bit from the field-seeding kernels. Separate from the particle
// sampling word below because the two report different objects and the callers
// raise different exceptions.
inline constexpr int kPicSeedFieldNotFinite = 1;

// Evaluate one deck generator over the padded Yee lattices. Writes every one of
// the six components: the generator's own get their profile, the rest are
// zeroed, matching the host path's fresh-zero-buffer-per-component behaviour.
void launch_pic_seed_initial_fields(const PicInitialFieldSpec& spec,
                                    YeeField2D<Real>& fields, int* status,
                                    backend::stream_t stream);

// Cylindrical standing-mode magnetic half step. MUST run after the caller has
// filled the field ghosts: the radial stencil reads them, which is what makes
// the boundary parity the configured closure's rather than a private copy.
void launch_pic_seed_radial_half_step(const PicRadialHalfStepSpec& spec,
                                      YeeField2D<Real>& fields, int* status,
                                      backend::stream_t stream);

// Status bits from the sampling kernels.
inline constexpr int kPicSampleNonFinitePosition = 1;
inline constexpr int kPicSampleNonFiniteVelocity = 2;
inline constexpr int kPicSampleSuperluminal = 4;
inline constexpr int kPicSampleOutsideDomain = 8;

void launch_pic_quiet_positions(PicQuietStartSpec spec, Real* x, Real* y,
                                int* status, backend::stream_t stream);

void launch_pic_maxwellian_velocities(std::uint64_t count,
                                      PicMaxwellianSpec spec, Real* vx,
                                      Real* vy, Real* vz, int* status,
                                      backend::stream_t stream);

void launch_pic_velocity_perturbation(std::uint64_t count,
                                      PicVelocityPerturbationSpec spec,
                                      const Real* x, const Real* y, Real* vx,
                                      Real* vy, Real* vz, int* status,
                                      backend::stream_t stream);

// |v| >= 1 leaves the nonrelativistic Boris model. Checked here so the sampled
// velocities never have to come back to the host to be inspected.
void launch_pic_check_subluminal(std::uint64_t count, const Real* vx,
                                 const Real* vy, const Real* vz, int* status,
                                 backend::stream_t stream);

// Every LIVE particle must have a finite position inside the closed physical
// domain [x_lo,x_hi] x [y_lo,y_hi]. Dead slots are skipped.
//
// This is the admissibility gate a species must pass before its initial charge
// is deposited: a live centre outside the domain would have its shape tail
// silently clipped or folded by a boundary it has not actually crossed.
// Boundary points themselves are admissible -- periodic high faces alias low,
// and wall faces are physical -- so the comparisons are inclusive.
//
// It exists so the check can read the resident planes instead of downloading
// all eleven of them to look at two, which is what the host predicate it
// replaces did.
void launch_pic_check_initial_domain(std::uint64_t count, const Real* x,
                                     const Real* y,
                                     const std::uint8_t* alive, Real x_lo,
                                     Real x_hi, Real y_lo, Real y_hi,
                                     int* status, backend::stream_t stream);

// Fill the planes a seeded species needs beyond position and velocity:
// previous positions equal to the initial ones, vphi_deposit equal to vz, a
// uniform macro weight, alive flags, and identity ids.
void launch_pic_finalize_seed(std::uint64_t count, Real weight, const Real* x,
                              const Real* y, const Real* vz, Real* x_prev,
                              Real* y_prev, Real* vphi_deposit, Real* weights,
                              std::uint8_t* alive, std::uint64_t* id,
                              backend::stream_t stream);

}  // namespace quasar::pic

// The field-data ABI is phrased in quasar::Real (and YeeField2D<Real> /
// JField2D<Real>), not literal double, so the kernel boundary tracks the same
// precision typedef as the rest of the PIC stack: if Real ever changes, the
// solver's Real* device pointers cannot silently mismatch a double* ABI slot.
extern "C" {

// -- FDTD field updates ------------------------------------------------------
void launch_pic_fdtd_b_order2(const quasar::Grid2D&, quasar::Real*, quasar::Real*,
                              quasar::Real*, const quasar::Real*, const quasar::Real*,
                              const quasar::Real*, quasar::Real, quasar_stream_t);
void launch_pic_fdtd_b_order4(const quasar::Grid2D&, quasar::Real*, quasar::Real*,
                              quasar::Real*, const quasar::Real*, const quasar::Real*,
                              const quasar::Real*, quasar::Real, quasar_stream_t);
void launch_pic_fdtd_e_order2(const quasar::Grid2D&, quasar::Real*, quasar::Real*,
                              quasar::Real*, const quasar::Real*, const quasar::Real*,
                              const quasar::Real*, const quasar::Real*, const quasar::Real*,
                              const quasar::Real*, quasar::Real, quasar_stream_t);
void launch_pic_fdtd_e_order4(const quasar::Grid2D&, quasar::Real*, quasar::Real*,
                              quasar::Real*, const quasar::Real*, const quasar::Real*,
                              const quasar::Real*, const quasar::Real*, const quasar::Real*,
                              const quasar::Real*, quasar::Real, quasar_stream_t);

// -- Cylindrical (r,z) m=0 FDTD field updates --------------------------------
// Axisymmetric counterparts of launch_pic_fdtd_{b,e}: the curls use the
// 1/r and (1/r) d(r .)/dr radial operators with the on-axis (i=0) regularized
// closure, reading the radius from Grid2D's r_at_* helpers.  The order-four
// radial divergence is algebraically D(A)/dr+M(A)/r: it is equivalent to
// applying the staggered derivative to r*A and dividing by the cell-centre
// radius, but never materializes the potentially overflowing product r*A.
void launch_pic_fdtd_b_cyl_order2(const quasar::Grid2D&, quasar::Real*, quasar::Real*,
                                  quasar::Real*, const quasar::Real*, const quasar::Real*,
                                  const quasar::Real*, quasar::Real, quasar_stream_t);
void launch_pic_fdtd_b_cyl_order4(const quasar::Grid2D&, quasar::Real*, quasar::Real*,
                                  quasar::Real*, const quasar::Real*, const quasar::Real*,
                                  const quasar::Real*, quasar::Real, quasar_stream_t);
void launch_pic_fdtd_e_cyl_order2(const quasar::Grid2D&, quasar::Real*, quasar::Real*,
                                  quasar::Real*, const quasar::Real*, const quasar::Real*,
                                  const quasar::Real*, const quasar::Real*, const quasar::Real*,
                                  const quasar::Real*, quasar::Real, quasar_stream_t);
void launch_pic_fdtd_e_cyl_order4(const quasar::Grid2D&, quasar::Real*, quasar::Real*,
                                  quasar::Real*, const quasar::Real*, const quasar::Real*,
                                  const quasar::Real*, const quasar::Real*, const quasar::Real*,
                                  const quasar::Real*, quasar::Real, quasar_stream_t);

// -- Cylindrical (r,z) particle gather + push --------------------------------
// Boris push in (r,z) with the vz ABI slot carrying v_phi and the coordinate
// (centrifugal/azimuthal) terms applied across r. `periodic_x`/`periodic_y`
// mirror the Cartesian gather flags: the axial (y) axis can still be periodic
// while the radial axis at i=0 is on-axis (never periodic).
void launch_pic_gather_push_cyl_shape1(const quasar::Grid2D&, quasar::pic::ParticleSpecies&,
                                       const quasar::YeeField2D<quasar::Real>&,
                                       const quasar::YeeField2D<quasar::Real>&,
                                       const quasar::BField2D<quasar::Real>&, int periodic_x,
                                       int periodic_y, quasar::Real force_dt,
                                       quasar::Real position_dt,
                                       quasar::Real previous_b_weight,
                                       quasar::Real current_b_weight, quasar_stream_t);
void launch_pic_gather_push_cyl_shape2(const quasar::Grid2D&, quasar::pic::ParticleSpecies&,
                                       const quasar::YeeField2D<quasar::Real>&,
                                       const quasar::YeeField2D<quasar::Real>&,
                                       const quasar::BField2D<quasar::Real>&, int periodic_x,
                                       int periodic_y, quasar::Real force_dt,
                                       quasar::Real position_dt,
                                       quasar::Real previous_b_weight,
                                       quasar::Real current_b_weight, quasar_stream_t);

// -- Cylindrical (r,z) current deposition ------------------------------------
// Charge-conserving Esirkepov deposit with ring/volume weights proportional to
// the radius (cell_volume(i)), so the discrete cylindrical continuity residual
// stays within tolerance. `periodic_x`/`periodic_y` mirror the Cartesian flags.
void launch_pic_deposit_cyl_shape1(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                                   quasar::JField2D<quasar::Real>&, quasar::Real, int periodic_x,
                                   int periodic_y, quasar_stream_t);
void launch_pic_deposit_cyl_shape2(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                                   quasar::JField2D<quasar::Real>&, quasar::Real, int periodic_x,
                                   int periodic_y, quasar_stream_t);
void launch_pic_charge_cyl_shape1(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                                  quasar::ScalarGrid2D<quasar::Real>&, int periodic_x,
                                  int periodic_y, quasar_stream_t);
void launch_pic_charge_cyl_shape2(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                                  quasar::ScalarGrid2D<quasar::Real>&, int periodic_x,
                                  int periodic_y, quasar_stream_t);

// -- Cylindrical on-axis (r=0) boundary --------------------------------------
// Field closure: enforces the r=0 parity of the field components in the i=0
// ghost/edge so the 1/r curls stay regular on the axis. Run after each curl
// (and as a ghost fill) on the x_lo side only.
void launch_pic_boundary_axis_fields(const quasar::Grid2D&,
                                     quasar::YeeField2D<quasar::Real>&, quasar_stream_t);
// Particle closure: folds particles that cross r=0 back into the domain
// (r -> -r, vr -> -vr), keeping the on-axis approach reflectionless. Run on the
// x_lo side only.
void launch_pic_boundary_axis_particles(const quasar::Grid2D&,
                                        quasar::pic::ParticleSpecies&, quasar_stream_t);

// -- Particle gather + push --------------------------------------------------
// `periodic_x`/`periodic_y` (0/1) select per-axis field-gather indexing: a
// periodic axis wraps; a non-periodic (wall) axis reads the padded interpolation
// stencil (the evolved field's boundary closure and the prescribed field sampled
// at those same ghost coordinates), clamping only at the allocation limit for
// low-level caller safety instead of wrapping to the far edge. Mirrors the
// deposit.
void launch_pic_gather_push_shape1(const quasar::Grid2D&, quasar::pic::ParticleSpecies&,
                                   const quasar::YeeField2D<quasar::Real>&,
                                   const quasar::YeeField2D<quasar::Real>&,
                                   const quasar::BField2D<quasar::Real>&, int periodic_x,
                                   int periodic_y, quasar::Real force_dt,
                                   quasar::Real position_dt,
                                   quasar::Real previous_b_weight,
                                   quasar::Real current_b_weight, quasar_stream_t);
void launch_pic_gather_push_shape2(const quasar::Grid2D&, quasar::pic::ParticleSpecies&,
                                   const quasar::YeeField2D<quasar::Real>&,
                                   const quasar::YeeField2D<quasar::Real>&,
                                   const quasar::BField2D<quasar::Real>&, int periodic_x,
                                   int periodic_y, quasar::Real force_dt,
                                   quasar::Real position_dt,
                                   quasar::Real previous_b_weight,
                                   quasar::Real current_b_weight, quasar_stream_t);
// Snapshot B^{n-1/2} immediately before Faraday advances it to B^{n+1/2}.
void launch_pic_copy_b(const quasar::Grid2D&, const quasar::YeeField2D<quasar::Real>&,
                       quasar::BField2D<quasar::Real>&, quasar_stream_t);

// -- Current deposition ------------------------------------------------------
// `periodic_x`/`periodic_y` (0/1) select per-axis node indexing: a periodic axis
// wraps (historical behaviour), a non-periodic (wall) axis deposits into ghost
// cells without wrapping so launch_pic_boundary_specular_foldback can reflect the
// boundary-crossing current back into the interior.
void launch_pic_deposit_shape1(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                               quasar::JField2D<quasar::Real>&, quasar::Real, int periodic_x,
                               int periodic_y, quasar_stream_t);
void launch_pic_deposit_shape2(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                               quasar::JField2D<quasar::Real>&, quasar::Real, int periodic_x,
                               int periodic_y, quasar_stream_t);
void launch_pic_charge_shape1(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                              quasar::ScalarGrid2D<quasar::Real>&, int periodic_x,
                              int periodic_y, quasar_stream_t);
void launch_pic_charge_shape2(const quasar::Grid2D&, const quasar::pic::ParticleSpecies&,
                              quasar::ScalarGrid2D<quasar::Real>&, int periodic_x,
                              int periodic_y, quasar_stream_t);
void launch_pic_add_uniform_charge(const quasar::Grid2D&,
                                   quasar::ScalarGrid2D<quasar::Real>&,
                                   quasar::Real density, quasar_stream_t);
// Synchronously rejects non-finite source values after all deposit-side
// transformations (wall foldback, filtering/correction, periodic-face restore,
// and background addition). Either source pointer may be null; `error` is a
// caller-owned one-element device buffer that is reset on every invocation.
void launch_pic_validate_finite_sources(
    const quasar::Grid2D&,
    const quasar::JField2D<quasar::Real>* current,
    const quasar::ScalarGrid2D<quasar::Real>* charge,
    unsigned int* error, quasar_stream_t);
// The fourth-order staggered derivative factors as D4+ = D2+ S with
// S=(13/12)I-(1/24)(T+ + T-). This solves Sx*Jx=Jx_raw and Sy*Jy=Jy_raw after
// the ordinary Esirkepov prefix deposit, making its forward D2 continuity
// identity exactly compatible with the order-four Ampere/Gauss operator.
void launch_pic_current_correct_order4(const quasar::Grid2D&,
                                       quasar::JField2D<quasar::Real>&,
                                       quasar::Real* rhs_x, quasar::Real* rhs_y,
                                       quasar::Real* iter_x, quasar::Real* iter_y,
                                       int x_lo_mode, int x_hi_mode,
                                       int y_lo_mode, int y_hi_mode,
                                       quasar_stream_t);
// Boundary modes used by both compact correction launchers are 0=periodic,
// 1=even normal-E continuation (PEC), 2=linear continuation (outflow),
// 3=the cylindrical r=0 axis, and 4=an exchanged internal-tile guard. The
// correction must use the same continuation as the field ghost closure or
// D4(J_corrected)=D2(J_raw) fails at boundary cells.
// Cylindrical variant solves the radial compact system without materialising
// r*Jr, and the axial system for Jz.
void launch_pic_current_correct_cyl_order4(const quasar::Grid2D&,
                                           quasar::JField2D<quasar::Real>&,
                                           quasar::Real* rhs_r, quasar::Real* rhs_z,
                                           quasar::Real* iter_r, quasar::Real* iter_z,
                                           int r_lo_mode, int r_hi_mode,
                                           int z_lo_mode, int z_hi_mode,
                                           quasar_stream_t);
// Restore the duplicate physical high normal-current faces on periodic axes.
// Must run after every filter/order correction and immediately before Ampere.
void launch_pic_current_periodic_high_faces(
    const quasar::Grid2D&, quasar::JField2D<quasar::Real>&,
    int periodic_x, int periodic_y, quasar_stream_t);
// Reads + clears the species' persistent deposit-error flag and throws a
// std::runtime_error if a deposit had an unrepresentable input/result or spilled
// outside its fixed window. This is deliberately separate from the source scan
// above, which covers arithmetic performed after the atomic deposits complete.
void launch_pic_deposit_overflow_check(const quasar::pic::ParticleSpecies&,
                                       quasar_stream_t);
// Drains the gather/push sticky state-error flag and throws synchronously. The
// public pusher calls this after every launch so invalid state never reaches a
// boundary/deposit kernel, including for direct C++ callers outside EmPic2D3V.
void launch_pic_particle_error_check(const quasar::pic::ParticleSpecies&,
                                     quasar_stream_t);

// -- Current filtering -------------------------------------------------------
// `scratch` is caller-owned ping-pong storage of at least grid.storage_size()
// Real values, hoisted out of the per-step path. Only `jz` is smoothed: in 2D it
// does not enter charge continuity, whereas convolving the in-plane Jx/Jy pair
// without applying the same operator to both endpoint charge densities would
// change div(J) and violate Gauss's law. Jx/Jy are therefore left untouched, and
// one strip is enough. `periodic_x`/`periodic_y` select whether the smoothing
// stencil wraps on that axis; non-periodic axes clamp at the edge so a filter
// cannot couple current across a wall.
void launch_pic_filter_binomial(const quasar::Grid2D&, quasar::JField2D<quasar::Real>&,
                                quasar::Real* scratch, int, int periodic_x, int periodic_y,
                                int cylindrical, quasar_stream_t);
void launch_pic_filter_compensated(const quasar::Grid2D&, quasar::JField2D<quasar::Real>&,
                                   quasar::Real* scratch, int, int periodic_x, int periodic_y,
                                   int cylindrical, quasar_stream_t);

// -- Particle boundary conditions --------------------------------------------
void launch_pic_boundary_absorb_particles(const quasar::Grid2D&,
                                          quasar::pic::ParticleSpecies&, int, quasar_stream_t);
void launch_pic_boundary_prepare_absorb(const quasar::Grid2D&,
                                        quasar::pic::ParticleSpecies&, int side,
                                        int shape_order, quasar_stream_t);
void launch_pic_boundary_specular_particles(const quasar::Grid2D&,
                                            quasar::pic::ParticleSpecies&, int, quasar_stream_t);
// Reflects current deposited into one reflecting side's ghost cells back into the
// interior (image-charge fold) and zeroes those ghosts. Run after the deposit and
// before the current filter / E-update on every specular side.
void launch_pic_boundary_specular_foldback(const quasar::Grid2D&,
                                           quasar::JField2D<quasar::Real>&, int side,
                                           int cylindrical, quasar_stream_t);
void launch_pic_boundary_specular_foldback_charge(
    const quasar::Grid2D&, quasar::ScalarGrid2D<quasar::Real>&, int side,
    int cylindrical, quasar_stream_t);
// `side` is the Side enum (0=x_lo,1=x_hi,2=y_lo,3=y_hi). A per-side periodic BC
// wraps only particles that cross that side, so a one-sided periodic wall does
// not hide exits through the opposite non-periodic wall.
void launch_pic_boundary_periodic_particles(const quasar::Grid2D&,
                                            quasar::pic::ParticleSpecies&, int side,
                                            quasar_stream_t);

// -- Field boundary conditions -----------------------------------------------
void launch_pic_boundary_periodic_fields(const quasar::Grid2D&,
                                         quasar::YeeField2D<quasar::Real>&,
                                         int, quasar_stream_t);

// Stagger-aware non-periodic ghost closures. PEC applies the exact even/odd
// continuation at the physical wall; outflow supplies linear halo continuation
// before its characteristic tangential-E update. `cylindrical` selects the
// radial staggering of the axisymmetric scheme.
void launch_pic_boundary_pec_fields(const quasar::Grid2D&,
                                    quasar::YeeField2D<quasar::Real>&,
                                    int side, int cylindrical,
                                    quasar_stream_t);
void launch_pic_boundary_outflow_fill_fields(const quasar::Grid2D&,
                                             quasar::YeeField2D<quasar::Real>&,
                                             int side, int cylindrical,
                                             quasar_stream_t);
// Backward-compatible internal cylindrical spellings.
void launch_pic_boundary_cyl_pec_fields(const quasar::Grid2D&,
                                        quasar::YeeField2D<quasar::Real>&,
                                        int side, quasar_stream_t);
void launch_pic_boundary_cyl_outflow_fields(const quasar::Grid2D&,
                                            quasar::YeeField2D<quasar::Real>&,
                                            int side, quasar_stream_t);

// First-order Mur characteristic update for the two tangential-E component
// lattices. `stride` is max tangential line length (normal-axis-independent),
// and `cylindrical` enables sqrt(r)-scaled radial characteristics.
void launch_pic_boundary_outflow_correct_e(
    const quasar::Grid2D&, quasar::YeeField2D<quasar::Real>&, int side,
    quasar::Real dt, quasar::Real* mur_strips, int stride, int init,
    int skip_lo, int skip_hi, int cylindrical, quasar_stream_t);
// Diagonal Mur closure for a corner shared by two outflow sides. `corner_mask`
// bits enumerate (xlo,ylo), (xhi,ylo), (xlo,yhi), (xhi,yhi).
void launch_pic_boundary_outflow_corners(const quasar::Grid2D&,
                                         quasar::YeeField2D<quasar::Real>&,
                                         quasar::Real dt, unsigned int corner_mask,
                                         quasar::Real* history, int init,
                                         int cylindrical,
                                         quasar_stream_t);

// -- Particle array compaction -----------------------------------------------
// Compacts alive particles to the front of every species array (order is not
// preserved — irrelevant for PIC) and shrinks the active count. Returns the new
// alive count via set_count() on the species.
void launch_pic_particle_compact(quasar::pic::ParticleSpecies&, quasar_stream_t);

// -- Alive-particle count ----------------------------------------------------
// Single-pass device reduction of the alive flags; returns the count to the
// host. Cheaper than a full HostSnapshot when only the scalar is needed.
std::size_t launch_pic_alive_count(const quasar::pic::ParticleSpecies&, quasar_stream_t);

// Counts live particles outside a tile's half-open ownership box. Physical
// high faces include their exact endpoint; internal high faces do not. The
// scalar reduction lets the distributed runtime skip full particle snapshots
// on the overwhelmingly common no-migration step.
std::size_t launch_pic_particle_departure_count(
    const quasar::pic::ParticleSpecies&, int include_x_high,
    int include_y_high, quasar_stream_t);

// -- Prescribed external-field sampling --------------------------------------
//
// Backing kernels for pic::sample_external_field, in
// src/backend/hip/pic/external_field_hip.hip. Sample points, evaluator answers
// and materialized components are all indexed in Grid2D::index(i, j) order over
// the padded extent, so the mapping stage is a flat elementwise map.
//
// The checking entry points cannot throw, so each ORs bits into a zeroed `int*`
// with an integer atomic -- exact and order-independent, hence independent of
// the launch geometry -- and the host raises the matching exception. Each bit
// set is documented on its own launcher.

// Builds one component's Yee-lattice sample points over the whole padded
// lattice, already scaled from internal length units to SI. Offsets are in
// cells from the domain origin (0 = face/node, 1/2 = centre).
//
// The padded extent is evaluated, not just the physical subset: a wall-adjacent
// finite-size gather legitimately reads the boundary-filled ghosts of the
// evolved field, so the prescribed field must supply values at those same
// coordinates. Edge replication would reduce a nonuniform analytic or file
// field to a first-order constant continuation.
//
// Status bits: 1 = a scaled sample coordinate is not finite;
//              2 = a nonzero coordinate underflowed under length scaling.
void launch_pic_yee_points(const quasar::Grid2D& g, quasar::Real offset_x,
                           quasar::Real offset_y, quasar::Real length_scale,
                           int plane_is_xz, quasar::Real* px, quasar::Real* py,
                           quasar::Real* pz, int* status, quasar_stream_t);

// Rotates a sample-point set about the configured symmetry axis, for the
// cylindrical covariance probes. Status bit 1 = a rotated coordinate is not
// finite.
void launch_pic_rotate_points(int plane_is_xz, quasar::Real cosine,
                              quasar::Real sine, int exact_quarter_turn,
                              const quasar::Real* px, const quasar::Real* py,
                              const quasar::Real* pz, int M,
                              quasar::Real* ox, quasar::Real* oy,
                              quasar::Real* oz, int* status, quasar_stream_t);

// Checks that the field sampled on the rotated points equals the rotation of
// the field sampled on the meridional points. Status bits:
//   1 = the evaluator returned a non-finite field;
//   2 = the field is not rotationally covariant about the symmetry axis.
void launch_pic_check_rotational_covariance(
    int plane_is_xz, quasar::Real cosine, quasar::Real sine,
    int exact_quarter_turn,
    const quasar::Real* mx, const quasar::Real* my, const quasar::Real* mz,
    const quasar::Real* rx, const quasar::Real* ry, const quasar::Real* rz,
    int M, quasar::Real relative_tolerance, quasar::Real absolute_floor,
    int* status, quasar_stream_t);

// Maps one PIC-frame component out of the evaluator's lab-frame planes: reads
// lab axis `axis` (0=x, 1=y, 2=z) and scales by `sign` and 1/field_scale.
// Status bits: 1 = the evaluator returned a non-finite field;
//              2 = the scaled value is not finite;
//              4 = a nonzero field value underflowed solver units.
void launch_pic_materialize_external_component(
    const quasar::Real* vx, const quasar::Real* vy, const quasar::Real* vz,
    int total, int axis, quasar::Real sign, quasar::Real field_scale,
    quasar::Real* out, int* status, quasar_stream_t);

// Verifies and then enforces radial parity on one materialized component of a
// cylindrical prescribed field. `face_odd` selects odd parity with an exact
// zero on the axis face; otherwise even parity about the first cell centre.
// Ghosts are overwritten with the exact mirrored value after the check, so the
// prescribed halo is bit-for-bit consistent with the closure the evolved field
// uses. Status bits: 1 = the axis face does not vanish;
//                    2 = the radial parity is violated.
void launch_pic_enforce_axis_parity(const quasar::Grid2D& g,
                                    quasar::Real* plane, int face_odd,
                                    quasar::Real relative_tolerance,
                                    int* status, quasar_stream_t);

// Checks trace(grad B) = 0 pointwise on a component-major 9*M Jacobian. This is
// Maxwell's continuous constraint on a prescribed force-only field, deliberately
// not a discrete Yee divergence: a smooth divergence-free field must not be
// rejected merely because samples of it carry the stencil's truncation error.
// Status bits: 1 = a non-finite gradient entry; 2 = the trace does not vanish.
void launch_pic_check_continuous_solenoidality(
    const quasar::Real* G, int M, quasar::Real relative_tolerance, int* status,
    quasar_stream_t);

// Sets `*flag` to 1 if any component of the sampled field is nonzero.
void launch_pic_field_has_nonzero(const quasar::Real* x, const quasar::Real* y,
                                  const quasar::Real* z, int M, int* flag,
                                  quasar_stream_t);

}  // extern "C"
