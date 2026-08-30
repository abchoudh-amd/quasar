#pragma once

// Device-resident initial sampling for one PIC species.
//
// This replaces the NumPy block that used to sit in `quasar.pic.cli`: a rank-1
// quiet-start lattice, a Maxwellian velocity sample, an optional sinusoidal
// perturbation, a speed check and a uniform macro weight, all evaluated on the
// host for every particle and then uploaded. Nothing here touches host memory
// per particle. What remains on the host is O(1) configuration arithmetic --
// the lattice stride, the per-particle block measure, the thermal speed -- and
// that is deliberate: those are single scalars derived from the deck, not work
// that scales with the population.
//
// One behavioural change comes with the move, and it is not cosmetic. The
// velocity sample is now Philox4x32-10 counted by particle index rather than
// `numpy.random.default_rng`'s PCG64 stream, so seeded decks draw different
// velocities than they did before. See kernels.hpp for why a stream generator
// cannot be evaluated in parallel at an arbitrary index, and CHANGELOG.md for
// the affected references.

#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"
#include "quasar/physics/pic/species.hpp"

#include <cstdint>

namespace quasar::pic {

// Everything one species needs, in the solver's units.
struct ParticleSampleConfig {
  std::size_t count{0};

  // Block bounds: (x, y) on a Cartesian grid, (r, z) on a cylindrical one. A
  // cylindrical block samples uniformly in r^2 so an equal weight is an equal
  // ring volume.
  Real x_min{0}, x_max{1};
  Real y_min{0}, y_max{1};
  bool cylindrical{false};

  Real thermal_speed{0};
  Real drift_x{0}, drift_y{0}, drift_z{0};
  // Philox key. `species_key` separates species drawn under one deck seed.
  std::uint64_t seed{0};
  std::uint64_t species_key{0};

  // Optional velocity perturbation:
  //   v += sin(2*pi*(mx*(x-ox)/lx + my*(y-oy)/ly) + phase) * amplitude.
  bool perturb{false};
  Real mode_x{0}, mode_y{0}, phase{0};
  Real amplitude_x{0}, amplitude_y{0}, amplitude_z{0};
  Real domain_origin_x{0}, domain_origin_y{0};
  Real domain_lx{1}, domain_ly{1};

  // Uniform macro weight: density times the per-particle block measure.
  Real weight{0};
};

// Fill `species` with `config.count` quiet-start particles at t = 0.
//
// Velocities are the physical distribution at t = 0 and are NOT pre-staggered;
// the solver owns the initial half-step Boris kick.
//
// Throws std::invalid_argument for a malformed configuration, and for a sample
// the kernels reject: a non-finite coordinate or velocity, or a speed at or
// above c = 1, which leaves the nonrelativistic Boris model. Those three are
// the same rejections `ParticleSpecies::set_host_particles` performed on the
// uploaded arrays, moved to where the values are produced.
void sample_species(ParticleSpecies& species,
                    const ParticleSampleConfig& config,
                    backend::stream_t stream = nullptr);

// The rank-1 lattice stride: the golden-ratio multiple of `count`, raised to
// the next value coprime with it. O(1) integer work, so it stays on the host --
// every particle needs the same answer.
[[nodiscard]] std::uint64_t quiet_start_stride(std::uint64_t count);

// Area (Cartesian) or ring volume (cylindrical) represented by each
// equal-weight particle. A single scalar, evaluated through a scaled product so
// a valid result is not lost to a false intermediate overflow.
[[nodiscard]] Real quiet_block_measure(const ParticleSampleConfig& config);

}  // namespace quasar::pic
