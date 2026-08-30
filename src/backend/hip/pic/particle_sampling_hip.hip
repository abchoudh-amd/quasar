// Quiet-start positions and Maxwellian velocities, sampled on the device.
//
// Positions come from a rank-1 lattice: coordinate one walks the midpoint
// strata in index order and coordinate two walks them in the order given by a
// stride coprime with the count, so both coordinates use every stratum exactly
// once and the block's centre of mass is exact for every N. Nothing in that is
// sequential, which is why it ports directly.
//
// Velocities come from Philox4x32-10, a counter-based generator: the draw for
// particle p is a pure function of (seed, species, p) rather than of a position
// in a stream. That is the property the host `np.random.default_rng` sample did
// not have and the reason the values change with this port -- a stateful PCG64
// stream cannot be evaluated at an arbitrary index without advancing through
// it. The counter form also makes the sample independent of block size, of
// thread scheduling, and of how the population is later split across devices.
//
// Normals are Box-Muller. Two Philox calls per particle-pair give four
// uniforms and hence four normals, of which three are used. Box-Muller rather
// than a ziggurat because it is a closed form with no rejection loop and no
// table, so every particle costs the same and the result does not depend on how
// many rejections a neighbouring lane happened to take.

#include "quasar/backend/device.hpp"
#include "quasar/physics/pic/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace quasar::pic {
namespace {

constexpr int kBlock = 256;

// -- Philox4x32-10 -----------------------------------------------------------

__device__ inline void philox_round(std::uint32_t ctr[4],
                                    const std::uint32_t key[2]) {
  constexpr std::uint32_t kMul0 = 0xD2511F53u;
  constexpr std::uint32_t kMul1 = 0xCD9E8D57u;
  const std::uint64_t product0 =
      static_cast<std::uint64_t>(kMul0) * static_cast<std::uint64_t>(ctr[0]);
  const std::uint64_t product1 =
      static_cast<std::uint64_t>(kMul1) * static_cast<std::uint64_t>(ctr[2]);
  const std::uint32_t low0 = static_cast<std::uint32_t>(product0);
  const std::uint32_t high0 = static_cast<std::uint32_t>(product0 >> 32);
  const std::uint32_t low1 = static_cast<std::uint32_t>(product1);
  const std::uint32_t high1 = static_cast<std::uint32_t>(product1 >> 32);
  const std::uint32_t out0 = high1 ^ ctr[1] ^ key[0];
  const std::uint32_t out1 = low1;
  const std::uint32_t out2 = high0 ^ ctr[3] ^ key[1];
  const std::uint32_t out3 = low0;
  ctr[0] = out0;
  ctr[1] = out1;
  ctr[2] = out2;
  ctr[3] = out3;
}

// Ten rounds, the key bumped by the two Weyl constants between them. This is
// the standard Random123 schedule; the constants are part of the algorithm.
__device__ inline void philox4x32_10(std::uint32_t ctr[4],
                                     std::uint32_t key[2]) {
  constexpr std::uint32_t kWeyl0 = 0x9E3779B9u;
  constexpr std::uint32_t kWeyl1 = 0xBB67AE85u;
  for (int round = 0; round < 10; ++round) {
    if (round > 0) {
      key[0] += kWeyl0;
      key[1] += kWeyl1;
    }
    philox_round(ctr, key);
  }
}

// One draw of four 32-bit words for the given counter, keyed on the seed and
// the species. Both halves of the 64-bit seed and species key are used, so two
// seeds differing only in their high word give different streams.
__device__ inline void philox_draw(std::uint64_t seed, std::uint64_t species,
                                   std::uint64_t counter, std::uint32_t lane,
                                   std::uint32_t out[4]) {
  std::uint32_t key[2] = {static_cast<std::uint32_t>(seed),
                          static_cast<std::uint32_t>(seed >> 32)};
  key[0] ^= static_cast<std::uint32_t>(species);
  key[1] ^= static_cast<std::uint32_t>(species >> 32);
  out[0] = static_cast<std::uint32_t>(counter);
  out[1] = static_cast<std::uint32_t>(counter >> 32);
  out[2] = lane;
  out[3] = 0x517CC1B7u;
  philox4x32_10(out, key);
}

// Open interval (0, 1). The +0.5 keeps both endpoints out, which matters
// because Box-Muller takes a logarithm of the first uniform.
__device__ inline double uniform_open(std::uint32_t high, std::uint32_t low) {
  const std::uint64_t bits =
      (static_cast<std::uint64_t>(high) << 21) ^ (low >> 11);
  const std::uint64_t mantissa = bits & ((1ull << 53) - 1ull);
  return (static_cast<double>(mantissa) + 0.5) * (1.0 / 9007199254740992.0);
}

__device__ inline void box_muller(double u1, double u2, double* z0,
                                  double* z1) {
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  const double radius = ::sqrt(-2.0 * ::log(u1));
  double sine = 0.0;
  double cosine = 0.0;
  ::sincos(kTwoPi * u2, &sine, &cosine);
  *z0 = radius * cosine;
  *z1 = radius * sine;
}

// Three standard normals for pair index `pair`.
__device__ inline void normals3(std::uint64_t seed, std::uint64_t species,
                                std::uint64_t pair, double out[3]) {
  std::uint32_t words[4];
  philox_draw(seed, species, pair, 0u, words);
  double z0 = 0.0;
  double z1 = 0.0;
  box_muller(uniform_open(words[0], words[1]),
             uniform_open(words[2], words[3]), &z0, &z1);
  out[0] = z0;
  out[1] = z1;

  // A second draw on the same counter but a different lane word. Box-Muller
  // yields normals in pairs, so the fourth is generated and discarded rather
  // than carried to the next particle -- carrying it would make a particle's
  // draw depend on its neighbour's, which is exactly the stream dependence this
  // generator exists to avoid.
  philox_draw(seed, species, pair, 1u, words);
  double z2 = 0.0;
  double unused = 0.0;
  box_muller(uniform_open(words[0], words[1]),
             uniform_open(words[2], words[3]), &z2, &unused);
  out[2] = z2;
}

// -- Kernels -----------------------------------------------------------------

__global__ void quiet_positions_kernel(PicQuietStartSpec spec,
                                       double* __restrict__ x,
                                       double* __restrict__ y,
                                       int* __restrict__ status) {
  const std::uint64_t k =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= spec.count) return;
  const double inverse_count = 1.0 / static_cast<double>(spec.count);
  const std::uint64_t rank = (spec.stride * k) % spec.count;
  const double u = (static_cast<double>(k) + 0.5) * inverse_count;
  const double v = (static_cast<double>(rank) + 0.5) * inverse_count;

  double px = 0.0;
  double py = 0.0;
  if (spec.cylindrical != 0) {
    // Uniform in r^2: r^2 - r_min^2 = u * (r_max^2 - r_min^2). Formed through
    // the ratio r_min/r_max so a thin annulus at large radius keeps
    // (r_max - r_min) even when the ratio rounds to one, and so r^2 never
    // overflows. Same algebra the host helper used.
    const double radial_extent = spec.x_max - spec.x_min;
    const double ratio = spec.x_min / spec.x_max;
    const double sum_factor = 1.0 + ratio;
    const double scaled_increment = (radial_extent / spec.x_max) * sum_factor * u;
    const double scaled_radius = ::sqrt(ratio * ratio + scaled_increment);
    px = spec.x_min + radial_extent * sum_factor * u / (scaled_radius + ratio);
    py = spec.y_min + v * (spec.y_max - spec.y_min);
  } else {
    // Lower bound plus a fraction of the extent, not a convex combination of
    // the bounds: on a large translated domain the convex form forms two
    // origin-sized products and loses their separation.
    px = spec.x_min + u * (spec.x_max - spec.x_min);
    py = spec.y_min + v * (spec.y_max - spec.y_min);
  }

  if (!(::isfinite(px) && ::isfinite(py))) {
    atomicOr(status, kPicSampleNonFinitePosition);
  } else if (px < spec.domain_x_min || px > spec.domain_x_max ||
             py < spec.domain_y_min || py > spec.domain_y_max) {
    atomicOr(status, kPicSampleOutsideDomain);
  }
  x[k] = px;
  y[k] = py;
}

__global__ void maxwellian_kernel(std::uint64_t count, PicMaxwellianSpec spec,
                                  double* __restrict__ vx,
                                  double* __restrict__ vy,
                                  double* __restrict__ vz,
                                  int* __restrict__ status) {
  const std::uint64_t p =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= count) return;

  double thermal[3] = {0.0, 0.0, 0.0};
  const std::uint64_t pairs = count / 2;
  if (p < 2 * pairs) {
    // Antithetic pairing: 2j takes the draw, 2j+1 takes its negation, so the
    // thermal sample's bulk momentum is exactly zero rather than zero to within
    // sampling error. An odd population leaves its last particle at zero.
    const std::uint64_t pair = p / 2;
    const bool negate = (p % 2) != 0;
    normals3(spec.seed, spec.species_key, pair, thermal);
    const double sign = negate ? -1.0 : 1.0;
    thermal[0] *= sign * spec.thermal_speed;
    thermal[1] *= sign * spec.thermal_speed;
    thermal[2] *= sign * spec.thermal_speed;
  }

  const double ux = thermal[0] + spec.drift_x;
  const double uy = thermal[1] + spec.drift_y;
  const double uz = thermal[2] + spec.drift_z;
  if (!(::isfinite(ux) && ::isfinite(uy) && ::isfinite(uz))) {
    atomicOr(status, kPicSampleNonFiniteVelocity);
  }
  vx[p] = ux;
  vy[p] = uy;
  vz[p] = uz;
}

__global__ void perturbation_kernel(std::uint64_t count,
                                    PicVelocityPerturbationSpec spec,
                                    const double* __restrict__ x,
                                    const double* __restrict__ y,
                                    double* __restrict__ vx,
                                    double* __restrict__ vy,
                                    double* __restrict__ vz,
                                    int* __restrict__ status) {
  const std::uint64_t p =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= count) return;
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  const double phase =
      kTwoPi *
          (spec.mode_x * (x[p] - spec.origin_x) / spec.lx +
           spec.mode_y * (y[p] - spec.origin_y) / spec.ly) +
      spec.phase;
  const double factor = ::sin(phase);
  const double ux = vx[p] + factor * spec.amplitude_x;
  const double uy = vy[p] + factor * spec.amplitude_y;
  const double uz = vz[p] + factor * spec.amplitude_z;
  if (!(::isfinite(ux) && ::isfinite(uy) && ::isfinite(uz))) {
    atomicOr(status, kPicSampleNonFiniteVelocity);
  }
  vx[p] = ux;
  vy[p] = uy;
  vz[p] = uz;
}

__global__ void subluminal_kernel(std::uint64_t count,
                                  const double* __restrict__ vx,
                                  const double* __restrict__ vy,
                                  const double* __restrict__ vz,
                                  int* __restrict__ status) {
  const std::uint64_t p =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= count) return;
  // Nested hypot, matching ParticleSpecies::validate_snapshot exactly: for a
  // speed within an ulp of one the two formulas can disagree, and this check
  // replaces that one rather than approximating it.
  const double speed = ::hypot(::hypot(vx[p], vy[p]), vz[p]);
  if (!::isfinite(speed)) {
    atomicOr(status, kPicSampleNonFiniteVelocity);
  } else if (speed >= 1.0) {
    atomicOr(status, kPicSampleSuperluminal);
  }
}

// The remaining planes a freshly seeded species needs. Previous positions are
// seeded to the initial positions so the first deposit sees zero displacement
// and no startup current; vphi_deposit is seeded to vz for the same reason on
// the cylindrical path. Ids are the identity permutation, which is unique by
// construction, so the host uniqueness check has nothing left to do.
__global__ void finalize_seed_kernel(std::uint64_t count, double weight,
                                     const double* __restrict__ x,
                                     const double* __restrict__ y,
                                     const double* __restrict__ vz,
                                     double* __restrict__ x_prev,
                                     double* __restrict__ y_prev,
                                     double* __restrict__ vphi_deposit,
                                     double* __restrict__ weights,
                                     std::uint8_t* __restrict__ alive,
                                     std::uint64_t* __restrict__ id) {
  const std::uint64_t p =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= count) return;
  x_prev[p] = x[p];
  y_prev[p] = y[p];
  vphi_deposit[p] = vz[p];
  weights[p] = weight;
  alive[p] = 1;
  id[p] = p;
}

dim3 blocks_for(std::uint64_t n) {
  return dim3(static_cast<unsigned int>((n + kBlock - 1) / kBlock));
}

}  // namespace

void launch_pic_quiet_positions(PicQuietStartSpec spec, Real* x, Real* y,
                                int* status, backend::stream_t stream_in) {
  if (spec.count == 0) return;
  const auto stream = static_cast<hipStream_t>(stream_in);
  hipLaunchKernelGGL(quiet_positions_kernel, blocks_for(spec.count),
                     dim3(kBlock), 0, stream, spec, x, y, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_pic_maxwellian_velocities(std::uint64_t count,
                                      PicMaxwellianSpec spec, Real* vx,
                                      Real* vy, Real* vz, int* status,
                                      backend::stream_t stream_in) {
  if (count == 0) return;
  const auto stream = static_cast<hipStream_t>(stream_in);
  hipLaunchKernelGGL(maxwellian_kernel, blocks_for(count), dim3(kBlock), 0,
                     stream, count, spec, vx, vy, vz, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_pic_velocity_perturbation(std::uint64_t count,
                                      PicVelocityPerturbationSpec spec,
                                      const Real* x, const Real* y, Real* vx,
                                      Real* vy, Real* vz, int* status,
                                      backend::stream_t stream_in) {
  if (count == 0 || spec.active == 0) return;
  const auto stream = static_cast<hipStream_t>(stream_in);
  hipLaunchKernelGGL(perturbation_kernel, blocks_for(count), dim3(kBlock), 0,
                     stream, count, spec, x, y, vx, vy, vz, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_pic_check_subluminal(std::uint64_t count, const Real* vx,
                                 const Real* vy, const Real* vz, int* status,
                                 backend::stream_t stream_in) {
  if (count == 0) return;
  const auto stream = static_cast<hipStream_t>(stream_in);
  hipLaunchKernelGGL(subluminal_kernel, blocks_for(count), dim3(kBlock), 0,
                     stream, count, vx, vy, vz, status);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void launch_pic_finalize_seed(std::uint64_t count, Real weight, const Real* x,
                              const Real* y, const Real* vz, Real* x_prev,
                              Real* y_prev, Real* vphi_deposit, Real* weights,
                              std::uint8_t* alive, std::uint64_t* id,
                              backend::stream_t stream_in) {
  if (count == 0) return;
  const auto stream = static_cast<hipStream_t>(stream_in);
  hipLaunchKernelGGL(finalize_seed_kernel, blocks_for(count), dim3(kBlock), 0,
                     stream, count, weight, x, y, vz, x_prev, y_prev,
                     vphi_deposit, weights, alive, id);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace quasar::pic
