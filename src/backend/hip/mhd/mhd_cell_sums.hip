// Integrated conserved totals (mass and energy) over the owned cells.
//
// These are the deck-facing invariants a benchmark run reports at t=0 and
// compares its final state against. They used to be computed twice, in two
// different ways: the serial CLI summed the downloaded interior planes in NumPy
// with no geometric weight, while the distributed runtime downloaded the same
// planes per tile and ran an uncompensated host loop that DID apply the
// axisymmetric cell volume. On a cylindrical deck the two therefore disagreed by
// far more than rounding, and on any deck they disagreed by however much a naive
// traversal-order-dependent fp64 sweep drifts.
//
// One kernel now serves both paths, so there is a single definition of what
// "total mass" means, the volume weight cannot be applied on one path and not
// the other, and the accumulation is compensated in a shared exponent frame so a
// one-tile and an N-tile run agree to a meaningful tolerance.
//
// -ffp-contract=off is required, per scaled_reduction_detail.hpp: the Kahan
// compensation recovers exactly the bits a fused multiply-add would silently
// keep or drop, so contraction makes the correction term meaningless.

#include "quasar/backend/memory.hpp"
#include "quasar/core/grid.hpp"
#include "quasar/physics/mhd/kernels.hpp"
#include "quasar/physics/mhd/mhd_field.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/reduction_detail.hpp"
#include "backend/hip/scaled_reduction_detail.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace {

namespace hd = quasar::backend::hip_detail;

using quasar::Grid2D;
using quasar::Real;
using hd::ScaledAcc;
using hd::ScaledTerm;
using hd::scaled_acc_add;
using hd::scaled_acc_merge;
using hd::scaled_signed_product;

inline constexpr unsigned kThreads = hd::kReduceThreads;

// Mass and energy in one pass so both totals provably see the same state.
//
// scaled_signed_product, not scaled_product: a conserved density is expected to
// be positive, but a run that has gone bad can carry a negative energy, and the
// diagnostic's job is to report that faithfully rather than poison the sum and
// hide it behind a representability error.
__global__ void QUASAR_LAUNCH_BOUNDS_256 conserved_cell_sums_pass1(
    Grid2D g, int cylindrical, const double* __restrict__ rho,
    const double* __restrict__ energy, ScaledAcc* __restrict__ partial_mass,
    ScaledAcc* __restrict__ partial_energy) {
  __shared__ ScaledAcc s_mass[kThreads];
  __shared__ ScaledAcc s_energy[kThreads];
  const unsigned tid = threadIdx.x;

  ScaledAcc mass{};
  ScaledAcc total_energy{};

  const std::size_t cells =
      static_cast<std::size_t>(g.nx) * static_cast<std::size_t>(g.ny);
  for (std::size_t c = blockIdx.x * blockDim.x + tid; c < cells;
       c += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const int i = static_cast<int>(c % static_cast<std::size_t>(g.nx));
    const int j = static_cast<int>(c / static_cast<std::size_t>(g.nx));
    // Column volume, not per-cell: on a Cartesian grid every owned cell carries
    // unit weight, which keeps the totals a plain cell sum exactly as before.
    const Real weight = cylindrical != 0 ? g.cell_volume(i) : Real{1};
    const std::size_t k = g.index(i, j);

    const Real mass_factors[2] = {weight, rho[k]};
    const Real energy_factors[2] = {weight, energy[k]};
    scaled_acc_add(mass, scaled_signed_product(mass_factors, 2));
    scaled_acc_add(total_energy, scaled_signed_product(energy_factors, 2));
  }

  s_mass[tid] = mass;
  s_energy[tid] = total_energy;
  __syncthreads();
  hd::block_scaled_acc_reduce<kThreads>(s_mass, tid);
  hd::block_scaled_acc_reduce<kThreads>(s_energy, tid);
  if (tid == 0) {
    partial_mass[blockIdx.x] = s_mass[0];
    partial_energy[blockIdx.x] = s_energy[0];
  }
}

__global__ void QUASAR_LAUNCH_BOUNDS_256 conserved_cell_sums_pass2(
    const ScaledAcc* __restrict__ partial, ScaledAcc* __restrict__ out,
    unsigned blocks) {
  __shared__ ScaledAcc sdata[kThreads];
  const unsigned tid = threadIdx.x;
  ScaledAcc acc{};
  for (unsigned k = tid; k < blocks; k += kThreads) {
    scaled_acc_merge(acc, partial[k]);
  }
  sdata[tid] = acc;
  __syncthreads();
  hd::block_scaled_acc_reduce<kThreads>(sdata, tid);
  if (tid == 0) out[0] = sdata[0];
}

quasar::mhd::MhdScaledSum to_public(const ScaledAcc& a) {
  quasar::mhd::MhdScaledSum out;
  out.sum = a.sum;
  out.correction = a.correction;
  out.exponent = a.exponent;
  out.initialized = a.initialized;
  out.invalid = a.invalid || a.nonfinite_input;
  return out;
}

}  // namespace

namespace quasar::mhd {

MhdConservedSums launch_mhd_conserved_cell_sums(const MhdField2D<Real>& u,
                                                int cylindrical,
                                                stream_t stream) {
  const Grid2D g = u.grid;
  const std::size_t cells =
      static_cast<std::size_t>(g.nx) * static_cast<std::size_t>(g.ny);
  if (cells == 0) return MhdConservedSums{};

  const auto hip_stream = static_cast<hipStream_t>(stream);
  const unsigned blocks = hd::reduce_block_count(cells);

  backend::DeviceBuffer<ScaledAcc> partial_mass{blocks};
  backend::DeviceBuffer<ScaledAcc> partial_energy{blocks};
  backend::DeviceBuffer<ScaledAcc> result_mass{1};
  backend::DeviceBuffer<ScaledAcc> result_energy{1};

  hipLaunchKernelGGL(conserved_cell_sums_pass1, dim3{blocks}, dim3{kThreads}, 0,
                     hip_stream, g, cylindrical, u.rho.device_ptr(),
                     u.energy.device_ptr(), partial_mass.device_ptr(),
                     partial_energy.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());

  hipLaunchKernelGGL(conserved_cell_sums_pass2, dim3{1}, dim3{kThreads}, 0,
                     hip_stream, partial_mass.device_ptr(),
                     result_mass.device_ptr(), blocks);
  QUASAR_HIP_CHECK(::hipGetLastError());
  hipLaunchKernelGGL(conserved_cell_sums_pass2, dim3{1}, dim3{kThreads}, 0,
                     hip_stream, partial_energy.device_ptr(),
                     result_energy.device_ptr(), blocks);
  QUASAR_HIP_CHECK(::hipGetLastError());
  QUASAR_HIP_CHECK(::hipStreamSynchronize(hip_stream));

  ScaledAcc host_mass{};
  ScaledAcc host_energy{};
  result_mass.copy_to_host(&host_mass, 1);
  result_energy.copy_to_host(&host_energy, 1);

  MhdConservedSums out;
  out.mass = to_public(host_mass);
  out.energy = to_public(host_energy);
  return out;
}

}  // namespace quasar::mhd
