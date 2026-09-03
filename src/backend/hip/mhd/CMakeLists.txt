if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_mhd_hip_sources
  mhd_reconstruct.hip
  mhd_riemann.hip
  mhd_ct.hip
  mhd_halo.hip
  mhd_background_validate.hip
  mhd_initial_conditions.hip
  mhd_background_build.hip
  mhd_collocation.hip
  mhd_background_momentum.hip
  mhd_cylindrical_momentum.hip
  mhd_split_energy.hip
  mhd_update.hip
  mhd_reduce.hip
  mhd_cell_sums.hip
  mhd_boundary_hip.hip
)

quasar_add_hip_module(mhd_hip SOURCES ${_quasar_mhd_hip_sources})
quasar_add_backend_src_include(mhd_hip)

# -ffp-contract=off for the background validation sweeps and the seeded initial
# state only.
#
# These decide whether a supplied B0 is discretely divergence-free by comparing
# a residual against the representation roundoff of its own operands. Both sides
# of that comparison are computed from the same scaled-arithmetic helpers, and
# the whole judgement is meaningful only if the arithmetic rounds where the
# metric says it rounds. Contracting a multiply-add there changes the residual
# and the roundoff bound by different amounts, which can turn a field that is
# solenoidal to the last bit into a rejected one, or the reverse.
#
# The seeded initial state shares those scaled-arithmetic helpers through the
# face-to-cell collocation, and its positivity preflight decides admissibility
# by comparing a difference of same-magnitude quantities against zero -- the
# same reason, one step earlier in the run. The background construction is on
# the list for a third variant of it: the vacuum projection's conjugate gradient
# uses compensated inner products, and contraction there degrades the
# compensation exactly as it does in a reduction.
#
# The conserved cell totals are on the list for a fourth variant: their Kahan
# compensation recovers exactly the bits a fused multiply-add would keep or drop
# silently, so contraction does not merely perturb the correction term, it makes
# it meaningless. This is the requirement scaled_reduction_detail.hpp states for
# every consumer of ScaledAcc.
#
# Scoped per-source: the MHD hot-path kernels in this module want contraction.
set_source_files_properties(mhd_background_validate.hip mhd_initial_conditions.hip
  mhd_background_build.hip mhd_collocation.hip mhd_cell_sums.hip
  TARGET_DIRECTORY quasar_mhd_hip
  PROPERTIES COMPILE_OPTIONS
    "$<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>")
