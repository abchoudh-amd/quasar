if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_mhd_hip_sources
  mhd_reconstruct.hip
  mhd_riemann.hip
  mhd_ct.hip
  mhd_halo.hip
  mhd_background_validate.hip
  mhd_background_momentum.hip
  mhd_cylindrical_momentum.hip
  mhd_split_energy.hip
  mhd_update.hip
  mhd_reduce.hip
  mhd_boundary_hip.hip
)

quasar_add_hip_module(mhd_hip SOURCES ${_quasar_mhd_hip_sources})
quasar_add_backend_src_include(mhd_hip)

# -ffp-contract=off for the background validation sweeps only.
#
# These decide whether a supplied B0 is discretely divergence-free by comparing
# a residual against the representation roundoff of its own operands. Both sides
# of that comparison are computed from the same scaled-arithmetic helpers, and
# the whole judgement is meaningful only if the arithmetic rounds where the
# metric says it rounds. Contracting a multiply-add there changes the residual
# and the roundoff bound by different amounts, which can turn a field that is
# solenoidal to the last bit into a rejected one, or the reverse.
#
# Scoped per-source: the MHD hot-path kernels in this module want contraction.
set_source_files_properties(mhd_background_validate.hip
  TARGET_DIRECTORY quasar_mhd_hip
  PROPERTIES COMPILE_OPTIONS
    "$<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>")
