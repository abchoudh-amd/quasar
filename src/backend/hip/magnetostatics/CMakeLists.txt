if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

include(QuasarLaunchParams)

set(_quasar_magnetostatics_hip_sources
  biot_savart_hip.hip
  biot_savart_grad_hip.hip
  geometry_hip.hip
)

quasar_add_hip_module(magnetostatics_hip
  SOURCES ${_quasar_magnetostatics_hip_sources})
quasar_add_backend_src_include(magnetostatics_hip)

# Stamp per-gfx launch parameters as PRIVATE compile definitions so both the
# host launcher (kernel<<<grid,block,...>>>) and the device kernel
# (__shared__ Real s_ax[kTileSegments]) see the same constants.
quasar_set_launch_params(quasar_magnetostatics_hip)

# -ffp-contract=off on the geometry kernels. Their resolvability checks compare
# a recovered displacement against the one that was requested and reject a ratio
# outside [1/2, 3/2]; that is a statement about what rounding did to each
# individual operation, so contracting them changes what the check is testing.
# The generated vertices are also expected to agree with the host accessors that
# echo the same geometry.
set_source_files_properties(geometry_hip.hip
  TARGET_DIRECTORY quasar_magnetostatics_hip
  PROPERTIES COMPILE_OPTIONS
    "$<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>")
