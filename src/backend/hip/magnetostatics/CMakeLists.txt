if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

include(QuasarLaunchParams)

set(_quasar_magnetostatics_hip_sources
  biot_savart_hip.hip
  biot_savart_grad_hip.hip
)

quasar_add_hip_module(magnetostatics_hip
  SOURCES ${_quasar_magnetostatics_hip_sources})
quasar_add_backend_src_include(magnetostatics_hip)

# Stamp per-gfx launch parameters as PRIVATE compile definitions so both the
# host launcher (kernel<<<grid,block,...>>>) and the device kernel
# (__shared__ Real s_ax[kTileSegments]) see the same constants.
quasar_set_launch_params(quasar_magnetostatics_hip)
