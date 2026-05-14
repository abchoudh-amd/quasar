if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_magnetostatics_hip_sources
  biot_savart_hip.hip
)

quasar_add_hip_module(magnetostatics_hip
  SOURCES ${_quasar_magnetostatics_hip_sources})
