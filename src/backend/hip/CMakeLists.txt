if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_backend_hip_sources
  device.cpp
  device_memory.cpp
)

# The .cpp file uses the HIP runtime API; route it through the HIP compiler so
# the runtime library is linked in automatically.
set_source_files_properties(${_quasar_backend_hip_sources}
  PROPERTIES LANGUAGE HIP)

quasar_add_module(backend_hip SOURCES ${_quasar_backend_hip_sources})
quasar_add_backend_src_include(backend_hip)

add_subdirectory(magnetostatics)
add_subdirectory(pic)
add_subdirectory(mhd)
add_subdirectory(equilibrium)
# numerics precedes stability: the stability backend links quasar_numerics_hip
# by target name, so that target must already exist.
add_subdirectory(numerics)
add_subdirectory(stability)
