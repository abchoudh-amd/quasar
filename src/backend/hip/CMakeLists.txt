if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_backend_hip_sources
  device.cpp
)

# The .cpp file uses the HIP runtime API; route it through the HIP compiler so
# the runtime library is linked in automatically.
set_source_files_properties(${_quasar_backend_hip_sources}
  PROPERTIES LANGUAGE HIP)

quasar_add_module(backend_hip SOURCES ${_quasar_backend_hip_sources})
