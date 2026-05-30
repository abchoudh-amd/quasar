if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_magnetostatics_sources
  conductor.cpp
  geometry_generators.cpp
  biot_savart_evaluator.cpp
)

# The evaluator orchestrator owns device buffers and links the biot_savart HIP
# kernel module; compile it with the HIP toolchain so the runtime library links
# in transitively. conductor.cpp / geometry_generators.cpp stay plain C++
# (host-only); the observation types now live in core.
set_source_files_properties(biot_savart_evaluator.cpp
  PROPERTIES LANGUAGE HIP)

quasar_add_module(magnetostatics REGISTERS SOURCES ${_quasar_magnetostatics_sources})

# Pull in the HIP kernel translation unit, which defines the extern "C"
# launch_biot_savart_B symbol referenced by biot_savart_evaluator.cpp.
if(TARGET quasar_magnetostatics_hip)
  target_link_libraries(quasar_magnetostatics PUBLIC quasar_magnetostatics_hip)
endif()
