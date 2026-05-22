if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_pic_sources
  species.cpp
  pic_solver.cpp
  external_field_sampler.cpp
  diagnostics.cpp
)

set_source_files_properties(${_quasar_pic_sources}
  PROPERTIES LANGUAGE HIP)

quasar_add_module(pic SOURCES ${_quasar_pic_sources})

if(TARGET quasar_pic_hip)
  target_link_libraries(quasar_pic PUBLIC quasar_pic_hip)
endif()
