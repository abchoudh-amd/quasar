set(_quasar_boundary_sources
  periodic.cpp
  wall.cpp
)

set_source_files_properties(${_quasar_boundary_sources} PROPERTIES LANGUAGE HIP)

quasar_add_module(boundary SOURCES ${_quasar_boundary_sources})

if(TARGET quasar_pic_hip)
  target_link_libraries(quasar_boundary PUBLIC quasar_pic_hip)
endif()
