set(_quasar_numerics_sources
  filter.cpp
)

set_source_files_properties(filter.cpp PROPERTIES LANGUAGE HIP)

quasar_add_module(numerics SOURCES ${_quasar_numerics_sources})

if(TARGET quasar_pic_hip)
  target_link_libraries(quasar_numerics PUBLIC quasar_pic_hip)
endif()
