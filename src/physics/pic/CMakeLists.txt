if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_pic_sources
  species.cpp
  pic_solver.cpp
  external_field_sampler.cpp
  diagnostics.cpp
  particle_sampling.cpp
)

set_source_files_properties(${_quasar_pic_sources}
  PROPERTIES LANGUAGE HIP)

# REGISTERS: pic_solver.cpp owns the field-solver, pusher, and deposit factories.
# A consumer may construct those schemes solely through Registry<...> without
# referencing an EmPic2D3V symbol, so keep the registration TU in every link.
quasar_add_module(pic REGISTERS SOURCES ${_quasar_pic_sources})

if(TARGET quasar_pic_hip)
  target_link_libraries(quasar_pic PUBLIC quasar_pic_hip)
endif()
