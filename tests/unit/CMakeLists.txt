add_subdirectory(core)
add_subdirectory(backend)
add_subdirectory(numerics)
add_subdirectory(boundary)
add_subdirectory(physics)

if(QUASAR_DISTRIBUTED_AVAILABLE)
  add_subdirectory(distributed)
endif()
