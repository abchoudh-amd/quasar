if(NOT QUASAR_BUILD_BENCHMARKS OR NOT QUASAR_ENABLE_HIP)
  return()
endif()

add_subdirectory(micro)
