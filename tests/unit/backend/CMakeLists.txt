if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

add_executable(quasar_test_backend_device_memory test_device_memory.cpp)

# The test uses HIP runtime API (hipMalloc/hipMemcpy via DeviceBuffer<T>); route
# the .cpp through the HIP compiler so the runtime library links in.
set_source_files_properties(test_device_memory.cpp PROPERTIES LANGUAGE HIP)

target_link_libraries(quasar_test_backend_device_memory
  PRIVATE
    quasar::core
    GTest::gtest_main
)

add_test(
  NAME    backend_test_device_memory
  COMMAND quasar_test_backend_device_memory
)
