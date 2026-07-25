add_executable(quasar_test_core_registry test_registry.cpp)

target_link_libraries(quasar_test_core_registry
  PRIVATE
    quasar::core
    GTest::gtest_main
)

add_test(
  NAME    core_test_registry
  COMMAND quasar_test_core_registry
)

foreach(_test_name IN ITEMS test_field test_grid_2d test_grid_cylindrical test_normalization test_types test_yee_field)
  set(_target "quasar_${_test_name}")
  add_executable(${_target} "${_test_name}.cpp")
  if(_test_name STREQUAL "test_yee_field")
    set_source_files_properties("${_test_name}.cpp" PROPERTIES LANGUAGE HIP)
  endif()
  target_link_libraries(${_target}
    PRIVATE
      quasar::core
      GTest::gtest_main
  )
  add_test(
    NAME    "core_${_test_name}"
    COMMAND ${_target}
  )
endforeach()
