foreach(_test_name IN ITEMS test_cic_shape test_tsc_shape test_stencil_order test_filter_impulse_response)
  set(_target "quasar_${_test_name}")
  add_executable(${_target} "${_test_name}.cpp")
  if(_test_name STREQUAL "test_filter_impulse_response")
    set_source_files_properties("${_test_name}.cpp" PROPERTIES LANGUAGE HIP)
  endif()
  target_link_libraries(${_target}
    PRIVATE
      quasar::core
      GTest::gtest_main
  )
  add_test(NAME "numerics_${_test_name}" COMMAND ${_target})
endforeach()
