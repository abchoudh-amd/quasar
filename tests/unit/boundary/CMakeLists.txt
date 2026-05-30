foreach(_test_name IN ITEMS test_pec_ghosts test_registry_linkage)
  set(_target "quasar_${_test_name}")
  add_executable(${_target} "${_test_name}.cpp")
  set_source_files_properties("${_test_name}.cpp" PROPERTIES LANGUAGE HIP)
  target_link_libraries(${_target}
    PRIVATE
      quasar::core
      GTest::gtest_main
  )
  add_test(NAME "boundary_${_test_name}" COMMAND ${_target})
endforeach()
