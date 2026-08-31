set(_quasar_mhd_tests
  test_mhd_solver_conservation
  test_mhd_divergence_free
  test_mhd_cylindrical_source
  test_mhd_scheme_registry_linkage
  test_mhd_background_field
  test_mhd_background_builder
  test_mhd_initial_conditions
  test_mhd_onesided_boundary
  test_mhd_boundary
  test_mhd_device_compute
  test_mhd_positivity_preservation
  test_mhd_cylindrical_high_order
)

foreach(_test_name IN LISTS _quasar_mhd_tests)
  set(_target "quasar_${_test_name}")
  add_executable(${_target} "${_test_name}.cpp")
  set_source_files_properties("${_test_name}.cpp" PROPERTIES LANGUAGE HIP)
  target_link_libraries(${_target}
    PRIVATE
      quasar::core
      GTest::gtest_main
  )
  add_test(NAME "mhd_${_test_name}" COMMAND ${_target})
endforeach()
