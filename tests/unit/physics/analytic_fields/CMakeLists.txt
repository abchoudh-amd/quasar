foreach(_test_name IN ITEMS test_uniform_evaluator test_dipole_evaluator
        test_gradient_evaluator test_file_grid_evaluator
        test_analytic_device_accuracy)
  set(_target "quasar_${_test_name}")
  add_executable(${_target} "${_test_name}.cpp")
  target_link_libraries(${_target}
    PRIVATE
      quasar::core
      quasar_test_support
      GTest::gtest_main
  )
  add_test(NAME "analytic_fields_${_test_name}" COMMAND ${_target})
endforeach()
