foreach(_test_name IN ITEMS test_uniform_evaluator test_dipole_evaluator)
  set(_target "quasar_${_test_name}")
  add_executable(${_target} "${_test_name}.cpp")
  target_link_libraries(${_target}
    PRIVATE
      quasar::core
      GTest::gtest_main
  )
  add_test(NAME "analytic_fields_${_test_name}" COMMAND ${_target})
endforeach()
