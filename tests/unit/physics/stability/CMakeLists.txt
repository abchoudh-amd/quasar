set(_quasar_stability_tests
  test_flux_coordinates
)

foreach(_test_name IN LISTS _quasar_stability_tests)
  set(_target "quasar_${_test_name}")
  add_executable(${_target} "${_test_name}.cpp")
  target_link_libraries(${_target}
    PRIVATE
      quasar::core
      GTest::gtest_main
  )
  add_test(NAME "stability_${_test_name}" COMMAND ${_target})
endforeach()

# Each test builds a converged free-boundary equilibrium before it can construct
# any coordinates, so they inherit the solver's runtime.
set_tests_properties(
  stability_test_flux_coordinates
  PROPERTIES TIMEOUT 600)
