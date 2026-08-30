set(_quasar_stability_tests
  test_flux_coordinates
  test_spectral_flux_coordinates
  test_radial_domains
  test_chebyshev
  test_spectral_layout
  test_newcomb_energy
  test_toroidal_equilibrium
  test_toroidal_energy
  test_stability_solver
)

# These binaries touch the device before their first per-test skip opportunity.
# Probe at CTest execution time rather than freezing configure-time visibility.
set(_quasar_stability_hip_runtime_tests
  test_flux_coordinates
  test_spectral_flux_coordinates
  test_radial_domains
  test_chebyshev
)

foreach(_test_name IN LISTS _quasar_stability_tests)
  set(_target "quasar_${_test_name}")
  add_executable(${_target} "${_test_name}.cpp")
  target_link_libraries(${_target}
    PRIVATE
      quasar::core
      GTest::gtest_main
  )
  if(_test_name IN_LIST _quasar_stability_hip_runtime_tests)
    quasar_add_hip_runtime_test(
      NAME "stability_${_test_name}"
      TARGET "${_target}")
  else()
    add_test(NAME "stability_${_test_name}" COMMAND ${_target})
  endif()
endforeach()

# Each test builds a converged free-boundary equilibrium before it can construct
# any coordinates, so they inherit the solver's runtime.
set_tests_properties(
  stability_test_flux_coordinates
  stability_test_radial_domains
  stability_test_stability_solver
  PROPERTIES TIMEOUT 600)
