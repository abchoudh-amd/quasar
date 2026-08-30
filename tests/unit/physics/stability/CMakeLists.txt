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
  test_shift_invert_stability
)

# These binaries touch the device before their first per-test skip opportunity.
# Probe at CTest execution time rather than freezing configure-time visibility.
#
# That is every stability test except the pure-host DOF-layout one: the energy,
# equilibrium-field, and solver tests allocate device buffers or launch HIP
# kernels unconditionally too, so they are gated on the same terms as the
# coordinate tests and as the equilibrium suite.
set(_quasar_stability_hip_runtime_tests
  test_flux_coordinates
  test_spectral_flux_coordinates
  test_radial_domains
  test_chebyshev
  test_newcomb_energy
  test_toroidal_equilibrium
  test_toroidal_energy
  test_stability_solver
  test_shift_invert_stability
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

# Each of these builds a converged free-boundary equilibrium, or resolves the
# analytic geometry gate at high contour/Fourier resolution, before it can
# construct any coordinates; they inherit the solver's runtime and receive a
# generous timeout.
#
# They deliberately carry no `slow` label, for the same reason the equilibrium
# suite does not: the default test presets exclude that label, and the stability
# slice is the headline feature these tests exist to cover. The `slow` label is
# reserved for tests that dominate a full run outright, currently only
# `python_test_examples` at over an hour.
set_tests_properties(
  stability_test_flux_coordinates
  stability_test_spectral_flux_coordinates
  stability_test_radial_domains
  stability_test_toroidal_equilibrium
  stability_test_toroidal_energy
  stability_test_stability_solver
  stability_test_shift_invert_stability
  PROPERTIES TIMEOUT 600)
