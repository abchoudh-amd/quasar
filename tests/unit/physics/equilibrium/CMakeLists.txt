set(_quasar_equilibrium_tests
  test_gs_fixed_boundary
  test_free_boundary_gs
  test_flux_surfaces
  test_mhd_seeding
  test_gs_operator_device
  test_gs_reduce_device
  test_gs_greens_device
  test_gs_source_device
  test_gs_critical_device
  test_gs_multigrid_device
  test_gs_solver_invariants
  test_gs_contour_device
  test_gs_port_gate
)

foreach(_test_name IN LISTS _quasar_equilibrium_tests)
  set(_target "quasar_${_test_name}")
  add_executable(${_target} "${_test_name}.cpp")
  target_link_libraries(${_target}
    PRIVATE
      quasar::core
      GTest::gtest_main
  )
  add_test(NAME "equilibrium_${_test_name}" COMMAND ${_target})
endforeach()

# The free-boundary solve is a few-hundred-iteration nonlinear loop with an
# O(N_boundary * N_interior) Green's-function integral per step, so these tests
# receive a generous timeout. They deliberately carry no `slow` label: default
# presets exclude that label, and the headline feature must remain covered.
set_tests_properties(
  equilibrium_test_free_boundary_gs
  equilibrium_test_flux_surfaces
  equilibrium_test_mhd_seeding
  equilibrium_test_gs_solver_invariants
  equilibrium_test_gs_contour_device
  equilibrium_test_gs_port_gate
  PROPERTIES TIMEOUT 600)
