if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

# Host-only tests (plain C++, no HIP runtime needed at test time).
set(_quasar_magnetostatics_host_tests
  test_soa_conversion
  test_geometry_generators
  test_observation_sets
)

foreach(_test_name IN LISTS _quasar_magnetostatics_host_tests)
  set(_target "quasar_test_magnetostatics_${_test_name}")
  add_executable(${_target} "${_test_name}.cpp")
  target_link_libraries(${_target}
    PRIVATE
      quasar::core
      quasar_test_support
      GTest::gtest_main
  )
  add_test(
    NAME    "magnetostatics_${_test_name}"
    COMMAND ${_target}
  )
endforeach()

# The HIP-launching analytical tests pull <hip/hip_runtime.h> transitively via
# quasar/physics/magnetostatics/biot_savart.hpp -> quasar/backend/device.hpp,
# so compile them with the HIP toolchain.
set(_quasar_magnetostatics_hip_tests
  test_finite_segment
  test_circular_loop_on_axis
  test_segment_gradient
  test_vector_potential
  test_infinite_wire_limit
  test_helmholtz_pair
  test_solenoid_inside
  test_polyline_convergence
  test_fp32_vs_fp64
  test_singular_and_empty
)

foreach(_test_name IN LISTS _quasar_magnetostatics_hip_tests)
  set(_src "${_test_name}.cpp")
  set(_target "quasar_test_magnetostatics_${_test_name}")

  add_executable(${_target} ${_src})
  set_source_files_properties(${_src} PROPERTIES LANGUAGE HIP)
  target_link_libraries(${_target}
    PRIVATE
      quasar::core
      quasar_test_support
      GTest::gtest_main
  )

  add_test(
    NAME    magnetostatics_${_test_name}
    COMMAND ${_target}
  )
endforeach()
