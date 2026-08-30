foreach(_test_name IN ITEMS test_cic_shape test_tsc_shape test_stencil_order test_onesided_stencil test_filter_impulse_response test_filter_registry_linkage test_mhd_state test_mhd_eigensystem test_hlld_riemann test_flux_reconstruction test_flux_reconstruction_cylindrical test_mhd_multidimensional_flux test_mhd_axis_high_order test_mp_limiter test_mp_limiter_cylindrical test_radial_moments test_radial_tables test_mhd_cylindrical_collocation test_ct_divergence test_ssprk3_integrator test_positivity_limiter test_mhd_background_eos test_mhd_background_profile test_characteristic_projection test_pade_derivative test_geometric_multigrid test_defect_correction test_elliptic_tile test_generalized_eigensolver test_block_tridiagonal test_condition_estimator test_shift_invert_lanczos)
  set(_target "quasar_${_test_name}")
  add_executable(${_target} "${_test_name}.cpp")
  if(_test_name STREQUAL "test_filter_impulse_response" OR
     _test_name STREQUAL "test_radial_tables")
    set_source_files_properties("${_test_name}.cpp" PROPERTIES LANGUAGE HIP)
  endif()
  target_link_libraries(${_target}
    PRIVATE
      quasar::core
      GTest::gtest_main
  )
  # The dense linear-algebra tests call hipBLAS and hipSOLVER unconditionally,
  # so they need a real device rather than only a HIP compiler. Gate them the
  # way the equilibrium and stability suites are gated: a configure and build on
  # a login node stays valid, and a run without a GPU reports a clean skip
  # instead of a spurious failure.
  if(_test_name STREQUAL "test_generalized_eigensolver" OR
     _test_name STREQUAL "test_block_tridiagonal" OR
     _test_name STREQUAL "test_condition_estimator" OR
     _test_name STREQUAL "test_shift_invert_lanczos")
    quasar_add_hip_runtime_test(
      NAME "numerics_${_test_name}"
      TARGET "${_target}")
  else()
    add_test(NAME "numerics_${_test_name}" COMMAND ${_target})
  endif()
endforeach()
