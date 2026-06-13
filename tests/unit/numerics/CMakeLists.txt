foreach(_test_name IN ITEMS test_cic_shape test_tsc_shape test_stencil_order test_onesided_stencil test_filter_impulse_response test_filter_registry_linkage test_mhd_state test_mhd_eigensystem test_hlld_riemann test_flux_reconstruction test_mp_limiter test_ct_divergence test_ssprk3_integrator test_positivity_limiter test_mhd_background_eos test_mhd_background_profile test_characteristic_projection)
  set(_target "quasar_${_test_name}")
  add_executable(${_target} "${_test_name}.cpp")
  if(_test_name STREQUAL "test_filter_impulse_response")
    set_source_files_properties("${_test_name}.cpp" PROPERTIES LANGUAGE HIP)
  endif()
  target_link_libraries(${_target}
    PRIVATE
      quasar::core
      GTest::gtest_main
  )
  add_test(NAME "numerics_${_test_name}" COMMAND ${_target})
endforeach()
