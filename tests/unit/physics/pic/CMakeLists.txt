set(_quasar_pic_tests
  test_em_wave_dispersion
  test_fdtd_order_convergence
  test_single_particle_uniform_B
  test_single_particle_ExB_drift
  test_charge_conservation
  test_specular_charge_conservation
  test_tsc_noise_reduction
  test_pec_plane_wave_reflection
  test_outflow_field_absorption
  test_field_ghost_periodic_equivalence
  test_specular_particle
  test_absorbing_particle_decay
  test_mixed_bc
  test_external_uniform_b
  test_external_biot_savart
  test_external_plane_mapping
  test_multi_species_external
  test_filter_charge_conservation
  test_filter_through_solver
  test_filter_two_stream
  test_normalization_invariance
  test_energy_conservation
  test_two_stream_growth_rate
  test_weibel_growth_rate
  test_fp32_vs_fp64
  test_scheme_registry_linkage
  test_cyl_schemes_registry
  test_cyl_diagnostics
  test_cyl_charge_conservation
)

foreach(_test_name IN LISTS _quasar_pic_tests)
  set(_target "quasar_${_test_name}")
  add_executable(${_target} "${_test_name}.cpp")
  set_source_files_properties("${_test_name}.cpp" PROPERTIES LANGUAGE HIP)
  target_link_libraries(${_target}
    PRIVATE
      quasar::core
      GTest::gtest_main
  )
  add_test(NAME "pic_${_test_name}" COMMAND ${_target})
endforeach()
