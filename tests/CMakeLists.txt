if(QUASAR_DISTRIBUTED_AVAILABLE)
  add_test(
    NAME distributed_configuration_enabled
    COMMAND "${CMAKE_COMMAND}" -E echo
            "Distributed tests enabled (QUASAR_ENABLE_DISTRIBUTED=${QUASAR_ENABLE_DISTRIBUTED})"
  )
  set_tests_properties(distributed_configuration_enabled PROPERTIES
    LABELS "configuration;distributed"
  )
else()
  set(_quasar_distributed_test_status
      "Distributed tests unavailable (QUASAR_ENABLE_DISTRIBUTED=${QUASAR_ENABLE_DISTRIBUTED})")
  if(QUASAR_DISTRIBUTED_FAILURE_REASONS)
    string(APPEND _quasar_distributed_test_status
           ": ${QUASAR_DISTRIBUTED_FAILURE_REASONS}")
  endif()
  add_test(
    NAME distributed_configuration_unavailable
    COMMAND "${CMAKE_COMMAND}" -E echo
            "${_quasar_distributed_test_status}"
  )
  set_tests_properties(distributed_configuration_unavailable PROPERTIES
    LABELS "configuration;distributed"
  )
endif()

add_subdirectory(unit)

if(QUASAR_BUILD_PYTHON)
  add_subdirectory(python)
endif()
