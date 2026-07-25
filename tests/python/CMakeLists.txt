if(NOT QUASAR_BUILD_PYTHON)
  return()
endif()

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import numpy; import yaml"
  RESULT_VARIABLE _quasar_python_dependencies_result
  ERROR_VARIABLE  _quasar_python_dependencies_error
)
if(NOT _quasar_python_dependencies_result EQUAL 0)
  string(STRIP "${_quasar_python_dependencies_error}"
         _quasar_python_dependencies_error)
  message(FATAL_ERROR
    "Quasar's Python tests require NumPy and PyYAML for the selected "
    "Python interpreter (${Python3_EXECUTABLE}). Install the complete test "
    "environment with:\n"
    "  ${Python3_EXECUTABLE} -m pip install -r "
    "${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt\n"
    "Import error: ${_quasar_python_dependencies_error}"
  )
endif()

# Smoke tests driven through unittest (no pytest dependency).
set(_quasar_python_tests
  test_magnetostatics_bindings
  test_binding_real_arrays
  test_field_grid
  test_coil_cli
  test_postprocess
  test_examples
  test_mhd_background_bindings
  test_mhd_background_io
  test_mhd_bindings
  test_mhd_cli
  test_mhd_io
  test_pic_io
  test_pic_io_cylindrical
  test_pic_initial_conditions
  test_pic_cli
  test_pic_numerics
  test_pic_numerics_cylindrical
  test_pic_periodic_save
  test_pic_periodic_save_integration
  test_pic_postprocess
  test_pic_units
  test_pic_smoke
)

foreach(_test IN LISTS _quasar_python_tests)
  add_test(
    NAME    python_${_test}
    COMMAND ${Python3_EXECUTABLE} -m unittest -v ${_test}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_property(TEST python_${_test}
    PROPERTY ENVIRONMENT
      "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
      "QUASAR_HAS_HIP_RUNTIME=$<IF:$<BOOL:${QUASAR_HAS_HIP_RUNTIME}>,1,0>"
  )
  set_property(TEST python_${_test}
    APPEND PROPERTY DEPENDS core_test_registry
  )
endforeach()
