if(NOT QUASAR_BUILD_PYTHON)
  return()
endif()

set(_quasar_ctest_gpu_launcher
    "${PROJECT_SOURCE_DIR}/tests/run_with_ctest_gpus.sh")

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
  test_distributed
  test_mhd_distributed_runner
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
  test_pic_distributed_runner
  test_pic_numerics
  test_pic_numerics_cylindrical
  test_pic_periodic_save
  test_pic_periodic_save_integration
  test_pic_postprocess
  test_pic_units
  test_pic_smoke
)

if(QUASAR_DISTRIBUTED_AVAILABLE)
  list(APPEND _quasar_python_tests test_distributed_native)
endif()

# Tests labelled "slow" are excluded by the default ctest presets because they
# dominate a full run (test_examples drives every example CLI end-to-end and
# takes over an hour). Run them with the "-all" test presets, or directly with
# `ctest -L slow`.
set(_quasar_python_slow_tests
  test_examples
)

foreach(_test IN LISTS _quasar_python_tests)
  if(_test STREQUAL "test_distributed_native")
    # This module's direct entry point maps unittest resource skips to code 77.
    add_test(
      NAME python_${_test}
      COMMAND "${_quasar_ctest_gpu_launcher}" --
              ${Python3_EXECUTABLE}
              "${CMAKE_CURRENT_SOURCE_DIR}/test_distributed_native.py"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
  else()
    add_test(
      NAME    python_${_test}
      COMMAND ${Python3_EXECUTABLE} -m unittest -v ${_test}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
  endif()
  set_property(TEST python_${_test}
    PROPERTY ENVIRONMENT
      "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
      "QUASAR_HAS_HIP_RUNTIME=$<IF:$<BOOL:${QUASAR_HAS_HIP_RUNTIME}>,1,0>"
  )
  set_property(TEST python_${_test}
    APPEND PROPERTY DEPENDS core_test_registry
  )
  if(_test IN_LIST _quasar_python_slow_tests)
    set_property(TEST python_${_test} APPEND PROPERTY LABELS slow)
  endif()
endforeach()

if(QUASAR_DISTRIBUTED_AVAILABLE)
  set_tests_properties(python_test_distributed_native PROPERTIES
    LABELS "distributed;multigpu"
    RESOURCE_GROUPS "gpus:1"
    SKIP_RETURN_CODE 77
    TIMEOUT 30
  )

  add_test(
    NAME python_test_distributed_native_mpi
    COMMAND
      "${_quasar_ctest_gpu_launcher}" --
      ${MPIEXEC_EXECUTABLE}
      ${MPIEXEC_NUMPROC_FLAG} 2
      ${MPIEXEC_PREFLAGS}
      ${Python3_EXECUTABLE}
      "${CMAKE_CURRENT_SOURCE_DIR}/test_distributed_native.py"
      ${MPIEXEC_POSTFLAGS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_tests_properties(python_test_distributed_native_mpi PROPERTIES
    DEPENDS core_test_registry
    ENVIRONMENT
      "PYTHONPATH=${CMAKE_BINARY_DIR}/python;QUASAR_HAS_HIP_RUNTIME=$<IF:$<BOOL:${QUASAR_HAS_HIP_RUNTIME}>,1,0>"
    LABELS "distributed;multirank;multigpu"
    RESOURCE_GROUPS "2,gpus:1"
    SKIP_RETURN_CODE 77
    TIMEOUT 30
  )
endif()
