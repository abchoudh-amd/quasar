if(NOT QUASAR_BUILD_PYTHON)
  return()
endif()

# Smoke tests driven through unittest (no pytest dependency).
set(_quasar_python_tests
  test_magnetostatics_bindings
  test_coil_cli
  test_postprocess
  test_examples
  test_pic_io
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
    APPEND PROPERTY DEPENDS quasar_test_core_registry
  )
endforeach()
