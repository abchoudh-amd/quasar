if(NOT QUASAR_BUILD_PYTHON OR NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_pybind_sources
  bind_magnetostatics.cpp
)

# The bindings TU includes quasar/backend/* headers, which pull in
# <hip/hip_runtime.h>; route it through the HIP compiler so the HIP runtime
# library links into the shared object.
set_source_files_properties(${_quasar_pybind_sources}
  PROPERTIES LANGUAGE HIP)

pybind11_add_module(quasar_core_py MODULE ${_quasar_pybind_sources})

set_target_properties(quasar_core_py PROPERTIES
  OUTPUT_NAME            "_core"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/quasar"
)

target_link_libraries(quasar_core_py
  PRIVATE
    quasar::core
)

# Stage the pure-Python package files alongside the built extension so that
#   PYTHONPATH=${CMAKE_BINARY_DIR}/python python -c "import quasar"
# works from the build tree.
set(_quasar_py_sources
  __init__.py
  coil/__init__.py
  coil/io.py
  coil/cli.py
  coil/postprocess.py
)

foreach(_py IN LISTS _quasar_py_sources)
  configure_file(
    "${PROJECT_SOURCE_DIR}/python/quasar/${_py}"
    "${CMAKE_BINARY_DIR}/python/quasar/${_py}"
    COPYONLY
  )
endforeach()
