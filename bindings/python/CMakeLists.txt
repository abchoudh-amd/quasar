if(NOT QUASAR_BUILD_PYTHON OR NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_pybind_sources
  bind_magnetostatics.cpp
  bind_pic.cpp
  bind_mhd.cpp
)

# The bindings TU includes quasar/backend/* headers, which pull in
# <hip/hip_runtime.h>; route it through the HIP compiler so the HIP runtime
# library links into the shared object.
set_source_files_properties(${_quasar_pybind_sources}
  PROPERTIES LANGUAGE HIP)

# CMake's Python3_add_library owns the ABI suffix.  WITH_SOABI deliberately
# bypasses pybind11 2.x's separately cached PYTHON_MODULE_EXTENSION, which can
# otherwise retain the suffix of an interpreter used by an older configure.
pybind11_add_module(quasar_core_py MODULE WITH_SOABI ${_quasar_pybind_sources})

set_target_properties(quasar_core_py PROPERTIES
  OUTPUT_NAME            "_core"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/quasar"
)

target_link_libraries(quasar_core_py
  PRIVATE
    quasar::core
)

if(QUASAR_DISTRIBUTED_AVAILABLE)
  pybind11_add_module(quasar_distributed_py MODULE WITH_SOABI
    bind_distributed.cpp
  )
  set_target_properties(quasar_distributed_py PROPERTIES
    OUTPUT_NAME              "_distributed"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python/quasar"
    # Keep this host-only binding TU out of cross-toolchain slim LTO. Mixing a
    # C++ LTO object with the HIP-linked runtime can otherwise omit the module
    # initializer without producing a link error.
    INTERPROCEDURAL_OPTIMIZATION OFF
  )
  target_link_libraries(quasar_distributed_py
    PRIVATE
      quasar::distributed
  )
  target_include_directories(quasar_distributed_py
    PRIVATE
      "${PROJECT_SOURCE_DIR}/src/distributed"
  )
  # Availability means both complete physics runners exist, not merely that the
  # MPI/HDF5 foundation compiled. Flip each definition only with its real binding.
  target_compile_definitions(quasar_distributed_py
    PRIVATE
      QUASAR_DISTRIBUTED_PIC_RUNNER_AVAILABLE=1
      QUASAR_DISTRIBUTED_MHD_RUNNER_AVAILABLE=1
  )
endif()

# Stage the pure-Python package files alongside the built extension so that
#   PYTHONPATH=${CMAKE_BINARY_DIR}/python python -c "import quasar"
# works from the build tree.
set(_quasar_py_sources
  __init__.py
  distributed.py
  _checkpoint_diagnostics.py
  _distributed_helpers.py
  _paths.py
  _deck.py
  _field_grid.py
  _plotting.py
  coil/__init__.py
  coil/io.py
  coil/cli.py
  coil/postprocess.py
  pic/__init__.py
  pic/io.py
  pic/initial_conditions.py
  pic/numerics.py
  pic/postprocess.py
  pic/cli.py
  pic/_units.py
  pic/_distributed_runner.py
  mhd/__init__.py
  mhd/io.py
  mhd/numerics.py
  mhd/_units.py
  mhd/_distributed_runner.py
  mhd/cli.py
)

set(_quasar_staged_py_sources)
foreach(_py IN LISTS _quasar_py_sources)
  set(_src "${PROJECT_SOURCE_DIR}/python/quasar/${_py}")
  set(_dst "${CMAKE_BINARY_DIR}/python/quasar/${_py}")
  get_filename_component(_dst_dir "${_dst}" DIRECTORY)

  # Keep the build-tree package synchronized after the initial configure. A
  # configure_file(COPYONLY) copy has no build dependency, so editing (for
  # example) pic/io.py and running only `cmake --build` would otherwise execute
  # the stale copy left in build/python/quasar.
  add_custom_command(
    OUTPUT "${_dst}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${_dst_dir}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_src}" "${_dst}"
    # copy_if_different intentionally preserves the destination timestamp when
    # only the source timestamp changed. Mark the declared output fresh as well,
    # otherwise Make will rerun this rule on every subsequent build.
    COMMAND "${CMAKE_COMMAND}" -E touch "${_dst}"
    DEPENDS "${_src}"
    COMMENT "Staging Python source quasar/${_py}"
    VERBATIM
  )
  list(APPEND _quasar_staged_py_sources "${_dst}")
endforeach()

add_custom_target(quasar_python_sources ALL
  DEPENDS ${_quasar_staged_py_sources})
add_dependencies(quasar_core_py quasar_python_sources)
if(TARGET quasar_distributed_py)
  add_dependencies(
    quasar_distributed_py quasar_core_py quasar_python_sources)
  # A link can succeed even if the extension lost PyInit__distributed. Import
  # the just-built module so that failure is reported by the build itself.
  add_custom_command(TARGET quasar_distributed_py POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E env
            "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
            "${Python3_EXECUTABLE}" -c "import quasar._distributed"
    COMMENT "Checking that quasar._distributed imports"
    VERBATIM
  )
endif()
