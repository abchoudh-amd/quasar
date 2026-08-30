# QuasarHipRuntime.cmake
#
# Configure-time detection of a usable HIP runtime device. Sets the cache
# variable QUASAR_HAS_HIP_RUNTIME to ON when the HIP toolchain can both
# compile a probe program *and* run it successfully reporting at least one
# visible device. CTest exports the result to the Python tests' runtime skip
# guards; native device tests also query the runtime and skip when unavailable.

include_guard(GLOBAL)

# Register a native test behind an execution-time HIP runtime probe.  This is
# deliberately not keyed from QUASAR_HAS_HIP_RUNTIME: a build configured on a
# login node must still run the test later on a GPU node.  CTest interprets the
# launcher's exit code 77 as a skip only when the runtime is unavailable.
function(quasar_add_hip_runtime_test)
  cmake_parse_arguments(QAHRT "" "NAME;TARGET" "ARGUMENTS" ${ARGN})
  if(NOT QAHRT_NAME OR NOT QAHRT_TARGET)
    message(FATAL_ERROR
      "quasar_add_hip_runtime_test requires NAME and TARGET.")
  endif()
  if(NOT TARGET "${QAHRT_TARGET}")
    message(FATAL_ERROR
      "quasar_add_hip_runtime_test: unknown target ${QAHRT_TARGET}.")
  endif()

  if(NOT TARGET quasar_hip_test_launcher)
    set(_quasar_hip_test_launcher_source
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/QuasarHipTestLauncher.cpp")
    add_executable(quasar_hip_test_launcher
      "${_quasar_hip_test_launcher_source}")
    set_source_files_properties("${_quasar_hip_test_launcher_source}"
      PROPERTIES LANGUAGE HIP)
    # Make the small launcher a representative final consumer of the pinned
    # linear-algebra runtime so CTest can verify the resulting ELF dynamic tag.
    target_link_libraries(quasar_hip_test_launcher
      PRIVATE quasar::linear_algebra)
  endif()

  # A targeted build of the real test must also produce the launcher named in
  # its CTest command; add_test generator expressions do not create this build
  # dependency by themselves.
  add_dependencies("${QAHRT_TARGET}" quasar_hip_test_launcher)

  add_test(
    NAME "${QAHRT_NAME}"
    COMMAND "$<TARGET_FILE:quasar_hip_test_launcher>"
            "$<TARGET_FILE:${QAHRT_TARGET}>"
            ${QAHRT_ARGUMENTS}
  )
  set_tests_properties("${QAHRT_NAME}" PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

function(_quasar_add_ctest_include_to_directory_tree directory include_file)
  set_property(DIRECTORY "${directory}" APPEND PROPERTY
    TEST_INCLUDE_FILES "${include_file}")

  get_property(_subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
  foreach(_subdirectory IN LISTS _subdirectories)
    _quasar_add_ctest_include_to_directory_tree(
      "${_subdirectory}" "${include_file}")
  endforeach()
endfunction()

function(quasar_configure_test_rocm_runtime)
  if(NOT QUASAR_ENABLE_HIP OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    return()
  endif()
  if(NOT DEFINED QUASAR_ROCM_ROOT OR QUASAR_ROCM_ROOT STREQUAL "")
    message(FATAL_ERROR
      "QuasarHipRuntime: the pinned ROCm root is unavailable; "
      "cannot configure a consistent CTest runtime environment.")
  endif()

  cmake_path(ABSOLUTE_PATH QUASAR_ROCM_ROOT
             NORMALIZE OUTPUT_VARIABLE _quasar_rocm_root)
  set(_quasar_rocm_library_dirs)
  foreach(_candidate IN ITEMS lib lib64)
    if(IS_DIRECTORY "${_quasar_rocm_root}/${_candidate}")
      file(REAL_PATH "${_quasar_rocm_root}/${_candidate}"
           _quasar_rocm_library_dir)
      cmake_path(IS_PREFIX _quasar_rocm_root
                 "${_quasar_rocm_library_dir}" NORMALIZE
                 _quasar_library_dir_in_root)
      if(NOT _quasar_library_dir_in_root)
        message(FATAL_ERROR
          "QuasarHipRuntime: ${_quasar_rocm_root}/${_candidate} resolves "
          "outside the pinned ROCm root.")
      endif()
      list(APPEND _quasar_rocm_library_dirs
           "${_quasar_rocm_library_dir}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _quasar_rocm_library_dirs)
  if(NOT _quasar_rocm_library_dirs)
    message(FATAL_ERROR
      "QuasarHipRuntime: no lib or lib64 directory exists below "
      "${_quasar_rocm_root}.")
  endif()
  list(JOIN _quasar_rocm_library_dirs ":"
       QUASAR_CTEST_ROCM_LIBRARY_PATH)
  set(_quasar_ctest_runtime_directory "${PROJECT_BINARY_DIR}/cmake")
  set(_quasar_ctest_runtime_include
      "${_quasar_ctest_runtime_directory}/QuasarTestRocmRuntime.cmake")
  file(MAKE_DIRECTORY "${_quasar_ctest_runtime_directory}")
  configure_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/QuasarTestRocmRuntime.cmake.in"
    "${_quasar_ctest_runtime_include}"
    @ONLY)

  if(TARGET quasar_hip_test_launcher)
    find_program(_quasar_readelf NAMES readelf llvm-readelf)
    if(_quasar_readelf)
      add_test(
        NAME configuration_rocm_rpath
        COMMAND "${CMAKE_COMMAND}"
          "-DQUASAR_READELF=${_quasar_readelf}"
          "-DQUASAR_TEST_BINARY=$<TARGET_FILE:quasar_hip_test_launcher>"
          "-DQUASAR_EXPECTED_RPATH=${QUASAR_CTEST_ROCM_LIBRARY_PATH}"
          -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/QuasarVerifyRpath.cmake")
      set_tests_properties(configuration_rocm_rpath PROPERTIES
        LABELS "configuration;rocm")
    endif()
  endif()

  # Install the hook in every configured directory so the policy also applies
  # when CTest is launched from a subdirectory of the build tree.  The runtime
  # script is idempotent because a normal top-level CTest traversal includes it
  # once per directory.
  _quasar_add_ctest_include_to_directory_tree(
    "${PROJECT_SOURCE_DIR}" "${_quasar_ctest_runtime_include}")
endfunction()

function(quasar_check_hip_runtime)
  set(_dir "${CMAKE_BINARY_DIR}/cmake-probes/hip_runtime")
  set(_probe "${_dir}/probe.hip")
  file(MAKE_DIRECTORY "${_dir}")
  file(WRITE "${_probe}"
"#include <hip/hip_runtime.h>\n"
"int main() {\n"
"  int n = 0;\n"
"  hipError_t e = hipGetDeviceCount(&n);\n"
"  if (e != hipSuccess) return 2;\n"
"  return (n > 0) ? 0 : 1;\n"
"}\n"
  )

  try_run(_run_result _compile_result
    SOURCES "${_probe}"
    NO_CACHE
    RUN_OUTPUT_VARIABLE _hip_out
    COMPILE_OUTPUT_VARIABLE _hip_log
  )

  if(NOT _compile_result)
    message(STATUS
      "QuasarHipRuntime: HIP probe failed to compile; QUASAR_HAS_HIP_RUNTIME=OFF")
    set(QUASAR_HAS_HIP_RUNTIME OFF CACHE BOOL
      "HIP runtime device visible at configure time" FORCE)
    return()
  endif()

  if(_run_result EQUAL 0)
    message(STATUS
      "QuasarHipRuntime: HIP runtime visible; QUASAR_HAS_HIP_RUNTIME=ON")
    set(QUASAR_HAS_HIP_RUNTIME ON CACHE BOOL
      "HIP runtime device visible at configure time" FORCE)
  else()
    message(STATUS
      "QuasarHipRuntime: HIP runtime not visible at configure time "
      "(run exit=${_run_result}); HIP-launching tests will be skipped. "
      "QUASAR_HAS_HIP_RUNTIME=OFF")
    set(QUASAR_HAS_HIP_RUNTIME OFF CACHE BOOL
      "HIP runtime device visible at configure time" FORCE)
  endif()
endfunction()
