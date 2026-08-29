# QuasarHipRuntime.cmake
#
# Configure-time detection of a usable HIP runtime device. Sets the cache
# variable QUASAR_HAS_HIP_RUNTIME to ON when the HIP toolchain can both
# compile a probe program *and* run it successfully reporting at least one
# visible device. CTest exports the result to the Python tests' runtime skip
# guards; native device tests also query the runtime and skip when unavailable.

include_guard(GLOBAL)

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
  if(NOT DEFINED CMAKE_HIP_COMPILER_ROCM_ROOT OR
     CMAKE_HIP_COMPILER_ROCM_ROOT STREQUAL "")
    message(FATAL_ERROR
      "QuasarHipRuntime: CMAKE_HIP_COMPILER_ROCM_ROOT is unavailable; "
      "cannot configure a consistent CTest runtime environment.")
  endif()

  cmake_path(ABSOLUTE_PATH CMAKE_HIP_COMPILER_ROCM_ROOT
             NORMALIZE OUTPUT_VARIABLE _quasar_rocm_root)
  set(_quasar_rocm_library_dirs)
  foreach(_candidate IN ITEMS lib lib64)
    if(IS_DIRECTORY "${_quasar_rocm_root}/${_candidate}")
      list(APPEND _quasar_rocm_library_dirs
           "${_quasar_rocm_root}/${_candidate}")
    endif()
  endforeach()
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
