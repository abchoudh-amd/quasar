# QuasarHipRuntime.cmake
#
# Configure-time detection of a usable HIP runtime device. Sets the cache
# variable QUASAR_HAS_HIP_RUNTIME to ON when the HIP toolchain can both
# compile a probe program *and* run it successfully reporting at least one
# visible device.  CTest uses this to decide whether HIP-launching unit tests
# should be active or marked DISABLED.

include_guard(GLOBAL)

function(quasar_check_hip_runtime)
  if(DEFINED CACHE{QUASAR_HAS_HIP_RUNTIME})
    return()
  endif()

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
    "${_dir}"
    SOURCES "${_probe}"
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
