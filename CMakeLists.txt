cmake_minimum_required(VERSION 3.27)

project(quasar
  VERSION 0.1.0
  DESCRIPTION "HIP-accelerated numerical simulation framework"
  LANGUAGES CXX
)

# Resolve the HIP compiler after the project toolchain has been loaded but
# before the HIP language is enabled. Development hosts commonly expose a
# moving ROCm nightly through a `current` symlink; retaining that spelling in
# CMake's compiler metadata lets an already-configured build silently switch
# compiler and runtime when the symlink moves.
set(_quasar_hip_compiler_request "")
set(_quasar_hip_compiler_args "")
if(DEFINED CMAKE_HIP_COMPILER AND NOT CMAKE_HIP_COMPILER STREQUAL "")
  set(_quasar_hip_compiler_request "${CMAKE_HIP_COMPILER}")
elseif(DEFINED ENV{HIPCXX} AND NOT "$ENV{HIPCXX}" STREQUAL "")
  set(_quasar_hip_compiler_request "$ENV{HIPCXX}")
else()
  find_program(_quasar_hipconfig NAMES hipconfig NO_CACHE)
  if(_quasar_hipconfig)
    execute_process(
      COMMAND "${_quasar_hipconfig}" --hipclangpath
      RESULT_VARIABLE _quasar_hipconfig_result
      OUTPUT_VARIABLE _quasar_hipclang_path
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_quasar_hipconfig_result EQUAL 0
       AND EXISTS "${_quasar_hipclang_path}/clang++")
      set(_quasar_hip_compiler_request
          "${_quasar_hipclang_path}/clang++")
    endif()
  endif()
  if(_quasar_hip_compiler_request STREQUAL "")
    find_program(_quasar_hip_compiler_request NAMES clang++ NO_CACHE)
  endif()
endif()

if(NOT _quasar_hip_compiler_request STREQUAL "")
  set(_quasar_existing_hip_compiler_arg1 "")
  if(DEFINED CMAKE_HIP_COMPILER_ARG1 AND
     NOT CMAKE_HIP_COMPILER_ARG1 STREQUAL "")
    set(_quasar_existing_hip_compiler_arg1
        "${CMAKE_HIP_COMPILER_ARG1}")
  endif()
  get_filename_component(_quasar_hip_compiler
    "${_quasar_hip_compiler_request}" PROGRAM
    PROGRAM_ARGS _quasar_hip_compiler_args)
  if(_quasar_hip_compiler_args STREQUAL "")
    set(_quasar_hip_compiler_args
        "${_quasar_existing_hip_compiler_arg1}")
  endif()
  string(STRIP "${_quasar_hip_compiler_args}" _quasar_hip_compiler_args)
  if(NOT _quasar_hip_compiler OR NOT EXISTS "${_quasar_hip_compiler}")
    message(FATAL_ERROR
      "Quasar: HIP compiler does not exist: ${_quasar_hip_compiler_request}")
  endif()
  if(_quasar_hip_compiler MATCHES "(^|[/\\\\])hipcc([.]bat|[.]exe)?$")
    message(FATAL_ERROR
      "Quasar: hipcc is not a supported CMAKE_HIP_COMPILER; use the ROCm "
      "Clang driver directly.")
  endif()
  file(REAL_PATH "${_quasar_hip_compiler}" _quasar_hip_compiler)

  # CMake preserves compiler arguments separately in CMAKE_HIP_COMPILER_ARG1.
  # Canonicalize every ROCm/HIP path option before compiler identification so
  # a moving symlink cannot survive in the generated compile and link rules.
  # Parse only to identify complete option values; replacements happen in the
  # original string so quoting and unrelated arguments remain unchanged.
  if(NOT _quasar_hip_compiler_args STREQUAL "")
    separate_arguments(_quasar_hip_compiler_arg_list NATIVE_COMMAND
      "${_quasar_hip_compiler_args}")
    set(_quasar_path_option_pending "")
    foreach(_quasar_arg IN LISTS _quasar_hip_compiler_arg_list)
      if(NOT _quasar_path_option_pending STREQUAL "")
        if(NOT IS_ABSOLUTE "${_quasar_arg}" OR NOT EXISTS "${_quasar_arg}")
          message(FATAL_ERROR
            "Quasar: ${_quasar_path_option_pending} requires an existing "
            "absolute path, got '${_quasar_arg}'.")
        endif()
        file(REAL_PATH "${_quasar_arg}" _quasar_arg_real)
        string(REPLACE "${_quasar_arg}" "${_quasar_arg_real}"
          _quasar_hip_compiler_args "${_quasar_hip_compiler_args}")
        set(_quasar_path_option_pending "")
        continue()
      endif()

      foreach(_quasar_path_option IN ITEMS
          --rocm-path
          --rocm-device-lib-path
          --hip-path
          --hipstdpar-path
          --hipstdpar-prim-path
          --hipstdpar-thrust-path)
        if(_quasar_arg STREQUAL "${_quasar_path_option}")
          set(_quasar_path_option_pending "${_quasar_path_option}")
          break()
        endif()

        string(FIND "${_quasar_arg}" "${_quasar_path_option}="
          _quasar_path_option_position)
        if(_quasar_path_option_position EQUAL 0)
          string(LENGTH "${_quasar_path_option}="
            _quasar_path_option_prefix_length)
          string(SUBSTRING "${_quasar_arg}"
            ${_quasar_path_option_prefix_length} -1 _quasar_arg_path)
          if(_quasar_arg_path STREQUAL "" OR
             NOT IS_ABSOLUTE "${_quasar_arg_path}" OR
             NOT EXISTS "${_quasar_arg_path}")
            message(FATAL_ERROR
              "Quasar: ${_quasar_path_option} requires an existing absolute "
              "path, got '${_quasar_arg_path}'.")
          endif()
          file(REAL_PATH "${_quasar_arg_path}" _quasar_arg_real)
          string(REPLACE "${_quasar_arg_path}" "${_quasar_arg_real}"
            _quasar_hip_compiler_args "${_quasar_hip_compiler_args}")
          break()
        endif()
      endforeach()
    endforeach()
    if(NOT _quasar_path_option_pending STREQUAL "")
      message(FATAL_ERROR
        "Quasar: ${_quasar_path_option_pending} requires a path argument.")
    endif()
  endif()

  # Check the old pin before changing CMAKE_HIP_COMPILER. CMake otherwise sees
  # the compiler change during enable_language(), clears most of CMakeCache.txt,
  # and reruns configuration before a later comparison can diagnose it. Besides
  # defeating the pin, that reset silently restores unrelated project options
  # to their defaults.
  if(DEFINED QUASAR_PINNED_HIP_COMPILER AND
     NOT "${QUASAR_PINNED_HIP_COMPILER}" STREQUAL
         "${_quasar_hip_compiler}")
    message(FATAL_ERROR
      "Quasar: the HIP compiler changed from ${QUASAR_PINNED_HIP_COMPILER} to "
      "${_quasar_hip_compiler}; use a fresh build tree.")
  endif()
  if(DEFINED QUASAR_PINNED_HIP_COMPILER_ARG1 AND
     NOT "${QUASAR_PINNED_HIP_COMPILER_ARG1}" STREQUAL
         "${_quasar_hip_compiler_args}")
    message(FATAL_ERROR
      "Quasar: the HIP compiler arguments changed from "
      "'${QUASAR_PINNED_HIP_COMPILER_ARG1}' to "
      "'${_quasar_hip_compiler_args}'; use a fresh build tree.")
  endif()

  set(CMAKE_HIP_COMPILER "${_quasar_hip_compiler}" CACHE FILEPATH
      "Physical HIP compiler pinned for this build tree" FORCE)
  if(NOT _quasar_hip_compiler_args STREQUAL "")
    set(CMAKE_HIP_COMPILER_ARG1 "${_quasar_hip_compiler_args}" CACHE STRING
        "Arguments accompanying the pinned HIP compiler" FORCE)
  else()
    unset(CMAKE_HIP_COMPILER_ARG1 CACHE)
    unset(CMAKE_HIP_COMPILER_ARG1)
  endif()
  set(QUASAR_PINNED_HIP_COMPILER "${_quasar_hip_compiler}" CACHE INTERNAL
      "Physical HIP compiler fixed for this build tree" FORCE)
  set(QUASAR_PINNED_HIP_COMPILER_ARG1 "${_quasar_hip_compiler_args}"
      CACHE INTERNAL "HIP compiler arguments fixed for this build tree" FORCE)
endif()

# Define CTest's standard BUILD_TESTING option before deriving Quasar's project-
# specific switch from it.  Without this, a plain (non-preset) configure expands
# the undefined variable to an empty default and silently omits every test.
include(CTest)

option(QUASAR_ENABLE_HIP      "Build with HIP (ROCm) support" ON)
option(QUASAR_BUILD_TESTING   "Build Quasar unit tests" ${BUILD_TESTING})
option(QUASAR_BUILD_PYTHON    "Build Python bindings (pybind11)" ON)
option(QUASAR_BUILD_BENCHMARKS
       "Build Quasar micro-benchmarks (Google Benchmark)" OFF)

if(NOT QUASAR_ENABLE_HIP)
  message(FATAL_ERROR
    "Quasar currently requires the HIP backend. Configure with "
    "-DQUASAR_ENABLE_HIP=ON; a host-only backend is not implemented.")
endif()

enable_language(HIP)

if(NOT DEFINED CMAKE_HIP_COMPILER_ROCM_ROOT OR
   CMAKE_HIP_COMPILER_ROCM_ROOT STREQUAL "")
  message(FATAL_ERROR
    "Quasar: CMAKE_HIP_COMPILER_ROCM_ROOT is unavailable; cannot pin ROCm.")
endif()
file(REAL_PATH "${CMAKE_HIP_COMPILER}" _quasar_hip_compiler_real)
file(REAL_PATH "${CMAKE_HIP_COMPILER_ROCM_ROOT}" _quasar_rocm_root_real)
if(DEFINED QUASAR_PINNED_HIP_COMPILER AND
   NOT "${QUASAR_PINNED_HIP_COMPILER}" STREQUAL
       "${_quasar_hip_compiler_real}")
  message(FATAL_ERROR
    "Quasar: the HIP compiler changed from ${QUASAR_PINNED_HIP_COMPILER} to "
    "${_quasar_hip_compiler_real}; use a fresh build tree.")
endif()
if(DEFINED QUASAR_PINNED_ROCM_ROOT AND
   NOT "${QUASAR_PINNED_ROCM_ROOT}" STREQUAL "${_quasar_rocm_root_real}")
  message(FATAL_ERROR
    "Quasar: the ROCm root changed from ${QUASAR_PINNED_ROCM_ROOT} to "
    "${_quasar_rocm_root_real}; use a fresh build tree.")
endif()
set(QUASAR_PINNED_HIP_COMPILER "${_quasar_hip_compiler_real}" CACHE INTERNAL
    "Physical HIP compiler fixed for this build tree")
set(QUASAR_PINNED_ROCM_ROOT "${_quasar_rocm_root_real}" CACHE INTERNAL
    "Physical ROCm root fixed for this build tree")
set(QUASAR_ROCM_ROOT "${QUASAR_PINNED_ROCM_ROOT}")

set(CMAKE_HIP_COMPILER "${QUASAR_PINNED_HIP_COMPILER}")
set(CMAKE_HIP_COMPILER_ROCM_ROOT "${QUASAR_ROCM_ROOT}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
  set(CMAKE_HIP_ARCHITECTURES "gfx942" CACHE STRING
    "AMDGPU targets for HIP device code" FORCE)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(QuasarAddModule)
include(QuasarHipRuntime)
include(QuasarLinearAlgebra)
include(QuasarDistributed)

quasar_check_hip_runtime()
quasar_find_linear_algebra()

if(QUASAR_BUILD_TESTING)
  enable_testing()
endif()

if(QUASAR_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  # pybind11 2.x otherwise falls back to its legacy FindPythonLibs path, which
  # can select a different interpreter/ABI from the Python3 package above when
  # several Python versions are installed (and can even reuse stale cache
  # entries).  Reuse the modern FindPython result so the extension suffix,
  # headers, and test interpreter always describe the same Python runtime.
  set(PYBIND11_FINDPYTHON ON CACHE BOOL
      "Use CMake FindPython for pybind11" FORCE)
endif()

add_subdirectory(extern)
add_subdirectory(src)

if(QUASAR_ENABLE_HIP)
  add_subdirectory(apps)
endif()

if(QUASAR_BUILD_PYTHON)
  add_subdirectory(bindings)
endif()

if(QUASAR_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(QUASAR_BUILD_TESTING)
  add_subdirectory(tests)
  quasar_configure_test_rocm_runtime()
endif()

message(STATUS "Quasar configured:")
message(STATUS "  HIP architectures     : ${CMAKE_HIP_ARCHITECTURES}")
message(STATUS "  HIP runtime visible   : ${QUASAR_HAS_HIP_RUNTIME}")
message(STATUS "  Distributed request   : ${QUASAR_ENABLE_DISTRIBUTED}")
message(STATUS "  Distributed available : ${QUASAR_DISTRIBUTED_AVAILABLE}")
message(STATUS "  C++ standard          : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build tests           : ${QUASAR_BUILD_TESTING}")
message(STATUS "  Build Python bindings : ${QUASAR_BUILD_PYTHON}")
message(STATUS "  Build benchmarks      : ${QUASAR_BUILD_BENCHMARKS}")
