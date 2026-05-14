cmake_minimum_required(VERSION 3.27)

project(quasar
  VERSION 0.1.0
  DESCRIPTION "HIP-accelerated numerical simulation framework"
  LANGUAGES CXX
)

option(QUASAR_ENABLE_HIP      "Build with HIP (ROCm) support" ON)
option(QUASAR_BUILD_TESTING   "Build Quasar unit tests" ${BUILD_TESTING})
option(QUASAR_BUILD_PYTHON    "Build Python bindings (pybind11)" ON)
option(QUASAR_BUILD_BENCHMARKS
       "Build Quasar micro-benchmarks (Google Benchmark)" OFF)

if(NOT QUASAR_ENABLE_HIP)
  message(FATAL_ERROR
    "Phase 1 of Quasar is HIP-only. Configure with -DQUASAR_ENABLE_HIP=ON. "
    "Host backends arrive after the magnetostatics vertical slice lands.")
endif()

enable_language(HIP)

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

quasar_check_hip_runtime()

if(QUASAR_BUILD_TESTING)
  include(CTest)
  enable_testing()
endif()

if(QUASAR_BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
endif()

add_subdirectory(extern)
add_subdirectory(src)

if(QUASAR_BUILD_PYTHON)
  add_subdirectory(bindings)
endif()

if(QUASAR_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(QUASAR_BUILD_TESTING)
  add_subdirectory(tests)
endif()

message(STATUS "Quasar configured:")
message(STATUS "  HIP architectures     : ${CMAKE_HIP_ARCHITECTURES}")
message(STATUS "  HIP runtime visible   : ${QUASAR_HAS_HIP_RUNTIME}")
message(STATUS "  C++ standard          : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Build tests           : ${QUASAR_BUILD_TESTING}")
message(STATUS "  Build Python bindings : ${QUASAR_BUILD_PYTHON}")
message(STATUS "  Build benchmarks      : ${QUASAR_BUILD_BENCHMARKS}")
