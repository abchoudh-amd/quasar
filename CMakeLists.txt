cmake_minimum_required(VERSION 3.27)

project(quasar
  VERSION 0.1.0
  DESCRIPTION "HIP-accelerated numerical simulation framework"
  LANGUAGES CXX
)

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
include(QuasarDistributed)

quasar_check_hip_runtime()

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
