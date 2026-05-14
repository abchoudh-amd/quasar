# Third-party dependencies fetched via FetchContent at configure time.

include(FetchContent)

if(QUASAR_BUILD_TESTING)
  set(_googletest_tag "v1.14.0")
  message(STATUS "extern: fetching googletest ${_googletest_tag}")

  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        ${_googletest_tag}
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(googletest)
endif()

if(QUASAR_BUILD_PYTHON)
  set(_pybind11_tag "v2.13.6")
  message(STATUS "extern: fetching pybind11 ${_pybind11_tag}")

  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        ${_pybind11_tag}
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(pybind11)
endif()

if(QUASAR_BUILD_BENCHMARKS)
  set(_benchmark_tag "v1.9.0")
  message(STATUS "extern: fetching google benchmark ${_benchmark_tag}")

  set(BENCHMARK_ENABLE_TESTING       OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS   OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL       OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_USE_BUNDLED_GTEST    OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        ${_benchmark_tag}
    GIT_SHALLOW    TRUE
  )
  FetchContent_MakeAvailable(googlebenchmark)
endif()
