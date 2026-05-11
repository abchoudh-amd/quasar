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
