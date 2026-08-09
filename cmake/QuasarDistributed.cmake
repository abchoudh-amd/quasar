# QuasarDistributed.cmake
#
# Resolve the tri-state distributed-runtime configuration.  This module is
# intentionally evaluated at top-level file scope because HDF5's C component
# requires the C language to be enabled, and enable_language() is not legal
# from inside a CMake function.

include_guard(GLOBAL)

set(QUASAR_ENABLE_DISTRIBUTED "AUTO" CACHE STRING
    "Build distributed MPI/multi-GPU support (AUTO, ON, or OFF)")
set_property(CACHE QUASAR_ENABLE_DISTRIBUTED PROPERTY STRINGS AUTO ON OFF)
option(QUASAR_DISTRIBUTED_TEST_HOOKS
       "Compile distributed fault-injection hooks (test builds only)" OFF)

string(TOUPPER "${QUASAR_ENABLE_DISTRIBUTED}" _quasar_distributed_mode)
if(NOT _quasar_distributed_mode MATCHES "^(AUTO|ON|OFF)$")
  message(FATAL_ERROR
    "QUASAR_ENABLE_DISTRIBUTED must be AUTO, ON, or OFF; got "
    "'${QUASAR_ENABLE_DISTRIBUTED}'.")
endif()
set(QUASAR_ENABLE_DISTRIBUTED "${_quasar_distributed_mode}" CACHE STRING
    "Build distributed MPI/multi-GPU support (AUTO, ON, or OFF)" FORCE)
set_property(CACHE QUASAR_ENABLE_DISTRIBUTED PROPERTY STRINGS AUTO ON OFF)

set(_quasar_distributed_failures)
set(_quasar_distributed_available OFF)

if(NOT QUASAR_ENABLE_DISTRIBUTED STREQUAL "OFF")
  find_package(Threads QUIET)
  if(NOT Threads_FOUND)
    list(APPEND _quasar_distributed_failures "Threads was not found")
  endif()

  find_package(MPI 3.1 QUIET COMPONENTS CXX)
  if(NOT MPI_FOUND OR NOT MPI_CXX_FOUND)
    if(DEFINED MPI_CXX_VERSION AND NOT MPI_CXX_VERSION STREQUAL "")
      list(APPEND _quasar_distributed_failures
        "MPI C++ >= 3.1 was not found (detected ${MPI_CXX_VERSION})")
    else()
      list(APPEND _quasar_distributed_failures "MPI C++ >= 3.1 was not found")
    endif()
  elseif(NOT DEFINED MPI_CXX_VERSION OR MPI_CXX_VERSION VERSION_LESS "3.1")
    list(APPEND _quasar_distributed_failures
      "MPI C++ >= 3.1 is required (detected ${MPI_CXX_VERSION})")
  endif()

  # FindHDF5's C component needs C enabled.  Check first so AUTO can degrade
  # gracefully when a C compiler is unavailable instead of failing configure.
  include(CheckLanguage)
  check_language(C)
  if(CMAKE_C_COMPILER)
    enable_language(C)
    set(HDF5_PREFER_PARALLEL TRUE)
    find_package(HDF5 1.10 QUIET COMPONENTS C)

    if(NOT HDF5_FOUND)
      list(APPEND _quasar_distributed_failures
        "parallel HDF5 C >= 1.10 was not found")
    elseif(NOT HDF5_IS_PARALLEL AND NOT HDF5_ENABLE_PARALLEL)
      list(APPEND _quasar_distributed_failures
        "parallel HDF5 C >= 1.10 is required (detected serial HDF5 ${HDF5_VERSION})")
    endif()
  else()
    list(APPEND _quasar_distributed_failures
      "a working C compiler is required to discover HDF5 C")
  endif()

  if(NOT _quasar_distributed_failures)
    set(_quasar_distributed_probe_source [=[
#include <hdf5.h>
#include <mpi.h>

int main(int argc, char** argv) {
  int provided = MPI_THREAD_SINGLE;
  const int mpi_status =
      MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if (mpi_status != MPI_SUCCESS) {
    return 1;
  }

  const hid_t access = H5Pcreate(H5P_FILE_ACCESS);
  if (access < 0) {
    MPI_Finalize();
    return 2;
  }
  const herr_t hdf5_status =
      H5Pset_fapl_mpio(access, MPI_COMM_WORLD, MPI_INFO_NULL);
  H5Pclose(access);
  MPI_Finalize();
  return hdf5_status < 0 ? 3 : 0;
}
]=])

    try_compile(_quasar_distributed_probe_ok
      SOURCE_FROM_VAR quasar_distributed_probe.cpp
                      _quasar_distributed_probe_source
      LINK_LIBRARIES MPI::MPI_CXX Threads::Threads HDF5::HDF5
      CXX_STANDARD 20
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
      LOG_DESCRIPTION
        "Checking MPI_Init_thread and H5Pset_fapl_mpio compile/link support"
      NO_CACHE
      OUTPUT_VARIABLE _quasar_distributed_probe_output
    )

    if(_quasar_distributed_probe_ok)
      set(_quasar_distributed_available ON)
    else()
      list(APPEND _quasar_distributed_failures
        "the combined MPI_Init_thread/H5Pset_fapl_mpio compile/link probe failed (see CMakeFiles/CMakeConfigureLog.yaml)")
      message(DEBUG
        "Quasar distributed compile/link probe output:\n"
        "${_quasar_distributed_probe_output}")
    endif()
  endif()
endif()

set(QUASAR_DISTRIBUTED_AVAILABLE ${_quasar_distributed_available} CACHE INTERNAL
    "Whether the complete Quasar distributed runtime can be built" FORCE)
string(JOIN "; " _quasar_distributed_failure_summary
       ${_quasar_distributed_failures})
set(QUASAR_DISTRIBUTED_FAILURE_REASONS
    "${_quasar_distributed_failure_summary}" CACHE INTERNAL
    "Why AUTO distributed support resolved to unavailable" FORCE)

if(_quasar_distributed_failures)
  string(JOIN "\n  - " _quasar_distributed_failure_text
         ${_quasar_distributed_failures})
  if(QUASAR_ENABLE_DISTRIBUTED STREQUAL "ON")
    message(FATAL_ERROR
      "QUASAR_ENABLE_DISTRIBUTED=ON requires the complete distributed "
      "dependency set:\n"
      "  - ${_quasar_distributed_failure_text}\n"
      "Install compatible MPI and parallel-HDF5 development packages, or "
      "configure with QUASAR_ENABLE_DISTRIBUTED=AUTO/OFF.")
  else()
    message(STATUS
      "Quasar distributed support: AUTO resolved to OFF:\n"
      "  - ${_quasar_distributed_failure_text}")
  endif()
elseif(QUASAR_ENABLE_DISTRIBUTED STREQUAL "OFF")
  message(STATUS "Quasar distributed support: disabled (OFF)")
else()
  message(STATUS
    "Quasar distributed support: enabled "
    "(MPI ${MPI_CXX_VERSION}, parallel HDF5 ${HDF5_VERSION})")
endif()
