# QuasarLinearAlgebra.cmake
#
# Resolve hipBLAS and hipSOLVER from the same ROCm installation that CMake's
# HIP compiler selected.  On development hosts `current` may point at a newer
# nightly than an already-configured gfx target; allowing normal prefix search
# here can therefore compile against one release and load another release's
# libraries.  CMAKE_HIP_COMPILER_ROCM_ROOT is CMake's authoritative toolchain
# prefix and is the only package root accepted below.

include_guard(GLOBAL)

function(quasar_find_linear_algebra)
  if(NOT QUASAR_ENABLE_HIP)
    return()
  endif()
  if(NOT DEFINED CMAKE_HIP_COMPILER_ROCM_ROOT OR
     CMAKE_HIP_COMPILER_ROCM_ROOT STREQUAL "")
    message(FATAL_ERROR
      "QuasarLinearAlgebra: CMAKE_HIP_COMPILER_ROCM_ROOT is unavailable; "
      "cannot select hipBLAS/hipSOLVER consistently with the HIP compiler.")
  endif()

  cmake_path(ABSOLUTE_PATH CMAKE_HIP_COMPILER_ROCM_ROOT
             NORMALIZE OUTPUT_VARIABLE _quasar_rocm_root)
  foreach(_package IN ITEMS hip hipblas-common hipblas hipsolver)
    set(${_package}_DIR
        "${_quasar_rocm_root}/lib/cmake/${_package}")
  endforeach()

  # CMAKE_PREFIX_PATH is local to this function.  It guides dependencies loaded
  # by the package configs (notably `hip`) without polluting later discovery.
  list(PREPEND CMAKE_PREFIX_PATH "${_quasar_rocm_root}")
  find_package(hipblas CONFIG REQUIRED
    PATHS "${_quasar_rocm_root}/lib/cmake/hipblas"
    NO_DEFAULT_PATH)
  find_package(hipsolver CONFIG REQUIRED
    PATHS "${_quasar_rocm_root}/lib/cmake/hipsolver"
    NO_DEFAULT_PATH)

  if(NOT TARGET roc::hipblas OR NOT TARGET roc::hipsolver)
    message(FATAL_ERROR
      "QuasarLinearAlgebra: ROCm package configs did not define "
      "roc::hipblas and roc::hipsolver.")
  endif()

  # Deterministic mode is part of the solver-handle contract.  hipSOLVER's
  # package-version compatibility changed across ROCm releases, so a numeric
  # find_package minimum can reject a newer installation (for example 3.x when
  # asking for 2.3).  Check the exact API we require instead and fail during
  # configuration rather than later in a HIP translation unit.
  include(CheckCXXSourceCompiles)
  set(_quasar_saved_required_libraries "${CMAKE_REQUIRED_LIBRARIES}")
  set(CMAKE_REQUIRED_LIBRARIES roc::hipsolver)
  # The selected ROCm root may change inside an existing build tree.  Do not
  # reuse a successful probe from a different installation.
  unset(QUASAR_HIPSOLVER_HAS_DETERMINISTIC_MODE CACHE)
  check_cxx_source_compiles(
    "#include <hipsolver/hipsolver.h>
     int main() {
       auto* function = &hipsolverSetDeterministicMode;
       (void)function;
       return 0;
     }"
    QUASAR_HIPSOLVER_HAS_DETERMINISTIC_MODE)
  set(CMAKE_REQUIRED_LIBRARIES "${_quasar_saved_required_libraries}")
  unset(_quasar_saved_required_libraries)
  if(NOT QUASAR_HIPSOLVER_HAS_DETERMINISTIC_MODE)
    message(FATAL_ERROR
      "QuasarLinearAlgebra: hipSOLVER lacks "
      "hipsolverSetDeterministicMode (requires hipSOLVER >= 2.3).")
  endif()

  # Fail at configure time if a stale cache entry nevertheless injected a
  # library from another ROCm tree.
  file(REAL_PATH "${_quasar_rocm_root}" _quasar_rocm_root_real)
  foreach(_target IN ITEMS roc::hipblas roc::hipsolver)
    get_target_property(_location ${_target} IMPORTED_LOCATION_RELEASE)
    if(NOT _location)
      get_target_property(_location ${_target} IMPORTED_LOCATION)
    endif()
    if(_location)
      file(REAL_PATH "${_location}" _location_real)
      string(FIND "${_location_real}" "${_quasar_rocm_root_real}/"
             _prefix_position)
      if(NOT _prefix_position EQUAL 0)
        message(FATAL_ERROR
          "QuasarLinearAlgebra: ${_target} resolved to ${_location_real}, "
          "outside HIP compiler root ${_quasar_rocm_root_real}.")
      endif()
    endif()
  endforeach()

  add_library(quasar_linear_algebra INTERFACE)
  add_library(quasar::linear_algebra ALIAS quasar_linear_algebra)
  target_link_libraries(quasar_linear_algebra
    INTERFACE
      roc::hipsolver
      roc::hipblas
  )

  # Linking an absolute .so still records its SONAME.  These development hosts
  # commonly put `rocm-nightly/current/lib` first in LD_LIBRARY_PATH, which
  # would otherwise replace the compiler-root hipSOLVER/hipBLAS at load time.
  # Old-style DT_RPATH is intentional here: unlike DT_RUNPATH it takes
  # precedence over LD_LIBRARY_PATH for these direct dependencies.  Tests also
  # prepend this root to LD_LIBRARY_PATH because ROCm DSOs carry their own
  # RUNPATH and must resolve their transitive dependencies from the same tree.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(quasar_linear_algebra INTERFACE
      "LINKER:--disable-new-dtags"
      "LINKER:-rpath,${_quasar_rocm_root}/lib"
    )
    if(IS_DIRECTORY "${_quasar_rocm_root}/lib64")
      target_link_options(quasar_linear_algebra INTERFACE
        "LINKER:-rpath,${_quasar_rocm_root}/lib64")
    endif()
  endif()

  message(STATUS
    "QuasarLinearAlgebra: hipBLAS/hipSOLVER from ${_quasar_rocm_root}")
endfunction()
