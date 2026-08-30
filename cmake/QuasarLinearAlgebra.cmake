# QuasarLinearAlgebra.cmake
#
# Resolve hipBLAS and hipSOLVER from the same ROCm installation that CMake's
# HIP compiler selected.  On development hosts `current` may point at a newer
# nightly than an already-configured gfx target; allowing normal prefix search
# here can therefore compile against one release and load another release's
# libraries.  CMAKE_HIP_COMPILER_ROCM_ROOT is CMake's authoritative toolchain
# prefix is checked once at the top-level configuration boundary and persisted
# as QUASAR_ROCM_ROOT, which is the only package root accepted below.

include_guard(GLOBAL)

function(_quasar_require_rocm_path path description root)
  if(path MATCHES "[$]<")
    message(FATAL_ERROR
      "QuasarLinearAlgebra: ${description} contains an unvalidated generator "
      "expression: ${path}")
  endif()
  if(NOT IS_ABSOLUTE "${path}" OR NOT EXISTS "${path}")
    message(FATAL_ERROR
      "QuasarLinearAlgebra: ${description} is not an existing absolute path: "
      "${path}")
  endif()
  file(REAL_PATH "${path}" _quasar_path_real)
  cmake_path(IS_PREFIX root "${_quasar_path_real}" NORMALIZE
             _quasar_path_in_root)
  if(NOT _quasar_path_in_root)
    message(FATAL_ERROR
      "QuasarLinearAlgebra: ${description} resolved to ${_quasar_path_real}, "
      "outside pinned ROCm root ${root}.")
  endif()
endfunction()

function(_quasar_validate_rocm_imported_target target root)
  if(NOT TARGET "${target}")
    message(FATAL_ERROR
      "QuasarLinearAlgebra: ROCm package configs did not define ${target}.")
  endif()
  get_target_property(_quasar_imported "${target}" IMPORTED)
  if(NOT _quasar_imported)
    message(FATAL_ERROR
      "QuasarLinearAlgebra: ${target} already exists but is not an imported "
      "target from the pinned ROCm installation.")
  endif()

  set(_quasar_imported_locations)
  get_target_property(_quasar_imported_configs
    "${target}" IMPORTED_CONFIGURATIONS)
  if(NOT _quasar_imported_configs)
    set(_quasar_imported_configs)
  endif()
  list(APPEND _quasar_imported_configs
    NOCONFIG DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  list(REMOVE_DUPLICATES _quasar_imported_configs)
  foreach(_config IN LISTS _quasar_imported_configs)
    string(TOUPPER "${_config}" _config_upper)
    foreach(_property IN ITEMS IMPORTED_LOCATION IMPORTED_IMPLIB)
      get_target_property(_location
        "${target}" "${_property}_${_config_upper}")
      if(_location)
        list(APPEND _quasar_imported_locations "${_location}")
      endif()
    endforeach()
  endforeach()
  foreach(_property IN ITEMS IMPORTED_LOCATION IMPORTED_IMPLIB)
    get_target_property(_location "${target}" "${_property}")
    if(_location)
      list(APPEND _quasar_imported_locations "${_location}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _quasar_imported_locations)
  if(NOT _quasar_imported_locations)
    message(FATAL_ERROR
      "QuasarLinearAlgebra: ${target} has no imported library location.")
  endif()
  foreach(_location IN LISTS _quasar_imported_locations)
    _quasar_require_rocm_path("${_location}"
      "${target} imported library" "${root}")
  endforeach()
endfunction()

function(_quasar_validate_rocm_target_paths target root)
  get_target_property(_quasar_imported "${target}" IMPORTED)
  if(NOT _quasar_imported)
    message(FATAL_ERROR
      "QuasarLinearAlgebra: ${target} already exists but is not an imported "
      "target from the pinned ROCm installation.")
  endif()

  foreach(_property IN ITEMS
      INTERFACE_INCLUDE_DIRECTORIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
    get_target_property(_paths "${target}" "${_property}")
    if(_paths)
      foreach(_path IN LISTS _paths)
        _quasar_require_rocm_path("${_path}"
          "${target} ${_property}" "${root}")
      endforeach()
    endif()
  endforeach()

  set(_quasar_link_properties INTERFACE_LINK_LIBRARIES)
  get_target_property(_quasar_imported_configs
    "${target}" IMPORTED_CONFIGURATIONS)
  if(_quasar_imported_configs)
    foreach(_config IN LISTS _quasar_imported_configs)
      string(TOUPPER "${_config}" _config_upper)
      list(APPEND _quasar_link_properties
        "IMPORTED_LINK_DEPENDENT_LIBRARIES_${_config_upper}")
    endforeach()
  endif()
  list(APPEND _quasar_link_properties IMPORTED_LINK_DEPENDENT_LIBRARIES)
  list(REMOVE_DUPLICATES _quasar_link_properties)
  foreach(_property IN LISTS _quasar_link_properties)
    get_target_property(_links "${target}" "${_property}")
    if(_links)
      foreach(_link IN LISTS _links)
        if(_link MATCHES "[$]<")
          if(target STREQUAL "hip::device" AND
             _link MATCHES
               "^[$]<[$]<LINK_LANGUAGE:CXX>:--(hip-link|offload-(arch|jobs)=[^>]*)>$")
            continue()
          endif()
          message(FATAL_ERROR
            "QuasarLinearAlgebra: ${target} ${_property} contains an "
            "unvalidated generator expression: ${_link}")
        endif()
        if(IS_ABSOLUTE "${_link}")
          _quasar_require_rocm_path("${_link}"
            "${target} ${_property}" "${root}")
        endif()
      endforeach()
    endif()
  endforeach()
endfunction()

function(_quasar_require_link_target target dependency)
  get_target_property(_quasar_links "${target}" INTERFACE_LINK_LIBRARIES)
  if(NOT _quasar_links)
    set(_quasar_links)
  endif()
  list(FIND _quasar_links "${dependency}" _quasar_dependency_position)
  if(_quasar_dependency_position EQUAL -1)
    message(FATAL_ERROR
      "QuasarLinearAlgebra: ${target} does not link the required pinned "
      "target ${dependency}.")
  endif()
endfunction()

function(quasar_find_linear_algebra)
  if(NOT QUASAR_ENABLE_HIP)
    return()
  endif()
  if(NOT DEFINED QUASAR_ROCM_ROOT OR QUASAR_ROCM_ROOT STREQUAL "")
    message(FATAL_ERROR
      "QuasarLinearAlgebra: the pinned ROCm root is unavailable; cannot "
      "select hipBLAS/hipSOLVER consistently with the HIP compiler.")
  endif()

  cmake_path(ABSOLUTE_PATH QUASAR_ROCM_ROOT
             NORMALIZE OUTPUT_VARIABLE _quasar_rocm_root)

  # ROCm installations use either lib/cmake or lib64/cmake depending on the
  # distribution.  Build both the package-search paths and runtime search paths
  # from directories that actually exist instead of assuming the former.
  set(_quasar_rocm_library_dirs)
  set(_quasar_rocm_cmake_dirs)
  foreach(_library_dir IN ITEMS lib lib64)
    set(_candidate "${_quasar_rocm_root}/${_library_dir}")
    if(IS_DIRECTORY "${_candidate}")
      file(REAL_PATH "${_candidate}" _candidate_real)
      _quasar_require_rocm_path("${_candidate_real}"
        "ROCm ${_library_dir} directory" "${_quasar_rocm_root}")
      list(APPEND _quasar_rocm_library_dirs "${_candidate_real}")
    endif()
    if(IS_DIRECTORY "${_candidate}/cmake")
      list(APPEND _quasar_rocm_cmake_dirs "${_candidate}/cmake")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _quasar_rocm_library_dirs)
  if(NOT _quasar_rocm_library_dirs OR NOT _quasar_rocm_cmake_dirs)
    message(FATAL_ERROR
      "QuasarLinearAlgebra: no ROCm lib[64]/cmake package layout exists below "
      "${_quasar_rocm_root}.")
  endif()

  # rocblas/rocsolver are required in addition to the hip* wrappers: the batched
  # LU used by the radial-moment tables has no hipSOLVER entry point (hipSOLVER
  # exposes only single-matrix getrf/getrs), so it calls
  # rocsolver_dgetrf_strided_batched directly. hipSOLVER already loads rocSOLVER
  # underneath, but that is an implementation detail of a different package and
  # is not a target we may link.
  set(_quasar_required_rocm_packages
      hip hipblas-common hipblas hipsolver rocblas rocsolver)
  if(NOT WIN32)
    list(APPEND _quasar_required_rocm_packages
      AMDDeviceLibs amd_comgr hsa-runtime64)
  endif()
  foreach(_package IN LISTS _quasar_required_rocm_packages)
    unset(_quasar_package_dir)
    foreach(_cmake_dir IN LISTS _quasar_rocm_cmake_dirs)
      if(IS_DIRECTORY "${_cmake_dir}/${_package}")
        set(_quasar_package_dir "${_cmake_dir}/${_package}")
        break()
      endif()
    endforeach()
    if(NOT _quasar_package_dir)
      message(FATAL_ERROR
        "QuasarLinearAlgebra: ${_package} package configuration is missing "
        "below ${_quasar_rocm_root}/lib[64]/cmake.")
    endif()
    # Replace stale cache entries when an existing build tree changes compiler
    # roots; allowing an old *_DIR to survive defeats the exact-root contract.
    set(${_package}_DIR "${_quasar_package_dir}" CACHE PATH
        "${_package} package directory pinned to the HIP compiler root" FORCE)
    # CMP0126 NEW preserves a caller's normal variable when the cache is set.
    # Set both scopes so a parent project cannot shadow the pinned directory.
    set(${_package}_DIR "${_quasar_package_dir}")
  endforeach()

  # CMAKE_PREFIX_PATH is local to this function.  It guides dependencies loaded
  # by the package configs (notably `hip`) without polluting later discovery.
  list(PREPEND CMAKE_PREFIX_PATH "${_quasar_rocm_root}")
  # hip-config-amd.cmake gives ENV{ROCM_PATH} precedence over its own package
  # prefix when locating AMDDeviceLibs, amd_comgr, and hsa-runtime64. Pin that
  # environment only while loading the ROCm package graph, then restore it.
  if(NOT WIN32)
    set(_quasar_had_rocm_path FALSE)
    if(DEFINED ENV{ROCM_PATH})
      set(_quasar_had_rocm_path TRUE)
      set(_quasar_saved_rocm_path "$ENV{ROCM_PATH}")
    endif()
    set(ENV{ROCM_PATH} "${_quasar_rocm_root}")
    set(ROCM_PATH "${_quasar_rocm_root}")
  endif()
  set(_quasar_hipblas_search_paths)
  set(_quasar_hipsolver_search_paths)
  set(_quasar_rocblas_search_paths)
  set(_quasar_rocsolver_search_paths)
  foreach(_cmake_dir IN LISTS _quasar_rocm_cmake_dirs)
    list(APPEND _quasar_hipblas_search_paths "${_cmake_dir}/hipblas")
    list(APPEND _quasar_hipsolver_search_paths "${_cmake_dir}/hipsolver")
    list(APPEND _quasar_rocblas_search_paths "${_cmake_dir}/rocblas")
    list(APPEND _quasar_rocsolver_search_paths "${_cmake_dir}/rocsolver")
  endforeach()
  find_package(hipblas CONFIG REQUIRED
    PATHS ${_quasar_hipblas_search_paths}
    NO_DEFAULT_PATH)
  find_package(hipsolver CONFIG REQUIRED
    PATHS ${_quasar_hipsolver_search_paths}
    NO_DEFAULT_PATH)
  find_package(rocblas CONFIG REQUIRED
    PATHS ${_quasar_rocblas_search_paths}
    NO_DEFAULT_PATH)
  find_package(rocsolver CONFIG REQUIRED
    PATHS ${_quasar_rocsolver_search_paths}
    NO_DEFAULT_PATH)
  if(NOT WIN32)
    if(_quasar_had_rocm_path)
      set(ENV{ROCM_PATH} "${_quasar_saved_rocm_path}")
    else()
      unset(ENV{ROCM_PATH})
    endif()
  endif()

  set(_quasar_required_rocm_targets
      roc::hipblas roc::hipsolver roc::hipblas-common
      roc::rocblas roc::rocsolver
      hip::amdhip64 hip::host hip::device)
  if(NOT WIN32)
    list(APPEND _quasar_required_rocm_targets
      amd_comgr hsa-runtime64::hsa-runtime64 ocml)
  endif()
  foreach(_target IN LISTS _quasar_required_rocm_targets)
    if(NOT TARGET "${_target}")
      message(FATAL_ERROR
        "QuasarLinearAlgebra: ROCm package configs did not define ${_target}.")
    endif()
  endforeach()

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

  # Fail at configure time if pre-existing package targets or stale cache
  # entries injected either libraries or headers from another ROCm tree.
  set(_quasar_rocm_library_targets
      roc::hipblas roc::hipsolver roc::rocblas roc::rocsolver hip::amdhip64)
  if(NOT WIN32)
    list(APPEND _quasar_rocm_library_targets
      amd_comgr hsa-runtime64::hsa-runtime64 ocml)
  endif()
  foreach(_target IN LISTS _quasar_rocm_library_targets)
    _quasar_validate_rocm_imported_target(
      "${_target}" "${_quasar_rocm_root}")
  endforeach()
  foreach(_target IN LISTS _quasar_required_rocm_targets)
    _quasar_validate_rocm_target_paths(
      "${_target}" "${_quasar_rocm_root}")
  endforeach()
  _quasar_require_link_target(hip::host hip::amdhip64)
  _quasar_require_link_target(hip::device hip::host)
  _quasar_require_link_target(roc::hipblas roc::hipblas-common)
  _quasar_require_link_target(roc::hipblas hip::host)
  _quasar_require_link_target(roc::hipsolver hip::host)
  _quasar_require_link_target(roc::rocsolver roc::rocblas)

  add_library(quasar_linear_algebra INTERFACE)
  add_library(quasar::linear_algebra ALIAS quasar_linear_algebra)
  target_link_libraries(quasar_linear_algebra
    INTERFACE
      roc::hipsolver
      roc::hipblas
      roc::rocsolver
      roc::rocblas
  )

  # Linking an absolute .so still records its SONAME.  These development hosts
  # commonly put `rocm-nightly/current/lib` first in LD_LIBRARY_PATH, which
  # would otherwise replace the compiler-root hipSOLVER/hipBLAS at load time.
  # Old-style DT_RPATH is intentional here: unlike DT_RUNPATH it takes
  # precedence over LD_LIBRARY_PATH for these direct dependencies.  Tests also
  # prepend this root to LD_LIBRARY_PATH because ROCm DSOs carry their own
  # RUNPATH and must resolve their transitive dependencies from the same tree.
  #
  # ROCm Clang's default rocm.cfg appends --enable-new-dtags after ordinary
  # command-line linker options.  Supplying --disable-new-dtags only through
  # target_link_options therefore still produces DT_RUNPATH.  Load a second
  # Clang config whose appended option is processed after the default config;
  # retain the ordinary linker option as the portable path for non-Clang
  # toolchains.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_quasar_linker_config_dir "${PROJECT_BINARY_DIR}/cmake")
    set(_quasar_disable_new_dtags_config
        "${_quasar_linker_config_dir}/QuasarDisableNewDtags.cfg")
    file(MAKE_DIRECTORY "${_quasar_linker_config_dir}")
    configure_file(
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/QuasarDisableNewDtags.cfg.in"
      "${_quasar_disable_new_dtags_config}"
      COPYONLY)

    set(_quasar_rocm_cxx_config_option)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      file(REAL_PATH "${CMAKE_CXX_COMPILER}" _quasar_cxx_compiler_real)
      string(FIND "${_quasar_cxx_compiler_real}"
             "${_quasar_rocm_root}/" _quasar_cxx_in_rocm_root)
      if(_quasar_cxx_in_rocm_root EQUAL 0)
        list(APPEND _quasar_rocm_cxx_config_option
          "$<$<LINK_LANGUAGE:CXX>:--config=${_quasar_disable_new_dtags_config}>")
      endif()
    endif()

    target_link_options(quasar_linear_algebra INTERFACE
      "LINKER:--disable-new-dtags"
      ${_quasar_rocm_cxx_config_option}
      "$<$<LINK_LANG_AND_ID:HIP,Clang>:--config=${_quasar_disable_new_dtags_config}>"
    )
    foreach(_library_dir IN LISTS _quasar_rocm_library_dirs)
      target_link_options(quasar_linear_algebra INTERFACE
        "LINKER:-rpath,${_library_dir}")
    endforeach()
  endif()

  message(STATUS
    "QuasarLinearAlgebra: hipBLAS/hipSOLVER from ${_quasar_rocm_root}")
endfunction()
