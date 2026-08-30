if(NOT DEFINED QUASAR_READELF OR QUASAR_READELF STREQUAL "")
  message(FATAL_ERROR "QUASAR_READELF is required")
endif()
if(NOT DEFINED QUASAR_TEST_BINARY OR QUASAR_TEST_BINARY STREQUAL "")
  message(FATAL_ERROR "QUASAR_TEST_BINARY is required")
endif()

execute_process(
  COMMAND "${QUASAR_READELF}" -d "${QUASAR_TEST_BINARY}"
  RESULT_VARIABLE _readelf_result
  OUTPUT_VARIABLE _dynamic_section
  ERROR_VARIABLE _readelf_error)
if(NOT _readelf_result EQUAL 0)
  message(FATAL_ERROR
    "readelf failed for ${QUASAR_TEST_BINARY}: ${_readelf_error}")
endif()

if(_dynamic_section MATCHES "\\(RUNPATH\\)")
  message(FATAL_ERROR
    "${QUASAR_TEST_BINARY} contains DT_RUNPATH; compiler-root ROCm libraries "
    "can be overridden through LD_LIBRARY_PATH")
endif()
string(REGEX MATCH "\\(RPATH\\)[^\n]*\\[([^]]*)\\]"
       _rpath_entry "${_dynamic_section}")
if(NOT _rpath_entry)
  message(FATAL_ERROR
    "${QUASAR_TEST_BINARY} does not contain the required DT_RPATH")
endif()
set(_actual_rpath "${CMAKE_MATCH_1}")
string(REPLACE ":" ";" _actual_directories "${_actual_rpath}")

if(DEFINED QUASAR_EXPECTED_RPATH AND NOT QUASAR_EXPECTED_RPATH STREQUAL "")
  string(REPLACE ":" ";" _expected_directories "${QUASAR_EXPECTED_RPATH}")

  set(_actual_normalized)
  foreach(_directory IN LISTS _actual_directories)
    if(_directory STREQUAL "")
      message(FATAL_ERROR
        "${QUASAR_TEST_BINARY} DT_RPATH contains an empty search component")
    endif()
    set(_normalized_directory "${_directory}")
    cmake_path(NORMAL_PATH _normalized_directory)
    list(APPEND _actual_normalized "${_normalized_directory}")
  endforeach()
  list(REMOVE_DUPLICATES _actual_normalized)

  set(_expected_normalized)
  foreach(_directory IN LISTS _expected_directories)
    if(_directory STREQUAL "")
      message(FATAL_ERROR "QUASAR_EXPECTED_RPATH contains an empty component")
    endif()
    set(_normalized_directory "${_directory}")
    cmake_path(NORMAL_PATH _normalized_directory)
    list(APPEND _expected_normalized "${_normalized_directory}")
  endforeach()
  list(REMOVE_DUPLICATES _expected_normalized)

  if(NOT "${_actual_normalized}" STREQUAL "${_expected_normalized}")
    message(FATAL_ERROR
      "${QUASAR_TEST_BINARY} DT_RPATH is '${_actual_rpath}', expected exactly "
      "'${QUASAR_EXPECTED_RPATH}' after normalization")
  endif()
endif()
