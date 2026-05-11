# QuasarAddModule.cmake
#
# Helpers for declaring per-module STATIC libraries that auto-attach to the
# top-level `quasar_core` INTERFACE target.
#
#   quasar_add_module(<name> SOURCES <files...>)
#     - Plain C++ sources only.
#
#   quasar_add_hip_module(<name> SOURCES <files...>)
#     - Tags every *.hip source with LANGUAGE HIP before adding the target.

include_guard(GLOBAL)

function(_quasar_define_module_target name sources is_hip)
  set(_target "quasar_${name}")
  add_library(${_target} STATIC ${sources})

  if(is_hip)
    foreach(_src IN LISTS sources)
      if(_src MATCHES "\\.hip$")
        set_source_files_properties("${_src}" PROPERTIES LANGUAGE HIP)
      endif()
    endforeach()
  endif()

  target_include_directories(${_target}
    PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    PRIVATE
      ${PROJECT_SOURCE_DIR}/src
  )

  target_compile_features(${_target} PUBLIC cxx_std_20)

  if(TARGET quasar_core)
    target_link_libraries(quasar_core INTERFACE ${_target})
  endif()
endfunction()

function(quasar_add_module name)
  cmake_parse_arguments(QAM "" "" "SOURCES" ${ARGN})
  if(NOT QAM_SOURCES)
    message(FATAL_ERROR "quasar_add_module(${name}): SOURCES is required.")
  endif()
  _quasar_define_module_target("${name}" "${QAM_SOURCES}" FALSE)
endfunction()

function(quasar_add_hip_module name)
  cmake_parse_arguments(QAHM "" "" "SOURCES" ${ARGN})
  if(NOT QAHM_SOURCES)
    message(FATAL_ERROR "quasar_add_hip_module(${name}): SOURCES is required.")
  endif()
  if(NOT QUASAR_ENABLE_HIP)
    message(FATAL_ERROR
      "quasar_add_hip_module(${name}) requires QUASAR_ENABLE_HIP=ON.")
  endif()
  _quasar_define_module_target("${name}" "${QAHM_SOURCES}" TRUE)
endfunction()
