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

function(_quasar_define_module_target name sources is_hip registers)
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
    if(registers)
      # This module self-registers concrete schemes via namespace-scope static
      # initializers (core/registry.hpp). Those objects carry no externally
      # referenced symbol, so a plain static-archive link drops them and the
      # registrations vanish. WHOLE_ARCHIVE forces every object in, keeping the
      # registry populated for Registry<Base>::create(name) at the deck boundary.
      target_link_libraries(quasar_core
        INTERFACE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${_target}>")
    else()
      target_link_libraries(quasar_core INTERFACE ${_target})
    endif()
  endif()
endfunction()

function(quasar_add_module name)
  cmake_parse_arguments(QAM "REGISTERS" "" "SOURCES" ${ARGN})
  if(NOT QAM_SOURCES)
    message(FATAL_ERROR "quasar_add_module(${name}): SOURCES is required.")
  endif()
  _quasar_define_module_target("${name}" "${QAM_SOURCES}" FALSE "${QAM_REGISTERS}")
endfunction()

function(quasar_add_hip_module name)
  cmake_parse_arguments(QAHM "REGISTERS" "" "SOURCES" ${ARGN})
  if(NOT QAHM_SOURCES)
    message(FATAL_ERROR "quasar_add_hip_module(${name}): SOURCES is required.")
  endif()
  if(NOT QUASAR_ENABLE_HIP)
    message(FATAL_ERROR
      "quasar_add_hip_module(${name}) requires QUASAR_ENABLE_HIP=ON.")
  endif()
  _quasar_define_module_target("${name}" "${QAHM_SOURCES}" TRUE "${QAHM_REGISTERS}")
endfunction()
