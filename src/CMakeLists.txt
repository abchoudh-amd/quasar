# Aggregate INTERFACE target that downstream consumers (apps, tests, bindings)
# link against. Per-module STATIC libraries register themselves via
# quasar_add_module / quasar_add_hip_module, which calls
# target_link_libraries(quasar_core INTERFACE ...).

add_library(quasar_core INTERFACE)
add_library(quasar::core ALIAS quasar_core)

target_include_directories(quasar_core
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_compile_features(quasar_core INTERFACE cxx_std_20)

add_subdirectory(backend)
add_subdirectory(physics)

# Note: `core/` has no .cpp files (header-only types/Field/Registry), so it
# does not need its own subdirectory.
