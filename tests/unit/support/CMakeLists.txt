# Header-only helpers shared by unit tests and benchmarks. INTERFACE so it adds
# only an include path; nothing here is compiled into a library.
add_library(quasar_test_support INTERFACE)
target_include_directories(quasar_test_support
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
