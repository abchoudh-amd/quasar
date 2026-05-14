add_executable(quasar_test_core_registry test_registry.cpp)

target_link_libraries(quasar_test_core_registry
  PRIVATE
    quasar::core
    GTest::gtest_main
)

add_test(
  NAME    core_test_registry
  COMMAND quasar_test_core_registry
)
