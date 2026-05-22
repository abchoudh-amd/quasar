if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

add_executable(quasar_pic_app quasar_pic/main.cpp)
set_source_files_properties(quasar_pic/main.cpp PROPERTIES LANGUAGE HIP)
set_target_properties(quasar_pic_app PROPERTIES OUTPUT_NAME quasar_pic)
target_link_libraries(quasar_pic_app PRIVATE quasar::core)
