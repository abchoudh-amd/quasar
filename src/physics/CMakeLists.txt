if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

add_subdirectory(analytic_fields)
add_subdirectory(magnetostatics)
add_subdirectory(pic)
add_subdirectory(mhd)
add_subdirectory(equilibrium)
