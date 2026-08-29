if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

# Plain C++: this module is the orchestration layer. All arithmetic lives in
# the kernels under src/backend/hip/equilibrium/, reached through the launch
# ABI in physics/equilibrium/kernels.hpp.
set(_quasar_equilibrium_sources
  gs_solver.cpp
)

quasar_add_module(equilibrium SOURCES ${_quasar_equilibrium_sources})

if(TARGET quasar_equilibrium_hip)
  target_link_libraries(quasar_equilibrium PUBLIC quasar_equilibrium_hip)
endif()
