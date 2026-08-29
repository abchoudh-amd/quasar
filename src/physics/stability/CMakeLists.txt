if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

# Plain-C++ orchestration.  All field construction, matrix assembly, and dense
# linear algebra are dispatched to the HIP modules through their public APIs.
quasar_add_module(stability SOURCES
  stability_solver.cpp
)

target_link_libraries(quasar_stability
  PUBLIC
    quasar_equilibrium
    quasar_stability_hip
    quasar_numerics_hip
)
