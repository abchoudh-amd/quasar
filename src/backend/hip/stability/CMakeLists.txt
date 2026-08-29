if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_stability_hip_sources
  flux_coordinates.hip
  radial_domains.hip
  chebyshev.hip
)

quasar_add_hip_module(stability_hip SOURCES ${_quasar_stability_hip_sources})
quasar_add_backend_src_include(stability_hip)

# Same reasoning as the equilibrium HIP module: the stability path builds on
# equilibrium output whose kernels are compiled without FP contraction, and the
# metric is a chain of differences whose cancellation makes contraction's effect
# unpredictable rather than merely small. Keeping the two modules on the same
# footing also means a quantity can be traced across the boundary without
# wondering which side reassociated it.
target_compile_options(quasar_stability_hip
  PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)
