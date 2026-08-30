if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_stability_hip_sources
  flux_coordinates.hip
  radial_domains.hip
  chebyshev.hip
  newcomb_energy.hip
  toroidal_equilibrium.hip
  toroidal_energy.hip
  spectral_blocks.hip
)

quasar_add_hip_module(stability_hip SOURCES ${_quasar_stability_hip_sources})
quasar_add_backend_src_include(stability_hip)
# quasar_numerics_hip supplies the public Cholesky, block-tridiagonal, and
# shift-invert entry points this module calls.  It is linked as a normal public
# dependency precisely so that nothing here has to reach into that module's
# private headers under src/.
target_link_libraries(quasar_stability_hip
  PUBLIC
    quasar::linear_algebra
    quasar_numerics_hip
)

# Same reasoning as the equilibrium HIP module: the stability path builds on
# equilibrium output whose kernels are compiled without FP contraction, and the
# metric is a chain of differences whose cancellation makes contraction's effect
# unpredictable rather than merely small. Keeping the two modules on the same
# footing also means a quantity can be traced across the boundary without
# wondering which side reassociated it.
target_compile_options(quasar_stability_hip
  PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)
