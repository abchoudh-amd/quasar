if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_analytic_fields_hip_sources
  analytic_fields_hip.hip
)

quasar_add_hip_module(analytic_fields_hip
  SOURCES ${_quasar_analytic_fields_hip_sources})
quasar_add_backend_src_include(analytic_fields_hip)

# -ffp-contract=off is load-bearing here for the same reason it is in
# src/backend/hip/equilibrium/: these kernels reduce through the exact
# expansions in include/quasar/numerics/scaled_arithmetic.hpp, whose two-sum is
# only error-free if each operation rounds separately. Contracting a*b+c into an
# FMA does not shift the answer by an ulp -- it silently turns the compensated
# sum back into a naive one, and the far-field dipole cancellation and the
# trilinear stencil are exactly the places where that matters.
target_compile_options(quasar_analytic_fields_hip
  PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)
