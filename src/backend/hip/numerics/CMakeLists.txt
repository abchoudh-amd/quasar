if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

quasar_add_hip_module(numerics_hip SOURCES
  generalized_eigensolver.hip
  block_tridiagonal.hip
  condition_estimator.hip
  cholesky.hip
  shift_invert_lanczos.hip
  batched_lu.hip
  radial_moments_hip.hip
)
quasar_add_backend_src_include(numerics_hip)

target_link_libraries(quasar_numerics_hip
  PUBLIC
    quasar::linear_algebra
)
