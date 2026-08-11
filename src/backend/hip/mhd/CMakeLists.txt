if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_mhd_hip_sources
  mhd_reconstruct.hip
  mhd_riemann.hip
  mhd_ct.hip
  mhd_halo.hip
  mhd_background_momentum.hip
  mhd_cylindrical_momentum.hip
  mhd_split_energy.hip
  mhd_update.hip
  mhd_reduce.hip
  mhd_boundary_hip.hip
)

quasar_add_hip_module(mhd_hip SOURCES ${_quasar_mhd_hip_sources})
quasar_add_backend_src_include(mhd_hip)
