if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_pic_hip_sources
  fdtd_b_hip.hip
  fdtd_e_hip.hip
  gather_push_hip.hip
  deposit_hip.hip
  boundary_periodic_hip.hip
  boundary_pec_hip.hip
  boundary_outflow_hip.hip
  boundary_specular_hip.hip
  boundary_absorb_hip.hip
  particle_compact_hip.hip
  halo_hip.hip
  filter_hip.hip
  fdtd_b_cyl_hip.hip
  fdtd_e_cyl_hip.hip
  gather_push_cyl_hip.hip
  deposit_cyl_hip.hip
  boundary_axis_hip.hip
)

quasar_add_hip_module(pic_hip SOURCES ${_quasar_pic_hip_sources})
quasar_add_backend_src_include(pic_hip)
