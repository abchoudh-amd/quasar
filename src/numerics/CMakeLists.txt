set(_quasar_numerics_sources
  field_solver.cpp
  filter.cpp
  mhd_state.cpp
  hlld_riemann.cpp
  flux_reconstruction.cpp
  ct_scheme.cpp
  ssprk_integrator.cpp
  positivity_limiter.cpp
  mhd_background_profile.cpp
)

set_source_files_properties(filter.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(mhd_state.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(hlld_riemann.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(flux_reconstruction.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(ct_scheme.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(ssprk_integrator.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(positivity_limiter.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(mhd_background_profile.cpp PROPERTIES LANGUAGE HIP)

quasar_add_module(numerics REGISTERS SOURCES ${_quasar_numerics_sources})

# filter.cpp calls launch_pic_* device kernels (the filter scheme's only consumer
# is EM-PIC), so numerics links the PIC backend. This inverts the nominal
# physics->numerics dependency direction; it is an accepted single-consumer
# tradeoff documented in CLAUDE.md (axis-orthogonality note). Split when a non-PIC
# consumer of these schemes appears.
if(TARGET quasar_pic_hip)
  target_link_libraries(quasar_numerics PUBLIC quasar_pic_hip)
endif()
