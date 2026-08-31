if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_mhd_sources
  mhd_solver.cpp
  initial_conditions.cpp
  background_builder.cpp
  mhd_geometric_source.cpp
  mhd_boundary.cpp
)

set_source_files_properties(${_quasar_mhd_sources}
  PROPERTIES LANGUAGE HIP)

# REGISTERS: the per-side MHD boundary self-registrations (mhd_boundary.cpp) carry
# no externally referenced symbol, so WHOLE_ARCHIVE keeps them in the link and the
# registry populated for Registry<IMhd*Boundary>::create(name) at the deck boundary.
quasar_add_module(mhd REGISTERS SOURCES ${_quasar_mhd_sources})

if(TARGET quasar_mhd_hip)
  target_link_libraries(quasar_mhd PUBLIC quasar_mhd_hip)
endif()
