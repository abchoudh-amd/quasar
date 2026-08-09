set(_quasar_boundary_sources
  periodic.cpp
  internal.cpp
  wall.cpp
  outflow.cpp
  axis.cpp
)

set_source_files_properties(${_quasar_boundary_sources} PROPERTIES LANGUAGE HIP)

quasar_add_module(boundary REGISTERS SOURCES ${_quasar_boundary_sources})

# The boundary BCs (periodic/wall/outflow/axis) call launch_pic_* device kernels
# and operate on pic::ParticleSpecies / YeeField2D (EM-PIC is their only
# consumer), so boundary links the PIC backend. This inverts the nominal
# physics->boundary dependency direction; it is an accepted single-consumer
# tradeoff documented in CLAUDE.md (axis-orthogonality note). Split when a non-PIC
# consumer of these boundaries appears.
if(TARGET quasar_pic_hip)
  target_link_libraries(quasar_boundary PUBLIC quasar_pic_hip)
endif()
