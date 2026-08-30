set(_quasar_analytic_fields_sources
  uniform.cpp
  gradient.cpp
  dipole.cpp
  file_grid.cpp
)

quasar_add_module(analytic_fields REGISTERS SOURCES ${_quasar_analytic_fields_sources})

# The evaluators are now pure orchestrators over launch_analytic_* kernels; the
# arithmetic lives in the backend module. Unlike the numerics/boundary link into
# quasar_pic_hip (an accepted inversion documented in CLAUDE.md), this is the
# ordinary direction: a physics module linking its own backend.
if(TARGET quasar_analytic_fields_hip)
  target_link_libraries(quasar_analytic_fields PUBLIC quasar_analytic_fields_hip)
endif()
