set(_quasar_analytic_fields_sources
  uniform.cpp
  gradient.cpp
  dipole.cpp
  file_grid.cpp
)

quasar_add_module(analytic_fields REGISTERS SOURCES ${_quasar_analytic_fields_sources})
