set(_quasar_core_sources
  normalization.cpp
  observations.cpp
  device_observations.cpp
)

quasar_add_module(core_norm SOURCES ${_quasar_core_sources})
