if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_core_hip_sources
  observations_hip.hip
)

quasar_add_hip_module(core_hip SOURCES ${_quasar_core_hip_sources})
quasar_add_backend_src_include(core_hip)

# -ffp-contract=off. device_lerp reproduces the C++20 std::lerp algorithm so a
# device-expanded LineProbe agrees with LineProbe::point_at bit for bit; that
# guarantee depends on each operation rounding separately, and contraction would
# break it silently -- the last point would drift off `end` by an ulp with
# nothing failing.
target_compile_options(quasar_core_hip
  PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)
