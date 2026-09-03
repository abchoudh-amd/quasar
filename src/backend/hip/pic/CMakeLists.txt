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
  particle_migration_hip.hip
  particle_sampling_hip.hip
  halo_hip.hip
  filter_hip.hip
  fdtd_b_cyl_hip.hip
  fdtd_e_cyl_hip.hip
  gather_push_cyl_hip.hip
  deposit_cyl_hip.hip
  boundary_axis_hip.hip
  diagnostics_hip.hip
  external_field_hip.hip
  pic_seed_fields.hip
)

quasar_add_hip_module(pic_hip SOURCES ${_quasar_pic_hip_sources})
quasar_add_backend_src_include(pic_hip)

# -ffp-contract=off, scoped to the diagnostic reductions.
#
# diagnostics_hip.hip accumulates with the Knuth two-sum error-free
# transformation (backend/hip/reduction_detail.hpp) and a Kahan compensation in
# a scaled exponent frame (scaled_reduction_detail.hpp). Both identities hold
# only if each operation rounds separately. If the compiler contracts the
# sequence into an FMA, the intermediate is evaluated at higher precision, the
# recovered term stops being the exact residual, and the compensation quietly
# degrades into a naive sum. Nothing fails; the accuracy claim just becomes
# false -- and the host reference these kernels replaced has been deleted, so
# the equivalence tests against a long-double oracle are the only thing left
# holding them honest. See src/backend/hip/equilibrium/CMakeLists.txt for the
# same requirement and the measurement behind it.
#
# Applied per-source rather than to the whole module because the PIC hot-path
# kernels (FDTD, gather/push, deposit) legitimately want contraction.
#
# external_field_hip.hip needs the same treatment for the same reason: its
# continuous-solenoidality check reduces trace(grad B) through the exact
# expansions in numerics/scaled_arithmetic.hpp, and the whole point of that
# reduction is that an identically vanishing trace collapses to exactly zero
# rather than to one ulp of an intermediate. Contraction would put the ulp back.
set_source_files_properties(diagnostics_hip.hip external_field_hip.hip
  TARGET_DIRECTORY quasar_pic_hip
  PROPERTIES COMPILE_OPTIONS
    "$<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>")
