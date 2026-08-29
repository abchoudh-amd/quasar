if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_equilibrium_hip_sources
  gs_operator.hip
  gs_reduce.hip
  gs_greens.hip
  gs_source.hip
)

quasar_add_hip_module(equilibrium_hip SOURCES ${_quasar_equilibrium_hip_sources})
quasar_add_backend_src_include(equilibrium_hip)

# -ffp-contract=off is load-bearing, not a micro-optimization knob.
#
# The equilibrium port is a HARD REPLACE: the host implementation is deleted at
# the end of the port, so the only thing pinning each kernel to the behaviour it
# replaces is the host-vs-device equivalence test. Allowing the compiler to
# contract a*b+c into an FMA would make the device answer differ from the host
# answer in the last ulp, which turns that test from an EQUALITY into a
# tolerance -- and a tolerance cannot distinguish "correctly ported" from
# "slightly wrong in a way that happens to be small on this grid".
#
# The Pade line solve is where this bites hardest: it is a long serial
# elimination chain, so a single contracted operation early on propagates
# through every subsequent node of the line.
#
# gs_reduce.hip has an INDEPENDENT and harder requirement on the same flag: its
# two-sum error-free transformation is only error-free if each operation rounds
# separately. Contraction there does not shift the answer by an ulp, it silently
# turns the compensated sum back into a naive one. Do not relax this flag for
# one file without checking the other.
target_compile_options(quasar_equilibrium_hip
  PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)
#
# Verified, not assumed: flipping this to -ffp-contract=fast makes
# GsOperatorDevice.ApplyMatchesHostBitExactly fail with a max host-device
# difference of 8.8e-11 on the 65x65 case. That is comfortably inside any
# tolerance one would naturally reach for, which is precisely the problem.
