if(NOT QUASAR_ENABLE_HIP)
  return()
endif()

set(_quasar_equilibrium_hip_sources
  gs_operator.hip
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
target_compile_options(quasar_equilibrium_hip
  PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)
#
# Verified, not assumed: flipping this to -ffp-contract=fast makes
# GsOperatorDevice.ApplyMatchesHostBitExactly fail with a max host-device
# difference of 8.8e-11 on the 65x65 case. That is comfortably inside any
# tolerance one would naturally reach for, which is precisely the problem.
