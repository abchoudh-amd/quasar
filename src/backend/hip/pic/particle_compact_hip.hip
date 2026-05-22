#include "quasar/physics/pic/species.hpp"

#include <hip/hip_runtime.h>

extern "C" void launch_pic_particle_compact(quasar::pic::ParticleSpecies&, hipStream_t) {}
