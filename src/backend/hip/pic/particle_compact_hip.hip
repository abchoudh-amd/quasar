#include "quasar/physics/pic/species.hpp"

#include "backend/hip/pic/launch.hpp"

#include <hip/hip_runtime.h>

extern "C" void launch_pic_particle_compact(quasar::pic::ParticleSpecies&, hipStream_t) {}
