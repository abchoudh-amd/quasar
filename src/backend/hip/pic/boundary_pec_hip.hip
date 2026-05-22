#include "quasar/core/yee_field.hpp"

#include <hip/hip_runtime.h>

extern "C" void launch_pic_boundary_pec_fields(const quasar::Grid2D&,
                                               quasar::YeeField2D<double>&,
                                               int, hipStream_t) {}
