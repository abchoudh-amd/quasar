// The PIC kernel-launch ABI now lives in the installed public header
// include/quasar/backend/pic_kernels.hpp so callers reach the backend through a
// public abstraction instead of this private src/ path. This shim remains only
// for the backend .hip definitions under src/backend/hip/pic/, which include it
// by relative path; it forwards to the public declaration so a signature drift
// between a .hip definition and the ABI is still a compile error.
#pragma once

#include "quasar/backend/pic_kernels.hpp"
