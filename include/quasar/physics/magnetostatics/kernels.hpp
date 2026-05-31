// Backend-neutral declaration of the magnetostatics kernel-launch ABI.
//
// This is a per-physics seam (it names the magnetostatics launch entry points),
// so it lives under physics/magnetostatics/ rather than backend/ — the backend
// axis stays physics-neutral (device.hpp / memory.hpp only), mirroring
// physics/pic/kernels.hpp. Each launch_biot_savart_* entry point defined under
// src/backend/hip/magnetostatics/ is declared here exactly once and included
// both by its .hip definition (so a signature drift is a compile error) and by
// the host orchestrator (biot_savart_evaluator.cpp). Do not hand-redeclare these
// extern "C" prototypes anywhere else.
#pragma once

#include "quasar/backend/device.hpp"

// The launch ABI speaks the backend-neutral stream handle so callers never
// include a HIP header. The .hip definitions cast it back internally.
using quasar_ms_stream_t = ::quasar::backend::stream_t;

extern "C" {

// Defined in src/backend/hip/magnetostatics/biot_savart_hip.hip.
void launch_biot_savart_B_f64(
    const double* ax, const double* ay, const double* az,
    const double* bx, const double* by, const double* bz,
    const double* I_, int N,
    const double* px, const double* py, const double* pz, int M,
    double* Bx, double* By, double* Bz,
    quasar_ms_stream_t stream);

void launch_biot_savart_B_f32(
    const float* ax, const float* ay, const float* az,
    const float* bx, const float* by, const float* bz,
    const float* I_, int N,
    const float* px, const float* py, const float* pz, int M,
    float* Bx, float* By, float* Bz,
    quasar_ms_stream_t stream);

// Defined in src/backend/hip/magnetostatics/biot_savart_grad_hip.hip.
void launch_biot_savart_gradB_f64(
    const double* ax, const double* ay, const double* az,
    const double* bx, const double* by, const double* bz,
    const double* I_, int N,
    const double* px, const double* py, const double* pz, int M,
    double* G,
    quasar_ms_stream_t stream);

void launch_biot_savart_gradB_f32(
    const float* ax, const float* ay, const float* az,
    const float* bx, const float* by, const float* bz,
    const float* I_, int N,
    const float* px, const float* py, const float* pz, int M,
    float* G,
    quasar_ms_stream_t stream);

}  // extern "C"
