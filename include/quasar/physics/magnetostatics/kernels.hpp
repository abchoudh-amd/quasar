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
    double* Bx, double* By, double* Bz, int* status,
    quasar_ms_stream_t stream);

void launch_biot_savart_B_f32(
    const float* ax, const float* ay, const float* az,
    const float* bx, const float* by, const float* bz,
    const float* I_, int N,
    const float* px, const float* py, const float* pz, int M,
    float* Bx, float* By, float* Bz, int* status,
    quasar_ms_stream_t stream);

// Vector potential A (B = curl A). Defined in
// src/backend/hip/magnetostatics/biot_savart_hip.hip alongside the B launchers;
// the output SoA Ax/Ay/Az mirrors the B entry points.
void launch_biot_savart_A_f64(
    const double* ax, const double* ay, const double* az,
    const double* bx, const double* by, const double* bz,
    const double* I_, int N,
    const double* px, const double* py, const double* pz, int M,
    double* Ax, double* Ay, double* Az, int* status,
    quasar_ms_stream_t stream);

void launch_biot_savart_A_f32(
    const float* ax, const float* ay, const float* az,
    const float* bx, const float* by, const float* bz,
    const float* I_, int N,
    const float* px, const float* py, const float* pz, int M,
    float* Ax, float* Ay, float* Az, int* status,
    quasar_ms_stream_t stream);

// Defined in src/backend/hip/magnetostatics/biot_savart_grad_hip.hip.
void launch_biot_savart_gradB_f64(
    const double* ax, const double* ay, const double* az,
    const double* bx, const double* by, const double* bz,
    const double* I_, int N,
    const double* px, const double* py, const double* pz, int M,
    double* G, int* status,
    quasar_ms_stream_t stream);

void launch_biot_savart_gradB_f32(
    const float* ax, const float* ay, const float* az,
    const float* bx, const float* by, const float* bz,
    const float* I_, int N,
    const float* px, const float* py, const float* pz, int M,
    float* G, int* status,
    quasar_ms_stream_t stream);

// -- Filament geometry generation --------------------------------------------
//
// Defined in src/backend/hip/magnetostatics/geometry_hip.hip. The generators
// used to walk their vertices on the host, calling sin/cos and two
// resolvability checks per point; that is the last per-point host arithmetic in
// the magnetostatics slice, so it runs here and writes device SoA planes
// directly. What stays on the host is scalar: the orthonormal basis built once
// from the axis vector, and the handful of corner points a racetrack needs.
//
// The resolvability checks are the reason these kernels are not one line each.
// A displacement requested at a large centre can lose an entire local dimension
// to rounding while the surviving coordinates still form non-zero segments, so
// each vertex verifies that every material component of its local offset
// survives both the scaling and the translation with at least one useful binary
// digit. The host threw immediately with the offending dimension named; the
// kernels OR a bit per dimension into a status word instead.
//
// Status bits, shared by the generator kernels:
//   1 = a radial displacement is not resolvable at the requested centre
//   2 = an axial displacement is not resolvable at the requested centre
//   4 = a generated coordinate overflowed

// Orthonormal frame: the unit axis plus two unit vectors spanning its normal
// plane. Built on the host from one axis vector, which is scalar configuration
// math.
struct QuasarMsBasis {
  double axis[3];
  double u[3];
  double v[3];
};

// Circular arc vertices. Vertex k lies at
//   centre + radius * (cos(theta_k) * u + sin(theta_k) * v),
//   theta_k = theta_bias + theta_scale * index_k / theta_divisor,
// where index_k = k + index_offset, reduced modulo index_modulus when that is
// nonzero. Reducing the integer index before forming the angle -- rather than
// accumulating an increment -- is what keeps a many-turn helix free of
// argument-reduction drift, so the modulus is part of the contract.
struct QuasarMsArcParams {
  double centre[3];
  double u[3];
  double v[3];
  double radius;
  double theta_bias;
  double theta_scale;
  int index_offset;
  int index_modulus;
  int theta_divisor;
  // Components below this fraction of the unit direction are immaterial and are
  // not required to survive; this admits the expected sin(pi) residue.
  double material_component;
};

void launch_ms_arc_points(
    QuasarMsArcParams params, int count,
    double* px, double* py, double* pz, int* status,
    quasar_ms_stream_t stream);

// Helix vertices: the arc above, but about a centre that advances along the
// axis with the vertex index.
struct QuasarMsHelixParams {
  double centre[3];
  double axis[3];
  double u[3];
  double v[3];
  double radius;
  double half_length;
  long long n_total;
  int segments_per_turn;
  double material_component;
};

void launch_ms_helix_points(
    QuasarMsHelixParams params, int count,
    double* px, double* py, double* pz, int* status,
    quasar_ms_stream_t stream);

// Rejects a consecutive vertex pair that coincides exactly. Status bit 1.
void launch_ms_validate_segments(
    const double* px, const double* py, const double* pz, int n_points,
    int* status, quasar_ms_stream_t stream);

// Flattens one filament's vertices into the shared per-segment planes at
// `offset`. Status bits: 1 = a non-finite coordinate; 2 = an endpoint
// displacement overflows; 4 = a zero-length segment.
void launch_ms_flatten_filament(
    const double* px, const double* py, const double* pz, int n_points,
    double current, long long offset,
    double* ax, double* ay, double* az,
    double* bx, double* by, double* bz, double* I_,
    int* status, quasar_ms_stream_t stream);

// fp32 narrowing for BiotSavartEvaluatorF. A common double-precision origin is
// subtracted before the cast so a rigid translation is invisible to the
// narrowing rather than collapsing a short segment into one float coordinate.
// Status bits: 1 = not representable after fp32 origin shifting;
//              2 = a segment collapses after fp32 narrowing.
void launch_ms_narrow_segments(
    const double* ax, const double* ay, const double* az,
    const double* bx, const double* by, const double* bz,
    const double* I_, int n,
    double ox, double oy, double oz,
    float* fax, float* fay, float* faz,
    float* fbx, float* fby, float* fbz, float* fI,
    int* status, quasar_ms_stream_t stream);

void launch_ms_narrow_points(
    const double* px, const double* py, const double* pz, int n,
    double ox, double oy, double oz,
    float* fpx, float* fpy, float* fpz,
    int* status, quasar_ms_stream_t stream);

}  // extern "C"
