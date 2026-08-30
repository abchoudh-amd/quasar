// Backend-neutral declaration of the analytic-field kernel-launch ABI.
//
// Mirrors physics/magnetostatics/kernels.hpp and physics/pic/kernels.hpp: a
// per-physics seam naming the launch entry points defined under
// src/backend/hip/analytic_fields/, included both by those .hip definitions (so
// a signature drift is a compile error) and by the host orchestrators in
// src/physics/analytic_fields/. Do not hand-redeclare these prototypes.
//
// Every launcher takes device-resident observation planes (px, py, pz) of
// length M and writes device-resident output planes of the same length. The
// evaluator parameters -- a dipole moment, a gradient tensor, a file-grid
// descriptor -- are small PODs passed by value, so no per-launch upload is
// needed for them.
//
// Status word. These kernels cannot throw, so each reports failure through an
// `int*` bit field written with integer atomics (exact and order-independent,
// hence launch-geometry independent). The bits are the contract shared with
// core::throw_on_evaluator_status:
//
//   bit 0  the quantity is singular at some observation point
//   bit 1  the quantity is finite mathematically but not representable
//   bit 2  an observation coordinate is not finite
//   bit 3  an observation point lies outside the configured grid
//
// The status buffer must be zero-initialized before the launch.
#pragma once

#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"

#include <cstddef>

using quasar_af_stream_t = ::quasar::backend::stream_t;

extern "C" {

// Row-major 3x3 gradient plus the reference point it is measured from.
struct QuasarAfGradientParams {
  ::quasar::Real b0[3];
  ::quasar::Real grad[9];
  ::quasar::Real origin[3];
};

struct QuasarAfDipoleParams {
  ::quasar::Real moment[3];
  ::quasar::Real origin[3];
  // mu0/(4*pi). Passed rather than redefined on the device so the host constant
  // in analytic_fields/dipole.hpp stays the single source of truth.
  ::quasar::Real mu0_over_4pi;
};

// Rectilinear nodal grid. The node values live in three device planes of
// length dims[0]*dims[1]*dims[2], indexed x-fastest:
// `i + dims[0] * (j + dims[1] * k)`.
struct QuasarAfFileGridParams {
  ::quasar::Real origin[3];
  ::quasar::Real spacing[3];
  int dims[3];
  // Per axis: 0 samples the interpolation weights, 1 samples their derivative.
  // A gradient column is one launch with a single axis set.
  int derivative_axis[3];
};

// Fills three planes with a constant vector. Serves both the uniform
// evaluator's B and its E.
void launch_analytic_uniform_fill(
    ::quasar::Real vx, ::quasar::Real vy, ::quasar::Real vz, int M,
    ::quasar::Real* ox, ::quasar::Real* oy, ::quasar::Real* oz,
    quasar_af_stream_t stream);

// B(p) = b0 + grad . (p - origin), each row summed in scaled form so a large
// cancelling displacement term does not round away the offset.
void launch_analytic_gradient_B(
    QuasarAfGradientParams params,
    const ::quasar::Real* px, const ::quasar::Real* py,
    const ::quasar::Real* pz, int M,
    ::quasar::Real* Bx, ::quasar::Real* By, ::quasar::Real* Bz,
    int* status, quasar_af_stream_t stream);

// Ideal point dipole. Singular at `origin` (status bit 0).
void launch_analytic_dipole_B(
    QuasarAfDipoleParams params,
    const ::quasar::Real* px, const ::quasar::Real* py,
    const ::quasar::Real* pz, int M,
    ::quasar::Real* Bx, ::quasar::Real* By, ::quasar::Real* Bz,
    int* status, quasar_af_stream_t stream);

// Dipole Jacobian, written component-major into a 9*M buffer:
// entry (i, j) of point p at `G[(3 * i + j) * M + p]`.
void launch_analytic_dipole_gradB(
    QuasarAfDipoleParams params,
    const ::quasar::Real* px, const ::quasar::Real* py,
    const ::quasar::Real* pz, int M,
    ::quasar::Real* G, int* status, quasar_af_stream_t stream);

// Trilinear sample of a rectilinear nodal map. With every derivative_axis zero
// this is the interpolated field; with exactly one set it is the corresponding
// column dB/dcoordinate, which is how the Jacobian is assembled (three launches
// writing into strided views of the same 9*M buffer).
void launch_analytic_file_grid_sample(
    QuasarAfFileGridParams params,
    const ::quasar::Real* vx, const ::quasar::Real* vy,
    const ::quasar::Real* vz,
    const ::quasar::Real* px, const ::quasar::Real* py,
    const ::quasar::Real* pz, int M,
    ::quasar::Real* ox, ::quasar::Real* oy, ::quasar::Real* oz,
    int* status, quasar_af_stream_t stream);

// One-time admissibility sweep over an uploaded map, run once at configure
// time. Two questions, both answered per cell vertex with no cross-thread
// reduction: are all node values finite, and is the componentwise trilinear
// interpolant solenoidal to within `relative_tolerance`?
//
// This reports through its own status bits rather than the shared ones, because
// the exceptions it raises name the map rather than an observation point:
//
//   bit 4  a node value is not finite
//   bit 5  the trilinear map is not solenoidal
//
// A boolean verdict OR-reduced with an integer atomic is exact and
// order-independent, so unlike a floating-point max this needs no deterministic
// tree -- there is nothing to round.
void launch_analytic_file_grid_validate(
    QuasarAfFileGridParams params,
    const ::quasar::Real* vx, const ::quasar::Real* vy,
    const ::quasar::Real* vz,
    ::quasar::Real relative_tolerance, int* status,
    quasar_af_stream_t stream);

}  // extern "C"
