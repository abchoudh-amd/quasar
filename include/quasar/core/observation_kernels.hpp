// Backend-neutral declaration of the observation-set kernel-launch ABI.
//
// The structured observation sets (ObservationGrid, PlaneSlice, LineProbe) are
// descriptions, not data: each expands to a point per lattice site through a
// short affine formula. Expanding them on the host and uploading the result was
// the last per-point host arithmetic left in front of the device-resident field
// evaluators, so the expansion happens on the device and writes a
// core::DevicePointCloud directly.
//
// This is the core-axis analogue of physics/<module>/kernels.hpp: declared once
// here, included by the .hip definitions under src/backend/hip/core/ and by the
// host methods in src/core/observations.cpp.
//
// Status word, shared by all three: bit 0 means a generated coordinate
// overflowed. Set with an integer atomic, so it is exact and independent of the
// launch geometry. The buffer must be zeroed before the launch.
#pragma once

#include "quasar/backend/device.hpp"
#include "quasar/core/types.hpp"

using quasar_obs_stream_t = ::quasar::backend::stream_t;

extern "C" {

// Site (i, j, k) at origin + (i*spacing.x, j*spacing.y, k*spacing.z).
// Flat layout is x-fastest: i + dims[0] * (j + dims[1] * k).
struct QuasarObsGridParams {
  ::quasar::Real origin[3];
  ::quasar::Real spacing[3];
  int dims[3];
};

// Site (i, j) at origin + i*u_step + j*v_step, u-fastest: i + nu * j.
struct QuasarObsPlaneParams {
  ::quasar::Real origin[3];
  ::quasar::Real u_step[3];
  ::quasar::Real v_step[3];
  int nu;
  int nv;
};

// n_points samples linearly interpolated from start to end, inclusive at both
// ends. Requires n_points >= 2, which the host validates.
struct QuasarObsLineParams {
  ::quasar::Real start[3];
  ::quasar::Real end[3];
  int n_points;
};

void launch_observation_grid_points(
    QuasarObsGridParams params, ::quasar::Real* px, ::quasar::Real* py,
    ::quasar::Real* pz, int* status, quasar_obs_stream_t stream);

void launch_observation_plane_points(
    QuasarObsPlaneParams params, ::quasar::Real* px, ::quasar::Real* py,
    ::quasar::Real* pz, int* status, quasar_obs_stream_t stream);

void launch_observation_line_points(
    QuasarObsLineParams params, ::quasar::Real* px, ::quasar::Real* py,
    ::quasar::Real* pz, int* status, quasar_obs_stream_t stream);

}  // extern "C"
