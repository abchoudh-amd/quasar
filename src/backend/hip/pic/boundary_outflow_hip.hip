#include "quasar/core/grid.hpp"
#include "quasar/core/yee_field.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

#include <cmath>

// First-order characteristic (Mur) outflow for the component-specific Yee
// lattices.  The ghost continuation used by the curls lives in
// boundary_pec_hip.hip; this file advances the two tangential electric
// components after Ampere.
//
// A radial cylindrical wave has asymptotic amplitude u~r^{-1/2}.  Applying Mur
// to w=sqrt(r)u rather than u includes that geometric spreading while retaining
// the usual local one-way characteristic.  Axial and Cartesian faces use unit
// scaling.  This is the metric-consistent first-order absorbing condition; it is
// not a claim of exact transparency for oblique or dispersive waves.

namespace {

__device__ inline double mur_coefficient(double dt, double spacing) {
  // (dt-h)/(dt+h), evaluated without overflowing dt+h. The ratio form also
  // remains well-defined when either positive input is subnormal.
  if (dt <= spacing) {
    const double ratio = dt / spacing;
    return (ratio - 1.0) / (ratio + 1.0);
  }
  const double ratio = spacing / dt;
  return (1.0 - ratio) / (1.0 + ratio);
}

__device__ inline void mur_advance(double* comp, double* old_wall,
                                   double* old_adjacent, std::size_t kw,
                                   std::size_t ka, double coeff, bool init,
                                   double wall_scale, double adjacent_scale) {
  const double new_adjacent = adjacent_scale * comp[ka];
  if (init) {
    *old_wall = wall_scale * comp[kw];
    *old_adjacent = new_adjacent;
    return;
  }
  const double new_wall = *old_adjacent + coeff * (new_adjacent - *old_wall);
  comp[kw] = new_wall / wall_scale;
  *old_wall = new_wall;
  *old_adjacent = new_adjacent;
}

__device__ inline double radial_scale(const quasar::Grid2D& g, int i,
                                      bool face) {
  const double r = face ? g.r_at_edge(i) : g.r_at_cell_center(i);
  return sqrt(r);
}

__global__ void outflow_mur_x_kernel(quasar::Grid2D g, int side, double dt,
                                     int cylindrical, int stride,
                                     double* ey, double* ez, double* mur,
                                     int init, int skip_lo, int skip_hi) {
  const int line = blockIdx.x * blockDim.x + threadIdx.x;
  if (line >= stride) return;
  const bool low = side == 0;
  const double coeff = mur_coefficient(dt, g.dx());
  const std::size_t line_offset = static_cast<std::size_t>(line);
  const std::size_t strip = static_cast<std::size_t>(stride);

  // Ey is x-centred and y-face-located in both geometries.  It is tangential to
  // an x/r face and includes the two physical axial high/low face samples.
  if (line <= g.ny) {
    const int iw = low ? 0 : g.nx - 1;
    const int ia = low ? 1 : g.nx - 2;
    double sw = 1.0, sa = 1.0;
    if (cylindrical != 0) {
      sw = radial_scale(g, iw, false);
      sa = radial_scale(g, ia, false);
    }
    mur_advance(ey, mur + line_offset, mur + strip + line_offset,
                g.index(iw, line), g.index(ia, line), coeff, init != 0,
                sw, sa);
  }

  // Cartesian Ez is x-centred; cylindrical Ephi is radial-face-located and its
  // high wall is the real i=nx face.  At an outflow/outflow corner the diagonal
  // kernel owns this doubly-tangential component, so neither face updates it.
  if (line < g.ny) {
    const bool corner = (line == 0 && skip_lo != 0)
                     || (line == g.ny - 1 && skip_hi != 0);
    if (!corner) {
      const bool radial_face = cylindrical != 0;
      const int iw = radial_face ? (low ? 0 : g.nx)
                                 : (low ? 0 : g.nx - 1);
      const int ia = radial_face ? (low ? 1 : g.nx - 1)
                                 : (low ? 1 : g.nx - 2);
      double sw = 1.0, sa = 1.0;
      if (radial_face) {
        sw = radial_scale(g, iw, true);
        sa = radial_scale(g, ia, true);
      }
      mur_advance(ez, mur + 2u * strip + line_offset,
                  mur + 3u * strip + line_offset,
                  g.index(iw, line), g.index(ia, line), coeff, init != 0,
                  sw, sa);
    }
  }
}

__global__ void outflow_mur_y_kernel(quasar::Grid2D g, int side, double dt,
                                     int cylindrical, int stride,
                                     double* ex, double* ez, double* mur,
                                     int init, int skip_lo, int skip_hi) {
  const int line = blockIdx.x * blockDim.x + threadIdx.x;
  if (line >= stride) return;
  const bool low = side == 2;
  const int jw = low ? 0 : g.ny - 1;
  const int ja = low ? 1 : g.ny - 2;
  const double coeff = mur_coefficient(dt, g.dy());
  const std::size_t line_offset = static_cast<std::size_t>(line);
  const std::size_t strip = static_cast<std::size_t>(stride);

  // Ex/Er is x/r-face-located and tangential to a y/z face.
  if (line <= g.nx) {
    mur_advance(ex, mur + line_offset, mur + strip + line_offset,
                g.index(line, jw), g.index(line, ja), coeff, init != 0,
                1.0, 1.0);
  }

  // Cartesian Ez is x-centred; cylindrical Ephi is radial-face-located.
  const int ez_last = cylindrical != 0 ? g.nx : g.nx - 1;
  if (line <= ez_last) {
    const bool corner = (line == 0 && skip_lo != 0)
                     || (line == ez_last && skip_hi != 0);
    if (!corner) {
      mur_advance(ez, mur + 2u * strip + line_offset,
                  mur + 3u * strip + line_offset,
                  g.index(line, jw), g.index(line, ja), coeff, init != 0,
                  1.0, 1.0);
    }
  }
}

// Diagonal characteristic for the scalar component tangential to both faces.
// Bits enumerate (xlo,ylo), (xhi,ylo), (xlo,yhi), (xhi,yhi).
__global__ void outflow_corner_kernel(quasar::Grid2D g, double dt,
                                      unsigned int mask, int cylindrical,
                                      double* ez, double* history, int init) {
  const int c = threadIdx.x;
  if (c >= 4 || (mask & (1u << c)) == 0u) return;
  const bool hi_x = (c & 1) != 0;
  const bool hi_y = (c & 2) != 0;
  const bool radial_face = cylindrical != 0;
  const int iw = radial_face ? (hi_x ? g.nx : 0)
                             : (hi_x ? g.nx - 1 : 0);
  const int ia = radial_face ? (hi_x ? g.nx - 1 : 1)
                             : (hi_x ? g.nx - 2 : 1);
  const int jw = hi_y ? g.ny - 1 : 0;
  const int ja = hi_y ? g.ny - 2 : 1;
  const std::size_t kw = g.index(iw, jw);
  const std::size_t ka = g.index(ia, ja);
  double sw = 1.0, sa = 1.0;
  if (radial_face) {
    sw = radial_scale(g, iw, true);
    sa = radial_scale(g, ia, true);
  }
  const double new_adjacent = sa * ez[ka];
  if (init != 0) {
    history[c] = sw * ez[kw];
    history[4 + c] = new_adjacent;
    return;
  }
  // Normalize the diagonal distance before hypot: the true hypot may exceed
  // DBL_MAX even though its ratio with dt, and hence the Mur coefficient, is
  // perfectly representable.
  const double distance_scale = fmax(g.dx(), g.dy());
  const double distance_scaled = hypot(g.dx() / distance_scale,
                                       g.dy() / distance_scale);
  const double coeff = mur_coefficient(dt / distance_scale, distance_scaled);
  const double new_corner = history[4 + c]
                          + coeff * (new_adjacent - history[c]);
  ez[kw] = new_corner / sw;
  history[c] = new_corner;
  history[4 + c] = new_adjacent;
}

}  // namespace

extern "C" void launch_pic_boundary_outflow_correct_e(
    const quasar::Grid2D& g, quasar::YeeField2D<quasar::Real>& f, int side,
    quasar::Real dt, quasar::Real* mur, int stride, int init, int skip_lo,
    int skip_hi, int cylindrical, quasar_stream_t stream) {
  if (stride <= 0) return;
  const dim3 block(quasar::backend::pic::kLaunchBlock);
  const dim3 grid = quasar::backend::pic::grid_1d(
      static_cast<std::size_t>(stride));
  const auto hip_stream = static_cast<hipStream_t>(stream);
  if (side < 2) {
    hipLaunchKernelGGL(outflow_mur_x_kernel, grid, block, 0, hip_stream, g,
                       side, dt, cylindrical, stride, f.ey.device_ptr(),
                       f.ez.device_ptr(), mur, init, skip_lo, skip_hi);
  } else {
    hipLaunchKernelGGL(outflow_mur_y_kernel, grid, block, 0, hip_stream, g,
                       side, dt, cylindrical, stride, f.ex.device_ptr(),
                       f.ez.device_ptr(), mur, init, skip_lo, skip_hi);
  }
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_boundary_outflow_corners(
    const quasar::Grid2D& g, quasar::YeeField2D<quasar::Real>& f,
    quasar::Real dt, unsigned int corner_mask, quasar::Real* history,
    int init, int cylindrical, quasar_stream_t stream) {
  if (corner_mask == 0u) return;
  hipLaunchKernelGGL(outflow_corner_kernel, dim3(1), dim3(4), 0,
                     static_cast<hipStream_t>(stream), g, dt, corner_mask,
                     cylindrical, f.ez.device_ptr(), history, init);
  QUASAR_HIP_CHECK(::hipGetLastError());
}
