#include "quasar/core/grid.hpp"
#include "quasar/core/yee_field.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

// Ghost closures for non-periodic Yee boundaries.
//
// A PEC condition is imposed at the physical wall, not at the outermost
// cell-centred sample.  Consequently tangential cell-centred E and normal
// cell-centred B are continued oddly across the wall; their interpolation at
// the wall is zero, while the first physical sample remains an evolved degree
// of freedom.  Normal E and tangential B use even continuation.  A component
// that actually lives on the wall is pinned only when its parity is odd.
//
// Outflow uses linear continuation.  It supplies every stencil halo before the
// curls; the characteristic (Mur) update of tangential E is applied separately
// after Ampere.  Linear continuation makes the order-two boundary derivative
// equal the one-sided interior derivative and exactly differentiates a linear
// outgoing profile at either order.
//
// Every side kernel spans the complete padded transverse axis.  This is
// intentional: Bz(nx,ny) is a physical corner and order-four high-face Ampere
// also reads the halo beyond it.  Filling only nx or ny entries left mixed-BC
// corner values uninitialised.

namespace {

enum class FillKind : int { pec, outflow };

__device__ inline double linear_low(const quasar::Grid2D& g, const double* f,
                                    const auto& index, int distance, int n) {
  const double wall = f[index(0)];
  if (n <= 1) return wall;
  return (static_cast<double>(distance) + 1.0) * wall
       - static_cast<double>(distance) * f[index(1)];
}

__device__ inline double linear_high_center(const quasar::Grid2D& g,
                                            const double* f, const auto& index,
                                            int distance, int n) {
  const double wall = f[index(n - 1)];
  if (n <= 1) return wall;
  return (static_cast<double>(distance) + 1.0) * wall
       - static_cast<double>(distance) * f[index(n - 2)];
}

__device__ inline double linear_high_face(const quasar::Grid2D& g,
                                          const double* f, const auto& index,
                                          int distance, int n) {
  const double wall = f[index(n)];
  if (n <= 0) return wall;
  return (static_cast<double>(distance) + 1.0) * wall
       - static_cast<double>(distance) * f[index(n - 1)];
}

// `face_normal` describes staggering along the boundary-normal axis.  For a
// centred component there are n physical samples 0..n-1; for a face component
// there are n+1 physical samples 0..n.
__device__ inline void fill_component(const quasar::Grid2D& g, double* f,
                                      int side, int line, bool face_normal,
                                      double parity, FillKind kind) {
  const bool xside = side < 2;
  const bool low = side == 0 || side == 2;
  const int n = xside ? g.nx : g.ny;
  const auto index = [&](int normal) {
    return xside ? g.index(normal, line) : g.index(line, normal);
  };

  if (low) {
    if (kind == FillKind::pec && face_normal && parity < 0.0) {
      f[index(0)] = 0.0;
    }
    for (int gh = 1; gh <= g.nghost; ++gh) {
      if (kind == FillKind::outflow) {
        f[index(-gh)] = linear_low(g, f, index, gh, n);
      } else {
        const int src = face_normal ? gh : gh - 1;
        f[index(-gh)] = parity * f[index(src)];
      }
    }
    return;
  }

  if (!face_normal) {
    for (int gh = 1; gh <= g.nghost; ++gh) {
      if (kind == FillKind::outflow) {
        f[index(n - 1 + gh)] = linear_high_center(g, f, index, gh, n);
      } else {
        f[index(n - 1 + gh)] = parity * f[index(n - gh)];
      }
    }
    return;
  }

  // index n is the physical high face, not a ghost.  Even PEC and outflow leave
  // it untouched; odd PEC pins it at the wall.  Only n+1.. are continuation.
  if (kind == FillKind::pec && parity < 0.0) f[index(n)] = 0.0;
  for (int q = 1; q < g.nghost; ++q) {
    if (kind == FillKind::outflow) {
      f[index(n + q)] = linear_high_face(g, f, index, q, n);
    } else {
      f[index(n + q)] = parity * f[index(n - q)];
    }
  }
}

__global__ void boundary_fields_kernel(quasar::Grid2D g, int side,
                                       int cylindrical, FillKind kind,
                                       double* ex, double* ey, double* ez,
                                       double* bx, double* by, double* bz) {
  const int transverse_storage = side < 2 ? g.height() : g.pitch();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= transverse_storage) return;
  const int line = tid - g.nghost;

  if (kind == FillKind::outflow) {
    if (cylindrical != 0 && side < 2) {
      fill_component(g, ex, side, line, true,  +1.0, kind);
      fill_component(g, ey, side, line, false, +1.0, kind);
      fill_component(g, ez, side, line, true,  +1.0, kind);
      fill_component(g, bx, side, line, true,  +1.0, kind);
      fill_component(g, by, side, line, false, +1.0, kind);
      fill_component(g, bz, side, line, true,  +1.0, kind);
    } else if (cylindrical != 0) {
      fill_component(g, ex, side, line, false, +1.0, kind);
      fill_component(g, ey, side, line, true,  +1.0, kind);
      fill_component(g, ez, side, line, false, +1.0, kind);
      fill_component(g, bx, side, line, true,  +1.0, kind);
      fill_component(g, by, side, line, false, +1.0, kind);
      fill_component(g, bz, side, line, true,  +1.0, kind);
    } else if (side < 2) {
      fill_component(g, ex, side, line, true,  +1.0, kind);
      fill_component(g, ey, side, line, false, +1.0, kind);
      fill_component(g, ez, side, line, false, +1.0, kind);
      fill_component(g, bx, side, line, false, +1.0, kind);
      fill_component(g, by, side, line, true,  +1.0, kind);
      fill_component(g, bz, side, line, true,  +1.0, kind);
    } else {
      fill_component(g, ex, side, line, false, +1.0, kind);
      fill_component(g, ey, side, line, true,  +1.0, kind);
      fill_component(g, ez, side, line, false, +1.0, kind);
      fill_component(g, bx, side, line, true,  +1.0, kind);
      fill_component(g, by, side, line, false, +1.0, kind);
      fill_component(g, bz, side, line, true,  +1.0, kind);
    }
    return;
  }

  if (cylindrical != 0 && side < 2) {
    // Cylindrical radial parity: Er/Ephi/Br/Bphi live on radial faces;
    // Ez/Bz live at radial centres.
    fill_component(g, ex, side, line, true,  +1.0, kind);  // normal E_r
    fill_component(g, ey, side, line, false, -1.0, kind);  // tangential E_z
    fill_component(g, ez, side, line, true,  -1.0, kind);  // tangential E_phi
    fill_component(g, bx, side, line, true,  -1.0, kind);  // normal B_r
    fill_component(g, by, side, line, false, +1.0, kind);  // tangential B_z
    fill_component(g, bz, side, line, true,  +1.0, kind);  // tangential B_phi
  } else if (cylindrical != 0) {
    fill_component(g, ex, side, line, false, -1.0, kind);
    fill_component(g, ey, side, line, true,  +1.0, kind);
    fill_component(g, ez, side, line, false, -1.0, kind);
    fill_component(g, bx, side, line, true,  +1.0, kind);
    fill_component(g, by, side, line, false, -1.0, kind);
    fill_component(g, bz, side, line, true,  +1.0, kind);
  } else if (side < 2) {
    fill_component(g, ex, side, line, true,  +1.0, kind);  // normal E_x
    fill_component(g, ey, side, line, false, -1.0, kind);  // tangential E_y
    fill_component(g, ez, side, line, false, -1.0, kind);  // tangential E_z
    fill_component(g, bx, side, line, false, -1.0, kind);  // normal B_x
    fill_component(g, by, side, line, true,  +1.0, kind);  // tangential B_y
    fill_component(g, bz, side, line, true,  +1.0, kind);  // tangential B_z
  } else {
    fill_component(g, ex, side, line, false, -1.0, kind);
    fill_component(g, ey, side, line, true,  +1.0, kind);
    fill_component(g, ez, side, line, false, -1.0, kind);
    fill_component(g, bx, side, line, true,  +1.0, kind);
    fill_component(g, by, side, line, false, -1.0, kind);
    fill_component(g, bz, side, line, true,  +1.0, kind);
  }
}

void launch_boundary_fields(const quasar::Grid2D& g,
                            quasar::YeeField2D<quasar::Real>& f, int side,
                            int cylindrical, FillKind kind,
                            quasar_stream_t stream) {
  const int n = side < 2 ? g.height() : g.pitch();
  if (n <= 0) return;
  const dim3 block(quasar::backend::pic::kLaunchBlock);
  const dim3 grid = quasar::backend::pic::grid_1d(static_cast<std::size_t>(n));
  hipLaunchKernelGGL(boundary_fields_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream), g, side, cylindrical,
                     kind, f.ex.device_ptr(), f.ey.device_ptr(),
                     f.ez.device_ptr(), f.bx.device_ptr(), f.by.device_ptr(),
                     f.bz.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

}  // namespace

extern "C" void launch_pic_boundary_pec_fields(
    const quasar::Grid2D& g, quasar::YeeField2D<quasar::Real>& f, int side,
    int cylindrical, quasar_stream_t stream) {
  launch_boundary_fields(g, f, side, cylindrical, FillKind::pec, stream);
}

extern "C" void launch_pic_boundary_outflow_fill_fields(
    const quasar::Grid2D& g, quasar::YeeField2D<quasar::Real>& f, int side,
    int cylindrical, quasar_stream_t stream) {
  launch_boundary_fields(g, f, side, cylindrical, FillKind::outflow, stream);
}

// Backward-compatible internal spellings retained for the existing boundary
// dispatch seam.  New callers should pass the geometry explicitly above.
extern "C" void launch_pic_boundary_cyl_pec_fields(
    const quasar::Grid2D& g, quasar::YeeField2D<quasar::Real>& f, int side,
    quasar_stream_t stream) {
  launch_boundary_fields(g, f, side, 1, FillKind::pec, stream);
}

extern "C" void launch_pic_boundary_cyl_outflow_fields(
    const quasar::Grid2D& g, quasar::YeeField2D<quasar::Real>& f, int side,
    quasar_stream_t stream) {
  launch_boundary_fields(g, f, side, 1, FillKind::outflow, stream);
}
