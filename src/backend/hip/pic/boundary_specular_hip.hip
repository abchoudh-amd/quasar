#include "quasar/core/grid.hpp"
#include "quasar/physics/pic/species.hpp"

#include "backend/hip/hip_check.hpp"
#include "backend/hip/pic/launch.hpp"
#include "backend/hip/pic/launch_helpers.hpp"

#include <hip/hip_runtime.h>

namespace {

__global__ void specular_kernel(quasar::Grid2D g, std::size_t n, int side,
                                double* x, double* y, double* x_prev, double* y_prev,
                                double* vx, double* vy, unsigned char* alive) {
  const std::size_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n || alive[p] == 0) return;
  // Reflect the post-move position and the normal velocity about the wall. The
  // pre-move position (x_prev) is deliberately LEFT inside the domain: the
  // charge-conserving deposit then forms current from the true in-domain start to
  // the just-crossed end, and the part of the deposit stencil that spills past the
  // wall lands in ghost cells, which boundary_specular_foldback reflects back into
  // the interior (image-charge current). Mirroring x_prev too (the old behaviour)
  // put the start point outside the domain and the deposit teleported current to
  // the opposite edge via the periodic wrap.
  if (side == 0 && x[p] < g.origin_x) {
    // FMA evaluates 2*wall-position with one final rounding. Materialising
    // 2*wall first can overflow for a large finite translated domain even when
    // the reflected coordinate is close to the wall and representable.
    x[p] = fma(2.0, g.origin_x, -x[p]);
    vx[p] = -vx[p];
  } else if (side == 1 && x[p] > g.origin_x + g.lx) {
    const double w = g.origin_x + g.lx;
    x[p] = fma(2.0, w, -x[p]);
    vx[p] = -vx[p];
  } else if (side == 2 && y[p] < g.origin_y) {
    y[p] = fma(2.0, g.origin_y, -y[p]);
    vy[p] = -vy[p];
  } else if (side == 3 && y[p] > g.origin_y + g.ly) {
    const double w = g.origin_y + g.ly;
    y[p] = fma(2.0, w, -y[p]);
    vy[p] = -vy[p];
  }
}

__device__ inline double radial_volume_ratio(const quasar::Grid2D& g,
                                             int ghost_i, int image_i,
                                             bool cylindrical) {
  if (!cylindrical) return 1.0;
  // The common 2*pi*dr*dz factors cancel exactly. Avoid forming either ring
  // volume: both may overflow on a large-radius thin annulus while their radius
  // ratio remains close to one and perfectly representable.
  const double ri = g.r_at_cell_center(image_i);
  return ri > 0.0 ? g.r_at_cell_center(ghost_i) / ri : 0.0;
}

// Fold centred/tangential current densities from ghost cells to their mirror.
// The normal current lives on faces and is handled separately below: its value
// on the wall is zero, while faces beyond the wall mirror antisymmetrically.
__global__ void foldback_kernel(quasar::Grid2D g, int side,
                                bool cylindrical,
                                double* jx, double* jy, double* jz) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int len = (side < 2) ? g.height() : g.pitch();
  const int total = g.nghost * len;
  if (tid >= total) return;
  const int gh = tid / len + 1;     // ghost layer, 1..nghost
  const int line = tid % len;

  int gi, gj, ii, ij;
  if (side < 2) {
    const int j = line - g.nghost;
    gi = side == 0 ? -gh : g.nx - 1 + gh;
    ii = side == 0 ? gh - 1 : g.nx - gh;
    gj = ij = j;
  } else {
    const int i = line - g.nghost;
    gj = side == 2 ? -gh : g.ny - 1 + gh;
    ij = side == 2 ? gh - 1 : g.ny - gh;
    gi = ii = i;
  }

  const std::size_t kg = g.index(gi, gj);
  const std::size_t ki = g.index(ii, ij);
  if (side < 2) {
    const double ratio = radial_volume_ratio(g, gi, ii, cylindrical);
    jy[ki] += ratio * jy[kg];
    jy[kg] = 0.0;
    if (!cylindrical) {
      jz[ki] += jz[kg];
      jz[kg] = 0.0;
    } else {
      // Jphi is radial-face centred. Its true ghosts mirror about the wall face
      // (i=0 or i=nx), and its density is normalized by a face dual annulus,
      // whose common 2*pi*dr*dz factors again reduce to a radius ratio. The high
      // physical face itself is not a ghost; with the uniform padded allocation
      // there are nghost-1 true high face ghosts versus nghost low ghosts.
      const bool has_face_ghost = side == 0 || gh < g.nghost;
      if (has_face_ghost) {
        const int src_face = side == 0 ? -gh : g.nx + gh;
        const int dst_face = side == 0 ? gh : g.nx - gh;
        const double rs = fabs(g.r_at_edge(src_face));
        const double rd = fabs(g.r_at_edge(dst_face));
        const auto ks = g.index(src_face, gj);
        const auto kd = g.index(dst_face, ij);
        if (rd > 0.0) jz[kd] += (rs / rd) * jz[ks];
        jz[ks] = 0.0;
      }
    }

    // Radial/Cartesian-x face current: odd parity about the wall face.
    if (side == 0) {
      const int src_face = -gh;
      const int dst_face = gh;
      const auto ks = g.index(src_face, gj);
      const auto kd = g.index(dst_face, ij);
      if (cylindrical) {
        const double rd = g.r_at_edge(dst_face);
        if (rd != 0.0) {
          jx[kd] -= (g.r_at_edge(src_face) / rd) * jx[ks];
        }
      } else {
        jx[kd] -= jx[ks];
      }
      jx[ks] = 0.0;
      if (gh == 1) jx[g.index(0, gj)] = 0.0;
    } else if (gh == 1) {
      jx[g.index(g.nx, gj)] = 0.0;
    } else {
      const int src_face = g.nx + gh - 1;
      const int dst_face = g.nx - gh + 1;
      const auto ks = g.index(src_face, gj);
      const auto kd = g.index(dst_face, ij);
      if (cylindrical) {
        const double rd = g.r_at_edge(dst_face);
        if (rd != 0.0) {
          jx[kd] -= (g.r_at_edge(src_face) / rd) * jx[ks];
        }
      } else {
        jx[kd] -= jx[ks];
      }
      jx[ks] = 0.0;
    }
  } else {
    // z/y reflection: radial metric is unchanged at a fixed i.
    jx[ki] += jx[kg];
    jz[ki] += jz[kg];
    jx[kg] = 0.0;
    jz[kg] = 0.0;

    if (side == 2) {
      const auto ks = g.index(gi, -gh);
      const auto kd = g.index(ii, gh);
      jy[kd] -= jy[ks];
      jy[ks] = 0.0;
      if (gh == 1) jy[g.index(gi, 0)] = 0.0;
    } else if (gh == 1) {
      jy[g.index(gi, g.ny)] = 0.0;
    } else {
      const int src_face = g.ny + gh - 1;
      const int dst_face = g.ny - gh + 1;
      const auto ks = g.index(gi, src_face);
      const auto kd = g.index(ii, dst_face);
      jy[kd] -= jy[ks];
      jy[ks] = 0.0;
    }
  }
}

__global__ void foldback_charge_kernel(quasar::Grid2D g, int side,
                                       bool cylindrical, double* rho) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int len = (side < 2) ? g.height() : g.pitch();
  const int total = g.nghost * len;
  if (tid >= total) return;
  const int gh = tid / len + 1;
  const int line = tid % len;
  int gi, gj, ii, ij;
  if (side < 2) {
    const int j = line - g.nghost;
    gi = side == 0 ? -gh : g.nx - 1 + gh;
    ii = side == 0 ? gh - 1 : g.nx - gh;
    gj = ij = j;
  } else {
    const int i = line - g.nghost;
    gj = side == 2 ? -gh : g.ny - 1 + gh;
    ij = side == 2 ? gh - 1 : g.ny - gh;
    gi = ii = i;
  }
  const auto kg = g.index(gi, gj);
  const auto ki = g.index(ii, ij);
  const double ratio = side < 2
      ? radial_volume_ratio(g, gi, ii, cylindrical) : 1.0;
  rho[ki] += ratio * rho[kg];
  rho[kg] = 0.0;
}

}  // namespace

extern "C" void launch_pic_boundary_specular_particles(const quasar::Grid2D& g,
                                                       quasar::pic::ParticleSpecies& s,
                                                       int side, quasar_stream_t stream) {
  if (s.size() == 0) return;
  const dim3 block(quasar::backend::pic::kLaunchBlock);
  const dim3 grid = quasar::backend::pic::grid_1d(s.size());
  hipLaunchKernelGGL(specular_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, s.size(), side, s.x(), s.y(), s.x_prev(), s.y_prev(),
                     s.vx(), s.vy(), s.alive());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_boundary_specular_foldback(const quasar::Grid2D& g,
                                                      quasar::JField2D<quasar::Real>& j,
                                                      int side, int cylindrical,
                                                      quasar_stream_t stream) {
  const int len = (side < 2) ? g.height() : g.pitch();
  const int total = g.nghost * len;
  if (total <= 0) return;
  const dim3 block(quasar::backend::pic::kLaunchBlock);
  const dim3 grid = quasar::backend::pic::grid_1d(static_cast<std::size_t>(total));
  hipLaunchKernelGGL(foldback_kernel, grid, block, 0, static_cast<hipStream_t>(stream),
                     g, side, cylindrical != 0, j.jx.device_ptr(),
                     j.jy.device_ptr(), j.jz.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}

extern "C" void launch_pic_boundary_specular_foldback_charge(
    const quasar::Grid2D& g, quasar::ScalarGrid2D<quasar::Real>& rho,
    int side, int cylindrical, quasar_stream_t stream) {
  const int len = (side < 2) ? g.height() : g.pitch();
  const int total = g.nghost * len;
  if (total <= 0) return;
  const dim3 block(quasar::backend::pic::kLaunchBlock);
  const dim3 grid = quasar::backend::pic::grid_1d(
      static_cast<std::size_t>(total));
  hipLaunchKernelGGL(foldback_charge_kernel, grid, block, 0,
                     static_cast<hipStream_t>(stream), g, side,
                     cylindrical != 0, rho.values.device_ptr());
  QUASAR_HIP_CHECK(::hipGetLastError());
}
