#include "quasar/physics/pic/diagnostics.hpp"

#include "quasar/physics/pic/kernels.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace quasar::pic {

namespace {

// Snapshot one device field component into a host vector sized to the grid's
// padded storage. Diagnostics are host-side reductions; the per-step hot path
// does not call these.
std::vector<Real> component_to_host(const backend::DeviceBuffer<Real>& buf) {
  std::vector<Real> h(buf.size());
  if (!h.empty()) {
    buf.copy_to_host(h.data(), h.size());
  }
  return h;
}

}  // namespace

std::size_t alive_count(const ParticleSpecies& species) {
  // Device-side reduction over the alive flags — avoids copying all seven
  // particle arrays to the host just to count survivors.
  return ::launch_pic_alive_count(species, nullptr);
}

Real total_kinetic_energy(const ParticleSpecies& species) {
  const auto snap = species.to_host();
  const Real m = species.mass();
  Real ke = Real{0};
  for (std::size_t p = 0; p < snap.x.size(); ++p) {
    if (snap.alive[p] == 0) continue;
    const Real v2 = snap.vx[p] * snap.vx[p] + snap.vy[p] * snap.vy[p] +
                    snap.vz[p] * snap.vz[p];
    ke += Real{0.5} * m * v2 * snap.weight[p];
  }
  return ke;
}

Real total_em_energy(const YeeField2D<Real>& fields, const Grid2D& grid,
                     bool cylindrical) {
  // Normalized natural units (c = eps0 = mu0 = 1): u = 0.5*(E^2 + B^2),
  // integrated over the interior cell volume. Components are colocated at first
  // order for this scalar energy budget, which is the convention the magnitude
  // tests pin. Cartesian uses the flat area dA = dx*dy; cylindrical (r,z) uses
  // the axisymmetric ring volume cell_volume(i) = 2*pi*r*dr*dz, the same metric
  // the cylindrical deposit normalizes by, so the field energy is physical.
  // The field may be default-constructed (empty buffers) while the grid still
  // reports a nominal 1x1 extent; never index past the actual storage.
  if (fields.ex.size() < grid.storage_size()) return Real{0};

  const auto ex = component_to_host(fields.ex);
  const auto ey = component_to_host(fields.ey);
  const auto ez = component_to_host(fields.ez);
  const auto bx = component_to_host(fields.bx);
  const auto by = component_to_host(fields.by);
  const auto bz = component_to_host(fields.bz);

  const Real dA = grid.dx() * grid.dy();
  Real energy = Real{0};
  for (int j = 0; j < grid.ny; ++j) {
    for (int i = 0; i < grid.nx; ++i) {
      const std::size_t k = grid.index(i, j);
      const Real e2 = ex[k] * ex[k] + ey[k] * ey[k] + ez[k] * ez[k];
      const Real b2 = bx[k] * bx[k] + by[k] * by[k] + bz[k] * bz[k];
      const Real dV = cylindrical ? grid.cell_volume(i) : dA;
      energy += Real{0.5} * (e2 + b2) * dV;
    }
  }
  return energy;
}

Real gauss_residual(const YeeField2D<Real>& fields, const JField2D<Real>& current,
                    bool cylindrical) {
  // Gauss's law residual ‖∇·E − ρ‖_2 over the interior. The solver does not
  // carry a standalone charge-density field (ρ = Σ q·w·S(x) lives on the
  // species, not here), so this reports the vacuum residual ‖∇·E‖_2 — the
  // quantity that must stay near zero away from sources and that drifts if the
  // field update violates the discrete divergence constraint.
  (void)current;
  const auto& grid = fields.grid;
  if (fields.ex.size() < grid.storage_size()) return Real{0};
  const auto ex = component_to_host(fields.ex);
  const auto ey = component_to_host(fields.ey);

  const Real inv_dx = Real{1} / grid.dx();
  const Real inv_dy = Real{1} / grid.dy();
  Real sum_sq = Real{0};
  for (int j = 0; j < grid.ny; ++j) {
    for (int i = 0; i < grid.nx; ++i) {
      Real div;
      if (cylindrical) {
        // Axisymmetric divergence (1/r) d(r E_r)/dr + d E_z/dz, using the same
        // backward ring-flux radial operator the deposit's continuity is built
        // on (E_r = ex maps to the radial face component, E_z = ey to axial).
        // At i=0 the inner face radius r_at_edge(0)=0 zeroes the inside-axis
        // term, the natural on-axis closure, so no neighbour past r=0 is read.
        Real radial = Real{0};
        if (i > 0) {
          radial = (grid.r_at_edge(i) * ex[grid.index(i, j)] -
                    grid.r_at_edge(i - 1) * ex[grid.index(i - 1, j)]) /
                   (grid.r_at_cell_center(i) * grid.dx());
        }
        // z stays a plain (periodic-or-wall) axis; periodic_index wraps it
        // legitimately and never crosses the radial axis.
        const Real dEzdz =
            (ey[grid.index(i, j)] - ey[grid.periodic_index(i, j - 1)]) * inv_dy;
        div = radial + dEzdz;
      } else {
        // Cartesian Yee divergence at the cell corner: backward differences of
        // the face-centred E components, wrapped periodically for neighbours.
        const Real dExdx =
            (ex[grid.index(i, j)] - ex[grid.periodic_index(i - 1, j)]) * inv_dx;
        const Real dEydy =
            (ey[grid.index(i, j)] - ey[grid.periodic_index(i, j - 1)]) * inv_dy;
        div = dExdx + dEydy;
      }
      sum_sq += div * div;
    }
  }
  return std::sqrt(sum_sq);
}

}  // namespace quasar::pic
