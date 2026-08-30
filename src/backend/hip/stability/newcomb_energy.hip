// Cylindrical Newcomb energy coefficients and deterministic weak-form assembly.
//
// The local Chebyshev arrays deliberately retain duplicate interface samples:
// each adjacent subdomain owns a quadrature endpoint and therefore contributes
// its own endpoint weight.  The xi_r degree of freedom is merged at a regular
// interface and split for the resonant harmonic.  A global matrix entry is
// assembled by exactly one thread, which walks domains in ascending order and
// needs no atomics.

#include "quasar/physics/stability/energy_functional.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace quasar::stability {
namespace {

constexpr unsigned kCoefficientBlock = 256;
constexpr unsigned kMatrixBlock = 16;

std::size_t checked_product(std::size_t a, std::size_t b,
                            const char* message) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::length_error{message};
  }
  return a * b;
}

struct BasisShape {
  int n_nodes{0};
  int n_domains{0};
  int order{0};
  int regular_n_free{0};
  std::size_t n_local{0};
};

struct RadialTopology {
  int n_radial{0};
  int n_free{0};
  std::vector<int> local_to_dof{};
};

void require_size(const backend::DeviceBuffer<Real>& buffer,
                  std::size_t expected, const char* name) {
  if (buffer.size() != expected) {
    throw std::invalid_argument{std::string{name} + " has size "
                                + std::to_string(buffer.size())
                                + "; expected " + std::to_string(expected)};
  }
}

template <class T>
void require_owner(const backend::DeviceBuffer<T>& buffer, int expected,
                   const char* name) {
  if (buffer.owner_device() != expected) {
    throw std::invalid_argument{std::string{name}
                                + " is allocated on a different device"};
  }
}

BasisShape validate_basis(const ChebyshevBasis& basis) {
  if (basis.order < 1
      || static_cast<long long>(basis.n_nodes)
             != static_cast<long long>(basis.order) + 1LL
      || basis.n_domains < 1) {
    throw std::invalid_argument{
        "cylindrical Newcomb operations require a non-empty Chebyshev basis "
        "with n_nodes == order + 1 and order >= 1"};
  }

  const std::size_t n_local = checked_product(
      static_cast<std::size_t>(basis.n_domains),
      static_cast<std::size_t>(basis.n_nodes),
      "cylindrical Newcomb local-node count overflows size_t");
  const std::size_t n_diff = checked_product(
      n_local, static_cast<std::size_t>(basis.n_nodes),
      "cylindrical Newcomb differentiation storage overflows size_t");
  require_size(basis.nodes, n_local, "ChebyshevBasis::nodes");
  require_size(basis.weights, n_local, "ChebyshevBasis::weights");
  require_size(basis.diff, n_diff, "ChebyshevBasis::diff");

  // There are n_domains*order radial intervals and one more global node.  The
  // two endpoint nodes are fixed, hence n_domains*order-1 free DOFs.  A
  // first-order single domain legitimately has zero free DOFs; coefficient
  // evaluation is still well-defined even though matrix assembly is not.
  const std::size_t n_intervals = checked_product(
      static_cast<std::size_t>(basis.n_domains),
      static_cast<std::size_t>(basis.order),
      "cylindrical Newcomb global-node count overflows size_t");
  if (n_intervals - 1
      > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error{
        "cylindrical Newcomb free-DOF count does not fit in int"};
  }

  const int owner = basis.nodes.owner_device();
  if (owner < 0) {
    throw std::invalid_argument{
        "cylindrical Newcomb Chebyshev basis is not device-resident"};
  }
  require_owner(basis.weights, owner, "ChebyshevBasis::weights");
  require_owner(basis.diff, owner, "ChebyshevBasis::diff");
  if (backend::current_device() != owner) {
    throw std::invalid_argument{
        "cylindrical Newcomb basis is not on the current device"};
  }

  return BasisShape{basis.n_nodes, basis.n_domains, basis.order,
                    static_cast<int>(n_intervals - 1), n_local};
}

RadialTopology build_radial_topology(const BasisShape& shape,
                                     const RadialDomains& domains, int m) {
  if (domains.n_domains != shape.n_domains
      || domains.n_domains > RadialDomains::kMaxDomains) {
    throw std::invalid_argument{
        "cylindrical Newcomb RadialDomains do not match Chebyshev basis"};
  }
  if (domains.overflow) {
    throw std::invalid_argument{
        "cylindrical Newcomb cannot use an overflowed radial topology"};
  }
  if (domains.resonance_count < 0
      || domains.resonance_count > RadialDomains::kMaxResonanceTags) {
    throw std::invalid_argument{
        "cylindrical Newcomb RadialDomains have an invalid resonance count"};
  }
  for (int breakpoint = 1; breakpoint <= domains.n_domains; ++breakpoint) {
    if (!std::isfinite(domains.breakpoints[breakpoint - 1])
        || !std::isfinite(domains.breakpoints[breakpoint])
        || !(domains.breakpoints[breakpoint]
             > domains.breakpoints[breakpoint - 1])) {
      throw std::invalid_argument{
          "cylindrical Newcomb radial breakpoints must be finite and ascending"};
    }
  }

  if (domains.resonance_offsets[0] != 0) {
    throw std::invalid_argument{
        "cylindrical Newcomb resonance offsets must begin at zero"};
  }
  int previous = 0;
  for (int breakpoint = 0; breakpoint <= domains.n_domains; ++breakpoint) {
    const int next = domains.resonance_offsets[breakpoint + 1];
    if (next < previous || next > domains.resonance_count) {
      throw std::invalid_argument{
          "cylindrical Newcomb RadialDomains have invalid resonance offsets"};
    }
    previous = next;
  }
  if (previous != domains.resonance_count) {
    throw std::invalid_argument{
        "cylindrical Newcomb resonance offsets do not cover every tag"};
  }

  std::vector<int> split(static_cast<std::size_t>(shape.n_domains + 1), 0);
  int split_count = 0;
  for (int interface = 1; interface < shape.n_domains; ++interface) {
    const int begin = domains.resonance_offsets[interface];
    const int end = domains.resonance_offsets[interface + 1];
    for (int tag = begin; tag < end; ++tag) {
      if (domains.resonant_m[tag] == m) {
        split[static_cast<std::size_t>(interface)] = 1;
        ++split_count;
        break;
      }
    }
  }

  const std::size_t regular_radial =
      static_cast<std::size_t>(shape.n_domains)
          * static_cast<std::size_t>(shape.order)
      + std::size_t{1};
  const std::size_t n_radial = regular_radial
                             + static_cast<std::size_t>(split_count);
  if (n_radial < 2
      || n_radial - 2
             > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error{
        "cylindrical Newcomb radial DOF count does not fit in int"};
  }

  RadialTopology topology;
  topology.n_radial = static_cast<int>(n_radial);
  topology.n_free = static_cast<int>(n_radial - 2);
  topology.local_to_dof.assign(shape.n_local, -1);

  int prior_splits = 0;
  for (int domain = 0; domain < shape.n_domains; ++domain) {
    if (domain > 0) {
      prior_splits += split[static_cast<std::size_t>(domain)];
    }
    for (int local = 0; local < shape.n_nodes; ++local) {
      const int radial = domain * shape.order + (shape.order - local)
                       + prior_splits;
      const std::size_t local_index =
          static_cast<std::size_t>(domain) * shape.n_nodes + local;
      if (radial != 0 && radial != topology.n_radial - 1) {
        topology.local_to_dof[local_index] = radial - 1;
      }
    }
  }
  return topology;
}

enum CoordinateStatus : int {
  kCoordinateValid = 0,
  kInvalidRadius = 1 << 0,
  kInvalidJacobian = 1 << 1,
};

__global__ void validate_coordinate_kernel(
    const Real* __restrict__ r, const Real* __restrict__ dr_dcoordinate,
    std::size_t n, int* __restrict__ status) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  // Validation is a setup-time O(n) pass over tens to hundreds of nodes.  A
  // single ordered scan is cheaper than allocating a reduction workspace and
  // keeps the entire stability path free of atomics.
  int result = kCoordinateValid;
  for (std::size_t i = 0; i < n; ++i) {
    if (!isfinite(r[i]) || !(r[i] > Real{0})) {
      result |= static_cast<int>(kInvalidRadius);
    }
    if (dr_dcoordinate != nullptr
        && (!isfinite(dr_dcoordinate[i])
            || !(dr_dcoordinate[i] > Real{0}))) {
      result |= static_cast<int>(kInvalidJacobian);
    }
  }
  *status = result;
}

void validate_coordinate_on_device(
    const backend::DeviceBuffer<Real>& r,
    const backend::DeviceBuffer<Real>* dr_dcoordinate, std::size_t n,
    stream_t stream) {
  backend::DeviceBuffer<int> device_status{1};
  hipLaunchKernelGGL(
      validate_coordinate_kernel, dim3{1}, dim3{1}, 0,
      static_cast<hipStream_t>(stream), r.device_ptr(),
      dr_dcoordinate == nullptr ? nullptr : dr_dcoordinate->device_ptr(), n,
      device_status.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  int status = kCoordinateValid;
  device_status.copy_to_host_async(&status, 1, stream);
  backend::device_synchronize(stream);
  if ((status & kInvalidRadius) != 0) {
    throw std::invalid_argument{
        "cylindrical Newcomb is restricted to finite annuli with r > 0"};
  }
  if ((status & kInvalidJacobian) != 0) {
    throw std::invalid_argument{
        "cylindrical Newcomb requires finite dr/dcoordinate > 0"};
  }
}

__global__ void evaluate_coefficients_kernel(
    const Real* __restrict__ r, const Real* __restrict__ b_theta,
    const Real* __restrict__ b_z, const Real* __restrict__ dp_dr,
    std::size_t n, int m, Real k, Real* __restrict__ f,
    Real* __restrict__ g) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                      + threadIdx.x;
  if (i >= n) return;

  const Real radius = r[i];
  const Real bt = b_theta[i];
  const Real bz = b_z[i];
  const Real pressure_gradient = dp_dr[i];
  const Real mr = static_cast<Real>(m);
  const Real k2 = k * k;
  const Real r2 = radius * radius;
  const Real m2 = mr * mr;
  const Real d = m2 + k2 * r2;
  const Real c_plus = k * radius * bz + mr * bt;
  const Real c_plus2 = c_plus * c_plus;
  const Real inv_mu0 = Real{1} / mu0;

  // Newcomb Eq. (16), with C_+ = k r B_z + m B_theta and SI conversion.
  f[i] = radius * c_plus2 * inv_mu0 / d;

  // Newcomb Eq. (18), in the physical xi_r / bare-dr convention.  The
  // pressure-gradient term is already in SI and therefore has no 1/mu0.
  const Real pressure = Real{2} * k2 * r2 / d * pressure_gradient;
  const Real field_line = c_plus2 * inv_mu0 / radius * (d - Real{1}) / d;
  const Real coupling = Real{2} * k2 * radius * inv_mu0 / (d * d)
                      * (k2 * r2 * bz * bz - m2 * bt * bt);
  g[i] = pressure + field_line + coupling;
}

__global__ void assemble_matrix_kernel(
    const Real* __restrict__ weights, const Real* __restrict__ diff,
    const Real* __restrict__ dr_dcoordinate, const Real* __restrict__ f,
    const Real* __restrict__ g, const int* __restrict__ local_to_dof,
    int n_nodes, int n_domains, int n_free, Real* __restrict__ matrix) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int column = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (row >= n_free || column >= n_free || row > column) return;

  Real sum = Real{0};

  for (int domain = 0; domain < n_domains; ++domain) {
    const std::size_t node_base =
        static_cast<std::size_t>(domain) * n_nodes;
    int local_row = -1;
    int local_column = -1;
    for (int local = 0; local < n_nodes; ++local) {
      const int dof = local_to_dof[node_base + local];
      if (dof == row) local_row = local;
      if (dof == column) local_column = local;
    }
    if (local_row < 0 || local_column < 0) continue;

    const std::size_t matrix_base = node_base * n_nodes;

    // Fixed loop order is part of determinism.  At a regular interface both
    // local copies map to the same DOF and their endpoint contributions are
    // accumulated in ascending domain order.  At a rational interface they
    // map to independent one-sided DOFs.
    for (int q = 0; q < n_nodes; ++q) {
      const std::size_t local_q = node_base + q;
      const std::size_t diff_q = matrix_base
                               + static_cast<std::size_t>(q) * n_nodes;
      const Real jacobian = dr_dcoordinate[local_q];
      sum += weights[local_q] * (f[local_q] / jacobian)
           * diff[diff_q + local_row] * diff[diff_q + local_column];
      if (q == local_row && local_row == local_column) {
        sum += weights[local_q] * (g[local_q] * jacobian);
      }
    }
  }

  // Column-major, with the lower entry mirrored from the exact same computed
  // scalar so symmetry is bit-for-bit rather than merely within tolerance.
  matrix[static_cast<std::size_t>(row)
         + static_cast<std::size_t>(column) * n_free] = sum;
  matrix[static_cast<std::size_t>(column)
         + static_cast<std::size_t>(row) * n_free] = sum;
}

}  // namespace

void CylindricalNewcombCoefficients::resize(std::size_t n_local_nodes) {
  f = backend::DeviceBuffer<Real>{n_local_nodes};
  g = backend::DeviceBuffer<Real>{n_local_nodes};
}

void CylindricalNewcombMatrix::resize(int n_xi_r_dofs_in) {
  if (n_xi_r_dofs_in < 0) {
    throw std::invalid_argument{
        "CylindricalNewcombMatrix DOF count must be non-negative"};
  }
  const std::size_t n = static_cast<std::size_t>(n_xi_r_dofs_in);
  const std::size_t entries = checked_product(
      n, n, "CylindricalNewcombMatrix storage overflows size_t");
  values = backend::DeviceBuffer<Real>{entries};
  local_to_dof = backend::DeviceBuffer<int>{};
  n_xi_r_dofs = n_xi_r_dofs_in;
}

int cylindrical_newcomb_fixed_boundary_dof_count(const ChebyshevBasis& basis) {
  return validate_basis(basis).regular_n_free;
}

int cylindrical_newcomb_fixed_boundary_dof_count(
    const ChebyshevBasis& basis, const RadialDomains& domains, int m) {
  const BasisShape shape = validate_basis(basis);
  return build_radial_topology(shape, domains, m).n_free;
}

void launch_evaluate_cylindrical_newcomb_coefficients(
    const ChebyshevBasis& basis, const backend::DeviceBuffer<Real>& r,
    const backend::DeviceBuffer<Real>& b_theta,
    const backend::DeviceBuffer<Real>& b_z,
    const backend::DeviceBuffer<Real>& dp_dr, CylindricalNewcombMode mode,
    CylindricalNewcombCoefficients& out, stream_t stream) {
  const BasisShape shape = validate_basis(basis);
  if (!std::isfinite(mode.k) || (mode.m == 0 && mode.k == Real{0})) {
    throw std::invalid_argument{
        "cylindrical Newcomb mode requires finite k and (m,k) != (0,0)"};
  }

  require_size(r, shape.n_local, "r");
  require_size(b_theta, shape.n_local, "B_theta");
  require_size(b_z, shape.n_local, "B_z");
  require_size(dp_dr, shape.n_local, "dp_dr");
  require_size(out.f, shape.n_local, "Newcomb f output");
  require_size(out.g, shape.n_local, "Newcomb g output");

  const int owner = basis.nodes.owner_device();
  require_owner(r, owner, "r");
  require_owner(b_theta, owner, "B_theta");
  require_owner(b_z, owner, "B_z");
  require_owner(dp_dr, owner, "dp_dr");
  require_owner(out.f, owner, "Newcomb f output");
  require_owner(out.g, owner, "Newcomb g output");

  validate_coordinate_on_device(r, nullptr, shape.n_local, stream);

  const unsigned blocks = static_cast<unsigned>(
      (shape.n_local + kCoefficientBlock - 1) / kCoefficientBlock);
  hipLaunchKernelGGL(
      evaluate_coefficients_kernel, dim3{blocks}, dim3{kCoefficientBlock}, 0,
      static_cast<hipStream_t>(stream), r.device_ptr(), b_theta.device_ptr(),
      b_z.device_ptr(), dp_dr.device_ptr(), shape.n_local, mode.m, mode.k,
      out.f.device_ptr(), out.g.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());
}

void launch_assemble_cylindrical_newcomb_matrix(
    const ChebyshevBasis& basis, const RadialDomains& domains, int m,
    const backend::DeviceBuffer<Real>& r,
    const backend::DeviceBuffer<Real>& dr_dcoordinate,
    const CylindricalNewcombCoefficients& coefficients,
    CylindricalNewcombMatrix& out, stream_t stream) {
  const BasisShape shape = validate_basis(basis);
  const RadialTopology topology = build_radial_topology(shape, domains, m);
  if (topology.n_free < 1) {
    throw std::invalid_argument{
        "cylindrical Newcomb fixed boundaries leave no free xi_r DOF"};
  }
  require_size(r, shape.n_local, "r");
  require_size(dr_dcoordinate, shape.n_local, "dr/dcoordinate");
  require_size(coefficients.f, shape.n_local, "Newcomb f coefficients");
  require_size(coefficients.g, shape.n_local, "Newcomb g coefficients");
  if (out.n_xi_r_dofs != topology.n_free) {
    throw std::invalid_argument{
        "CylindricalNewcombMatrix DOF count does not match harmonic topology"};
  }
  const std::size_t matrix_entries = checked_product(
      static_cast<std::size_t>(topology.n_free),
      static_cast<std::size_t>(topology.n_free),
      "cylindrical Newcomb matrix storage overflows size_t");
  require_size(out.values, matrix_entries, "CylindricalNewcombMatrix::values");

  const int owner = basis.nodes.owner_device();
  require_owner(r, owner, "r");
  require_owner(dr_dcoordinate, owner, "dr/dcoordinate");
  require_owner(coefficients.f, owner, "Newcomb f coefficients");
  require_owner(coefficients.g, owner, "Newcomb g coefficients");
  require_owner(out.values, owner, "CylindricalNewcombMatrix::values");

  validate_coordinate_on_device(r, &dr_dcoordinate, shape.n_local, stream);

  if (out.local_to_dof.size() != shape.n_local) {
    out.local_to_dof = backend::DeviceBuffer<int>{shape.n_local};
  }
  require_owner(out.local_to_dof, owner,
                "CylindricalNewcombMatrix::local_to_dof");
  out.local_to_dof.copy_from_host(topology.local_to_dof.data(),
                                  topology.local_to_dof.size());

  const dim3 block{kMatrixBlock, kMatrixBlock};
  const dim3 grid{
      (static_cast<unsigned>(topology.n_free) + block.x - 1) / block.x,
      (static_cast<unsigned>(topology.n_free) + block.y - 1) / block.y};
  hipLaunchKernelGGL(
      assemble_matrix_kernel, grid, block, 0,
      static_cast<hipStream_t>(stream), basis.weights.device_ptr(),
      basis.diff.device_ptr(), dr_dcoordinate.device_ptr(),
      coefficients.f.device_ptr(), coefficients.g.device_ptr(),
      out.local_to_dof.device_ptr(), shape.n_nodes, shape.n_domains,
      topology.n_free, out.values.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());
}

}  // namespace quasar::stability
