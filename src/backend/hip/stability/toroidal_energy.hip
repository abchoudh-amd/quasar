#include "quasar/physics/stability/toroidal_energy.hpp"

#include "backend/hip/hip_check.hpp"
#include "quasar/numerics/cholesky.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace quasar::stability {
namespace {

constexpr unsigned kBlock1D = 128;
constexpr unsigned kBlock2D = 8;
constexpr int kMaximumOccurrences = 2;

// Block size for the dense-matrix validation reduction.  Must be a power of two
// so the shared-memory tree reduction is exact and complete.
constexpr unsigned kBlockReduce = 256;
static_assert((kBlockReduce & (kBlockReduce - 1)) == 0,
              "kBlockReduce must be a power of two");

// Cap on the grid of the validation scan.  The reduction is min/max/or only, so
// the result is independent of how many blocks the work is spread over; this
// simply bounds the size of the partial-result buffer.
constexpr unsigned kMaximumScanBlocks = 1024;

struct DeviceDiagnostics {
  std::uint64_t status;
  Real minimum_jacobian;
  Real minimum_density;
  Real maximum_field_pitch_deviation;
  Real maximum_flux_scale_deviation;
  Real maximum_pest_deviation;
  Real minimum_mass_diagonal;
  Real hermitian_residual;
  std::size_t first_invalid_metric_point;
  int nonfinite_inertia;
};

struct ComplexValue {
  Real real;
  Real imag;
};

__host__ __device__ ComplexValue operator+(ComplexValue a, ComplexValue b) {
  return {a.real + b.real, a.imag + b.imag};
}

__host__ __device__ ComplexValue operator-(ComplexValue a, ComplexValue b) {
  return {a.real - b.real, a.imag - b.imag};
}

__host__ __device__ ComplexValue operator*(ComplexValue a, ComplexValue b) {
  return {a.real * b.real - a.imag * b.imag,
          a.real * b.imag + a.imag * b.real};
}

__host__ __device__ ComplexValue operator*(Real a, ComplexValue b) {
  return {a * b.real, a * b.imag};
}

__host__ __device__ ComplexValue operator*(ComplexValue a, Real b) {
  return b * a;
}

__host__ __device__ ComplexValue operator/(ComplexValue a, Real b) {
  return {a.real / b, a.imag / b};
}

__host__ __device__ ComplexValue& operator+=(ComplexValue& a,
                                              ComplexValue b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

__host__ __device__ ComplexValue conjugate(ComplexValue a) {
  return {a.real, -a.imag};
}

__host__ __device__ ComplexValue multiply_i(ComplexValue a) {
  return {-a.imag, a.real};
}

__host__ __device__ ComplexValue conjugate_multiply(ComplexValue a,
                                                     ComplexValue b) {
  return conjugate(a) * b;
}

struct BasisFields {
  ComplexValue xi_lambda{};
  ComplexValue xi_theta{};
  ComplexValue xi_phi{};
  ComplexValue q_lambda{};
  ComplexValue q_theta{};
  ComplexValue q_phi{};
  ComplexValue divergence{};
};

struct BilinearValues {
  ComplexValue stiffness{};
  ComplexValue inertia{};
};

std::size_t checked_multiply(std::size_t a, std::size_t b,
                             const char* message) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::length_error{message};
  }
  return a * b;
}

std::size_t checked_add(std::size_t a, std::size_t b,
                        const char* message) {
  if (b > std::numeric_limits<std::size_t>::max() - a) {
    throw std::length_error{message};
  }
  return a + b;
}

void require_size(const backend::DeviceBuffer<Real>& buffer,
                  std::size_t expected, const char* name) {
  if (buffer.size() != expected) {
    throw std::invalid_argument{std::string{name} + " has the wrong size"};
  }
}

void require_size(const backend::DeviceBuffer<int>& buffer,
                  std::size_t expected, const char* name) {
  if (buffer.size() != expected) {
    throw std::invalid_argument{std::string{name} + " has the wrong size"};
  }
}

template <class T>
void require_owner(const backend::DeviceBuffer<T>& buffer, int owner,
                   const char* name) {
  if (!buffer.empty() && buffer.owner_device() != owner) {
    throw std::invalid_argument{std::string{name}
                                + " resides on a different device"};
  }
}

struct ProblemShape {
  int n_domains;
  int n_nodes;
  int n_lambda;
  int n_theta;
  int owner;
  std::size_t geometry_points;
};

ProblemShape validate_shape(const ChebyshevBasis& basis,
                            const RadialDomains& domains,
                            const FluxCoordinateGrid& coords,
                            const ToroidalEquilibriumFields& equilibrium,
                            const SpectralDofLayout& layout) {
  if (basis.order <= 0
      || static_cast<long long>(basis.n_nodes)
             != static_cast<long long>(basis.order) + 1LL
      || basis.n_domains <= 0 || domains.n_domains != basis.n_domains) {
    throw std::invalid_argument{
        "toroidal assembly requires a valid Chebyshev/domain layout"};
  }
  if (layout.n_domains() != basis.n_domains
      || layout.chebyshev_order() != basis.order
      || layout.n_theta() != coords.n_theta) {
    throw std::invalid_argument{
        "toroidal assembly layout does not match the spectral grid"};
  }
  const std::size_t radial = checked_multiply(
      static_cast<std::size_t>(basis.n_domains),
      static_cast<std::size_t>(basis.n_nodes),
      "toroidal assembly radial size overflows");
  if (radial > static_cast<std::size_t>(std::numeric_limits<int>::max())
      || coords.n_psi != static_cast<int>(radial) || coords.n_theta <= 0
      || equilibrium.n_lambda != coords.n_psi
      || equilibrium.n_theta != coords.n_theta) {
    throw std::invalid_argument{
        "toroidal assembly coordinate/equilibrium shape mismatch"};
  }
  const std::size_t geometry = checked_multiply(
      radial, static_cast<std::size_t>(coords.n_theta),
      "toroidal assembly geometry size overflows");

  require_size(basis.nodes, radial, "ChebyshevBasis::nodes");
  require_size(basis.weights, radial, "ChebyshevBasis::weights");
  require_size(basis.diff,
               checked_multiply(radial,
                                static_cast<std::size_t>(basis.n_nodes),
                                "Chebyshev differentiation size overflows"),
               "ChebyshevBasis::diff");
  require_size(coords.psi_n, radial, "FluxCoordinateGrid::psi_n");
  require_size(coords.q, radial, "FluxCoordinateGrid::q");
  require_size(coords.valid, radial, "FluxCoordinateGrid::valid");
  require_size(coords.r, geometry, "FluxCoordinateGrid::r");

#define QUASAR_REQUIRE_RADIAL(member)                                          \
  require_size(equilibrium.member, radial,                                    \
               "ToroidalEquilibriumFields::" #member)
#define QUASAR_REQUIRE_GEOMETRY(member)                                        \
  require_size(equilibrium.member, geometry,                                  \
               "ToroidalEquilibriumFields::" #member)
  QUASAR_REQUIRE_RADIAL(pressure);
  QUASAR_REQUIRE_RADIAL(pressure_lambda);
  QUASAR_REQUIRE_RADIAL(density);
  QUASAR_REQUIRE_GEOMETRY(g_lambda_lambda);
  QUASAR_REQUIRE_GEOMETRY(g_lambda_theta);
  QUASAR_REQUIRE_GEOMETRY(g_theta_theta);
  QUASAR_REQUIRE_GEOMETRY(g_phi_phi);
  QUASAR_REQUIRE_GEOMETRY(jacobian);
  QUASAR_REQUIRE_GEOMETRY(b_theta);
  QUASAR_REQUIRE_GEOMETRY(b_phi);
  QUASAR_REQUIRE_GEOMETRY(j_theta);
  QUASAR_REQUIRE_GEOMETRY(j_phi);
#undef QUASAR_REQUIRE_GEOMETRY
#undef QUASAR_REQUIRE_RADIAL

  const int owner = basis.nodes.owner_device();
  require_owner(basis.weights, owner, "ChebyshevBasis::weights");
  require_owner(basis.diff, owner, "ChebyshevBasis::diff");
  require_owner(coords.psi_n, owner, "FluxCoordinateGrid::psi_n");
  require_owner(coords.q, owner, "FluxCoordinateGrid::q");
  require_owner(coords.valid, owner, "FluxCoordinateGrid::valid");
  require_owner(coords.r, owner, "FluxCoordinateGrid::r");

#define QUASAR_REQUIRE_OWNER(member)                                           \
  require_owner(equilibrium.member, owner,                                    \
                "ToroidalEquilibriumFields::" #member)
  QUASAR_REQUIRE_OWNER(pressure);
  QUASAR_REQUIRE_OWNER(pressure_lambda);
  QUASAR_REQUIRE_OWNER(density);
  QUASAR_REQUIRE_OWNER(g_lambda_lambda);
  QUASAR_REQUIRE_OWNER(g_lambda_theta);
  QUASAR_REQUIRE_OWNER(g_theta_theta);
  QUASAR_REQUIRE_OWNER(g_phi_phi);
  QUASAR_REQUIRE_OWNER(jacobian);
  QUASAR_REQUIRE_OWNER(b_theta);
  QUASAR_REQUIRE_OWNER(b_phi);
  QUASAR_REQUIRE_OWNER(j_theta);
  QUASAR_REQUIRE_OWNER(j_phi);
#undef QUASAR_REQUIRE_OWNER

  return {basis.n_domains, basis.n_nodes, static_cast<int>(radial),
          coords.n_theta, owner, geometry};
}

ToroidalAssemblyStatus validate_topology(
    const RadialDomains& domains, const SpectralDofLayout& layout) noexcept {
  if (domains.overflow || domains.n_domains <= 0
      || domains.n_domains > RadialDomains::kMaxDomains
      || domains.resonance_count < 0
      || domains.resonance_count > RadialDomains::kMaxResonanceTags
      || domains.resonance_offsets[0] != 0) {
    return ToroidalAssemblyStatus::invalid_topology;
  }

  for (int breakpoint = 0; breakpoint <= domains.n_domains; ++breakpoint) {
    if (!std::isfinite(domains.breakpoints[breakpoint])) {
      return ToroidalAssemblyStatus::invalid_topology;
    }
    if (breakpoint != 0
        && !(domains.breakpoints[breakpoint]
             > domains.breakpoints[breakpoint - 1])) {
      return ToroidalAssemblyStatus::invalid_topology;
    }
  }

  int previous_offset = 0;
  for (int breakpoint = 0; breakpoint <= domains.n_domains; ++breakpoint) {
    const int next_offset = domains.resonance_offsets[breakpoint + 1];
    if (next_offset < previous_offset
        || next_offset > domains.resonance_count) {
      return ToroidalAssemblyStatus::invalid_topology;
    }
    previous_offset = next_offset;
  }
  if (previous_offset != domains.resonance_count) {
    return ToroidalAssemblyStatus::invalid_topology;
  }
  for (int tag = 0; tag < domains.resonance_count; ++tag) {
    if (!std::isfinite(domains.resonant_psi_n[tag])) {
      return ToroidalAssemblyStatus::invalid_topology;
    }
  }

  const std::size_t domain_count =
      static_cast<std::size_t>(domains.n_domains);
  const std::size_t order =
      static_cast<std::size_t>(layout.chebyshev_order());
  if (order != 0
      && domain_count
             > (std::numeric_limits<std::size_t>::max() - std::size_t{1})
                   / order) {
    return ToroidalAssemblyStatus::invalid_topology;
  }
  const std::size_t regular_radial = domain_count * order + std::size_t{1};
  std::size_t total_radial = 0;

  try {
    for (int m = -layout.m_max(); m <= layout.m_max(); ++m) {
      std::vector<unsigned char> split(
          static_cast<std::size_t>(domains.n_domains + 1), 0);
      std::size_t split_count = 0;
      for (int interface = 1; interface < domains.n_domains; ++interface) {
        const int begin = domains.resonance_offsets[interface];
        const int end = domains.resonance_offsets[interface + 1];
        for (int tag = begin; tag < end; ++tag) {
          if (domains.resonant_m[tag] == m) {
            split[static_cast<std::size_t>(interface)] = 1;
            ++split_count;
            break;
          }
        }
        if (layout.interface_is_split(interface, m)
            != (split[static_cast<std::size_t>(interface)] != 0)) {
          return ToroidalAssemblyStatus::invalid_topology;
        }
      }

      if (split_count
          > std::numeric_limits<std::size_t>::max() - regular_radial) {
        return ToroidalAssemblyStatus::invalid_topology;
      }
      const std::size_t expected_radial = regular_radial + split_count;
      if (layout.n_radial(m) != expected_radial
          || expected_radial
                 > std::numeric_limits<std::size_t>::max() - total_radial) {
        return ToroidalAssemblyStatus::invalid_topology;
      }
      total_radial += expected_radial;

      std::size_t prior_splits = 0;
      for (int domain = 0; domain < domains.n_domains; ++domain) {
        if (domain > 0) {
          prior_splits += split[static_cast<std::size_t>(domain)];
        }
        for (int local = 0; local <= layout.chebyshev_order(); ++local) {
          const std::size_t expected =
              static_cast<std::size_t>(domain) * order
              + static_cast<std::size_t>(layout.chebyshev_order() - local)
              + prior_splits;
          if (layout.global_radial(domain, local, m) != expected) {
            return ToroidalAssemblyStatus::invalid_topology;
          }
        }
      }
    }
  } catch (const std::exception&) {
    return ToroidalAssemblyStatus::invalid_topology;
  }

  if (total_radial
      > std::numeric_limits<std::size_t>::max()
            / static_cast<std::size_t>(SpectralDofLayout::kComponentCount)
      || layout.complex_dof_count()
             != total_radial
                    * static_cast<std::size_t>(
                        SpectralDofLayout::kComponentCount)) {
    return ToroidalAssemblyStatus::invalid_topology;
  }
  return ToroidalAssemblyStatus::success;
}

struct HostOccurrences {
  std::vector<int> count;
  std::vector<int> domain;
  std::vector<int> local_node;
  std::vector<int> harmonic;
  std::vector<int> component;
};

HostOccurrences build_occurrences(const SpectralDofLayout& layout,
                                  const FixedBoundaryDofMap& boundary_map) {
  const std::size_t count = boundary_map.complex_dof_count();
  HostOccurrences result;
  result.count.assign(count, 0);
  result.domain.assign(kMaximumOccurrences * count, -1);
  result.local_node.assign(kMaximumOccurrences * count, -1);
  result.harmonic.assign(count, 0);
  result.component.assign(count, -1);

  for (int m = -layout.m_max(); m <= layout.m_max(); ++m) {
    for (int domain = 0; domain < layout.n_domains(); ++domain) {
      for (int local = 0; local <= layout.chebyshev_order(); ++local) {
        const std::size_t radial = layout.global_radial(domain, local, m);
        for (int component = 0;
             component < SpectralDofLayout::kComponentCount; ++component) {
          const auto displacement =
              static_cast<DisplacementComponent>(component);
          const std::size_t full =
              layout.complex_dof(radial, m, displacement);
          const std::size_t free = boundary_map.free_complex_dof(full);
          if (free == FixedBoundaryDofMap::eliminated) continue;

          const int occurrence = result.count[free]++;
          if (occurrence >= kMaximumOccurrences) {
            throw std::logic_error{
                "a spectral DOF has more than two local occurrences"};
          }
          const std::size_t slot =
              kMaximumOccurrences * free + static_cast<std::size_t>(occurrence);
          result.domain[slot] = domain;
          result.local_node[slot] = local;
          if (result.component[free] < 0) {
            result.harmonic[free] = m;
            result.component[free] = component;
          } else if (result.harmonic[free] != m
                     || result.component[free] != component) {
            throw std::logic_error{
                "spectral occurrence map aliases distinct basis functions"};
          }
        }
      }
    }
  }

  for (std::size_t i = 0; i < count; ++i) {
    if (result.count[i] <= 0 || result.count[i] > kMaximumOccurrences
        || result.component[i] < 0) {
      throw std::logic_error{"incomplete spectral occurrence map"};
    }
  }
  return result;
}

__device__ Real relative_deviation(Real value, Real reference) {
  if (!isfinite(value) || !isfinite(reference)) {
    return std::numeric_limits<Real>::infinity();
  }
  const Real scale = fmax(fabs(value), fabs(reference));
  if (scale == Real{0}) return Real{0};
  return fabs(value - reference) / scale;
}

__device__ Real complex_magnitude(ComplexValue value) {
  return hypot(value.real, value.imag);
}

__device__ Real relative_deviation(ComplexValue value,
                                   ComplexValue reference) {
  const Real value_magnitude = complex_magnitude(value);
  const Real reference_magnitude = complex_magnitude(reference);
  if (!isfinite(value_magnitude) || !isfinite(reference_magnitude)) {
    return std::numeric_limits<Real>::infinity();
  }
  const Real scale = fmax(value_magnitude, reference_magnitude);
  if (scale == Real{0}) return Real{0};
  return complex_magnitude(value - reference) / scale;
}

__device__ Real fourier_derivative_coefficient(int row, int column,
                                               int count) {
  if (row == column || count <= 1) return Real{0};
  const int difference = row - column;
  const Real sign = (abs(difference) & 1) == 0 ? Real{1} : Real{-1};
  const Real angle = pi * static_cast<Real>(difference)
                   / static_cast<Real>(count);
  if ((count & 1) == 0) {
    return Real{0.5} * sign / tan(angle);
  }
  return Real{0.5} * sign / sin(angle);
}

__global__ void build_derivatives_kernel(
    const Real* __restrict__ q, const Real* __restrict__ jacobian,
    const Real* __restrict__ differentiation, int n_domains, int n_nodes,
    int n_theta, Real* __restrict__ q_lambda,
    Real* __restrict__ jacobian_lambda,
    Real* __restrict__ jacobian_theta) {
  const int theta = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int radial = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  const int n_lambda = n_domains * n_nodes;
  if (radial >= n_lambda || theta >= n_theta) return;

  const int domain = radial / n_nodes;
  const int row = radial - domain * n_nodes;
  const std::size_t dbase = static_cast<std::size_t>(domain) * n_nodes * n_nodes
                          + static_cast<std::size_t>(row) * n_nodes;
  const std::size_t radial_base =
      static_cast<std::size_t>(domain) * n_nodes;

  Real dq = Real{0};
  Real dj_lambda = Real{0};
  for (int column = 0; column < n_nodes; ++column) {
    const Real coefficient = differentiation[dbase + column];
    const std::size_t source_radial = radial_base + column;
    dq += coefficient * q[source_radial];
    dj_lambda +=
        coefficient * jacobian[source_radial * n_theta + theta];
  }
  if (theta == 0) q_lambda[radial] = dq;

  Real dj_stored_theta = Real{0};
  const std::size_t geometry_base = static_cast<std::size_t>(radial) * n_theta;
  for (int column = 0; column < n_theta; ++column) {
    dj_stored_theta +=
        fourier_derivative_coefficient(theta, column, n_theta)
        * jacobian[geometry_base + column];
  }
  const std::size_t geometry = geometry_base + theta;
  jacobian_lambda[geometry] = dj_lambda;
  // theta_source = -theta_stored.
  jacobian_theta[geometry] = -dj_stored_theta;
}

__global__ void validate_inputs_kernel(
    const Real* __restrict__ basis_nodes,
    const Real* __restrict__ coordinate_nodes,
    const Real* __restrict__ breakpoints,
    const Real* __restrict__ q, const int* __restrict__ valid,
    const Real* __restrict__ pressure,
    const Real* __restrict__ pressure_lambda,
    const Real* __restrict__ density,
    const Real* __restrict__ g_lambda_lambda,
    const Real* __restrict__ g_lambda_theta,
    const Real* __restrict__ g_theta_theta,
    const Real* __restrict__ g_phi_phi,
    const Real* __restrict__ jacobian,
    const Real* __restrict__ b_theta,
    const Real* __restrict__ b_phi,
    const Real* __restrict__ j_theta,
    const Real* __restrict__ j_phi,
    const Real* __restrict__ q_lambda,
    const Real* __restrict__ jacobian_lambda,
    const Real* __restrict__ jacobian_theta,
    int n_domains, int n_nodes, int n_lambda, int n_theta,
    Real signed_flux_scale,
    Real field_pitch_tolerance, Real flux_scale_tolerance,
    Real pest_tolerance, DeviceDiagnostics* __restrict__ out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  DeviceDiagnostics diagnostics{};
  diagnostics.minimum_jacobian = std::numeric_limits<Real>::infinity();
  diagnostics.minimum_density = std::numeric_limits<Real>::infinity();
  diagnostics.first_invalid_metric_point =
      std::numeric_limits<std::size_t>::max();

  for (int radial = 0; radial < n_lambda; ++radial) {
    const int domain = radial / n_nodes;
    const int local = radial - domain * n_nodes;
    bool endpoint_mismatch = false;
    if (domain < 0 || domain >= n_domains) {
      endpoint_mismatch = true;
    } else if (local == 0) {
      endpoint_mismatch = basis_nodes[radial] != breakpoints[domain + 1];
    } else if (local == n_nodes - 1) {
      endpoint_mismatch = basis_nodes[radial] != breakpoints[domain];
    }
    if (valid[radial] != 1 || basis_nodes[radial] != coordinate_nodes[radial]
        || endpoint_mismatch) {
      diagnostics.status |= static_cast<std::uint64_t>(
          ToroidalAssemblyStatus::invalid_topology);
    }
    if (!isfinite(coordinate_nodes[radial]) || !isfinite(q[radial])
        || !isfinite(pressure[radial])
        || !isfinite(pressure_lambda[radial])
        || !isfinite(density[radial]) || !isfinite(q_lambda[radial])) {
      diagnostics.status |= static_cast<std::uint64_t>(
          ToroidalAssemblyStatus::nonfinite_input);
    }
    diagnostics.minimum_density =
        fmin(diagnostics.minimum_density, density[radial]);
    if (!(density[radial] > Real{0})) {
      diagnostics.status |= static_cast<std::uint64_t>(
          ToroidalAssemblyStatus::nonpositive_density);
    }

    Real pest_sum = Real{0};
    bool pest_finite = true;
    for (int theta = 0; theta < n_theta; ++theta) {
      const std::size_t index =
          static_cast<std::size_t>(radial) * n_theta + theta;
      const Real gll = g_lambda_lambda[index];
      const Real glt = g_lambda_theta[index];
      const Real gtt = g_theta_theta[index];
      const Real gpp = g_phi_phi[index];
      const Real jac = jacobian[index];
      const Real bth = b_theta[index];
      const Real bph = b_phi[index];
      const Real jth = j_theta[index];
      const Real jph = j_phi[index];
      const bool metric_finite =
          isfinite(gll) && isfinite(glt) && isfinite(gtt) && isfinite(gpp);
      if (!metric_finite || !isfinite(jac) || !isfinite(bth)
          || !isfinite(bph) || !isfinite(jth) || !isfinite(jph)
          || !isfinite(jacobian_lambda[index])
          || !isfinite(jacobian_theta[index])) {
        diagnostics.status |= static_cast<std::uint64_t>(
            ToroidalAssemblyStatus::nonfinite_input);
      }
      if (metric_finite
          && (!(gll > Real{0}) || !(gtt > Real{0}) || !(gpp > Real{0})
              || !(gll * gtt - glt * glt > Real{0}))) {
        diagnostics.status |= static_cast<std::uint64_t>(
            ToroidalAssemblyStatus::invalid_metric);
        if (diagnostics.first_invalid_metric_point
            == std::numeric_limits<std::size_t>::max()) {
          diagnostics.first_invalid_metric_point = index;
        }
      }
      diagnostics.minimum_jacobian =
          fmin(diagnostics.minimum_jacobian, jac);
      if (!(jac > Real{0})) {
        diagnostics.status |= static_cast<std::uint64_t>(
            ToroidalAssemblyStatus::nonpositive_jacobian);
      }
      const Real pitch = bph / bth;
      const Real pitch_deviation = relative_deviation(pitch, q[radial]);
      const Real scale_deviation =
          relative_deviation(jac * bth, signed_flux_scale);
      diagnostics.maximum_field_pitch_deviation =
          fmax(diagnostics.maximum_field_pitch_deviation, pitch_deviation);
      diagnostics.maximum_flux_scale_deviation =
          fmax(diagnostics.maximum_flux_scale_deviation, scale_deviation);
      const Real pest = jac / gpp;
      if (!isfinite(pest)) pest_finite = false;
      pest_sum += pest;
    }

    if (!pest_finite || !isfinite(pest_sum)) {
      diagnostics.maximum_pest_deviation =
          std::numeric_limits<Real>::infinity();
    } else {
      const Real pest_mean = pest_sum / static_cast<Real>(n_theta);
      for (int theta = 0; theta < n_theta; ++theta) {
        const std::size_t index =
            static_cast<std::size_t>(radial) * n_theta + theta;
        diagnostics.maximum_pest_deviation = fmax(
            diagnostics.maximum_pest_deviation,
            relative_deviation(jacobian[index] / g_phi_phi[index],
                               pest_mean));
      }
    }
  }

  if (diagnostics.maximum_field_pitch_deviation
      > field_pitch_tolerance) {
    diagnostics.status |= static_cast<std::uint64_t>(
        ToroidalAssemblyStatus::inconsistent_field_pitch);
  }
  if (diagnostics.maximum_flux_scale_deviation
      > flux_scale_tolerance) {
    diagnostics.status |= static_cast<std::uint64_t>(
        ToroidalAssemblyStatus::inconsistent_flux_scale);
  }
  if (diagnostics.maximum_pest_deviation > pest_tolerance) {
    diagnostics.status |= static_cast<std::uint64_t>(
        ToroidalAssemblyStatus::non_pest_geometry);
  }
  *out = diagnostics;
}

__device__ int local_occurrence_for_domain(
    int free_dof, int domain, const int* occurrence_count,
    const int* occurrence_domain, const int* occurrence_local_node) {
  const int count = occurrence_count[free_dof];
  for (int occurrence = 0; occurrence < count; ++occurrence) {
    const int slot = kMaximumOccurrences * free_dof + occurrence;
    if (occurrence_domain[slot] == domain) {
      return occurrence_local_node[slot];
    }
  }
  return -1;
}

__device__ BasisFields evaluate_basis_fields(
    int harmonic, int component, int local_node, int quadrature_node,
    int theta_index, int n_theta, int n_toroidal, Real signed_flux_scale,
    Real q, Real q_lambda, Real jacobian, Real jacobian_lambda,
    Real jacobian_theta, const Real* differentiation_row) {
  const Real theta_source =
      -Real{2} * pi * static_cast<Real>(theta_index)
      / static_cast<Real>(n_theta);
  const Real phase_angle = static_cast<Real>(harmonic) * theta_source;
  ComplexValue phase{cos(phase_angle), sin(phase_angle)};
  ComplexValue value{};
  if (quadrature_node == local_node) value = phase;
  const ComplexValue derivative = differentiation_row[local_node] * phase;
  const Real inv_jacobian = Real{1} / jacobian;

  BasisFields fields{};
  if (component == static_cast<int>(DisplacementComponent::psi)) {
    fields.xi_lambda = value;
    fields.q_lambda =
        (signed_flux_scale * static_cast<Real>(harmonic - n_toroidal * q)
         * inv_jacobian)
        * multiply_i(value);
    fields.q_theta = (-signed_flux_scale * inv_jacobian) * derivative;
    fields.q_phi = (-signed_flux_scale * inv_jacobian)
                 * (q * derivative + q_lambda * value);
    fields.divergence =
        derivative + (jacobian_lambda * inv_jacobian) * value;
  } else if (component
             == static_cast<int>(DisplacementComponent::theta)) {
    fields.xi_theta = value;
    fields.q_theta =
        (-static_cast<Real>(n_toroidal) * signed_flux_scale * q
         * inv_jacobian)
        * multiply_i(value);
    fields.q_phi =
        (-static_cast<Real>(harmonic) * signed_flux_scale * q
         * inv_jacobian)
        * multiply_i(value);
    fields.divergence =
        static_cast<Real>(harmonic) * multiply_i(value)
        + (jacobian_theta * inv_jacobian) * value;
  } else {
    fields.xi_phi = value;
    fields.q_theta =
        (static_cast<Real>(n_toroidal) * signed_flux_scale * inv_jacobian)
        * multiply_i(value);
    fields.q_phi =
        (static_cast<Real>(harmonic) * signed_flux_scale * inv_jacobian)
        * multiply_i(value);
    fields.divergence =
        (-static_cast<Real>(n_toroidal)) * multiply_i(value);
  }
  return fields;
}

__device__ ComplexValue metric_inner_product(
    ComplexValue a_lambda, ComplexValue a_theta, ComplexValue a_phi,
    ComplexValue b_lambda, ComplexValue b_theta, ComplexValue b_phi,
    Real gll, Real glt, Real gtt, Real gpp) {
  ComplexValue value{};
  value += gll * conjugate_multiply(a_lambda, b_lambda);
  value += glt * conjugate_multiply(a_lambda, b_theta);
  value += glt * conjugate_multiply(a_theta, b_lambda);
  value += gtt * conjugate_multiply(a_theta, b_theta);
  value += gpp * conjugate_multiply(a_phi, b_phi);
  return value;
}

__device__ ComplexValue current_cross_first(
    const BasisFields& a, const BasisFields& b, Real jacobian,
    Real j_theta, Real j_phi) {
  ComplexValue value{};
  value += (jacobian * j_theta)
         * (conjugate(a.xi_phi) * b.q_lambda
            - conjugate(a.xi_lambda) * b.q_phi);
  value += (jacobian * j_phi)
         * (conjugate(a.xi_lambda) * b.q_theta
            - conjugate(a.xi_theta) * b.q_lambda);
  return value;
}

__device__ BilinearValues evaluate_bilinear_values(
    const BasisFields& test, const BasisFields& trial, Real gamma, Real pressure,
    Real pressure_lambda, Real density, Real g_lambda_lambda,
    Real g_lambda_theta, Real g_theta_theta, Real g_phi_phi, Real jacobian,
    Real j_theta, Real j_phi) {
  BilinearValues values;
  values.stiffness =
      (Real{1} / mu0)
      * metric_inner_product(
          test.q_lambda, test.q_theta, test.q_phi, trial.q_lambda,
          trial.q_theta, trial.q_phi, g_lambda_lambda, g_lambda_theta,
          g_theta_theta, g_phi_phi);
  values.stiffness +=
      (gamma * pressure)
      * conjugate_multiply(test.divergence, trial.divergence);
  values.stiffness +=
      (Real{0.5} * pressure_lambda)
      * (conjugate(test.xi_lambda) * trial.divergence
         + conjugate(test.divergence) * trial.xi_lambda);
  values.stiffness +=
      Real{0.5}
      * (current_cross_first(test, trial, jacobian, j_theta, j_phi)
         + conjugate(current_cross_first(
             trial, test, jacobian, j_theta, j_phi)));

  values.inertia =
      density
      * metric_inner_product(
          test.xi_lambda, test.xi_theta, test.xi_phi, trial.xi_lambda,
          trial.xi_theta, trial.xi_phi, g_lambda_lambda, g_lambda_theta,
          g_theta_theta, g_phi_phi);
  return values;
}

__device__ void write_realified_hermitian_entry(
    Real* matrix, int real_order, int row_complex, int column_complex,
    ComplexValue value) {
  const int row_cos = 2 * row_complex;
  const int row_sin = row_cos + 1;
  const int column_cos = 2 * column_complex;
  const int column_sin = column_cos + 1;

  if (row_complex == column_complex) {
    matrix[static_cast<std::size_t>(row_cos)
           + static_cast<std::size_t>(column_cos) * real_order] = value.real;
    matrix[static_cast<std::size_t>(row_sin)
           + static_cast<std::size_t>(column_sin) * real_order] = value.real;
    matrix[static_cast<std::size_t>(row_cos)
           + static_cast<std::size_t>(column_sin) * real_order] = Real{0};
    matrix[static_cast<std::size_t>(row_sin)
           + static_cast<std::size_t>(column_cos) * real_order] = Real{0};
    return;
  }

  // z = cosine - i*sine, hence [[Re, Im],[-Im, Re]].
  matrix[static_cast<std::size_t>(row_cos)
         + static_cast<std::size_t>(column_cos) * real_order] = value.real;
  matrix[static_cast<std::size_t>(row_cos)
         + static_cast<std::size_t>(column_sin) * real_order] = value.imag;
  matrix[static_cast<std::size_t>(row_sin)
         + static_cast<std::size_t>(column_cos) * real_order] = -value.imag;
  matrix[static_cast<std::size_t>(row_sin)
         + static_cast<std::size_t>(column_sin) * real_order] = value.real;

  matrix[static_cast<std::size_t>(column_cos)
         + static_cast<std::size_t>(row_cos) * real_order] = value.real;
  matrix[static_cast<std::size_t>(column_sin)
         + static_cast<std::size_t>(row_cos) * real_order] = value.imag;
  matrix[static_cast<std::size_t>(column_cos)
         + static_cast<std::size_t>(row_sin) * real_order] = -value.imag;
  matrix[static_cast<std::size_t>(column_sin)
         + static_cast<std::size_t>(row_sin) * real_order] = value.real;
}

__global__ void assemble_matrices_kernel(
    const Real* __restrict__ weights,
    const Real* __restrict__ differentiation,
    const Real* __restrict__ q,
    const Real* __restrict__ pressure,
    const Real* __restrict__ pressure_lambda,
    const Real* __restrict__ density,
    const Real* __restrict__ g_lambda_lambda,
    const Real* __restrict__ g_lambda_theta,
    const Real* __restrict__ g_theta_theta,
    const Real* __restrict__ g_phi_phi,
    const Real* __restrict__ jacobian,
    const Real* __restrict__ j_theta,
    const Real* __restrict__ j_phi,
    const Real* __restrict__ q_lambda,
    const Real* __restrict__ jacobian_lambda,
    const Real* __restrict__ jacobian_theta,
    const int* __restrict__ occurrence_count,
    const int* __restrict__ occurrence_domain,
    const int* __restrict__ occurrence_local_node,
    const int* __restrict__ harmonic,
    const int* __restrict__ component,
    int complex_order, int n_domains, int n_nodes, int n_theta,
    int n_toroidal, Real gamma, Real signed_flux_scale,
    Real* __restrict__ stiffness, Real* __restrict__ inertia,
    Real* __restrict__ hermitian_residuals) {
  const int column =
      static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (row >= complex_order || column >= complex_order || row > column) return;

  ComplexValue stiffness_ab{};
  ComplexValue inertia_ab{};
  ComplexValue stiffness_ba{};
  ComplexValue inertia_ba{};
  for (int domain = 0; domain < n_domains; ++domain) {
    const int row_local = local_occurrence_for_domain(
        row, domain, occurrence_count, occurrence_domain,
        occurrence_local_node);
    const int column_local = local_occurrence_for_domain(
        column, domain, occurrence_count, occurrence_domain,
        occurrence_local_node);
    if (row_local < 0 || column_local < 0) continue;

    const std::size_t radial_base =
        static_cast<std::size_t>(domain) * n_nodes;
    const std::size_t differentiation_base =
        static_cast<std::size_t>(domain) * n_nodes * n_nodes;
    for (int quadrature = 0; quadrature < n_nodes; ++quadrature) {
      const int radial = static_cast<int>(radial_base) + quadrature;
      const Real radial_weight = weights[radial];
      const Real q_value = q[radial];
      const Real q_derivative = q_lambda[radial];
      const Real p = pressure[radial];
      const Real p_lambda = pressure_lambda[radial];
      const Real rho = density[radial];
      const Real* differentiation_row =
          differentiation + differentiation_base
          + static_cast<std::size_t>(quadrature) * n_nodes;

      for (int theta = 0; theta < n_theta; ++theta) {
        const std::size_t geometry =
            static_cast<std::size_t>(radial) * n_theta + theta;
        const Real jac = jacobian[geometry];
        const BasisFields a = evaluate_basis_fields(
            harmonic[row], component[row], row_local, quadrature, theta,
            n_theta, n_toroidal, signed_flux_scale, q_value, q_derivative,
            jac, jacobian_lambda[geometry], jacobian_theta[geometry],
            differentiation_row);
        const BasisFields b = evaluate_basis_fields(
            harmonic[column], component[column], column_local, quadrature,
            theta, n_theta, n_toroidal, signed_flux_scale, q_value,
            q_derivative, jac, jacobian_lambda[geometry],
            jacobian_theta[geometry], differentiation_row);

        const BilinearValues ab = evaluate_bilinear_values(
            a, b, gamma, p, p_lambda, rho, g_lambda_lambda[geometry],
            g_lambda_theta[geometry], g_theta_theta[geometry],
            g_phi_phi[geometry], jac, j_theta[geometry], j_phi[geometry]);
        const BilinearValues ba = evaluate_bilinear_values(
            b, a, gamma, p, p_lambda, rho, g_lambda_lambda[geometry],
            g_lambda_theta[geometry], g_theta_theta[geometry],
            g_phi_phi[geometry], jac, j_theta[geometry], j_phi[geometry]);
        const Real volume_weight =
            radial_weight * (Real{4} * pi * pi / static_cast<Real>(n_theta))
            * jac;
        stiffness_ab += volume_weight * ab.stiffness;
        inertia_ab += volume_weight * ab.inertia;
        stiffness_ba += volume_weight * ba.stiffness;
        inertia_ba += volume_weight * ba.inertia;
      }
    }
  }

  const ComplexValue stiffness_ba_conjugate = conjugate(stiffness_ba);
  const ComplexValue inertia_ba_conjugate = conjugate(inertia_ba);
  const Real stiffness_residual =
      relative_deviation(stiffness_ab, stiffness_ba_conjugate);
  const Real inertia_residual =
      relative_deviation(inertia_ab, inertia_ba_conjugate);
  hermitian_residuals[static_cast<std::size_t>(row)
                      + static_cast<std::size_t>(column) * complex_order] =
      fmax(stiffness_residual, inertia_residual);

  const ComplexValue stiffness_entry =
      Real{0.5} * (stiffness_ab + stiffness_ba_conjugate);
  const ComplexValue inertia_entry =
      Real{0.5} * (inertia_ab + inertia_ba_conjugate);
  const int real_order = 2 * complex_order;
  write_realified_hermitian_entry(stiffness, real_order, row, column,
                                  stiffness_entry);
  write_realified_hermitian_entry(inertia, real_order, row, column,
                                  inertia_entry);
}

// Partial result of one thread block's scan over the dense matrices.
//
// Only min, max, and logical-or reductions appear here.  All three are exactly
// associative and commutative on IEEE doubles, so the block decomposition below
// returns the same bits as the serial scan it replaces regardless of how the
// grid is scheduled.  That is what allows this validation to be parallel
// without weakening the module's determinism contract.
struct MatrixScanPartial {
  Real maximum_residual;
  Real minimum_diagonal;
  int nonfinite;
  int nonfinite_inertia;
};

__device__ MatrixScanPartial combine(MatrixScanPartial a,
                                     MatrixScanPartial b) {
  MatrixScanPartial result;
  result.maximum_residual = fmax(a.maximum_residual, b.maximum_residual);
  result.minimum_diagonal = fmin(a.minimum_diagonal, b.minimum_diagonal);
  result.nonfinite = a.nonfinite | b.nonfinite;
  result.nonfinite_inertia = a.nonfinite_inertia | b.nonfinite_inertia;
  return result;
}

__device__ MatrixScanPartial identity_partial() {
  return MatrixScanPartial{-std::numeric_limits<Real>::infinity(),
                           std::numeric_limits<Real>::infinity(), 0, 0};
}

// fmax/fmin propagate the non-NaN operand, so a NaN residual would vanish in
// the reduction.  Map it onto infinity, which the tolerance comparison rejects.
__device__ Real reducible_residual(Real residual) {
  return isnan(residual) ? std::numeric_limits<Real>::infinity() : residual;
}

template <unsigned BlockSize>
__device__ MatrixScanPartial reduce_block(MatrixScanPartial value) {
  __shared__ MatrixScanPartial shared[BlockSize];
  const unsigned lane = threadIdx.x;
  shared[lane] = value;
  __syncthreads();
  for (unsigned stride = BlockSize / 2; stride > 0; stride >>= 1) {
    if (lane < stride) {
      shared[lane] = combine(shared[lane], shared[lane + stride]);
    }
    __syncthreads();
  }
  return shared[0];
}

// Grid-stride scan of the two dense real matrices: finiteness, the inertia
// diagonal minimum, and the symmetry residual against the transposed entry.
__global__ void scan_matrices_kernel(
    const Real* __restrict__ stiffness, const Real* __restrict__ inertia,
    int order, MatrixScanPartial* __restrict__ partials) {
  const std::size_t total =
      static_cast<std::size_t>(order) * static_cast<std::size_t>(order);
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  MatrixScanPartial value = identity_partial();
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                         + threadIdx.x;
       index < total; index += stride) {
    const std::size_t row = index % static_cast<std::size_t>(order);
    const std::size_t column = index / static_cast<std::size_t>(order);
    const std::size_t transpose =
        column + row * static_cast<std::size_t>(order);
    const Real stiffness_entry = stiffness[index];
    const Real inertia_entry = inertia[index];

    MatrixScanPartial contribution = identity_partial();
    if (!isfinite(stiffness_entry) || !isfinite(inertia_entry)) {
      contribution.nonfinite = 1;
    }
    if (!isfinite(inertia_entry)) contribution.nonfinite_inertia = 1;
    if (row == column) contribution.minimum_diagonal = inertia_entry;
    contribution.maximum_residual = fmax(
        reducible_residual(
            relative_deviation(stiffness_entry, stiffness[transpose])),
        reducible_residual(
            relative_deviation(inertia_entry, inertia[transpose])));
    value = combine(value, contribution);
  }

  value = reduce_block<kBlockReduce>(value);
  if (threadIdx.x == 0) partials[blockIdx.x] = value;
}

// Grid-stride scan of the per-entry Hermitian residuals recorded by the
// assembly.  Only the upper triangle (row <= column) was written.
__global__ void scan_assembly_residuals_kernel(
    const Real* __restrict__ assembly_hermitian_residuals, int complex_order,
    MatrixScanPartial* __restrict__ partials) {
  const std::size_t total = static_cast<std::size_t>(complex_order)
                          * static_cast<std::size_t>(complex_order);
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  MatrixScanPartial value = identity_partial();
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                         + threadIdx.x;
       index < total; index += stride) {
    const std::size_t row = index % static_cast<std::size_t>(complex_order);
    const std::size_t column =
        index / static_cast<std::size_t>(complex_order);
    if (row > column) continue;
    MatrixScanPartial contribution = identity_partial();
    contribution.maximum_residual =
        reducible_residual(assembly_hermitian_residuals[index]);
    value = combine(value, contribution);
  }

  value = reduce_block<kBlockReduce>(value);
  if (threadIdx.x == 0) partials[blockIdx.x] = value;
}

// Single-block final reduction over the two partial arrays, followed by the
// status decisions the serial kernel used to make.
__global__ void combine_matrix_scan_kernel(
    const MatrixScanPartial* __restrict__ matrix_partials,
    int matrix_partial_count,
    const MatrixScanPartial* __restrict__ residual_partials,
    int residual_partial_count, Real hermitian_tolerance,
    DeviceDiagnostics* __restrict__ diagnostics) {
  MatrixScanPartial value = identity_partial();
  for (int index = static_cast<int>(threadIdx.x);
       index < matrix_partial_count;
       index += static_cast<int>(blockDim.x)) {
    value = combine(value, matrix_partials[index]);
  }
  for (int index = static_cast<int>(threadIdx.x);
       index < residual_partial_count;
       index += static_cast<int>(blockDim.x)) {
    value = combine(value, residual_partials[index]);
  }
  value = reduce_block<kBlockReduce>(value);
  if (threadIdx.x != 0) return;

  // The input-validation kernel already recorded a residual; keep the maximum.
  const Real hermitian_residual =
      fmax(diagnostics->hermitian_residual, value.maximum_residual);
  diagnostics->minimum_mass_diagonal = value.minimum_diagonal;
  diagnostics->hermitian_residual = hermitian_residual;
  diagnostics->nonfinite_inertia = value.nonfinite_inertia;
  if (value.nonfinite != 0) {
    diagnostics->status |= static_cast<std::uint64_t>(
        ToroidalAssemblyStatus::nonfinite_matrix);
  }
  if (!(value.minimum_diagonal > Real{0})) {
    diagnostics->status |= static_cast<std::uint64_t>(
        ToroidalAssemblyStatus::nonpositive_mass_diagonal);
  }
  if (!(hermitian_residual <= hermitian_tolerance)) {
    diagnostics->status |= static_cast<std::uint64_t>(
        ToroidalAssemblyStatus::nonhermitian_matrix);
  }
}

// Host-side driver replacing the former single-thread validate kernel.  The
// dense scan is O(order^2) work; running it on one lane made validation the
// dominant cost of the assembly at production resolutions.
void launch_matrix_validation(
    const backend::DeviceBuffer<Real>& stiffness,
    const backend::DeviceBuffer<Real>& inertia,
    const backend::DeviceBuffer<Real>* assembly_hermitian_residuals,
    int complex_order, int order, Real hermitian_tolerance, int owner,
    backend::DeviceBuffer<DeviceDiagnostics>& diagnostics,
    stream_t stream) {
  auto* hip_stream = static_cast<hipStream_t>(stream);
  const std::size_t matrix_entries =
      static_cast<std::size_t>(order) * static_cast<std::size_t>(order);
  const int matrix_blocks = static_cast<int>(std::min<std::size_t>(
      kMaximumScanBlocks,
      (matrix_entries + kBlockReduce - 1) / kBlockReduce));

  backend::DeviceBuffer<MatrixScanPartial> matrix_partials{
      static_cast<std::size_t>(matrix_blocks), backend::uninitialized,
      backend::on_device(owner)};
  hipLaunchKernelGGL(scan_matrices_kernel, dim3{
                         static_cast<unsigned>(matrix_blocks)},
                     dim3{kBlockReduce}, 0, hip_stream,
                     stiffness.device_ptr(), inertia.device_ptr(), order,
                     matrix_partials.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  int residual_blocks = 0;
  backend::DeviceBuffer<MatrixScanPartial> residual_partials{};
  if (assembly_hermitian_residuals != nullptr && complex_order > 0) {
    const std::size_t residual_entries =
        static_cast<std::size_t>(complex_order)
        * static_cast<std::size_t>(complex_order);
    residual_blocks = static_cast<int>(std::min<std::size_t>(
        kMaximumScanBlocks,
        (residual_entries + kBlockReduce - 1) / kBlockReduce));
    residual_partials = backend::DeviceBuffer<MatrixScanPartial>{
        static_cast<std::size_t>(residual_blocks), backend::uninitialized,
        backend::on_device(owner)};
    hipLaunchKernelGGL(scan_assembly_residuals_kernel,
                       dim3{static_cast<unsigned>(residual_blocks)},
                       dim3{kBlockReduce}, 0, hip_stream,
                       assembly_hermitian_residuals->device_ptr(),
                       complex_order, residual_partials.device_ptr());
    QUASAR_HIP_CHECK(hipGetLastError());
  }

  hipLaunchKernelGGL(
      combine_matrix_scan_kernel, dim3{1}, dim3{kBlockReduce}, 0, hip_stream,
      matrix_partials.device_ptr(), matrix_blocks,
      residual_blocks > 0 ? residual_partials.device_ptr() : nullptr,
      residual_blocks, hermitian_tolerance, diagnostics.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  // The partial buffers are freed when this scope exits, so the caller must
  // observe the combine kernel before that happens.
  backend::device_synchronize(stream);
}

DeviceDiagnostics download_diagnostics(
    const backend::DeviceBuffer<DeviceDiagnostics>& device, stream_t stream) {
  DeviceDiagnostics host{};
  device.copy_to_host_async(&host, 1, stream);
  backend::device_synchronize(stream);
  return host;
}

ToroidalAssemblyDiagnostics to_public_diagnostics(
    const DeviceDiagnostics& host, std::size_t dense_storage_bytes) {
  ToroidalAssemblyDiagnostics result;
  result.status = static_cast<ToroidalAssemblyStatus>(host.status);
  result.minimum_jacobian = host.minimum_jacobian;
  result.minimum_density = host.minimum_density;
  result.maximum_field_pitch_deviation =
      host.maximum_field_pitch_deviation;
  result.maximum_flux_scale_deviation = host.maximum_flux_scale_deviation;
  result.maximum_pest_deviation = host.maximum_pest_deviation;
  result.minimum_mass_diagonal = host.minimum_mass_diagonal;
  result.hermitian_residual = host.hermitian_residual;
  result.first_invalid_metric_point = host.first_invalid_metric_point;
  result.dense_storage_bytes = dense_storage_bytes;
  return result;
}

constexpr std::uint64_t matrix_status_mask() {
  return static_cast<std::uint64_t>(
             ToroidalAssemblyStatus::nonfinite_matrix)
       | static_cast<std::uint64_t>(
             ToroidalAssemblyStatus::nonpositive_mass_diagonal)
       | static_cast<std::uint64_t>(
             ToroidalAssemblyStatus::mass_not_positive_definite)
       | static_cast<std::uint64_t>(
             ToroidalAssemblyStatus::nonhermitian_matrix);
}

void clear_matrix_diagnostics(ToroidalAssemblyDiagnostics& diagnostics) {
  diagnostics.status = static_cast<ToroidalAssemblyStatus>(
      static_cast<std::uint64_t>(diagnostics.status)
      & ~matrix_status_mask());
  diagnostics.minimum_mass_diagonal = Real{0};
  diagnostics.hermitian_residual = Real{0};
  diagnostics.mass_failed_leading_minor_order = 0;
}

DeviceDiagnostics to_device_diagnostics(
    const ToroidalAssemblyDiagnostics& diagnostics) {
  DeviceDiagnostics result{};
  result.status = static_cast<std::uint64_t>(diagnostics.status);
  result.minimum_jacobian = diagnostics.minimum_jacobian;
  result.minimum_density = diagnostics.minimum_density;
  result.maximum_field_pitch_deviation =
      diagnostics.maximum_field_pitch_deviation;
  result.maximum_flux_scale_deviation =
      diagnostics.maximum_flux_scale_deviation;
  result.maximum_pest_deviation = diagnostics.maximum_pest_deviation;
  result.minimum_mass_diagonal = diagnostics.minimum_mass_diagonal;
  result.hermitian_residual = diagnostics.hermitian_residual;
  result.first_invalid_metric_point =
      diagnostics.first_invalid_metric_point;
  return result;
}

void validate_mass_positive_definite(
    ToroidalMatrixPair& matrices, bool inertia_is_finite, stream_t stream) {
  if (!inertia_is_finite) return;

  const numerics::CholeskyResult factorization =
      numerics::check_symmetric_positive_definite(
          matrices.inertia, matrices.real_order,
          numerics::MatrixTriangle::lower, stream);
  if (factorization.status
      == numerics::CholeskyStatus::invalid_solver_argument) {
    throw std::runtime_error{
        "hipSOLVER potrf rejected argument "
        + std::to_string(factorization.invalid_argument_position)};
  }
  if (factorization.status
      == numerics::CholeskyStatus::not_positive_definite) {
    matrices.diagnostics.status |=
        ToroidalAssemblyStatus::mass_not_positive_definite;
    matrices.diagnostics.mass_failed_leading_minor_order =
        factorization.failed_leading_minor_order;
  }
}

}  // namespace

FixedBoundaryDofMap::FixedBoundaryDofMap(
    const SpectralDofLayout& layout)
    : full_to_free_complex_(layout.complex_dof_count(), eliminated) {
  std::vector<unsigned char> remove(layout.complex_dof_count(), 0);
  for (int m = -layout.m_max(); m <= layout.m_max(); ++m) {
    const std::size_t outer = layout.n_radial(m) - 1;
    remove[layout.complex_dof(outer, m, DisplacementComponent::psi)] = 1;
  }
  for (std::size_t full = 0; full < remove.size(); ++full) {
    if (remove[full] != 0) continue;
    full_to_free_complex_[full] = free_to_full_complex_.size();
    free_to_full_complex_.push_back(full);
  }
}

std::size_t FixedBoundaryDofMap::free_complex_dof(
    std::size_t full_complex_dof) const {
  if (full_complex_dof >= full_to_free_complex_.size()) {
    throw std::out_of_range{"full complex DOF is out of range"};
  }
  return full_to_free_complex_[full_complex_dof];
}

std::size_t FixedBoundaryDofMap::full_complex_dof(
    std::size_t free_complex_dof) const {
  if (free_complex_dof >= free_to_full_complex_.size()) {
    throw std::out_of_range{"free complex DOF is out of range"};
  }
  return free_to_full_complex_[free_complex_dof];
}

std::size_t FixedBoundaryDofMap::complex_dof(
    const SpectralDofLayout& layout, std::size_t radial, int m,
    DisplacementComponent component) const {
  if (layout.complex_dof_count() != full_to_free_complex_.size()) {
    throw std::invalid_argument{
        "FixedBoundaryDofMap was built for a different spectral layout"};
  }
  return free_complex_dof(layout.complex_dof(radial, m, component));
}

std::size_t FixedBoundaryDofMap::dof(
    const SpectralDofLayout& layout, std::size_t radial, int m,
    DisplacementComponent component, FourierQuadrature quadrature) const {
  const std::size_t complex = complex_dof(layout, radial, m, component);
  if (complex == eliminated) return eliminated;
  const int q = static_cast<int>(quadrature);
  if (q < 0 || q >= SpectralDofLayout::kQuadratureCount) {
    throw std::out_of_range{"Fourier quadrature is out of range"};
  }
  return 2 * complex + static_cast<std::size_t>(q);
}

bool FixedBoundaryDofMap::is_eliminated(std::size_t full_complex_dof) const {
  return free_complex_dof(full_complex_dof) == eliminated;
}

ToroidalAssemblyStatus toroidal_dense_storage_bytes(
    std::size_t real_order, std::size_t& bytes) noexcept {
  bytes = 0;
  const std::size_t maximum = std::numeric_limits<std::size_t>::max();
  if (real_order != 0 && real_order > maximum / real_order) {
    return ToroidalAssemblyStatus::storage_overflow;
  }
  const std::size_t entries = real_order * real_order;
  if (entries > maximum / sizeof(Real)) {
    return ToroidalAssemblyStatus::storage_overflow;
  }
  const std::size_t matrix_bytes = entries * sizeof(Real);
  if (matrix_bytes > maximum / std::size_t{2}) {
    return ToroidalAssemblyStatus::storage_overflow;
  }
  bytes = std::size_t{2} * matrix_bytes;
  return ToroidalAssemblyStatus::success;
}

void launch_validate_toroidal_matrix_pair(
    ToroidalMatrixPair& matrices, Real hermitian_tolerance, stream_t stream) {
  if (matrices.real_order <= 0) {
    throw std::invalid_argument{
        "toroidal matrix validation requires a positive real order"};
  }
  if (matrices.complex_order < 0
      || (matrices.complex_order != 0
          && (matrices.complex_order
                  > std::numeric_limits<int>::max() / 2
              || matrices.real_order != 2 * matrices.complex_order))) {
    throw std::invalid_argument{
        "toroidal matrix validation found inconsistent real/complex orders"};
  }
  if (!(hermitian_tolerance >= Real{0})
      || !std::isfinite(hermitian_tolerance)) {
    throw std::invalid_argument{
        "toroidal matrix validation tolerance must be finite and non-negative"};
  }

  const std::size_t order = static_cast<std::size_t>(matrices.real_order);
  if (order > std::numeric_limits<std::size_t>::max() / order) {
    throw std::invalid_argument{
        "toroidal matrix validation order overflows matrix storage"};
  }
  const std::size_t matrix_entries = order * order;
  require_size(matrices.stiffness, matrix_entries,
               "ToroidalMatrixPair::stiffness");
  require_size(matrices.inertia, matrix_entries,
               "ToroidalMatrixPair::inertia");
  if (matrices.stiffness.owner_device() != matrices.inertia.owner_device()) {
    throw std::invalid_argument{
        "toroidal stiffness and inertia reside on different devices"};
  }

  clear_matrix_diagnostics(matrices.diagnostics);
  const int owner = matrices.stiffness.owner_device();
  backend::DeviceGuard guard{owner};
  backend::DeviceBuffer<DeviceDiagnostics> device_diagnostics{
      1, backend::on_device(owner)};
  const DeviceDiagnostics initial =
      to_device_diagnostics(matrices.diagnostics);
  device_diagnostics.copy_from_host_async(&initial, 1, stream);

  launch_matrix_validation(matrices.stiffness, matrices.inertia, nullptr, 0,
                           matrices.real_order, hermitian_tolerance, owner,
                           device_diagnostics, stream);
  const DeviceDiagnostics host =
      download_diagnostics(device_diagnostics, stream);
  matrices.diagnostics =
      to_public_diagnostics(host, matrices.diagnostics.dense_storage_bytes);
  validate_mass_positive_definite(
      matrices, host.nonfinite_inertia == 0, stream);
}

void launch_assemble_fixed_boundary_toroidal_matrices(
    const ChebyshevBasis& basis, const RadialDomains& domains,
    const FluxCoordinateGrid& coords,
    const ToroidalEquilibriumFields& equilibrium,
    const SpectralDofLayout& layout, const ToroidalAssemblyConfig& config,
    ToroidalMatrixPair& out, stream_t stream) {
  out = ToroidalMatrixPair{};
  const ProblemShape shape =
      validate_shape(basis, domains, coords, equilibrium, layout);

  ToroidalAssemblyStatus host_status = validate_topology(domains, layout);
  if (config.n_toroidal == 0) {
    host_status |= ToroidalAssemblyStatus::invalid_toroidal_mode;
  }
  if (!(config.gamma > Real{0}) || !std::isfinite(config.gamma)) {
    host_status |= ToroidalAssemblyStatus::invalid_gamma;
  }
  if (equilibrium.signed_flux_scale == Real{0}
      || !std::isfinite(equilibrium.signed_flux_scale)) {
    host_status |= ToroidalAssemblyStatus::zero_flux_scale;
  }
  if (!(domains.breakpoints[0] > Real{0})) {
    host_status |= ToroidalAssemblyStatus::unsupported_magnetic_axis;
  }
  if (!(config.field_pitch_tolerance >= Real{0})
      || !(config.flux_scale_tolerance >= Real{0})
      || !(config.pest_tolerance >= Real{0})
      || !(config.hermitian_tolerance >= Real{0})
      || !std::isfinite(config.field_pitch_tolerance)
      || !std::isfinite(config.flux_scale_tolerance)
      || !std::isfinite(config.pest_tolerance)
      || !std::isfinite(config.hermitian_tolerance)) {
    throw std::invalid_argument{
        "toroidal assembly tolerances must be finite and non-negative"};
  }

  std::size_t complex_order = 0;
  if (layout.n_harmonics() > layout.complex_dof_count()) {
    host_status |= ToroidalAssemblyStatus::invalid_topology;
  } else {
    complex_order = layout.complex_dof_count() - layout.n_harmonics();
  }

  std::size_t real_order_size = 0;
  std::size_t dense_bytes = 0;
  if (complex_order > std::numeric_limits<std::size_t>::max() / 2
      || complex_order
             > static_cast<std::size_t>(
                   std::numeric_limits<int>::max() / 2)) {
    host_status |= ToroidalAssemblyStatus::storage_overflow;
  } else {
    real_order_size = std::size_t{2} * complex_order;
    host_status |=
        toroidal_dense_storage_bytes(real_order_size, dense_bytes);
    if (!has_status(host_status, ToroidalAssemblyStatus::storage_overflow)
        && config.max_dense_storage_bytes != 0
        && dense_bytes > config.max_dense_storage_bytes) {
      host_status |= ToroidalAssemblyStatus::storage_limit_exceeded;
    }
  }
  out.diagnostics.status = host_status;
  out.diagnostics.dense_storage_bytes = dense_bytes;
  if (host_status != ToroidalAssemblyStatus::success) return;

  FixedBoundaryDofMap boundary_map{layout};
  if (boundary_map.complex_dof_count() != complex_order) {
    out.diagnostics.status |= ToroidalAssemblyStatus::invalid_topology;
    return;
  }
  out.full_to_free_complex = boundary_map.full_to_free_complex();
  out.free_to_full_complex = boundary_map.free_to_full_complex();
  const HostOccurrences occurrences =
      build_occurrences(layout, boundary_map);
  const int complex_order_int = static_cast<int>(complex_order);
  const int real_order = static_cast<int>(real_order_size);
  out.complex_order = complex_order_int;
  out.real_order = real_order;

  backend::DeviceGuard guard{shape.owner};
  out.q_lambda = backend::DeviceBuffer<Real>{
      static_cast<std::size_t>(shape.n_lambda), backend::uninitialized,
      backend::on_device(shape.owner)};
  out.jacobian_lambda = backend::DeviceBuffer<Real>{
      shape.geometry_points, backend::uninitialized,
      backend::on_device(shape.owner)};
  out.jacobian_theta = backend::DeviceBuffer<Real>{
      shape.geometry_points, backend::uninitialized,
      backend::on_device(shape.owner)};
  out.occurrence_count = backend::DeviceBuffer<int>{
      complex_order, backend::uninitialized, backend::on_device(shape.owner)};
  out.occurrence_domain = backend::DeviceBuffer<int>{
      kMaximumOccurrences * complex_order, backend::uninitialized,
      backend::on_device(shape.owner)};
  out.occurrence_local_node = backend::DeviceBuffer<int>{
      kMaximumOccurrences * complex_order, backend::uninitialized,
      backend::on_device(shape.owner)};
  out.harmonic = backend::DeviceBuffer<int>{
      complex_order, backend::uninitialized, backend::on_device(shape.owner)};
  out.component = backend::DeviceBuffer<int>{
      complex_order, backend::uninitialized, backend::on_device(shape.owner)};

  out.occurrence_count.copy_from_host_async(
      occurrences.count.data(), occurrences.count.size(), stream);
  out.occurrence_domain.copy_from_host_async(
      occurrences.domain.data(), occurrences.domain.size(), stream);
  out.occurrence_local_node.copy_from_host_async(
      occurrences.local_node.data(), occurrences.local_node.size(), stream);
  out.harmonic.copy_from_host_async(
      occurrences.harmonic.data(), occurrences.harmonic.size(), stream);
  out.component.copy_from_host_async(
      occurrences.component.data(), occurrences.component.size(), stream);

  backend::DeviceBuffer<Real> device_breakpoints{
      static_cast<std::size_t>(shape.n_domains + 1), backend::uninitialized,
      backend::on_device(shape.owner)};
  device_breakpoints.copy_from_host_async(
      domains.breakpoints, static_cast<std::size_t>(shape.n_domains + 1),
      stream);

  auto* hip_stream = static_cast<hipStream_t>(stream);
  const dim3 derivative_block{kBlock2D, kBlock2D};
  const dim3 derivative_grid{
      (static_cast<unsigned>(shape.n_theta) + derivative_block.x - 1)
          / derivative_block.x,
      (static_cast<unsigned>(shape.n_lambda) + derivative_block.y - 1)
          / derivative_block.y};
  hipLaunchKernelGGL(
      build_derivatives_kernel, derivative_grid, derivative_block, 0,
      hip_stream, coords.q.device_ptr(), equilibrium.jacobian.device_ptr(),
      basis.diff.device_ptr(), shape.n_domains, shape.n_nodes, shape.n_theta,
      out.q_lambda.device_ptr(), out.jacobian_lambda.device_ptr(),
      out.jacobian_theta.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  backend::DeviceBuffer<DeviceDiagnostics> device_diagnostics{
      1, backend::on_device(shape.owner)};
  hipLaunchKernelGGL(
      validate_inputs_kernel, dim3{1}, dim3{1}, 0, hip_stream,
      basis.nodes.device_ptr(), coords.psi_n.device_ptr(),
      device_breakpoints.device_ptr(),
      coords.q.device_ptr(), coords.valid.device_ptr(),
      equilibrium.pressure.device_ptr(),
      equilibrium.pressure_lambda.device_ptr(),
      equilibrium.density.device_ptr(),
      equilibrium.g_lambda_lambda.device_ptr(),
      equilibrium.g_lambda_theta.device_ptr(),
      equilibrium.g_theta_theta.device_ptr(),
      equilibrium.g_phi_phi.device_ptr(),
      equilibrium.jacobian.device_ptr(),
      equilibrium.b_theta.device_ptr(), equilibrium.b_phi.device_ptr(),
      equilibrium.j_theta.device_ptr(), equilibrium.j_phi.device_ptr(),
      out.q_lambda.device_ptr(), out.jacobian_lambda.device_ptr(),
      out.jacobian_theta.device_ptr(), shape.n_domains, shape.n_nodes,
      shape.n_lambda, shape.n_theta,
      equilibrium.signed_flux_scale, config.field_pitch_tolerance,
      config.flux_scale_tolerance, config.pest_tolerance,
      device_diagnostics.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());
  DeviceDiagnostics host_diagnostics =
      download_diagnostics(device_diagnostics, stream);
  out.diagnostics = to_public_diagnostics(host_diagnostics, dense_bytes);
  if (!out.diagnostics.ok()) return;

  const std::size_t matrix_entries = real_order_size * real_order_size;
  const std::size_t residual_entries = complex_order * complex_order;
  out.stiffness = backend::DeviceBuffer<Real>{
      matrix_entries, backend::uninitialized, backend::on_device(shape.owner)};
  out.inertia = backend::DeviceBuffer<Real>{
      matrix_entries, backend::uninitialized, backend::on_device(shape.owner)};
  backend::DeviceBuffer<Real> assembly_hermitian_residuals{
      residual_entries, backend::uninitialized, backend::on_device(shape.owner)};

  const dim3 assembly_block{kBlock2D, kBlock2D};
  const dim3 assembly_grid{
      (static_cast<unsigned>(complex_order_int) + assembly_block.x - 1)
          / assembly_block.x,
      (static_cast<unsigned>(complex_order_int) + assembly_block.y - 1)
          / assembly_block.y};
  hipLaunchKernelGGL(
      assemble_matrices_kernel, assembly_grid, assembly_block, 0, hip_stream,
      basis.weights.device_ptr(), basis.diff.device_ptr(),
      coords.q.device_ptr(), equilibrium.pressure.device_ptr(),
      equilibrium.pressure_lambda.device_ptr(),
      equilibrium.density.device_ptr(),
      equilibrium.g_lambda_lambda.device_ptr(),
      equilibrium.g_lambda_theta.device_ptr(),
      equilibrium.g_theta_theta.device_ptr(),
      equilibrium.g_phi_phi.device_ptr(),
      equilibrium.jacobian.device_ptr(), equilibrium.j_theta.device_ptr(),
      equilibrium.j_phi.device_ptr(), out.q_lambda.device_ptr(),
      out.jacobian_lambda.device_ptr(), out.jacobian_theta.device_ptr(),
      out.occurrence_count.device_ptr(), out.occurrence_domain.device_ptr(),
      out.occurrence_local_node.device_ptr(), out.harmonic.device_ptr(),
      out.component.device_ptr(), complex_order_int, shape.n_domains,
      shape.n_nodes, shape.n_theta, config.n_toroidal, config.gamma,
      equilibrium.signed_flux_scale, out.stiffness.device_ptr(),
      out.inertia.device_ptr(), assembly_hermitian_residuals.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());

  launch_matrix_validation(out.stiffness, out.inertia,
                           &assembly_hermitian_residuals, complex_order_int,
                           real_order, config.hermitian_tolerance,
                           shape.owner, device_diagnostics, stream);
  host_diagnostics = download_diagnostics(device_diagnostics, stream);
  out.diagnostics = to_public_diagnostics(host_diagnostics, dense_bytes);
  if (config.verify_mass_positive_definite) {
    validate_mass_positive_definite(
        out, host_diagnostics.nonfinite_inertia == 0, stream);
  }
}

}  // namespace quasar::stability
