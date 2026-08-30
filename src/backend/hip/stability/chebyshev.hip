// Chebyshev spectral basis on each radial subinterval.
//
// Nodes, Clenshaw-Curtis quadrature weights, and the spectral differentiation
// matrix, all mapped from the reference interval [-1, 1] onto each subinterval
// laid out by RadialDomains.
//
// -- The negative-sum trick is not optional -----------------------------------
// The closed-form diagonal of the Chebyshev differentiation matrix,
//
//   D_ii = -x_i / (2 (1 - x_i^2))    for interior nodes,
//
// is analytically correct and numerically poor: at high order it is a
// difference of large nearly-equal quantities, and the resulting matrix fails
// to differentiate a constant to exactly zero. Since the whole point of the
// operator is that it annihilates constants, that error appears as a spurious
// zero-order term in everything assembled from it.
//
// The fix is standard: build the off-diagonal entries from the closed form,
// then SET each diagonal to minus the sum of the rest of its row.
//
// This does not make the row sum exactly zero in floating point, and it is
// worth being precise about why. The diagonal is computed by summing the
// off-diagonals in one order; a caller applying the matrix to a vector of ones
// sums in a different order, so the two disagree at the last bit. What the
// trick guarantees is that the residual stays at the level of rounding on the
// ROW MAGNITUDE and does not grow with order -- which is exactly what the
// closed-form diagonal fails to do, since its cancellation error worsens as the
// order rises. Entries scale like order^2 over the interval width, so the
// residual must be judged relative to that, never against an absolute bound.
//
// -- Why Gauss-Lobatto ---------------------------------------------------------
// Lobatto nodes include the endpoints, so continuity across a subinterval
// interface is a condition on shared node values rather than an interpolation.
// The interfaces sit at resonant surfaces, where an interpolation error would
// land exactly on the singularity the subdivision exists to isolate.

#include "quasar/physics/stability/kernels.hpp"

#include "backend/hip/hip_check.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace quasar::stability {
namespace {

inline constexpr Real kPi = Real{3.14159265358979323846};

template <class T>
void require_size(const backend::DeviceBuffer<T>& buffer,
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

struct BasisLaunchShape {
  int n_nodes{0};
  int n_domains{0};
};

BasisLaunchShape validate_basis_launch(const RadialDomains* d_domains,
                                       const ChebyshevBasis& out) {
  if (d_domains == nullptr) {
    throw std::invalid_argument{
        "Chebyshev basis domains must be a non-null device pointer"};
  }
  if (out.order < 1
      || static_cast<long long>(out.n_nodes)
             != static_cast<long long>(out.order) + 1LL
      || out.n_domains < 1
      || out.n_domains > RadialDomains::kMaxDomains) {
    throw std::invalid_argument{
        "Chebyshev basis launch requires order >= 1, n_nodes == order + 1, "
        "and a supported positive domain count"};
  }

  const std::size_t n_nodes = static_cast<std::size_t>(out.n_nodes);
  const std::size_t n_domains = static_cast<std::size_t>(out.n_domains);
  const std::size_t n_local = backend::detail::checked_size_product(
      n_domains, n_nodes, "Chebyshev basis node count overflows size_t");
  const std::size_t n_diff = backend::detail::checked_size_product(
      n_local, n_nodes,
      "Chebyshev basis differentiation storage overflows size_t");

  require_size(out.nodes, n_local, "ChebyshevBasis::nodes");
  require_size(out.weights, n_local, "ChebyshevBasis::weights");
  require_size(out.diff, n_diff, "ChebyshevBasis::diff");

  const int owner = out.nodes.owner_device();
  if (owner < 0) {
    throw std::invalid_argument{
        "Chebyshev basis storage is not device-resident"};
  }
  require_owner(out.weights, owner, "ChebyshevBasis::weights");
  require_owner(out.diff, owner, "ChebyshevBasis::diff");
  if (backend::current_device() != owner) {
    throw std::invalid_argument{
        "Chebyshev basis storage is not on the current device"};
  }

  return BasisLaunchShape{out.n_nodes, out.n_domains};
}

// Reference-interval node: x_i = cos(pi i / N), descending from +1 to -1.
//
// Kept in the classical descending order rather than flipped to ascending,
// because the closed-form differentiation weights below are derived for it and
// reordering silently changes their signs.
__device__ inline Real cheb_node(int i, int n) {
  return cos(kPi * static_cast<Real>(i) / static_cast<Real>(n));
}

// Clenshaw-Curtis weights on [-1, 1] for N+1 Lobatto nodes, by the standard
// cosine series. Exact for polynomials up to degree N.
__device__ inline Real clenshaw_curtis_weight(int i, int n) {
  const Real ci = (i == 0 || i == n) ? Real{1} : Real{2};
  Real sum = Real{0};
  for (int k = 1; k <= n / 2; ++k) {
    const Real bk = (Real{2} * static_cast<Real>(k) == static_cast<Real>(n))
                        ? Real{1}
                        : Real{2};
    sum += bk / (Real{4} * static_cast<Real>(k) * static_cast<Real>(k) - Real{1})
         * cos(Real{2} * static_cast<Real>(k) * kPi * static_cast<Real>(i)
               / static_cast<Real>(n));
  }
  return ci / static_cast<Real>(n) * (Real{1} - sum);
}

// One block per domain. Threads cooperate over the rows of that domain's
// differentiation matrix.
__global__ void build_basis_kernel(const RadialDomains* __restrict__ domains,
                                   int n_nodes, Real* __restrict__ nodes,
                                   Real* __restrict__ weights,
                                   Real* __restrict__ diff) {
  const int d = static_cast<int>(blockIdx.x);
  if (d >= domains->n_domains) return;

  const int n = n_nodes - 1;  // polynomial order
  const Real a = domains->breakpoints[d];
  const Real b = domains->breakpoints[d + 1];
  const Real half = Real{0.5} * (b - a);
  const Real mid = Real{0.5} * (b + a);

  const std::size_t nbase = static_cast<std::size_t>(d) * n_nodes;
  const std::size_t mbase = static_cast<std::size_t>(d) * n_nodes * n_nodes;

  // Nodes and weights, mapped onto the physical subinterval.
  for (int i = static_cast<int>(threadIdx.x); i < n_nodes;
       i += static_cast<int>(blockDim.x)) {
    // The endpoints are written from the breakpoints DIRECTLY rather than
    // through the affine map. cos(0) and cos(pi) are exactly +-1, but
    // mid + half and mid - half are not exactly b and a in floating point, and
    // the discrepancy lands precisely at the subinterval interface. Since the
    // reason for choosing Lobatto nodes is that continuity can be imposed on a
    // SHARED node value rather than by interpolating, the two sides of an
    // interface have to agree bit-for-bit or that reason evaporates.
    if (i == 0) {
      nodes[nbase + i] = b;
    } else if (i == n) {
      nodes[nbase + i] = a;
    } else {
      nodes[nbase + i] = mid + half * cheb_node(i, n);
    }
    // dx_physical = half * dx_reference, so the weights scale by the same
    // factor. `half` is positive because breakpoints ascend.
    weights[nbase + i] = half * clenshaw_curtis_weight(i, n);
  }
  __syncthreads();

  // Off-diagonal entries on the reference interval:
  //   D_ij = (c_i / c_j) * (-1)^(i+j) / (x_i - x_j),  i != j
  // with c_0 = c_N = 2 and c_i = 1 otherwise.
  for (int i = static_cast<int>(threadIdx.x); i < n_nodes;
       i += static_cast<int>(blockDim.x)) {
    const Real xi = cheb_node(i, n);
    const Real ci = (i == 0 || i == n) ? Real{2} : Real{1};
    for (int j = 0; j < n_nodes; ++j) {
      if (i == j) continue;
      const Real xj = cheb_node(j, n);
      const Real cj = (j == 0 || j == n) ? Real{2} : Real{1};
      const Real sign = ((i + j) & 1) ? Real{-1} : Real{1};
      diff[mbase + static_cast<std::size_t>(i) * n_nodes + j] =
          (ci / cj) * sign / (xi - xj);
    }
  }
  __syncthreads();

  // Scale the OFF-DIAGONALS from the reference interval to physical units
  // first: d/dx_phys = (1/half) d/dx_ref.
  //
  // Order matters here. Applying the negative-sum trick before scaling makes
  // the reference matrix sum to zero, but scaling every entry afterwards
  // re-rounds each product independently and the sum drifts off zero again.
  // Scaling first and fixing the diagonal last makes the row sum of the matrix
  // callers actually use as close to zero as floating point allows.
  const Real inv_half = Real{1} / half;
  for (int i = static_cast<int>(threadIdx.x); i < n_nodes;
       i += static_cast<int>(blockDim.x)) {
    for (int j = 0; j < n_nodes; ++j) {
      if (i == j) continue;
      diff[mbase + static_cast<std::size_t>(i) * n_nodes + j] *= inv_half;
    }
  }
  __syncthreads();

  // Negative-sum trick: the diagonal is whatever makes the row sum zero.
  //
  // Summed from the far end inward so the smallest-magnitude terms are
  // accumulated first, which is the ordering that loses the least to
  // cancellation across a row whose entries span several orders of magnitude.
  for (int i = static_cast<int>(threadIdx.x); i < n_nodes;
       i += static_cast<int>(blockDim.x)) {
    Real sum = Real{0};
    for (int j = n_nodes; j-- > 0;) {
      if (j == i) continue;
      sum += diff[mbase + static_cast<std::size_t>(i) * n_nodes + j];
    }
    diff[mbase + static_cast<std::size_t>(i) * n_nodes + i] = -sum;
  }
}

}  // namespace

void ChebyshevBasis::resize(int order_in, int n_domains_in) {
  if (order_in <= 0 || n_domains_in <= 0) {
    throw std::invalid_argument{
        "ChebyshevBasis dimensions must be positive"};
  }

  const std::size_t nodes_per_domain =
      static_cast<std::size_t>(order_in) + 1;
  if (nodes_per_domain
      > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::length_error{
        "ChebyshevBasis node count does not fit in int"};
  }
  const std::size_t nn = backend::detail::checked_size_product(
      static_cast<std::size_t>(n_domains_in), nodes_per_domain,
      "ChebyshevBasis node storage overflows size_t");
  const std::size_t diff_size = backend::detail::checked_size_product(
      nn, nodes_per_domain,
      "ChebyshevBasis differentiation storage overflows size_t");

  // Complete every arithmetic check before asking the backend to allocate.
  // In particular, an extreme but non-negative order must fail as a size
  // contract violation even on a host with no visible HIP device.
  (void)backend::detail::checked_buffer_bytes<Real>(nn);
  (void)backend::detail::checked_buffer_bytes<Real>(diff_size);

  nodes = backend::DeviceBuffer<Real>{nn};
  weights = backend::DeviceBuffer<Real>{nn};
  diff = backend::DeviceBuffer<Real>{diff_size};

  order = order_in;
  n_nodes = static_cast<int>(nodes_per_domain);
  n_domains = n_domains_in;
}

void launch_build_chebyshev_basis(const RadialDomains* d_domains,
                                  ChebyshevBasis& out, stream_t stream) {
  const BasisLaunchShape shape = validate_basis_launch(d_domains, out);
  const unsigned threads = shape.n_nodes < 256
                               ? static_cast<unsigned>(shape.n_nodes)
                               : 256u;
  hipLaunchKernelGGL(build_basis_kernel,
                     dim3{static_cast<unsigned>(shape.n_domains)},
                     dim3{threads},
                     0, static_cast<hipStream_t>(stream), d_domains,
                     shape.n_nodes, out.nodes.device_ptr(),
                     out.weights.device_ptr(), out.diff.device_ptr());
  QUASAR_HIP_CHECK(hipGetLastError());
}

}  // namespace quasar::stability
