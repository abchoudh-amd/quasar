#pragma once

// Indexing contract for the Chebyshev x Fourier ideal-MHD displacement basis.
//
// Radially, every Chebyshev subdomain stores order + 1 Lobatto nodes in
// descending physical order.  Adjacent domains ordinarily share their common
// endpoint.  At a rational interface, however, Newcomb's small and large
// solutions permit independent one-sided limits for the RESONANT harmonic.
// Radial topology is therefore harmonic-specific: an interface tagged with m
// is split only in the m block and remains merged in every other block.
//
// Since harmonic blocks can contain different radial counts, they are packed
// using prefix offsets.  Within a harmonic block, radial point is the slow
// index, displacement component the middle index, and real/imaginary
// quadrature the fast index.  Dense matrices use the column-major convention
// required by hipSOLVER.

#include "quasar/physics/stability/kernels.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace quasar::stability {

enum class DisplacementComponent : int {
  psi = 0,
  theta = 1,
  phi = 2,
};

enum class FourierQuadrature : int {
  cosine = 0,
  sine = 1,
};

class SpectralDofLayout {
 public:
  static constexpr int kComponentCount = 3;
  static constexpr int kQuadratureCount = 2;

  // Backward-compatible all-regular layout: every duplicated interface is
  // merged for every harmonic.
  SpectralDofLayout(int n_domains, int chebyshev_order, int m_max,
                    int n_theta)
      : n_domains_{n_domains}, order_{chebyshev_order}, m_max_{m_max},
        n_theta_{n_theta} {
    initialize(nullptr);
  }

  // Harmonic-aware layout using the resonance provenance attached to each
  // RadialDomains breakpoint.
  SpectralDofLayout(const RadialDomains& domains, int chebyshev_order,
                    int m_max, int n_theta)
      : n_domains_{domains.n_domains}, order_{chebyshev_order},
        m_max_{m_max}, n_theta_{n_theta} {
    validate_domains(domains);
    initialize(&domains);
  }

  [[nodiscard]] int n_domains() const noexcept { return n_domains_; }
  [[nodiscard]] int chebyshev_order() const noexcept { return order_; }
  [[nodiscard]] int m_max() const noexcept { return m_max_; }
  [[nodiscard]] int n_theta() const noexcept { return n_theta_; }

  // Count for the all-regular topology, retained for callers constructing the
  // legacy all-regular layout.  Harmonic-aware callers should use n_radial(m).
  [[nodiscard]] std::size_t n_radial() const noexcept {
    return regular_n_radial_;
  }

  [[nodiscard]] std::size_t n_radial(int m) const {
    return harmonic_radial_counts_[harmonic_slot(m)];
  }

  [[nodiscard]] std::size_t n_harmonics() const noexcept {
    return n_harmonics_;
  }
  [[nodiscard]] std::size_t complex_dof_count() const noexcept {
    return complex_dofs_;
  }
  [[nodiscard]] std::size_t dof_count() const noexcept { return real_dofs_; }

  // Existing ChebyshevBasis convention: local node 0 is the upper endpoint,
  // local node `order` is the lower endpoint.  This overload is unambiguous
  // only for an all-regular layout.
  [[nodiscard]] std::size_t global_radial(int domain,
                                           int local_node) const {
    check_local_node(domain, local_node);
    if (has_any_split_) {
      throw std::logic_error{
          "SpectralDofLayout: harmonic m is required for a rational topology"};
    }
    return regular_global_radial(domain, local_node);
  }

  // Harmonic-specific local-to-global radial map.  At interface `domain`, the
  // upper endpoint of domain-1 precedes the independent lower endpoint of
  // domain when that interface is resonant for m.
  [[nodiscard]] std::size_t global_radial(int domain, int local_node,
                                           int m) const {
    check_local_node(domain, local_node);
    const std::size_t slot = harmonic_slot(m);
    std::size_t radial = regular_global_radial(domain, local_node);
    for (int interface = 1; interface <= domain; ++interface) {
      if (split_interfaces_[split_index(slot, interface)] != 0) {
        radial = checked_add(radial, std::size_t{1});
      }
    }
    return radial;
  }

  [[nodiscard]] bool interface_is_split(int interface, int m) const {
    if (interface <= 0 || interface >= n_domains_) {
      throw std::out_of_range{
          "SpectralDofLayout: interface out of range"};
    }
    return split_interfaces_[split_index(harmonic_slot(m), interface)] != 0;
  }

  [[nodiscard]] std::size_t harmonic_slot(int m) const {
    if (m < -m_max_ || m > m_max_) {
      throw std::out_of_range{"SpectralDofLayout: harmonic out of range"};
    }
    return static_cast<std::size_t>(m + m_max_);
  }

  [[nodiscard]] std::size_t complex_dof(
      std::size_t radial, int m, DisplacementComponent component) const {
    const std::size_t slot = harmonic_slot(m);
    check_radial(radial, slot);
    const int component_index = static_cast<int>(component);
    if (component_index < 0 || component_index >= kComponentCount) {
      throw std::out_of_range{
          "SpectralDofLayout: displacement component out of range"};
    }
    const std::size_t harmonic_radial = checked_add(
        harmonic_radial_offsets_[slot], radial);
    return checked_add(
        static_cast<std::size_t>(component_index),
        checked_multiply(static_cast<std::size_t>(kComponentCount),
                         harmonic_radial));
  }

  [[nodiscard]] std::size_t dof(
      std::size_t radial, int m, DisplacementComponent component,
      FourierQuadrature quadrature) const {
    const int quadrature_index = static_cast<int>(quadrature);
    if (quadrature_index < 0 || quadrature_index >= kQuadratureCount) {
      throw std::out_of_range{
          "SpectralDofLayout: Fourier quadrature out of range"};
    }
    return checked_add(
        static_cast<std::size_t>(quadrature_index),
        checked_multiply(static_cast<std::size_t>(kQuadratureCount),
                         complex_dof(radial, m, component)));
  }

  [[nodiscard]] std::size_t matrix_index(std::size_t row,
                                          std::size_t column) const {
    if (row >= real_dofs_ || column >= real_dofs_) {
      throw std::out_of_range{"SpectralDofLayout: matrix index out of range"};
    }
    return checked_add(row, checked_multiply(real_dofs_, column));
  }

  // Eigenvectors are stored as a square column-major matrix with one column
  // per generalized eigenpair.
  [[nodiscard]] std::size_t eigenfunction_index(
      std::size_t eigenpair, std::size_t radial, int m,
      DisplacementComponent component, FourierQuadrature quadrature) const {
    if (eigenpair >= real_dofs_) {
      throw std::out_of_range{
          "SpectralDofLayout: eigenpair out of range"};
    }
    return checked_add(dof(radial, m, component, quadrature),
                       checked_multiply(real_dofs_, eigenpair));
  }

 private:
  static std::size_t checked_multiply(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
      throw std::length_error{"SpectralDofLayout: size overflow"};
    }
    return a * b;
  }

  static std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
      throw std::length_error{"SpectralDofLayout: size overflow"};
    }
    return a + b;
  }

  void validate_common() const {
    if (n_domains_ <= 0) {
      throw std::invalid_argument{
          "SpectralDofLayout: n_domains must be positive"};
    }
    if (order_ <= 0) {
      throw std::invalid_argument{
          "SpectralDofLayout: chebyshev_order must be positive"};
    }
    if (m_max_ < 0) {
      throw std::invalid_argument{
          "SpectralDofLayout: m_max must be non-negative"};
    }
    if (n_theta_ <= 0) {
      throw std::invalid_argument{
          "SpectralDofLayout: n_theta must be positive"};
    }

    const long long minimum_theta = 4LL * m_max_ + 1LL;
    if (static_cast<long long>(n_theta_) < minimum_theta) {
      throw std::invalid_argument{
          "SpectralDofLayout: n_theta must be at least 4*m_max + 1"};
    }
  }

  void validate_domains(const RadialDomains& domains) const {
    if (domains.n_domains <= 0
        || domains.n_domains > RadialDomains::kMaxDomains) {
      throw std::invalid_argument{
          "SpectralDofLayout: invalid RadialDomains::n_domains"};
    }
    if (domains.resonance_count < 0
        || domains.resonance_count > RadialDomains::kMaxResonanceTags) {
      throw std::invalid_argument{
          "SpectralDofLayout: invalid RadialDomains resonance count"};
    }

    int previous = 0;
    if (domains.resonance_offsets[0] != 0) {
      throw std::invalid_argument{
          "SpectralDofLayout: resonance offsets must begin at zero"};
    }
    for (int breakpoint = 0; breakpoint <= domains.n_domains; ++breakpoint) {
      const int next = domains.resonance_offsets[breakpoint + 1];
      if (next < previous || next > domains.resonance_count) {
        throw std::invalid_argument{
            "SpectralDofLayout: invalid resonance offset range"};
      }
      previous = next;
    }
    if (previous != domains.resonance_count) {
      throw std::invalid_argument{
          "SpectralDofLayout: resonance offsets do not cover all tags"};
    }
  }

  void initialize(const RadialDomains* domains) {
    validate_common();

    regular_n_radial_ = checked_add(
        checked_multiply(static_cast<std::size_t>(n_domains_),
                         static_cast<std::size_t>(order_)),
        std::size_t{1});
    n_harmonics_ = checked_add(
        checked_multiply(std::size_t{2}, static_cast<std::size_t>(m_max_)),
        std::size_t{1});

    const std::size_t interfaces_per_harmonic =
        checked_add(static_cast<std::size_t>(n_domains_), std::size_t{1});
    split_interfaces_.assign(
        checked_multiply(n_harmonics_, interfaces_per_harmonic), 0);
    harmonic_radial_counts_.assign(n_harmonics_, regular_n_radial_);
    harmonic_radial_offsets_.assign(
        checked_add(n_harmonics_, std::size_t{1}), std::size_t{0});

    for (std::size_t slot = 0; slot < n_harmonics_; ++slot) {
      const int m = static_cast<int>(slot) - m_max_;
      std::size_t split_count = 0;
      if (domains != nullptr) {
        for (int interface = 1; interface < n_domains_; ++interface) {
          const int begin = domains->resonance_offsets[interface];
          const int end = domains->resonance_offsets[interface + 1];
          for (int tag = begin; tag < end; ++tag) {
            if (domains->resonant_m[tag] == m) {
              split_interfaces_[split_index(slot, interface)] = 1;
              ++split_count;
              has_any_split_ = true;
              break;
            }
          }
        }
      }
      harmonic_radial_counts_[slot] =
          checked_add(regular_n_radial_, split_count);
      harmonic_radial_offsets_[slot + 1] = checked_add(
          harmonic_radial_offsets_[slot], harmonic_radial_counts_[slot]);
    }

    complex_dofs_ = checked_multiply(
        harmonic_radial_offsets_.back(),
        static_cast<std::size_t>(kComponentCount));
    real_dofs_ = checked_multiply(
        complex_dofs_, static_cast<std::size_t>(kQuadratureCount));
  }

  [[nodiscard]] std::size_t split_index(std::size_t harmonic,
                                         int interface) const {
    return checked_add(
        static_cast<std::size_t>(interface),
        checked_multiply(harmonic,
                         static_cast<std::size_t>(n_domains_ + 1)));
  }

  void check_local_node(int domain, int local_node) const {
    if (domain < 0 || domain >= n_domains_) {
      throw std::out_of_range{"SpectralDofLayout: domain out of range"};
    }
    if (local_node < 0 || local_node > order_) {
      throw std::out_of_range{
          "SpectralDofLayout: local Chebyshev node out of range"};
    }
  }

  [[nodiscard]] std::size_t regular_global_radial(int domain,
                                                   int local_node) const {
    return checked_add(
        checked_multiply(static_cast<std::size_t>(domain),
                         static_cast<std::size_t>(order_)),
        static_cast<std::size_t>(order_ - local_node));
  }

  void check_radial(std::size_t radial, std::size_t harmonic) const {
    if (radial >= harmonic_radial_counts_[harmonic]) {
      throw std::out_of_range{"SpectralDofLayout: radial index out of range"};
    }
  }

  int n_domains_{0};
  int order_{0};
  int m_max_{0};
  int n_theta_{0};
  std::size_t regular_n_radial_{0};
  std::size_t n_harmonics_{0};
  std::size_t complex_dofs_{0};
  std::size_t real_dofs_{0};
  bool has_any_split_{false};
  std::vector<unsigned char> split_interfaces_{};
  std::vector<std::size_t> harmonic_radial_counts_{};
  std::vector<std::size_t> harmonic_radial_offsets_{};
};

}  // namespace quasar::stability
