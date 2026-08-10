#pragma once

// Shared sampling rules for the staggered finite-volume MHD magnetic field.
//
// bx_face(i,j) is the x-low face average of cell (i,j), by_face(i,j) is the
// y-low face average, and bz_cell(i,j) is a cell average. Thermodynamic
// operations need a volume-average magnetic field collocated with the other
// conserved cell averages. A two-face arithmetic mean is only second-order:
// MP5/MP7 grids therefore use polynomial-exact 6th/8th-order quadrature. In
// Cartesian geometry that width is inferred from the working halo. In the
// cylindrical radial direction it is selected explicitly by RadialTables so
// an overpadded MP5 grid still uses its native six-point row; axial stencils
// remain halo-selected Cartesian rows. A Riemann problem still uses the single
// CT normal face shared by its two cells; only tangential background components
// require the matching cell-average-to-face interpolation.
//
// The padded storage has no separate upper face for its outermost high ghost
// cell. Centered quadrature is used wherever its face stencil is available; in
// the outer ghost layers a same-order one-sided polynomial integral is evaluated
// with four-point Gauss quadrature and barycentric interpolation. Thus boundary
// stencils have an explicit closure instead of silently dropping to a point
// sample. Smooth periodic ghost fills are simply the smooth periodic extension
// consumed by that closure.

#include "quasar/core/grid.hpp"
#include "quasar/numerics/mhd_state.hpp"
#include "quasar/numerics/radial_tables.hpp"

namespace quasar::mhd {

#if defined(__HIPCC__)
#define QUASAR_MHD_STAGGER_NOINLINE __attribute__((noinline))
#else
#define QUASAR_MHD_STAGGER_NOINLINE
#endif

namespace detail {

inline constexpr int kAverage2[2] = {1, 1};
inline constexpr int kFaceToCell4[4] = {-1, 13, 13, -1};
inline constexpr int kFaceToCell6[6] = {11, -93, 802, 802, -93, 11};
inline constexpr int kFaceToCell8[8] = {
    -191, 1879, -9531, 68323, 68323, -9531, 1879, -191};
inline constexpr int kCellToFace4[4] = {-1, 7, 7, -1};
inline constexpr int kCellToFace6[6] = {1, -8, 37, 37, -8, 1};
inline constexpr int kCellToFace8[8] = {-3, 29, -139, 533,
                                         533, -139, 29, -3};

// Evaluate sum_k weights[k]*sample(k)/denominator without materializing a
// weighted product, a same-sign partial sum, or a prematurely rounded tiny
// quotient. Retain the integer-coefficient numerator as an exact expansion,
// round that numerator once to a scaled value, then divide once by the common
// rational denominator. Dividing every term first can destroy an exact
// distributed cancellation when the denominator is not a power of two
// (notably the MP5 /60 stencil). The final divide is range-safe, though the
// rounded-numerator-then-divide sequence may differ by one ulp from a fully
// correctly rounded exact rational in a double-rounding edge case.
template <int Count, class Sample>
QUASAR_HOST_DEVICE inline Real scaled_rational_weighted_sum(
    const int (&weights)[Count], Real denominator, const Sample& sample) {
  numerics::ScaledProductQuotientAccumulator<2 * Count> numerator_sum;
  for (int k = 0; k < Count; ++k) {
    numerics::append_scaled_exact_product(
        numerator_sum, static_cast<Real>(weights[k]), sample(k));
  }
  const numerics::ScaledValue numerator =
      numerics::finish_scaled_exact_product_sum_to_value(numerator_sum);
  return numerics::scaled_value_divide(numerator, denominator);
}

// Radius-dependent rows are stored directly as binary64 coefficients instead
// of small integer rationals.  Retain the same range-safe reduction and exact
// constant-field short circuit used by the transverse quadrature kernels.
template <class Sample>
QUASAR_HOST_DEVICE inline Real scaled_radial_weighted_sum(
    const Real* weights, int count, const Sample& sample) {
  const Real reference = sample(count / 2);
  bool constant = true;
  numerics::ScaledProductQuotientAccumulator<8> sum;
  for (int k = 0; k < count; ++k) {
    const Real value = sample(k);
    constant = constant && value == reference;
    numerics::append_scaled_product_quotient(
        sum, weights[k], value, Real{1}, Real{1});
  }
  return constant ? reference
                  : numerics::finish_scaled_product_quotient_sum(sum);
}

QUASAR_HOST_DEVICE inline int collocation_width(int extent, int nghost) {
  const int requested = (nghost >= 4) ? 8 : (nghost >= 3) ? 6 :
                        (nghost >= 2) ? 4 : 2;
  if (requested >= 8 && extent >= 8) return 8;
  if (requested >= 6 && extent >= 6) return 6;
  if (requested >= 4 && extent >= 4) return 4;
  return extent >= 2 ? 2 : 1;
}

QUASAR_HOST_DEVICE inline Real barycentric_weight(int width, int k) {
  // Common-scale barycentric weights (-1)^k C(width-1,k) on integer nodes.
  if (width == 8) {
    constexpr int w[8] = {1, -7, 21, -35, 35, -21, 7, -1};
    return static_cast<Real>(w[k]);
  }
  if (width == 6) {
    constexpr int w[6] = {1, -5, 10, -10, 5, -1};
    return static_cast<Real>(w[k]);
  }
  if (width == 4) {
    constexpr int w[4] = {1, -3, 3, -1};
    return static_cast<Real>(w[k]);
  }
  return k == 0 ? Real{1} : Real{-1};
}

template <class T>
QUASAR_HOST_DEVICE inline Real axis_sample(const Grid2D& g, const T* face,
                                            int axis, int q, int fixed) {
  return axis == 0
      ? static_cast<Real>(face[g.index(q, fixed)])
      : static_cast<Real>(face[g.index(fixed, q)]);
}

// Integrate the degree-(width-1) interpolant through equally-spaced face
// samples over [cell,cell+1]. This is used only for outer-ghost one-sided
// closures; the common centered path below has exact rational coefficients.
template <class T>
QUASAR_HOST_DEVICE QUASAR_MHD_STAGGER_NOINLINE inline Real one_sided_face_integral(
    const Grid2D& g, const T* face, int axis, int fixed, int start,
    int width, int cell, bool radial_weighted = false,
    bool preserve_constant = false) {
  constexpr Real xq[4] = {
      Real{0.0694318442029737124}, Real{0.3300094782075718676},
      Real{0.6699905217924281324}, Real{0.9305681557970262876}};
  constexpr Real wq[4] = {
      Real{0.1739274225687269287}, Real{0.3260725774312730713},
      Real{0.3260725774312730713}, Real{0.1739274225687269287}};
  const int relative_cell = cell - start;
  if (preserve_constant) {
    const Real reference = axis_sample(g, face, axis, start, fixed);
    bool constant = true;
    for (int k = 1; k < width; ++k) {
      constant = constant &&
          axis_sample(g, face, axis, start + k, fixed) == reference;
    }
    if (constant) return reference;
  }
  Real radial_normalization = Real{1};
  Real radial_weight[4] = {wq[0], wq[1], wq[2], wq[3]};
  if (radial_weighted) {
    radial_normalization = Real{0};
    const Real edge_rho = g.r_at_edge(cell) / g.dx();
    for (int q = 0; q < 4; ++q) {
      radial_weight[q] = wq[q] * fabs(edge_rho + xq[q]);
      radial_normalization += radial_weight[q];
    }
  }
  // Flatten all four barycentric interpolants and Gauss weights into one
  // reduction.  Computing numerator/denominator at each point first can
  // overflow the numerator, while summing the four already-rounded point
  // values can overflow before opposite-sign extrapolation terms cancel in the
  // integral.  At most 4*8 terms are active.
  numerics::ScaledProductQuotientAccumulator<32> integral;
  for (int q = 0; q < 4; ++q) {
    const Real x = static_cast<Real>(relative_cell) + xq[q];
    numerics::ScaledProductQuotientAccumulator<8> denominator_sum;
    for (int k = 0; k < width; ++k) {
      numerics::append_scaled_product_quotient(
          denominator_sum, barycentric_weight(width, k), Real{1},
          x - static_cast<Real>(k), Real{1});
    }
    const Real denominator =
        numerics::finish_scaled_product_quotient_sum(denominator_sum);
    for (int k = 0; k < width; ++k) {
      const Real weighted_basis =
          (radial_weight[q] / radial_normalization) *
          barycentric_weight(width, k);
      numerics::append_scaled_product_quotient(
          integral, weighted_basis,
          axis_sample(g, face, axis, start + k, fixed),
          x - static_cast<Real>(k), denominator);
    }
  }
  return numerics::finish_scaled_product_quotient_sum(integral);
}

template <class T>
QUASAR_HOST_DEVICE QUASAR_MHD_STAGGER_NOINLINE inline Real face_samples_to_cell_average(
    const Grid2D& g, const T* face, int axis, int cell, int fixed,
    numerics::RadialTablesView radial_tables = {}) {
  const int n = axis == 0 ? g.nx : g.ny;
  const int extent = n + 2 * g.nghost;
  const bool radial_weighted = axis == 0 && radial_tables.active != 0;
  const int width = radial_weighted
      ? radial_tables.collocation_width
      : collocation_width(extent, g.nghost);
  const int lo = -g.nghost;
  const int hi = n + g.nghost - 1;
  if (width == 1) return axis_sample(g, face, axis, cell, fixed);

  const int left = width / 2 - 1;
  const int centered_start = cell - left;
  if (centered_start >= lo && centered_start + width - 1 <= hi) {
    if (axis == 0 && radial_tables.active != 0 &&
        radial_tables.contains(cell) &&
        radial_tables.collocation_width == width) {
      return scaled_radial_weighted_sum(
          radial_tables.r4_row(cell), width, [&](int k) {
            return axis_sample(g, face, axis, centered_start + k, fixed);
          });
    }
    if (width == 8) {
      // Exact through degree seven. The third coefficient is -9531/120960;
      // -9504 would not even preserve constants.
      return scaled_rational_weighted_sum(
          kFaceToCell8, Real{120960}, [&](int k) {
            return axis_sample(g, face, axis, centered_start + k, fixed);
          });
    }
    if (width == 6) {
      return scaled_rational_weighted_sum(
          kFaceToCell6, Real{1440}, [&](int k) {
            return axis_sample(g, face, axis, centered_start + k, fixed);
          });
    }
    if (width == 4) {
      return scaled_rational_weighted_sum(
          kFaceToCell4, Real{24}, [&](int k) {
            return axis_sample(g, face, axis, centered_start + k, fixed);
          });
    }
    return scaled_rational_weighted_sum(
        kAverage2, Real{2}, [&](int k) {
          return axis_sample(g, face, axis, centered_start + k, fixed);
        });
  }

  int start = centered_start;
  if (start < lo) start = lo;
  if (start + width - 1 > hi) start = hi - width + 1;
  return one_sided_face_integral(
      g, face, axis, fixed, start, width, cell,
      radial_weighted, radial_tables.active != 0);
}

// Bounded first-order Godunov collocation used only by the low-order positivity
// retry. Interior cells average their two bounding CT faces. The outermost high
// ghost cell has no stored upper face, so use its sole face sample; the
// first-order physical-interface stencils never need that closure, but keeping
// it defined prevents an out-of-bounds read for stand-alone padded fields.
template <class T>
QUASAR_HOST_DEVICE inline Real low_order_face_to_cell_average(
    const Grid2D& g, const T* face, int axis, int cell, int fixed) {
  const int n = axis == 0 ? g.nx : g.ny;
  const int hi = n + g.nghost - 1;
  const Real lo_face = axis_sample(g, face, axis, cell, fixed);
  if (cell >= hi) return lo_face;
  return scaled_rational_weighted_sum(
      kAverage2, Real{2}, [&](int k) {
        return k == 0 ? lo_face
                      : axis_sample(g, face, axis, cell + 1, fixed);
      });
}

}  // namespace detail

template <class T>
QUASAR_HOST_DEVICE QUASAR_MHD_STAGGER_NOINLINE inline Real cell_bx(
                                       const Grid2D& g, const T* bx_face,
                                       int i, int j,
                                       numerics::RadialTablesView radial_tables,
                                       int collocation_order = 0) {
  (void)radial_tables;
  if (collocation_order == 1) {
    return detail::low_order_face_to_cell_average(g, bx_face, 0, i, j);
  }
  return detail::face_samples_to_cell_average(
      g, bx_face, 0, i, j, radial_tables);
}

template <class T>
QUASAR_HOST_DEVICE QUASAR_MHD_STAGGER_NOINLINE inline Real cell_by(
                                       const Grid2D& g, const T* by_face,
                                       int i, int j,
                                       numerics::RadialTablesView radial_tables,
                                       int collocation_order = 0) {
  if (collocation_order == 1) {
    return detail::low_order_face_to_cell_average(g, by_face, 1, j, i);
  }
  return detail::face_samples_to_cell_average(
      g, by_face, 1, j, i, radial_tables);
}

// Compatibility overloads deliberately select an inactive view.  Cartesian
// callers therefore retain the exact pre-table arithmetic until they opt into
// the cylindrical overload above.
template <class T>
QUASAR_HOST_DEVICE QUASAR_MHD_STAGGER_NOINLINE inline Real cell_bx(
                                       const Grid2D& g, const T* bx_face,
                                       int i, int j,
                                       int collocation_order = 0) {
  return cell_bx(g, bx_face, i, j, numerics::RadialTablesView{},
                 collocation_order);
}

template <class T>
QUASAR_HOST_DEVICE QUASAR_MHD_STAGGER_NOINLINE inline Real cell_by(
                                       const Grid2D& g, const T* by_face,
                                       int i, int j,
                                       int collocation_order = 0) {
  return cell_by(g, by_face, i, j, numerics::RadialTablesView{},
                 collocation_order);
}

template <class T>
QUASAR_HOST_DEVICE inline Real normal_face_b(const Grid2D& g, int dir,
                                             const T* bx_face,
                                             const T* by_face,
                                             int i, int j) {
  const T* normal = (dir == 0) ? bx_face : by_face;
  return static_cast<Real>(normal[g.index(i, j)]);
}

template <class T>
QUASAR_HOST_DEVICE inline numerics::MhdState load_cell_state(
    const Grid2D& g, const T* rho, const T* mx, const T* my, const T* mz,
    const T* energy, const T* bx_face, const T* by_face, const T* bz_cell,
    int i, int j, numerics::RadialTablesView radial_tables,
    int collocation_order = 0) {
  const std::size_t k = g.index(i, j);
  numerics::MhdState s;
  s.rho = static_cast<Real>(rho[k]);
  s.mx = static_cast<Real>(mx[k]);
  s.my = static_cast<Real>(my[k]);
  s.mz = static_cast<Real>(mz[k]);
  s.energy = static_cast<Real>(energy[k]);
  s.bx = cell_bx(g, bx_face, i, j, radial_tables, collocation_order);
  s.by = cell_by(g, by_face, i, j, radial_tables, collocation_order);
  s.bz = static_cast<Real>(bz_cell[k]);
  return s;
}

template <class T>
QUASAR_HOST_DEVICE inline numerics::MhdState load_cell_state(
    const Grid2D& g, const T* rho, const T* mx, const T* my, const T* mz,
    const T* energy, const T* bx_face, const T* by_face, const T* bz_cell,
    int i, int j, int collocation_order = 0) {
  return load_cell_state(
      g, rho, mx, my, mz, energy, bx_face, by_face, bz_cell, i, j,
      numerics::RadialTablesView{}, collocation_order);
}

// Cell-centred B0 for the split formulation.
template <class T>
QUASAR_HOST_DEVICE inline numerics::MhdBackground load_cell_background(
    const Grid2D& g, const T* b0x_face, const T* b0y_face,
    const T* b0z_cell, int i, int j,
    numerics::RadialTablesView radial_tables,
    int collocation_order = 0) {
  numerics::MhdBackground b0;
  b0.b0x = cell_bx(g, b0x_face, i, j, radial_tables, collocation_order);
  b0.b0y = cell_by(g, b0y_face, i, j, radial_tables, collocation_order);
  b0.b0z = static_cast<Real>(b0z_cell[g.index(i, j)]);
  return b0;
}

template <class T>
QUASAR_HOST_DEVICE inline numerics::MhdBackground load_cell_background(
    const Grid2D& g, const T* b0x_face, const T* b0y_face,
    const T* b0z_cell, int i, int j, int collocation_order = 0) {
  return load_cell_background(
      g, b0x_face, b0y_face, b0z_cell, i, j,
      numerics::RadialTablesView{}, collocation_order);
}

// Interpolate cell averages to the face between cells face-1 and face. The
// centered formulas are exact through degree 3/5/7 for 4/6/8 samples.
template <class CellSample>
QUASAR_HOST_DEVICE inline Real cell_averages_to_face(
    const Grid2D& g, int axis, int face, CellSample sample,
    numerics::RadialTablesView radial_tables,
    int collocation_order = 0) {
  (void)radial_tables;
  const int n = axis == 0 ? g.nx : g.ny;
  const int lo = -g.nghost;
  const int hi = n + g.nghost - 1;
  if (collocation_order == 1) {
    int left = face - 1;
    int right = face;
    if (left < lo) left = lo;
    if (right > hi) right = hi;
    return detail::scaled_rational_weighted_sum(
        detail::kAverage2, Real{2}, [&](int k) {
          return sample(k == 0 ? left : right);
        });
  }
  const int width = axis == 0 && radial_tables.active != 0
      ? radial_tables.collocation_width
      : detail::collocation_width(n + 2 * g.nghost, g.nghost);
  const int start = face - width / 2;
  if (axis == 0 && radial_tables.active != 0 &&
      radial_tables.contains(face) &&
      radial_tables.collocation_width == width &&
      start >= lo && start + width - 1 <= hi) {
    return detail::scaled_radial_weighted_sum(
        radial_tables.r5_row(face), width,
        [&](int k) { return sample(start + k); });
  }
  if (width == 8 && start >= lo && start + 7 <= hi) {
    return detail::scaled_rational_weighted_sum(
        detail::kCellToFace8, Real{840}, [&](int k) {
          return sample(start + k);
        });
  }
  if (width == 6 && start >= lo && start + 5 <= hi) {
    return detail::scaled_rational_weighted_sum(
        detail::kCellToFace6, Real{60}, [&](int k) {
          return sample(start + k);
        });
  }
  if (width == 4 && start >= lo && start + 3 <= hi) {
    return detail::scaled_rational_weighted_sum(
        detail::kCellToFace4, Real{12}, [&](int k) {
          return sample(start + k);
        });
  }
  // This path is used only by deliberately under-padded stand-alone fields;
  // solver grids always have width/2 ghost cells at a physical interface.
  int left = face - 1;
  int right = face;
  if (left < lo) left = lo;
  if (right > hi) right = hi;
  return detail::scaled_rational_weighted_sum(
      detail::kAverage2, Real{2}, [&](int k) {
        return sample(k == 0 ? left : right);
      });
}

template <class CellSample>
QUASAR_HOST_DEVICE inline Real cell_averages_to_face(
    const Grid2D& g, int axis, int face, CellSample sample,
    int collocation_order = 0) {
  return cell_averages_to_face(
      g, axis, face, sample, numerics::RadialTablesView{}, collocation_order);
}

// B0 collocated with a directional Riemann face. The normal component is the
// exact single staggered CT face. Tangential components use the order-matched
// cell-average-to-face interpolation above.
template <class T>
QUASAR_HOST_DEVICE QUASAR_MHD_STAGGER_NOINLINE inline numerics::MhdBackground
load_interface_background(
    const Grid2D& g, int dir, const T* b0x_face, const T* b0y_face,
    const T* b0z_cell, int i, int j,
    numerics::RadialTablesView radial_tables,
    int collocation_order = 0) {
  numerics::MhdBackground b0;
  if (dir == 0) {
    b0.b0x = static_cast<Real>(b0x_face[g.index(i, j)]);
    b0.b0y = cell_averages_to_face(g, 0, i, [&](int ii) {
      return cell_by(g, b0y_face, ii, j, radial_tables, collocation_order);
    }, radial_tables, collocation_order);
    b0.b0z = cell_averages_to_face(g, 0, i, [&](int ii) {
      return static_cast<Real>(b0z_cell[g.index(ii, j)]);
    }, radial_tables, collocation_order);
  } else {
    b0.b0x = cell_averages_to_face(g, 1, j, [&](int jj) {
      return cell_bx(g, b0x_face, i, jj, radial_tables, collocation_order);
    }, radial_tables, collocation_order);
    b0.b0y = static_cast<Real>(b0y_face[g.index(i, j)]);
    b0.b0z = cell_averages_to_face(g, 1, j, [&](int jj) {
      return static_cast<Real>(b0z_cell[g.index(i, jj)]);
    }, radial_tables, collocation_order);
  }
  return b0;
}

template <class T>
QUASAR_HOST_DEVICE QUASAR_MHD_STAGGER_NOINLINE inline numerics::MhdBackground
load_interface_background(
    const Grid2D& g, int dir, const T* b0x_face, const T* b0y_face,
    const T* b0z_cell, int i, int j, int collocation_order = 0) {
  return load_interface_background(
      g, dir, b0x_face, b0y_face, b0z_cell, i, j,
      numerics::RadialTablesView{}, collocation_order);
}

#undef QUASAR_MHD_STAGGER_NOINLINE

}  // namespace quasar::mhd
