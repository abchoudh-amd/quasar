#include "quasar/numerics/filter.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/boundary/boundary_condition.hpp"

#include <hip/hip_runtime.h>

extern "C" void launch_pic_filter_binomial(const quasar::Grid2D&,
                                           quasar::JField2D<double>&,
                                           int, hipStream_t);
extern "C" void launch_pic_filter_compensated(const quasar::Grid2D&,
                                              quasar::JField2D<double>&,
                                              int, hipStream_t);

namespace quasar::numerics {

void BinomialFilter::apply(JField2D<Real>& current, const boundary::BoundarySpec&) const {
  ::launch_pic_filter_binomial(current.grid, current, n_passes_, nullptr);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void CompensatedBinomialFilter::apply(JField2D<Real>& current, const boundary::BoundarySpec&) const {
  ::launch_pic_filter_compensated(current.grid, current, n_passes_, nullptr);
  QUASAR_HIP_CHECK(::hipGetLastError());
}

void FilterPipeline::apply(JField2D<Real>& current, const boundary::BoundarySpec& bc) const {
  for (const auto& filter : filters_) {
    filter->apply(current, bc);
  }
}

QUASAR_REGISTER_CURRENT_FILTER("binomial", BinomialFilter)
QUASAR_REGISTER_CURRENT_FILTER("compensated_binomial", CompensatedBinomialFilter)

}  // namespace quasar::numerics
