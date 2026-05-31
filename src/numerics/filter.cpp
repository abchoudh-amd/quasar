#include "quasar/numerics/filter.hpp"

#include "quasar/backend/device.hpp"
#include "quasar/boundary/boundary_condition.hpp"

#include "quasar/physics/pic/kernels.hpp"

namespace quasar::numerics {

namespace {
// Grow the scratch buffer to at least `n` doubles, reusing it across calls.
Real* ensure_scratch(backend::DeviceBuffer<Real>& scratch, std::size_t n) {
  if (scratch.size() < n) {
    scratch = backend::DeviceBuffer<Real>{n};
  }
  return scratch.device_ptr();
}
}  // namespace

void BinomialFilter::apply(JField2D<Real>& current, const boundary::BoundarySpec&) const {
  Real* scratch = ensure_scratch(scratch_, current.grid.storage_size());
  ::launch_pic_filter_binomial(current.grid, current, scratch, n_passes_, nullptr);
}

void CompensatedBinomialFilter::apply(JField2D<Real>& current, const boundary::BoundarySpec&) const {
  Real* scratch = ensure_scratch(scratch_, current.grid.storage_size());
  ::launch_pic_filter_compensated(current.grid, current, scratch, n_passes_, nullptr);
}

void FilterPipeline::apply(JField2D<Real>& current, const boundary::BoundarySpec& bc) const {
  for (const auto& filter : filters_) {
    filter->apply(current, bc);
  }
}

QUASAR_REGISTER_CURRENT_FILTER("binomial", BinomialFilter)
QUASAR_REGISTER_CURRENT_FILTER("compensated_binomial", CompensatedBinomialFilter)

}  // namespace quasar::numerics
