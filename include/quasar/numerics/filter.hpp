#pragma once

#include "quasar/core/registry.hpp"
#include "quasar/core/yee_field.hpp"

#include <memory>
#include <string>
#include <vector>

namespace quasar::boundary {
struct BoundarySpec;
}

namespace quasar::numerics {

class ICurrentFilter {
 public:
  virtual ~ICurrentFilter() = default;
  virtual void apply(JField2D<Real>& current, const boundary::BoundarySpec& bc) const = 0;
};

class BinomialFilter final : public ICurrentFilter {
 public:
  explicit BinomialFilter(int n_passes = 1) : n_passes_{n_passes} {}
  void apply(JField2D<Real>& current, const boundary::BoundarySpec& bc) const override;
  int passes() const noexcept { return n_passes_; }

 private:
  int n_passes_{1};
};

class CompensatedBinomialFilter final : public ICurrentFilter {
 public:
  explicit CompensatedBinomialFilter(int n_passes = 1) : n_passes_{n_passes} {}
  void apply(JField2D<Real>& current, const boundary::BoundarySpec& bc) const override;
  int passes() const noexcept { return n_passes_; }

 private:
  int n_passes_{1};
};

class FilterPipeline {
 public:
  void add(std::unique_ptr<ICurrentFilter> filter) { filters_.push_back(std::move(filter)); }
  bool empty() const noexcept { return filters_.empty(); }
  std::size_t size() const noexcept { return filters_.size(); }
  void apply(JField2D<Real>& current, const boundary::BoundarySpec& bc) const;

 private:
  std::vector<std::unique_ptr<ICurrentFilter>> filters_{};
};

}  // namespace quasar::numerics

#define QUASAR_REGISTER_CURRENT_FILTER(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::numerics::ICurrentFilter, Name, Class)
