#pragma once

#include "quasar/backend/memory.hpp"
#include "quasar/core/registry.hpp"
#include "quasar/core/yee_field.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace quasar::boundary {
struct BoundarySpec;
}

namespace quasar::numerics {

class ICurrentFilter {
 public:
  virtual ~ICurrentFilter() = default;
  virtual void apply(JField2D<Real>& current, const boundary::BoundarySpec& bc,
                     bool cylindrical = false) const = 0;
  // Sets the number of smoothing passes. Registry-created filters are default
  // constructed (one pass); the deck loader calls this to apply the configured
  // pass count without needing a type-specific constructor.
  virtual void set_passes(int n_passes) = 0;
};

class BinomialFilter final : public ICurrentFilter {
 public:
  explicit BinomialFilter(int n_passes = 1) { set_passes(n_passes); }
  void apply(JField2D<Real>& current, const boundary::BoundarySpec& bc,
             bool cylindrical = false) const override;
  void set_passes(int n_passes) override {
    if (n_passes < 1) {
      throw std::invalid_argument{"BinomialFilter: passes must be >= 1"};
    }
    n_passes_ = n_passes;
  }
  int passes() const noexcept { return n_passes_; }

 private:
  int n_passes_{1};
  // Ping-pong scratch, allocated once on first use and reused across steps so the
  // per-step filter does not hipMalloc a full grid every call.
  mutable backend::DeviceBuffer<Real> scratch_{};
};

class CompensatedBinomialFilter final : public ICurrentFilter {
 public:
  explicit CompensatedBinomialFilter(int n_passes = 1) { set_passes(n_passes); }
  void apply(JField2D<Real>& current, const boundary::BoundarySpec& bc,
             bool cylindrical = false) const override;
  void set_passes(int n_passes) override {
    if (n_passes < 1) {
      throw std::invalid_argument{"CompensatedBinomialFilter: passes must be >= 1"};
    }
    n_passes_ = n_passes;
  }
  int passes() const noexcept { return n_passes_; }

 private:
  int n_passes_{1};
  mutable backend::DeviceBuffer<Real> scratch_{};
};

class FilterPipeline {
 public:
  void add(std::unique_ptr<ICurrentFilter> filter) { filters_.push_back(std::move(filter)); }
  bool empty() const noexcept { return filters_.empty(); }
  std::size_t size() const noexcept { return filters_.size(); }
  void apply(JField2D<Real>& current, const boundary::BoundarySpec& bc,
             bool cylindrical = false) const;

 private:
  std::vector<std::unique_ptr<ICurrentFilter>> filters_{};
};

}  // namespace quasar::numerics

#define QUASAR_REGISTER_CURRENT_FILTER(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::numerics::ICurrentFilter, Name, Class)
