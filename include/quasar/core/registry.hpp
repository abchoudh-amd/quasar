#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace quasar {

template <class Base>
class Registry {
 public:
  using Factory = std::function<std::unique_ptr<Base>()>;

  static Registry& instance() {
    static Registry singleton;
    return singleton;
  }

  // Returns true unconditionally so it can drive a namespace-scope static initializer.
  // Replaces any prior registration under the same name.
  bool register_factory(std::string name, Factory factory) {
    factories_.insert_or_assign(std::move(name), std::move(factory));
    return true;
  }

  std::unique_ptr<Base> create(std::string_view name) const {
    const auto it = factories_.find(std::string{name});
    if (it == factories_.end()) {
      throw std::out_of_range{std::string{"quasar::Registry: no factory named '"}
                              + std::string{name} + "'"};
    }
    return (it->second)();
  }

  bool contains(std::string_view name) const noexcept {
    return factories_.find(std::string{name}) != factories_.end();
  }

  std::vector<std::string> names() const {
    std::vector<std::string> out;
    out.reserve(factories_.size());
    for (const auto& kv : factories_) {
      out.push_back(kv.first);
    }
    return out;
  }

  std::size_t size() const noexcept { return factories_.size(); }

 private:
  Registry()                           = default;
  ~Registry()                          = default;
  Registry(const Registry&)            = delete;
  Registry& operator=(const Registry&) = delete;
  Registry(Registry&&)                 = delete;
  Registry& operator=(Registry&&)      = delete;

  std::unordered_map<std::string, Factory> factories_{};
};

}  // namespace quasar

#define QUASAR_REGISTRY_DETAIL_PASTE_INNER(a, b) a##b
#define QUASAR_REGISTRY_DETAIL_PASTE(a, b) QUASAR_REGISTRY_DETAIL_PASTE_INNER(a, b)

// Generic registration sugar. Concrete axis macros (e.g.
// QUASAR_REGISTER_FIELD_EVALUATOR) live next to their base in 1.C+.
#define QUASAR_REGISTRY_REGISTER(Base, Name, Class)                                     \
  namespace {                                                                           \
  const bool QUASAR_REGISTRY_DETAIL_PASTE(_quasar_reg_, __LINE__) =                     \
      ::quasar::Registry<Base>::instance().register_factory(                            \
          (Name), []() -> std::unique_ptr<Base> { return std::make_unique<Class>(); }); \
  }
