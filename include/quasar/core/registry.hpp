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

  // Type-keyed registration. Each Derived instantiates a distinct factory
  // function, so the resulting std::function targets cannot be folded together
  // by the compiler/linker (identical-code folding would otherwise collapse the
  // stateless `make_unique<Derived>` lambdas of several types into one, making
  // every create() build whichever type the folded body kept — a real bug we hit
  // with the boundary registry). Keying the factory on &make<Derived> defeats
  // the fold because each address-taken specialization must stay distinct.
  template <class Derived>
  bool register_type(std::string name) {
    return register_factory(std::move(name), &make<Derived>);
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
  template <class Derived>
  static std::unique_ptr<Base> make() {
    return std::make_unique<Derived>();
  }

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
      ::quasar::Registry<Base>::instance().template register_type<Class>((Name));       \
  }
