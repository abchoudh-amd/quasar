#pragma once

#include <functional>
#include <memory>
#include <mutex>
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

  // Returns true on success so it can drive a namespace-scope static
  // initializer. Duplicate names are rejected: silently replacing one makes
  // the selected implementation depend on cross-translation-unit/plugin load
  // order, which is unspecified.
  bool register_factory(std::string name, Factory factory) {
    if (name.empty()) {
      throw std::invalid_argument{"quasar::Registry: factory name must not be empty"};
    }
    if (!factory) {
      throw std::invalid_argument{"quasar::Registry: factory must be callable"};
    }
    const std::scoped_lock lock{mutex_};
    const auto [it, inserted] =
        factories_.try_emplace(std::move(name), std::move(factory));
    if (!inserted) {
      throw std::invalid_argument{
          std::string{"quasar::Registry: duplicate factory name '"}
          + it->first + "'"};
    }
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
    Factory factory;
    {
      const std::scoped_lock lock{mutex_};
      const auto it = factories_.find(std::string{name});
      if (it == factories_.end()) {
        throw std::out_of_range{std::string{"quasar::Registry: no factory named '"}
                                + std::string{name} + "'"};
      }
      // Invoke outside the lock: a plugin factory may itself consult another
      // registry, and user construction must not serialize unrelated lookups.
      factory = it->second;
    }
    auto result = factory();
    if (!result) {
      throw std::runtime_error{std::string{"quasar::Registry: factory named '"}
                               + std::string{name} + "' returned null"};
    }
    return result;
  }

  // Looking up a string_view currently materializes a std::string for the
  // unordered-map key, so allocation failure is allowed to propagate rather
  // than terminating through a false noexcept promise.
  bool contains(std::string_view name) const {
    const std::scoped_lock lock{mutex_};
    return factories_.find(std::string{name}) != factories_.end();
  }

  std::vector<std::string> names() const {
    const std::scoped_lock lock{mutex_};
    std::vector<std::string> out;
    out.reserve(factories_.size());
    for (const auto& kv : factories_) {
      out.push_back(kv.first);
    }
    return out;
  }

  std::size_t size() const {
    const std::scoped_lock lock{mutex_};
    return factories_.size();
  }

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
  mutable std::mutex mutex_{};
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

// MHD-axis registration sugar. Each macro only pastes a fully-qualified interface
// name into the generic QUASAR_REGISTRY_REGISTER expansion above; it does not
// include or forward-declare the interface type, so this header compiles even
// before those interface headers exist. The macros are expanded later in each
// scheme's own .cpp (which includes its interface header).
#define QUASAR_REGISTER_RIEMANN_SOLVER(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::numerics::IRiemannSolver, Name, Class)

#define QUASAR_REGISTER_FLUX_RECONSTRUCTION(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::numerics::IFluxReconstruction, Name, Class)

#define QUASAR_REGISTER_INTEGRATOR(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::numerics::ISsprkIntegrator, Name, Class)

#define QUASAR_REGISTER_CT_SCHEME(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::numerics::ICtScheme, Name, Class)

#define QUASAR_REGISTER_POSITIVITY_LIMITER(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::numerics::IPositivityLimiter, Name, Class)

#define QUASAR_REGISTER_MHD_FLUID_BOUNDARY(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::boundary::IMhdFluidBoundary, Name, Class)

#define QUASAR_REGISTER_MHD_FIELD_BOUNDARY(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::boundary::IMhdFieldBoundary, Name, Class)

#define QUASAR_REGISTER_MHD_BACKGROUND_PROFILE(Name, Class) \
  QUASAR_REGISTRY_REGISTER(::quasar::numerics::IMhdBackgroundProfile, Name, Class)
