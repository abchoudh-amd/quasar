#include "quasar/core/registry.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace {

// Test-local abstract base; isolated from production Registry<...> instances.
class IDummy {
 public:
  virtual ~IDummy() = default;
  virtual int magic() const = 0;
};

class DummyA : public IDummy {
 public:
  int magic() const override { return 42; }
};

class DummyB : public IDummy {
 public:
  int magic() const override { return 7; }
};

class INullDummy {
 public:
  virtual ~INullDummy() = default;
};

}  // namespace

QUASAR_REGISTRY_REGISTER(IDummy, "dummy_a", DummyA)
QUASAR_REGISTRY_REGISTER(IDummy, "dummy_b", DummyB)

TEST(Registry, ContainsRegisteredNames) {
  auto& reg = ::quasar::Registry<IDummy>::instance();
  EXPECT_TRUE(reg.contains("dummy_a"));
  EXPECT_TRUE(reg.contains("dummy_b"));
  EXPECT_FALSE(reg.contains("nope"));
}

TEST(Registry, CreatesByNameAndReturnsCorrectType) {
  auto& reg = ::quasar::Registry<IDummy>::instance();

  auto a = reg.create("dummy_a");
  ASSERT_NE(a, nullptr);
  EXPECT_EQ(a->magic(), 42);

  auto b = reg.create("dummy_b");
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(b->magic(), 7);
}

TEST(Registry, ThrowsOnMissingName) {
  auto& reg = ::quasar::Registry<IDummy>::instance();
  EXPECT_THROW(reg.create("not_here"), std::out_of_range);
}

TEST(Registry, RejectsInvalidRegistrationsWithoutMutation) {
  auto& reg = ::quasar::Registry<IDummy>::instance();
  const auto original_size = reg.size();

  EXPECT_THROW(reg.register_factory("", [] { return std::make_unique<DummyA>(); }),
               std::invalid_argument);
  EXPECT_THROW(reg.register_factory("empty_factory", {}), std::invalid_argument);
  EXPECT_EQ(reg.size(), original_size);
  EXPECT_FALSE(reg.contains("empty_factory"));
}

TEST(Registry, RejectsDuplicateNamesWithoutReplacingTheOriginal) {
  auto& reg = ::quasar::Registry<IDummy>::instance();
  EXPECT_THROW(reg.register_factory(
                   "dummy_a", [] { return std::make_unique<DummyB>(); }),
               std::invalid_argument);
  EXPECT_EQ(reg.create("dummy_a")->magic(), 42);
}

TEST(Registry, RejectsAFactoryThatReturnsNull) {
  auto& reg = ::quasar::Registry<INullDummy>::instance();
  ASSERT_TRUE(reg.register_factory(
      "null_factory", [] { return std::unique_ptr<INullDummy>{}; }));
  EXPECT_THROW((void)reg.create("null_factory"), std::runtime_error);
}

TEST(Registry, NamesListIncludesBothEntries) {
  auto names = ::quasar::Registry<IDummy>::instance().names();
  ASSERT_EQ(names.size(), 2u);

  bool seen_a = false;
  bool seen_b = false;
  for (const auto& n : names) {
    if (n == "dummy_a") seen_a = true;
    if (n == "dummy_b") seen_b = true;
  }
  EXPECT_TRUE(seen_a);
  EXPECT_TRUE(seen_b);
}

TEST(Registry, IsSingletonAcrossLookups) {
  auto* r1 = &::quasar::Registry<IDummy>::instance();
  auto* r2 = &::quasar::Registry<IDummy>::instance();
  EXPECT_EQ(r1, r2);
}
