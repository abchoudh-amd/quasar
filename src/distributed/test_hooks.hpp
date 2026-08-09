#pragma once

#include <cstdlib>

namespace quasar::distributed {

// Ambient fault injection is compiled out unless the dedicated, default-OFF
// test-hook option is enabled. Keeping this check shared prevents one publisher
// from accidentally honoring test environment variables in release builds.
inline bool distributed_test_failure_enabled(const char* name) noexcept {
#if defined(QUASAR_DISTRIBUTED_TEST_HOOKS)
  const char* value = std::getenv(name);
  return value != nullptr && value[0] == '1' && value[1] == '\0';
#else
  (void)name;
  return false;
#endif
}

}  // namespace quasar::distributed
