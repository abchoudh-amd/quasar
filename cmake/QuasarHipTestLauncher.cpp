#include <hip/hip_runtime.h>

#include <cerrno>
#include <cstdio>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace {

constexpr int kCTestSkipReturnCode = 77;

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fputs("quasar HIP test launcher: missing test executable\n", stderr);
    return 2;
  }

  int device_count = 0;
  const hipError_t status = hipGetDeviceCount(&device_count);
  if (status == hipErrorNoDevice || (status == hipSuccess && device_count == 0)) {
    std::fputs("quasar HIP test skipped: no ROCm-capable device is visible\n",
               stderr);
    return kCTestSkipReturnCode;
  }
  if (status != hipSuccess) {
    std::fprintf(stderr,
                 "quasar HIP test launcher: hipGetDeviceCount failed: %s "
                 "(%d): %s\n",
                 hipGetErrorName(status), static_cast<int>(status),
                 hipGetErrorString(status));
    return 1;
  }
  if (device_count < 0) {
    std::fprintf(stderr,
                 "quasar HIP test launcher: hipGetDeviceCount returned an "
                 "invalid count: %d\n",
                 device_count);
    return 1;
  }

#if defined(_WIN32)
  std::vector<const char*> child_argv;
  child_argv.reserve(static_cast<std::size_t>(argc));
  for (int i = 1; i < argc; ++i) child_argv.push_back(argv[i]);
  child_argv.push_back(nullptr);
  const intptr_t result =
      _spawnv(_P_WAIT, argv[1], child_argv.data());
  if (result >= 0) return static_cast<int>(result);
#else
  execv(argv[1], argv + 1);
#endif

  const int saved_errno = errno;
  std::perror("quasar HIP test launcher: could not execute test");
  return saved_errno == ENOENT ? 127 : 126;
}
