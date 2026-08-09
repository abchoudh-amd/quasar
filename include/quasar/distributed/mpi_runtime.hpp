#pragma once

#include "quasar/distributed/collective_error.hpp"

#include <array>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

namespace quasar::distributed {

namespace detail {
class MpiRuntimeNativeAccess;
}

class MpiError : public std::runtime_error {
 public:
  MpiError(std::string operation, int code, std::string detail);

  [[nodiscard]] int code() const noexcept { return code_; }
  [[nodiscard]] const std::string& operation() const noexcept {
    return operation_;
  }

 private:
  std::string operation_;
  int code_{0};
};

// Uses fixed inline message storage so constructing the synchronized exception
// cannot itself fail on only one rank before a caller's rollback/close phase.
class DistributedCollectiveError : public std::exception {
 public:
  explicit DistributedCollectiveError(
      CollectiveResolution resolution) noexcept;

  [[nodiscard]] const char* what() const noexcept override {
    return message_.data();
  }

  [[nodiscard]] const CollectiveResolution& resolution() const noexcept {
    return resolution_;
  }

 private:
  CollectiveResolution resolution_{};
  std::array<char, 512> message_{};
};

// Owns Quasar's MPI lifetime and shared-memory communicator.  Every member that
// calls MPI verifies that it is running on the construction thread.  close()
// is collective and explicit; the destructor intentionally never calls MPI.
class MpiRuntime {
 public:
  explicit MpiRuntime(int* argc = nullptr, char*** argv = nullptr);
  ~MpiRuntime() noexcept;

  MpiRuntime(const MpiRuntime&) = delete;
  MpiRuntime& operator=(const MpiRuntime&) = delete;
  MpiRuntime(MpiRuntime&&) = delete;
  MpiRuntime& operator=(MpiRuntime&&) = delete;

  [[nodiscard]] int rank() const noexcept;
  [[nodiscard]] int size() const noexcept;
  [[nodiscard]] int node_rank() const noexcept;
  [[nodiscard]] int node_size() const noexcept;
  [[nodiscard]] int thread_level() const noexcept;
  [[nodiscard]] bool owns_mpi() const noexcept;
  [[nodiscard]] bool closed() const noexcept;

  void require_orchestration_thread() const;
  void barrier() const;

  [[nodiscard]] CollectiveResolution consensus(
      CollectiveErrorRecord local) const;
  void require_collective_success(CollectiveErrorRecord local) const;

  [[nodiscard]] double allreduce_max(double local) const;
  [[nodiscard]] double allreduce_min(double local) const;
  [[nodiscard]] double allreduce_sum(double local) const;
  [[nodiscard]] std::uint64_t allreduce_sum(std::uint64_t local) const;
  [[nodiscard]] bool allreduce_all(bool local) const;

  // Completes communicator teardown on every rank and finalizes MPI only when
  // this object initialized it.  Calling close twice is harmless.
  void close();

 private:
  friend class detail::MpiRuntimeNativeAccess;

  class Impl;

  void require_open() const;

  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::string mpi_error_string(int code);
void check_mpi(int code, const char* operation);

}  // namespace quasar::distributed
