#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstring>
#include <exception>
#include <span>
#include <string_view>
#include <system_error>
#include <utility>

namespace quasar::distributed {

// Allocation-free bounded text used while ranks are converging on a common
// collective outcome. All variants share assign/append semantics so error
// paths cannot drift between checkpoint, diagnostics, MPI, and bindings.
template <std::size_t Capacity>
class FixedMessage {
  static_assert(Capacity > 0);

 public:
  FixedMessage() = default;
  explicit FixedMessage(std::string_view text) noexcept { assign(text); }

  void clear() noexcept {
    size_ = 0;
    storage_.front() = '\0';
  }

  void assign(std::string_view text) noexcept {
    clear();
    append(text);
  }

  void append(std::string_view text) noexcept {
    const std::size_t available = storage_.size() - size_ - 1;
    const std::size_t count = std::min(text.size(), available);
    if (count != 0) {
      std::memcpy(storage_.data() + size_, text.data(), count);
      size_ += count;
    }
    storage_[size_] = '\0';
  }

  template <class Integer>
  void append_integer(Integer value) noexcept {
    std::array<char, 32> text{};
    const auto converted =
        std::to_chars(text.data(), text.data() + text.size(), value);
    if (converted.ec == std::errc{}) {
      append({text.data(),
              static_cast<std::size_t>(converted.ptr - text.data())});
    }
  }

  [[nodiscard]] std::string_view view() const noexcept {
    return {storage_.data(), size_};
  }

  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] const char* c_str() const noexcept { return storage_.data(); }
  [[nodiscard]] operator std::string_view() const noexcept { return view(); }

 private:
  std::array<char, Capacity> storage_{};
  std::size_t size_{0};
};

// Same bounded writer semantics for storage owned by another public object,
// such as DistributedCollectiveError's fixed ABI message array.
class FixedMessageWriter {
 public:
  explicit FixedMessageWriter(std::span<char> storage) noexcept
      : storage_{storage} {
    clear();
  }

  void clear() noexcept {
    size_ = 0;
    if (!storage_.empty()) storage_.front() = '\0';
  }

  void append(std::string_view text) noexcept {
    if (storage_.empty() || size_ >= storage_.size() - 1) return;
    const std::size_t available = storage_.size() - size_ - 1;
    const std::size_t count = std::min(text.size(), available);
    if (count != 0) {
      std::memcpy(storage_.data() + size_, text.data(), count);
      size_ += count;
    }
    storage_[size_] = '\0';
  }

  template <class Integer>
  void append_integer(Integer value) noexcept {
    std::array<char, 32> text{};
    const auto converted =
        std::to_chars(text.data(), text.data() + text.size(), value);
    if (converted.ec == std::errc{}) {
      append({text.data(),
              static_cast<std::size_t>(converted.ptr - text.data())});
    }
  }

 private:
  std::span<char> storage_{};
  std::size_t size_{0};
};

enum class LocalFailureKind {
  none,
  standard_exception,
  non_standard_exception,
};

// Preserve whether a local operation threw a standard exception carrying
// diagnostic text or a non-standard exception carrying none. Some collective
// protocols intentionally retain an earlier, more useful message in the
// latter case.
template <std::size_t Capacity, class Function>
LocalFailureKind capture_local_exception(
    FixedMessage<Capacity>& message, std::string_view fallback,
    Function&& function) noexcept {
  try {
    std::forward<Function>(function)();
    return LocalFailureKind::none;
  } catch (const std::exception& error) {
    const char* detail = error.what();
    message.assign(detail == nullptr ? fallback : std::string_view{detail});
  } catch (...) {
    return LocalFailureKind::non_standard_exception;
  }
  return LocalFailureKind::standard_exception;
}

template <std::size_t Capacity, class Function>
bool capture_local_failure(FixedMessage<Capacity>& message,
                           std::string_view fallback,
                           Function&& function) noexcept {
  const LocalFailureKind failure = capture_local_exception(
      message, fallback, std::forward<Function>(function));
  if (failure == LocalFailureKind::non_standard_exception) {
    message.assign(fallback);
  }
  return failure == LocalFailureKind::none;
}

template <class Function>
bool local_operation_succeeds(Function&& function) noexcept {
  try {
    std::forward<Function>(function)();
    return true;
  } catch (...) {
    return false;
  }
}

// Some collective contracts deliberately expose one stable error string for
// every local exception. Keep that policy distinct from capture_local_failure,
// which preserves std::exception::what() when available.
template <std::size_t Capacity, class Function>
bool capture_local_failure_with_fallback(
    FixedMessage<Capacity>& message, std::string_view fallback,
    Function&& function) noexcept {
  const bool success =
      local_operation_succeeds(std::forward<Function>(function));
  if (!success) message.assign(fallback);
  return success;
}

// Local parsing, validation, and serialization may fail before a rank enters
// its next collective.  Keep both the flag and diagnostic in one shared,
// allocation-free representation until every rank has agreed on the outcome.
inline constexpr std::size_t collective_local_error_capacity = 1024;

class CollectiveLocalError {
 public:
  void capture(
      std::string_view detail,
      std::string_view fallback = "local collective operation failed") noexcept {
    failed_ = true;
    message_.assign(detail.empty() ? fallback : detail);
  }

  void capture(
      const std::exception& error,
      std::string_view fallback = "local collective operation failed") noexcept {
    const char* detail = error.what();
    capture(detail == nullptr ? std::string_view{} : std::string_view{detail},
            fallback);
  }

  void capture_unknown(
      std::string_view message =
          "unknown non-standard collective exception") noexcept {
    capture(message);
  }

  [[nodiscard]] bool ok() const noexcept { return !failed_; }
  [[nodiscard]] std::string_view message() const noexcept {
    return message_.view();
  }

 private:
  FixedMessage<collective_local_error_capacity> message_{};
  bool failed_{false};
};

template <class Operation>
CollectiveLocalError capture_collective_local_error(
    Operation&& operation) noexcept {
  CollectiveLocalError error;
  try {
    std::forward<Operation>(operation)();
  } catch (const std::exception& exception) {
    error.capture(exception);
  } catch (...) {
    error.capture_unknown();
  }
  return error;
}

}  // namespace quasar::distributed
