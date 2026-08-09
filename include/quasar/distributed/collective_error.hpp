#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>

namespace quasar::distributed {

inline constexpr std::size_t collective_phase_capacity   = 64;
inline constexpr std::size_t collective_message_capacity = 256;
inline constexpr std::int32_t collective_protocol_error  = -32000;

// Severity is intentionally ordered so a maximum reduction implements the
// common accept/retry/fail decision.
enum class ErrorDisposition : std::uint8_t {
  success = 0,
  retry   = 1,
  fatal   = 2,
};

enum class CollectiveDecision : std::uint8_t {
  accept = 0,
  retry  = 1,
  fail   = 2,
};

// Fixed-size, trivially-copyable rank status suitable for MPI_Allgather as
// bytes.  Text is always NUL-terminated and is deterministically truncated.
// `endpoint == -1` denotes a rank-wide error.
struct CollectiveErrorRecord {
  std::uint64_t epoch{0};
  std::int32_t  rank{-1};
  std::int32_t  endpoint{-1};
  std::int32_t  code{0};
  ErrorDisposition disposition{ErrorDisposition::success};
  std::array<char, collective_phase_capacity> phase{};
  std::array<char, collective_message_capacity> message{};

  [[nodiscard]] static CollectiveErrorRecord success(
      std::uint64_t epoch, std::int32_t rank, std::string_view phase);
  [[nodiscard]] static CollectiveErrorRecord retry(
      std::uint64_t epoch, std::int32_t rank, std::int32_t endpoint,
      std::int32_t code, std::string_view phase, std::string_view message);
  [[nodiscard]] static CollectiveErrorRecord failure(
      std::uint64_t epoch, std::int32_t rank, std::int32_t endpoint,
      std::int32_t code, std::string_view phase, std::string_view message);

  [[nodiscard]] std::string_view phase_text() const noexcept;
  [[nodiscard]] std::string_view message_text() const noexcept;
  [[nodiscard]] bool ok() const noexcept {
    return disposition == ErrorDisposition::success;
  }
};

static_assert(std::is_trivially_copyable_v<CollectiveErrorRecord>);

struct CollectiveResolution {
  CollectiveDecision     decision{CollectiveDecision::accept};
  CollectiveErrorRecord  representative{};
  std::size_t            participant_count{0};
  std::size_t            retry_count{0};
  std::size_t            failure_count{0};

  [[nodiscard]] bool accepted() const noexcept {
    return decision == CollectiveDecision::accept;
  }
};

// Resolve already-gathered rank records into one deterministic decision.  The
// input must contain exactly one record for every contiguous rank [0,N), with
// a common epoch and phase.  Protocol disagreement is converted into a fatal
// resolution instead of throwing, so every rank that gathered the same bytes
// reaches the same branch.
[[nodiscard]] CollectiveResolution resolve_collective_errors(
    std::span<const CollectiveErrorRecord> records) noexcept;

}  // namespace quasar::distributed
